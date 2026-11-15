from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT

from .pair_reader_r1b import PairKeyValueReaderR1B, PairReaderConfig
from .transform_r3b0_model import SupportConditionedTransformPointerR3B0


@dataclass(frozen=True)
class GateR3B2Config:
    evidence_dim: int = 23
    hidden_1: int = 128
    hidden_2: int = 64


READER_EVIDENCE_NAMES = [
    "reader_retrieval_logit_sorted_0",
    "reader_retrieval_logit_sorted_1",
    "reader_retrieval_logit_sorted_2",
    "reader_alpha_sorted_0",
    "reader_alpha_sorted_1",
    "reader_alpha_sorted_2",
    "reader_max_raw_match_score",
    "reader_top1_minus_top2_match_margin",
    "reader_alpha_entropy",
    "reader_min_q_to_key_distance",
    "reader_mean_q_to_key_distance",
    "reader_confidence_mean_max_color_probability",
    "reader_output_cell_entropy_mean",
    "reader_output_cell_entropy_max",
]

TRANSFORM_EVIDENCE_NAMES = [
    "transform_task_context_norm",
    "transform_task_context_mean",
    "transform_task_context_std",
    "transform_mean_max_pointer_probability",
    "transform_mean_pointer_entropy",
    "transform_max_pointer_entropy",
    "transform_output_confidence_mean_max_color_probability",
    "transform_output_cell_entropy_mean",
    "transform_output_cell_entropy_max",
]

EVIDENCE_FEATURE_NAMES = READER_EVIDENCE_NAMES + TRANSFORM_EVIDENCE_NAMES

FORBIDDEN_GATE_FORWARD_KEYS = {
    "target",
    "target_grid",
    "target_grids",
    "target_colors",
    "family",
    "family_label",
    "route_family",
    "transform_spec",
    "raw_support_output_grids",
    "support_output_grids",
    "raw_reader_output_grid",
    "raw_transform_output_grid",
    "candidate_grids",
    "candidate_labels",
    "query_type",
    "split",
    "episode_id",
    "counterfactual_group_id",
    "pair_id",
    "pair_position",
}


def load_reader_checkpoint(path: Path, device: torch.device) -> PairKeyValueReaderR1B:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = PairReaderConfig(**payload.get("config", {}))
    model = PairKeyValueReaderR1B(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_transform_checkpoint(path: Path, device: torch.device) -> SupportConditionedTransformPointerR3B0:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = SupportConditionedTransformPointerR3B0().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def cell_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs.clamp_min(eps) * torch.log(probs.clamp_min(eps))).sum(dim=-1)


def nll_per_episode(probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).flatten(1).mean(dim=1)


class FrozenReaderExpertR3B2(nn.Module):
    """Read-only R1B wrapper that exposes only permitted evidence summaries."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.reader = load_reader_checkpoint(self.checkpoint_path, device)

    def train(self, mode: bool = True) -> "FrozenReaderExpertR3B2":
        super().train(False)
        self.reader.eval()
        return self

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.reader.eval()
        outputs = self.reader(query_grids, support_input_grids, support_output_grids)
        probs = outputs["retrieved_color_probabilities"]
        alpha = outputs["retrieval_weights"]
        logits = outputs["retrieval_logits"]
        sorted_logits = torch.sort(logits, dim=-1, descending=True).values
        sorted_alpha = torch.sort(alpha, dim=-1, descending=True).values
        distances = (query_grids[:, None, :, :] != support_input_grids).flatten(2).sum(dim=-1).to(torch.float32)
        top2 = sorted_logits[:, :2]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(alpha.clamp_min(1e-8) * torch.log(alpha.clamp_min(1e-8))).sum(dim=-1)
        output_entropy = cell_entropy(probs)
        confidence = probs.max(dim=-1).values.mean(dim=(1, 2))
        temperature = float(self.reader.temperature.detach().cpu().item())
        raw_scores = logits * max(temperature, 1e-6)
        evidence = torch.cat(
            [
                sorted_logits,
                sorted_alpha,
                raw_scores.max(dim=-1).values[:, None],
                margin[:, None],
                entropy[:, None],
                distances.min(dim=-1).values[:, None],
                distances.mean(dim=-1)[:, None],
                confidence[:, None],
                output_entropy.mean(dim=(1, 2))[:, None],
                output_entropy.flatten(1).max(dim=-1).values[:, None],
            ],
            dim=-1,
        )
        return {
            "reader_probs": probs,
            "reader_decoded_grid": probs.argmax(dim=-1),
            "reader_alpha": alpha,
            "retrieval_logits": logits,
            "reader_qk_distances": distances,
            "reader_evidence": evidence,
            "reader_evidence_summary": {
                "feature_names": READER_EVIDENCE_NAMES,
                "shape": tuple(evidence.shape),
            },
        }


class FrozenTransformExpertR3B2(nn.Module):
    """Read-only R3B-0 wrapper that exposes only permitted evidence summaries."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.transform = load_transform_checkpoint(self.checkpoint_path, device)
        self.loo_transform_consistency_available = False

    def train(self, mode: bool = True) -> "FrozenTransformExpertR3B2":
        super().train(False)
        self.transform.eval()
        return self

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.transform.eval()
        outputs = self.transform(query_grids, support_input_grids, support_output_grids)
        probs = outputs["output_probs"]
        context = outputs["task_context"]
        pointer = outputs["pointer_attention"]
        pointer_entropy = cell_entropy(pointer)
        output_entropy = cell_entropy(probs)
        evidence = torch.cat(
            [
                context.norm(dim=-1)[:, None],
                context.mean(dim=-1)[:, None],
                context.std(dim=-1, unbiased=False)[:, None],
                pointer.max(dim=-1).values.mean(dim=(1, 2))[:, None],
                pointer_entropy.mean(dim=(1, 2))[:, None],
                pointer_entropy.flatten(1).max(dim=-1).values[:, None],
                probs.max(dim=-1).values.mean(dim=(1, 2))[:, None],
                output_entropy.mean(dim=(1, 2))[:, None],
                output_entropy.flatten(1).max(dim=-1).values[:, None],
            ],
            dim=-1,
        )
        return {
            "transform_probs": probs,
            "transform_decoded_grid": probs.argmax(dim=-1),
            "transform_task_context": context,
            "transform_pointer_attention": pointer,
            "transform_evidence": evidence,
            "transform_evidence_summary": {
                "feature_names": TRANSFORM_EVIDENCE_NAMES,
                "shape": tuple(evidence.shape),
                "loo_transform_consistency_available": self.loo_transform_consistency_available,
            },
        }


class EvidenceGateR3B2(nn.Module):
    """Small MLP gate over frozen-expert evidence summaries."""

    def __init__(self, config: GateR3B2Config | None = None) -> None:
        super().__init__()
        self.config = config or GateR3B2Config()
        self.network = nn.Sequential(
            nn.LayerNorm(self.config.evidence_dim),
            nn.Linear(self.config.evidence_dim, self.config.hidden_1),
            nn.GELU(),
            nn.Linear(self.config.hidden_1, self.config.hidden_2),
            nn.GELU(),
            nn.Linear(self.config.hidden_2, 1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected final gate layer")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or evidence.shape[-1] != self.config.evidence_dim:
            raise ValueError(f"expected evidence [batch,{self.config.evidence_dim}], got {tuple(evidence.shape)}")
        return torch.sigmoid(self.network(evidence).squeeze(-1))

    def initialization_report(self) -> dict[str, Any]:
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        return {
            "architecture": "R3B-2 Evidence Gate MLP",
            "config": asdict(self.config),
            "evidence_feature_names": EVIDENCE_FEATURE_NAMES,
            "forbidden_forward_keys": sorted(FORBIDDEN_GATE_FORWARD_KEYS),
            "final_layer_weight_abs_sum": float(final.weight.detach().abs().sum().item()),
            "final_layer_bias_abs_sum": float(final.bias.detach().abs().sum().item()),
        }


class MixedRouterR3B2(nn.Module):
    """Frozen R1B reader + frozen R3B-0 transform expert + trainable evidence gate."""

    def __init__(self, reader_checkpoint: Path, transform_checkpoint: Path, device: torch.device) -> None:
        super().__init__()
        self.reader_expert = FrozenReaderExpertR3B2(reader_checkpoint, device)
        self.transform_expert = FrozenTransformExpertR3B2(transform_checkpoint, device)
        self.gate = EvidenceGateR3B2().to(device)
        self.freeze_experts()

    def freeze_experts(self) -> None:
        for parameter in self.reader_expert.parameters():
            parameter.requires_grad = False
        for parameter in self.transform_expert.parameters():
            parameter.requires_grad = False
        self.reader_expert.eval()
        self.transform_expert.eval()

    def train(self, mode: bool = True) -> "MixedRouterR3B2":
        super().train(mode)
        self.reader_expert.eval()
        self.transform_expert.eval()
        return self

    def gate_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.gate.parameters() if parameter.requires_grad]

    def expert_parameters(self) -> list[nn.Parameter]:
        return list(self.reader_expert.parameters()) + list(self.transform_expert.parameters())

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        *,
        force_gate: float | None = None,
        reader_disabled: bool = False,
        transform_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.reader_expert.eval()
        self.transform_expert.eval()
        with torch.no_grad():
            reader_outputs = self.reader_expert(query_grids, support_input_grids, support_output_grids)
            transform_outputs = self.transform_expert(query_grids, support_input_grids, support_output_grids)
        evidence = torch.cat(
            [reader_outputs["reader_evidence"], transform_outputs["transform_evidence"]],
            dim=-1,
        ).detach()
        if force_gate is None:
            gate = self.gate(evidence)
        else:
            if force_gate < 0.0 or force_gate > 1.0:
                raise ValueError("force_gate must be in [0,1]")
            gate = evidence.new_full((evidence.shape[0],), float(force_gate))
        reader_probs = reader_outputs["reader_probs"].detach()
        transform_probs = transform_outputs["transform_probs"].detach()
        reader_probs_for_mix = (
            torch.full_like(reader_probs, 1.0 / ARC_COLOR_COUNT)
            if reader_disabled
            else reader_probs
        )
        transform_probs_for_mix = (
            torch.full_like(transform_probs, 1.0 / ARC_COLOR_COUNT)
            if transform_disabled
            else transform_probs
        )
        soft_probs = gate[:, None, None, None] * reader_probs_for_mix + (1.0 - gate[:, None, None, None]) * transform_probs_for_mix
        hard_reader = gate >= 0.5
        hard_probs = torch.where(hard_reader[:, None, None, None], reader_probs_for_mix, transform_probs_for_mix)
        return {
            "soft_probs": soft_probs,
            "soft_decoded_grid": soft_probs.argmax(dim=-1),
            "hard_route_probs": hard_probs,
            "hard_decoded_grid": hard_probs.argmax(dim=-1),
            "gate": gate,
            "hard_route_reader": hard_reader,
            "evidence": evidence,
            "reader_probs": reader_probs,
            "reader_probs_for_mix": reader_probs_for_mix,
            "reader_decoded_grid": reader_outputs["reader_decoded_grid"],
            "reader_alpha": reader_outputs["reader_alpha"],
            "retrieval_logits": reader_outputs["retrieval_logits"],
            "reader_qk_distances": reader_outputs["reader_qk_distances"],
            "reader_evidence": reader_outputs["reader_evidence"],
            "transform_probs": transform_probs,
            "transform_probs_for_mix": transform_probs_for_mix,
            "transform_decoded_grid": transform_outputs["transform_decoded_grid"],
            "transform_task_context": transform_outputs["transform_task_context"],
            "transform_pointer_attention": transform_outputs["transform_pointer_attention"],
            "transform_evidence": transform_outputs["transform_evidence"],
        }

    def initialization_report(self, sample_gate: torch.Tensor | None = None) -> dict[str, Any]:
        return {
            "architecture": "R3B-2 Frozen Neural Reader + Frozen Neural Transform Expert Mixed Router",
            "reader_checkpoint": str(self.reader_expert.checkpoint_path),
            "transform_checkpoint": str(self.transform_expert.checkpoint_path),
            "reader_frozen": all(not parameter.requires_grad for parameter in self.reader_expert.parameters()),
            "transform_frozen": all(not parameter.requires_grad for parameter in self.transform_expert.parameters()),
            "frozen_experts_eval_mode": (not self.reader_expert.training and not self.transform_expert.training),
            "trainable_gate_parameters": sum(parameter.numel() for parameter in self.gate.parameters() if parameter.requires_grad),
            "expert_trainable_parameters": sum(parameter.numel() for parameter in self.expert_parameters() if parameter.requires_grad),
            "gate": self.gate.initialization_report(),
            "mean_initial_gate_on_sample": None if sample_gate is None else float(sample_gate.detach().mean().item()),
            "loo_transform_consistency_available": self.transform_expert.loo_transform_consistency_available,
        }


def router_output_loss(soft_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = soft_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        total += value * value
    return math.sqrt(total)
