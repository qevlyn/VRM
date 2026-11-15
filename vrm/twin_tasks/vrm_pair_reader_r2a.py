from __future__ import annotations

import hashlib
import io
from dataclasses import asdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT, MAX_GRID_HEIGHT, MAX_GRID_TOKENS, MAX_GRID_WIDTH
from vrm.model import VRM, VRMConfig

from .pair_reader_r1 import PairReaderConfig
from .pair_reader_r1b import PairKeyValueReaderR1B

VRM_FORWARD_KEYS = {
    "support_evidence_colors",
    "support_evidence_rows",
    "support_evidence_cols",
    "support_evidence_roles",
    "support_evidence_pair_indices",
    "support_evidence_mask",
    "query_inputs",
    "query_input_mask",
}


class VRMWithFrozenPairReaderR2A(nn.Module):
    """Frozen VRM and frozen Pair Reader with a trainable output-route gate."""

    def __init__(
        self,
        *,
        legacy_vrm: VRM | None = None,
        reader: PairKeyValueReaderR1B | None = None,
        vrm_config: VRMConfig | None = None,
        reader_config: PairReaderConfig | None = None,
        constant_gate_feature: float = 100.0,
    ) -> None:
        super().__init__()
        self.legacy_vrm = legacy_vrm or VRM(vrm_config or VRMConfig())
        self.reader = reader or PairKeyValueReaderR1B(reader_config or PairReaderConfig())
        self.constant_gate_feature = float(constant_gate_feature)
        hidden_size = self.legacy_vrm.config.hidden_size
        self.gate_head = nn.Linear(hidden_size + 4, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.freeze_legacy_and_reader()

    def freeze_legacy_and_reader(self) -> None:
        for parameter in self.legacy_vrm.parameters():
            parameter.requires_grad = False
        for parameter in self.reader.parameters():
            parameter.requires_grad = False
        self.legacy_vrm.eval()
        self.reader.eval()

    def train(self, mode: bool = True) -> "VRMWithFrozenPairReaderR2A":
        super().train(mode)
        self.legacy_vrm.eval()
        self.reader.eval()
        return self

    def _validate_vrm_inputs(self, vrm_inputs: dict[str, torch.Tensor]) -> None:
        keys = set(vrm_inputs)
        if keys != VRM_FORWARD_KEYS:
            missing = sorted(VRM_FORWARD_KEYS - keys)
            extra = sorted(keys - VRM_FORWARD_KEYS)
            raise ValueError(f"invalid VRM input keys; missing={missing}, extra={extra}")

    def _reader_probs_full(self, reader_probs: torch.Tensor) -> torch.Tensor:
        batch, height, width, colors = reader_probs.shape
        if colors != ARC_COLOR_COUNT:
            raise ValueError("reader color count mismatch")
        full = reader_probs.new_zeros((batch, MAX_GRID_HEIGHT, MAX_GRID_WIDTH, ARC_COLOR_COUNT))
        full[..., 0] = 1.0
        full[:, :height, :width, :] = reader_probs
        return full

    def _gate_from_legacy_and_reader(
        self,
        final_canvas: torch.Tensor,
        reader_alpha: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = final_canvas.shape
        top2 = torch.topk(reader_alpha, k=min(2, reader_alpha.shape[1]), dim=-1).values
        top1 = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
        entropy = -(reader_alpha.clamp_min(1e-8) * torch.log(reader_alpha.clamp_min(1e-8))).sum(dim=-1)
        confidence = torch.stack(
            [
                top1,
                margin,
                entropy,
                torch.full_like(top1, self.constant_gate_feature),
            ],
            dim=-1,
        )
        confidence = confidence[:, None, :].expand(batch, tokens, -1)
        features = torch.cat([final_canvas, confidence], dim=-1)
        gate = torch.sigmoid(self.gate_head(features))
        return gate.view(batch, MAX_GRID_HEIGHT, MAX_GRID_WIDTH, 1)

    def forward(
        self,
        vrm_inputs: dict[str, torch.Tensor],
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        force_gate: float | None = None,
        reader_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._validate_vrm_inputs(vrm_inputs)
        legacy_outputs = self.legacy_vrm(**vrm_inputs)
        legacy_probs = legacy_outputs["color_probabilities"].view(
            -1, MAX_GRID_HEIGHT, MAX_GRID_WIDTH, ARC_COLOR_COUNT
        )
        reader_outputs = self.reader(query_grids, support_input_grids, support_output_grids)
        reader_probs_raw = reader_outputs["retrieved_color_probabilities"]
        reader_probs = self._reader_probs_full(reader_probs_raw)
        if reader_disabled:
            reader_probs_for_mix = torch.full_like(reader_probs, 1.0 / ARC_COLOR_COUNT)
        else:
            reader_probs_for_mix = reader_probs
        if force_gate is None:
            gate = self._gate_from_legacy_and_reader(legacy_outputs["final_canvas"], reader_outputs["retrieval_weights"])
        else:
            gate_value = float(force_gate)
            if gate_value < 0.0 or gate_value > 1.0:
                raise ValueError("force_gate must be between 0 and 1")
            gate = legacy_probs.new_full((legacy_probs.shape[0], MAX_GRID_HEIGHT, MAX_GRID_WIDTH, 1), gate_value)
        final_probs = (1.0 - gate) * legacy_probs + gate * reader_probs_for_mix
        decoded_full = final_probs.argmax(dim=-1)
        height = support_output_grids.shape[-2]
        width = support_output_grids.shape[-1]
        return {
            "legacy_probs": legacy_probs,
            "reader_probs": reader_probs,
            "reader_probs_raw": reader_probs_raw,
            "reader_probs_for_mix": reader_probs_for_mix,
            "final_probs": final_probs,
            "final_log_probs": torch.log(final_probs.clamp_min(1e-8)),
            "gate": gate,
            "reader_alpha": reader_outputs["retrieval_weights"],
            "reader_logits": reader_outputs["retrieval_logits"],
            "reader_decoded_grid": reader_outputs["decoded_grid"],
            "decoded_full_grid": decoded_full,
            "decoded_grid": decoded_full[:, :height, :width],
            "legacy_decoded_grid": legacy_probs.argmax(dim=-1)[:, :height, :width],
            "height_logits": legacy_outputs["height_logits"],
            "width_logits": legacy_outputs["width_logits"],
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "VRM Pair Reader R2A",
            "legacy_vrm_config": asdict(self.legacy_vrm.config),
            "reader_config": asdict(self.reader.config),
            "legacy_parameters_trainable": sum(parameter.numel() for parameter in self.legacy_vrm.parameters() if parameter.requires_grad),
            "reader_parameters_trainable": sum(parameter.numel() for parameter in self.reader.parameters() if parameter.requires_grad),
            "gate_parameters_trainable": sum(parameter.numel() for parameter in self.gate_head.parameters() if parameter.requires_grad),
            "gate_initialization": "final linear weights zero, bias zero; sigmoid output 0.5",
            "constant_gate_feature": self.constant_gate_feature,
        }


def r2a_output_loss(final_probs: torch.Tensor, target_grids: torch.Tensor) -> torch.Tensor:
    height = target_grids.shape[-2]
    width = target_grids.shape[-1]
    target_probs = final_probs[:, :height, :width, :]
    gathered = target_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(1e-8)).mean()


def tensor_tree_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in sorted(state_dict.items())}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        total += value * value
    return total**0.5

