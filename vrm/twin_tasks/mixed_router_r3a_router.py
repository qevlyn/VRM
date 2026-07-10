from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vrm.data import ARC_COLOR_COUNT

from .mixed_router_r3a_reader import PairKeyValueReaderR3A


class MixedRouteRouterR3A(nn.Module):
    """Frozen reader + deterministic transform oracle probabilities + trainable scalar gate."""

    def __init__(self, reader: PairKeyValueReaderR3A) -> None:
        super().__init__()
        self.reader = reader
        self.gate_head = nn.Linear(6, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)
        self.freeze_reader()

    def freeze_reader(self) -> None:
        for parameter in self.reader.parameters():
            parameter.requires_grad = False
        self.reader.eval()

    def train(self, mode: bool = True) -> "MixedRouteRouterR3A":
        super().train(mode)
        self.reader.eval()
        return self

    def gate_from_reader_features(self, reader_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = reader_outputs["reader_match_features"]
        logits = self.gate_head(features).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        transform_oracle_probs: torch.Tensor,
        *,
        force_gate: float | None = None,
        reader_disabled: bool = False,
        transform_disabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        reader_outputs = self.reader(query_grids, support_input_grids, support_output_grids)
        reader_probs = reader_outputs["reader_probs"]
        if reader_disabled:
            reader_probs_for_mix = torch.full_like(reader_probs, 1.0 / ARC_COLOR_COUNT)
        else:
            reader_probs_for_mix = reader_probs
        if transform_disabled:
            transform_probs_for_mix = torch.full_like(transform_oracle_probs, 1.0 / ARC_COLOR_COUNT)
        else:
            transform_probs_for_mix = transform_oracle_probs
        if force_gate is None:
            gate = self.gate_from_reader_features(reader_outputs)
        else:
            gate_value = float(force_gate)
            if gate_value < 0.0 or gate_value > 1.0:
                raise ValueError("force_gate must be in [0,1]")
            gate = reader_probs.new_full((reader_probs.shape[0],), gate_value)
        final_probs = gate[:, None, None, None] * reader_probs_for_mix + (1.0 - gate[:, None, None, None]) * transform_probs_for_mix
        return {
            "final_probs": final_probs,
            "decoded_grid": final_probs.argmax(dim=-1),
            "gate": gate,
            "reader_probs": reader_probs,
            "reader_probs_for_mix": reader_probs_for_mix,
            "reader_alpha": reader_outputs["retrieval_weights"],
            "retrieval_logits": reader_outputs["retrieval_logits"],
            "reader_match_features": reader_outputs["reader_match_features"],
            "qk_distances": reader_outputs["qk_distances"],
            "raw_scores": reader_outputs["raw_scores"],
            "reader_decoded_grid": reader_outputs["decoded_grid"],
            "transform_oracle_probs": transform_oracle_probs,
            "transform_probs_for_mix": transform_probs_for_mix,
            "transform_decoded_grid": transform_oracle_probs.argmax(dim=-1),
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "Mixed Router R3A",
            "reader_frozen": all(not parameter.requires_grad for parameter in self.reader.parameters()),
            "gate_parameters_trainable": sum(parameter.numel() for parameter in self.gate_head.parameters() if parameter.requires_grad),
            "gate_input_features": [
                "max retrieval logit / 500",
                "top1-top2 retrieval margin / 20",
                "alpha entropy",
                "100 * exp(-minimum q-key Hamming distance)",
                "100 * minimum q-key Hamming distance",
                "mean q-key Hamming distance",
            ],
            "gate_initialization": "linear weights zero and bias zero; sigmoid output 0.5",
        }


def router_output_loss(final_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = final_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        total += value * value
    return total**0.5
