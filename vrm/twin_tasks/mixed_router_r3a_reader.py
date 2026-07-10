from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT


@dataclass(frozen=True)
class R3AReaderConfig:
    grid_size: int = 7
    color_count: int = ARC_COLOR_COUNT
    initial_temperature: float = 0.1


class PairKeyValueReaderR3A(nn.Module):
    """R3A key-value reader using shared Q/X grid features and raw Y values."""

    def __init__(self, config: R3AReaderConfig | None = None) -> None:
        super().__init__()
        self.config = config or R3AReaderConfig()
        if self.config.initial_temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(self.config.initial_temperature), dtype=torch.float32))

    def encode(self, grids: torch.Tensor) -> torch.Tensor:
        if grids.ndim != 3:
            raise ValueError(f"expected [batch,7,7], got {tuple(grids.shape)}")
        if grids.shape[1:] != (self.config.grid_size, self.config.grid_size):
            raise ValueError("R3A reader expects exactly 7x7 grids")
        return F.one_hot(grids.long(), num_classes=self.config.color_count).to(torch.float32).flatten(1)

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if support_input_grids.ndim != 4 or support_output_grids.ndim != 4:
            raise ValueError("support grids must be [batch,pairs,7,7]")
        if support_input_grids.shape != support_output_grids.shape:
            raise ValueError("support input and output shapes differ")
        batch, pair_count, height, width = support_input_grids.shape
        if (height, width) != (self.config.grid_size, self.config.grid_size):
            raise ValueError("R3A reader expects exactly 7x7 support grids")
        query_features = self.encode(query_grids)
        key_features = self.encode(support_input_grids.reshape(batch * pair_count, height, width)).reshape(batch, pair_count, -1)
        raw_scores = torch.einsum("bd,bpd->bp", query_features, key_features)
        temperature = self.log_temperature.exp().clamp_min(1e-4)
        retrieval_logits = raw_scores / temperature
        alpha = torch.softmax(retrieval_logits, dim=-1)
        values = F.one_hot(support_output_grids.long(), num_classes=self.config.color_count).to(torch.float32)
        reader_probs = torch.sum(alpha[:, :, None, None, None] * values, dim=1)
        decoded = reader_probs.argmax(dim=-1)
        distances = (query_grids[:, None, :, :] != support_input_grids).flatten(2).sum(dim=-1).to(torch.float32)
        top2 = torch.topk(retrieval_logits, k=min(2, pair_count), dim=-1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else top2[:, 0]
        entropy = -(alpha.clamp_min(1e-8) * torch.log(alpha.clamp_min(1e-8))).sum(dim=-1)
        match_features = torch.stack(
            [
                retrieval_logits.max(dim=-1).values / 500.0,
                margin / 20.0,
                entropy,
                torch.exp(-distances.min(dim=-1).values) * 100.0,
                distances.min(dim=-1).values * 100.0,
                distances.mean(dim=-1),
            ],
            dim=-1,
        )
        return {
            "query_features": query_features,
            "key_features": key_features,
            "raw_scores": raw_scores,
            "retrieval_logits": retrieval_logits,
            "retrieval_weights": alpha,
            "reader_probs": reader_probs,
            "retrieved_color_probabilities": reader_probs,
            "decoded_grid": decoded,
            "qk_distances": distances,
            "reader_match_features": match_features,
            "temperature": temperature.detach(),
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "Pair Key-Value Reader R3A",
            "config": asdict(self.config),
            "training_objective": "output-only pixel NLL from retrieved probabilities to target grid",
            "retrieval_supervision_used": False,
            "value_route": "raw support output one-hot grids mixed by retrieval alpha",
            "parameter_count": {
                "total": sum(parameter.numel() for parameter in self.parameters()),
                "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            },
        }


def reader_output_loss(reader_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = reader_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def reader_training_batch_keys() -> list[str]:
    return ["query_grids", "support_input_grids", "support_output_grids", "target_grids"]
