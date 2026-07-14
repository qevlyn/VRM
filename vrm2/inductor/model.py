from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class InductorModelConfig:
    color_embedding_dim: int = 32
    hidden_dim: int = 128
    attention_heads: int = 4
    pair_encoder_blocks: int = 2
    pair_summary_slots: int = 4
    maximum_supports: int = 3
    dropout: float = 0.0


def model_config_from_mapping(config: Mapping[str, Any]) -> InductorModelConfig:
    return InductorModelConfig(
        **{field: config[field] for field in InductorModelConfig.__dataclass_fields__}
    )


class SpatialResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.convolution = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        return spatial + self.convolution(self.norm(spatial))


class SharedPairEncoder(nn.Module):
    def __init__(self, config: InductorModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        color = config.color_embedding_dim
        self.color_embedding = nn.Embedding(10, color)
        self.feature_projection = nn.Linear(color * 4 + 1, hidden)
        self.row_embedding = nn.Embedding(8, hidden)
        self.column_embedding = nn.Embedding(8, hidden)
        self.spatial_blocks = nn.ModuleList(
            [SpatialResidualBlock(hidden) for _ in range(config.pair_encoder_blocks)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.pair_encoder_blocks, enable_nested_tensor=False
        )
        self.summary_slots = nn.Parameter(
            torch.empty(1, config.pair_summary_slots, hidden)
        )
        self.summary_attention = nn.MultiheadAttention(
            hidden, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.summary_norm = nn.LayerNorm(hidden)
        self.summary_mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        nn.init.normal_(self.summary_slots, std=0.02)

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        x = self.color_embedding(input_grid)
        y = self.color_embedding(output_grid)
        same = input_grid.eq(output_grid).to(x.dtype).unsqueeze(-1)
        features = torch.cat((x, y, y - x, x * y, same), dim=-1)
        spatial = self.feature_projection(features).permute(0, 3, 1, 2)
        for block in self.spatial_blocks:
            spatial = block(spatial)
        tokens = spatial.flatten(2).transpose(1, 2)
        rows = torch.arange(8, device=tokens.device).repeat_interleave(8)
        columns = torch.arange(8, device=tokens.device).repeat(8)
        tokens = (
            tokens
            + self.row_embedding(rows).unsqueeze(0)
            + self.column_embedding(columns).unsqueeze(0)
        )
        tokens = self.transformer(tokens)
        slots = self.summary_slots.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.summary_attention(slots, tokens, tokens, need_weights=False)
        pooled = self.summary_norm(slots + pooled)
        pooled = pooled + self.summary_mlp(pooled)
        return pooled.mean(dim=1)


class SupervisedTypedProgramInductorV0(nn.Module):
    """Permutation-invariant support-set encoder with factorized typed-program heads."""

    model_version = "supervised-typed-program-inductor-v0"

    def __init__(self, config: InductorModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or InductorModelConfig()
        self.pair_encoder = SharedPairEncoder(self.config)
        hidden = self.config.hidden_dim
        self.pair_set_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.task_projection = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.length_head = nn.Linear(hidden, 2)
        self.operator_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(3)])
        self.selector_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(3)])
        self.argument_type_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(3)])
        self.argument_value_heads = nn.ModuleList([nn.Linear(hidden, 10) for _ in range(3)])

    def forward(
        self,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if support_input_grids.ndim != 4 or support_input_grids.shape[2:] != (8, 8):
            raise ValueError("support_input_grids must have shape [batch, supports, 8, 8]")
        if support_output_grids.shape != support_input_grids.shape:
            raise ValueError("support input/output tensor shapes must match")
        batch, supports = support_input_grids.shape[:2]
        if not 1 <= supports <= self.config.maximum_supports:
            raise ValueError("support count exceeds the frozen maximum")
        pair_features = self.pair_encoder(
            support_input_grids.reshape(batch * supports, 8, 8),
            support_output_grids.reshape(batch * supports, 8, 8),
        ).reshape(batch, supports, self.config.hidden_dim)
        pair_features = self.pair_set_projection(pair_features)
        if support_mask is None:
            support_mask = torch.ones(batch, supports, device=pair_features.device, dtype=torch.bool)
        weights = support_mask.to(pair_features.dtype).unsqueeze(-1)
        task = (pair_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        task = self.task_projection(task)
        return {
            "length": self.length_head(task),
            "operator": torch.stack([head(task) for head in self.operator_heads], dim=1),
            "selector": torch.stack([head(task) for head in self.selector_heads], dim=1),
            "argument_type": torch.stack(
                [head(task) for head in self.argument_type_heads], dim=1
            ),
            "argument_value": torch.stack(
                [head(task) for head in self.argument_value_heads], dim=1
            ),
        }


def _masked_argmax(logits: torch.Tensor, valid_ids: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    masked = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked[:, list(valid_ids)] = logits[:, list(valid_ids)]
    probabilities = masked.softmax(dim=-1)
    prediction = probabilities.argmax(dim=-1)
    confidence = probabilities.gather(1, prediction.unsqueeze(1)).squeeze(1)
    return prediction, confidence


@torch.no_grad()
def materialize_hard_program(
    logits: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    lengths = logits["length"].argmax(dim=-1) + 2
    batch = lengths.shape[0]
    programs = torch.zeros(batch, 3, 4, dtype=torch.long, device=lengths.device)
    confidence = torch.zeros(batch, 3, 4, dtype=torch.float32, device=lengths.device)
    length_probabilities = logits["length"].softmax(dim=-1)
    length_confidence = length_probabilities.gather(
        1, (lengths - 2).unsqueeze(1)
    ).squeeze(1)
    for step in range(3):
        active = step < lengths
        operators, operator_confidence = _masked_argmax(logits["operator"][:, step], (1, 2, 3))
        programs[active, step, 0] = operators[active]
        confidence[active, step, 0] = operator_confidence[active].float()
        for operator_id, selector_ids, argument_type_id, value_ids in (
            (1, (1, 2), 1, (1, 2, 3, 4)),
            (2, (1, 2), 2, (1, 2, 3, 4, 5, 6, 7, 8, 9)),
            (3, (3,), 3, (1, 2)),
        ):
            selected = active & operators.eq(operator_id)
            selectors, selector_confidence = _masked_argmax(
                logits["selector"][:, step], selector_ids
            )
            values, value_confidence = _masked_argmax(
                logits["argument_value"][:, step], value_ids
            )
            programs[selected, step, 1] = selectors[selected]
            programs[selected, step, 2] = argument_type_id
            programs[selected, step, 3] = values[selected]
            confidence[selected, step, 1] = selector_confidence[selected].float()
            argument_probabilities = logits["argument_type"][:, step].softmax(dim=-1)
            confidence[selected, step, 2] = argument_probabilities[
                selected, argument_type_id
            ].float()
            confidence[selected, step, 3] = value_confidence[selected].float()
    return lengths.long(), programs, {
        "length": length_confidence.float(), "fields": confidence
    }


def inductor_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
