from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from vrm2.inductor.model import InductorModelConfig, SharedPairEncoder
from vrm2.inductor.sequential_decoder import (
    SequentialDecoderConfig,
    SequentialResidualProgramDecoder,
)


@dataclass(frozen=True)
class InductorV1ModelConfig:
    color_embedding_dim: int = 32
    hidden_dim: int = 128
    attention_heads: int = 4
    pair_encoder_blocks: int = 2
    pair_summary_slots: int = 4
    maximum_supports: int = 3
    dropout: float = 0.0
    instruction_embedding_dim: int = 16
    disable_residual_update: bool = False


def model_config_from_mapping(config: Mapping[str, Any]) -> InductorV1ModelConfig:
    return InductorV1ModelConfig(
        **{field: config[field] for field in InductorV1ModelConfig.__dataclass_fields__}
    )


class SequentialResidualInductorV1(nn.Module):
    model_version = "sequential-residual-inductor-v1"

    def __init__(self, config: InductorV1ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or InductorV1ModelConfig()
        encoder_config = InductorModelConfig(
            color_embedding_dim=self.config.color_embedding_dim,
            hidden_dim=self.config.hidden_dim,
            attention_heads=self.config.attention_heads,
            pair_encoder_blocks=self.config.pair_encoder_blocks,
            pair_summary_slots=self.config.pair_summary_slots,
            maximum_supports=self.config.maximum_supports,
            dropout=self.config.dropout,
        )
        self.pair_encoder = SharedPairEncoder(encoder_config)
        hidden = self.config.hidden_dim
        self.pair_set_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.task_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.decoder = SequentialResidualProgramDecoder(
            SequentialDecoderConfig(
                hidden_dim=hidden,
                instruction_embedding_dim=self.config.instruction_embedding_dim,
                disable_residual_update=self.config.disable_residual_update,
            )
        )

    def encode_supports(
        self,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            support_mask = torch.ones(
                batch, supports, device=pair_features.device, dtype=torch.bool
            )
        weights = support_mask.to(pair_features.dtype).unsqueeze(-1)
        task = (pair_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.task_projection(task)

    def forward(
        self,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        target_typed_program_ids: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        task = self.encode_supports(
            support_input_grids, support_output_grids, support_mask
        )
        return self.decoder.teacher_forced(task, target_typed_program_ids)

    def forward_autoregressive(
        self,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        support_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        task = self.encode_supports(
            support_input_grids, support_output_grids, support_mask
        )
        return self.decoder.autoregressive(task)


def component_parameter_counts(model: SequentialResidualInductorV1) -> dict[str, int]:
    decoder = model.decoder
    components = {
        "pair_encoder": sum(item.numel() for item in model.pair_encoder.parameters()),
        "set_aggregator": sum(item.numel() for item in model.pair_set_projection.parameters())
        + sum(item.numel() for item in model.task_projection.parameters()),
        "residual_initializer": sum(item.numel() for item in decoder.residual_initializer.parameters()),
        "shared_decoder_cell": sum(item.numel() for item in decoder.decoder_input_projection.parameters())
        + sum(item.numel() for item in decoder.decoder_cell.parameters()),
        "shared_residual_update": sum(item.numel() for item in decoder.residual_update.parameters())
        + sum(item.numel() for item in decoder.residual_norm.parameters()),
        "instruction_embeddings": decoder.begin_program_embedding.numel()
        + sum(item.numel() for item in decoder.operator_embedding.parameters())
        + sum(item.numel() for item in decoder.selector_embedding.parameters())
        + sum(item.numel() for item in decoder.argument_type_embedding.parameters())
        + sum(item.numel() for item in decoder.argument_value_embedding.parameters())
        + sum(item.numel() for item in decoder.instruction_projection.parameters()),
        "shared_field_heads": sum(item.numel() for item in decoder.operator_head.parameters())
        + sum(item.numel() for item in decoder.selector_head.parameters())
        + sum(item.numel() for item in decoder.argument_type_head.parameters())
        + sum(item.numel() for item in decoder.argument_value_head.parameters()),
    }
    components["total"] = sum(components.values())
    return components
