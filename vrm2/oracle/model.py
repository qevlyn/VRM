from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class OracleModelConfig:
    grid_size: int = 8
    hidden_dim: int = 96
    color_embedding_dim: int = 32
    workspace_tokens: int = 4
    attention_heads: int = 4
    shared_execution_layers: int = 2
    maximum_program_length: int = 3
    dropout: float = 0.0


def model_config_from_mapping(config: Mapping[str, Any]) -> OracleModelConfig:
    return OracleModelConfig(**{field: config[field] for field in OracleModelConfig.__dataclass_fields__})


class InstructionEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.operator_embedding = nn.Embedding(4, hidden_dim)
        self.selector_embedding = nn.Embedding(4, hidden_dim)
        self.argument_type_embedding = nn.Embedding(4, hidden_dim)
        self.argument_value_embedding = nn.Embedding(10, hidden_dim)
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, instruction_ids: torch.Tensor) -> torch.Tensor:
        factors = (
            self.operator_embedding(instruction_ids[:, 0])
            + self.selector_embedding(instruction_ids[:, 1])
            + self.argument_type_embedding(instruction_ids[:, 2])
            + self.argument_value_embedding(instruction_ids[:, 3])
        )
        return self.fuse(factors)


class ConditionalExecutionLayer(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int, dropout: float, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.spatial_tokens = grid_size * grid_size
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.convolution_norm = nn.LayerNorm(hidden_dim)
        self.convolution_film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.convolution = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp_film = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _condition(normalized: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        scale, shift = film.chunk(2, dim=-1)
        return normalized * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        attended_input = self._condition(self.attention_norm(state), self.attention_film(instruction))
        attended, _ = self.attention(attended_input, attended_input, attended_input, need_weights=False)
        state = state + self.dropout(attended)

        spatial = state[:, : self.spatial_tokens]
        convolved_input = self._condition(
            self.convolution_norm(spatial), self.convolution_film(instruction)
        )
        convolved_input = convolved_input.transpose(1, 2).reshape(
            state.shape[0], state.shape[2], self.grid_size, self.grid_size
        )
        convolved = self.convolution(convolved_input).flatten(2).transpose(1, 2)
        state = torch.cat((spatial + self.dropout(convolved), state[:, self.spatial_tokens :]), dim=1)

        mlp_input = self._condition(self.mlp_norm(state), self.mlp_film(instruction))
        return state + self.dropout(self.mlp(mlp_input))


class SharedExecutionBlock(nn.Module):
    def __init__(self, config: OracleModelConfig) -> None:
        super().__init__()
        self.spatial_tokens = config.grid_size * config.grid_size
        self.instruction_to_workspace = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                ConditionalExecutionLayer(
                    config.hidden_dim,
                    config.attention_heads,
                    config.dropout,
                    config.grid_size,
                )
                for _ in range(config.shared_execution_layers)
            ]
        )

    def forward(self, state: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        spatial = state[:, : self.spatial_tokens]
        workspace = state[:, self.spatial_tokens :] + self.instruction_to_workspace(instruction).unsqueeze(1)
        state = torch.cat((spatial, workspace), dim=1)
        for layer in self.layers:
            state = layer(state, instruction)
        return state


class OracleInterpreterV0(nn.Module):
    """One shared learned execution block reused for every hard typed instruction."""

    def __init__(self, config: OracleModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or OracleModelConfig()
        if self.config.grid_size != 8:
            raise ValueError("OracleInterpreterV0 is frozen to 8x8 Stage-0 grids")
        hidden_dim = self.config.hidden_dim
        self.color_embedding = nn.Embedding(10, self.config.color_embedding_dim)
        self.color_projection = nn.Linear(self.config.color_embedding_dim, hidden_dim)
        self.row_embedding = nn.Embedding(self.config.grid_size, hidden_dim)
        self.column_embedding = nn.Embedding(self.config.grid_size, hidden_dim)
        self.workspace = nn.Parameter(torch.empty(1, self.config.workspace_tokens, hidden_dim))
        self.instruction_encoder = InstructionEncoder(hidden_dim)
        self.shared_execution_block = SharedExecutionBlock(self.config)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 10))
        self.feedback_projection = nn.Sequential(
            nn.Linear(self.config.color_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feedback_gate = nn.Parameter(torch.tensor(-2.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.workspace, mean=0.0, std=0.02)

    def _initial_state(self, input_grid: torch.Tensor) -> torch.Tensor:
        batch_size = input_grid.shape[0]
        spatial = self.color_projection(self.color_embedding(input_grid)).reshape(
            batch_size, self.config.grid_size * self.config.grid_size, self.config.hidden_dim
        )
        rows = torch.arange(self.config.grid_size, device=input_grid.device).repeat_interleave(
            self.config.grid_size
        )
        columns = torch.arange(self.config.grid_size, device=input_grid.device).repeat(
            self.config.grid_size
        )
        spatial = spatial + self.row_embedding(rows).unsqueeze(0) + self.column_embedding(columns).unsqueeze(0)
        return torch.cat((spatial, self.workspace.expand(batch_size, -1, -1)), dim=1)

    def forward(self, input_grid: torch.Tensor, typed_program_ids: torch.Tensor) -> torch.Tensor:
        if input_grid.ndim != 3 or input_grid.shape[1:] != (8, 8):
            raise ValueError("input_grid must have shape [batch, 8, 8]")
        if typed_program_ids.ndim != 3 or typed_program_ids.shape[0] != input_grid.shape[0]:
            raise ValueError("typed_program_ids must have shape [batch, program_length, 4]")
        if typed_program_ids.shape[2] != 4:
            raise ValueError("typed instructions must contain four integer factors")
        if not 1 <= typed_program_ids.shape[1] <= self.config.maximum_program_length:
            raise ValueError("program length is outside the frozen Stage-0 range")

        state = self._initial_state(input_grid)
        step_logits = []
        spatial_tokens = self.config.grid_size * self.config.grid_size
        for step_index in range(typed_program_ids.shape[1]):
            instruction = self.instruction_encoder(typed_program_ids[:, step_index])
            state = self.shared_execution_block(state, instruction)
            spatial = state[:, :spatial_tokens]
            logits = self.output_head(spatial)
            step_logits.append(
                logits.transpose(1, 2).reshape(
                    input_grid.shape[0], 10, self.config.grid_size, self.config.grid_size
                )
            )
            probabilities = logits.softmax(dim=-1)
            predicted_colors = probabilities @ self.color_embedding.weight
            feedback = self.feedback_projection(predicted_colors)
            spatial = spatial + torch.sigmoid(self.feedback_gate) * feedback
            state = torch.cat((spatial, state[:, spatial_tokens:]), dim=1)
        return torch.stack(step_logits, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
