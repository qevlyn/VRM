from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from vrm2.oracle.model import InstructionEncoder, OracleModelConfig, SharedExecutionBlock


class OracleInterpreterV1(nn.Module):
    """Shared interpreter with a hard canonical grid boundary after every step."""

    model_version = "v1"

    def __init__(self, config: OracleModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or OracleModelConfig()
        if self.config.grid_size != 8:
            raise ValueError("OracleInterpreterV1 is frozen to 8x8 Stage-0 grids")
        hidden_dim = self.config.hidden_dim
        self.color_embedding = nn.Embedding(10, self.config.color_embedding_dim)
        self.color_projection = nn.Linear(self.config.color_embedding_dim, hidden_dim)
        self.row_embedding = nn.Embedding(self.config.grid_size, hidden_dim)
        self.column_embedding = nn.Embedding(self.config.grid_size, hidden_dim)
        self.workspace_seed = nn.Parameter(
            torch.empty(1, self.config.workspace_tokens, hidden_dim)
        )
        self.workspace_from_grid = nn.Linear(hidden_dim, hidden_dim)
        self.workspace_from_instruction = nn.Linear(hidden_dim, hidden_dim)
        self.instruction_encoder = InstructionEncoder(hidden_dim)
        self.shared_execution_block = SharedExecutionBlock(self.config)
        self.output_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 10))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.workspace_seed, mean=0.0, std=0.02)

    def _validate_program_inputs(
        self, input_grid: torch.Tensor, typed_program_ids: torch.Tensor
    ) -> None:
        if input_grid.ndim != 3 or input_grid.shape[1:] != (8, 8):
            raise ValueError("input_grid must have shape [batch, 8, 8]")
        if typed_program_ids.ndim != 3 or typed_program_ids.shape[0] != input_grid.shape[0]:
            raise ValueError("typed_program_ids must have shape [batch, program_length, 4]")
        if typed_program_ids.shape[2] != 4:
            raise ValueError("typed instructions must contain four integer factors")
        if not 1 <= typed_program_ids.shape[1] <= self.config.maximum_program_length:
            raise ValueError("program length is outside the frozen Stage-0 range")

    def canonicalize_input_grid(self, input_grid: torch.Tensor) -> torch.Tensor:
        """Map integer colors to the public hard one-hot canonical representation."""

        if input_grid.ndim != 3 or input_grid.shape[1:] != (8, 8):
            raise ValueError("input_grid must have shape [batch, 8, 8]")
        return functional.one_hot(input_grid, num_classes=10).to(
            dtype=self.color_embedding.weight.dtype
        )

    def canonicalize_predicted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Use hard colors forward and a softmax straight-through gradient in training."""

        if logits.ndim != 4 or logits.shape[1:] != (10, 8, 8):
            raise ValueError("logits must have shape [batch, 10, 8, 8]")
        soft = logits.softmax(dim=1).permute(0, 2, 3, 1)
        hard = functional.one_hot(soft.argmax(dim=-1), num_classes=10).to(soft.dtype)
        if self.training:
            return hard + soft - soft.detach()
        return hard

    def encode_canonical_grid(
        self, canonical_grid: torch.Tensor, instruction: torch.Tensor
    ) -> torch.Tensor:
        """Construct fresh spatial and workspace activations for one instruction."""

        expected = (canonical_grid.shape[0], 8, 8, 10)
        if tuple(canonical_grid.shape) != expected:
            raise ValueError("canonical_grid must have shape [batch, 8, 8, 10]")
        if instruction.shape != (canonical_grid.shape[0], self.config.hidden_dim):
            raise ValueError("instruction embedding has an invalid shape")
        batch_size = canonical_grid.shape[0]
        color_features = canonical_grid @ self.color_embedding.weight
        spatial = self.color_projection(color_features).reshape(
            batch_size, self.config.grid_size * self.config.grid_size, self.config.hidden_dim
        )
        rows = torch.arange(self.config.grid_size, device=canonical_grid.device).repeat_interleave(
            self.config.grid_size
        )
        columns = torch.arange(self.config.grid_size, device=canonical_grid.device).repeat(
            self.config.grid_size
        )
        spatial = (
            spatial
            + self.row_embedding(rows).unsqueeze(0)
            + self.column_embedding(columns).unsqueeze(0)
        )
        grid_summary = spatial.mean(dim=1)
        workspace_update = self.workspace_from_grid(grid_summary) + self.workspace_from_instruction(
            instruction
        )
        workspace = self.workspace_seed.expand(batch_size, -1, -1) + workspace_update.unsqueeze(1)
        return torch.cat((spatial, workspace), dim=1)

    def execute_canonical_step(
        self, canonical_grid: torch.Tensor, instruction_ids: torch.Tensor
    ) -> torch.Tensor:
        """Execute one typed instruction from only a canonical current grid."""

        if instruction_ids.ndim != 2 or instruction_ids.shape != (canonical_grid.shape[0], 4):
            raise ValueError("instruction_ids must have shape [batch, 4]")
        instruction = self.instruction_encoder(instruction_ids)
        fresh_state = self.encode_canonical_grid(canonical_grid, instruction)
        next_state = self.shared_execution_block(fresh_state, instruction)
        spatial = next_state[:, : self.config.grid_size * self.config.grid_size]
        logits = self.output_head(spatial)
        return logits.transpose(1, 2).reshape(
            canonical_grid.shape[0], 10, self.config.grid_size, self.config.grid_size
        )

    def execute_integer_step(
        self, input_grid: torch.Tensor, instruction_ids: torch.Tensor
    ) -> torch.Tensor:
        """Run a transition through the same fresh route used at rollout boundaries."""

        return self.execute_canonical_step(
            self.canonicalize_input_grid(input_grid), instruction_ids
        )

    def forward(self, input_grid: torch.Tensor, typed_program_ids: torch.Tensor) -> torch.Tensor:
        self._validate_program_inputs(input_grid, typed_program_ids)
        canonical_grid = self.canonicalize_input_grid(input_grid)
        step_logits = []
        for step_index in range(typed_program_ids.shape[1]):
            logits = self.execute_canonical_step(
                canonical_grid, typed_program_ids[:, step_index]
            )
            step_logits.append(logits)
            if step_index + 1 < typed_program_ids.shape[1]:
                canonical_grid = self.canonicalize_predicted_logits(logits)
        return torch.stack(step_logits, dim=1)
