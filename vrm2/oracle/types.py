from __future__ import annotations

from dataclasses import dataclass

import torch

from vrm2.data.types import Grid, Split

TypedInstructionIds = tuple[int, int, int, int]
TypedProgramIds = tuple[TypedInstructionIds, ...]


@dataclass(frozen=True)
class OracleExample:
    input_grid: Grid
    typed_program_ids: TypedProgramIds
    target_output_grid: Grid
    intermediate_target_grids: tuple[Grid, ...]
    episode_seed: int
    split: Split
    composition_signature: str
    diagnostic_bucket: str

    @property
    def program_length(self) -> int:
        return len(self.typed_program_ids)


@dataclass(frozen=True)
class OracleBatch:
    input_grid: torch.Tensor
    typed_program_ids: torch.Tensor
    target_output_grid: torch.Tensor
    intermediate_target_grids: torch.Tensor

    def model_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_grid, self.typed_program_ids

    def to(self, device: torch.device) -> "OracleBatch":
        return OracleBatch(
            input_grid=self.input_grid.to(device, non_blocking=True),
            typed_program_ids=self.typed_program_ids.to(device, non_blocking=True),
            target_output_grid=self.target_output_grid.to(device, non_blocking=True),
            intermediate_target_grids=self.intermediate_target_grids.to(device, non_blocking=True),
        )
