from __future__ import annotations

from dataclasses import dataclass

from vrm2.data.types import Grid
from vrm2.oracle.types import TypedInstructionIds

PaddedTypedProgramIds = tuple[TypedInstructionIds, TypedInstructionIds, TypedInstructionIds]


@dataclass(frozen=True)
class SupportEpisode:
    episode_id: str
    support_input_grids: tuple[Grid, Grid, Grid]
    support_output_grids: tuple[Grid, Grid, Grid]
    target_program_length: int
    target_typed_program_ids: PaddedTypedProgramIds


@dataclass(frozen=True)
class QueryEpisode:
    episode_id: str
    query_input_grid: Grid
    query_target_grid: Grid
    evaluator_program_length: int
    evaluator_typed_program_ids: PaddedTypedProgramIds
