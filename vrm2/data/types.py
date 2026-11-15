from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, TypeAlias

from vrm2.data.program_schema import ProgramInstruction

Grid: TypeAlias = tuple[tuple[int, ...], ...]
Split: TypeAlias = Literal["train", "dev", "sealed"]


@dataclass(frozen=True)
class GridPair:
    input_grid: Grid
    output_grid: Grid


@dataclass(frozen=True)
class OperationStep:
    name: str
    parameters: Mapping[str, Any]
    typed_instruction: ProgramInstruction | None = None


@dataclass(frozen=True)
class HiddenProgram:
    composition_signature: str
    steps: tuple[OperationStep, ...]

    @property
    def typed_instructions(self) -> tuple[ProgramInstruction, ...]:
        instructions = tuple(step.typed_instruction for step in self.steps)
        if any(instruction is None for instruction in instructions):
            raise ValueError("hidden program has not been factorized into typed instructions")
        return tuple(instruction for instruction in instructions if instruction is not None)


@dataclass(frozen=True)
class TaskEpisode:
    support: tuple[GridPair, ...]
    query_input: Grid
    query_output: Grid
    hidden_program: HiddenProgram
    episode_seed: int
    split: Split


@dataclass(frozen=True)
class EpisodeDiagnostics:
    episode_attempt: int
    scene_attempts: tuple[int, ...]
    rejection_counts: Mapping[str, int]


def make_grid(rows: Iterable[Iterable[int]], expected_size: int | None = None) -> Grid:
    grid = tuple(tuple(int(value) for value in row) for row in rows)
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be a non-empty rectangle")
    if expected_size is not None and (len(grid) != expected_size or len(grid[0]) != expected_size):
        raise ValueError(f"grid must have shape {expected_size}x{expected_size}")
    if any(value < 0 or value > 9 for row in grid for value in row):
        raise ValueError("grid colors must be integers in [0, 9]")
    return grid


def grid_as_lists(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]
