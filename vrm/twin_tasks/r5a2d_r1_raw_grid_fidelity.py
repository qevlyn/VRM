from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]


def freeze_grid(raw: Iterable[Iterable[int]]) -> Grid:
    return tuple(tuple(int(value) for value in row) for row in raw)


def grid_shape(grid: Grid) -> tuple[int, int | None]:
    height = len(grid)
    if height == 0:
        return 0, 0
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        return height, None
    return height, next(iter(widths))


def canonical_grid(grid: Grid) -> dict[str, Any]:
    height, width = grid_shape(grid)
    return {
        "height": height,
        "width": width,
        "row_lengths": [len(row) for row in grid],
        "values_in_row_major_order": [int(value) for row in grid for value in row],
    }


def canonical_grid_sha256(grid: Grid) -> str:
    encoded = json.dumps(canonical_grid(grid), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RawGridFidelityEvidence:
    context: str
    canonical_source_grid_sha256: str
    canonical_expert_grid_sha256: str
    same_height: bool
    same_width: bool
    same_row_major_order: bool
    same_cell_value_at_every_coordinate: bool
    passed: bool
    source_height: int
    source_width: int | None
    expert_height: int
    expert_width: int | None
    mismatch_count: int
    first_mismatch: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "canonical_source_grid_sha256": self.canonical_source_grid_sha256,
            "canonical_expert_grid_sha256": self.canonical_expert_grid_sha256,
            "same_height": self.same_height,
            "same_width": self.same_width,
            "same_row_major_order": self.same_row_major_order,
            "same_cell_value_at_every_coordinate": self.same_cell_value_at_every_coordinate,
            "passed": self.passed,
            "source_height": self.source_height,
            "source_width": self.source_width,
            "expert_height": self.expert_height,
            "expert_width": self.expert_width,
            "mismatch_count": self.mismatch_count,
            "first_mismatch": self.first_mismatch,
        }


def compare_raw_grid_fidelity(source_grid: Grid, expert_grid: Grid, *, context: str) -> RawGridFidelityEvidence:
    source_height, source_width = grid_shape(source_grid)
    expert_height, expert_width = grid_shape(expert_grid)
    same_height = source_height == expert_height
    same_width = source_width == expert_width

    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    max_height = max(source_height, expert_height)
    for y in range(max_height):
        source_row = source_grid[y] if y < source_height else None
        expert_row = expert_grid[y] if y < expert_height else None
        max_width = max(len(source_row) if source_row is not None else 0, len(expert_row) if expert_row is not None else 0)
        for x in range(max_width):
            source_present = source_row is not None and x < len(source_row)
            expert_present = expert_row is not None and x < len(expert_row)
            source_value = int(source_row[x]) if source_present else None
            expert_value = int(expert_row[x]) if expert_present else None
            if source_value != expert_value:
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "x": x,
                        "y": y,
                        "source_present": source_present,
                        "expert_present": expert_present,
                        "source_value": source_value,
                        "expert_value": expert_value,
                    }

    source_hash = canonical_grid_sha256(source_grid)
    expert_hash = canonical_grid_sha256(expert_grid)
    same_row_major_order = canonical_grid(source_grid)["values_in_row_major_order"] == canonical_grid(expert_grid)["values_in_row_major_order"]
    same_cells = mismatch_count == 0
    passed = same_height and same_width and same_row_major_order and same_cells and source_hash == expert_hash
    return RawGridFidelityEvidence(
        context=context,
        canonical_source_grid_sha256=source_hash,
        canonical_expert_grid_sha256=expert_hash,
        same_height=same_height,
        same_width=same_width,
        same_row_major_order=same_row_major_order,
        same_cell_value_at_every_coordinate=same_cells,
        passed=passed,
        source_height=source_height,
        source_width=source_width,
        expert_height=expert_height,
        expert_width=expert_width,
        mismatch_count=mismatch_count,
        first_mismatch=first_mismatch,
    )


def fidelity_json(source_grid: Grid, expert_grid: Grid, *, context: str) -> dict[str, Any]:
    return compare_raw_grid_fidelity(source_grid, expert_grid, context=context).to_json()
