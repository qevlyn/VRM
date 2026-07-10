from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .r5a3a_scene_hypotheses import (
    Grid,
    SceneHypothesis,
    canonical_grid_sha256,
    extract_objects,
    freeze_grid,
    grid_shape,
    stable_hash,
    valid_raw_grid,
)

MAX_PARSE_HYPOTHESES = 4
PARSE_MENU = {
    "background_candidates": ["modal_color", "color_0_when_distinct_from_modal"],
    "connectivity": [4, 8],
}


@dataclass(frozen=True)
class ParseState:
    parse_id: str
    background_color: int
    connectivity: int
    object_count: int
    object_state_hash: str
    objects: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "parse_id": self.parse_id,
            "background_color": self.background_color,
            "connectivity": self.connectivity,
            "object_count": self.object_count,
            "object_state_hash": self.object_state_hash,
            "objects": list(self.objects),
        }


def modal_color(grid: Grid) -> int:
    values = [int(value) for row in grid for value in row]
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def parse_background_candidates(grid: Grid) -> list[int]:
    modal = modal_color(grid)
    rows = [modal]
    if modal != 0:
        rows.append(0)
    return rows


def object_json_with_masks(grid: Grid, hypothesis: SceneHypothesis) -> tuple[dict[str, Any], ...]:
    height, width = grid_shape(grid)
    assert width is not None
    rows = []
    for obj in extract_objects(grid, hypothesis):
        min_x, min_y, max_x, max_y = obj.bbox
        mask = [[0 for _ in range(max_x - min_x + 1)] for _ in range(max_y - min_y + 1)]
        for x, y in obj.cells:
            mask[y - min_y][x - min_x] = 1
        rows.append(
            {
                "object_id": obj.object_id,
                "color": obj.color,
                "cell_count": len(obj.cells),
                "bounding_box": list(obj.bbox),
                "height": max_y - min_y + 1,
                "width": max_x - min_x + 1,
                "mask_shape_canonicalization": mask,
                "top_left_position": [min_x, min_y],
                "centroid_approximation": [
                    sum(x for x, _ in obj.cells) / len(obj.cells),
                    sum(y for _, y in obj.cells) / len(obj.cells),
                ],
                "border_touch_flag": obj.touches_border,
                "hole_count": 0,
                "shape_hash": obj.shape_hash,
                "cells": [list(cell) for cell in obj.cells],
            }
        )
    return tuple(sorted(rows, key=lambda item: (item["color"], item["bounding_box"], item["shape_hash"])))


def enumerate_parse_states(grid: Grid) -> tuple[ParseState, ...]:
    ok, reasons = valid_raw_grid(grid)
    if not ok:
        raise ValueError(f"invalid raw grid for R5A-3B parse enumeration: {reasons}")
    rows = []
    seen = set()
    for background in parse_background_candidates(grid):
        for connectivity in (4, 8):
            hyp = SceneHypothesis(f"bg={background}|conn={connectivity}", int(background), int(connectivity))
            objects = object_json_with_masks(grid, hyp)
            object_state_hash = stable_hash(objects)
            if object_state_hash in seen:
                continue
            seen.add(object_state_hash)
            parse_id = f"bg={background}|conn={connectivity}|objects={len(objects)}|hash={object_state_hash[:12]}"
            rows.append(ParseState(parse_id, int(background), int(connectivity), len(objects), object_state_hash, objects))
            if len(rows) >= MAX_PARSE_HYPOTHESES:
                return tuple(rows)
    return tuple(rows)


def parse_report_for_grid(grid: Grid) -> dict[str, Any]:
    return {
        "grid_hash": canonical_grid_sha256(grid),
        "grid_shape": list(grid_shape(grid)),
        "parse_states": [state.to_json() for state in enumerate_parse_states(grid)],
    }
