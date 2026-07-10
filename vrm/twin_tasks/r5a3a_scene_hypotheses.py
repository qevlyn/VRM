from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

Grid = tuple[tuple[int, ...], ...]

MAX_GRID_SIZE = 30
MIN_GRID_SIZE = 1
PALETTE = tuple(range(10))
MAX_SCENE_HYPOTHESES = 1


@dataclass(frozen=True)
class SceneHypothesis:
    hypothesis_id: str
    background_color: int
    connectivity: int

    def to_json(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "background_color": self.background_color,
            "connectivity": self.connectivity,
        }


@dataclass(frozen=True)
class ObjectRegion:
    object_id: str
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    touches_border: bool
    shape_hash: str

    def to_json(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "color": self.color,
            "cells": [list(cell) for cell in self.cells],
            "bbox": list(self.bbox),
            "touches_border": self.touches_border,
            "shape_hash": self.shape_hash,
            "area": len(self.cells),
        }


@dataclass(frozen=True)
class SceneParse:
    grid_shape: tuple[int, int]
    hypothesis: SceneHypothesis
    objects: tuple[ObjectRegion, ...]
    color_histogram: tuple[tuple[int, int], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "grid_shape": list(self.grid_shape),
            "hypothesis": self.hypothesis.to_json(),
            "objects": [obj.to_json() for obj in self.objects],
            "color_histogram": [[color, count] for color, count in self.color_histogram],
        }


def freeze_grid(raw: Iterable[Iterable[int]]) -> Grid:
    return tuple(tuple(int(value) for value in row) for row in raw)


def grid_to_list(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grid_shape(grid: Grid) -> tuple[int, int | None]:
    if not grid:
        return 0, 0
    widths = {len(row) for row in grid}
    if len(widths) != 1:
        return len(grid), None
    return len(grid), next(iter(widths))


def valid_raw_grid(grid: Grid) -> tuple[bool, list[str]]:
    height, width = grid_shape(grid)
    reasons: list[str] = []
    if width is None:
        reasons.append("ragged_grid")
    if not (MIN_GRID_SIZE <= height <= MAX_GRID_SIZE):
        reasons.append("height_outside_1_to_30")
    if width is None or not (MIN_GRID_SIZE <= int(width) <= MAX_GRID_SIZE):
        reasons.append("width_outside_1_to_30")
    values = [int(value) for row in grid for value in row]
    if any(value not in PALETTE for value in values):
        reasons.append("palette_outside_0_to_9")
    return not reasons, reasons


def canonical_grid(grid: Grid) -> dict[str, Any]:
    height, width = grid_shape(grid)
    return {
        "height": height,
        "width": width,
        "values_in_row_major_order": [int(value) for row in grid for value in row],
    }


def canonical_grid_sha256(grid: Grid) -> str:
    return stable_hash(canonical_grid(grid))


def scene_hypotheses_for_grid(grid: Grid) -> tuple[SceneHypothesis, ...]:
    ok, reasons = valid_raw_grid(grid)
    if not ok:
        raise ValueError(f"invalid raw ARC grid for scene hypotheses: {reasons}")
    values = [int(value) for row in grid for value in row]
    counts = Counter(values)
    candidates = [0, counts.most_common(1)[0][0]]
    candidates.extend(color for color, _ in counts.most_common() if color not in candidates)
    rows: list[SceneHypothesis] = []
    for background in candidates:
        for connectivity in (4, 8):
            hyp_id = f"bg={background}|conn={connectivity}"
            hyp = SceneHypothesis(hyp_id, int(background), int(connectivity))
            if hyp not in rows:
                rows.append(hyp)
            if len(rows) >= MAX_SCENE_HYPOTHESES:
                return tuple(rows)
    return tuple(rows)


def neighbors(x: int, y: int, height: int, width: int, connectivity: int) -> list[tuple[int, int]]:
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    if connectivity == 8:
        deltas.extend([(1, -1), (1, 1), (-1, 1), (-1, -1)])
    out = []
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            out.append((nx, ny))
    return out


def _shape_hash(cells: Iterable[tuple[int, int]]) -> str:
    rows = list(cells)
    if not rows:
        return stable_hash([])
    min_x = min(x for x, _ in rows)
    min_y = min(y for _, y in rows)
    normalized = sorted((x - min_x, y - min_y) for x, y in rows)
    return stable_hash(normalized)


def extract_objects(grid: Grid, hypothesis: SceneHypothesis) -> tuple[ObjectRegion, ...]:
    ok, reasons = valid_raw_grid(grid)
    if not ok:
        raise ValueError(f"invalid raw ARC grid for object extraction: {reasons}")
    height, width = grid_shape(grid)
    assert width is not None
    unseen = {
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if int(value) != hypothesis.background_color
    }
    objects: list[ObjectRegion] = []
    while unseen:
        start = min(unseen, key=lambda item: (item[1], item[0]))
        color = int(grid[start[1]][start[0]])
        queue = deque([start])
        unseen.remove(start)
        cells = {start}
        while queue:
            x, y = queue.popleft()
            for nx, ny in neighbors(x, y, height, width, hypothesis.connectivity):
                if (nx, ny) in unseen and int(grid[ny][nx]) == color:
                    unseen.remove((nx, ny))
                    cells.add((nx, ny))
                    queue.append((nx, ny))
        ordered = tuple(sorted(cells, key=lambda item: (item[1], item[0])))
        min_x = min(x for x, _ in ordered)
        max_x = max(x for x, _ in ordered)
        min_y = min(y for _, y in ordered)
        max_y = max(y for _, y in ordered)
        touches = any(x in {0, width - 1} or y in {0, height - 1} for x, y in ordered)
        object_id = f"color={color}|bbox={min_x},{min_y},{max_x},{max_y}|area={len(ordered)}|shape={_shape_hash(ordered)[:12]}"
        objects.append(
            ObjectRegion(
                object_id=object_id,
                color=color,
                cells=ordered,
                bbox=(min_x, min_y, max_x, max_y),
                touches_border=touches,
                shape_hash=_shape_hash(ordered),
            )
        )
    return tuple(sorted(objects, key=lambda obj: (obj.color, obj.bbox, len(obj.cells), obj.shape_hash)))


def parse_scene(grid: Grid, hypothesis: SceneHypothesis) -> SceneParse:
    values = [int(value) for row in grid for value in row]
    hist = tuple(sorted(Counter(values).items()))
    height, width = grid_shape(grid)
    assert width is not None
    return SceneParse((height, width), hypothesis, extract_objects(grid, hypothesis), hist)


def object_count(grid: Grid) -> int:
    hypothesis = scene_hypotheses_for_grid(grid)[0]
    return len(extract_objects(grid, hypothesis))


def bbox_for_cells(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = list(cells)
    if not rows:
        return None
    return min(x for x, _ in rows), min(y for _, y in rows), max(x for x, _ in rows), max(y for _, y in rows)
