from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from vrm2.data.types import Grid, OperationStep, make_grid

OPERATION_REGISTRY_VERSION = "vrm2-operations-v1"
OPERATION_NAMES = (
    "translate_leftmost",
    "translate_topmost",
    "recolor_leftmost",
    "recolor_topmost",
    "reflect_rows",
    "reflect_columns",
)

DIRECTIONS: Mapping[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class SceneValidationError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class SceneObject:
    cells: frozenset[tuple[int, int]]
    color: int

    @property
    def min_row(self) -> int:
        return min(row for row, _ in self.cells)

    @property
    def min_col(self) -> int:
        return min(col for _, col in self.cells)


def _neighbors(row: int, col: int) -> Iterable[tuple[int, int]]:
    yield row - 1, col
    yield row + 1, col
    yield row, col - 1
    yield row, col + 1


def find_objects(grid: Grid) -> tuple[SceneObject, ...]:
    size = len(grid)
    make_grid(grid, expected_size=size)
    occupied = {(row, col) for row in range(size) for col in range(size) if grid[row][col] != 0}
    if not occupied:
        raise SceneValidationError("empty_scene")

    unseen = set(occupied)
    objects: list[SceneObject] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        cells: set[tuple[int, int]] = set()
        while stack:
            cell = stack.pop()
            cells.add(cell)
            for neighbor in _neighbors(*cell):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        colors = {grid[row][col] for row, col in cells}
        if len(colors) != 1:
            raise SceneValidationError("objects_touching", "edge-connected colors do not define separate objects")
        objects.append(SceneObject(frozenset(cells), colors.pop()))
    return tuple(sorted(objects, key=lambda item: (item.min_row, item.min_col, item.color)))


def _select_unique(objects: tuple[SceneObject, ...], relation: str) -> SceneObject:
    if relation == "leftmost":
        values = [obj.min_col for obj in objects]
    elif relation == "topmost":
        values = [obj.min_row for obj in objects]
    else:
        raise ValueError(f"unknown relation: {relation}")
    best = min(values)
    candidates = [obj for obj, value in zip(objects, values, strict=True) if value == best]
    if len(candidates) != 1:
        raise SceneValidationError("selector_tie", f"{relation} relation is not unique")
    return candidates[0]


def _mutable(grid: Grid) -> list[list[int]]:
    return [list(row) for row in grid]


def _finish(before: Grid, rows: list[list[int]]) -> Grid:
    after = make_grid(rows, expected_size=len(before))
    if after == before:
        raise SceneValidationError("no_visible_change")
    return after


def _translate(grid: Grid, relation: str, parameters: Mapping[str, Any]) -> Grid:
    direction = parameters.get("direction")
    if direction not in DIRECTIONS:
        raise SceneValidationError("invalid_parameter", f"unknown direction {direction!r}")
    objects = find_objects(grid)
    selected = _select_unique(objects, relation)
    delta_row, delta_col = DIRECTIONS[str(direction)]
    moved = {(row + delta_row, col + delta_col) for row, col in selected.cells}
    size = len(grid)
    if any(row < 0 or row >= size or col < 0 or col >= size for row, col in moved):
        raise SceneValidationError("out_of_bounds")

    occupied_other = set().union(*(obj.cells for obj in objects if obj is not selected)) if len(objects) > 1 else set()
    if moved & occupied_other:
        raise SceneValidationError("object_overlap")
    if any(neighbor in occupied_other for cell in moved for neighbor in _neighbors(*cell)):
        raise SceneValidationError("objects_touching")

    rows = _mutable(grid)
    for row, col in selected.cells:
        rows[row][col] = 0
    for row, col in moved:
        rows[row][col] = selected.color
    return _finish(grid, rows)


def _recolor(grid: Grid, relation: str, parameters: Mapping[str, Any]) -> Grid:
    color = parameters.get("color")
    if not isinstance(color, int) or color < 1 or color > 9:
        raise SceneValidationError("invalid_parameter", f"invalid color {color!r}")
    selected = _select_unique(find_objects(grid), relation)
    if color == selected.color:
        raise SceneValidationError("irrelevant_parameter", "sampled color equals selected object color")
    rows = _mutable(grid)
    for row, col in selected.cells:
        rows[row][col] = color
    return _finish(grid, rows)


def _reflect(grid: Grid, axis: str) -> Grid:
    occupied = [(row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value != 0]
    if not occupied:
        raise SceneValidationError("empty_scene")
    min_row = min(row for row, _ in occupied)
    max_row = max(row for row, _ in occupied)
    min_col = min(col for _, col in occupied)
    max_col = max(col for _, col in occupied)
    rows = [[0 for _ in row] for row in grid]
    for row, col in occupied:
        if axis == "rows":
            target = (min_row + max_row - row, col)
        elif axis == "columns":
            target = (row, min_col + max_col - col)
        else:
            raise ValueError(f"unknown reflection axis: {axis}")
        target_row, target_col = target
        rows[target_row][target_col] = grid[row][col]
    return _finish(grid, rows)


def execute_operation(grid: Grid, step: OperationStep) -> Grid:
    if step.name == "translate_leftmost":
        return _translate(grid, "leftmost", step.parameters)
    if step.name == "translate_topmost":
        return _translate(grid, "topmost", step.parameters)
    if step.name == "recolor_leftmost":
        return _recolor(grid, "leftmost", step.parameters)
    if step.name == "recolor_topmost":
        return _recolor(grid, "topmost", step.parameters)
    if step.name == "reflect_rows":
        return _reflect(grid, "rows")
    if step.name == "reflect_columns":
        return _reflect(grid, "columns")
    raise SceneValidationError("unknown_operation", step.name)


def execute_program(grid: Grid, steps: tuple[OperationStep, ...]) -> Grid:
    states = [grid]
    for step in steps:
        states.append(execute_operation(states[-1], step))
        if len(states) >= 3 and states[-1] == states[-3]:
            raise SceneValidationError("consecutive_cancellation")
    return states[-1]
