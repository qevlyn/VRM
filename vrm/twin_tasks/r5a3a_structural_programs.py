from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .r5a3a_scene_hypotheses import (
    Grid,
    ObjectRegion,
    SceneHypothesis,
    bbox_for_cells,
    canonical_grid_sha256,
    extract_objects,
    freeze_grid,
    grid_shape,
    stable_hash,
)

MAX_PROGRAM_DEPTH = 2
MAX_PROGRAM_ENUMERATION_COUNT = 8
K_VALUES = [1, 4, 16, 64, 128]
SELECTOR_FAMILIES = (
    "all_objects",
    "objects_by_color",
    "largest_object",
    "smallest_object",
    "unique_object_by_shape",
    "unique_object_by_color",
    "border_touching_objects",
    "objects_nearest_to_marker",
)
TRANSFORM_FAMILIES = (
    "keep",
    "delete",
    "recolor",
    "translate_by_inferred_vector",
    "copy_by_inferred_vector",
    "rotate_90_180_270",
    "reflect_horizontal_vertical",
    "fill_bounding_box",
    "outline_bounding_box",
)
RENDER_FAMILIES = (
    "same_canvas",
    "selected_object_bounding_box_canvas",
    "union_of_selected_objects_bounding_box_canvas",
)


@dataclass(frozen=True)
class SelectorSpec:
    kind: str
    value: Any = None

    def program_id(self) -> str:
        return self.kind if self.value is None else f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    value: Any = None

    def program_id(self) -> str:
        return self.kind if self.value is None else f"{self.kind}:{self.value}"


@dataclass(frozen=True)
class StructuralProgram:
    program_id: str
    selector: SelectorSpec
    transforms: tuple[TransformSpec, ...]
    render: str
    depth: int

    def to_json(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "selector": self.selector.program_id(),
            "transforms": [transform.program_id() for transform in self.transforms],
            "render": self.render,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class ExecutionTrace:
    program_id: str
    selected_object_ids: tuple[str, ...]
    changed_cells: tuple[tuple[int, int], ...]
    output_hash: str
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "selected_object_ids": list(self.selected_object_ids),
            "changed_cells": [list(cell) for cell in self.changed_cells],
            "output_hash": self.output_hash,
            "status": self.status,
        }


def _colors_from_grids(grids: Iterable[Grid]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for grid in grids for row in grid for value in row}))


def _object_shape_counts(grids: Iterable[Grid], hypothesis: SceneHypothesis) -> Counter[str]:
    counts: Counter[str] = Counter()
    for grid in grids:
        for obj in extract_objects(grid, hypothesis):
            counts[obj.shape_hash] += 1
    return counts


def infer_vectors(support_inputs: tuple[Grid, ...], support_outputs: tuple[Grid, ...], hypothesis: SceneHypothesis) -> tuple[tuple[int, int], ...]:
    vectors: Counter[tuple[int, int]] = Counter()
    for before, after in zip(support_inputs, support_outputs):
        before_objects = extract_objects(before, hypothesis)
        after_objects = extract_objects(after, hypothesis)
        for b_obj in before_objects:
            matches = [a_obj for a_obj in after_objects if a_obj.color == b_obj.color and len(a_obj.cells) == len(b_obj.cells)]
            for a_obj in matches:
                dx = a_obj.bbox[0] - b_obj.bbox[0]
                dy = a_obj.bbox[1] - b_obj.bbox[1]
                if (dx, dy) != (0, 0) and abs(dx) <= 30 and abs(dy) <= 30:
                    vectors[(dx, dy)] += 1
    if not vectors:
        return ((1, 0), (0, 1), (-1, 0), (0, -1))
    return tuple(vector for vector, _ in vectors.most_common(8))


def enumerate_selectors(
    grids: tuple[Grid, ...],
    hypothesis: SceneHypothesis,
    *,
    allow_relations: bool = True,
) -> tuple[SelectorSpec, ...]:
    colors = _colors_from_grids(grids)
    selectors: list[SelectorSpec] = [SelectorSpec("all_objects")]
    selectors.extend(SelectorSpec("objects_by_color", color) for color in colors)
    selectors.extend([SelectorSpec("largest_object"), SelectorSpec("smallest_object")])
    if allow_relations:
        selectors.append(SelectorSpec("border_touching_objects"))
        counts = _object_shape_counts(grids, hypothesis)
        selectors.extend(SelectorSpec("unique_object_by_shape", shape) for shape, count in sorted(counts.items())[:6] if count >= 1)
        selectors.extend(SelectorSpec("unique_object_by_color", color) for color in colors)
        selectors.extend(SelectorSpec("objects_nearest_to_marker", color) for color in colors)
    seen = set()
    ordered = []
    for selector in selectors:
        key = selector.program_id()
        if key not in seen:
            seen.add(key)
            ordered.append(selector)
    return tuple(ordered)


def enumerate_transforms(
    support_inputs: tuple[Grid, ...],
    support_outputs: tuple[Grid, ...],
    query_inputs: tuple[Grid, ...],
    hypothesis: SceneHypothesis,
) -> tuple[TransformSpec, ...]:
    colors = _colors_from_grids(support_inputs + support_outputs + query_inputs)
    raw_output_colors = _colors_from_grids(support_outputs) or colors
    output_colors = tuple([color for color in raw_output_colors if color != hypothesis.background_color] + [color for color in raw_output_colors if color == hypothesis.background_color])
    transforms: list[TransformSpec] = [TransformSpec("keep"), TransformSpec("delete")]
    transforms.extend(TransformSpec("recolor", color) for color in output_colors)
    for vector in infer_vectors(support_inputs, support_outputs, hypothesis):
        transforms.append(TransformSpec("translate", vector))
        transforms.append(TransformSpec("copy", vector))
    transforms.extend(TransformSpec("rotate", angle) for angle in (90, 180, 270))
    transforms.extend(TransformSpec("reflect", axis) for axis in ("horizontal", "vertical"))
    for color in output_colors:
        transforms.append(TransformSpec("fill_bbox", color))
        transforms.append(TransformSpec("outline_bbox", color))
    seen = set()
    ordered = []
    for transform in transforms:
        key = transform.program_id()
        if key not in seen:
            seen.add(key)
            ordered.append(transform)
    return tuple(ordered)


def enumerate_programs(
    support_inputs: tuple[Grid, ...],
    support_outputs: tuple[Grid, ...],
    query_inputs: tuple[Grid, ...],
    hypothesis: SceneHypothesis,
    *,
    max_depth: int = MAX_PROGRAM_DEPTH,
    allow_relations: bool = True,
    allow_changed_shape_render: bool = True,
    identity_only: bool = False,
    max_programs: int = MAX_PROGRAM_ENUMERATION_COUNT,
) -> tuple[StructuralProgram, ...]:
    render_modes = ["same_canvas"]
    if allow_changed_shape_render:
        render_modes.extend(["selected_bbox_canvas", "union_bbox_canvas"])
    if identity_only:
        program = StructuralProgram(
            "select=all_objects|ops=keep|render=same_canvas",
            SelectorSpec("all_objects"),
            (TransformSpec("keep"),),
            "same_canvas",
            1,
        )
        return (program,)
    selectors = enumerate_selectors(support_inputs + query_inputs, hypothesis, allow_relations=allow_relations)
    transforms = enumerate_transforms(support_inputs, support_outputs, query_inputs, hypothesis)
    rows: list[StructuralProgram] = []

    def add_program(selector: SelectorSpec, ops: tuple[TransformSpec, ...], render: str) -> bool:
        program_id = f"select={selector.program_id()}|ops={'+'.join(op.program_id() for op in ops)}|render={render}"
        if any(existing.program_id == program_id for existing in rows):
            return len(rows) >= max_programs
        rows.append(StructuralProgram(program_id, selector, tuple(ops), render, len(ops)))
        return len(rows) >= max_programs

    for selector in selectors:
        by_kind: dict[str, list[TransformSpec]] = {}
        for transform in transforms:
            by_kind.setdefault(transform.kind, []).append(transform)
        keep = by_kind.get("keep", [transforms[0]])[0]
        recolor = by_kind.get("recolor", [keep])[0]
        translate = by_kind.get("translate", [keep])[0]
        copy = by_kind.get("copy", [keep])[0]
        rotate = by_kind.get("rotate", [keep])[0]
        reflect = by_kind.get("reflect", [keep])[0]
        priority: list[tuple[tuple[TransformSpec, ...], str]] = [
            ((keep,), "same_canvas"),
            ((keep,), "selected_bbox_canvas" if allow_changed_shape_render else "same_canvas"),
            ((recolor,), "same_canvas"),
            ((translate,), "same_canvas"),
            ((copy,), "same_canvas"),
            ((rotate,), "same_canvas"),
            ((reflect,), "same_canvas"),
        ]
        if max_depth >= 2:
            priority.append(((recolor, translate), "same_canvas"))
        for ops, render in priority:
            if add_program(selector, ops, render):
                return tuple(rows)
        for transform in transforms:
            if add_program(selector, (transform,), "same_canvas"):
                return tuple(rows)
        if allow_changed_shape_render:
            for transform in transforms[:4]:
                for render in ("selected_bbox_canvas", "union_bbox_canvas"):
                    if add_program(selector, (transform,), render):
                        return tuple(rows)
        if max_depth >= 2:
            priority = transforms[:8]
            for left, right in itertools.product(priority, repeat=2):
                if left.program_id() == right.program_id():
                    continue
                if add_program(selector, (left, right), "same_canvas"):
                    return tuple(rows)
    return tuple(rows)


def select_objects(objects: tuple[ObjectRegion, ...], selector: SelectorSpec) -> tuple[ObjectRegion, ...]:
    if selector.kind == "all_objects":
        return objects
    if selector.kind == "objects_by_color":
        return tuple(obj for obj in objects if obj.color == int(selector.value))
    if selector.kind == "largest_object":
        if not objects:
            return tuple()
        max_area = max(len(obj.cells) for obj in objects)
        return tuple(obj for obj in objects if len(obj.cells) == max_area)
    if selector.kind == "smallest_object":
        if not objects:
            return tuple()
        min_area = min(len(obj.cells) for obj in objects)
        return tuple(obj for obj in objects if len(obj.cells) == min_area)
    if selector.kind == "unique_object_by_shape":
        matches = tuple(obj for obj in objects if obj.shape_hash == selector.value)
        return matches if len(matches) == 1 else tuple()
    if selector.kind == "unique_object_by_color":
        matches = tuple(obj for obj in objects if obj.color == int(selector.value))
        return matches if len(matches) == 1 else tuple()
    if selector.kind == "border_touching_objects":
        return tuple(obj for obj in objects if obj.touches_border)
    if selector.kind == "objects_nearest_to_marker":
        markers = [obj for obj in objects if obj.color == int(selector.value)]
        others = [obj for obj in objects if obj.color != int(selector.value)]
        if not markers or not others:
            return tuple()
        marker_cells = [cell for marker in markers for cell in marker.cells]
        scored = []
        for obj in others:
            dist = min(abs(x - mx) + abs(y - my) for x, y in obj.cells for mx, my in marker_cells)
            scored.append((dist, obj.object_id, obj))
        best = min(score for score, _, _ in scored)
        return tuple(obj for score, _, obj in sorted(scored) if score == best)
    return tuple()


def _blank_like(grid: Grid, color: int) -> list[list[int]]:
    height, width = grid_shape(grid)
    assert width is not None
    return [[color for _ in range(width)] for _ in range(height)]


def _put_cell(canvas: list[list[int]], x: int, y: int, color: int) -> None:
    if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
        canvas[y][x] = int(color)


def _transform_cells(cells: tuple[tuple[int, int], ...], bbox: tuple[int, int, int, int], transform: TransformSpec) -> tuple[tuple[int, int], ...]:
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    if transform.kind == "translate" or transform.kind == "copy":
        dx, dy = transform.value
        return tuple((x + int(dx), y + int(dy)) for x, y in cells)
    if transform.kind == "rotate":
        angle = int(transform.value)
        out = []
        for x, y in cells:
            lx, ly = x - min_x, y - min_y
            if angle == 90:
                out.append((min_x + height - ly, min_y + lx))
            elif angle == 180:
                out.append((min_x + width - lx, min_y + height - ly))
            elif angle == 270:
                out.append((min_x + ly, min_y + width - lx))
        return tuple(out)
    if transform.kind == "reflect":
        if transform.value == "horizontal":
            return tuple((min_x + (max_x - x), y) for x, y in cells)
        return tuple((x, min_y + (max_y - y)) for x, y in cells)
    return cells


def execute_program(grid: Grid, hypothesis: SceneHypothesis, program: StructuralProgram) -> tuple[Grid, ExecutionTrace]:
    objects = extract_objects(grid, hypothesis)
    selected = select_objects(objects, program.selector)
    selected_cells = tuple(cell for obj in selected for cell in obj.cells)
    canvas = [list(row) for row in grid]
    affected = set(selected_cells)
    current_cells = tuple(selected_cells)
    current_colors = {cell: int(grid[cell[1]][cell[0]]) for cell in selected_cells}
    bbox = bbox_for_cells(current_cells)
    status = "OK"
    for transform in program.transforms:
        if not current_cells and transform.kind != "keep":
            status = "EMPTY_SELECTION"
            continue
        if transform.kind == "keep":
            continue
        if transform.kind == "delete":
            for x, y in current_cells:
                _put_cell(canvas, x, y, hypothesis.background_color)
                affected.add((x, y))
            current_cells = tuple()
            current_colors = {}
            bbox = None
            continue
        if bbox is None:
            status = "EMPTY_SELECTION"
            continue
        if transform.kind == "recolor":
            for x, y in current_cells:
                _put_cell(canvas, x, y, int(transform.value))
                current_colors[(x, y)] = int(transform.value)
                affected.add((x, y))
            continue
        if transform.kind in {"translate", "rotate", "reflect"}:
            for x, y in current_cells:
                _put_cell(canvas, x, y, hypothesis.background_color)
                affected.add((x, y))
            new_cells = _transform_cells(current_cells, bbox, transform)
            new_colors = {}
            for old, new in zip(current_cells, new_cells):
                color = current_colors.get(old, int(grid[old[1]][old[0]]))
                _put_cell(canvas, new[0], new[1], color)
                affected.add(new)
                new_colors[new] = color
            current_cells = tuple(sorted(new_colors))
            current_colors = new_colors
            bbox = bbox_for_cells(current_cells)
            continue
        if transform.kind == "copy":
            new_cells = _transform_cells(current_cells, bbox, transform)
            for old, new in zip(current_cells, new_cells):
                color = current_colors.get(old, int(grid[old[1]][old[0]]))
                _put_cell(canvas, new[0], new[1], color)
                affected.add(new)
            continue
        if transform.kind in {"fill_bbox", "outline_bbox"}:
            min_x, min_y, max_x, max_y = bbox
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    if transform.kind == "fill_bbox" or x in {min_x, max_x} or y in {min_y, max_y}:
                        _put_cell(canvas, x, y, int(transform.value))
                        affected.add((x, y))
            current_cells = tuple((x, y) for y in range(min_y, max_y + 1) for x in range(min_x, max_x + 1))
            current_colors = {cell: int(transform.value) for cell in current_cells}
            bbox = bbox_for_cells(current_cells)
            continue
    output = freeze_grid(canvas)
    if program.render in {"selected_bbox_canvas", "union_bbox_canvas"}:
        crop_bbox = bbox_for_cells(current_cells if program.render == "selected_bbox_canvas" else affected)
        if crop_bbox is None:
            output = ((hypothesis.background_color,),)
        else:
            min_x, min_y, max_x, max_y = crop_bbox
            output = freeze_grid([row[min_x : max_x + 1] for row in output[min_y : max_y + 1]])
    trace = ExecutionTrace(
        program_id=program.program_id,
        selected_object_ids=tuple(obj.object_id for obj in selected),
        changed_cells=tuple(sorted(cell for cell in affected if 0 <= cell[1] < len(grid) and 0 <= cell[0] < len(grid[0]))),
        output_hash=canonical_grid_sha256(output),
        status=status,
    )
    return output, trace


def program_surface_spec() -> dict[str, Any]:
    payload = {
        "K": K_VALUES,
        "maximum_program_depth": MAX_PROGRAM_DEPTH,
        "maximum_program_enumeration_count_before_ranking": MAX_PROGRAM_ENUMERATION_COUNT,
        "select_operator_family": SELECTOR_FAMILIES,
        "transform_operator_family": TRANSFORM_FAMILIES,
        "render_operator_family": RENDER_FAMILIES,
    }
    payload["surface_sha256"] = stable_hash(payload)
    return payload
