from __future__ import annotations

import itertools
import json
from typing import Any

GRID_SIZE = 7
BACKGROUND = 0
INPUT_COLORS = (1, 2, 3, 4)
OUTPUT_COLORS = (5, 6, 7, 8)
COLOR_COUNT = 9

OP_C = "C"
OP_F = "F"
OP_L = "L"
OP_U = "U"

SINGLE_CLASSES = (OP_C, OP_F, OP_L, OP_U)
SEEN_MIXED_CLASSES = ("C->L", "F->U", "L->F", "U->C")
HELD_OUT_MIXED_CLASSES = ("L->C", "U->F", "F->L", "C->U")
MIXED_CLASSES = SEEN_MIXED_CLASSES + HELD_OUT_MIXED_CLASSES
TRAIN_VISIBLE_CLASSES = SINGLE_CLASSES + SEEN_MIXED_CLASSES
SEALED_CLASSES = TRAIN_VISIBLE_CLASSES + HELD_OUT_MIXED_CLASSES
COUNTERFACTUAL_CLASSES = MIXED_CLASSES


def parse_sequence(sequence_class: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in sequence_class.split("->"))
    if not parts or any(part not in SINGLE_CLASSES for part in parts):
        raise ValueError(f"unknown R5A-0 operation class {sequence_class}")
    return parts


def coord_c(x: int, y: int) -> tuple[int, int]:
    return (GRID_SIZE - 1 - y, x)


def coord_f(x: int, y: int) -> tuple[int, int]:
    return (GRID_SIZE - 1 - x, y)


def apply_primitive_to_cell(op: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    if color == BACKGROUND:
        return x, y, color
    if op == OP_C:
        out_x, out_y = coord_c(x, y)
        return out_x, out_y, color
    if op == OP_F:
        out_x, out_y = coord_f(x, y)
        return out_x, out_y, color
    if op == OP_L:
        if x <= 3 and color in INPUT_COLORS:
            return x, y, color + 4
        return x, y, color
    if op == OP_U:
        if y <= 3 and color in INPUT_COLORS:
            return x, y, color + 4
        return x, y, color
    raise ValueError(f"unknown primitive {op}")


def apply_sequence_to_cell(sequence_class: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    out_x, out_y, out_color = x, y, color
    for op in parse_sequence(sequence_class):
        out_x, out_y, out_color = apply_primitive_to_cell(op, out_x, out_y, out_color)
        if not (0 <= out_x < GRID_SIZE and 0 <= out_y < GRID_SIZE):
            raise ValueError(f"{sequence_class} produced out-of-bounds coordinate {(out_x, out_y)}")
    return out_x, out_y, out_color


def empty_grid() -> list[list[int]]:
    return [[BACKGROUND for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def freeze_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def grid_to_list(grid: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(row) for row in grid]


def parse_grid(raw: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return freeze_grid([[int(value) for value in row] for row in raw])


def foreground_coords(grid: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]]:
    return [(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value != BACKGROUND]


def no_edge_foreground(grid: tuple[tuple[int, ...], ...]) -> bool:
    return all(0 < x < GRID_SIZE - 1 and 0 < y < GRID_SIZE - 1 for x, y in foreground_coords(grid))


def apply_sequence_to_grid(grid: tuple[tuple[int, ...], ...], sequence_class: str) -> tuple[tuple[int, ...], ...]:
    result = empty_grid()
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == BACKGROUND:
                continue
            out_x, out_y, out_color = apply_sequence_to_cell(sequence_class, x, y, int(color))
            if result[out_y][out_x] != BACKGROUND:
                raise ValueError("operation produced colliding foreground cells")
            result[out_y][out_x] = int(out_color)
    return freeze_grid(result)


def source_index_for_output(sequence_class: str, out_x: int, out_y: int) -> int:
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            try:
                cand_x, cand_y, _ = apply_sequence_to_cell(sequence_class, x, y, 1)
            except ValueError:
                continue
            if (cand_x, cand_y) == (out_x, out_y):
                return y * GRID_SIZE + x
    return -1


def source_coord_for_output(sequence_class: str, out_x: int, out_y: int) -> tuple[int, int] | None:
    index = source_index_for_output(sequence_class, out_x, out_y)
    if index < 0:
        return None
    y, x = divmod(index, GRID_SIZE)
    return x, y


def cell_signature(sequence_class: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for y, x, color in itertools.product(range(GRID_SIZE), range(GRID_SIZE), INPUT_COLORS):
        out_x, out_y, out_color = apply_sequence_to_cell(sequence_class, x, y, color)
        rows.append([x, y, color, out_x, out_y, out_color])
    return rows


def stable_signature_key(sequence_class: str) -> str:
    return json.dumps(cell_signature(sequence_class), sort_keys=True, separators=(",", ":"))


def operation_function_audit() -> dict[str, Any]:
    all_classes = SINGLE_CLASSES + MIXED_CLASSES
    signatures = {sequence: stable_signature_key(sequence) for sequence in all_classes}
    collisions = [
        {"left": left, "right": right}
        for left, right in itertools.combinations(all_classes, 2)
        if signatures[left] == signatures[right]
    ]
    mixed_unique = len({signatures[sequence] for sequence in MIXED_CLASSES}) == len(MIXED_CLASSES)
    no_heldout_equals_seen = all(signatures[heldout] != signatures[seen] for heldout in HELD_OUT_MIXED_CLASSES for seen in SEEN_MIXED_CLASSES)
    no_mixed_equals_single = all(signatures[mixed] != signatures[single] for mixed in MIXED_CLASSES for single in SINGLE_CLASSES)
    no_heldout_equals_train_visible = all(signatures[heldout] != signatures[visible] for heldout in HELD_OUT_MIXED_CLASSES for visible in TRAIN_VISIBLE_CLASSES)
    passed = mixed_unique and no_heldout_equals_seen and no_mixed_equals_single and no_heldout_equals_train_visible and not collisions
    return {
        "primitive_definitions": {
            "C": "ROT90_CW: (x,y,c)->(6-y,x,c)",
            "F": "HFLIP: (x,y,c)->(6-x,y,c)",
            "L": "LEFT_REGION_RECOLOR: if current x <= 3 and c in {1,2,3,4}, c->c+4",
            "U": "UPPER_REGION_RECOLOR: if current y <= 3 and c in {1,2,3,4}, c->c+4",
            "sequence_notation": "A->B means apply A first, then B",
        },
        "signature_domain": {
            "x": list(range(GRID_SIZE)),
            "y": list(range(GRID_SIZE)),
            "foreground_input_colors": list(INPUT_COLORS),
            "signature_entry": "(x,y,c)->(x_final,y_final,c_final)",
        },
        "single_classes": list(SINGLE_CLASSES),
        "seen_mixed_classes": list(SEEN_MIXED_CLASSES),
        "held_out_mixed_classes": list(HELD_OUT_MIXED_CLASSES),
        "mixed_signatures_all_unique": mixed_unique,
        "no_heldout_mixed_signature_equals_seen_mixed_signature": no_heldout_equals_seen,
        "no_mixed_signature_equals_single_signature": no_mixed_equals_single,
        "no_heldout_class_functionally_identical_to_train_visible_class": no_heldout_equals_train_visible,
        "collisions": collisions,
        "mapping_signatures": {sequence: cell_signature(sequence) for sequence in all_classes},
        "proof": {
            "passed": passed,
            "statement": "All R5A-0 mixed operation signatures are exact final position/color functions over the valid primitive-input state domain.",
        },
    }


def identity_transport_matrix(device: Any = None) -> Any:
    import torch

    return torch.eye(GRID_SIZE * GRID_SIZE, dtype=torch.float32, device=device)


def identity_color_transition(source_colors: Any) -> Any:
    import torch
    from torch.nn import functional as F

    return F.one_hot(source_colors.long().clamp(0, COLOR_COUNT - 1), num_classes=COLOR_COUNT).to(torch.float32)
