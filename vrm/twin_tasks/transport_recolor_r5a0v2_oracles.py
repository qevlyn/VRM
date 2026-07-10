from __future__ import annotations

import itertools
import json
from typing import Any

GRID_SIZE = 7
BACKGROUND = 0
INPUT_COLORS = (1, 2, 3, 4)
RECOLORED_COLORS = (5, 6, 7, 8)
FOREGROUND_COLORS = INPUT_COLORS + RECOLORED_COLORS
COLOR_COUNT = 9

OP_C = "C"
OP_H = "H"
OP_V = "V"
OP_L = "L"
OP_K = "K"

SINGLE_CLASSES = (OP_C, OP_H, OP_V, OP_L, OP_K)
SEEN_MIXED_CLASSES = ("C->L", "L->H", "H->K", "K->V")
HELD_OUT_MIXED_CLASSES = ("L->C", "H->L", "C->K", "K->C")
MIXED_CLASSES = SEEN_MIXED_CLASSES + HELD_OUT_MIXED_CLASSES
TRAIN_VISIBLE_CLASSES = SINGLE_CLASSES + SEEN_MIXED_CLASSES
SEALED_CLASSES = TRAIN_VISIBLE_CLASSES + HELD_OUT_MIXED_CLASSES
COUNTERFACTUAL_CLASSES = MIXED_CLASSES

EXPECTED_MIXED_PREDICATES = {
    "C->L": "y >= 3",
    "L->C": "x <= 3",
    "C->K": "x <= 3 and y >= 3",
    "K->C": "x <= 3 and y <= 3",
    "H->L": "x >= 3",
    "L->H": "x <= 3",
    "H->K": "x >= 3 and y <= 3",
    "K->V": "x <= 3 and y <= 3",
}


def parse_sequence(sequence_class: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in sequence_class.split("->"))
    if not parts or any(part not in SINGLE_CLASSES for part in parts):
        raise ValueError(f"unknown R5A-0 V2 operation class {sequence_class}")
    return parts


def coord_c(x: int, y: int) -> tuple[int, int]:
    return GRID_SIZE - 1 - y, x


def coord_h(x: int, y: int) -> tuple[int, int]:
    return GRID_SIZE - 1 - x, y


def coord_v(x: int, y: int) -> tuple[int, int]:
    return x, GRID_SIZE - 1 - y


def apply_primitive_to_cell(op: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    if color == BACKGROUND:
        return x, y, color
    if op == OP_C:
        out_x, out_y = coord_c(x, y)
        return out_x, out_y, color
    if op == OP_H:
        out_x, out_y = coord_h(x, y)
        return out_x, out_y, color
    if op == OP_V:
        out_x, out_y = coord_v(x, y)
        return out_x, out_y, color
    if op == OP_L:
        if x <= 3 and color in INPUT_COLORS:
            return x, y, color + 4
        return x, y, color
    if op == OP_K:
        if x <= 3 and y <= 3 and color in INPUT_COLORS:
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
            cand_x, cand_y, _ = apply_sequence_to_cell(sequence_class, x, y, 1)
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
    for y, x, color in itertools.product(range(GRID_SIZE), range(GRID_SIZE), FOREGROUND_COLORS):
        out_x, out_y, out_color = apply_sequence_to_cell(sequence_class, x, y, color)
        rows.append([x, y, color, out_x, out_y, out_color])
    return rows


def signature_hash(sequence_class: str) -> str:
    import hashlib

    encoded = json.dumps(cell_signature(sequence_class), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def final_geometry_map(sequence_class: str) -> dict[str, str]:
    mapping = {(x, y): apply_sequence_to_cell(sequence_class, x, y, 1)[:2] for y in range(GRID_SIZE) for x in range(GRID_SIZE)}
    if mapping == {(x, y): coord_c(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)}:
        return {"name": "C", "formula": "(6-y, x)"}
    if mapping == {(x, y): coord_h(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)}:
        return {"name": "H", "formula": "(6-x, y)"}
    if mapping == {(x, y): coord_v(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)}:
        return {"name": "V", "formula": "(x, 6-y)"}
    if mapping == {(x, y): (x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)}:
        return {"name": "I", "formula": "(x, y)"}
    return {"name": "other", "formula": "finite-state map"}


def predicate_truth_set(sequence_class: str) -> set[tuple[int, int]]:
    return {
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        if apply_sequence_to_cell(sequence_class, x, y, 1)[2] == 5
    }


def predicate_formula_from_truth_set(truth: set[tuple[int, int]]) -> str:
    formulas = {
        "false": set(),
        "x <= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x <= 3},
        "x >= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x >= 3},
        "y <= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if y <= 3},
        "y >= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if y >= 3},
        "x <= 3 and y <= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x <= 3 and y <= 3},
        "x <= 3 and y >= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x <= 3 and y >= 3},
        "x >= 3 and y <= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x >= 3 and y <= 3},
        "x >= 3 and y >= 3": {(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) if x >= 3 and y >= 3},
    }
    for name, values in formulas.items():
        if truth == values:
            return name
    return "finite-state predicate"


def color_transition_table(sequence_class: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for x, y in [(0, 0), (3, 3), (6, 6), (2, 4), (4, 2)]:
        table[f"x={x},y={y}"] = {str(color): apply_sequence_to_cell(sequence_class, x, y, color)[2] for color in FOREGROUND_COLORS}
    return table


def operation_function_audit() -> dict[str, Any]:
    all_classes = SINGLE_CLASSES + MIXED_CLASSES
    hashes = {sequence: signature_hash(sequence) for sequence in all_classes}
    collisions = [
        {"left": left, "right": right}
        for left, right in itertools.combinations(all_classes, 2)
        if hashes[left] == hashes[right]
    ]
    mixed_unique = len({hashes[sequence] for sequence in MIXED_CLASSES}) == len(MIXED_CLASSES)
    singles_unique = len({hashes[sequence] for sequence in SINGLE_CLASSES}) == len(SINGLE_CLASSES)
    no_mixed_equals_single = all(hashes[mixed] != hashes[single] for mixed in MIXED_CLASSES for single in SINGLE_CLASSES)
    no_heldout_equals_seen = all(hashes[heldout] != hashes[seen] for heldout in HELD_OUT_MIXED_CLASSES for seen in SEEN_MIXED_CLASSES)
    identity_hash = json.dumps([[x, y, color, x, y, color] for y, x, color in itertools.product(range(GRID_SIZE), range(GRID_SIZE), FOREGROUND_COLORS)], sort_keys=True, separators=(",", ":"))
    degenerates = [sequence for sequence in all_classes if json.dumps(cell_signature(sequence), sort_keys=True, separators=(",", ":")) == identity_hash]
    entries = []
    predicate_matches = {}
    for sequence in all_classes:
        predicate = predicate_formula_from_truth_set(predicate_truth_set(sequence))
        expected = EXPECTED_MIXED_PREDICATES.get(sequence)
        if expected is not None:
            predicate_matches[sequence] = predicate == expected
        entries.append(
            {
                "class": sequence,
                "final_geometry_map": final_geometry_map(sequence),
                "original_coordinate_recolor_predicate": predicate,
                "expected_original_coordinate_recolor_predicate": expected,
                "expected_predicate_matches": predicate == expected if expected is not None else None,
                "color_transition_table_samples": color_transition_table(sequence),
                "exact_signature_hash": hashes[sequence],
            }
        )
    passed = mixed_unique and singles_unique and no_mixed_equals_single and no_heldout_equals_seen and not degenerates and not collisions and all(predicate_matches.values())
    return {
        "primitive_definitions": {
            "C": "ROT90_CW: (x,y,c)->(6-y,x,c)",
            "H": "HFLIP: (x,y,c)->(6-x,y,c)",
            "V": "VFLIP: (x,y,c)->(x,6-y,c)",
            "L": "LEFT_REGION_RECOLOR: if current x <= 3 and c in {1,2,3,4}, c->c+4",
            "K": "UPPER_LEFT_QUADRANT_RECOLOR: if current x <= 3 and current y <= 3 and c in {1,2,3,4}, c->c+4",
            "sequence_notation": "A->B means apply A first, then B",
        },
        "signature_domain": {"x": list(range(GRID_SIZE)), "y": list(range(GRID_SIZE)), "foreground_colors": list(FOREGROUND_COLORS)},
        "single_classes": list(SINGLE_CLASSES),
        "seen_mixed_classes": list(SEEN_MIXED_CLASSES),
        "held_out_mixed_classes": list(HELD_OUT_MIXED_CLASSES),
        "operation_entries": entries,
        "mixed_signatures_all_unique": mixed_unique,
        "single_signatures_all_unique": singles_unique,
        "no_mixed_signature_equals_single_signature": no_mixed_equals_single,
        "no_heldout_mixed_signature_equals_seen_mixed_signature": no_heldout_equals_seen,
        "no_operation_degenerates_to_identity": not degenerates,
        "identity_degenerate_classes": degenerates,
        "collisions": collisions,
        "expected_predicate_matches": predicate_matches,
        "proof": {
            "passed": passed,
            "statement": "R5A-0 V2 operation signatures are exact finite-state final position/color functions over x,y in 0..6 and c in 1..8.",
        },
    }


def identity_transport_matrix(device: Any = None) -> Any:
    import torch

    return torch.eye(GRID_SIZE * GRID_SIZE, dtype=torch.float32, device=device)
