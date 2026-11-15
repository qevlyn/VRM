from __future__ import annotations

import itertools
import json
from typing import Any

from .composition_r4a0r_data import (
    GRID_SIZE,
    apply_sequence_to_coord,
    apply_sequence_to_grid,
    grid_to_list,
    no_edge_foreground,
    stable_hash,
)

OP_ORDER = ("C", "F", "T")
IDENTITY_AFFINE = (((1, 0), (0, 1)), (0, 0))
OP_AFFINES: dict[str, tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]] = {
    "C": (((0, -1), (1, 0)), (6, 0)),
    "F": (((-1, 0), (0, 1)), (6, 0)),
    "T": (((1, 0), (0, 1)), (1, 0)),
}


def compact_word(word: str) -> str:
    cleaned = word.replace("->", "").replace(" ", "")
    if any(part not in OP_ORDER for part in cleaned):
        raise ValueError(f"unknown composition word {word!r}")
    return cleaned


def arrow_word(word: str) -> str:
    cleaned = compact_word(word)
    if not cleaned:
        return "I"
    return "->".join(cleaned)


def enumerate_length3_words() -> list[str]:
    return ["".join(parts) for parts in itertools.product(OP_ORDER, repeat=3)]


def matmul(
    left: tuple[tuple[int, int], tuple[int, int]],
    right: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def matvec(matrix: tuple[tuple[int, int], tuple[int, int]], vector: tuple[int, int]) -> tuple[int, int]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    )


def compose_affine(
    first: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]],
    second: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    """Return second(first(p)) for integer affine maps."""
    first_a, first_b = first
    second_a, second_b = second
    out_a = matmul(second_a, first_a)
    second_on_first_b = matvec(second_a, first_b)
    out_b = (second_on_first_b[0] + second_b[0], second_on_first_b[1] + second_b[1])
    return out_a, out_b


def affine_for_word(word: str) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
    affine = IDENTITY_AFFINE
    for op in compact_word(word):
        affine = compose_affine(affine, OP_AFFINES[op])
    return affine


def affine_key(affine: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]) -> str:
    matrix, offset = affine
    return json.dumps({"A": matrix, "b": offset}, sort_keys=True, separators=(",", ":"))


def affine_payload(word: str) -> dict[str, Any]:
    matrix, offset = affine_for_word(word)
    return {
        "word": compact_word(word),
        "sequence_class": arrow_word(word),
        "A": [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]],
        "b": [offset[0], offset[1]],
        "affine_key": affine_key((matrix, offset)),
        "number_of_T_applications": compact_word(word).count("T"),
    }


def apply_word_to_coord(word: str, coord: tuple[int, int]) -> tuple[int, int]:
    return apply_sequence_to_coord(arrow_word(word), coord)


def source_index_for_output_exact(word: str, out_x: int, out_y: int) -> tuple[int, int, int]:
    matches: list[tuple[int, int]] = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            try:
                if apply_word_to_coord(word, (x, y)) == (out_x, out_y):
                    matches.append((x, y))
            except ValueError:
                continue
    if len(matches) != 1:
        return -1, -1, len(matches)
    x, y = matches[0]
    return y * GRID_SIZE + x, x, y


def is_scene_safe_for_word(grid: tuple[tuple[int, ...], ...], word: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not no_edge_foreground(grid):
        reasons.append("input_foreground_touches_edge")
    current = grid
    for step, op in enumerate(compact_word(word), start=1):
        try:
            current = apply_sequence_to_grid(current, op)
        except ValueError as exc:
            reasons.append(f"step_{step}_{op}_invalid:{exc}")
            return False, sorted(set(reasons))
        if not no_edge_foreground(current):
            reasons.append(f"step_{step}_{op}_foreground_touches_edge")
    try:
        output = apply_sequence_to_grid(grid, arrow_word(word))
    except ValueError as exc:
        reasons.append(f"final_invalid:{exc}")
        return False, sorted(set(reasons))
    if output == grid:
        reasons.append("final_output_equals_input")
    return not reasons, sorted(set(reasons))


def output_hash_for_word(grid: tuple[tuple[int, ...], ...], word: str) -> str:
    return stable_hash(grid_to_list(apply_sequence_to_grid(grid, arrow_word(word))))
