from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterable

GRID_SIZE = 7
BACKGROUND = 0
INPUT_COLORS = (1, 2, 3, 4)
RECOLORED_COLORS = (5, 6, 7, 8)
FOREGROUND_COLORS = INPUT_COLORS + RECOLORED_COLORS
PALETTE = (0,) + FOREGROUND_COLORS

PRIMITIVE_PROGRAMS = ("C", "H", "V", "L", "K")
SEEN_MIXED_PROGRAMS = ("C->L", "L->H", "H->K", "K->V")
HELDOUT_MIXED_PROGRAMS = ("L->C", "H->L", "C->K", "K->C")
DECLARED_PROGRAMS = PRIMITIVE_PROGRAMS + SEEN_MIXED_PROGRAMS + HELDOUT_MIXED_PROGRAMS
FRONTIER_TRANSFORM_PROGRAMS = ("C", "H", "V", "C->L")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_grid(grid: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def empty_grid() -> list[list[int]]:
    return [[BACKGROUND for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def grid_to_list(grid: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(row) for row in grid]


def parse_program(program: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in program.split("->"))
    if not parts or any(item not in PRIMITIVE_PROGRAMS for item in parts):
        raise ValueError(f"unknown declared frontier program {program}")
    return parts


def apply_primitive_to_cell(op: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    if color == BACKGROUND:
        return x, y, color
    if op == "C":
        return GRID_SIZE - 1 - y, x, color
    if op == "H":
        return GRID_SIZE - 1 - x, y, color
    if op == "V":
        return x, GRID_SIZE - 1 - y, color
    if op == "L":
        return (x, y, color + 4) if x <= 3 and color in INPUT_COLORS else (x, y, color)
    if op == "K":
        return (x, y, color + 4) if x <= 3 and y <= 3 and color in INPUT_COLORS else (x, y, color)
    raise ValueError(f"unknown primitive {op}")


def apply_program_to_cell(program: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    out_x, out_y, out_color = int(x), int(y), int(color)
    for op in parse_program(program):
        out_x, out_y, out_color = apply_primitive_to_cell(op, out_x, out_y, out_color)
        if not (0 <= out_x < GRID_SIZE and 0 <= out_y < GRID_SIZE):
            raise ValueError(f"{program} produced out-of-bounds coordinate {(out_x, out_y)}")
        if out_color not in PALETTE:
            raise ValueError(f"{program} produced invalid color {out_color}")
    return out_x, out_y, out_color


def apply_program_to_grid(grid: tuple[tuple[int, ...], ...], program: str) -> tuple[tuple[int, ...], ...]:
    result = empty_grid()
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == BACKGROUND:
                continue
            out_x, out_y, out_color = apply_program_to_cell(program, x, y, int(color))
            if result[out_y][out_x] != BACKGROUND:
                raise ValueError("declared closure transform produced colliding foreground cells")
            result[out_y][out_x] = out_color
    return freeze_grid(result)


def foreground_cells(grid: tuple[tuple[int, ...], ...]) -> list[tuple[int, int, int]]:
    return [(x, y, int(value)) for y, row in enumerate(grid) for x, value in enumerate(row) if int(value) != BACKGROUND]


def cell_signature(program: str, *, colors: tuple[int, ...] = PALETTE) -> tuple[tuple[int, int, int, int, int, int], ...]:
    return tuple(
        (x, y, color, *apply_program_to_cell(program, x, y, color))
        for y, x, color in itertools.product(range(GRID_SIZE), range(GRID_SIZE), colors)
    )


def program_signature_hash(program: str) -> str:
    return stable_hash(cell_signature(program))


def enumerate_declared_programs() -> list[dict[str, Any]]:
    rows = []
    for program in DECLARED_PROGRAMS:
        signature = cell_signature(program)
        rows.append(
            {
                "program": program,
                "primitive_sequence": list(parse_program(program)),
                "program_length": len(parse_program(program)),
                "cell_signature_sha256": stable_hash(signature),
            }
        )
    return rows


def coordinate_lift_report() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for program in DECLARED_PROGRAMS:
        destinations = {}
        out_of_bounds = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                out_x, out_y, _ = apply_program_to_cell(program, x, y, 1)
                if not (0 <= out_x < GRID_SIZE and 0 <= out_y < GRID_SIZE):
                    out_of_bounds.append({"x": x, "y": y, "x_out": out_x, "y_out": out_y})
                destinations.setdefault((out_x, out_y), []).append((x, y))
        duplicated = [
            {"x_out": x, "y_out": y, "sources": [{"x": sx, "y": sy} for sx, sy in sources]}
            for (x, y), sources in sorted(destinations.items())
            if len(sources) > 1
        ]
        dropped = [{"x_out": x, "y_out": y} for y in range(GRID_SIZE) for x in range(GRID_SIZE) if (x, y) not in destinations]
        rows[program] = {
            "coordinate_map_total": len(out_of_bounds) == 0,
            "coordinate_map_bijective": not out_of_bounds and not duplicated and not dropped,
            "out_of_bounds": out_of_bounds,
            "duplicated_destinations": duplicated,
            "dropped_destinations": dropped,
        }
    return {"by_program": rows, "all_coordinate_maps_bijective": all(row["coordinate_map_bijective"] for row in rows.values())}


def declared_grammar_spec() -> dict[str, Any]:
    return {
        "primitive_names": list(PRIMITIVE_PROGRAMS),
        "primitive_preconditions": {
            "C": "foreground cell rotates clockwise around the 7x7 grid center; background remains background",
            "H": "foreground cell reflects horizontally; background remains background",
            "V": "foreground cell reflects vertically; background remains background",
            "L": "input-color foreground cells with current x <= 3 are recolored c -> c + 4",
            "K": "input-color foreground cells with current x <= 3 and current y <= 3 are recolored c -> c + 4",
        },
        "input_output_shape_assumptions": "7x7 integer grids with palette 0..8 and shared input/output size",
        "object_color_coordinate_assumptions": "the closure checker is cellwise and object-agnostic; collision-free whole-grid lifting is required for multi-cell foreground grids",
        "composition_order": "A->B applies A first, then B",
        "allowed_program_lengths": [1, 2],
        "declared_programs": list(DECLARED_PROGRAMS),
        "support_query_assumptions": "normal V2 raw support input grids, support output grids, and query grids; no labels or metadata",
        "known_limitations": [
            "bounded to the declared class tables audited in R5A-0 V2, not all syntactic strings accepted by a parser",
            "closure membership is exact only for the finite state/grid domain reported here",
            "whole-grid lifting requires destination-coordinate bijection and no foreground collision",
        ],
        "reused_production_code": False,
        "imports_v2_model_or_oracle": False,
    }


def closure_report() -> dict[str, Any]:
    rows = enumerate_declared_programs()
    by_hash: dict[str, list[str]] = {}
    for row in rows:
        by_hash.setdefault(row["cell_signature_sha256"], []).append(row["program"])
    collisions = [
        {"programs": programs, "cell_signature_sha256": signature}
        for signature, programs in sorted(by_hash.items())
        if len(programs) > 1
    ]
    lift = coordinate_lift_report()
    return {
        "grammar_spec": declared_grammar_spec(),
        "enumerated_domain": "single-cell finite state signatures lifted to collision-free 7x7 grids",
        "grid_sizes": [[GRID_SIZE, GRID_SIZE]],
        "palette_domain": list(PALETTE),
        "object_domain": "object-agnostic cellwise domain; probe object validity is audited separately",
        "program_length_bound": 2,
        "number_of_programs": len(rows),
        "whether_exhaustive": True,
        "whether_bounded_complete": True,
        "semantic_collision_count": len(collisions),
        "semantic_collisions": collisions,
        "whole_grid_lift_conditions_if_any": {
            "coordinate_maps_bijective": lift["all_coordinate_maps_bijective"],
            "requires_no_foreground_destination_collision": True,
            "requires_independent_cellwise_application": True,
        },
        "programs": rows,
        "coordinate_lift_report": lift,
        "limitations": declared_grammar_spec()["known_limitations"],
    }


def representability_check(
    support_inputs: list[tuple[tuple[int, ...], ...]],
    support_outputs: list[tuple[tuple[int, ...], ...]],
    query_grid: tuple[tuple[int, ...], ...],
    target_grid: tuple[tuple[int, ...], ...],
    *,
    candidate_programs: tuple[str, ...] = DECLARED_PROGRAMS,
) -> dict[str, Any]:
    matches = []
    failures = []
    for program in candidate_programs:
        try:
            support_ok = all(apply_program_to_grid(before, program) == after for before, after in zip(support_inputs, support_outputs))
            query_ok = apply_program_to_grid(query_grid, program) == target_grid
        except ValueError as exc:
            failures.append({"program": program, "exception": repr(exc)})
            continue
        if support_ok and query_ok:
            matches.append(program)
    return {
        "representable_by_declared_v2_closure": bool(matches),
        "matching_programs": matches,
        "collision_or_ambiguity_details": {
            "match_count": len(matches),
            "ambiguous": len(matches) > 1,
            "program_failures": failures,
        },
        "checker_evidence": {
            "candidate_program_count": len(candidate_programs),
            "support_pair_count": len(support_inputs),
            "query_checked": True,
            "checker": "independent R5A-2A handwritten closure checker",
        },
    }


def selected_object_transform_matches(
    before_cells: Iterable[tuple[int, int, int]],
    after_cells: Iterable[tuple[int, int, int]],
    *,
    candidate_programs: tuple[str, ...] = FRONTIER_TRANSFORM_PROGRAMS,
) -> list[str]:
    before = sorted((int(x), int(y), int(color)) for x, y, color in before_cells)
    observed = sorted((int(x), int(y), int(color)) for x, y, color in after_cells)
    matches = []
    for program in candidate_programs:
        predicted = sorted(apply_program_to_cell(program, x, y, color) for x, y, color in before)
        if predicted == observed:
            matches.append(program)
    return matches
