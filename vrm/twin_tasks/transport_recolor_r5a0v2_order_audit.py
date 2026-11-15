from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from pathlib import Path
from typing import Any, Iterable

GRID_SIZE = 7
BACKGROUND = 0
INPUT_COLORS = (1, 2, 3, 4)
RECOLORED_COLORS = (5, 6, 7, 8)
ALL_COLORS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
FOREGROUND_COLORS = INPUT_COLORS + RECOLORED_COLORS

PRIMITIVE_PROGRAMS = ("C", "H", "V", "L", "K")
SEEN_MIXED_PROGRAMS = ("C->L", "L->H", "H->K", "K->V")
HELDOUT_MIXED_PROGRAMS = ("L->C", "H->L", "C->K", "K->C")
MIXED_PROGRAMS = SEEN_MIXED_PROGRAMS + HELDOUT_MIXED_PROGRAMS
TRAIN_PROGRAMS = PRIMITIVE_PROGRAMS + SEEN_MIXED_PROGRAMS
DEV_PROGRAMS = TRAIN_PROGRAMS
SEALED_PROGRAMS = TRAIN_PROGRAMS + HELDOUT_MIXED_PROGRAMS

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
AUDIT_VERSION = "R5A0V2_ORDER_AUDIT_V1"
PASS_VERDICT = "R5A0V2_ORDER_AUDIT_PASS"
FAIL_VERDICT = "R5A0V2_ORDER_CLAIM_NOT_ESTABLISHED"
INCONCLUSIVE_VERDICT = "INCONCLUSIVE"

SOURCE_FILES = (
    "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    "vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py",
    "tests/test_transport_recolor_r5a0v2.py",
    "tests/test_transport_recolor_r5a0v2_order_audit.py",
    "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    "docs/TRANSPORT_RECOLOR_R5A0V2_ORDER_AUDIT.md",
)

REUSED_IMMUTABLE_CONSTANTS: tuple[str, ...] = ()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def parse_program(program: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in program.split("->"))
    if not parts or any(item not in PRIMITIVE_PROGRAMS for item in parts):
        raise ValueError(f"unknown audit program {program}")
    return parts


def reverse_program(program: str) -> str:
    return "->".join(reversed(parse_program(program)))


def agreement_programs() -> tuple[str, ...]:
    return tuple(sorted(set(SEALED_PROGRAMS) | {reverse_program(program) for program in MIXED_PROGRAMS}))


def independent_apply_primitive(op: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    if color == BACKGROUND:
        return x, y, color
    if op == "C":
        return GRID_SIZE - 1 - y, x, color
    if op == "H":
        return GRID_SIZE - 1 - x, y, color
    if op == "V":
        return x, GRID_SIZE - 1 - y, color
    if op == "L":
        if x <= 3 and color in INPUT_COLORS:
            return x, y, color + 4
        return x, y, color
    if op == "K":
        if x <= 3 and y <= 3 and color in INPUT_COLORS:
            return x, y, color + 4
        return x, y, color
    raise ValueError(f"unknown primitive {op}")


def independent_apply_program_to_cell(program: str, x: int, y: int, color: int) -> tuple[int, int, int]:
    out_x, out_y, out_color = x, y, color
    for op in parse_program(program):
        out_x, out_y, out_color = independent_apply_primitive(op, out_x, out_y, out_color)
        if not (0 <= out_x < GRID_SIZE and 0 <= out_y < GRID_SIZE):
            raise ValueError(f"{program} produced out-of-bounds coordinate {(out_x, out_y)}")
    return out_x, out_y, out_color


def independent_empty_grid() -> list[list[int]]:
    return [[BACKGROUND for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def freeze_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def independent_apply_program_to_grid(grid: tuple[tuple[int, ...], ...], program: str) -> tuple[tuple[int, ...], ...]:
    result = independent_empty_grid()
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == BACKGROUND:
                continue
            out_x, out_y, out_color = independent_apply_program_to_cell(program, x, y, int(color))
            if result[out_y][out_x] != BACKGROUND:
                raise ValueError("independent checker produced colliding foreground cells")
            result[out_y][out_x] = out_color
    return freeze_grid(result)


def coordinate_bijection_report_for_primitive(op: str) -> dict[str, Any]:
    failures: list[str] = []
    by_color: dict[str, Any] = {}
    overwritten_destinations: list[dict[str, Any]] = []
    total_state_count = 0
    in_bounds_count = 0
    valid_color_count = 0
    for color in ALL_COLORS:
        destinations: dict[tuple[int, int], list[tuple[int, int]]] = {}
        out_of_bounds: list[dict[str, int]] = []
        invalid_colors: list[dict[str, int]] = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                total_state_count += 1
                out_x, out_y, out_color = independent_apply_primitive(op, x, y, color)
                if 0 <= out_x < GRID_SIZE and 0 <= out_y < GRID_SIZE:
                    in_bounds_count += 1
                else:
                    out_of_bounds.append({"x": x, "y": y, "color": color, "x_out": out_x, "y_out": out_y})
                if out_color in ALL_COLORS:
                    valid_color_count += 1
                else:
                    invalid_colors.append({"x": x, "y": y, "color": color, "color_out": out_color})
                destinations.setdefault((out_x, out_y), []).append((x, y))
        duplicate_destinations = [
            {"x_out": x, "y_out": y, "sources": [{"x": sx, "y": sy} for sx, sy in sources]}
            for (x, y), sources in sorted(destinations.items())
            if len(sources) > 1
        ]
        dropped_destinations = [
            {"x_out": x, "y_out": y}
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
            if (x, y) not in destinations
        ]
        by_color[str(color)] = {
            "coordinate_bijection": not out_of_bounds and not duplicate_destinations and not dropped_destinations,
            "out_of_bounds": out_of_bounds,
            "dropped_destinations": dropped_destinations,
            "duplicated_destinations": duplicate_destinations,
            "invalid_colors": invalid_colors,
        }
    foreground_bijection = all(by_color[str(color)]["coordinate_bijection"] for color in FOREGROUND_COLORS)
    background_bijection = by_color[str(BACKGROUND)]["coordinate_bijection"]
    if total_state_count != len(ALL_COLORS) * GRID_SIZE * GRID_SIZE:
        failures.append("mapping_not_total")
    if in_bounds_count != total_state_count:
        failures.append("destination_out_of_bounds")
    if valid_color_count != total_state_count:
        failures.append("invalid_output_color")
    if not foreground_bijection:
        failures.append("foreground_coordinate_map_not_bijective")
    if not background_bijection:
        failures.append("background_coordinate_map_not_bijective")
    for color, row in by_color.items():
        for duplicate in row["duplicated_destinations"]:
            overwritten_destinations.append({"color": int(color), **duplicate})
    return {
        "primitive": op,
        "state_count": total_state_count,
        "mapping_total": total_state_count == len(ALL_COLORS) * GRID_SIZE * GRID_SIZE,
        "destinations_in_bounds": in_bounds_count == total_state_count,
        "output_colors_valid": valid_color_count == total_state_count,
        "foreground_coordinate_bijection": foreground_bijection,
        "background_coordinate_bijection": background_bijection,
        "coordinate_bijection_by_color": by_color,
        "dropped_destinations": [item for color in by_color.values() for item in color["dropped_destinations"]],
        "duplicated_destinations": [item for color in by_color.values() for item in color["duplicated_destinations"]],
        "colliding_destinations": overwritten_destinations,
        "overwritten_destinations": overwritten_destinations,
        "failure_reasons": failures,
    }


def primitive_cellwise_lift_report() -> dict[str, Any]:
    rows = {op: coordinate_bijection_report_for_primitive(op) for op in PRIMITIVE_PROGRAMS}
    return {
        "by_primitive": rows,
        "passed": all(not row["failure_reasons"] for row in rows.values()),
    }


def grid_from_cells(cells: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, ...], ...]:
    grid = independent_empty_grid()
    for x, y, color in cells:
        grid[y][x] = color
    return freeze_grid(grid)


def dense_all_color_grid() -> tuple[tuple[int, ...], ...]:
    grid = independent_empty_grid()
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid[y][x] = ((x + 3 * y) % 8) + 1
    return freeze_grid(grid)


def deterministic_random_dense_grid(seed: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    grid = independent_empty_grid()
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid[y][x] = rng.choice(ALL_COLORS)
    return freeze_grid(grid)


def multicell_fixtures() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = [
        {"fixture_id": "empty_background_only", "grid": freeze_grid(independent_empty_grid())},
        {"fixture_id": "one_active_cell", "grid": grid_from_cells([(3, 3, 1)])},
        {"fixture_id": "two_cells_distinct_colors", "grid": grid_from_cells([(2, 2, 1), (4, 4, 2)])},
    ]
    for y in range(GRID_SIZE):
        fixtures.append({"fixture_id": f"x_boundary_pair_y={y}", "grid": grid_from_cells([(3, y, 1), (4, y, 2)])})
    for x in range(GRID_SIZE):
        fixtures.append({"fixture_id": f"y_boundary_pair_x={x}", "grid": grid_from_cells([(x, 3, 3), (x, 4, 4)])})
    fixtures.extend(
        [
            {"fixture_id": "quadrant_boundary_corners", "grid": grid_from_cells([(3, 3, 1), (4, 3, 2), (3, 4, 3), (4, 4, 4)])},
            {"fixture_id": "dense_all_colors_all_branches", "grid": dense_all_color_grid()},
        ]
    )
    for seed in (20260628, 20260629, 20260630):
        fixtures.append({"fixture_id": f"deterministic_random_dense_seed_{seed}", "grid": deterministic_random_dense_grid(seed)})
    return fixtures


def multicell_production_agreement_report() -> dict[str, Any]:
    from .transport_recolor_r5a0v2_oracles import apply_sequence_to_grid

    failures: list[dict[str, Any]] = []
    count = 0
    for fixture in multicell_fixtures():
        for program in agreement_programs():
            independent = independent_apply_program_to_grid(fixture["grid"], program)
            production = apply_sequence_to_grid(fixture["grid"], program)
            count += 1
            if independent != production:
                failures.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "program": program,
                        "independent_output_hash": json_hash(independent),
                        "production_output_hash": json_hash(production),
                    }
                )
    return {
        "fixture_count": len(multicell_fixtures()),
        "program_count": len(agreement_programs()),
        "multicell_production_agreement_count": count - len(failures),
        "multicell_production_agreement_failures": failures,
    }


def source_grid_independence_report() -> dict[str, Any]:
    from .transport_recolor_r5a0v2_oracles import apply_primitive_to_cell

    source = inspect.getsource(apply_primitive_to_cell)
    signature = str(inspect.signature(apply_primitive_to_cell))
    forbidden_terms = ("grid", "row", "neighbor", "object", "occupancy", "foreground_coords", "count", "global")
    found = [term for term in forbidden_terms if term in source]
    passed = signature == "(op: 'str', x: 'int', y: 'int', color: 'int') -> 'tuple[int, int, int]'" and not found
    return {
        "production_primitive_signature": signature,
        "production_primitive_source_hash": sha256_bytes(source.encode("utf-8")),
        "forbidden_grid_state_terms_found": found,
        "no_neighbor_object_occupancy_or_global_state_reads": passed,
        "evidence": "Production primitive accepts only op, x, y, color; source contains no grid, neighbor, object, occupancy, foreground-count, or global-state read terms.",
    }


def cellwise_lift_validity_report() -> dict[str, Any]:
    primitive_report = primitive_cellwise_lift_report()
    multicell = multicell_production_agreement_report()
    source_report = source_grid_independence_report()
    failure_reasons: list[str] = []
    if not primitive_report["passed"]:
        failure_reasons.append("primitive_totality_or_coordinate_bijection_failed")
    if multicell["multicell_production_agreement_failures"]:
        failure_reasons.append("multicell_production_agreement_failed")
    if not source_report["no_neighbor_object_occupancy_or_global_state_reads"]:
        failure_reasons.append("production_primitive_grid_level_read_cannot_be_excluded")
    valid = not failure_reasons
    return {
        "single_cell_domain_size": len(probe_domain()),
        "cellwise_lift_valid": valid,
        "cellwise_lift_failure_reasons": failure_reasons,
        "coordinate_bijection_by_primitive": primitive_report["by_primitive"],
        "multicell_fixture_count": multicell["fixture_count"],
        "multicell_program_count": multicell["program_count"],
        "multicell_production_agreement_count": multicell["multicell_production_agreement_count"],
        "multicell_production_agreement_failures": multicell["multicell_production_agreement_failures"],
        "source_grid_independence": source_report,
        "whole_grid_exhaustiveness_claim": valid,
    }


def probe_domain() -> list[dict[str, Any]]:
    return [
        {"witness_id": f"cell:x={x},y={y},color={color}", "x": x, "y": y, "color": color}
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        for color in ALL_COLORS
    ]


def behavior_rows(program: str) -> list[list[int]]:
    return [
        [fixture["x"], fixture["y"], fixture["color"], *independent_apply_program_to_cell(program, fixture["x"], fixture["y"], fixture["color"])]
        for fixture in probe_domain()
    ]


def behavior_fingerprint(program: str) -> str:
    return json_hash(behavior_rows(program))


def first_divergence(program: str, other: str) -> dict[str, Any] | None:
    for fixture in probe_domain():
        original = independent_apply_program_to_cell(program, fixture["x"], fixture["y"], fixture["color"])
        reversed_output = independent_apply_program_to_cell(other, fixture["x"], fixture["y"], fixture["color"])
        if original != reversed_output:
            return {
                "first_divergence_witness_id": fixture["witness_id"],
                "first_divergence_input": {"x": fixture["x"], "y": fixture["y"], "color": fixture["color"]},
                "original_output": {"x": original[0], "y": original[1], "color": original[2]},
                "reversed_output": {"x": reversed_output[0], "y": reversed_output[1], "color": reversed_output[2]},
            }
    return None


GOLDEN_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "primitive_C", "program": "C", "input": {"x": 2, "y": 1, "color": 3}, "expected": {"x": 5, "y": 2, "color": 3}},
    {"case_id": "primitive_H", "program": "H", "input": {"x": 2, "y": 1, "color": 3}, "expected": {"x": 4, "y": 1, "color": 3}},
    {"case_id": "primitive_V", "program": "V", "input": {"x": 2, "y": 1, "color": 3}, "expected": {"x": 2, "y": 5, "color": 3}},
    {"case_id": "primitive_L_boundary_recolor", "program": "L", "input": {"x": 3, "y": 4, "color": 4}, "expected": {"x": 3, "y": 4, "color": 8}},
    {"case_id": "primitive_L_right_no_recolor", "program": "L", "input": {"x": 4, "y": 4, "color": 2}, "expected": {"x": 4, "y": 4, "color": 2}},
    {"case_id": "primitive_L_preserve_recolored", "program": "L", "input": {"x": 2, "y": 4, "color": 7}, "expected": {"x": 2, "y": 4, "color": 7}},
    {"case_id": "primitive_K_boundary_recolor", "program": "K", "input": {"x": 3, "y": 3, "color": 1}, "expected": {"x": 3, "y": 3, "color": 5}},
    {"case_id": "primitive_K_lower_no_recolor", "program": "K", "input": {"x": 3, "y": 4, "color": 1}, "expected": {"x": 3, "y": 4, "color": 1}},
    {"case_id": "mixed_C_then_L", "program": "C->L", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 4, "y": 2, "color": 1}},
    {"case_id": "mixed_L_then_C", "program": "L->C", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 4, "y": 2, "color": 5}},
    {"case_id": "mixed_L_then_H", "program": "L->H", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 4, "y": 2, "color": 5}},
    {"case_id": "mixed_H_then_L", "program": "H->L", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 4, "y": 2, "color": 1}},
    {"case_id": "mixed_H_then_K", "program": "H->K", "input": {"x": 4, "y": 2, "color": 1}, "expected": {"x": 2, "y": 2, "color": 5}},
    {"case_id": "mixed_K_then_H", "program": "K->H", "input": {"x": 4, "y": 2, "color": 1}, "expected": {"x": 2, "y": 2, "color": 1}},
    {"case_id": "mixed_K_then_V", "program": "K->V", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 2, "y": 4, "color": 5}},
    {"case_id": "mixed_V_then_K", "program": "V->K", "input": {"x": 2, "y": 2, "color": 1}, "expected": {"x": 2, "y": 4, "color": 1}},
    {"case_id": "mixed_C_then_K", "program": "C->K", "input": {"x": 2, "y": 4, "color": 1}, "expected": {"x": 2, "y": 2, "color": 5}},
    {"case_id": "mixed_K_then_C", "program": "K->C", "input": {"x": 2, "y": 4, "color": 1}, "expected": {"x": 2, "y": 2, "color": 1}},
)


def run_golden_checks() -> tuple[int, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    for case in GOLDEN_CASES:
        raw = case["input"]
        actual = independent_apply_program_to_cell(case["program"], raw["x"], raw["y"], raw["color"])
        expected = case["expected"]
        expected_tuple = (expected["x"], expected["y"], expected["color"])
        if actual != expected_tuple:
            failures.append({"case_id": case["case_id"], "program": case["program"], "actual": actual, "expected": expected})
    return len(GOLDEN_CASES) - len(failures), failures


def deterministic_generated_examples(limit_per_split_program: int = 2) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for split in ("train", "dev", "sealed"):
        path = ARTIFACT_ROOT / f"{split}.jsonl"
        seen: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                program = str(record["operation_class"])
                count = seen.get(program, 0)
                if count >= limit_per_split_program:
                    continue
                seen[program] = count + 1
                examples.append(
                    {
                        "example_id": str(record["episode_id"]),
                        "split": split,
                        "program": program,
                        "query": freeze_grid([[int(value) for value in row] for row in record["query"]]),
                        "target": freeze_grid([[int(value) for value in row] for row in record["target"]]),
                    }
                )
                expected_programs = TRAIN_PROGRAMS if split in {"train", "dev"} else SEALED_PROGRAMS
                if all(seen.get(program, 0) >= limit_per_split_program for program in expected_programs):
                    break
    return examples


def checker_agreement_report() -> dict[str, Any]:
    from .transport_recolor_r5a0v2_oracles import apply_sequence_to_grid

    agreement_count = 0
    _, failures = run_golden_checks()
    for case in GOLDEN_CASES:
        raw = case["input"]
        input_grid = grid_from_cells([(raw["x"], raw["y"], raw["color"])])
        expected = case["expected"]
        expected_grid = grid_from_cells([(expected["x"], expected["y"], expected["color"])])
        independent = independent_apply_program_to_grid(input_grid, case["program"])
        production = apply_sequence_to_grid(input_grid, case["program"])
        if independent == production == expected_grid:
            agreement_count += 1
            continue
        failures.append(
            {
                "case_id": case["case_id"],
                "program": case["program"],
                "independent_output_hash": json_hash(independent),
                "production_output_hash": json_hash(production),
                "expected_output_hash": json_hash(expected_grid),
            }
        )
    for example in deterministic_generated_examples():
        independent = independent_apply_program_to_grid(example["query"], example["program"])
        production = apply_sequence_to_grid(example["query"], example["program"])
        if independent == production == example["target"]:
            agreement_count += 1
            continue
        failures.append(
            {
                "example_id": example["example_id"],
                "program": example["program"],
                "independent_equals_production": independent == production,
                "independent_equals_target": independent == example["target"],
                "production_equals_target": production == example["target"],
            }
        )
    return {
        "checker_agreement_count": agreement_count,
        "checker_agreement_failures": failures,
        "generated_validation_example_count": len(deterministic_generated_examples()),
    }


def source_file_hashes() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        hashes[relative] = sha256_file(path) if path.exists() else None
    return hashes


def artifact_file_hashes() -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    if not ARTIFACT_ROOT.exists():
        return hashes
    for path in sorted(item for item in ARTIFACT_ROOT.iterdir() if item.is_file()):
        if path.name.startswith("order_audit"):
            continue
        hashes[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return hashes


def grammar_inventory() -> dict[str, Any]:
    return {
        "primitive_operations": {
            "C": {"syntactic": "ROT90_CW", "semantic": "(x,y,c)->(6-y,x,c)", "kind": "pointwise_coordinate_bijection"},
            "H": {"syntactic": "HFLIP", "semantic": "(x,y,c)->(6-x,y,c)", "kind": "pointwise_coordinate_bijection"},
            "V": {"syntactic": "VFLIP", "semantic": "(x,y,c)->(x,6-y,c)", "kind": "pointwise_coordinate_bijection"},
            "L": {"syntactic": "LEFT_REGION_RECOLOR", "semantic": "if current x <= 3 and c in 1..4 then c->c+4", "kind": "pointwise_coordinate_conditioned_recolor"},
            "K": {"syntactic": "UPPER_LEFT_QUADRANT_RECOLOR", "semantic": "if current x <= 3 and current y <= 3 and c in 1..4 then c->c+4", "kind": "pointwise_coordinate_conditioned_recolor"},
        },
        "syntactic_metadata": ["program_id", "split membership", "sequence string", "operation label"],
        "semantic_behavior": ["final coordinate map", "color transition as a function of current coordinate and color", "composition order"],
        "operation_context_sensitivity": {
            "pointwise": list(PRIMITIVE_PROGRAMS) + list(MIXED_PROGRAMS),
            "objectwise": [],
            "context_sensitive": [],
        },
        "sequence_notation": "A->B means apply A first, then B",
    }


def all_programs_by_split() -> dict[str, list[str]]:
    return {"train": list(TRAIN_PROGRAMS), "dev": list(DEV_PROGRAMS), "sealed": list(SEALED_PROGRAMS)}


def mixed_programs_by_split() -> dict[str, list[str]]:
    return {"train": list(SEEN_MIXED_PROGRAMS), "dev": list(SEEN_MIXED_PROGRAMS), "sealed": list(MIXED_PROGRAMS)}


def per_program_rows(fingerprints: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for program in MIXED_PROGRAMS:
        reversed_id = reverse_program(program)
        reversed_fp = fingerprints.setdefault(reversed_id, behavior_fingerprint(reversed_id))
        divergence = first_divergence(program, reversed_id)
        rows.append(
            {
                "program_id": program,
                "syntactic_signature": {"ordered_primitives": list(parse_program(program)), "sequence_string": program},
                "reversed_program_id": reversed_id,
                "original_behavior_fingerprint": fingerprints[program],
                "reversed_behavior_fingerprint": reversed_fp,
                "order_distinguishable": fingerprints[program] != reversed_fp,
                "first_divergence_witness_id": divergence["first_divergence_witness_id"] if divergence else None,
                "first_divergence_input": divergence["first_divergence_input"] if divergence else None,
                "original_output": divergence["original_output"] if divergence else None,
                "reversed_output": divergence["reversed_output"] if divergence else None,
            }
        )
    return rows


def per_split_order(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_program = {row["program_id"]: row for row in rows}
    result: dict[str, Any] = {}
    for split, programs in mixed_programs_by_split().items():
        distinguishable = sum(1 for program in programs if by_program[program]["order_distinguishable"])
        indistinguishable = len(programs) - distinguishable
        result[split] = {
            "mixed_program_count": len(programs),
            "order_distinguishable_count": distinguishable,
            "order_indistinguishable_count": indistinguishable,
            "order_distinguishable_rate": distinguishable / max(len(programs), 1),
            "by_program": {program: by_program[program]["order_distinguishable"] for program in programs},
        }
    return result


def collision_pairs(left_split: str, left_programs: Iterable[str], right_split: str, right_programs: Iterable[str], fingerprints: dict[str, str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left in left_programs:
        for right in right_programs:
            if fingerprints[left] != fingerprints[right]:
                continue
            pairs.append(
                {
                    "left_split": left_split,
                    "left_program": left,
                    "right_split": right_split,
                    "right_program": right,
                    "same_program_id": left == right,
                    "behavior_fingerprint": fingerprints[left],
                }
            )
    return pairs


def cross_split_overlap(fingerprints: dict[str, str]) -> dict[str, Any]:
    programs = all_programs_by_split()
    train_dev = collision_pairs("train", programs["train"], "dev", programs["dev"], fingerprints)
    train_sealed = collision_pairs("train", programs["train"], "sealed", programs["sealed"], fingerprints)
    dev_sealed = collision_pairs("dev", programs["dev"], "sealed", programs["sealed"], fingerprints)
    all_pairs = train_dev + train_sealed + dev_sealed
    return {
        "train_vs_dev_collision_count": len(train_dev),
        "train_vs_sealed_collision_count": len(train_sealed),
        "dev_vs_sealed_collision_count": len(dev_sealed),
        "collision_pairs": all_pairs,
        "collision_witness_summary": {
            "same_program_id_collision_count": sum(1 for item in all_pairs if item["same_program_id"]),
            "different_program_id_collision_count": sum(1 for item in all_pairs if not item["same_program_id"]),
            "note": "Train/dev and visible/sealed share the same visible program IDs, so same-program fingerprint overlaps are expected bookkeeping overlaps, not semantic aliasing between distinct programs.",
        },
    }


def heldout_vs_train_overlap(fingerprints: dict[str, str]) -> dict[str, Any]:
    pairs = [
        {
            "heldout_program": heldout,
            "train_program": train,
            "behavior_fingerprint": fingerprints[heldout],
        }
        for heldout in HELDOUT_MIXED_PROGRAMS
        for train in TRAIN_PROGRAMS
        if fingerprints[heldout] == fingerprints[train]
    ]
    return {
        "heldout_mixed_program_count": len(HELDOUT_MIXED_PROGRAMS),
        "heldout_mixed_collision_with_train_count": len(pairs),
        "collision_pairs": pairs,
    }


def build_samples(per_program: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "golden_cases": list(GOLDEN_CASES),
        "first_divergence_witnesses": {
            row["program_id"]: {
                "reversed_program_id": row["reversed_program_id"],
                "first_divergence_witness_id": row["first_divergence_witness_id"],
                "first_divergence_input": row["first_divergence_input"],
                "original_output": row["original_output"],
                "reversed_output": row["reversed_output"],
            }
            for row in per_program
        },
    }


def pytest_passed_from_path(path: Path | None) -> bool | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if " failed" in text or "ERROR" in text:
        return False
    return " passed" in text


def build_report(pytest_result_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    agreement = checker_agreement_report()
    cellwise = cellwise_lift_validity_report()
    programs_to_fingerprint = list(agreement_programs())
    fingerprints = {program: behavior_fingerprint(program) for program in programs_to_fingerprint}
    program_rows = per_program_rows(fingerprints)
    split_rows = per_split_order(program_rows)
    overlap = cross_split_overlap(fingerprints)
    heldout_overlap = heldout_vs_train_overlap(fingerprints)
    pytest_passed = pytest_passed_from_path(pytest_result_path)
    failed_gates: list[str] = []
    if pytest_passed is not True:
        failed_gates.append("pytest_q_not_confirmed" if pytest_passed is None else "pytest_q_failed")
    if not cellwise["cellwise_lift_valid"]:
        failed_gates.append("cellwise_lift_validity_failed")
    if agreement["checker_agreement_failures"]:
        failed_gates.append("checker_agreement_failed")
    for program in HELDOUT_MIXED_PROGRAMS:
        row = next(item for item in program_rows if item["program_id"] == program)
        if not row["order_distinguishable"]:
            failed_gates.append(f"heldout_reverse_not_distinguishable:{program}")
    if heldout_overlap["heldout_mixed_collision_with_train_count"]:
        failed_gates.append("heldout_mixed_collision_with_train")
    if any(not row["order_distinguishable"] for row in program_rows):
        failed_gates.append("mixed_program_commutes_with_reverse")

    verdict = PASS_VERDICT if not failed_gates else FAIL_VERDICT
    if failed_gates == ["pytest_q_not_confirmed"]:
        verdict = INCONCLUSIVE_VERDICT

    report = {
        "audit_version": AUDIT_VERSION,
        "source_file_hashes": source_file_hashes(),
        "existing_v2_artifact_hashes": artifact_file_hashes(),
        "reused_immutable_constants": list(REUSED_IMMUTABLE_CONSTANTS),
        "grammar_inventory": grammar_inventory(),
        "all_programs_by_split": all_programs_by_split(),
        "mixed_programs_by_split": mixed_programs_by_split(),
        "probe_domain_description": {
            "domain": "all independent cell states (x,y,color) with x,y in 0..6 and color in 0..8",
            "why_exhaustive": "Whole-grid exhaustiveness is claimed only if cellwise_lift_validity.cellwise_lift_valid is true. The gate verifies total in-bounds primitive maps, foreground coordinate bijections, no hidden grid-state reads in the production primitive, and deterministic multi-cell production agreement.",
            "fallback_if_lift_invalid": "If the cellwise lift gate fails, this same 441-state set is only a deterministic single-cell / branch-complete probe domain.",
            "coordinate_predicate_boundaries": {"x": [2, 3, 4], "y": [2, 3, 4]},
            "color_states": list(ALL_COLORS),
            "grid_size": GRID_SIZE,
        },
        "probe_domain_exhaustive": cellwise["whole_grid_exhaustiveness_claim"],
        "probe_fixture_count": len(probe_domain()),
        "cellwise_lift_validity": cellwise,
        "single_cell_domain_size": cellwise["single_cell_domain_size"],
        "cellwise_lift_valid": cellwise["cellwise_lift_valid"],
        "cellwise_lift_failure_reasons": cellwise["cellwise_lift_failure_reasons"],
        "coordinate_bijection_by_primitive": cellwise["coordinate_bijection_by_primitive"],
        "multicell_production_agreement_count": cellwise["multicell_production_agreement_count"],
        "multicell_production_agreement_failures": cellwise["multicell_production_agreement_failures"],
        "whole_grid_exhaustiveness_claim": cellwise["whole_grid_exhaustiveness_claim"],
        "golden_case_count": len(GOLDEN_CASES),
        "checker_agreement_count": agreement["checker_agreement_count"],
        "checker_agreement_failures": agreement["checker_agreement_failures"],
        "generated_validation_example_count": agreement["generated_validation_example_count"],
        "per_program": program_rows,
        "per_split": split_rows,
        "cross_split_function_overlap": overlap,
        "heldout_vs_train": heldout_overlap,
        "pytest_q_passed": pytest_passed,
        "failed_gates": sorted(set(failed_gates)),
        "verdict": verdict,
        "claim_boundary": {
            "established": "Semantic order distinguishability and heldout-vs-train non-overlap are established over the exact whole-grid semantic domain only when whole_grid_exhaustiveness_claim is true; otherwise they are established only over the deterministic single-cell branch-complete domain.",
            "not_established": "This audit does not validate model training, checkpoint quality, ARC generalization, or any future R5A/R5B design.",
        },
    }
    return report, build_samples(program_rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACT_ROOT / "order_audit_report.json")
    parser.add_argument("--samples-out", type=Path, default=ARTIFACT_ROOT / "order_audit_samples.json")
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)

    report, samples = build_report(args.pytest_result)
    write_json(args.out, report)
    write_json(args.samples_out, samples)
    print(json.dumps({"verdict": report["verdict"], "report_sha256": sha256_file(args.out), "samples_sha256": sha256_file(args.samples_out)}, sort_keys=True))
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())
