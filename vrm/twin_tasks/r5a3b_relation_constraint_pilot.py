from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .r5a2d_r1_corpus_resolver import sha256_file
from .r5a3a_arc_native_candidate_machine import SupportPair
from .r5a3a_scene_hypotheses import (
    Grid,
    canonical_grid_sha256,
    freeze_grid,
    grid_shape,
    grid_to_list,
    stable_hash,
)
from .r5a3b_constraint_acquisition import (
    CONSTRAINT_PREDICATES,
    ConstraintModel,
    acquire_constraints,
)
from .r5a3b_object_graph import EDGE_PREDICATES, build_object_graph
from .r5a3b_parse_hypotheses import (
    MAX_PARSE_HYPOTHESES,
    PARSE_MENU,
    enumerate_parse_states,
    modal_color,
)
from .r5a3b_relation_programs import (
    CONDITIONS,
    MATCHED_RANDOM_SALTS,
    MAX_PROGRAM_DEPTH,
    MAX_RAW_CANDIDATES_PER_FOLD,
    MAX_RAW_PROGRAMS_PER_PARSE,
    TOP_K,
    apply_conditions,
    build_raw_candidate_universe,
    recall_flags,
    relation_program_surface_spec,
    target_rank,
    visible_input_hash,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot"
PILOT_VERSION = "R5A3B_P0_RELATION_OBJECT_GRAPH_CONSTRAINT_CAUSAL_PILOT"
FOLD_SAFETY_CAP_SECONDS = 120.0
ARC_ROOT = PROJECT_ROOT / "arc-prize-2026-arc-agi-2"

NEW_SOURCE_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_parse_hypotheses.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_object_graph.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_relation_programs.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_constraint_acquisition.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_relation_constraint_pilot.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_r5a3b_relation_constraint_pilot.py",
    PROJECT_ROOT / "tests/test_r5a3b_relation_constraint_pilot.py",
    PROJECT_ROOT / "docs/R5A3B_P0_RELATION_CONSTRAINT_CAUSAL_PILOT.md",
)

PROTECTED_PRIOR_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_coverage_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_recall_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3a_arc_native_candidate_machine.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3a_scene_hypotheses.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3a_structural_programs.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3a_arc_audit.py",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/final_report.json",
    PROJECT_ROOT / "artifacts/r5a2d_r2_candidate_recall/final_report.json",
    PROJECT_ROOT / "artifacts/r5a3a_arc_native_candidate_machine/final_report.json",
)


@dataclass(frozen=True)
class PilotArcTask:
    task_id: str
    source_relative_path: str
    task_file_sha256: str
    train_inputs: tuple[Grid, ...]
    train_outputs: tuple[Grid, ...]
    test_inputs: tuple[Grid, ...]
    split_name: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_hashes(paths: tuple[Path, ...]) -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in paths}


def _split_name_from_challenge_file(path: Path) -> str:
    lower = path.name.lower()
    if "training" in lower:
        return "training"
    if "evaluation" in lower:
        return "evaluation"
    if "test" in lower:
        return "test"
    return path.stem


def load_arc_challenge_tasks_no_solutions() -> list[PilotArcTask]:
    tasks: list[PilotArcTask] = []
    for path in sorted(ARC_ROOT.glob("*_challenges.json")):
        split_name = _split_name_from_challenge_file(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_hash = sha256_file(path)
        relative = str(path.relative_to(PROJECT_ROOT))
        for task_id in sorted(payload):
            task_payload = payload[task_id]
            if not isinstance(task_payload, dict):
                continue
            train_inputs = []
            train_outputs = []
            for pair in task_payload.get("train", []):
                if "input" in pair and "output" in pair:
                    train_inputs.append(freeze_grid(pair["input"]))
                    train_outputs.append(freeze_grid(pair["output"]))
            test_inputs = tuple(freeze_grid(pair["input"]) for pair in task_payload.get("test", []) if "input" in pair)
            if train_inputs and len(train_inputs) == len(train_outputs):
                tasks.append(
                    PilotArcTask(
                        task_id=f"{split_name}:{task_id}",
                        source_relative_path=f"{relative}#{task_id}",
                        task_file_sha256=file_hash,
                        train_inputs=tuple(train_inputs),
                        train_outputs=tuple(train_outputs),
                        test_inputs=test_inputs,
                        split_name=split_name,
                    )
                )
    return sorted(tasks, key=lambda task: (task.source_relative_path, task.task_id))


def _object_count_modal_bg_conn4(grid: Grid) -> int:
    from .r5a3b_constraint_acquisition import parse_state_for_grid

    return parse_state_for_grid(grid, modal_color(grid), 4).object_count


def _task_input_hash(task: Any) -> str:
    return stable_hash(
        {
            "train_input_hashes": [canonical_grid_sha256(grid) for grid in task.train_inputs],
            "test_input_hashes": [canonical_grid_sha256(grid) for grid in task.test_inputs],
            "train_input_shapes": [list(grid_shape(grid)) for grid in task.train_inputs],
            "test_input_shapes": [list(grid_shape(grid)) for grid in task.test_inputs],
        }
    )


def _area_bin(task: Any) -> str:
    areas = []
    for grid in list(task.train_inputs) + list(task.test_inputs):
        height, width = grid_shape(grid)
        areas.append(int(height) * int(width or 0))
    area = max(areas) if areas else 0
    if area <= 100:
        return "small"
    if area <= 400:
        return "medium"
    return "large"


def _task_record_from_inputs(task: Any) -> dict[str, Any]:
    counts = [_object_count_modal_bg_conn4(grid) for grid in task.train_inputs]
    object_slice = "one_object" if counts and max(counts) <= 1 else "multi_object"
    return {
        "task_id": task.task_id,
        "source_relative_path": task.source_relative_path,
        "split_name": task.split_name,
        "support_pair_count": len(task.train_inputs),
        "train_input_count": len(task.train_inputs),
        "test_input_count": len(task.test_inputs),
        "input_only_object_counts_modal_bg_conn4": counts,
        "object_slice": object_slice,
        "area_bin": _area_bin(task),
        "input_only_selection_hash": _task_input_hash(task),
    }


def _round_robin_select(records: list[dict[str, Any]], desired: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[(record["area_bin"], record["support_pair_count"])].append(record)
    for key in buckets:
        buckets[key] = sorted(buckets[key], key=lambda row: row["input_only_selection_hash"])
    selected = []
    seen = set()
    keys = sorted(buckets)
    while len(selected) < desired:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            while bucket and bucket[0]["input_only_selection_hash"] in seen:
                bucket.pop(0)
            if bucket:
                row = bucket.pop(0)
                selected.append(row)
                seen.add(row["input_only_selection_hash"])
                progressed = True
                if len(selected) >= desired:
                    break
        if not progressed:
            break
    if len(selected) < desired:
        remainder = sorted(
            [row for row in records if row["input_only_selection_hash"] not in seen],
            key=lambda row: row["input_only_selection_hash"],
        )
        selected.extend(remainder[: desired - len(selected)])
    return selected[:desired]


def pilot_selection_spec() -> dict[str, Any]:
    payload = {
        "pilot_version": PILOT_VERSION,
        "requested_task_count": 24,
        "requested_multi_object_count": 18,
        "requested_one_object_count": 6,
        "selection_phase": "before_fold_hidden_outputs_are_used_for_metrics",
        "allowed_inputs_for_selection": [
            "train input grids",
            "test input grids",
            "train input grid shapes",
            "test input grid shapes",
            "train support-pair count",
            "split name",
        ],
        "forbidden_inputs_for_selection": [
            "train output grids",
            "held-out fold output grids",
            "ARC test outputs",
            "task id as tie-break or runtime input",
        ],
        "object_classifier": {
            "background": "modal color from raw input grid",
            "connectivity": 4,
            "object_count": "non-background same-color connected component count",
            "task_slice_rule": "one_object if every train input has at most one object; otherwise multi_object",
        },
        "grid_size_bins": {
            "small": "max input area <= 100",
            "medium": "100 < max input area <= 400",
            "large": "max input area > 400",
        },
        "tie_break": "SHA256 over raw input grids and input shapes only, never task id",
    }
    payload["selection_spec_sha256"] = stable_hash(payload)
    return payload


def select_pilot_manifest(tasks: list[Any]) -> dict[str, Any]:
    records = [
        _task_record_from_inputs(task)
        for task in tasks
        if task.split_name in {"training", "evaluation"} and len(task.train_inputs) >= 2
    ]
    multi = [row for row in records if row["object_slice"] == "multi_object"]
    one = [row for row in records if row["object_slice"] == "one_object"]
    selected = _round_robin_select(multi, 18) + _round_robin_select(one, 6)
    selected = sorted(selected, key=lambda row: row["input_only_selection_hash"])
    payload = {
        "pilot_version": PILOT_VERSION,
        "task_count": len(selected),
        "tasks": selected,
        "selection_limitations": [] if len(selected) == 24 else [f"selected_{len(selected)}_tasks_instead_of_24"],
        "counts": dict(Counter(row["object_slice"] for row in selected)),
        "area_bin_counts": dict(Counter(row["area_bin"] for row in selected)),
        "support_pair_count_distribution": dict(Counter(str(row["support_pair_count"]) for row in selected)),
        "task_ids_recorded_for_manifest_only": True,
        "task_ids_used_for_runtime_candidate_generation_or_ranking": False,
    }
    payload["pilot_manifest_sha256"] = stable_hash(payload)
    return payload


def pre_hidden_pilot_commitment(manifest: dict[str, Any], selection_spec: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "pilot_version": PILOT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "statement": "This commitment is written before any held-out fold output is used for pilot metrics.",
        "arc_loader": "challenge_files_only_no_solution_file_read_or_hash",
        "pilot_manifest_sha256": stable_hash(manifest),
        "pilot_selection_spec_sha256": stable_hash(selection_spec),
        "source_hashes": path_hashes(NEW_SOURCE_PATHS),
        "prior_protected_hashes_before": path_hashes(PROTECTED_PRIOR_PATHS),
        "parse_menu": PARSE_MENU,
        "max_parse_hypotheses": MAX_PARSE_HYPOTHESES,
        "relation_menu": list(EDGE_PREDICATES),
        "constraints": list(CONSTRAINT_PREDICATES),
        "grammar": relation_program_surface_spec(),
        "budgets": {
            "MAX_PARSE_HYPOTHESES": MAX_PARSE_HYPOTHESES,
            "MAX_RAW_PROGRAMS_PER_PARSE": MAX_RAW_PROGRAMS_PER_PARSE,
            "MAX_RAW_CANDIDATES_PER_FOLD": MAX_RAW_CANDIDATES_PER_FOLD,
            "TOP_K": TOP_K,
            "MAX_PROGRAM_DEPTH": MAX_PROGRAM_DEPTH,
        },
        "condition_definitions": list(CONDITIONS),
        "matched_random_salts": list(MATCHED_RANDOM_SALTS),
        "task_id_runtime_input": False,
        "metrics": [
            "acceptance_rates",
            "candidate_emission",
            "raw_candidate_distribution",
            "after_prune_counts",
            "unique_distribution",
            "Recall@K all/multi/one_object",
            "ExactHitsPer1000Candidates",
            "mean candidates evaluated/surviving",
            "hard_prune_rate",
            "soft_rank_effect",
            "primary relation/constraint/smart-prune deltas",
        ],
        "hidden_fold_outputs_used_before_this_commitment": False,
        "arc_test_outputs_used": False,
    }
    payload["commitment_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "created_utc"})
    return payload


def mechanism_preflight_report() -> dict[str, Any]:
    relation_grid = ((0, 0, 0), (1, 2, 0), (0, 0, 0))
    support = SupportPair(relation_grid, ((0, 0, 0), (1, 1, 0), (0, 0, 0)))
    universe = build_raw_candidate_universe((support,), relation_grid)
    relation_candidates = [row for row in universe["raw_candidate_json"] if row["program_family"] == "relation"]
    tie_grid = ((0, 1, 0), (2, 0, 3), (0, 0, 0))
    tie_parse = enumerate_parse_states(tie_grid)[0]
    tie_graph = build_object_graph(tie_parse)
    constraints = acquire_constraints((support,), background_color=0, connectivity=4)
    payload = {
        "pilot_version": PILOT_VERSION,
        "golden_cases": [
            {
                "name": "relation_candidate_generation_recolor_target",
                "relation_candidate_count": len(relation_candidates),
                "passed": any("recolor_target_to_source" in row["operator_sequence"] for row in relation_candidates),
            },
            {
                "name": "nearest_tie_is_represented",
                "nearest_tie_count": tie_graph.nearest_tie_count,
                "passed": tie_graph.nearest_tie_count > 0,
            },
            {
                "name": "constraints_are_visible_support_only",
                "hard_constraint_count": len(constraints.hard_constraints()),
                "passed": constraints.source == "visible_supports_only",
            },
        ],
        "no_training_or_model_components": True,
        "task_id_runtime_input": False,
        "arc_test_outputs_used": False,
    }
    payload["passed"] = all(row["passed"] for row in payload["golden_cases"])
    payload["report_sha256"] = stable_hash(payload)
    return payload


def property_test_report() -> dict[str, Any]:
    supports = (
        SupportPair(((1, 0, 2), (0, 0, 0), (3, 0, 0)), ((1, 0, 1), (0, 0, 0), (3, 0, 0))),
        SupportPair(((4, 0, 5), (0, 0, 0), (6, 0, 0)), ((4, 0, 4), (0, 0, 0), (6, 0, 0))),
    )
    query = ((7, 0, 8), (0, 0, 0), (9, 0, 0))
    universe = build_raw_candidate_universe(supports, query)
    parse_states = enumerate_parse_states(query)
    constraints = {
        parse_state.parse_id: acquire_constraints(supports, background_color=parse_state.background_color, connectivity=parse_state.connectivity)
        for parse_state in parse_states
    }
    visible_hash = visible_input_hash(supports, query)
    condition_rows = apply_conditions(
        raw_candidates=universe["raw_candidates"],
        query=query,
        constraint_by_parse=constraints,
        visible_hash=visible_hash,
    )
    raw_ids = {row["candidate_id"] for row in universe["raw_candidate_json"]}
    condition_ids_subset = all(
        set(run["candidate_ids"]).issubset(raw_ids)
        for key, row in condition_rows.items()
        for run in (row.get("runs") if key == "MATCHED_RANDOM_PRUNE" else [row])
    )
    random_counts_match = all(
        run["unique_candidate_count"] <= condition_rows["RELATION_HARD_CONSTRAINT"]["unique_candidate_count"]
        for run in condition_rows["MATCHED_RANDOM_PRUNE"]["runs"]
    )
    rows = [
        {
            "name": "budget_cap",
            "passed": universe["raw_candidate_count"] <= MAX_RAW_CANDIDATES_PER_FOLD,
            "observed_raw_candidate_count": universe["raw_candidate_count"],
        },
        {
            "name": "all_conditions_are_subsets_of_same_raw_universe",
            "passed": condition_ids_subset,
        },
        {
            "name": "matched_random_prune_never_exceeds_hard_survivor_count",
            "passed": random_counts_match,
        },
        {
            "name": "raw_grid_fidelity_shape_preserved_by_p0_renderer",
            "passed": all(grid_shape(row["output_grid"]) == grid_shape(query) for row in universe["raw_candidate_json"]),
        },
    ]
    payload = {
        "pilot_version": PILOT_VERSION,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
    payload["report_sha256"] = stable_hash(payload)
    return payload


def _fold_signature(manifest_row: dict[str, Any], query: Grid) -> dict[str, Any]:
    return {
        "object_slice": manifest_row["object_slice"],
        "area_bin": manifest_row["area_bin"],
        "query_input_hash": canonical_grid_sha256(query),
        "query_shape": list(grid_shape(query)),
    }


def build_pilot_ledgers(tasks: list[Any], manifest: dict[str, Any]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    raw_rows = []
    condition_rows = []
    parse_reports = []
    graph_reports = []
    constraint_reports = []
    runtime_rows = []
    safety_stop = None
    for manifest_row in manifest["tasks"]:
        task = task_by_id[manifest_row["task_id"]]
        for fold_index, query in enumerate(task.train_inputs):
            fold_start = time.perf_counter()
            support_indices = [index for index in range(len(task.train_inputs)) if index != fold_index]
            supports = tuple(SupportPair(task.train_inputs[index], task.train_outputs[index]) for index in support_indices)
            parse_start = time.perf_counter()
            parse_states = enumerate_parse_states(query)
            parse_seconds = time.perf_counter() - parse_start
            constraint_start = time.perf_counter()
            constraint_by_parse: dict[str, ConstraintModel] = {
                parse_state.parse_id: acquire_constraints(
                    supports,
                    background_color=parse_state.background_color,
                    connectivity=parse_state.connectivity,
                )
                for parse_state in parse_states
            }
            constraint_seconds = time.perf_counter() - constraint_start
            universe = build_raw_candidate_universe(supports, query)
            visible_hash = visible_input_hash(supports, query)
            condition_start = time.perf_counter()
            conditions = apply_conditions(
                raw_candidates=universe["raw_candidates"],
                query=query,
                constraint_by_parse=constraint_by_parse,
                visible_hash=visible_hash,
            )
            condition_seconds = time.perf_counter() - condition_start
            fold_seconds = time.perf_counter() - fold_start
            fold_key = f"{manifest_row['input_only_selection_hash'][:12]}:fold={fold_index}"
            raw_rows.append(
                {
                    "fold_key": fold_key,
                    "task_id_for_log_only": task.task_id,
                    "fold_index": fold_index,
                    "support_indices": support_indices,
                    "visible_input_hash": visible_hash,
                    "signature": _fold_signature(manifest_row, query),
                    "raw_candidate_count": universe["raw_candidate_count"],
                    "unique_output_count": universe["unique_output_count"],
                    "raw_candidates": universe["raw_candidate_json"],
                }
            )
            condition_rows.append(
                {
                    "fold_key": fold_key,
                    "task_id_for_log_only": task.task_id,
                    "fold_index": fold_index,
                    "signature": _fold_signature(manifest_row, query),
                    "conditions": conditions,
                }
            )
            parse_reports.append(
                {
                    "fold_key": fold_key,
                    "parse_count": len(parse_states),
                    "parse_states": [parse_state.to_json() for parse_state in parse_states],
                    "candidate_parse_rows": universe["parse_rows"],
                }
            )
            for parse_state in parse_states:
                graph = build_object_graph(parse_state)
                graph_reports.append(
                    {
                        "fold_key": fold_key,
                        "parse_id": parse_state.parse_id,
                        "node_count": len(graph.nodes),
                        "edge_count": len(graph.edges),
                        "nearest_tie_count": graph.nearest_tie_count,
                        "graph": graph.to_json(),
                    }
                )
            constraint_reports.append(
                {
                    "fold_key": fold_key,
                    "models": {parse_id: model.to_json() for parse_id, model in constraint_by_parse.items()},
                }
            )
            candidate_runtime = universe["runtime"]
            runtime_rows.append(
                {
                    "fold_key": fold_key,
                    "task_id_for_log_only": task.task_id,
                    "fold_index": fold_index,
                    "scene_parsing_seconds": parse_seconds + candidate_runtime["scene_parsing_seconds"],
                    "graph_construction_seconds": candidate_runtime["graph_construction_seconds"],
                    "constraint_acquisition_seconds": constraint_seconds,
                    "candidate_enumeration_seconds": candidate_runtime["candidate_enumeration_seconds"],
                    "execution_render_seconds": candidate_runtime["execution_render_seconds"],
                    "visible_support_scoring_seconds": candidate_runtime["visible_support_scoring_seconds"],
                    "pruning_dedup_hash_seconds": condition_seconds,
                    "serialization_seconds": 0.0,
                    "total_fold_seconds": fold_seconds,
                    "raw_candidate_count": universe["raw_candidate_count"],
                    "unique_output_count": universe["unique_output_count"],
                    "safety_cap_seconds": FOLD_SAFETY_CAP_SECONDS,
                    "safety_cap_exceeded": fold_seconds > FOLD_SAFETY_CAP_SECONDS,
                }
            )
            if fold_seconds > FOLD_SAFETY_CAP_SECONDS:
                safety_stop = {
                    "fold_key": fold_key,
                    "task_id_for_log_only": task.task_id,
                    "fold_index": fold_index,
                    "fold_seconds": fold_seconds,
                    "reason": "fold_runtime_exceeded_120_seconds",
                }
                break
        if safety_stop:
            break
    return {
        "raw_candidate_universe_ledger": raw_rows,
        "condition_candidate_ledgers": condition_rows,
        "parse_hypothesis_report": {
            "pilot_version": PILOT_VERSION,
            "fold_count": len(parse_reports),
            "rows": parse_reports,
        },
        "object_graph_report": {
            "pilot_version": PILOT_VERSION,
            "graph_count": len(graph_reports),
            "rows": graph_reports,
        },
        "constraint_acquisition_report": {
            "pilot_version": PILOT_VERSION,
            "fold_count": len(constraint_reports),
            "rows": constraint_reports,
        },
        "runtime_rows": runtime_rows,
        "safety_stop": safety_stop,
    }


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    rows = sorted(values)
    p95_index = min(len(rows) - 1, int(0.95 * (len(rows) - 1)))
    return {
        "count": len(rows),
        "mean": sum(rows) / len(rows),
        "median": statistics.median(rows),
        "p95": rows[p95_index],
        "max": max(rows),
    }


def runtime_profile(runtime_rows: list[dict[str, Any]], safety_stop: dict[str, Any] | None) -> dict[str, Any]:
    stages = [
        "scene_parsing_seconds",
        "graph_construction_seconds",
        "constraint_acquisition_seconds",
        "candidate_enumeration_seconds",
        "execution_render_seconds",
        "visible_support_scoring_seconds",
        "pruning_dedup_hash_seconds",
        "serialization_seconds",
        "total_fold_seconds",
    ]
    return {
        "pilot_version": PILOT_VERSION,
        "fold_count": len(runtime_rows),
        "stage_stats": {stage: _summary_stats([float(row[stage]) for row in runtime_rows]) for stage in stages},
        "slowest_10_folds": sorted(runtime_rows, key=lambda row: row["total_fold_seconds"], reverse=True)[:10],
        "safety_stop": safety_stop,
        "any_fold_exceeded_120s": any(row["safety_cap_exceeded"] for row in runtime_rows),
    }


def _empty_metric(condition: str) -> dict[str, Any]:
    return {
        "condition": condition,
        "fold_count": 0,
        "candidate_emission_rate": {"numerator": 0, "denominator": 0, "rate": 0.0},
        "unique_candidate_distribution": {},
        "after_prune_count_distribution": {},
        "recall": {f"recall_at_{k}": {"numerator": 0, "denominator": 0, "rate": 0.0} for k in TOP_K},
        "recall_splits": {},
        "exact_hits_per_1000_candidates": 0.0,
        "mean_candidates_surviving": 0.0,
        "mean_candidates_evaluated": 0.0,
    }


def _aggregate_condition_rows(condition: str, rows: list[dict[str, Any]], *, raw_counts: dict[str, int]) -> dict[str, Any]:
    if not rows:
        return _empty_metric(condition)
    fold_count = len(rows)
    recall = {}
    for k in TOP_K:
        key = f"recall_at_{k}"
        hits = sum(row["recalls"][key] for row in rows)
        recall[key] = {"numerator": hits, "denominator": fold_count, "rate": hits / fold_count}
    splits: dict[str, Any] = {}
    for slice_name in ("all", "multi_object", "one_object"):
        slice_rows = rows if slice_name == "all" else [row for row in rows if row["signature"]["object_slice"] == slice_name]
        splits[slice_name] = {
            f"recall_at_{k}": {
                "numerator": sum(row["recalls"][f"recall_at_{k}"] for row in slice_rows),
                "denominator": len(slice_rows),
                "rate": sum(row["recalls"][f"recall_at_{k}"] for row in slice_rows) / len(slice_rows) if slice_rows else 0.0,
            }
            for k in TOP_K
        }
    total_candidates = sum(row["candidate_count"] for row in rows)
    exact_hits = sum(row["target_rank"] is not None for row in rows)
    return {
        "condition": condition,
        "fold_count": fold_count,
        "candidate_emission_rate": {
            "numerator": sum(row["candidate_count"] > 0 for row in rows),
            "denominator": fold_count,
            "rate": sum(row["candidate_count"] > 0 for row in rows) / fold_count,
        },
        "unique_candidate_distribution": dict(Counter(str(row["candidate_count"]) for row in rows)),
        "after_prune_count_distribution": dict(Counter(str(row["candidate_count"]) for row in rows)),
        "recall": recall,
        "recall_splits": splits,
        "exact_hits_per_1000_candidates": exact_hits / total_candidates * 1000.0 if total_candidates else 0.0,
        "mean_candidates_surviving": total_candidates / fold_count,
        "mean_candidates_evaluated": sum(raw_counts[row["fold_key"]] for row in rows) / fold_count,
        "target_ranks": [row["target_rank"] for row in rows],
    }


def score_condition_ledgers(
    tasks: list[Any],
    raw_ledger: list[dict[str, Any]],
    condition_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    raw_counts = {row["fold_key"]: row["raw_candidate_count"] for row in raw_ledger}
    scored_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    random_scored_by_salt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_candidate_distribution = dict(Counter(str(row["raw_candidate_count"]) for row in raw_ledger))
    for row in condition_ledger:
        task = task_by_id[row["task_id_for_log_only"]]
        target = task.train_outputs[row["fold_index"]]
        for condition in ("OBJECT_ONLY", "RELATION_NO_PRUNE", "RELATION_SOFT_CONSTRAINT", "RELATION_HARD_CONSTRAINT"):
            condition_payload = row["conditions"][condition]
            candidates = condition_payload["top_candidates"]
            rank = target_rank(candidates, target)
            scored_by_condition[condition].append(
                {
                    "fold_key": row["fold_key"],
                    "signature": row["signature"],
                    "candidate_count": len(candidates),
                    "target_rank": rank,
                    "recalls": recall_flags(candidates, target),
                }
            )
        for run in row["conditions"]["MATCHED_RANDOM_PRUNE"]["runs"]:
            candidates = run["top_candidates"]
            rank = target_rank(candidates, target)
            random_scored_by_salt[run["salt"]].append(
                {
                    "fold_key": row["fold_key"],
                    "signature": row["signature"],
                    "candidate_count": len(candidates),
                    "target_rank": rank,
                    "recalls": recall_flags(candidates, target),
                }
            )
    metrics = {
        "pilot_version": PILOT_VERSION,
        "hidden_fold_outputs_used_only_in_this_scoring_stage": True,
        "acceptance_rates": {
            "fold_acceptance_rate": {
                "numerator": len(condition_ledger),
                "denominator": len(condition_ledger),
                "rate": 1.0 if condition_ledger else 0.0,
            }
        },
        "raw_candidate_distribution": raw_candidate_distribution,
        "conditions": {
            condition: _aggregate_condition_rows(condition, rows, raw_counts=raw_counts)
            for condition, rows in sorted(scored_by_condition.items())
        },
        "matched_random_by_salt": {
            salt: _aggregate_condition_rows("MATCHED_RANDOM_PRUNE", rows, raw_counts=raw_counts)
            for salt, rows in sorted(random_scored_by_salt.items())
        },
    }
    random_recall_at_128_rates = [
        payload["recall"]["recall_at_128"]["rate"]
        for payload in metrics["matched_random_by_salt"].values()
    ]
    metrics["conditions"]["MATCHED_RANDOM_PRUNE"] = {
        "condition": "MATCHED_RANDOM_PRUNE",
        "salt_count": len(metrics["matched_random_by_salt"]),
        "mean_recall_at_128": sum(random_recall_at_128_rates) / len(random_recall_at_128_rates) if random_recall_at_128_rates else 0.0,
        "min_recall_at_128": min(random_recall_at_128_rates) if random_recall_at_128_rates else 0.0,
        "max_recall_at_128": max(random_recall_at_128_rates) if random_recall_at_128_rates else 0.0,
    }
    no_prune = metrics["conditions"].get("RELATION_NO_PRUNE", _empty_metric("RELATION_NO_PRUNE"))
    soft = metrics["conditions"].get("RELATION_SOFT_CONSTRAINT", _empty_metric("RELATION_SOFT_CONSTRAINT"))
    hard = metrics["conditions"].get("RELATION_HARD_CONSTRAINT", _empty_metric("RELATION_HARD_CONSTRAINT"))
    obj = metrics["conditions"].get("OBJECT_ONLY", _empty_metric("OBJECT_ONLY"))
    metrics["hard_prune_rate"] = {
        "mean_removed_fraction": 1.0 - (hard["mean_candidates_surviving"] / no_prune["mean_candidates_surviving"])
        if no_prune["mean_candidates_surviving"]
        else 0.0
    }
    soft_ranks = [rank for rank in soft.get("target_ranks", []) if rank is not None]
    no_prune_ranks = [rank for rank in no_prune.get("target_ranks", []) if rank is not None]
    metrics["soft_rank_effect"] = {
        "mean_rank_no_prune_when_hit": sum(no_prune_ranks) / len(no_prune_ranks) if no_prune_ranks else None,
        "mean_rank_soft_when_hit": sum(soft_ranks) / len(soft_ranks) if soft_ranks else None,
    }
    metrics["primary_deltas"] = {
        "relation_delta_recall_at_128": no_prune["recall"]["recall_at_128"]["rate"] - obj["recall"]["recall_at_128"]["rate"],
        "constraint_delta_recall_at_128": hard["recall"]["recall_at_128"]["rate"] - no_prune["recall"]["recall_at_128"]["rate"],
        "smart_prune_delta_vs_random_mean_recall_at_128": hard["recall"]["recall_at_128"]["rate"]
        - metrics["conditions"]["MATCHED_RANDOM_PRUNE"]["mean_recall_at_128"],
    }
    metrics["metrics_sha256"] = stable_hash(metrics)
    return metrics


def matched_random_prune_report(condition_ledger: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for fold in condition_ledger:
        random_payload = fold["conditions"]["MATCHED_RANDOM_PRUNE"]
        rows.append(
            {
                "fold_key": fold["fold_key"],
                "matched_survivor_count": random_payload["matched_survivor_count"],
                "salt_candidate_counts": {run["salt"]: run["unique_candidate_count"] for run in random_payload["runs"]},
            }
        )
    return {
        "pilot_version": PILOT_VERSION,
        "salts": list(MATCHED_RANDOM_SALTS),
        "visible_hash_only_seed_material": True,
        "task_id_used_in_random_seed": False,
        "rows": rows,
        "metrics": metrics["conditions"]["MATCHED_RANDOM_PRUNE"],
    }


def causal_comparison_report(metrics: dict[str, Any]) -> dict[str, Any]:
    deltas = metrics["primary_deltas"]
    if deltas["relation_delta_recall_at_128"] > 0 or deltas["smart_prune_delta_vs_random_mean_recall_at_128"] > 0:
        next_action = "B"
    elif deltas["constraint_delta_recall_at_128"] < 0:
        next_action = "C"
    else:
        next_action = "A"
    return {
        "pilot_version": PILOT_VERSION,
        "primary_deltas": deltas,
        "interpretation": {
            "relation_delta": "RELATION_NO_PRUNE minus OBJECT_ONLY",
            "constraint_delta": "RELATION_HARD_CONSTRAINT minus RELATION_NO_PRUNE",
            "smart_prune_delta": "RELATION_HARD_CONSTRAINT minus MATCHED_RANDOM_PRUNE mean across salts",
        },
        "narrowest_justified_next_action": next_action,
    }


def representative_case_cards(condition_ledger: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    lines = ["# R5A-3B-P0 Representative Case Cards", ""]
    lines.append("Cases are keyed by input-only fold hashes; task ids are shown only as manifest/log references.")
    lines.append("")
    for row in condition_ledger[:8]:
        lines.append(f"## {row['fold_key']}")
        lines.append(f"- log task id: {row['task_id_for_log_only']}")
        lines.append(f"- fold: {row['fold_index']}")
        lines.append(f"- slice: {row['signature']['object_slice']}")
        for condition in ("OBJECT_ONLY", "RELATION_NO_PRUNE", "RELATION_HARD_CONSTRAINT"):
            payload = row["conditions"][condition]
            lines.append(f"- {condition}: {len(payload['top_candidates'])} unique candidates")
        lines.append("")
    lines.append("## Aggregate deltas")
    for key, value in metrics["primary_deltas"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def prior_integrity_verification(before: dict[str, str | None], after: dict[str, str | None]) -> dict[str, Any]:
    return {
        "pilot_version": PILOT_VERSION,
        "protected_prior_hashes_unchanged": before == after,
        "protected_prior_paths": list(before.keys()),
        "protected_prior_hashes_before": before,
        "protected_prior_hashes_after": after,
        "v2_topology_r5a2d_r5a3a_sources_or_artifacts_modified_by_pilot": before != after,
    }


def final_report(metrics: dict[str, Any], causal: dict[str, Any], integrity: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "Measured facts": {
            "pilot_version": PILOT_VERSION,
            "fold_count": metrics["acceptance_rates"]["fold_acceptance_rate"]["denominator"],
            "conditions": list(CONDITIONS),
            "recall_at_128_by_condition": {
                key: value["recall"]["recall_at_128"]["rate"]
                for key, value in metrics["conditions"].items()
                if key != "MATCHED_RANDOM_PRUNE"
            },
            "matched_random_mean_recall_at_128": metrics["conditions"]["MATCHED_RANDOM_PRUNE"]["mean_recall_at_128"],
            "primary_deltas": metrics["primary_deltas"],
            "protected_prior_hashes_unchanged": integrity["protected_prior_hashes_unchanged"],
            "any_fold_exceeded_120s": runtime["any_fold_exceeded_120s"],
        },
        "What this pilot establishes": [
            "A frozen 24-task input-selected causal pilot can compare object-only, relation-only, soft-constraint, hard-constraint, and matched-random-prune conditions from the same raw visible support information.",
            "The ledgers commit raw candidates and condition candidate lists before held-out fold outputs are used for exact metrics.",
        ],
        "What this pilot does not establish": [
            "It is not a full 4357-fold ARC corpus audit.",
            "It is not evidence of ARC solving, neural learning, hidden tool emergence, or transfer beyond this fixed pilot.",
            "It does not modify or reinterpret historical V2, topology, R5A-2D, or R5A-3A results.",
        ],
        "Narrowest justified next action": causal["narrowest_justified_next_action"],
    }


def strip_unstable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: strip_unstable(value)
            for key, value in payload.items()
            if key not in {"created_utc"}
            and not key.endswith("_seconds")
            and "seconds" not in key
            and key not in {"runtime_profile.json"}
        }
    if isinstance(payload, list):
        return [strip_unstable(value) for value in payload]
    return payload


def artifact_stable_hashes(artifact_root: Path) -> dict[str, str]:
    names = [
        "pilot_selection_spec.json",
        "pilot_manifest.json",
        "mechanism_preflight_report.json",
        "property_test_report.json",
        "pre_hidden_pilot_commitment.json",
        "parse_hypothesis_report.json",
        "object_graph_report.json",
        "constraint_acquisition_report.json",
        "raw_candidate_universe_ledger.json",
        "condition_candidate_ledgers.json",
        "matched_random_prune_report.json",
        "pilot_recall_metrics.json",
        "causal_comparison_report.json",
        "prior_integrity_verification.json",
        "final_report.json",
    ]
    rows = {}
    for name in names:
        path = artifact_root / name
        if path.exists():
            rows[name] = stable_hash(strip_unstable(load_json(path)))
    return rows


def run_pilot_once(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    protected_before = path_hashes(PROTECTED_PRIOR_PATHS)
    selection = pilot_selection_spec()
    tasks = load_arc_challenge_tasks_no_solutions()
    manifest = select_pilot_manifest(tasks)
    mechanism = mechanism_preflight_report()
    prop = property_test_report()
    commitment = pre_hidden_pilot_commitment(manifest, selection)
    write_json(artifact_root / "pilot_selection_spec.json", selection)
    write_json(artifact_root / "pilot_manifest.json", manifest)
    write_json(artifact_root / "mechanism_preflight_report.json", mechanism)
    write_json(artifact_root / "property_test_report.json", prop)
    write_json(artifact_root / "pre_hidden_pilot_commitment.json", commitment)
    ledgers = build_pilot_ledgers(tasks, manifest)
    write_json(artifact_root / "parse_hypothesis_report.json", ledgers["parse_hypothesis_report"])
    write_json(artifact_root / "object_graph_report.json", ledgers["object_graph_report"])
    write_json(artifact_root / "constraint_acquisition_report.json", ledgers["constraint_acquisition_report"])
    write_json(artifact_root / "raw_candidate_universe_ledger.json", ledgers["raw_candidate_universe_ledger"])
    write_json(artifact_root / "condition_candidate_ledgers.json", ledgers["condition_candidate_ledgers"])
    metrics = score_condition_ledgers(tasks, ledgers["raw_candidate_universe_ledger"], ledgers["condition_candidate_ledgers"])
    random_report = matched_random_prune_report(ledgers["condition_candidate_ledgers"], metrics)
    runtime = runtime_profile(ledgers["runtime_rows"], ledgers["safety_stop"])
    causal = causal_comparison_report(metrics)
    protected_after = path_hashes(PROTECTED_PRIOR_PATHS)
    integrity = prior_integrity_verification(protected_before, protected_after)
    final = final_report(metrics, causal, integrity, runtime)
    write_json(artifact_root / "matched_random_prune_report.json", random_report)
    write_json(artifact_root / "runtime_profile.json", runtime)
    write_json(artifact_root / "pilot_recall_metrics.json", metrics)
    write_json(artifact_root / "causal_comparison_report.json", causal)
    write_text(artifact_root / "representative_case_cards.md", representative_case_cards(ledgers["condition_candidate_ledgers"], metrics))
    write_json(artifact_root / "prior_integrity_verification.json", integrity)
    write_json(artifact_root / "final_report.json", final)
    return final


def run_pilot_twice(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    first = run_pilot_once(artifact_root=artifact_root)
    first_hashes = artifact_stable_hashes(artifact_root)
    second = run_pilot_once(artifact_root=artifact_root)
    second_hashes = artifact_stable_hashes(artifact_root)
    comparison = {
        "pilot_version": PILOT_VERSION,
        "two_identical_runs_stable": first_hashes == second_hashes,
        "differing_files": sorted(name for name in set(first_hashes) | set(second_hashes) if first_hashes.get(name) != second_hashes.get(name)),
        "run1": first_hashes,
        "run2": second_hashes,
        "first_report_next_action": first["Narrowest justified next action"],
        "second_report_next_action": second["Narrowest justified next action"],
    }
    write_json(artifact_root / "determinism_verification.json", comparison)
    final = load_json(artifact_root / "final_report.json")
    return final


def main() -> dict[str, Any]:
    return run_pilot_twice(artifact_root=ARTIFACT_ROOT)
