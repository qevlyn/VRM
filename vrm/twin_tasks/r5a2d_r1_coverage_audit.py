from __future__ import annotations

import copy
import hashlib
import io
import json
import time
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import compiled_topology_r5a2c_r2_data as topology_data
from . import compiled_topology_r5a2c_r2_executor as topology_executor
from . import transport_recolor_r5a0v2_oracles as v2_oracles
from .r5a2d_r1_corpus_resolver import (
    ARTIFACT_ROOT,
    PROJECT_ROOT,
    CorpusSelection,
    NoLocalArcCorpusError,
    resolve_and_precommit_corpus,
    sha256_file,
    stable_hash,
    write_json,
)
from .r5a2d_r1_raw_grid_fidelity import Grid, fidelity_json, freeze_grid, grid_shape

AUDIT_VERSION = "R5A2D_R1_FROZEN_LIBRARY_ARC_NATIVE_COVERAGE_AUDIT"
MEASURED_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_MEASURED"
NO_CORPUS_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_INCONCLUSIVE_NO_LOCAL_ARC_CORPUS"
CONTRACT_FAILURE_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_INCONCLUSIVE_CONTRACT_FAILURE"
NONDETERMINISTIC_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_INCONCLUSIVE_NONDETERMINISTIC"
INTEGRITY_FAILURE_VERDICT = "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_INCONCLUSIVE_INTEGRITY_FAILURE"

STRICT_VALIDATED = "STRICT_LOO_VALIDATED"
STRICT_FAILED = "STRICT_LOO_FAILED"
FULL_SUPPORT_ONLY = "FULL_SUPPORT_FIT_ONLY"
NO_UNIQUE = "NO_UNIQUE_HYPOTHESIS"
OUT_GRID = "OUT_OF_GRID_DOMAIN"
OUT_PALETTE = "OUT_OF_PALETTE_DOMAIN"
OUT_SUPPORT = "OUT_OF_SUPPORT_COUNT_DOMAIN"
RAW_REJECTED = "RAW_INTERFACE_REJECTED"
RUNTIME_ERROR = "RUNTIME_ERROR"
IN_DOMAIN = "IN_DOMAIN_AND_EXECUTED"
NOT_APPLICABLE = "NOT_APPLICABLE"

TOPO_RAW = "TOPOLOGY_RAW_DOMAIN_ELIGIBLE"
TOPO_FULL = "TOPOLOGY_FULL_SUPPORT_FIT"
TOPO_LOO_INFEASIBLE = "TOPOLOGY_LOO_INFEASIBLE_SUPPORT_CONTRACT"
TOPO_LOO_FAILED = "TOPOLOGY_LOO_ATTEMPTED_AND_FAILED"
TOPO_STRICT = "TOPOLOGY_STRICT_LOO_VALIDATED"

MANDATORY_SCOPE_BOUNDARY = """This audit measures coverage of one frozen two-specialist library on one
precommitted local ARC corpus under strict train-pair validation.

It does not measure:
- ARC generality;
- the value of an arbitrarily large future specialist library;
- the human design/search cost of inventing new primitives;
- learned routing;
- learned program synthesis;
- test-time training;
- zero-compute ARC solving;
- performance after task-specific primitive edits;
- whether all future compiled-specialist libraries fail to scale."""

MANDATORY_LOW_ZERO_BOUNDARY = """A low or zero strict-LOO coverage result for this two-expert frozen library
measures only this library under this corpus and this strict support contract.

It does not establish that all future compiled-specialist libraries fail to scale,
nor does it establish that an expert which is LOO-infeasible would necessarily fail
when allowed to use the complete ARC training set."""

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1a_preregistration.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_matchability_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_frontier_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_compiled_topology_r5a2c_r2_audit.py",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/r5a1a_preregistration_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_preflight/preflight_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_matchability_audit/matchability_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2a_frontier/final_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/closure_domain.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/shortcut_catalog.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/final_report.json",
)


@dataclass(frozen=True)
class ArcTaskOpen:
    task_id: str
    source_relative_path: str
    task_file_sha256: str
    train_inputs: tuple[Grid, ...]
    train_outputs: tuple[Grid, ...]
    test_inputs: tuple[Grid, ...]
    split_name: str
    solution_relative_path: str | None


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_to_list(grid: Grid | None) -> list[list[int]] | None:
    if grid is None:
        return None
    return [list(row) for row in grid]


def _grids_equal(left: Grid | None, right: Grid | None) -> bool:
    return left is not None and right is not None and left == right


def _artifact_path(name: str, artifact_root: Path = ARTIFACT_ROOT) -> Path:
    return artifact_root / name


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def protected_hashes() -> dict[str, str | None]:
    rows: dict[str, str | None] = {}
    for path in PROTECTED_PATHS:
        rows[str(path)] = sha256_file(path) if path.exists() else None
    return rows


def stable_json_hash_file(path: Path) -> str:
    payload = _load_json(path)
    return stable_hash(strip_unstable_fields(payload))


def strip_unstable_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: strip_unstable_fields(value)
            for key, value in payload.items()
            if key
            not in {
                "created_utc",
                "run_index",
                "observed_runtimes_seconds",
                "two_run_stable_hashes",
                "stable_two_run_artifact_hashes",
                "report_body_sha256_excluding_unstable_fields",
            }
        }
    if isinstance(payload, list):
        return [strip_unstable_fields(value) for value in payload]
    return payload


def _split_name_from_challenge_file(name: str) -> str:
    lower = name.lower()
    if "training" in lower:
        return "training"
    if "evaluation" in lower:
        return "evaluation"
    if "test" in lower:
        return "test"
    return Path(name).stem


def _solution_name_for_challenge(name: str) -> str | None:
    if not name.endswith("_challenges.json"):
        return None
    return name.replace("_challenges.json", "_solutions.json")


def _coerce_task(task_id: str, source_relative_path: str, task_file_sha256: str, payload: dict[str, Any], split_name: str, solution_relative_path: str | None) -> ArcTaskOpen:
    train_inputs = []
    train_outputs = []
    for pair in payload.get("train", []):
        if "input" in pair and "output" in pair:
            train_inputs.append(freeze_grid(pair["input"]))
            train_outputs.append(freeze_grid(pair["output"]))
    test_inputs = [freeze_grid(pair["input"]) for pair in payload.get("test", []) if "input" in pair]
    return ArcTaskOpen(
        task_id=task_id,
        source_relative_path=source_relative_path,
        task_file_sha256=task_file_sha256,
        train_inputs=tuple(train_inputs),
        train_outputs=tuple(train_outputs),
        test_inputs=tuple(test_inputs),
        split_name=split_name,
        solution_relative_path=solution_relative_path,
    )


def load_open_arc_tasks(selection: CorpusSelection) -> list[ArcTaskOpen]:
    tasks: list[ArcTaskOpen] = []
    files = sorted(selection.task_files, key=lambda item: item["relative_path"])
    for file_record in files:
        relative_path = file_record["relative_path"]
        raw_relative = Path(relative_path)
        corpus_relative = Path(file_record.get("corpus_relative_path", ""))
        if raw_relative.is_absolute():
            path = raw_relative
        elif corpus_relative and (selection.corpus_root / corpus_relative).exists():
            path = selection.corpus_root / corpus_relative
        else:
            path = PROJECT_ROOT / raw_relative
        if path.name.endswith("_solutions.json") or path.name == "sample_submission.json":
            continue
        payload = _load_json(path)
        solution_name = _solution_name_for_challenge(path.name)
        solution_relative = None
        if solution_name:
            solution_path = path.with_name(solution_name)
            if solution_path.exists():
                try:
                    solution_relative = str(solution_path.relative_to(PROJECT_ROOT))
                except ValueError:
                    solution_relative = str(solution_path)
        if isinstance(payload, dict) and "train" in payload and "test" in payload:
            tasks.append(_coerce_task(path.stem, relative_path, file_record["sha256"], payload, path.parent.name, solution_relative))
        elif isinstance(payload, dict):
            split_name = _split_name_from_challenge_file(path.name)
            for task_id in sorted(payload):
                task_payload = payload[task_id]
                if isinstance(task_payload, dict) and "train" in task_payload and "test" in task_payload:
                    full_task_id = f"{split_name}:{task_id}"
                    source = f"{relative_path}#{task_id}"
                    tasks.append(_coerce_task(full_task_id, source, file_record["sha256"], task_payload, split_name, solution_relative))
    return sorted(tasks, key=lambda item: (item.source_relative_path, item.task_id))


def _flatten_values(grids: Iterable[Grid]) -> list[int]:
    return [int(value) for grid in grids for row in grid for value in row]


def grid_shape_summary(task: ArcTaskOpen) -> dict[str, Any]:
    grids = list(task.train_inputs) + list(task.train_outputs) + list(task.test_inputs)
    counts = Counter(str(grid_shape(grid)) for grid in grids)
    return {"grid_count": len(grids), "shape_counts": dict(sorted(counts.items()))}


def palette_summary(task: ArcTaskOpen) -> dict[str, Any]:
    values = _flatten_values(list(task.train_inputs) + list(task.train_outputs) + list(task.test_inputs))
    return {
        "unique_values": sorted(set(values)),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "value_count": len(values),
    }


def _all_grids(task: ArcTaskOpen, *, include_test: bool = True) -> list[Grid]:
    grids = list(task.train_inputs) + list(task.train_outputs)
    if include_test:
        grids.extend(task.test_inputs)
    return grids


def _grid_size_ok(grid: Grid, size: int) -> bool:
    return grid_shape(grid) == (size, size)


def _palette_ok(grid: Grid, values: set[int]) -> bool:
    return all(int(value) in values for row in grid for value in row)


def _v2_domain_label(task: ArcTaskOpen) -> tuple[str, list[str]]:
    grids = _all_grids(task)
    if not all(_grid_size_ok(grid, v2_oracles.GRID_SIZE) for grid in grids):
        return OUT_GRID, ["v2_accepts_7x7_only"]
    if not all(_palette_ok(grid, set(range(v2_oracles.COLOR_COUNT))) for grid in grids):
        return OUT_PALETTE, ["v2_accepts_palette_0_to_8_only"]
    return IN_DOMAIN, []


def _topology_raw_domain(task: ArcTaskOpen) -> tuple[str, list[str], dict[str, bool]]:
    grids = _all_grids(task)
    grid_size_ok = all(_grid_size_ok(grid, topology_data.GRID_SIZE) for grid in grids)
    palette_ok = all(_palette_ok(grid, set(topology_data.PALETTE)) for grid in grids)
    marker_count_ok = all(sum(1 for row in grid for value in row if value in topology_data.MARKER_COLORS) <= topology_data.MAXIMUM_MARKER_COUNT for grid in grids)
    flags = {
        "grid_size_ok": grid_size_ok,
        "palette_ok": palette_ok,
        "marker_count_ok": marker_count_ok,
    }
    if not grid_size_ok:
        return OUT_GRID, ["topology_accepts_7x7_only"], flags
    if not palette_ok:
        return OUT_PALETTE, ["topology_accepts_palette_0_to_8_only"], flags
    if not marker_count_ok:
        return NOT_APPLICABLE, ["topology_marker_count_contract_exceeded"], flags
    return IN_DOMAIN, [], flags


def _fidelity_for_grids(grids: Iterable[Grid], *, expert_name: str, phase: str) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    for index, grid in enumerate(grids):
        evidence = fidelity_json(grid, grid, context=f"{expert_name}:{phase}:grid_{index}")
        rows.append(evidence)
    return rows, all(row["passed"] for row in rows)


def v2_consistent_sequences(support_inputs: list[Grid], support_outputs: list[Grid]) -> list[str]:
    matches: list[str] = []
    for sequence_class in v2_oracles.SEALED_CLASSES:
        ok = True
        for before, after in zip(support_inputs, support_outputs):
            try:
                predicted = v2_oracles.apply_sequence_to_grid(before, sequence_class)
            except Exception:
                ok = False
                break
            if predicted != after:
                ok = False
                break
        if ok:
            matches.append(sequence_class)
    return matches


def execute_v2_from_raw_grids(support_inputs: list[Grid], support_outputs: list[Grid], query_input: Grid) -> dict[str, Any]:
    matches = v2_consistent_sequences(support_inputs, support_outputs)
    if len(matches) != 1:
        return {
            "status": "ambiguous" if len(matches) > 1 else "no_consistent_sequence",
            "consistent_sequences": matches,
            "selected_sequence": None,
            "output_grid": None,
        }
    selected = matches[0]
    try:
        output = v2_oracles.apply_sequence_to_grid(query_input, selected)
    except Exception as exc:
        return {
            "status": "runtime_error",
            "consistent_sequences": matches,
            "selected_sequence": selected,
            "error": str(exc),
            "output_grid": None,
        }
    return {
        "status": "ok",
        "consistent_sequences": matches,
        "selected_sequence": selected,
        "output_grid": output,
    }


def _identity_evaluation(task: ArcTaskOpen) -> dict[str, Any]:
    fold_results = []
    for index, (query, target) in enumerate(zip(task.train_inputs, task.train_outputs)):
        fold_results.append(
            {
                "fold_index": index,
                "support_indices": [j for j in range(len(task.train_inputs)) if j != index],
                "status": "ok",
                "exact": query == target,
                "predicted_output_grid": _grid_to_list(query),
            }
        )
    strict = bool(fold_results) and all(row["exact"] for row in fold_results)
    full_support_fit = bool(task.train_inputs) and all(before == after for before, after in zip(task.train_inputs, task.train_outputs))
    test_candidates = [_grid_to_list(grid) for grid in task.test_inputs] if strict else []
    return {
        "primary_label": STRICT_VALIDATED if strict else STRICT_FAILED,
        "strict_label": STRICT_VALIDATED if strict else STRICT_FAILED,
        "full_support_fit": full_support_fit,
        "fold_results": fold_results,
        "loo_fold_count_attempted": len(fold_results),
        "test_candidates": test_candidates,
        "test_candidate_emitted": bool(test_candidates) or (strict and len(task.test_inputs) == 0),
    }


def _v2_full_support(task: ArcTaskOpen, raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    records, passed = _fidelity_for_grids(_all_grids(task, include_test=False), expert_name="v2", phase="full_support")
    raw_records.extend(records)
    if not passed:
        return {"attempted": False, "fit": False, "status": RAW_REJECTED, "selected_sequence": None}
    result = execute_v2_from_raw_grids(list(task.train_inputs), list(task.train_outputs), task.train_inputs[0]) if task.train_inputs else {"status": "no_train_pairs"}
    if result.get("status") != "ok":
        return {"attempted": True, "fit": False, "status": result.get("status"), "selected_sequence": None}
    selected = result["selected_sequence"]
    exact = all(v2_oracles.apply_sequence_to_grid(before, selected) == after for before, after in zip(task.train_inputs, task.train_outputs))
    return {"attempted": True, "fit": exact, "status": "ok", "selected_sequence": selected}


def _v2_evaluation(task: ArcTaskOpen) -> dict[str, Any]:
    domain_label, domain_reasons = _v2_domain_label(task)
    raw_records: list[dict[str, Any]] = []
    if domain_label != IN_DOMAIN:
        return {
            "primary_label": domain_label,
            "strict_label": domain_label,
            "full_support_fit": False,
            "full_support_status": domain_label,
            "loo_fold_count_attempted": 0,
            "fold_results": [],
            "raw_grid_fidelity_records": [],
            "test_candidates": [],
            "test_candidate_emitted": False,
            "domain_reasons": domain_reasons,
            "selected_sequences_by_fold": [],
        }
    full = _v2_full_support(task, raw_records)
    fold_results = []
    selected_sequences = []
    any_runtime_error = False
    any_no_unique = False
    any_mismatch = False
    any_raw_rejected = False
    for index, (query, target) in enumerate(zip(task.train_inputs, task.train_outputs)):
        support_indices = [j for j in range(len(task.train_inputs)) if j != index]
        support_inputs = [task.train_inputs[j] for j in support_indices]
        support_outputs = [task.train_outputs[j] for j in support_indices]
        records, passed = _fidelity_for_grids(support_inputs + support_outputs + [query], expert_name="v2", phase=f"loo_fold_{index}")
        raw_records.extend(records)
        if not passed:
            any_raw_rejected = True
            fold_results.append({"fold_index": index, "support_indices": support_indices, "status": RAW_REJECTED, "exact": False})
            continue
        result = execute_v2_from_raw_grids(support_inputs, support_outputs, query)
        status = result["status"]
        output = result.get("output_grid")
        exact = _grids_equal(output, target)
        selected_sequences.append(result.get("selected_sequence"))
        if status == "runtime_error":
            any_runtime_error = True
        elif status != "ok":
            any_no_unique = True
        elif not exact:
            any_mismatch = True
        fold_results.append(
            {
                "fold_index": index,
                "support_indices": support_indices,
                "status": status,
                "exact": exact,
                "selected_sequence": result.get("selected_sequence"),
                "consistent_sequence_count": len(result.get("consistent_sequences", [])),
                "predicted_output_grid": _grid_to_list(output),
            }
        )
    strict = bool(fold_results) and all(row["status"] == "ok" and row["exact"] for row in fold_results)
    if any_raw_rejected:
        primary = RAW_REJECTED
    elif any_runtime_error:
        primary = RUNTIME_ERROR
    elif strict:
        primary = STRICT_VALIDATED
    elif any_no_unique:
        primary = FULL_SUPPORT_ONLY if full["fit"] else NO_UNIQUE
    elif any_mismatch:
        primary = STRICT_FAILED
    else:
        primary = FULL_SUPPORT_ONLY if full["fit"] else NO_UNIQUE

    test_candidates: list[list[list[int]]] = []
    if strict:
        records, passed = _fidelity_for_grids(list(task.train_inputs) + list(task.train_outputs) + list(task.test_inputs), expert_name="v2", phase="test_candidate")
        raw_records.extend(records)
        if passed:
            for test_input in task.test_inputs:
                result = execute_v2_from_raw_grids(list(task.train_inputs), list(task.train_outputs), test_input)
                if result["status"] != "ok":
                    test_candidates = []
                    break
                test_candidates.append(_grid_to_list(result["output_grid"]))
    return {
        "primary_label": primary,
        "strict_label": STRICT_VALIDATED if strict else primary,
        "full_support_fit": bool(full["fit"]),
        "full_support_status": full["status"],
        "full_support_selected_sequence": full.get("selected_sequence"),
        "loo_fold_count_attempted": len(fold_results),
        "fold_results": fold_results,
        "raw_grid_fidelity_records": raw_records,
        "test_candidates": test_candidates,
        "test_candidate_emitted": bool(test_candidates) or (strict and len(task.test_inputs) == 0),
        "domain_reasons": domain_reasons,
        "selected_sequences_by_fold": [item for item in selected_sequences if item is not None],
    }


def _topology_full_support(task: ArcTaskOpen, raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(task.train_inputs) != topology_data.SUPPORT_PAIR_COUNT:
        return {"attempted": False, "fit": False, "status": "invalid_support_count"}
    records, passed = _fidelity_for_grids(_all_grids(task, include_test=False), expert_name="topology", phase="full_support")
    raw_records.extend(records)
    if not passed:
        return {"attempted": False, "fit": False, "status": RAW_REJECTED}
    result = topology_executor.execute_from_raw_grids(list(task.train_inputs), list(task.train_outputs), task.train_inputs[0])
    if result["status"] != "ok":
        return {"attempted": True, "fit": False, "status": result["status"], "selected_hypothesis": None}
    selected = result["selected_hypothesis"]
    exact = True
    for before, after in zip(task.train_inputs, task.train_outputs):
        output = topology_executor.execute_from_raw_grids(list(task.train_inputs), list(task.train_outputs), before).get("output_grid")
        if output != after:
            exact = False
            break
    return {"attempted": True, "fit": exact, "status": "ok", "selected_hypothesis": selected}


def _topology_subfamily(hypotheses: list[dict[str, Any]]) -> str | None:
    conventions = sorted({item.get("convention") for item in hypotheses if item})
    if not conventions:
        return None
    if all(convention == "seed_reachable_recolor" for convention in conventions):
        return "topology_seed_only"
    if all(convention in {"border_reachable_recolor", "enclosed_region_recolor"} for convention in conventions):
        return "topology_border_or_enclosed"
    return "topology_mixed_conventions"


def _topology_evaluation(task: ArcTaskOpen) -> dict[str, Any]:
    domain_label, domain_reasons, raw_flags = _topology_raw_domain(task)
    raw_records: list[dict[str, Any]] = []
    if domain_label != IN_DOMAIN:
        return {
            "primary_label": domain_label,
            "strict_label": domain_label,
            "loo_detail_status": TOPO_RAW if raw_flags["grid_size_ok"] and raw_flags["palette_ok"] else domain_label,
            "full_support_fit": False,
            "full_support_status": domain_label,
            "loo_fold_count_attempted": 0,
            "fold_results": [],
            "raw_grid_fidelity_records": [],
            "test_candidates": [],
            "test_candidate_emitted": False,
            "domain_reasons": domain_reasons,
            "raw_domain_flags": raw_flags,
            "topology_subfamily_if_any": None,
        }
    full = _topology_full_support(task, raw_records)
    remaining_support_count = len(task.train_inputs) - 1
    if remaining_support_count != topology_data.SUPPORT_PAIR_COUNT:
        return {
            "primary_label": OUT_SUPPORT,
            "strict_label": OUT_SUPPORT,
            "loo_detail_status": TOPO_LOO_INFEASIBLE,
            "full_support_fit": bool(full["fit"]),
            "full_support_status": full["status"],
            "loo_fold_count_attempted": 0,
            "fold_results": [],
            "raw_grid_fidelity_records": raw_records,
            "test_candidates": [],
            "test_candidate_emitted": False,
            "domain_reasons": ["topology_strict_loo_requires_exactly_3_remaining_support_pairs"],
            "raw_domain_flags": raw_flags,
            "topology_subfamily_if_any": _topology_subfamily([full.get("selected_hypothesis")] if full.get("selected_hypothesis") else []),
        }
    fold_results = []
    selected_hypotheses: list[dict[str, Any]] = []
    any_runtime_error = False
    any_no_unique = False
    any_mismatch = False
    any_raw_rejected = False
    for index, (query, target) in enumerate(zip(task.train_inputs, task.train_outputs)):
        support_indices = [j for j in range(len(task.train_inputs)) if j != index]
        support_inputs = [task.train_inputs[j] for j in support_indices]
        support_outputs = [task.train_outputs[j] for j in support_indices]
        records, passed = _fidelity_for_grids(support_inputs + support_outputs + [query], expert_name="topology", phase=f"loo_fold_{index}")
        raw_records.extend(records)
        if not passed:
            any_raw_rejected = True
            fold_results.append({"fold_index": index, "support_indices": support_indices, "status": RAW_REJECTED, "exact": False})
            continue
        try:
            result = topology_executor.execute_from_raw_grids(support_inputs, support_outputs, query)
        except Exception as exc:
            any_runtime_error = True
            fold_results.append({"fold_index": index, "support_indices": support_indices, "status": "runtime_error", "error": str(exc), "exact": False})
            continue
        output = result.get("output_grid")
        exact = _grids_equal(output, target)
        if result["status"] != "ok":
            any_no_unique = True
        elif not exact:
            any_mismatch = True
        if result.get("selected_hypothesis"):
            selected_hypotheses.append(result["selected_hypothesis"])
        fold_results.append(
            {
                "fold_index": index,
                "support_indices": support_indices,
                "status": result["status"],
                "exact": exact,
                "selected_hypothesis": result.get("selected_hypothesis"),
                "consistent_hypothesis_count": len(result.get("consistent_hypotheses", [])),
                "predicted_output_grid": _grid_to_list(output),
            }
        )
    strict = bool(fold_results) and all(row["status"] == "ok" and row["exact"] for row in fold_results)
    if any_raw_rejected:
        primary = RAW_REJECTED
        detail = TOPO_LOO_FAILED
    elif any_runtime_error:
        primary = RUNTIME_ERROR
        detail = TOPO_LOO_FAILED
    elif strict:
        primary = STRICT_VALIDATED
        detail = TOPO_STRICT
    elif any_no_unique:
        primary = FULL_SUPPORT_ONLY if full["fit"] else NO_UNIQUE
        detail = TOPO_LOO_FAILED
    elif any_mismatch:
        primary = STRICT_FAILED
        detail = TOPO_LOO_FAILED
    else:
        primary = FULL_SUPPORT_ONLY if full["fit"] else NO_UNIQUE
        detail = TOPO_LOO_FAILED

    test_candidates: list[list[list[int]]] = []
    if strict and len(task.train_inputs) == topology_data.SUPPORT_PAIR_COUNT:
        records, passed = _fidelity_for_grids(list(task.train_inputs) + list(task.train_outputs) + list(task.test_inputs), expert_name="topology", phase="test_candidate")
        raw_records.extend(records)
        if passed:
            for test_input in task.test_inputs:
                result = topology_executor.execute_from_raw_grids(list(task.train_inputs), list(task.train_outputs), test_input)
                if result["status"] != "ok":
                    test_candidates = []
                    break
                test_candidates.append(_grid_to_list(result["output_grid"]))
    return {
        "primary_label": primary,
        "strict_label": STRICT_VALIDATED if strict else primary,
        "loo_detail_status": detail,
        "full_support_fit": bool(full["fit"]),
        "full_support_status": full["status"],
        "full_support_selected_hypothesis": full.get("selected_hypothesis"),
        "loo_fold_count_attempted": len(fold_results),
        "fold_results": fold_results,
        "raw_grid_fidelity_records": raw_records,
        "test_candidates": test_candidates,
        "test_candidate_emitted": bool(test_candidates) or (strict and len(task.test_inputs) == 0),
        "domain_reasons": domain_reasons,
        "raw_domain_flags": raw_flags,
        "topology_subfamily_if_any": _topology_subfamily(selected_hypotheses),
    }


def _candidate_agreement(v2_eval: dict[str, Any], topology_eval: dict[str, Any]) -> tuple[str, str | None]:
    v2_strict = v2_eval["strict_label"] == STRICT_VALIDATED
    topo_strict = topology_eval["strict_label"] == STRICT_VALIDATED
    if not (v2_strict and topo_strict):
        return "not_multiple_strict_experts", None
    if not v2_eval["test_candidate_emitted"] or not topology_eval["test_candidate_emitted"]:
        return "not_agreement_eligible_missing_test_candidate", "one_or_more_strict_experts_could_not_emit_test_candidates_under_frozen_contract"
    if v2_eval["test_candidates"] == topology_eval["test_candidates"]:
        return "identical_test_candidates", None
    return "MULTI_EXPERT_CONFLICT", "strict_experts_emitted_different_test_candidates"


def _record_for_task(task: ArcTaskOpen) -> dict[str, Any]:
    identity = _identity_evaluation(task)
    v2_eval = _v2_evaluation(task)
    topology_eval = _topology_evaluation(task)
    agreement, conflict_reason = _candidate_agreement(v2_eval, topology_eval)
    return {
        "task_id": task.task_id,
        "source_relative_path": task.source_relative_path,
        "task_file_sha256": task.task_file_sha256,
        "train_pair_count": len(task.train_inputs),
        "test_input_count": len(task.test_inputs),
        "grid_shape_summary": grid_shape_summary(task),
        "palette_summary": palette_summary(task),
        "identity_primary_label": identity["primary_label"],
        "v2_primary_label": v2_eval["primary_label"],
        "topology_primary_label": topology_eval["primary_label"],
        "topology_loo_detail_status": topology_eval["loo_detail_status"],
        "topology_subfamily_if_any": topology_eval["topology_subfamily_if_any"],
        "full_support_fit_identity": identity["full_support_fit"],
        "full_support_fit_v2": v2_eval["full_support_fit"],
        "full_support_fit_topology": topology_eval["full_support_fit"],
        "full_support_status_v2": v2_eval["full_support_status"],
        "full_support_status_topology": topology_eval["full_support_status"],
        "loo_fold_count_attempted_identity": identity["loo_fold_count_attempted"],
        "loo_fold_count_attempted_v2": v2_eval["loo_fold_count_attempted"],
        "loo_fold_count_attempted_topology": topology_eval["loo_fold_count_attempted"],
        "loo_fold_results_identity": identity["fold_results"],
        "loo_fold_results_v2": v2_eval["fold_results"],
        "loo_fold_results_topology": topology_eval["fold_results"],
        "strict_label_identity": identity["strict_label"],
        "strict_label_v2": v2_eval["strict_label"],
        "strict_label_topology": topology_eval["strict_label"],
        "v2_raw_grid_fidelity_records": v2_eval["raw_grid_fidelity_records"],
        "topology_raw_grid_fidelity_records": topology_eval["raw_grid_fidelity_records"],
        "cross_expert_candidate_agreement": agreement,
        "cross_expert_conflict_reason_if_any": conflict_reason,
        "test_candidate_emitted_identity": identity["test_candidate_emitted"],
        "test_candidate_emitted_v2": v2_eval["test_candidate_emitted"],
        "test_candidate_emitted_topology": topology_eval["test_candidate_emitted"],
        "test_candidates_identity": identity["test_candidates"],
        "test_candidates_v2": v2_eval["test_candidates"],
        "test_candidates_topology": topology_eval["test_candidates"],
        "sealed_test_scoring_available": task.solution_relative_path is not None,
        "solution_relative_path_if_available": task.solution_relative_path,
        "v2_domain_reasons": v2_eval["domain_reasons"],
        "topology_domain_reasons": topology_eval["domain_reasons"],
        "topology_raw_domain_flags": topology_eval["raw_domain_flags"],
    }


def qualified_metric(selection: CorpusSelection, numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": denominator,
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _strict(record: dict[str, Any], expert: str) -> bool:
    return record[f"strict_label_{expert}"] == STRICT_VALIDATED


def _full(record: dict[str, Any], expert: str) -> bool:
    return bool(record[f"full_support_fit_{expert}"])


def compute_coverage_profile(selection: CorpusSelection, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(ledger)
    identity_strict = sum(_strict(row, "identity") for row in ledger)
    v2_strict = sum(_strict(row, "v2") for row in ledger)
    topology_strict = sum(_strict(row, "topology") for row in ledger)
    topology_seed = sum(_strict(row, "topology") and row["topology_subfamily_if_any"] == "topology_seed_only" for row in ledger)
    topology_border = sum(_strict(row, "topology") and row["topology_subfamily_if_any"] == "topology_border_or_enclosed" for row in ledger)
    oracle_union = sum(_strict(row, "v2") or _strict(row, "topology") for row in ledger)
    unrouted = 0
    conflicts = 0
    for row in ledger:
        v2_ok = _strict(row, "v2")
        topo_ok = _strict(row, "topology")
        if v2_ok + topo_ok == 1:
            unrouted += 1
        elif v2_ok and topo_ok:
            if row["cross_expert_candidate_agreement"] == "identical_test_candidates":
                unrouted += 1
            elif row["cross_expert_candidate_agreement"] == "MULTI_EXPERT_CONFLICT":
                conflicts += 1
    profile_rows = {
        "K = 0: identity baseline only": {
            "full_support_fit_rate": qualified_metric(selection, sum(_full(row, "identity") for row in ledger), n),
            "strict_LOO_validated_rate": qualified_metric(selection, identity_strict, n),
        },
        "K = 1a: V2 only": {
            "full_support_fit_rate": qualified_metric(selection, sum(_full(row, "v2") for row in ledger), n),
            "strict_LOO_validated_rate": qualified_metric(selection, v2_strict, n),
        },
        "K = 1b: topology only": {
            "full_support_fit_rate": qualified_metric(selection, sum(_full(row, "topology") for row in ledger), n),
            "strict_LOO_validated_rate": qualified_metric(selection, topology_strict, n),
        },
        "K = 2: V2 union topology": {
            "full_support_fit_rate": qualified_metric(selection, sum(_full(row, "v2") or _full(row, "topology") for row in ledger), n),
            "strict_LOO_validated_rate": qualified_metric(selection, oracle_union, n),
            "oracle_union_retrospective_rate": qualified_metric(selection, oracle_union, n),
            "unrouted_agreement_union_rate": qualified_metric(selection, unrouted, n),
        },
    }
    return {
        "audit_version": AUDIT_VERSION,
        "profile_name": "two-expert frozen-library coverage profile",
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": n,
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "C_identity": qualified_metric(selection, identity_strict, n),
        "C_V2": qualified_metric(selection, v2_strict, n),
        "C_topology_total": qualified_metric(selection, topology_strict, n),
        "C_topology_border_or_enclosed": qualified_metric(selection, topology_border, n),
        "C_topology_seed_only": qualified_metric(selection, topology_seed, n),
        "C_oracle_union_retrospective": qualified_metric(selection, oracle_union, n),
        "C_unrouted_agreement_union": qualified_metric(selection, unrouted, n),
        "cross_expert_conflict_count": conflicts,
        "profile": profile_rows,
        "ORACLE_UNION_RETROSPECTIVE": "analysis_only_not_deployable_routing_evidence",
    }


def _stage(stage: str, numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "stage": stage,
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
        "absolute_task_count": numerator,
    }


def compute_domain_attrition(selection: CorpusSelection, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(ledger)
    identity_applicable = n
    v2_raw = sum(row["v2_primary_label"] not in {OUT_GRID, OUT_PALETTE, RAW_REJECTED} for row in ledger)
    topo_raw_grid = sum(row["topology_raw_domain_flags"].get("grid_size_ok", False) for row in ledger)
    topo_raw_palette = sum(row["topology_raw_domain_flags"].get("grid_size_ok", False) and row["topology_raw_domain_flags"].get("palette_ok", False) for row in ledger)
    topo_raw = sum(row["topology_primary_label"] not in {OUT_GRID, OUT_PALETTE, RAW_REJECTED, NOT_APPLICABLE} for row in ledger)
    full_attempt = sum(row["full_support_status_v2"] not in {OUT_GRID, OUT_PALETTE} or row["full_support_status_topology"] not in {"invalid_support_count", OUT_GRID, OUT_PALETTE} for row in ledger)
    full_fit = sum(row["full_support_fit_v2"] or row["full_support_fit_topology"] or row["full_support_fit_identity"] for row in ledger)
    loo_feasible = sum(row["strict_label_v2"] != OUT_SUPPORT or row["strict_label_topology"] != OUT_SUPPORT for row in ledger)
    loo_attempt = sum(row["loo_fold_count_attempted_v2"] > 0 or row["loo_fold_count_attempted_topology"] > 0 or row["loo_fold_count_attempted_identity"] > 0 for row in ledger)
    loo_valid = sum(row["strict_label_v2"] == STRICT_VALIDATED or row["strict_label_topology"] == STRICT_VALIDATED or row["strict_label_identity"] == STRICT_VALIDATED for row in ledger)
    agreement_eligible = sum(row["strict_label_v2"] == STRICT_VALIDATED and row["strict_label_topology"] == STRICT_VALIDATED for row in ledger)
    unrouted = compute_coverage_profile(selection, ledger)["C_unrouted_agreement_union"]["numerator"]

    topology_full_support_eligible = sum(row["train_pair_count"] == topology_data.SUPPORT_PAIR_COUNT and row["topology_raw_domain_flags"].get("grid_size_ok", False) and row["topology_raw_domain_flags"].get("palette_ok", False) for row in ledger)
    topology_full_fit = sum(row["full_support_fit_topology"] for row in ledger)
    topology_loo_feasible = sum(row["topology_raw_domain_flags"].get("grid_size_ok", False) and row["topology_raw_domain_flags"].get("palette_ok", False) and row["train_pair_count"] - 1 == topology_data.SUPPORT_PAIR_COUNT for row in ledger)
    topology_loo_attempted = sum(row["loo_fold_count_attempted_topology"] > 0 for row in ledger)
    topology_loo_failed_after_execution = sum(row["topology_loo_detail_status"] == TOPO_LOO_FAILED for row in ledger)
    topology_strict = sum(row["strict_label_topology"] == STRICT_VALIDATED for row in ledger)

    reason_counts = {
        "grid_size": sum(row["v2_primary_label"] == OUT_GRID or row["topology_primary_label"] == OUT_GRID for row in ledger),
        "palette_range": sum(row["v2_primary_label"] == OUT_PALETTE or row["topology_primary_label"] == OUT_PALETTE for row in ledger),
        "support_pair_count": sum(row["topology_primary_label"] == OUT_SUPPORT for row in ledger),
        "query_count": sum(row["test_input_count"] != 1 for row in ledger),
        "raw_interface_rejection": sum(row["v2_primary_label"] == RAW_REJECTED or row["topology_primary_label"] == RAW_REJECTED for row in ledger),
        "runtime_interface_rejection": sum(row["v2_primary_label"] == RUNTIME_ERROR or row["topology_primary_label"] == RUNTIME_ERROR for row in ledger),
        "no_unique_topology_hypothesis": sum(row["topology_primary_label"] == NO_UNIQUE for row in ledger),
        "strict_LOO_mismatch": sum(row["v2_primary_label"] == STRICT_FAILED or row["topology_primary_label"] == STRICT_FAILED for row in ledger),
        "cross_expert_conflict": sum(row["cross_expert_candidate_agreement"] == "MULTI_EXPERT_CONFLICT" for row in ledger),
    }

    return {
        "audit_version": AUDIT_VERSION,
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": n,
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "overall_funnel": [
            _stage("all tasks", n, n),
            _stage("identity applicable", identity_applicable, n),
            _stage("V2 raw-domain applicable", v2_raw, n),
            _stage("topology raw-domain applicable", topo_raw, n),
            _stage("full-support reproduction attempted", full_attempt, n),
            _stage("full-support fit", full_fit, n),
            _stage("strict LOO feasible", loo_feasible, n),
            _stage("strict LOO attempted", loo_attempt, n),
            _stage("strict LOO validated", loo_valid, n),
            _stage("cross-expert agreement eligible", agreement_eligible, n),
            _stage("unrouted agreement union", unrouted, n),
        ],
        "attrition_breakdown": reason_counts,
        "mandatory_topology_funnel": [
            _stage("all corpus tasks", n, n),
            _stage("topology raw grid-size eligible", topo_raw_grid, n),
            _stage("topology raw palette eligible", topo_raw_palette, topo_raw_grid),
            _stage("topology raw support-count eligible for full support", topology_full_support_eligible, topo_raw_palette),
            _stage("topology full-support fit", topology_full_fit, topology_full_support_eligible),
            _stage("topology LOO feasible under frozen support contract", topology_loo_feasible, topo_raw_palette),
            _stage("topology LOO attempted", topology_loo_attempted, topology_loo_feasible),
            _stage("topology LOO failed after execution", topology_loo_failed_after_execution, topology_loo_attempted),
            _stage("topology strict LOO validated", topology_strict, topology_loo_attempted),
        ],
        "coverage_lost_due_to_frozen_grid_palette_support_count_contract": {
            "grid_or_palette_ineligible": n - topo_raw_palette,
            "support_count_ineligible_for_topology_LOO": topo_raw_palette - topology_loo_feasible,
        },
        "coverage_lost_after_topology_execution_wrong_hidden_output": topology_loo_failed_after_execution,
    }


def library_contract_report(selection: CorpusSelection) -> dict[str, Any]:
    closure_path = PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/closure_domain.json"
    shortcut_path = PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/shortcut_catalog.json"
    closure_payload = _load_json(closure_path) if closure_path.exists() else {}
    shortcut_payload = _load_json(shortcut_path) if shortcut_path.exists() else {}
    return {
        "audit_version": AUDIT_VERSION,
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": None,
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "frozen_library": ["identity_baseline", "V2_transport_recolor_executor", "R5A-2C-R2_compiled_topology_executor"],
        "identity_contract": {"runtime_inputs": ["query_input_grid"], "output": "input"},
        "v2_contract": {
            "accepted_grid_sizes": ["7x7"],
            "accepted_palette_range": [0, v2_oracles.COLOR_COUNT - 1],
            "accepted_support_pair_count_behavior": "variable raw support-pair count for exact support-consistency over the frozen SEALED_CLASSES operation set; no support duplication, deletion, or fabrication",
            "accepted_query_count_behavior": "one raw query grid per invocation; multiple ARC test inputs are invoked independently",
            "raw_input_fields": ["support_input_grids", "support_output_grids", "query_input_grid"],
            "runtime_error_behavior": "operation collision or non-unique support-consistent sequence returns no executable output",
            "frozen_operation_classes": list(v2_oracles.SEALED_CLASSES),
            "checkpoint_read": False,
            "neural_model_constructed": False,
        },
        "topology_contract": {
            "grid_size": f"{topology_data.GRID_SIZE}x{topology_data.GRID_SIZE}",
            "palette": list(topology_data.PALETTE),
            "support_pair_count_behavior": f"exactly {topology_data.SUPPORT_PAIR_COUNT} support pairs",
            "query_count_behavior": f"exactly {topology_data.QUERY_COUNT} query grid per invocation",
            "hypothesis_catalog_hash": closure_payload.get("closure_domain_sha256"),
            "closure_domain_hash": closure_payload.get("closure_domain_sha256"),
            "shortcut_catalog_hash": shortcut_payload.get("shortcut_catalog_sha256"),
            "runtime_error_behavior": "invalid support count, ambiguous support consistency, or no consistent hypothesis returns no executable output",
            "raw_input_fields": topology_executor.executor_runtime_contract()["runtime_inputs"],
        },
        "topology_seed_caveat": {
            "border_reachable_recolor": "reported in topology_border_or_enclosed",
            "enclosed_region_recolor": "reported in topology_border_or_enclosed",
            "seed_reachable_recolor": "reported separately as topology_seed_only and not presented as multi-step reachability coverage evidence",
        },
    }


def _render_grid(grid: list[list[int]] | Grid | None) -> str:
    if grid is None:
        return "<none>"
    rows = grid if isinstance(grid, list) else _grid_to_list(grid)
    return "\n".join("".join(str(value) for value in row) for row in rows)


def strict_examples_markdown(selection: CorpusSelection, ledger: list[dict[str, Any]]) -> str:
    lines = [
        "# R5A-2D-R1 strict coverage examples",
        "",
        f"corpus_name: {selection.corpus_name}",
        f"task_count: {len(ledger)}",
        f"corpus_manifest_sha256: {selection.corpus_manifest_sha256}",
        "",
    ]
    strict_rows = [row for row in ledger if row["strict_label_identity"] == STRICT_VALIDATED or row["strict_label_v2"] == STRICT_VALIDATED or row["strict_label_topology"] == STRICT_VALIDATED]
    if not strict_rows:
        lines.append("No task was counted as strict coverage by identity, V2, or topology.")
        return "\n".join(lines) + "\n"
    for row in strict_rows:
        experts = [name for name in ("identity", "v2", "topology") if row[f"strict_label_{name}"] == STRICT_VALIDATED]
        lines.extend(
            [
                f"## {row['task_id']}",
                "",
                f"source: {row['source_relative_path']}",
                f"strict_experts: {', '.join(experts)}",
                f"topology_subfamily_if_any: {row['topology_subfamily_if_any']}",
                f"cross_expert_agreement_required: {row['cross_expert_candidate_agreement']}",
                "",
                "Why applicable: raw interface fidelity passed for accepted expert invocations; all listed LOO folds passed exactly.",
                "",
                "First train input:",
                "```text",
                _render_grid(row["loo_fold_results_identity"][0]["predicted_output_grid"] if row["loo_fold_results_identity"] else None),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_sealed_outputs_for_task(task: ArcTaskOpen) -> list[Grid] | None:
    if not task.solution_relative_path:
        return None
    raw_path = Path(task.solution_relative_path)
    solution_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    if not solution_path.exists():
        return None
    payload = _load_json(solution_path)
    raw_task_id = task.task_id.split(":", 1)[1] if ":" in task.task_id else task.task_id
    if isinstance(payload, dict) and raw_task_id in payload:
        outputs = payload[raw_task_id]
        if isinstance(outputs, list):
            return [freeze_grid(grid) for grid in outputs]
    return None


def optional_sealed_scoring(selection: CorpusSelection, tasks: list[ArcTaskOpen], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    rows = []
    exact_counts = defaultdict(lambda: {"scored": 0, "exact": 0})
    for record in ledger:
        task = task_by_id[record["task_id"]]
        sealed = _load_sealed_outputs_for_task(task)
        if sealed is None:
            rows.append({"task_id": record["task_id"], "sealed_test_scoring_available": False})
            continue
        row: dict[str, Any] = {"task_id": record["task_id"], "sealed_test_scoring_available": True, "expert_scores": {}}
        for expert in ("identity", "v2", "topology"):
            candidates = record[f"test_candidates_{expert}"]
            if not candidates:
                row["expert_scores"][expert] = {"candidate_emitted": False, "scored": False}
                continue
            candidate_grids = [freeze_grid(candidate) for candidate in candidates]
            exact = len(candidate_grids) == len(sealed) and all(candidate == target for candidate, target in zip(candidate_grids, sealed))
            exact_counts[expert]["scored"] += 1
            exact_counts[expert]["exact"] += int(exact)
            row["expert_scores"][expert] = {"candidate_emitted": True, "scored": True, "exact": exact}
        rows.append(row)
    metrics = {
        expert: {
            "scored_task_count": counts["scored"],
            "exact_task_count": counts["exact"],
            "exact_rate": counts["exact"] / counts["scored"] if counts["scored"] else None,
        }
        for expert, counts in sorted(exact_counts.items())
    }
    return {
        "audit_version": AUDIT_VERSION,
        "metric_label": "SEALED_POST_HOC_TEST_SCORING",
        "post_hoc_only": True,
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": len(ledger),
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "scoring_available_task_count": sum(1 for row in rows if row.get("sealed_test_scoring_available")),
        "metrics_by_expert": metrics,
        "rows": rows,
        "sealed_outputs_used_to_change_labels_or_candidates": False,
    }


def _pre_score_commitment(selection: CorpusSelection, artifact_root: Path) -> dict[str, Any]:
    files = [
        "corpus_manifest_precommit.json",
        "library_contract_report.json",
        "task_ledger.json",
        "coverage_profile.json",
        "domain_attrition_report.json",
        "strict_coverage_examples.md",
    ]
    rows = []
    for name in files:
        path = artifact_root / name
        rows.append({"file": name, "sha256": sha256_file(path), "byte_size": path.stat().st_size})
    return {
        "audit_version": AUDIT_VERSION,
        "corpus_name": selection.corpus_name,
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "commitment_phase": "before_optional_sealed_test_scoring",
        "files": rows,
    }


def build_task_ledger(selection: CorpusSelection, tasks: list[ArcTaskOpen]) -> list[dict[str, Any]]:
    ledger = []
    for task in tasks:
        record = _record_for_task(task)
        record["corpus_name"] = selection.corpus_name
        record["corpus_manifest_sha256"] = selection.corpus_manifest_sha256
        ledger.append(record)
    return ledger


def _parameter_free_integrity() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_coverage_audit.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_corpus_resolver.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_raw_grid_fidelity.py",
        PROJECT_ROOT / "vrm/twin_tasks/run_r5a2d_r1_frozen_library_coverage.py",
    ]
    forbidden_terms = ("nn.Parameter", "torch.optim", ".backward(", "optimizer.step(", "torch.load", "torch.save")
    stripped_parts = []
    for path in paths:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        stripped_parts.extend(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string for token in tokens)
    stripped = " ".join(stripped_parts)
    found = [term for term in forbidden_terms if term in stripped]
    return {
        "no_parameter_optimizer_backward_or_checkpoint_io_introduced": not found,
        "forbidden_code_terms_found": found,
        "scan_excludes_strings_and_comments": True,
    }


def build_final_report(
    *,
    selection: CorpusSelection,
    ledger: list[dict[str, Any]],
    contract: dict[str, Any],
    profile: dict[str, Any],
    attrition: dict[str, Any],
    sealed: dict[str, Any],
    runtimes: dict[str, float],
    protected_before: dict[str, str | None],
    protected_after: dict[str, str | None],
    verdict: str,
    run_index: int,
    two_run_stable_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_failures = []
    for row in ledger:
        for expert in ("v2", "topology"):
            for evidence in row[f"{expert}_raw_grid_fidelity_records"]:
                if not evidence["passed"]:
                    raw_failures.append({"task_id": row["task_id"], "expert": expert, "evidence": evidence})
    report = {
        "audit_version": AUDIT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_index": run_index,
        "final_verdict": verdict,
        "mandatory_scope_boundary": MANDATORY_SCOPE_BOUNDARY,
        "mandatory_low_or_zero_boundary": MANDATORY_LOW_ZERO_BOUNDARY,
        "corpus_name": selection.corpus_name,
        "corpus_root_relative_or_configured_path": selection.corpus_root_relative_or_configured_path,
        "task_count": len(ledger),
        "corpus_manifest_sha256": selection.corpus_manifest_sha256,
        "library_contract_summary": contract,
        "coverage_profile": profile,
        "domain_attrition": attrition,
        "optional_sealed_post_hoc_scoring": {
            "metric_label": sealed["metric_label"],
            "scoring_available_task_count": sealed["scoring_available_task_count"],
            "metrics_by_expert": sealed["metrics_by_expert"],
        },
        "observed_runtimes_seconds": runtimes,
        "protected_hashes_unchanged": protected_before == protected_after,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "parameter_free_integrity": _parameter_free_integrity(),
        "raw_grid_fidelity_failure_count": len(raw_failures),
        "raw_grid_fidelity_failure_examples": raw_failures[:5],
        "cross_expert_conflict_examples": [
            {
                "task_id": row["task_id"],
                "source_relative_path": row["source_relative_path"],
                "reason": row["cross_expert_conflict_reason_if_any"],
            }
            for row in ledger
            if row["cross_expert_candidate_agreement"] == "MULTI_EXPERT_CONFLICT"
        ][:5],
        "full_support_reproduction_results_exploratory_only": {
            "identity": profile["profile"]["K = 0: identity baseline only"]["full_support_fit_rate"],
            "V2": profile["profile"]["K = 1a: V2 only"]["full_support_fit_rate"],
            "topology": profile["profile"]["K = 1b: topology only"]["full_support_fit_rate"],
        },
        "strict_LOO_coverage": {
            "identity": profile["C_identity"],
            "V2": profile["C_V2"],
            "topology_border_or_enclosed": profile["C_topology_border_or_enclosed"],
            "topology_seed_only": profile["C_topology_seed_only"],
            "topology_total": profile["C_topology_total"],
            "oracle_retrospective_union": profile["C_oracle_union_retrospective"],
            "unrouted_agreement_union": profile["C_unrouted_agreement_union"],
        },
        "stable_two_run_artifact_hashes": two_run_stable_hashes or {},
        "narrowest_justified_next_action": "Inspect the strict-LOO attrition rows for the first non-contract-mismatch failures without changing the frozen two-specialist library.",
    }
    report["report_body_sha256_excluding_unstable_fields"] = stable_hash(strip_unstable_fields(report))
    return report


def write_audit_artifacts(
    *,
    selection: CorpusSelection,
    tasks: list[ArcTaskOpen],
    ledger: list[dict[str, Any]],
    runtimes: dict[str, float],
    protected_before: dict[str, str | None],
    protected_after: dict[str, str | None],
    run_index: int,
    verdict: str,
    artifact_root: Path = ARTIFACT_ROOT,
    two_run_stable_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = library_contract_report(selection)
    contract["task_count"] = len(ledger)
    profile = compute_coverage_profile(selection, ledger)
    attrition = compute_domain_attrition(selection, ledger)
    write_json(artifact_root / "library_contract_report.json", contract)
    write_json(artifact_root / "task_ledger.json", ledger)
    write_json(artifact_root / "coverage_profile.json", profile)
    write_json(artifact_root / "domain_attrition_report.json", attrition)
    write_text(artifact_root / "strict_coverage_examples.md", strict_examples_markdown(selection, ledger))
    write_json(artifact_root / "pre_score_commitment.json", _pre_score_commitment(selection, artifact_root))
    sealed = optional_sealed_scoring(selection, tasks, ledger)
    write_json(artifact_root / "optional_sealed_test_scoring.json", sealed)
    final = build_final_report(
        selection=selection,
        ledger=ledger,
        contract=contract,
        profile=profile,
        attrition=attrition,
        sealed=sealed,
        runtimes=runtimes,
        protected_before=protected_before,
        protected_after=protected_after,
        verdict=verdict,
        run_index=run_index,
        two_run_stable_hashes=two_run_stable_hashes,
    )
    write_json(artifact_root / "final_report.json", final)
    return {
        "library_contract_report": contract,
        "task_ledger": ledger,
        "coverage_profile": profile,
        "domain_attrition_report": attrition,
        "optional_sealed_test_scoring": sealed,
        "final_report": final,
    }


def stable_artifact_hashes(artifact_root: Path = ARTIFACT_ROOT) -> dict[str, str]:
    names = [
        "corpus_manifest_precommit.json",
        "library_contract_report.json",
        "task_ledger.json",
        "coverage_profile.json",
        "domain_attrition_report.json",
        "optional_sealed_test_scoring.json",
        "final_report.json",
    ]
    rows = {}
    for name in names:
        path = artifact_root / name
        rows[name] = stable_json_hash_file(path)
    return rows


def _evaluate_once(*, artifact_root: Path, run_index: int) -> tuple[CorpusSelection, list[ArcTaskOpen], list[dict[str, Any]], dict[str, float]]:
    total_start = time.perf_counter()
    resolution_start = time.perf_counter()
    selection = resolve_and_precommit_corpus(project_root=PROJECT_ROOT, artifact_root=artifact_root)
    corpus_resolution_runtime_seconds = time.perf_counter() - resolution_start
    manifest_start = time.perf_counter()
    manifest_runtime_seconds = time.perf_counter() - manifest_start
    task_load_start = time.perf_counter()
    tasks = load_open_arc_tasks(selection)
    _ = time.perf_counter() - task_load_start

    v2_time = 0.0
    topology_time = 0.0
    ledger = []
    for task in tasks:
        start = time.perf_counter()
        identity = _identity_evaluation(task)
        v2_eval = _v2_evaluation(task)
        v2_time += time.perf_counter() - start
        start = time.perf_counter()
        topology_eval = _topology_evaluation(task)
        topology_time += time.perf_counter() - start
        agreement, conflict_reason = _candidate_agreement(v2_eval, topology_eval)
        record = {
            "task_id": task.task_id,
            "source_relative_path": task.source_relative_path,
            "task_file_sha256": task.task_file_sha256,
            "corpus_name": selection.corpus_name,
            "corpus_manifest_sha256": selection.corpus_manifest_sha256,
            "train_pair_count": len(task.train_inputs),
            "test_input_count": len(task.test_inputs),
            "grid_shape_summary": grid_shape_summary(task),
            "palette_summary": palette_summary(task),
            "identity_primary_label": identity["primary_label"],
            "v2_primary_label": v2_eval["primary_label"],
            "topology_primary_label": topology_eval["primary_label"],
            "topology_loo_detail_status": topology_eval["loo_detail_status"],
            "topology_subfamily_if_any": topology_eval["topology_subfamily_if_any"],
            "full_support_fit_identity": identity["full_support_fit"],
            "full_support_fit_v2": v2_eval["full_support_fit"],
            "full_support_fit_topology": topology_eval["full_support_fit"],
            "full_support_status_v2": v2_eval["full_support_status"],
            "full_support_status_topology": topology_eval["full_support_status"],
            "loo_fold_count_attempted_identity": identity["loo_fold_count_attempted"],
            "loo_fold_count_attempted_v2": v2_eval["loo_fold_count_attempted"],
            "loo_fold_count_attempted_topology": topology_eval["loo_fold_count_attempted"],
            "loo_fold_results_identity": identity["fold_results"],
            "loo_fold_results_v2": v2_eval["fold_results"],
            "loo_fold_results_topology": topology_eval["fold_results"],
            "strict_label_identity": identity["strict_label"],
            "strict_label_v2": v2_eval["strict_label"],
            "strict_label_topology": topology_eval["strict_label"],
            "v2_raw_grid_fidelity_records": v2_eval["raw_grid_fidelity_records"],
            "topology_raw_grid_fidelity_records": topology_eval["raw_grid_fidelity_records"],
            "cross_expert_candidate_agreement": agreement,
            "cross_expert_conflict_reason_if_any": conflict_reason,
            "test_candidate_emitted_identity": identity["test_candidate_emitted"],
            "test_candidate_emitted_v2": v2_eval["test_candidate_emitted"],
            "test_candidate_emitted_topology": topology_eval["test_candidate_emitted"],
            "test_candidates_identity": identity["test_candidates"],
            "test_candidates_v2": v2_eval["test_candidates"],
            "test_candidates_topology": topology_eval["test_candidates"],
            "sealed_test_scoring_available": task.solution_relative_path is not None,
            "solution_relative_path_if_available": task.solution_relative_path,
            "v2_domain_reasons": v2_eval["domain_reasons"],
            "topology_domain_reasons": topology_eval["domain_reasons"],
            "topology_raw_domain_flags": topology_eval["raw_domain_flags"],
            "run_index": run_index,
        }
        ledger.append(record)
    runtimes = {
        "corpus_resolution_runtime_seconds": corpus_resolution_runtime_seconds,
        "manifest_runtime_seconds": manifest_runtime_seconds,
        "V2_evaluation_runtime_seconds": v2_time,
        "topology_evaluation_runtime_seconds": topology_time,
        "total_audit_runtime_seconds": time.perf_counter() - total_start,
    }
    return selection, tasks, ledger, runtimes


def run_complete_audit_twice(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    protected_before = protected_hashes()
    try:
        selection1, tasks1, ledger1, runtimes1 = _evaluate_once(artifact_root=artifact_root, run_index=1)
    except NoLocalArcCorpusError:
        artifact_root.mkdir(parents=True, exist_ok=True)
        final = {
            "audit_version": AUDIT_VERSION,
            "final_verdict": NO_CORPUS_VERDICT,
            "mandatory_scope_boundary": MANDATORY_SCOPE_BOUNDARY,
            "mandatory_low_or_zero_boundary": MANDATORY_LOW_ZERO_BOUNDARY,
        }
        write_json(artifact_root / "final_report.json", final)
        return {"final_verdict": NO_CORPUS_VERDICT, "final_report": final}
    protected_mid = protected_hashes()
    payloads1 = write_audit_artifacts(
        selection=selection1,
        tasks=tasks1,
        ledger=ledger1,
        runtimes=runtimes1,
        protected_before=protected_before,
        protected_after=protected_mid,
        run_index=1,
        verdict=MEASURED_VERDICT,
        artifact_root=artifact_root,
    )
    hashes1 = stable_artifact_hashes(artifact_root)

    second_complete_start = time.perf_counter()
    selection2, tasks2, ledger2, runtimes2 = _evaluate_once(artifact_root=artifact_root, run_index=2)
    protected_after = protected_hashes()
    payloads2 = write_audit_artifacts(
        selection=selection2,
        tasks=tasks2,
        ledger=ledger2,
        runtimes=runtimes2,
        protected_before=protected_before,
        protected_after=protected_after,
        run_index=2,
        verdict=MEASURED_VERDICT,
        artifact_root=artifact_root,
    )
    hashes2 = stable_artifact_hashes(artifact_root)
    deterministic = hashes1 == hashes2
    integrity = protected_before == protected_after and payloads2["final_report"]["parameter_free_integrity"]["no_parameter_optimizer_backward_or_checkpoint_io_introduced"]
    verdict = MEASURED_VERDICT
    if not deterministic:
        verdict = NONDETERMINISTIC_VERDICT
    elif not integrity:
        verdict = INTEGRITY_FAILURE_VERDICT

    comparison = {
        "run1": hashes1,
        "run2": hashes2,
        "matching": deterministic,
        "differing_files": sorted(name for name in hashes1 if hashes1.get(name) != hashes2.get(name)),
    }
    final_payloads = write_audit_artifacts(
        selection=selection2,
        tasks=tasks2,
        ledger=ledger2,
        runtimes=runtimes2,
        protected_before=protected_before,
        protected_after=protected_after,
        run_index=2,
        verdict=verdict,
        artifact_root=artifact_root,
        two_run_stable_hashes=comparison,
    )
    runtimes2["total_audit_runtime_seconds"] = time.perf_counter() - second_complete_start
    final_payloads = write_audit_artifacts(
        selection=selection2,
        tasks=tasks2,
        ledger=ledger2,
        runtimes=runtimes2,
        protected_before=protected_before,
        protected_after=protected_after,
        run_index=2,
        verdict=verdict,
        artifact_root=artifact_root,
        two_run_stable_hashes=comparison,
    )
    return {
        "final_verdict": verdict,
        "selection": selection2,
        "tasks": tasks2,
        "ledger": ledger2,
        "observed_runtimes_seconds": runtimes2,
        "stable_two_run_artifact_hashes": comparison,
        "final_report": final_payloads["final_report"],
        "coverage_profile": final_payloads["coverage_profile"],
        "domain_attrition_report": final_payloads["domain_attrition_report"],
        "optional_sealed_test_scoring": final_payloads["optional_sealed_test_scoring"],
    }
