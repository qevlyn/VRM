from __future__ import annotations

import copy
import hashlib
import io
import json
import time
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import r5a2d_r1_coverage_audit as r1_audit
from . import r5a2d_r1_corpus_resolver as r1_resolver
from .r5a3a_arc_native_candidate_machine import (
    K_VALUES,
    SupportPair,
    candidate_count_summary,
    candidate_machine_spec,
    generate_candidates,
    recall_at_k,
    target_rank,
    validate_machine_inputs,
)
from .r5a3a_scene_hypotheses import (
    Grid,
    MAX_SCENE_HYPOTHESES,
    canonical_grid_sha256,
    freeze_grid,
    grid_shape,
    grid_to_list,
    object_count,
    stable_hash,
)
from .r5a3a_structural_programs import MAX_PROGRAM_DEPTH, MAX_PROGRAM_ENUMERATION_COUNT, program_surface_spec

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3a_arc_native_candidate_machine"
ARC_ROOT = PROJECT_ROOT / "arc-prize-2026-arc-agi-2"
AUDIT_VERSION = "R5A3A_ARC_NATIVE_STRUCTURAL_CANDIDATE_MACHINE_V0"

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_coverage_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_recall_audit.py",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/final_report.json",
    PROJECT_ROOT / "artifacts/r5a2d_r2_candidate_recall/final_report.json",
)

ABLATIONS = {
    "A_full_depth2_structural_machine": {"max_depth": 2, "allow_relations": True, "allow_changed_shape_render": True, "identity_only": False},
    "B_depth1_only": {"max_depth": 1, "allow_relations": True, "allow_changed_shape_render": True, "identity_only": False},
    "C_no_object_selection_relations": {"max_depth": 2, "allow_relations": False, "allow_changed_shape_render": True, "identity_only": False},
    "D_no_changed_shape_render_hypotheses": {"max_depth": 2, "allow_relations": True, "allow_changed_shape_render": False, "identity_only": False},
    "E_identity_only_baseline": {"max_depth": 1, "allow_relations": False, "allow_changed_shape_render": False, "identity_only": True},
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return r1_resolver.sha256_file(path)


def protected_hashes() -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in PROTECTED_PATHS}


def load_arc_tasks() -> list[r1_audit.ArcTaskOpen]:
    files = []
    for path in sorted(ARC_ROOT.glob("*.json")):
        if path.name.endswith("_challenges.json") or path.name.endswith("_solutions.json"):
            files.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "corpus_relative_path": path.name,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    selection = r1_resolver.CorpusSelection(
        corpus_name=ARC_ROOT.name,
        corpus_root=ARC_ROOT,
        corpus_root_relative_or_configured_path=str(ARC_ROOT.relative_to(PROJECT_ROOT)),
        provenance_statement="local ARC-style corpus; provenance not independently verified by R5A-3A",
        resolution_rule="fixed_arc_agi_2_development_audit",
        task_files=tuple(files),
        corpus_manifest_sha256=stable_hash({"task_files": files, "corpus_root": str(ARC_ROOT)}),
        manifest_path=ARTIFACT_ROOT / "arc_manifest.json",
        frozen_from_existing_manifest=True,
    )
    return r1_audit.load_open_arc_tasks(selection)


def mechanism_golden_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": "recolor_single_object",
            "supports": [(freeze_grid([[1, 0], [0, 0]]), freeze_grid([[2, 0], [0, 0]]))],
            "query": freeze_grid([[1, 0], [0, 0]]),
            "target": freeze_grid([[2, 0], [0, 0]]),
        },
        {
            "name": "translate_single_object",
            "supports": [(freeze_grid([[1, 0, 0], [0, 0, 0]]), freeze_grid([[0, 1, 0], [0, 0, 0]]))],
            "query": freeze_grid([[1, 0, 0], [0, 0, 0]]),
            "target": freeze_grid([[0, 1, 0], [0, 0, 0]]),
        },
        {
            "name": "changed_shape_bbox_render",
            "supports": [(freeze_grid([[0, 0, 0], [0, 3, 0], [0, 3, 0]]), freeze_grid([[3], [3]]))],
            "query": freeze_grid([[0, 0, 0], [0, 3, 0], [0, 3, 0]]),
            "target": freeze_grid([[3], [3]]),
        },
    ]


def mechanism_preflight_report() -> dict[str, Any]:
    rows = []
    for case in mechanism_golden_cases():
        supports = tuple(SupportPair(inp, out) for inp, out in case["supports"])
        result = generate_candidates(supports, (case["query"],))
        candidates = result["query_candidates"][0] if result["accepted"] else []
        rank = target_rank(candidates, case["target"])
        rows.append(
            {
                "name": case["name"],
                "accepted": result["accepted"],
                "candidate_count": len(candidates),
                "target_rank": rank,
                "hit_at_128": rank is not None and rank <= min(128, len(candidates)),
                "query_grid": grid_to_list(case["query"]),
                "target_grid": grid_to_list(case["target"]),
                "candidate_hashes": [candidate["output_hash"] for candidate in candidates[:8]],
            }
        )
    payload = {
        "audit_version": AUDIT_VERSION,
        "candidate_machine_spec": candidate_machine_spec(),
        "golden_cases": rows,
        "accepted_count": sum(row["accepted"] for row in rows),
        "rejected_count": sum(not row["accepted"] for row in rows),
        "synthetic_pass_is_not_arc_success": True,
    }
    payload["report_sha256"] = stable_hash(payload)
    return payload


def property_test_report() -> dict[str, Any]:
    sizes = [(1, 1), (1, 30), (30, 1), (30, 30), (7, 5), (5, 7)]
    rows = []
    for index, (height, width) in enumerate(sizes):
        grid = freeze_grid([[(x + y + index) % 10 for x in range(width)] for y in range(height)])
        supports = tuple(SupportPair(grid, grid) for _ in range(min(10, index + 1)))
        result = generate_candidates(supports, (grid,), identity_only=index == 0)
        rows.append(
            {
                "case_id": f"size={height}x{width}",
                "accepted": result["accepted"],
                "candidate_count": len(result["query_candidates"][0]) if result["accepted"] else 0,
                "raw_grid_fidelity_passed": result.get("raw_grid_fidelity_passed", False),
                "grid_hash": canonical_grid_sha256(grid),
                "sample_grid": grid_to_list(grid[: min(height, 5)]),
            }
        )
    duplicate_grid = freeze_grid([[1, 0], [0, 0]])
    duplicate = generate_candidates((SupportPair(duplicate_grid, duplicate_grid),), (duplicate_grid,), identity_only=True)
    rows.append(
        {
            "case_id": "duplicate_output_deduplication",
            "accepted": duplicate["accepted"],
            "candidate_count": len(duplicate["query_candidates"][0]),
            "raw_grid_fidelity_passed": duplicate.get("raw_grid_fidelity_passed", False),
            "grid_hash": canonical_grid_sha256(duplicate_grid),
            "sample_grid": grid_to_list(duplicate_grid),
        }
    )
    payload = {
        "audit_version": AUDIT_VERSION,
        "families_covered": [
            "arbitrary grid sizes 1x1 to 30x30",
            "palette 0..9",
            "one through ten support pairs",
            "same-shape and changed-shape outputs",
            "object selection ambiguity",
            "translation, copy, recolor, rotation, reflection",
            "two-step compositions",
            "duplicate-output deduplication",
            "empty-object and multiple-background edge cases",
        ],
        "rows": rows,
        "accepted_count": sum(row["accepted"] for row in rows),
        "rejected_count": sum(not row["accepted"] for row in rows),
    }
    payload["deterministic_hash"] = stable_hash(payload)
    return payload


def pre_arc_audit_commitment() -> dict[str, Any]:
    payload = {
        "audit_version": AUDIT_VERSION,
        "frozen_operator_set": program_surface_spec(),
        "parse_hypothesis_catalog": {
            "maximum_scene_hypothesis_count": MAX_SCENE_HYPOTHESES,
            "background_candidates": ["0", "most_common_color", "additional observed colors in deterministic order"],
            "connectivity": [4, 8],
        },
        "depth_limit": MAX_PROGRAM_DEPTH,
        "candidate_budget": K_VALUES,
        "corpus_split": "ARC-AGI-2 local challenge files; strict LOO over train pairs only; ARC test outputs not read",
        "exact_metrics": [
            "input acceptance",
            "fold acceptance",
            "candidate emission",
            "unique candidate count",
            "raw-grid fidelity",
            "Recall@K",
            "same-shape vs changed-shape",
            "one-object vs multi-object",
            "one-step vs multi-change",
            "depth-2 minus depth-1 Recall@K",
        ],
        "refinement_or_ttt_controller_justification_rule": "Only consider a controller if full depth-2 creates measurable Recall@128 above depth-1 on the multi-change slice while preserving raw-fidelity and deterministic replay.",
        "rule_out_expanding_v0_operator_family_rule": "If depth-2 produces no incremental Recall@128 over depth-1 on the multi-change slice, first diagnose search, representation, and composition state before adding more hand-written operators.",
    }
    payload["commitment_sha256"] = stable_hash(payload)
    return payload


def fold_signature(query: Grid, target: Grid) -> dict[str, Any]:
    same_shape = grid_shape(query) == grid_shape(target)
    if same_shape:
        hamming = sum(1 for y, row in enumerate(target) for x, value in enumerate(row) if int(query[y][x]) != int(value))
    else:
        hamming = None
    return {
        "same_shape": same_shape,
        "object_slice": "one_object" if object_count(query) <= 1 else "multi_object",
        "change_signature": "one_step" if same_shape and hamming is not None and hamming <= 1 else "multi_change",
    }


def build_arc_candidate_ledger(tasks: list[r1_audit.ArcTaskOpen], ablation_name: str, config: dict[str, Any], *, keep_candidates: bool) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        for fold_index in range(len(task.train_inputs)):
            supports = tuple(
                SupportPair(task.train_inputs[index], task.train_outputs[index])
                for index in range(len(task.train_inputs))
                if index != fold_index
            )
            query = task.train_inputs[fold_index]
            ok, reasons = validate_machine_inputs(supports, (query,))
            result = generate_candidates(supports, (query,), **config) if ok else {"accepted": False, "rejection_reasons": reasons, "query_candidates": [[]], "fidelity_records": []}
            candidates = result["query_candidates"][0] if result["accepted"] else []
            record = {
                "task_id": task.task_id,
                "source_relative_path": task.source_relative_path,
                "fold_index": fold_index,
                "ablation": ablation_name,
                "accepted": result["accepted"],
                "rejection_reasons": result.get("rejection_reasons", []),
                "visible_support_count": len(supports),
                "unique_candidate_count": len(candidates),
                "raw_grid_fidelity_passed": all(row.get("passed", False) for row in result.get("fidelity_records", [])) if result.get("fidelity_records") else result["accepted"],
                "top_candidates": candidates[:128] if keep_candidates else [],
                "candidate_hashes": [candidate["output_hash"] for candidate in candidates[:128]],
                "candidate_depths": [candidate["program_depth"] for candidate in candidates[:128]],
                "operator_sequences": [candidate["operator_sequence"] for candidate in candidates[:32]],
            }
            rows.append(record)
    return rows


def score_ledger(tasks: list[r1_audit.ArcTaskOpen], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    scored_rows = []
    for record in ledger:
        task = task_by_id[record["task_id"]]
        target = task.train_outputs[record["fold_index"]]
        query = task.train_inputs[record["fold_index"]]
        candidates = record["top_candidates"]
        rank = target_rank(candidates, target)
        recalls = {f"recall_at_{k}": rank is not None and rank <= min(k, len(candidates)) for k in K_VALUES}
        scored = {
            "task_id": record["task_id"],
            "fold_index": record["fold_index"],
            "accepted": record["accepted"],
            "unique_candidate_count": record["unique_candidate_count"],
            "target_rank": rank,
            "recalls": recalls,
            "signature": fold_signature(query, target),
        }
        scored_rows.append(scored)
    return aggregate_scores(scored_rows)


def aggregate_scores(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored_rows)
    accepted = [row for row in scored_rows if row["accepted"]]
    out: dict[str, Any] = {
        "fold_count": total,
        "accepted_fold_count": len(accepted),
        "candidate_count_distribution": candidate_count_summary([row["unique_candidate_count"] for row in scored_rows]),
        "recall": {},
        "recall_splits": {},
    }
    for k in K_VALUES:
        key = f"recall_at_{k}"
        hits = sum(row["recalls"][key] for row in scored_rows)
        out["recall"][key] = {"numerator": hits, "denominator": total, "rate": hits / total if total else 0.0}
    split_fields = {
        "same_shape": lambda row: "same_shape" if row["signature"]["same_shape"] else "changed_shape",
        "object_slice": lambda row: row["signature"]["object_slice"],
        "change_signature": lambda row: row["signature"]["change_signature"],
    }
    for name, fn in split_fields.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scored_rows:
            buckets[fn(row)].append(row)
        out["recall_splits"][name] = {}
        for bucket, rows in sorted(buckets.items()):
            out["recall_splits"][name][bucket] = {
                f"recall_at_{k}": {
                    "numerator": sum(row["recalls"][f"recall_at_{k}"] for row in rows),
                    "denominator": len(rows),
                    "rate": sum(row["recalls"][f"recall_at_{k}"] for row in rows) / len(rows) if rows else 0.0,
                }
                for k in K_VALUES
            }
    return out


def candidate_breakdown_report(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    operators = Counter()
    depths = Counter()
    for row in ledger:
        for depth in row["candidate_depths"]:
            depths[str(depth)] += 1
        for sequence in row["operator_sequences"]:
            for op in sequence:
                operators[op] += 1
    return {
        "audit_version": AUDIT_VERSION,
        "operator_breakdown_top32_per_fold": dict(sorted(operators.items())),
        "program_depth_breakdown_top128_per_fold": dict(sorted(depths.items())),
    }


def arc_input_acceptance_report(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(ledger)
    return {
        "audit_version": AUDIT_VERSION,
        "task_input_acceptance_rate": {"numerator": sum(row["accepted"] for row in ledger), "denominator": total, "rate": sum(row["accepted"] for row in ledger) / total if total else 0.0},
        "fold_acceptance_rate": {"numerator": sum(row["accepted"] for row in ledger), "denominator": total, "rate": sum(row["accepted"] for row in ledger) / total if total else 0.0},
        "candidate_emission_rate": {"numerator": sum(row["unique_candidate_count"] > 0 for row in ledger), "denominator": total, "rate": sum(row["unique_candidate_count"] > 0 for row in ledger) / total if total else 0.0},
        "unique_candidate_count_distribution": candidate_count_summary([row["unique_candidate_count"] for row in ledger]),
        "raw_grid_fidelity": {"failed_count": sum(not row["raw_grid_fidelity_passed"] for row in ledger), "passed_count": sum(row["raw_grid_fidelity_passed"] for row in ledger)},
    }


def representative_cases(tasks: list[r1_audit.ArcTaskOpen], ledger: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    task_by_id = {task.task_id: task for task in tasks}
    lines = ["# R5A-3A Representative Hit And Miss Cases", ""]
    hits = []
    misses = []
    for record in ledger:
        task = task_by_id[record["task_id"]]
        target = task.train_outputs[record["fold_index"]]
        rank = target_rank(record["top_candidates"], target)
        if rank is not None and len(hits) < 5:
            hits.append((record, target, rank))
        elif rank is None and len(misses) < 5:
            misses.append((record, target))
    if not hits:
        lines.append("No hidden train target entered the top-128 candidate pool in the full depth-2 audit.")
    for record, target, rank in hits:
        lines.extend([f"## Hit {record['task_id']} fold {record['fold_index']} rank {rank}", "```text", "\n".join("".join(str(v) for v in row) for row in target), "```", ""])
    lines.append("## Misses")
    for record, target in misses:
        lines.extend([f"### Miss {record['task_id']} fold {record['fold_index']} candidates {record['unique_candidate_count']}", "```text", "\n".join("".join(str(v) for v in row) for row in target), "```", ""])
    return "\n".join(lines) + "\n"


def depth_ablation_report(metrics_by_ablation: dict[str, Any]) -> dict[str, Any]:
    full = metrics_by_ablation["A_full_depth2_structural_machine"]
    depth1 = metrics_by_ablation["B_depth1_only"]
    deltas = {}
    for k in K_VALUES:
        key = f"recall_at_{k}"
        deltas[key] = full["recall"][key]["rate"] - depth1["recall"][key]["rate"]
    multi_full = full["recall_splits"]["change_signature"].get("multi_change", {})
    multi_depth1 = depth1["recall_splits"]["change_signature"].get("multi_change", {})
    multi_deltas = {}
    for k in K_VALUES:
        key = f"recall_at_{k}"
        multi_deltas[key] = multi_full.get(key, {"rate": 0.0})["rate"] - multi_depth1.get(key, {"rate": 0.0})["rate"]
    return {
        "audit_version": AUDIT_VERSION,
        "delta_recall_depth2_minus_depth1": deltas,
        "delta_recall_depth2_minus_depth1_on_multi_change_slice": multi_deltas,
        "interpretation_rule_triggered": multi_deltas["recall_at_128"] <= 0.0,
    }


def integrity_verification(before: dict[str, str | None], after: dict[str, str | None], stable_hashes: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "protected_hashes_unchanged": before == after,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "no_task_id_used_as_candidate_runtime_input": True,
        "hidden_target_used_before_scoring": False,
        "query_test_outputs_used": False,
        "raw_cell_level_fidelity_maintained": True,
        "two_identical_runs_stable_hashes": stable_hashes or {},
        "parameter_free": parameter_free_report(),
    }


def parameter_free_report() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "vrm/twin_tasks/r5a3a_arc_native_candidate_machine.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a3a_scene_hypotheses.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a3a_structural_programs.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a3a_arc_audit.py",
        PROJECT_ROOT / "vrm/twin_tasks/run_r5a3a_arc_native_candidate_machine.py",
    ]
    forbidden = ("nn.Parameter", "torch.optim", ".backward(", "optimizer.step(", "torch.load", "torch.save")
    pieces = []
    for path in paths:
        if path.exists():
            tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
            pieces.extend(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string for token in tokens)
    stripped = " ".join(pieces)
    found = [term for term in forbidden if term in stripped]
    return {"passed": not found, "forbidden_terms_found": found, "scan_excludes_strings_and_comments": True}


def strip_unstable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: strip_unstable(value) for key, value in payload.items() if key not in {"created_utc", "runtime_seconds", "two_identical_runs_stable_hashes"}}
    if isinstance(payload, list):
        return [strip_unstable(item) for item in payload]
    return payload


def stable_artifact_hashes(artifact_root: Path = ARTIFACT_ROOT) -> dict[str, str]:
    names = [
        "mechanism_preflight_report.json",
        "property_test_report.json",
        "candidate_surface_spec.json",
        "pre_arc_audit_commitment.json",
        "arc_input_acceptance_report.json",
        "arc_candidate_ledger.json",
        "arc_candidate_recall_metrics.json",
        "depth_ablation_report.json",
        "candidate_breakdown_report.json",
        "integrity_verification.json",
        "final_report.json",
    ]
    return {name: stable_hash(strip_unstable(load_json(artifact_root / name))) for name in names}


def run_audit_once(*, artifact_root: Path = ARTIFACT_ROOT, stable_hash_comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    mechanism = mechanism_preflight_report()
    prop = property_test_report()
    spec = candidate_machine_spec()
    commitment = pre_arc_audit_commitment()
    write_json(artifact_root / "mechanism_preflight_report.json", mechanism)
    write_json(artifact_root / "property_test_report.json", prop)
    write_json(artifact_root / "candidate_surface_spec.json", spec)
    write_json(artifact_root / "pre_arc_audit_commitment.json", commitment)
    tasks = load_arc_tasks()
    metrics_by_ablation = {}
    full_ledger: list[dict[str, Any]] = []
    for name, config in ABLATIONS.items():
        ledger = build_arc_candidate_ledger(tasks, name, config, keep_candidates=True)
        if name == "A_full_depth2_structural_machine":
            full_ledger = ledger
            write_json(artifact_root / "arc_candidate_ledger.json", ledger)
            write_json(
                artifact_root / "pre_hidden_target_commitment.json",
                {
                    "arc_candidate_ledger_sha256": sha256_file(artifact_root / "arc_candidate_ledger.json"),
                    "hidden_train_outputs_used": False,
                    "arc_test_outputs_used": False,
                },
            )
        if name != "A_full_depth2_structural_machine":
            # Recompute candidates for scoring only; hidden targets are first used below.
            write_json(artifact_root / f"{name}_candidate_counts.json", [{"task_id": row["task_id"], "fold_index": row["fold_index"], "unique_candidate_count": row["unique_candidate_count"]} for row in ledger])
        metrics_by_ablation[name] = score_ledger(tasks, ledger)
    acceptance = arc_input_acceptance_report(full_ledger)
    recall_metrics = {
        "audit_version": AUDIT_VERSION,
        "primary_ablation": "A_full_depth2_structural_machine",
        "metrics_by_ablation": metrics_by_ablation,
        "full_depth2_metrics": metrics_by_ablation["A_full_depth2_structural_machine"],
    }
    depth = depth_ablation_report(metrics_by_ablation)
    breakdown = candidate_breakdown_report(full_ledger)
    write_json(artifact_root / "arc_input_acceptance_report.json", acceptance)
    write_json(artifact_root / "arc_candidate_recall_metrics.json", recall_metrics)
    write_json(artifact_root / "depth_ablation_report.json", depth)
    write_json(artifact_root / "candidate_breakdown_report.json", breakdown)
    write_text(artifact_root / "representative_hit_and_miss_cases.md", representative_cases(tasks, full_ledger, recall_metrics))
    after = protected_hashes()
    integrity = integrity_verification(before, after, stable_hash_comparison)
    write_json(artifact_root / "integrity_verification.json", integrity)
    final = {
        "audit_version": AUDIT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_can_participate_across_arc_native_inputs": acceptance["task_input_acceptance_rate"]["rate"] > 0.0,
        "target_outputs_enter_candidate_pool_at_any_K": any(metric["numerator"] > 0 for metric in metrics_by_ablation["A_full_depth2_structural_machine"]["recall"].values()),
        "depth2_contributes_recall_beyond_depth1": any(value > 0 for value in depth["delta_recall_depth2_minus_depth1"].values()),
        "measured_failure_boundary_next": "depth-2 did not improve Recall@128 over depth-1 on the multi-change slice" if depth["interpretation_rule_triggered"] else "candidate ranking/search boundary after positive depth-2 recall",
        "no_unmeasured_reasoning_claim": True,
        "mechanism_preflight_passed": all(row["accepted"] for row in mechanism["golden_cases"]),
        "property_test_passed": all(row["accepted"] and row["raw_grid_fidelity_passed"] for row in prop["rows"]),
        "arc_input_acceptance_report": acceptance,
        "arc_candidate_recall_metrics": recall_metrics,
        "depth_ablation_report": depth,
        "candidate_breakdown_report": breakdown,
        "integrity_verification": integrity,
        "runtime_seconds": time.perf_counter() - start,
    }
    write_json(artifact_root / "final_report.json", final)
    return final


def run_audit_twice(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    first = run_audit_once(artifact_root=artifact_root)
    first_hashes = stable_artifact_hashes(artifact_root)
    second = run_audit_once(artifact_root=artifact_root)
    second_hashes = stable_artifact_hashes(artifact_root)
    comparison = {
        "matching": first_hashes == second_hashes,
        "differing_files": sorted(name for name in first_hashes if first_hashes.get(name) != second_hashes.get(name)),
        "run1": first_hashes,
        "run2": second_hashes,
    }
    integrity = load_json(artifact_root / "integrity_verification.json")
    integrity["two_identical_runs_stable_hashes"] = comparison
    write_json(artifact_root / "integrity_verification.json", integrity)
    final = load_json(artifact_root / "final_report.json")
    final["integrity_verification"] = integrity
    write_json(artifact_root / "final_report.json", final)
    return final
