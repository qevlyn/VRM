from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .r5a2d_r1_corpus_resolver import sha256_file
from .r5a3a_arc_native_candidate_machine import SupportPair
from .r5a3a_scene_hypotheses import Grid, canonical_grid_sha256, freeze_grid, stable_hash
from .r5a3b_p2a_balanced_relation_allocation import (
    MAX_FINAL_CANDIDATES_PER_FOLD,
    MAX_FINAL_PROGRAMS_PER_PARSE,
    MAX_PROGRAM_DEPTH,
    RELATION_ACTION_BUCKETS,
    TOP_K,
    allocation_spec,
    build_p2a_candidate_universe,
    candidate_trace_counts_json,
)
from .r5a3b_relation_programs import target_rank

PROJECT_ROOT = Path("/mnt/d/VRM")
ARC_ROOT = PROJECT_ROOT / "arc-prize-2026-arc-agi-2"
P0_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot"
P1_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_p1_exposure_audit"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_p2a_balanced_transport_allocation"
PYTHON = Path("/home/quyen/.venvs/vrm/bin/python")
AUDIT_VERSION = "R5A3B_P2A_BALANCED_RELATION_TRANSPORT_ALLOCATION_PILOT"

P2A_SOURCE_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_p2a_balanced_relation_allocation.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_p2a_transport_pilot.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_r5a3b_p2a_transport_pilot.py",
    PROJECT_ROOT / "tests/test_r5a3b_p2a_balanced_transport_allocation.py",
    PROJECT_ROOT / "docs/R5A3B_P2A_BALANCED_TRANSPORT_ALLOCATION.md",
)

PROTECTED_PRIOR_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_parse_hypotheses.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_object_graph.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_relation_programs.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_constraint_acquisition.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_relation_constraint_pilot.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a3b_p1_exposure_audit.py",
    PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot/final_report.json",
    PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot/raw_candidate_universe_ledger.json",
    PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot/condition_candidate_ledgers.json",
    PROJECT_ROOT / "artifacts/r5a3b_p1_exposure_audit/final_report.json",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def path_hashes(paths: tuple[Path, ...]) -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in paths}


def load_task_payload(source_relative_path: str) -> dict[str, Any]:
    source_path, task_id = source_relative_path.split("#", 1)
    payload = json.loads((PROJECT_ROOT / source_path).read_text(encoding="utf-8"))
    return payload[task_id]


def support_and_query_from_payload(payload: dict[str, Any], fold_index: int) -> tuple[tuple[SupportPair, ...], Grid]:
    supports = []
    query = None
    for index, pair in enumerate(payload.get("train", [])):
        if index == fold_index:
            query = freeze_grid(pair["input"])
        else:
            supports.append(SupportPair(freeze_grid(pair["input"]), freeze_grid(pair["output"])))
    if query is None:
        raise ValueError(f"missing fold index {fold_index}")
    return tuple(supports), query


def target_from_payload(payload: dict[str, Any], fold_index: int) -> Grid:
    return freeze_grid(payload["train"][fold_index]["output"])


def p0_object_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["parse_hypothesis_id"],
        tuple(candidate["source_program_ids"]),
        candidate["output_hash"],
        candidate["support_shape_mismatch_count"],
        candidate["support_cell_hamming_loss"],
        candidate["support_failed_pair_count"],
        tuple(candidate["operator_sequence"]),
        candidate["provenance"].get("source"),
    )


def verify_p0_object_branch_equivalence(
    manifest: dict[str, Any],
    p0_raw_ledger: list[dict[str, Any]],
    generated_by_fold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    p0_by_fold = {row["fold_key"]: row for row in p0_raw_ledger}
    rows = []
    mismatches = []
    for fold_key, generated in generated_by_fold.items():
        p0_objects = [candidate for candidate in p0_by_fold[fold_key]["raw_candidates"] if candidate["program_family"] == "object"]
        p2a_objects = generated["object_candidates"]
        p0_sigs = sorted(p0_object_signature(candidate) for candidate in p0_objects)
        p2a_sigs = sorted(p0_object_signature(candidate.to_json()) for candidate in p2a_objects)
        ok = p0_sigs == p2a_sigs
        rows.append(
            {
                "fold_key": fold_key,
                "p0_object_candidate_count": len(p0_objects),
                "p2a_object_candidate_count": len(p2a_objects),
                "matches": ok,
            }
        )
        if not ok:
            mismatches.append(
                {
                    "fold_key": fold_key,
                    "p0_signatures": p0_sigs[:20],
                    "p2a_signatures": p2a_sigs[:20],
                }
            )
            break
    return {
        "audit_version": AUDIT_VERSION,
        "status": "PASS" if not mismatches else "FAIL",
        "folds_checked": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "rows": rows,
    }


def pre_hidden_commitment(manifest: dict[str, Any], p0_confound: dict[str, Any], prior_hashes: dict[str, str | None]) -> dict[str, Any]:
    payload = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": now_utc(),
        "p0_p1_history_statement": "P0/P1 tested a relation grammar whose four relation slots were consumed by the first four lexical actions. P2A tests whether two previously unexposed vector-transfer actions contribute under the same final candidate budget.",
        "p0_pilot_manifest_sha256": stable_hash(manifest),
        "p0_allocation_confound_verification_sha256": stable_hash(p0_confound),
        "protected_prior_source_hashes": prior_hashes,
        "p2a_source_hashes": path_hashes(P2A_SOURCE_PATHS),
        "object_branch_equivalence_specification": {
            "slots_per_parse": 4,
            "programs": [
                "select=all_objects|ops=keep|render=same_canvas",
                "select=all_objects|ops=recolor:<support-derived>|render=same_canvas",
                "select=all_objects|ops=translate:<support-derived>|render=same_canvas",
                "select=all_objects|ops=copy:<support-derived>|render=same_canvas",
            ],
            "comparison_fields": ["output_hash", "program_id", "support_scores", "provenance"],
        },
        "allocation_spec": allocation_spec(),
        "final_candidate_caps": {
            "MAX_FINAL_PROGRAMS_PER_PARSE": MAX_FINAL_PROGRAMS_PER_PARSE,
            "MAX_FINAL_CANDIDATES_PER_FOLD": MAX_FINAL_CANDIDATES_PER_FOLD,
            "MAX_PROGRAM_DEPTH": MAX_PROGRAM_DEPTH,
            "TOP_K": TOP_K,
        },
        "ranking_tuple": [
            "support_shape_mismatch_count",
            "support_cell_hamming_loss",
            "support_failed_pair_count",
            "candidate_id",
            "output_hash",
        ],
        "tie_break_order": allocation_spec()["tie_break_order"],
        "test_query_firewall_policy": {
            "arc_test_outputs_read": False,
            "held_out_train_outputs_read_before_commitment": False,
            "candidate_generation_uses_visible_support_pairs_only": True,
        },
    }
    payload["commitment_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "timestamp_utc"})
    return payload


def _dedup_json(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        current = best.get(candidate["output_hash"])
        if current is None or tuple(candidate["base_rank_tuple"]) < tuple(current["base_rank_tuple"]):
            best[candidate["output_hash"]] = candidate
    return sorted(best.values(), key=lambda candidate: tuple(candidate["base_rank_tuple"]))


def recall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for k in TOP_K:
        key = f"Recall@{k}"
        hits = sum(row["target_rank"] is not None and row["target_rank"] <= min(k, row["candidate_count"]) for row in rows)
        summary[key] = {"numerator": hits, "denominator": len(rows), "rate": hits / len(rows) if rows else 0.0}
    ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
    summary["mean_target_rank_when_target_exists"] = sum(ranks) / len(ranks) if ranks else None
    summary["median_target_rank_when_target_exists"] = statistics.median(ranks) if ranks else None
    return summary


def score_candidates(candidates: list[dict[str, Any]], target: Grid) -> int | None:
    return target_rank(candidates, target)


def build_metrics(
    *,
    p0_condition_ledger: list[dict[str, Any]],
    p0_raw_ledger: list[dict[str, Any]],
    p2a_ledger: list[dict[str, Any]],
    targets_by_fold: dict[str, Grid],
) -> dict[str, Any]:
    p0_condition_by_fold = {row["fold_key"]: row for row in p0_condition_ledger}
    p0_raw_by_fold = {row["fold_key"]: row for row in p0_raw_ledger}
    p0_rows = []
    p2a_rows = []
    relation_only_hit_rows = []
    transport_counts = Counter()
    for row in p2a_ledger:
        fold_key = row["fold_key"]
        target = targets_by_fold[fold_key]
        p0_candidates = p0_condition_by_fold[fold_key]["conditions"]["RELATION_NO_PRUNE"]["top_candidates"]
        p2a_candidates = row["top_candidates"]
        p0_rank = score_candidates(p0_candidates, target)
        p2a_rank = score_candidates(p2a_candidates, target)
        p0_rows.append(
            {
                "fold_key": fold_key,
                "signature": row["signature"],
                "target_rank": p0_rank,
                "candidate_count": len(p0_candidates),
            }
        )
        p2a_rows.append(
            {
                "fold_key": fold_key,
                "signature": row["signature"],
                "target_rank": p2a_rank,
                "candidate_count": len(p2a_candidates),
            }
        )
        p0_object_hashes = {
            candidate["output_hash"]
            for candidate in p0_raw_by_fold[fold_key]["raw_candidates"]
            if candidate["program_family"] == "object"
        }
        transport_candidates = [
            candidate
            for candidate in row["raw_candidates"]
            if candidate["program_family"] == "relation"
            and candidate["operator_sequence"][1] in {"copy_source_by_relation_vector", "translate_target_by_relation_vector"}
        ]
        if any(candidate["operator_sequence"][1] == "copy_source_by_relation_vector" for candidate in transport_candidates):
            transport_counts["folds_with_transport_copy_emitted"] += 1
        if any(candidate["operator_sequence"][1] == "translate_target_by_relation_vector" for candidate in transport_candidates):
            transport_counts["folds_with_transport_translate_emitted"] += 1
        if transport_candidates:
            transport_counts["folds_with_at_least_one_transport_action_emitted"] += 1
        if any(candidate["output_hash"] not in p0_object_hashes for candidate in transport_candidates):
            transport_counts["folds_where_transport_created_relation_only_unique_output"] += 1
        if transport_candidates and all(candidate["output_hash"] in p0_object_hashes for candidate in transport_candidates):
            transport_counts["folds_where_transport_actions_only_duplicate_object_outputs"] += 1
        if any(candidate_trace_counts_json(candidate)["collision_count"] > 0 for candidate in transport_candidates):
            transport_counts["folds_where_transport_actions_have_collisions"] += 1
        if any(candidate_trace_counts_json(candidate)["dropped_out_of_bounds_count"] > 0 for candidate in transport_candidates):
            transport_counts["folds_where_transport_actions_drop_cells_out_of_bounds"] += 1
        for candidate in transport_candidates:
            if candidate["operator_sequence"][1] == "copy_source_by_relation_vector":
                transport_counts["transport_copy_programs_emitted"] += 1
            if candidate["operator_sequence"][1] == "translate_target_by_relation_vector":
                transport_counts["transport_translate_programs_emitted"] += 1
            if candidate["output_hash"] == canonical_grid_sha256(target) and freeze_grid(candidate["output_grid"]) == target:
                transport_counts["folds_where_transport_action_created_exact_hidden_target_hit"] += 1
                if candidate["output_hash"] not in p0_object_hashes:
                    relation_only_hit_rows.append(
                        {
                            "fold_key": fold_key,
                            "candidate_id": candidate["candidate_id"],
                            "operator_sequence": candidate["operator_sequence"],
                            "parse_hypothesis_id": candidate["parse_hypothesis_id"],
                            "target_rank": p2a_rank,
                        }
                    )
    def split(rows: list[dict[str, Any]], slice_name: str) -> list[dict[str, Any]]:
        if slice_name == "all":
            return rows
        return [row for row in rows if row["signature"]["object_slice"] == slice_name]
    metrics = {
        "audit_version": AUDIT_VERSION,
        "conditions": {},
        "primary_metrics": {},
        "transport_exposure": dict(transport_counts),
        "relation_only_hit_rows": relation_only_hit_rows,
    }
    for condition_name, rows in (("P0_SHALLOW_RELATION_REPLAY", p0_rows), ("P2A_BALANCED_TRANSPORT_RELATION", p2a_rows)):
        metrics["conditions"][condition_name] = {
            slice_name: recall_summary(split(rows, slice_name))
            for slice_name in ("all", "multi_object", "one_object")
        }
    p0_all = metrics["conditions"]["P0_SHALLOW_RELATION_REPLAY"]["all"]
    p2a_all = metrics["conditions"]["P2A_BALANCED_TRANSPORT_RELATION"]["all"]
    p0_multi = metrics["conditions"]["P0_SHALLOW_RELATION_REPLAY"]["multi_object"]
    p2a_multi = metrics["conditions"]["P2A_BALANCED_TRANSPORT_RELATION"]["multi_object"]
    metrics["primary_metrics"] = {
        "delta_recall_at_128_all": p2a_all["Recall@128"]["rate"] - p0_all["Recall@128"]["rate"],
        "delta_recall_at_128_multi_object": p2a_multi["Recall@128"]["rate"] - p0_multi["Recall@128"]["rate"],
        "delta_relation_only_hits": len(relation_only_hit_rows),
        "top_1_hit_change": p2a_all["Recall@1"]["rate"] - p0_all["Recall@1"]["rate"],
        "top_4_hit_change": p2a_all["Recall@4"]["rate"] - p0_all["Recall@4"]["rate"],
        "top_16_hit_change": p2a_all["Recall@16"]["rate"] - p0_all["Recall@16"]["rate"],
    }
    return metrics


def allocation_exposure_gate(candidate_ledger: list[dict[str, Any]], object_equiv: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if object_equiv["status"] != "PASS":
        failures.append("P0_OBJECT_BRANCH_NOT_EQUIVALENT")
    for row in candidate_ledger:
        any_eligible_vector = False
        any_emitted_vector = False
        for parse in row["microproposal_reports"]:
            buckets = {bucket_row["bucket"]: bucket_row for bucket_row in parse["bucket_rows"]}
            for bucket in ("RELATION_TRANSPORT_COPY", "RELATION_TRANSPORT_TRANSLATE"):
                if bucket not in buckets or not buckets[bucket]["attempted"]:
                    failures.append(f"{row['fold_key']}:{parse['parse_id']}:{bucket}:NOT_ATTEMPTED")
                if buckets[bucket]["legal_count"] > 0:
                    any_eligible_vector = True
            for bucket, actions in RELATION_ACTION_BUCKETS:
                selected = buckets[bucket]["selected_program"]
                if selected and selected["action"] not in actions:
                    failures.append(f"{row['fold_key']}:{parse['parse_id']}:{bucket}:SILENT_SUBSTITUTION")
            if any(
                buckets[bucket]["selected_program"]
                and buckets[bucket]["selected_program"]["action"] in {"copy_source_by_relation_vector", "translate_target_by_relation_vector"}
                for bucket in ("RELATION_TRANSPORT_COPY", "RELATION_TRANSPORT_TRANSLATE")
            ):
                any_emitted_vector = True
        if any_eligible_vector and not any_emitted_vector:
            failures.append(f"{row['fold_key']}:ELIGIBLE_VECTOR_BUCKET_WITHOUT_VECTOR_EMISSION")
    return {
        "status": "PASS" if not failures else "ALLOCATION_EXPOSURE_INVALID",
        "failure_count": len(failures),
        "failures": failures[:100],
        "gate_checks": [
            "P0 object branch is byte-equivalent in P2A.",
            "Vector-copy buckets are attempted when legal microproposals exist.",
            "Vector-translate buckets are attempted when legal microproposals exist.",
            "At least one vector action is emitted in every fold with a nonempty eligible vector bucket.",
            "No relation action bucket is silently replaced by another bucket.",
        ],
    }


def build_candidate_ledgers(manifest: dict[str, Any], p0_raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Grid]]:
    p0_by_fold = {row["fold_key"]: row for row in p0_raw}
    candidate_ledger = []
    generated_by_fold = {}
    runtime_rows = []
    targets_by_fold = {}
    for task_row in manifest["tasks"]:
        payload = load_task_payload(task_row["source_relative_path"])
        for fold_index in range(len(payload.get("train", []))):
            fold_start = time.perf_counter()
            fold_key = f"{task_row['input_only_selection_hash'][:12]}:fold={fold_index}"
            if fold_key not in p0_by_fold:
                continue
            supports, query = support_and_query_from_payload(payload, fold_index)
            universe = build_p2a_candidate_universe(supports, query)
            targets_by_fold[fold_key] = target_from_payload(payload, fold_index)
            generated_by_fold[fold_key] = universe
            raw_json = universe["raw_candidate_json"]
            top_json = universe["top_candidate_json"]
            candidate_ledger.append(
                {
                    "fold_key": fold_key,
                    "task_id_for_log_only": task_row["task_id"],
                    "fold_index": fold_index,
                    "signature": p0_by_fold[fold_key]["signature"],
                    "raw_candidate_count": len(raw_json),
                    "unique_candidate_count": len(top_json),
                    "microproposals_executed": universe["microproposal_count"],
                    "raw_candidates": raw_json,
                    "top_candidates": top_json,
                    "parse_reports": universe["parse_reports"],
                    "microproposal_reports": universe["microproposal_reports"],
                }
            )
            runtime_rows.append(
                {
                    "fold_key": fold_key,
                    "total_fold_seconds": time.perf_counter() - fold_start,
                    "candidate_generation_seconds": universe["runtime_seconds"],
                    "raw_candidate_count": len(raw_json),
                    "unique_candidate_count": len(top_json),
                    "microproposals_executed": universe["microproposal_count"],
                }
            )
    return candidate_ledger, generated_by_fold, runtime_rows, targets_by_fold


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(field: str) -> dict[str, Any]:
        vals = [float(row[field]) for row in rows]
        if not vals:
            return {"count": 0, "mean": None, "median": None, "max": None}
        return {"count": len(vals), "mean": sum(vals) / len(vals), "median": statistics.median(vals), "max": max(vals)}
    return {
        "audit_version": AUDIT_VERSION,
        "fold_count": len(rows),
        "stage_stats": {
            "total_fold_seconds": stats("total_fold_seconds"),
            "candidate_generation_seconds": stats("candidate_generation_seconds"),
            "microproposals_executed": stats("microproposals_executed"),
        },
        "slowest_10_folds": sorted(rows, key=lambda row: row["total_fold_seconds"], reverse=True)[:10],
    }


def transport_exposure_report(candidate_ledger: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for fold in candidate_ledger:
        transport = [
            candidate
            for candidate in fold["raw_candidates"]
            if candidate["program_family"] == "relation"
            and candidate["operator_sequence"][1] in {"copy_source_by_relation_vector", "translate_target_by_relation_vector"}
        ]
        rows.append(
            {
                "fold_key": fold["fold_key"],
                "transport_candidate_count": len(transport),
                "transport_actions": [candidate["operator_sequence"][1] for candidate in transport],
                "trace_counts": [candidate_trace_counts_json(candidate) for candidate in transport],
            }
        )
    return {"audit_version": AUDIT_VERSION, "summary": metrics["transport_exposure"], "rows": rows}


def bucket_allocation_report(candidate_ledger: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    bucket_counts = Counter()
    status_counts = Counter()
    for fold in candidate_ledger:
        for parse in fold["microproposal_reports"]:
            for row in parse["bucket_rows"]:
                bucket_counts[row["bucket"]] += int(row["selected_program"] is not None)
                status_counts[f"{row['bucket']}:{row['status']}"] += 1
    return {
        "audit_version": AUDIT_VERSION,
        "allocation_gate": gate,
        "selected_program_count_by_bucket": dict(bucket_counts),
        "bucket_status_counts": dict(status_counts),
        "no_bucket_silent_substitution": gate["status"] == "PASS",
    }


def microproposal_report(candidate_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for fold in candidate_ledger:
        for parse in fold["microproposal_reports"]:
            rows.append({"fold_key": fold["fold_key"], **parse})
    return {"audit_version": AUDIT_VERSION, "parse_count": len(rows), "rows": rows}


def comparison_report(metrics: dict[str, Any], object_equiv: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "object_branch_equivalence_status": object_equiv["status"],
        "allocation_exposure_gate_status": gate["status"],
        "primary_metrics": metrics["primary_metrics"],
        "p0_replay": metrics["conditions"]["P0_SHALLOW_RELATION_REPLAY"],
        "p2a": metrics["conditions"]["P2A_BALANCED_TRANSPORT_RELATION"],
    }


def representative_cards(candidate_ledger: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    lines = ["# R5A-3B-P2A Representative Transport Case Cards", ""]
    for row in candidate_ledger[:8]:
        transport = [
            candidate for candidate in row["raw_candidates"]
            if candidate["program_family"] == "relation"
            and candidate["operator_sequence"][1] in {"copy_source_by_relation_vector", "translate_target_by_relation_vector"}
        ]
        lines.append(f"## {row['fold_key']}")
        lines.append(f"- transport candidates: {len(transport)}")
        lines.append(f"- microproposals executed: {row['microproposals_executed']}")
        for candidate in transport[:4]:
            lines.append(f"- {candidate['operator_sequence'][1]} rank={candidate['base_rank_tuple']} trace={candidate_trace_counts_json(candidate)}")
        lines.append("")
    lines.append("## Primary Metrics")
    for key, value in metrics["primary_metrics"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def final_report(metrics: dict[str, Any], gate: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    if gate["status"] != "PASS":
        return {
            "Measured facts": {"allocation_exposure_gate_status": gate["status"], "failures": gate["failures"]},
            "What P2A establishes": ["The allocation exposure gate failed; recall was not interpreted."],
            "What P2A does not establish": [
                "It does not prove pair-specific edge binding is useless.",
                "It does not prove relation graphs are globally useful or useless.",
                "It does not prove more selectors in R5A-3A are useful.",
                "It is not a full-corpus ARC performance result.",
                "It does not establish the need for a neural controller or TTT.",
            ],
            "Narrowest justified next action": "C. Design a pair-specific edge-binding pilot because vector actions were exposed but global-edge semantics caused collision/bounds failure or no added coverage.",
        }
    relation_only_hits = metrics["primary_metrics"]["delta_relation_only_hits"]
    transport_summary = metrics["transport_exposure"]
    collisions_or_drops = transport_summary.get("folds_where_transport_actions_have_collisions", 0) or transport_summary.get("folds_where_transport_actions_drop_cells_out_of_bounds", 0)
    if relation_only_hits > 0:
        action = "B. Run a full R5A-3B audit only after P2A shows at least one relation-only hidden-target hit."
    elif collisions_or_drops:
        action = "C. Design a pair-specific edge-binding pilot because vector actions were exposed but global-edge semantics caused collision/bounds failure or no added coverage."
    else:
        action = "D. Stop global relation-transfer v0 because vector actions were adequately exposed and still produced no relation-only contribution."
    return {
        "Measured facts": {
            "allocation_exposure_gate_status": gate["status"],
            "primary_metrics": metrics["primary_metrics"],
            "transport_exposure": transport_summary,
            "recall": metrics["conditions"],
            "runtime": runtime["stage_stats"],
        },
        "What P2A establishes": [
            "Vector actions were genuinely exposed under the same final candidate budget.",
            "Balanced allocation improved or did not improve recall exactly as shown in the measured primary metrics.",
            "Transport collision and bounds behavior was logged for every emitted transport candidate.",
        ],
        "What P2A does not establish": [
            "It does not prove pair-specific edge binding is useless.",
            "It does not prove relation graphs are globally useful or useless.",
            "It does not prove more selectors in R5A-3A are useful.",
            "It is not a full-corpus ARC performance result.",
            "It does not establish the need for a neural controller or TTT.",
        ],
        "Narrowest justified next action": action,
    }


def prior_integrity(before: dict[str, str | None], after: dict[str, str | None]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "protected_prior_hashes_unchanged": before == after,
        "protected_prior_hashes_before": before,
        "protected_prior_hashes_after": after,
    }


def stable_artifact_hashes(root: Path) -> dict[str, str]:
    names = [
        "p0_object_branch_equivalence_verification.json",
        "relation_bucket_allocation_report.json",
        "relation_microproposal_report.json",
        "transport_exposure_report.json",
        "p2a_candidate_ledger.json",
        "p2a_recall_metrics.json",
        "p0_vs_p2a_comparison.json",
        "runtime_profile.json",
        "final_report.json",
    ]
    return {name: sha256_file(root / name) for name in names if (root / name).exists()}


def run_pilot_once(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    before = path_hashes(PROTECTED_PRIOR_PATHS)
    p0_confound = load_json(artifact_root / "p0_allocation_confound_verification.json")
    if p0_confound.get("status") != "PASS":
        raise RuntimeError("P0 allocation confound verification did not pass")
    manifest = load_json(P0_ARTIFACT_ROOT / "pilot_manifest.json")
    p0_raw = load_json(P0_ARTIFACT_ROOT / "raw_candidate_universe_ledger.json")
    p0_condition = load_json(P0_ARTIFACT_ROOT / "condition_candidate_ledgers.json")
    commitment = pre_hidden_commitment(manifest, p0_confound, before)
    write_json(artifact_root / "pre_hidden_p2a_commitment.json", commitment)
    candidate_ledger, generated_by_fold, runtime_rows, targets_by_fold = build_candidate_ledgers(manifest, p0_raw)
    object_equiv = verify_p0_object_branch_equivalence(manifest, p0_raw, generated_by_fold)
    write_json(artifact_root / "p0_object_branch_equivalence_verification.json", object_equiv)
    if object_equiv["status"] != "PASS":
        raise RuntimeError("P2A object branch failed equivalence verification")
    gate = allocation_exposure_gate(candidate_ledger, object_equiv)
    write_json(artifact_root / "p2a_candidate_ledger.json", candidate_ledger)
    relation_bucket = bucket_allocation_report(candidate_ledger, gate)
    micro = microproposal_report(candidate_ledger)
    write_json(artifact_root / "relation_bucket_allocation_report.json", relation_bucket)
    write_json(artifact_root / "relation_microproposal_report.json", micro)
    runtime = summarize_runtime(runtime_rows)
    write_json(artifact_root / "runtime_profile.json", runtime)
    if gate["status"] == "PASS":
        metrics = build_metrics(p0_condition_ledger=p0_condition, p0_raw_ledger=p0_raw, p2a_ledger=candidate_ledger, targets_by_fold=targets_by_fold)
    else:
        metrics = {"audit_version": AUDIT_VERSION, "status": "ALLOCATION_EXPOSURE_INVALID", "conditions": {}, "primary_metrics": {}, "transport_exposure": {}}
    transport = transport_exposure_report(candidate_ledger, metrics if gate["status"] == "PASS" else {"transport_exposure": {}})
    comparison = comparison_report(metrics, object_equiv, gate)
    write_json(artifact_root / "transport_exposure_report.json", transport)
    write_json(artifact_root / "p2a_recall_metrics.json", metrics)
    write_json(artifact_root / "p0_vs_p2a_comparison.json", comparison)
    write_text(artifact_root / "representative_transport_case_cards.md", representative_cards(candidate_ledger, metrics if gate["status"] == "PASS" else {"primary_metrics": {}}))
    final = final_report(metrics, gate, runtime)
    after = path_hashes(PROTECTED_PRIOR_PATHS)
    write_json(artifact_root / "prior_integrity_verification.json", prior_integrity(before, after))
    write_json(artifact_root / "final_report.json", final)
    write_json(
        artifact_root / "determinism_verification.json",
        {
            "audit_version": AUDIT_VERSION,
            "stable_artifact_hashes": stable_artifact_hashes(artifact_root),
            "deterministic_allocation_rules": True,
            "candidate_generation_rerun_for_determinism": False,
        },
    )
    return final


def main() -> dict[str, Any]:
    return run_pilot_once(artifact_root=ARTIFACT_ROOT)
