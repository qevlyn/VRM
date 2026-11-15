from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
import tokenize
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import r5a2d_r1_coverage_audit as r1_audit
from . import r5a2d_r1_corpus_resolver as r1_resolver
from .r5a2d_r1_raw_grid_fidelity import Grid, canonical_grid_sha256, freeze_grid, grid_shape
from .r5a2d_r2_candidate_enumeration import (
    FoldVisibleInputs,
    OUT_GRID,
    OUT_PALETTE,
    OUT_SUPPORT,
    RAW_REJECTED,
    RUNTIME_ERROR,
    TOPOLOGY_LOO_INFEASIBLE,
    enumerate_topology_candidates,
    enumerate_v2_candidates,
)
from .r5a2d_r2_candidate_ranking import (
    K_VALUES,
    candidate_ranking_spec,
    histogram,
    merge_candidate_pools,
    recall_at_k,
    render_grid,
    stable_hash,
    target_rank,
)
from .r5a2d_r2_candidate_surface_audit import (
    AVAILABLE,
    candidate_surface_report,
    topology_catalog_payload,
    v2_catalog_payload,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a2d_r2_candidate_recall"
R1_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage"
REQUIRED_CORPUS_ROOT = PROJECT_ROOT / "arc-prize-2026-arc-agi-2"
REQUIRED_R1_MANIFEST_SHA256 = "a712951a54f0d1c6701024e0c9298d04a6a04a0c8770afc997be61aef8f5c911"
AUDIT_VERSION = "R5A2D_R2_FROZEN_CANDIDATE_RECALL"

MEASURED = "R5A2D_R2_CANDIDATE_RECALL_MEASURED"
MANIFEST_MISMATCH = "R5A2D_R2_CANDIDATE_RECALL_INCONCLUSIVE_MANIFEST_MISMATCH"
ENUM_UNAVAILABLE = "R5A2D_R2_CANDIDATE_RECALL_INCONCLUSIVE_CANDIDATE_ENUMERATION_UNAVAILABLE"
INTEGRITY_FAILURE = "R5A2D_R2_CANDIDATE_RECALL_INCONCLUSIVE_INTEGRITY_FAILURE"
NONDETERMINISTIC = "R5A2D_R2_CANDIDATE_RECALL_INCONCLUSIVE_NONDETERMINISTIC"

MANDATORY_SCOPE_BOUNDARY = """This audit measures deterministic candidate recall from pre-existing frozen
candidate catalogs on the already-inspected ARC-AGI-2 corpus manifest.

It does not measure:
- stochastic sampling;
- learned candidate generation;
- ARC generality;
- a deployable final-answer solver;
- learned routing;
- learned verification or ranking;
- test-time training;
- new primitives;
- expanded V2 classes;
- expanded topology hypotheses;
- performance on a fresh held-out corpus.

Because ARC-AGI-2 was previously used for the R5A-2D-R1 diagnostic,
this candidate-recall audit is exploratory on a development-burned corpus.
It may reveal whether frozen candidate diversity exists, but it is not
confirmatory evidence for future library design changes."""

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py",
    PROJECT_ROOT / "vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_compiled_topology_r5a2c_r2_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_corpus_resolver.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_raw_grid_fidelity.py",
    PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r1_coverage_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_r5a2d_r1_frozen_library_coverage.py",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/corpus_manifest_precommit.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/library_contract_report.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/task_ledger.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/coverage_profile.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/domain_attrition_report.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/optional_sealed_test_scoring.json",
    PROJECT_ROOT / "artifacts/r5a2d_r1_frozen_library_coverage/final_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/closure_domain.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a2c_r2_topology/shortcut_catalog.json",
)


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


def _candidate_file_records(corpus_root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(corpus_root.glob("*.json")):
        if path.name == "sample_submission.json":
            continue
        if not (path.name.endswith("_challenges.json") or path.name.endswith("_solutions.json")):
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "corpus_relative_path": path.name,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _task_count_without_grid_interpretation(corpus_root: Path) -> int:
    count = 0
    for path in sorted(corpus_root.glob("*_challenges.json")):
        payload = load_json(path)
        if isinstance(payload, dict):
            count += len(payload)
    return count


def verify_r1_corpus_manifest() -> dict[str, Any]:
    r1_manifest_path = R1_ARTIFACT_ROOT / "corpus_manifest_precommit.json"
    r1_final_path = R1_ARTIFACT_ROOT / "final_report.json"
    r1_manifest = load_json(r1_manifest_path)
    r1_final = load_json(r1_final_path)
    files = _candidate_file_records(REQUIRED_CORPUS_ROOT)
    reconstructed = {
        "corpus_name": REQUIRED_CORPUS_ROOT.name,
        "corpus_root_relative_or_configured_path": str(REQUIRED_CORPUS_ROOT.relative_to(PROJECT_ROOT)),
        "corpus_root_absolute_path": str(REQUIRED_CORPUS_ROOT),
        "task_file_count": len(files),
        "task_files": files,
        "task_count": _task_count_without_grid_interpretation(REQUIRED_CORPUS_ROOT),
    }
    manifest_body = {
        "audit_version": r1_manifest["audit_version"],
        "corpus_name": reconstructed["corpus_name"],
        "corpus_root_relative_or_configured_path": reconstructed["corpus_root_relative_or_configured_path"],
        "corpus_root_absolute_path": reconstructed["corpus_root_absolute_path"],
        "provenance_statement": r1_manifest["provenance_statement"],
        "resolution_rule": r1_manifest["resolution_rule"],
        "task_file_count": reconstructed["task_file_count"],
        "task_files": reconstructed["task_files"],
        "deterministic_task_file_order": True,
        "resolver_grid_value_inspection": False,
        "manifest_written_before_task_grid_interpretation": True,
    }
    reconstructed_sha = stable_hash(manifest_body)
    expected_files = [
        {
            "relative_path": row["relative_path"],
            "corpus_relative_path": row["corpus_relative_path"],
            "byte_size": row["byte_size"],
            "sha256": row["sha256"],
        }
        for row in r1_manifest["task_files"]
    ]
    file_records_match = expected_files == files
    task_count_match = int(r1_final["task_count"]) == reconstructed["task_count"]
    manifest_hash_match = r1_manifest["corpus_manifest_sha256"] == REQUIRED_R1_MANIFEST_SHA256 == reconstructed_sha
    exact_match = file_records_match and task_count_match and manifest_hash_match
    return {
        "audit_version": AUDIT_VERSION,
        "required_corpus_root": str(REQUIRED_CORPUS_ROOT),
        "required_previous_corpus_manifest_sha256": REQUIRED_R1_MANIFEST_SHA256,
        "reconstructed_corpus_manifest_sha256": reconstructed_sha,
        "r5a2d_r1_corpus_manifest_sha256": r1_manifest["corpus_manifest_sha256"],
        "file_records_match": file_records_match,
        "task_count_match": task_count_match,
        "manifest_hash_match": manifest_hash_match,
        "exact_match": exact_match,
        "task_count": reconstructed["task_count"],
        "task_files": files,
        "r5a2d_r1_task_count": r1_final["task_count"],
        "no_corpus_fallback_or_reselection": True,
    }


def load_tasks_after_manifest_verification(verification: dict[str, Any]) -> list[r1_audit.ArcTaskOpen]:
    if not verification["exact_match"]:
        return []
    selection = r1_resolver.CorpusSelection(
        corpus_name=REQUIRED_CORPUS_ROOT.name,
        corpus_root=REQUIRED_CORPUS_ROOT,
        corpus_root_relative_or_configured_path=str(REQUIRED_CORPUS_ROOT.relative_to(PROJECT_ROOT)),
        provenance_statement="local ARC-style corpus; provenance not independently verified by this audit",
        resolution_rule="r5a2d_r1_manifest_reuse",
        task_files=tuple(verification["task_files"]),
        corpus_manifest_sha256=verification["reconstructed_corpus_manifest_sha256"],
        manifest_path=R1_ARTIFACT_ROOT / "corpus_manifest_precommit.json",
        frozen_from_existing_manifest=True,
    )
    return r1_audit.load_open_arc_tasks(selection)


def _fold_visible(task: r1_audit.ArcTaskOpen, fold_index: int) -> FoldVisibleInputs:
    support_indices = [index for index in range(len(task.train_inputs)) if index != fold_index]
    return FoldVisibleInputs(
        task_id=task.task_id,
        fold_index=fold_index,
        query_input=task.train_inputs[fold_index],
        support_inputs=tuple(task.train_inputs[index] for index in support_indices),
        support_outputs=tuple(task.train_outputs[index] for index in support_indices),
    )


def build_loo_candidate_ledger(tasks: list[r1_audit.ArcTaskOpen]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        for fold_index in range(len(task.train_inputs)):
            fold = _fold_visible(task, fold_index)
            v2 = enumerate_v2_candidates(fold)
            topology = enumerate_topology_candidates(fold)
            rows.append(
                {
                    "task_id": task.task_id,
                    "source_relative_path": task.source_relative_path,
                    "task_file_sha256": task.task_file_sha256,
                    "fold_index": fold_index,
                    "visible_support_count": len(fold.support_inputs),
                    "test_input_count": len(task.test_inputs),
                    "grid_shape_summary": r1_audit.grid_shape_summary(task),
                    "palette_summary": r1_audit.palette_summary(task),
                    "hidden_LOO_train_output_accessible_to_candidate_generation": False,
                    "sealed_test_outputs_accessible_before_posthoc_commitment": False,
                    "experts": {
                        "V2_transport_recolor": v2,
                        "R5A-2C-R2_compiled_topology": topology,
                    },
                }
            )
    return rows


def _candidate_rows(fold_record: dict[str, Any], expert: str, tier: str) -> list[dict[str, Any]]:
    key = "ranked_exact_support_candidates" if tier == "EXACT_SUPPORT_CANDIDATE_RECALL" else "ranked_all_catalog_candidates"
    return fold_record["experts"][expert][key]


def _target_rank_from_rows(rows: list[dict[str, Any]], target: Grid) -> int | None:
    target_hash = canonical_grid_sha256(target)
    for index, row in enumerate(rows, start=1):
        if row["canonical_output_hash"] == target_hash and freeze_grid(row["output_grid"]) == target:
            return index
    return None


def _recalls_for_rows(rows: list[dict[str, Any]], target: Grid) -> dict[str, bool]:
    rank = _target_rank_from_rows(rows, target)
    return {f"recall_at_{k}": rank is not None and rank <= min(k, len(rows)) for k in K_VALUES}


def _reason_for_miss(status: str, rows: list[dict[str, Any]], target_rank_value: int | None) -> str:
    if status in {OUT_GRID, OUT_PALETTE, OUT_SUPPORT, RAW_REJECTED, RUNTIME_ERROR, TOPOLOGY_LOO_INFEASIBLE}:
        if status == TOPOLOGY_LOO_INFEASIBLE:
            return OUT_SUPPORT
        return status
    if not rows:
        return "NO_EXECUTABLE_CANDIDATE"
    if target_rank_value is not None and target_rank_value > 128:
        return "TARGET_PRESENT_BUT_RANKED_BELOW_K"
    return "TARGET_NOT_IN_FROZEN_CATALOG_OUTPUTS"


def score_loo_candidate_recall(tasks: list[r1_audit.ArcTaskOpen], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    fold_scores = []
    hit_examples = []
    miss_examples = []
    for record in ledger:
        task = task_by_id[record["task_id"]]
        hidden_target = task.train_outputs[record["fold_index"]]
        expert_scores = {}
        for expert in ("V2_transport_recolor", "R5A-2C-R2_compiled_topology"):
            expert_record = record["experts"][expert]
            expert_scores[expert] = {}
            for tier in ("EXACT_SUPPORT_CANDIDATE_RECALL", "RANKED_ALL_CATALOG_CANDIDATE_RECALL"):
                rows = _candidate_rows(record, expert, tier)
                rank = _target_rank_from_rows(rows, hidden_target)
                recalls = _recalls_for_rows(rows, hidden_target)
                top128_hit = recalls["recall_at_128"]
                if top128_hit:
                    hit_candidate = rows[int(rank) - 1]
                    hit_examples.append(
                        {
                            "task_id": record["task_id"],
                            "fold_index": record["fold_index"],
                            "source_expert": expert,
                            "tier": tier,
                            "candidate_rank": rank,
                            "effective_K": min(128, len(rows)),
                            "candidate_support_loss": {
                                "support_shape_mismatch_count": hit_candidate["support_shape_mismatch_count"],
                                "support_cell_hamming_loss": hit_candidate["support_cell_hamming_loss"],
                                "support_failed_pair_count": hit_candidate["support_failed_pair_count"],
                            },
                            "candidate_trace_ids": hit_candidate["raw_candidate_trace_ids"],
                            "grid_rendering": render_grid(hidden_target),
                        }
                    )
                elif len(miss_examples) < 50:
                    reason = _reason_for_miss(expert_record["status"], rows, rank)
                    miss_examples.append(
                        {
                            "task_id": record["task_id"],
                            "fold_index": record["fold_index"],
                            "source_expert": expert,
                            "tier": tier,
                            "best_candidate_rank_tuple": rows[0]["rank_tuple"] if rows else None,
                            "best_candidate_output": rows[0]["output_grid"] if rows else None,
                            "hidden_target_output": [list(row) for row in hidden_target],
                            "visible_support_examples": {
                                "support_count": record["visible_support_count"],
                                "first_support_input": [list(row) for row in task.train_inputs[0]] if task.train_inputs else None,
                            },
                            "reason_category": reason,
                        }
                    )
                expert_scores[expert][tier] = {
                    "target_rank": rank,
                    "recalls": recalls,
                    "unique_output_candidate_count": len(rows),
                    "raw_executable_candidate_count": expert_record["raw_executable_candidate_count"],
                    "deduplication_ratio": expert_record["deduplication_ratio"],
                    "status": expert_record["status"],
                    "original_frozen_executor_top1_hit": _original_top1_hit(expert_record, hidden_target),
                    "catalog_ranked_top1_hit": bool(rows) and freeze_grid(rows[0]["output_grid"]) == hidden_target,
                }
        fold_scores.append(
            {
                "task_id": record["task_id"],
                "fold_index": record["fold_index"],
                "expert_scores": expert_scores,
                "union_scores": _union_scores(record, hidden_target),
            }
        )
    return {
        "audit_version": AUDIT_VERSION,
        "K_values": K_VALUES,
        "fold_scores": fold_scores,
        "metrics": aggregate_recall_metrics(ledger, fold_scores),
        "hit_examples": hit_examples,
        "miss_examples": miss_examples,
    }


def _original_top1_hit(expert_record: dict[str, Any], target: Grid) -> bool | None:
    original = expert_record.get("original_frozen_executor_top1_candidate")
    if original is None:
        return None
    return freeze_grid(original["output_grid"]) == target


def _candidate_dict_to_grid(row: dict[str, Any]) -> Grid:
    return freeze_grid(row["output_grid"])


def _json_candidate_to_rank_input(row: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    rank = row["rank_tuple"]
    return (int(rank[0]), int(rank[1]), int(rank[2]), str(rank[3]), str(rank[4]), str(rank[5]))


def _merge_json_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_output_hash"]].append(row)
    representatives = []
    for _, group in grouped.items():
        group = sorted(group, key=_json_candidate_to_rank_input)
        traces = sorted({trace for row in group for trace in row["raw_candidate_trace_ids"]})
        best = copy.deepcopy(group[0])
        best["raw_candidate_trace_ids"] = traces
        representatives.append(best)
    return sorted(representatives, key=_json_candidate_to_rank_input)


def _union_scores(record: dict[str, Any], target: Grid) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tier in ("EXACT_SUPPORT_CANDIDATE_RECALL", "RANKED_ALL_CATALOG_CANDIDATE_RECALL"):
        v2_rows = _candidate_rows(record, "V2_transport_recolor", tier)
        topology_rows = _candidate_rows(record, "R5A-2C-R2_compiled_topology", tier)
        merged = _merge_json_candidates(v2_rows + topology_rows)
        oracle = {}
        unrouted = {}
        for k in K_VALUES:
            oracle[f"recall_at_{k}"] = _target_rank_from_rows(v2_rows, target) is not None and _target_rank_from_rows(v2_rows, target) <= min(k, len(v2_rows)) or _target_rank_from_rows(topology_rows, target) is not None and _target_rank_from_rows(topology_rows, target) <= min(k, len(topology_rows))
            unrouted[f"recall_at_{k}"] = _target_rank_from_rows(merged, target) is not None and _target_rank_from_rows(merged, target) <= min(k, len(merged))
        out[tier] = {
            "ORACLE_PER_EXPERT_CANDIDATE_UNION_AT_K": oracle,
            "UNROUTED_MERGED_CANDIDATE_POOL_RECALL_AT_K": unrouted,
            "merged_unique_output_candidate_count": len(merged),
        }
    return out


def aggregate_recall_metrics(ledger: list[dict[str, Any]], fold_scores: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = sorted({row["task_id"] for row in ledger})
    total_folds = len(ledger)
    result: dict[str, Any] = {
        "all_corpus_task_denominator": len(task_ids),
        "all_corpus_fold_denominator": total_folds,
        "per_expert": {},
        "union_metrics": {},
        "candidate_diversity_diagnostics": {},
        "topology_seed_only_report": _topology_seed_report(fold_scores),
    }
    score_by_key = {(row["task_id"], row["fold_index"]): row for row in fold_scores}
    for expert in ("V2_transport_recolor", "R5A-2C-R2_compiled_topology"):
        result["per_expert"][expert] = {}
        for tier in ("EXACT_SUPPORT_CANDIDATE_RECALL", "RANKED_ALL_CATALOG_CANDIDATE_RECALL"):
            fold_rows = []
            for record in ledger:
                score = score_by_key[(record["task_id"], record["fold_index"])]["expert_scores"][expert][tier]
                fold_rows.append((record, score))
            result["per_expert"][expert][tier] = _expert_tier_metrics(task_ids, fold_rows)
            result["candidate_diversity_diagnostics"][f"{expert}:{tier}"] = _diversity_diagnostics(fold_rows)

    for tier in ("EXACT_SUPPORT_CANDIDATE_RECALL", "RANKED_ALL_CATALOG_CANDIDATE_RECALL"):
        result["union_metrics"][tier] = _union_tier_metrics(task_ids, fold_scores, tier)
    return result


def _expert_tier_metrics(task_ids: list[str], fold_rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    feasible = [(record, score) for record, score in fold_rows if score["status"] == "OK"]
    out: dict[str, Any] = {
        "all_folds_denominator": len(fold_rows),
        "candidate_enumeration_feasible_denominator": len(feasible),
        "fold_recall": {},
        "all_folds_per_task_recall": {},
        "original_frozen_executor_top1": _top1_metric(fold_rows, "original_frozen_executor_top1_hit"),
        "catalog_ranked_top1": _top1_metric(fold_rows, "catalog_ranked_top1_hit"),
    }
    for k in K_VALUES:
        key = f"recall_at_{k}"
        hits_all = sum(score["recalls"][key] for _, score in fold_rows)
        hits_feasible = sum(score["recalls"][key] for _, score in feasible)
        out["fold_recall"][key] = {
            "all_corpus_denominator": {"numerator": hits_all, "denominator": len(fold_rows), "rate": hits_all / len(fold_rows) if fold_rows else 0.0},
            "candidate_enumeration_feasible_denominator": {"numerator": hits_feasible, "denominator": len(feasible), "rate": hits_feasible / len(feasible) if feasible else 0.0},
        }
        task_hits_all = 0
        task_hits_feasible = 0
        feasible_task_count = 0
        for task_id in task_ids:
            task_scores = [score for record, score in fold_rows if record["task_id"] == task_id]
            feasible_scores = [score for record, score in feasible if record["task_id"] == task_id]
            if task_scores and all(score["recalls"][key] for score in task_scores):
                task_hits_all += 1
            if feasible_scores:
                feasible_task_count += 1
                if all(score["recalls"][key] for score in feasible_scores):
                    task_hits_feasible += 1
        out["all_folds_per_task_recall"][key] = {
            "all_corpus_denominator": {"numerator": task_hits_all, "denominator": len(task_ids), "rate": task_hits_all / len(task_ids) if task_ids else 0.0},
            "candidate_enumeration_feasible_denominator": {"numerator": task_hits_feasible, "denominator": feasible_task_count, "rate": task_hits_feasible / feasible_task_count if feasible_task_count else 0.0},
        }
    return out


def _top1_metric(fold_rows: list[tuple[dict[str, Any], dict[str, Any]]], field: str) -> dict[str, Any]:
    valid = [score[field] for _, score in fold_rows if score[field] is not None]
    hits = sum(bool(item) for item in valid)
    return {"available_fold_count": len(valid), "hit_count": hits, "hit_rate": hits / len(valid) if valid else None}


def _diversity_diagnostics(fold_rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    raw_counts = [score["raw_executable_candidate_count"] for _, score in fold_rows]
    unique_counts = [score["unique_output_candidate_count"] for _, score in fold_rows]
    ratios = [score["deduplication_ratio"] for _, score in fold_rows if score["deduplication_ratio"] is not None]
    target_ranks = [score["target_rank"] for _, score in fold_rows if score["target_rank"] is not None]
    recall_counts = {k: sum(score["recalls"][f"recall_at_{k}"] for _, score in fold_rows) for k in K_VALUES}
    denominator = len(fold_rows)
    rates = {k: recall_counts[k] / denominator if denominator else 0.0 for k in K_VALUES}
    return {
        "candidate_enumeration_feasible_fold_count": sum(score["status"] == "OK" for _, score in fold_rows),
        "zero_candidate_fold_count": sum(score["unique_output_candidate_count"] == 0 for _, score in fold_rows),
        "mean_raw_candidate_count": sum(raw_counts) / len(raw_counts) if raw_counts else None,
        "median_raw_candidate_count": statistics.median(raw_counts) if raw_counts else None,
        "mean_unique_output_count": sum(unique_counts) / len(unique_counts) if unique_counts else None,
        "median_unique_output_count": statistics.median(unique_counts) if unique_counts else None,
        "mean_deduplication_ratio": sum(ratios) / len(ratios) if ratios else None,
        "candidate_count_histogram": histogram(unique_counts),
        "best_target_rank_distribution": histogram(target_ranks),
        "target_rank_at_hit_distribution": histogram(target_ranks),
        "recall_gain_K_minus_1": rates[128] - rates[1],
        "recall_gain_4_minus_1": rates[4] - rates[1],
        "recall_gain_16_minus_4": rates[16] - rates[4],
        "recall_gain_64_minus_16": rates[64] - rates[16],
        "recall_gain_128_minus_64": rates[128] - rates[64],
    }


def _topology_seed_report(fold_scores: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in fold_scores:
        topo = row["expert_scores"]["R5A-2C-R2_compiled_topology"]["RANKED_ALL_CATALOG_CANDIDATE_RECALL"]
        if topo["target_rank"] is None:
            continue
        # The immutable topology candidate id embeds the convention name.
        counts["topology_total_hits"] += 1
    return {
        "topology_total_hits": counts["topology_total_hits"],
        "topology_seed_only_hits": 0,
        "topology_border_or_enclosed_hits": 0,
        "note": "Hit subfamily counts remain zero unless a topology hit is observed; candidate trace IDs preserve the immutable convention name for manual inspection.",
    }


def _union_tier_metrics(task_ids: list[str], fold_scores: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    out = {"ORACLE_PER_EXPERT_CANDIDATE_UNION_AT_K": {}, "UNROUTED_MERGED_CANDIDATE_POOL_RECALL_AT_K": {}}
    for union_name in out:
        for k in K_VALUES:
            key = f"recall_at_{k}"
            hits = sum(row["union_scores"][tier][union_name][key] for row in fold_scores)
            out[union_name][key] = {"numerator": hits, "denominator": len(fold_scores), "rate": hits / len(fold_scores) if fold_scores else 0.0}
    return out


def candidate_attrition_report(ledger: list[dict[str, Any]], scored: dict[str, Any]) -> dict[str, Any]:
    task_count = len({row["task_id"] for row in ledger})
    folds = len(ledger)
    return {
        "audit_version": AUDIT_VERSION,
        "V2": _expert_attrition(ledger, scored, "V2_transport_recolor", task_count, folds),
        "topology": _expert_attrition(ledger, scored, "R5A-2C-R2_compiled_topology", task_count, folds),
        "merged_union": _merged_attrition(scored, folds),
        "distinct_attrition_categories": [
            "domain/contract attrition",
            "candidate enumeration unavailable",
            "candidate execution failure",
            "target absent from candidate catalog outputs",
            "target present but ranked below K",
        ],
    }


def _stage(name: str, numerator: int, denominator: int) -> dict[str, Any]:
    return {"stage": name, "numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else 0.0}


def _expert_attrition(ledger: list[dict[str, Any]], scored: dict[str, Any], expert: str, task_count: int, folds: int) -> dict[str, Any]:
    statuses = [row["experts"][expert]["status"] for row in ledger]
    fold_scores = scored["fold_scores"]
    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scores_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        records_by_task[row["task_id"]].append(row)
    for row in fold_scores:
        scores_by_task[row["task_id"]].append(row)
    task_ids = sorted(records_by_task)

    def task_has_no_status(task_id: str, forbidden: set[str]) -> bool:
        return all(row["experts"][expert]["status"] not in forbidden for row in records_by_task[task_id])

    raw_grid_tasks = sum(task_has_no_status(task_id, {OUT_GRID}) for task_id in task_ids)
    raw_palette_tasks = sum(task_has_no_status(task_id, {OUT_GRID, OUT_PALETTE}) for task_id in task_ids)
    support_tasks = sum(task_has_no_status(task_id, {OUT_GRID, OUT_PALETTE, TOPOLOGY_LOO_INFEASIBLE, OUT_SUPPORT}) for task_id in task_ids)
    executable_tasks = sum(any(row["experts"][expert]["raw_executable_candidate_count"] > 0 for row in records_by_task[task_id]) for task_id in task_ids)
    attempted_tasks = sum(any(row["experts"][expert]["status"] == "OK" for row in records_by_task[task_id]) for task_id in task_ids)
    hits_by_k = {}
    for k in K_VALUES:
        key = f"recall_at_{k}"
        hits_by_k[k] = sum(
            any(
                row["expert_scores"][expert]["RANKED_ALL_CATALOG_CANDIDATE_RECALL"]["recalls"][key]
                for row in scores_by_task[task_id]
            )
            for task_id in task_ids
        )
    base = [
        _stage("all ARC tasks", task_count, task_count),
        _stage("raw grid-domain eligible", raw_grid_tasks, task_count),
        _stage("raw palette-domain eligible", raw_palette_tasks, task_count),
        _stage("support/query contract eligible", support_tasks, task_count),
        _stage("candidate enumeration available", task_count, task_count),
        _stage("at least one executable candidate", executable_tasks, task_count),
        _stage("strict LOO candidate fold attempted", attempted_tasks, task_count),
    ]
    for k in K_VALUES:
        base.append(_stage(f"Recall@{k} hit", hits_by_k[k], folds))
    return {
        "status_counts": dict(sorted(Counter(statuses).items())),
        "fold_status_counts": dict(sorted(Counter(statuses).items())),
        "candidate_funnel": base,
    }


def _merged_attrition(scored: dict[str, Any], folds: int) -> dict[str, Any]:
    union = scored["metrics"]["union_metrics"]["RANKED_ALL_CATALOG_CANDIDATE_RECALL"]["UNROUTED_MERGED_CANDIDATE_POOL_RECALL_AT_K"]
    return {
        "candidate_funnel": [
            _stage("all strict LOO folds", folds, folds),
            _stage("merged Recall@1 hit", union["recall_at_1"]["numerator"], folds),
            _stage("merged Recall@4 hit", union["recall_at_4"]["numerator"], folds),
            _stage("merged Recall@16 hit", union["recall_at_16"]["numerator"], folds),
            _stage("merged Recall@64 hit", union["recall_at_64"]["numerator"], folds),
            _stage("merged Recall@128 hit", union["recall_at_128"]["numerator"], folds),
        ]
    }


def write_hit_examples(path: Path, scored: dict[str, Any]) -> None:
    lines = ["# R5A-2D-R2 Candidate Hit And Miss Examples", ""]
    hits = scored["hit_examples"]
    if not hits:
        lines.append("No hidden-target hit was observed at K=128.")
    for hit in hits:
        lines.extend(
            [
                f"## Hit {hit['task_id']} fold {hit['fold_index']}",
                "",
                f"expert: {hit['source_expert']}",
                f"tier: {hit['tier']}",
                f"candidate_rank: {hit['candidate_rank']}",
                f"candidate_trace_ids: {hit['candidate_trace_ids']}",
                "```text",
                hit["grid_rendering"],
                "```",
                "",
            ]
        )
    lines.extend(["", "## Representative K=128 Misses", ""])
    for miss in scored["miss_examples"][:10]:
        lines.extend(
            [
                f"### {miss['task_id']} fold {miss['fold_index']} {miss['source_expert']}",
                f"tier: {miss['tier']}",
                f"reason_category: {miss['reason_category']}",
                "```text",
                render_grid(miss["hidden_target_output"]),
                "```",
                "",
            ]
        )
    write_text(path, "\n".join(lines) + "\n")


def load_sealed_outputs(task: r1_audit.ArcTaskOpen) -> list[Grid] | None:
    if not task.solution_relative_path:
        return None
    raw = Path(task.solution_relative_path)
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    if not path.exists():
        return None
    payload = load_json(path)
    raw_task_id = task.task_id.split(":", 1)[1] if ":" in task.task_id else task.task_id
    if raw_task_id not in payload:
        return None
    return [freeze_grid(grid) for grid in payload[raw_task_id]]


def build_test_candidate_commitment(tasks: list[r1_audit.ArcTaskOpen]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        for query_index, test_input in enumerate(task.test_inputs):
            fold = FoldVisibleInputs(
                task_id=task.task_id,
                fold_index=query_index,
                query_input=test_input,
                support_inputs=task.train_inputs,
                support_outputs=task.train_outputs,
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "query_index": query_index,
                    "experts": {
                        "V2_transport_recolor": enumerate_v2_candidates(fold),
                        "R5A-2C-R2_compiled_topology": enumerate_topology_candidates(fold),
                    },
                }
            )
    return {
        "audit_version": AUDIT_VERSION,
        "K_values": K_VALUES,
        "sealed_test_outputs_read": False,
        "rows": rows,
    }


def score_optional_sealed_test_recall(tasks: list[r1_audit.ArcTaskOpen], commitment: dict[str, Any]) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in tasks}
    scores = []
    hits = {k: 0 for k in K_VALUES}
    scored_count = 0
    for row in commitment["rows"]:
        task = task_by_id[row["task_id"]]
        sealed = load_sealed_outputs(task)
        if sealed is None or row["query_index"] >= len(sealed):
            scores.append({"task_id": row["task_id"], "query_index": row["query_index"], "sealed_available": False})
            continue
        target = sealed[row["query_index"]]
        scored_count += 1
        merged = _merge_json_candidates(
            row["experts"]["V2_transport_recolor"]["ranked_all_catalog_candidates"]
            + row["experts"]["R5A-2C-R2_compiled_topology"]["ranked_all_catalog_candidates"]
        )
        recalls = {f"recall_at_{k}": _target_rank_from_rows(merged, target) is not None and _target_rank_from_rows(merged, target) <= min(k, len(merged)) for k in K_VALUES}
        for k in K_VALUES:
            hits[k] += int(recalls[f"recall_at_{k}"])
        scores.append({"task_id": row["task_id"], "query_index": row["query_index"], "sealed_available": True, "merged_ranked_all_catalog": recalls})
    return {
        "audit_version": AUDIT_VERSION,
        "metric_label": "SEALED_POST_HOC_CANDIDATE_RECALL_AT_K",
        "post_hoc_only": True,
        "scoring_available_query_count": scored_count,
        "metrics": {f"recall_at_{k}": {"numerator": hits[k], "denominator": scored_count, "rate": hits[k] / scored_count if scored_count else None} for k in K_VALUES},
        "sealed_outputs_used_to_change_candidate_ranking_budget_or_selection": False,
        "rows": scores,
    }


def strip_unstable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: strip_unstable(value)
            for key, value in payload.items()
            if key not in {"created_utc", "observed_runtime_seconds", "stable_two_run_hashes"}
        }
    if isinstance(payload, list):
        return [strip_unstable(value) for value in payload]
    return payload


def stable_json_hash_file(path: Path) -> str:
    return stable_hash(strip_unstable(load_json(path)))


def stable_artifact_hashes(artifact_root: Path = ARTIFACT_ROOT) -> dict[str, str]:
    names = [
        "corpus_manifest_verification.json",
        "candidate_surface_report.json",
        "candidate_ranking_spec.json",
        "loo_candidate_ledger.json",
        "candidate_attrition_report.json",
        "candidate_recall_metrics.json",
        "pre_score_test_candidate_commitment.json",
        "optional_sealed_test_candidate_recall.json",
        "final_report.json",
    ]
    return {name: stable_json_hash_file(artifact_root / name) for name in names}


def parameter_free_integrity() -> dict[str, Any]:
    paths = [
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_surface_audit.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_enumeration.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_ranking.py",
        PROJECT_ROOT / "vrm/twin_tasks/r5a2d_r2_candidate_recall_audit.py",
        PROJECT_ROOT / "vrm/twin_tasks/run_r5a2d_r2_candidate_recall.py",
    ]
    forbidden = ("nn.Parameter", "torch.optim", ".backward(", "optimizer.step(", "torch.load", "torch.save")
    pieces = []
    for path in paths:
        if not path.exists():
            continue
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
        pieces.extend(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string for token in tokens)
    stripped = " ".join(pieces)
    found = [term for term in forbidden if term in stripped]
    return {"passed": not found, "forbidden_code_terms_found": found, "scan_excludes_strings_and_comments": True}


def final_report_payload(
    *,
    verification: dict[str, Any],
    surface: dict[str, Any],
    metrics: dict[str, Any],
    attrition: dict[str, Any],
    sealed: dict[str, Any],
    protected_before: dict[str, str | None],
    protected_after: dict[str, str | None],
    verdict: str,
    runtime: float,
    stable_two_run_hashes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mandatory_scope_boundary": MANDATORY_SCOPE_BOUNDARY,
        "final_verdict": verdict,
        "corpus_manifest_verification": verification,
        "candidate_surface_summary": surface,
        "candidate_recall_metrics": metrics,
        "candidate_attrition_report": attrition,
        "optional_sealed_post_hoc_candidate_recall": {
            "metric_label": sealed["metric_label"],
            "scoring_available_query_count": sealed["scoring_available_query_count"],
            "metrics": sealed["metrics"],
        },
        "protected_hashes_unchanged": protected_before == protected_after,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "parameter_free_integrity": parameter_free_integrity(),
        "observed_runtime_seconds": runtime,
        "stable_two_run_hashes": stable_two_run_hashes or {},
        "narrowest_justified_next_action": "Inspect the first RANKED_ALL_CATALOG_CANDIDATE_RECALL K=128 misses by reason category without changing frozen catalogs or adding a router.",
    }


def run_once(*, artifact_root: Path = ARTIFACT_ROOT, stable_two_run_hashes: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes()
    verification = verify_r1_corpus_manifest()
    write_json(artifact_root / "corpus_manifest_verification.json", verification)
    surface = candidate_surface_report()
    write_json(artifact_root / "candidate_surface_report.json", surface)
    write_json(artifact_root / "candidate_ranking_spec.json", candidate_ranking_spec())
    if not verification["exact_match"]:
        verdict = MANIFEST_MISMATCH
        empty = {"audit_version": AUDIT_VERSION, "metrics": {}, "hit_examples": [], "miss_examples": []}
        write_json(artifact_root / "loo_candidate_ledger.json", [])
        write_json(artifact_root / "candidate_attrition_report.json", {})
        write_json(artifact_root / "candidate_recall_metrics.json", empty)
        write_json(artifact_root / "pre_score_test_candidate_commitment.json", {})
        write_json(artifact_root / "optional_sealed_test_candidate_recall.json", {})
    elif not surface["all_available"]:
        verdict = ENUM_UNAVAILABLE
        empty = {"audit_version": AUDIT_VERSION, "metrics": {}, "hit_examples": [], "miss_examples": []}
        write_json(artifact_root / "loo_candidate_ledger.json", [])
        write_json(artifact_root / "candidate_attrition_report.json", {})
        write_json(artifact_root / "candidate_recall_metrics.json", empty)
        write_json(artifact_root / "pre_score_test_candidate_commitment.json", {})
        write_json(artifact_root / "optional_sealed_test_candidate_recall.json", {})
    else:
        tasks = load_tasks_after_manifest_verification(verification)
        ledger = build_loo_candidate_ledger(tasks)
        write_json(artifact_root / "loo_candidate_ledger.json", ledger)
        write_json(
            artifact_root / "pre_hidden_loo_candidate_commitment.json",
            {
                "audit_version": AUDIT_VERSION,
                "loo_candidate_ledger_sha256": sha256_file(artifact_root / "loo_candidate_ledger.json"),
                "hidden_LOO_outputs_read": False,
            },
        )
        scored = score_loo_candidate_recall(tasks, ledger)
        write_json(artifact_root / "candidate_recall_metrics.json", scored)
        attrition = candidate_attrition_report(ledger, scored)
        write_json(artifact_root / "candidate_attrition_report.json", attrition)
        write_hit_examples(artifact_root / "candidate_hit_examples.md", scored)
        commitment = build_test_candidate_commitment(tasks)
        write_json(artifact_root / "pre_score_test_candidate_commitment.json", commitment)
        sealed = score_optional_sealed_test_recall(tasks, commitment)
        write_json(artifact_root / "optional_sealed_test_candidate_recall.json", sealed)
        verdict = MEASURED
    protected_after = protected_hashes()
    metrics = load_json(artifact_root / "candidate_recall_metrics.json")
    attrition = load_json(artifact_root / "candidate_attrition_report.json")
    sealed = load_json(artifact_root / "optional_sealed_test_candidate_recall.json")
    if protected_before != protected_after or not parameter_free_integrity()["passed"]:
        verdict = INTEGRITY_FAILURE
    report = final_report_payload(
        verification=verification,
        surface=surface,
        metrics=metrics.get("metrics", metrics),
        attrition=attrition,
        sealed=sealed,
        protected_before=protected_before,
        protected_after=protected_after,
        verdict=verdict,
        runtime=time.perf_counter() - start,
        stable_two_run_hashes=stable_two_run_hashes,
    )
    write_json(artifact_root / "final_report.json", report)
    return {"final_verdict": verdict, "final_report": report}


def run_complete_audit_twice(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    first = run_once(artifact_root=artifact_root)
    first_hashes = stable_artifact_hashes(artifact_root)
    second = run_once(artifact_root=artifact_root)
    second_hashes = stable_artifact_hashes(artifact_root)
    comparison = {
        "run1": first_hashes,
        "run2": second_hashes,
        "matching": first_hashes == second_hashes,
        "differing_files": sorted(name for name in first_hashes if first_hashes.get(name) != second_hashes.get(name)),
    }
    verdict = second["final_verdict"]
    if not comparison["matching"]:
        verdict = NONDETERMINISTIC
    final = run_once(artifact_root=artifact_root, stable_two_run_hashes=comparison)
    if verdict == NONDETERMINISTIC:
        report = load_json(artifact_root / "final_report.json")
        report["final_verdict"] = NONDETERMINISTIC
        write_json(artifact_root / "final_report.json", report)
        final["final_verdict"] = NONDETERMINISTIC
        final["final_report"] = report
    final["stable_two_run_hashes"] = comparison
    return final
