from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .r5a2d_r1_corpus_resolver import sha256_file
from .r5a3a_scene_hypotheses import Grid, canonical_grid_sha256, freeze_grid, stable_hash
from .r5a3b_object_graph import build_object_graph, graph_signature
from .r5a3b_parse_hypotheses import enumerate_parse_states
from .r5a3b_relation_programs import (
    MATCHED_RANDOM_SALTS,
    TOP_K,
    enumerate_relation_programs,
    target_rank,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
P0_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_relation_constraint_pilot"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r5a3b_p1_exposure_audit"
ARC_ROOT = PROJECT_ROOT / "arc-prize-2026-arc-agi-2"
PYTHON = Path("/home/quyen/.venvs/vrm/bin/python")
AUDIT_VERSION = "R5A3B_P1_RELATION_EXPOSURE_CONSTRAINT_EFFICACY_AUDIT"

P0_INPUTS = (
    "pilot_manifest.json",
    "pre_hidden_pilot_commitment.json",
    "raw_candidate_universe_ledger.json",
    "condition_candidate_ledgers.json",
    "constraint_acquisition_report.json",
    "matched_random_prune_report.json",
    "pilot_recall_metrics.json",
    "runtime_profile.json",
    "final_report.json",
)

CONDITION_NAMES = (
    "OBJECT_ONLY",
    "RELATION_NO_PRUNE",
    "RELATION_SOFT_CONSTRAINT",
    "RELATION_HARD_CONSTRAINT",
)

RELATION_EXPOSURE_CATEGORIES = (
    "NO_RELATION_PROGRAM_EMITTED",
    "RELATION_PROGRAM_EMITTED_NO_EXECUTABLE_OUTPUT",
    "RELATION_OUTPUT_DUPLICATES_OBJECT_ONLY",
    "RELATION_DISTINCT_OUTPUT_ONLY_LOW_RANK",
    "RELATION_DISTINCT_OUTPUT_EXPOSED_NO_TARGET",
    "RELATION_TARGET_CONTRIBUTOR",
)


@dataclass(frozen=True)
class FoldTarget:
    task_id: str
    fold_index: int
    target_grid: Grid
    target_hash: str
    query_grid: Grid


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


def run_canonical_environment_reverification() -> dict[str, Any]:
    test_path = PROJECT_ROOT / "tests/test_r5a3b_relation_constraint_pilot.py"
    version_cmd = [str(PYTHON), "-c", "import pytest, torch; print(pytest.__version__, torch.__version__)"]
    pytest_cmd = [str(PYTHON), "-m", "pytest", "-q", "tests/test_r5a3b_relation_constraint_pilot.py"]
    version = subprocess.run(version_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    test = subprocess.run(pytest_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    parts = version.stdout.strip().split()
    payload = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": now_utc(),
        "interpreter_path": str(PYTHON),
        "pytest_version": parts[0] if len(parts) >= 1 else None,
        "torch_version": parts[1] if len(parts) >= 2 else None,
        "version_command": " ".join(version_cmd),
        "version_stdout": version.stdout,
        "version_stderr": version.stderr,
        "version_returncode": version.returncode,
        "pytest_command": "cd /mnt/d/VRM && /home/quyen/.venvs/vrm/bin/python -m pytest -q tests/test_r5a3b_relation_constraint_pilot.py",
        "pytest_stdout": test.stdout,
        "pytest_stderr": test.stderr,
        "pytest_returncode": test.returncode,
        "passed": version.returncode == 0 and test.returncode == 0,
        "focused_test_file_sha256": sha256_file(test_path),
        "does_not_modify_p0_report": True,
    }
    payload["artifact_sha256"] = stable_hash(payload)
    return payload


def p0_input_integrity() -> dict[str, Any]:
    files = []
    missing = []
    for name in P0_INPUTS:
        path = P0_ARTIFACT_ROOT / name
        if path.exists():
            files.append({"name": name, "path": str(path), "sha256": sha256_file(path), "byte_size": path.stat().st_size})
        else:
            missing.append(name)
    payload = {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": now_utc(),
        "p0_artifact_root": str(P0_ARTIFACT_ROOT),
        "required_input_count": len(P0_INPUTS),
        "present_input_count": len(files),
        "missing_inputs": missing,
        "input_hashes": files,
        "read_only_posthoc_audit": True,
        "arc_test_outputs_read": False,
    }
    payload["passed"] = not missing
    payload["input_manifest_sha256"] = stable_hash(files)
    return payload


def load_p0_inputs() -> dict[str, Any]:
    return {name: load_json(P0_ARTIFACT_ROOT / name) for name in P0_INPUTS}


def load_targets_from_manifest(manifest: dict[str, Any]) -> dict[tuple[str, int], FoldTarget]:
    cache: dict[Path, dict[str, Any]] = {}
    targets: dict[tuple[str, int], FoldTarget] = {}
    for task_row in manifest["tasks"]:
        source_path, raw_task_id = task_row["source_relative_path"].split("#", 1)
        path = PROJECT_ROOT / source_path
        if path not in cache:
            cache[path] = json.loads(path.read_text(encoding="utf-8"))
        task_payload = cache[path][raw_task_id]
        for index, pair in enumerate(task_payload.get("train", [])):
            if "input" not in pair or "output" not in pair:
                continue
            target = freeze_grid(pair["output"])
            query = freeze_grid(pair["input"])
            targets[(task_row["task_id"], index)] = FoldTarget(
                task_id=task_row["task_id"],
                fold_index=index,
                target_grid=target,
                target_hash=canonical_grid_sha256(target),
                query_grid=query,
            )
    return targets


def candidate_matches_target(candidate: dict[str, Any], target: FoldTarget) -> bool:
    return candidate.get("output_hash") == target.target_hash and freeze_grid(candidate.get("output_grid", [])) == target.target_grid


def condition_runs(condition_payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    if name == "MATCHED_RANDOM_PRUNE":
        return condition_payload["runs"]
    return [condition_payload]


def rank_in_candidates(candidates: list[dict[str, Any]], target: FoldTarget) -> int | None:
    for index, candidate in enumerate(candidates, start=1):
        if candidate_matches_target(candidate, target):
            return index
    return None


def relation_and_object_hashes(raw_candidates: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    object_hashes = {candidate["output_hash"] for candidate in raw_candidates if candidate["program_family"] == "object"}
    relation_hashes = {candidate["output_hash"] for candidate in raw_candidates if candidate["program_family"] == "relation"}
    return object_hashes, relation_hashes


def relation_surface_for_fold(raw_row: dict[str, Any], no_prune_payload: dict[str, Any]) -> dict[str, Any]:
    raw_candidates = raw_row["raw_candidates"]
    object_candidates = [candidate for candidate in raw_candidates if candidate["program_family"] == "object"]
    relation_candidates = [candidate for candidate in raw_candidates if candidate["program_family"] == "relation"]
    object_hashes, relation_hashes = relation_and_object_hashes(raw_candidates)
    relation_only_hashes = relation_hashes - object_hashes
    rank_by_hash = {candidate["output_hash"]: index for index, candidate in enumerate(no_prune_payload["top_candidates"], start=1)}
    relation_ranks = [rank_by_hash[h] for h in relation_only_hashes if h in rank_by_hash]
    object_ranks = [rank_by_hash[h] for h in object_hashes if h in rank_by_hash]
    return {
        "fold_key": raw_row["fold_key"],
        "object_only_raw_candidate_count": len(object_candidates),
        "relation_raw_candidate_count": len(relation_candidates),
        "object_only_unique_output_count": len(object_hashes),
        "relation_unique_output_count": len(relation_hashes),
        "relation_only_unique_output_count": len(relation_only_hashes),
        "duplicate_relation_output_count": sum(candidate["output_hash"] in object_hashes for candidate in relation_candidates),
        "duplicate_relation_unique_output_count": len(relation_hashes & object_hashes),
        "best_relation_candidate_rank": min(relation_ranks) if relation_ranks else None,
        "best_object_only_candidate_rank": min(object_ranks) if object_ranks else None,
        "relation_only_output_hashes": sorted(relation_only_hashes),
    }


def generated_relation_program_metadata(target: FoldTarget) -> dict[str, Any]:
    parse_states = enumerate_parse_states(target.query_grid)
    rows = []
    graph_hashes = set()
    relation_parse_ids = []
    for parse_state in parse_states:
        graph = build_object_graph(parse_state)
        graph_hashes.add(graph_signature(graph))
        programs = enumerate_relation_programs(graph, tuple(range(10)), max_programs=8)
        if programs:
            relation_parse_ids.append(parse_state.parse_id)
        rows.append(
            {
                "parse_id": parse_state.parse_id,
                "object_count": parse_state.object_count,
                "graph_signature": graph_signature(graph),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "generated_relation_program_count": len(programs),
            }
        )
    return {
        "raw_parse_hypothesis_count": len(parse_states),
        "distinct_object_graph_count": len(graph_hashes),
        "generated_relation_program_count": sum(row["generated_relation_program_count"] for row in rows),
        "parse_hypotheses_with_generated_relation_programs": relation_parse_ids,
        "parse_rows": rows,
    }


def classify_relation_exposure(
    *,
    surface: dict[str, Any],
    no_prune_payload: dict[str, Any],
    target: FoldTarget,
    generated_relation_program_count: int,
) -> str:
    if generated_relation_program_count == 0:
        return "NO_RELATION_PROGRAM_EMITTED"
    if surface["relation_raw_candidate_count"] == 0:
        return "RELATION_PROGRAM_EMITTED_NO_EXECUTABLE_OUTPUT"
    if surface["relation_only_unique_output_count"] == 0:
        return "RELATION_OUTPUT_DUPLICATES_OBJECT_ONLY"
    relation_only_hashes = set(surface.get("relation_only_output_hashes", []))
    top16_hashes = {candidate["output_hash"] for candidate in no_prune_payload["top_candidates"][:16]}
    relation_top16 = bool(relation_only_hashes & top16_hashes)
    if not relation_top16:
        return "RELATION_DISTINCT_OUTPUT_ONLY_LOW_RANK"
    if any(candidate_matches_target(candidate, target) and candidate.get("program_family") == "relation" for candidate in no_prune_payload["top_candidates"]):
        return "RELATION_TARGET_CONTRIBUTOR"
    return "RELATION_DISTINCT_OUTPUT_EXPOSED_NO_TARGET"


def hard_prune_target_survival(raw_candidates: list[dict[str, Any]], hard_payload: dict[str, Any], target: FoldTarget) -> str:
    raw_target_ids = {candidate["candidate_id"] for candidate in raw_candidates if candidate_matches_target(candidate, target)}
    if not raw_target_ids:
        return "NO_TARGET_IN_RAW_UNIVERSE"
    hard_ids = set(hard_payload["candidate_ids"])
    survivors = raw_target_ids & hard_ids
    if len(survivors) == len(raw_target_ids):
        return "HARD_PRUNE_PRESERVED_ALL_TARGET_CANDIDATES"
    if survivors:
        return "HARD_PRUNE_PRESERVED_AT_LEAST_ONE_TARGET_CANDIDATE"
    return "HARD_PRUNE_REMOVED_ALL_TARGET_CANDIDATES"


def parse_hit_classification(target_parse_ids: set[str]) -> str | None:
    if not target_parse_ids:
        return None
    canonical = {parse_id for parse_id in target_parse_ids if parse_id.startswith("bg=0|conn=4")}
    if len(target_parse_ids) > 1:
        return "MULTIPLE_PARSE_DUPLICATE_HIT"
    if canonical:
        return "CANONICAL_PARSE_HIT"
    return "NONCANONICAL_PARSE_HIT"


def recall_at_k_with_short_candidate_list(rank: int | None, candidate_count: int, k: int) -> bool:
    return rank is not None and rank <= min(k, candidate_count)


def ledger_completeness_report(raw_ledger: list[dict[str, Any]], condition_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    required_candidate_fields = {
        "parse_hypothesis_id",
        "source_program_ids",
        "operator_sequence",
        "program_family",
        "relation_predicates_used",
        "raw_ordinal",
        "output_hash",
        "support_shape_mismatch_count",
        "support_cell_hamming_loss",
        "support_failed_pair_count",
    }
    missing = []
    for row in raw_ledger:
        for candidate in row["raw_candidates"]:
            absent = sorted(required_candidate_fields - set(candidate))
            if absent:
                missing.append({"fold_key": row["fold_key"], "candidate_id": candidate.get("candidate_id"), "missing": absent})
    condition_missing = []
    for row in condition_ledger:
        for condition in CONDITION_NAMES:
            payload = row["conditions"][condition]
            for index, candidate in enumerate(payload["top_candidates"], start=1):
                absent = sorted(required_candidate_fields - set(candidate))
                if absent:
                    condition_missing.append({"fold_key": row["fold_key"], "condition": condition, "post_prune_ordinal": index, "missing": absent})
    payload = {
        "audit_version": AUDIT_VERSION,
        "raw_candidate_fields_checked": sorted(required_candidate_fields),
        "raw_candidate_missing_field_records": missing[:100],
        "condition_candidate_missing_field_records": condition_missing[:100],
        "hidden_target_flags_present_in_p0": False,
        "hidden_target_flags_computed_posthoc_in_p1": True,
        "generated_program_count_present_in_p0": False,
        "generated_program_count_recomputed_without_candidate_regeneration": True,
        "candidate_regeneration_required": False,
        "complete_for_candidate_provenance_audit": not missing and not condition_missing,
    }
    payload["report_sha256"] = stable_hash(payload)
    return payload


def _recall_summary(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored_rows:
        return {f"Recall@{k}": {"numerator": 0, "denominator": 0, "rate": 0.0} for k in TOP_K}
    return {
        f"Recall@{k}": {
            "numerator": sum(recall_at_k_with_short_candidate_list(row["target_rank"], row["candidate_count"], k) for row in scored_rows),
            "denominator": len(scored_rows),
            "rate": sum(recall_at_k_with_short_candidate_list(row["target_rank"], row["candidate_count"], k) for row in scored_rows) / len(scored_rows),
        }
        for k in TOP_K
    }


def _rank_stats(ranks: list[int]) -> dict[str, Any]:
    return {
        "mean_target_rank_when_target_exists": sum(ranks) / len(ranks) if ranks else None,
        "median_target_rank_when_target_exists": statistics.median(ranks) if ranks else None,
    }


def analyze_p0_ledgers(inputs: dict[str, Any]) -> dict[str, Any]:
    manifest = inputs["pilot_manifest.json"]
    raw_ledger = inputs["raw_candidate_universe_ledger.json"]
    condition_ledger = inputs["condition_candidate_ledgers.json"]
    constraint_report = inputs["constraint_acquisition_report.json"]
    runtime = inputs["runtime_profile.json"]
    targets = load_targets_from_manifest(manifest)
    raw_by_fold = {row["fold_key"]: row for row in raw_ledger}
    constraints_by_fold = {row["fold_key"]: row for row in constraint_report["rows"]}
    surface_rows = []
    exposure_rows = []
    hit_rows = []
    parse_rows = []
    hard_survival_rows = []
    condition_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    random_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    constraint_prune_counts: Counter[str] = Counter()
    pruned_examples: dict[str, dict[str, Any]] = {}
    eligibility_rows = []
    for condition_row in condition_ledger:
        raw_row = raw_by_fold[condition_row["fold_key"]]
        target = targets[(condition_row["task_id_for_log_only"], condition_row["fold_index"])]
        no_prune = condition_row["conditions"]["RELATION_NO_PRUNE"]
        surface = relation_surface_for_fold(raw_row, no_prune)
        metadata = generated_relation_program_metadata(target)
        category = classify_relation_exposure(
            surface=surface,
            no_prune_payload=no_prune,
            target=target,
            generated_relation_program_count=metadata["generated_relation_program_count"],
        )
        surface_rows.append(surface)
        exposure_rows.append(
            {
                "fold_key": condition_row["fold_key"],
                "task_id_for_log_only": condition_row["task_id_for_log_only"],
                "fold_index": condition_row["fold_index"],
                "signature": condition_row["signature"],
                "category": category,
                "generated_relation_program_count": metadata["generated_relation_program_count"],
                **surface,
            }
        )
        target_parse_ids = {
            candidate["parse_hypothesis_id"]
            for candidate in raw_row["raw_candidates"]
            if candidate_matches_target(candidate, target)
        }
        relation_parse_ids = {
            candidate["parse_hypothesis_id"]
            for candidate in raw_row["raw_candidates"]
            if candidate["program_family"] == "relation"
        }
        object_parse_ids = {
            candidate["parse_hypothesis_id"]
            for candidate in raw_row["raw_candidates"]
            if candidate["program_family"] == "object"
        }
        relation_unique_hashes = {
            candidate["output_hash"]
            for candidate in raw_row["raw_candidates"]
            if candidate["program_family"] == "relation"
        } - {
            candidate["output_hash"]
            for candidate in raw_row["raw_candidates"]
            if candidate["program_family"] == "object"
        }
        parse_rows.append(
            {
                "fold_key": condition_row["fold_key"],
                "number_of_raw_parse_hypotheses": metadata["raw_parse_hypothesis_count"],
                "number_of_distinct_object_graphs": metadata["distinct_object_graph_count"],
                "number_of_parse_hypotheses_emitting_object_only_candidates": len(object_parse_ids),
                "number_of_parse_hypotheses_emitting_relation_candidates": len(relation_parse_ids),
                "parse_hypothesis_ids_producing_unique_relation_outputs": sorted(
                    {
                        candidate["parse_hypothesis_id"]
                        for candidate in raw_row["raw_candidates"]
                        if candidate["program_family"] == "relation" and candidate["output_hash"] in relation_unique_hashes
                    }
                ),
                "parse_hypothesis_ids_producing_hidden_target_hits": sorted(target_parse_ids),
                "hit_classification": parse_hit_classification(target_parse_ids),
                "metadata_recomputed_without_candidate_regeneration": True,
            }
        )
        hard_behavior = hard_prune_target_survival(raw_row["raw_candidates"], condition_row["conditions"]["RELATION_HARD_CONSTRAINT"], target)
        random_behavior = {}
        raw_target_ids = {candidate["candidate_id"] for candidate in raw_row["raw_candidates"] if candidate_matches_target(candidate, target)}
        for run in condition_row["conditions"]["MATCHED_RANDOM_PRUNE"]["runs"]:
            survivors = raw_target_ids & set(run["candidate_ids"])
            random_behavior[run["salt"]] = {
                "target_survived": bool(survivors),
                "surviving_target_candidate_count": len(survivors),
            }
        hard_survival_rows.append(
            {
                "fold_key": condition_row["fold_key"],
                "behavior": hard_behavior,
                "raw_target_candidate_count": len(raw_target_ids),
                "random_prune_by_salt": random_behavior,
            }
        )
        for condition in CONDITION_NAMES:
            payload = condition_row["conditions"][condition]
            rank = rank_in_candidates(payload["top_candidates"], target)
            condition_scores[condition].append(
                {
                    "fold_key": condition_row["fold_key"],
                    "signature": condition_row["signature"],
                    "target_rank": rank,
                    "candidate_count": len(payload["top_candidates"]),
                    "before_prune_count": raw_row["raw_candidate_count"],
                    "after_prune_count": payload["unique_candidate_count"],
                }
            )
        for run in condition_row["conditions"]["MATCHED_RANDOM_PRUNE"]["runs"]:
            rank = rank_in_candidates(run["top_candidates"], target)
            random_scores[run["salt"]].append(
                {
                    "fold_key": condition_row["fold_key"],
                    "signature": condition_row["signature"],
                    "target_rank": rank,
                    "candidate_count": len(run["top_candidates"]),
                    "before_prune_count": raw_row["raw_candidate_count"],
                    "after_prune_count": run["unique_candidate_count"],
                }
            )
        target_conditions = {}
        for condition in CONDITION_NAMES:
            payload = condition_row["conditions"][condition]
            rank = rank_in_candidates(payload["top_candidates"], target)
            if rank is not None:
                target_conditions[condition] = rank
        random_survival = {}
        for run in condition_row["conditions"]["MATCHED_RANDOM_PRUNE"]["runs"]:
            random_survival[run["salt"]] = rank_in_candidates(run["top_candidates"], target)
        raw_hits = [candidate for candidate in raw_row["raw_candidates"] if candidate_matches_target(candidate, target)]
        if raw_hits:
            object_hashes, relation_hashes = relation_and_object_hashes(raw_row["raw_candidates"])
            for candidate in raw_hits:
                hit_rows.append(
                    {
                        "hit_fold_id": condition_row["fold_key"],
                        "task_id_for_log_only": condition_row["task_id_for_log_only"],
                        "fold_index": condition_row["fold_index"],
                        "conditions_where_target_exists": target_conditions,
                        "source_type": "relation-conditioned" if candidate["program_family"] == "relation" else "object-only",
                        "object_only_could_generate_same_output": candidate["output_hash"] in object_hashes,
                        "relation_conditioned_could_generate_same_output": candidate["output_hash"] in relation_hashes,
                        "relation_only_unique_target": candidate["output_hash"] in (relation_hashes - object_hashes),
                        "parse_hypothesis_id": candidate["parse_hypothesis_id"],
                        "program_id": candidate["source_program_ids"][0] if candidate["source_program_ids"] else None,
                        "operator_sequence": candidate["operator_sequence"],
                        "relation_predicates": candidate["relation_predicates_used"],
                        "raw_ordinal": candidate["raw_ordinal"],
                        "target_rank_before_pruning": target_conditions.get("RELATION_NO_PRUNE"),
                        "target_rank_after_soft_constraint": target_conditions.get("RELATION_SOFT_CONSTRAINT"),
                        "target_rank_after_hard_constraint": target_conditions.get("RELATION_HARD_CONSTRAINT"),
                        "target_survival_under_random_prune_salts": random_survival,
                    }
                )
        evaluations = condition_row["conditions"]["RELATION_HARD_CONSTRAINT"]["constraint_evaluations"]
        candidates_by_id = {candidate["candidate_id"]: candidate for candidate in raw_row["raw_candidates"]}
        for candidate_id, evaluation in evaluations.items():
            for violation in evaluation.get("hard_violations", []):
                predicate = violation["predicate"]
                constraint_prune_counts[predicate] += 1
                if predicate not in pruned_examples:
                    candidate = candidates_by_id.get(candidate_id)
                    pruned_examples[predicate] = {
                        "fold_key": condition_row["fold_key"],
                        "visible_support_evidence": constraints_by_fold[condition_row["fold_key"]]["models"],
                        "predicate_definition": predicate,
                        "why_it_passed_hard_constraint_eligibility": "Recorded hard=true in the frozen P0 constraint model for this parse.",
                        "candidate_that_was_pruned": candidate,
                        "pruned_output_matched_held_out_target": candidate_matches_target(candidate, target) if candidate else False,
                    }
        fold_constraints = []
        for model in constraints_by_fold[condition_row["fold_key"]]["models"].values():
            fold_constraints.extend(model["constraints"])
        eligibility_rows.append(
            {
                "fold_key": condition_row["fold_key"],
                "signature": condition_row["signature"],
                "candidate_independent_predicates_considered": len(fold_constraints),
                "rejected_for_ambiguous_correspondence": sum("ambiguous_correspondence_present" in c.get("hard_rejection_reasons", []) for c in fold_constraints),
                "rejected_for_non_universality": sum("not_universal_on_visible_supports" in c.get("hard_rejection_reasons", []) for c in fold_constraints),
                "rejected_for_non_contrastiveness": sum("not_contrastive_on_visible_supports" in c.get("hard_rejection_reasons", []) for c in fold_constraints),
                "rejected_for_non_mechanical_detectability": sum("not_mechanically_detectable" in c.get("hard_rejection_reasons", []) for c in fold_constraints),
                "retained_as_soft_signals": sum(not c.get("hard", False) for c in fold_constraints),
                "promoted_to_hard_constraints": sum(c.get("hard", False) for c in fold_constraints),
            }
        )
    return {
        "targets": targets,
        "surface_rows": surface_rows,
        "exposure_rows": exposure_rows,
        "hit_rows": hit_rows,
        "parse_rows": parse_rows,
        "hard_survival_rows": hard_survival_rows,
        "condition_scores": condition_scores,
        "random_scores": random_scores,
        "constraint_prune_counts": constraint_prune_counts,
        "pruned_examples": pruned_examples,
        "eligibility_rows": eligibility_rows,
        "runtime": runtime,
    }


def taxonomy_report(exposure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(row["category"] for row in rows)
        return {
            "fold_count": len(rows),
            "counts": {category: counts.get(category, 0) for category in RELATION_EXPOSURE_CATEGORIES},
            "rates": {category: counts.get(category, 0) / len(rows) if rows else 0.0 for category in RELATION_EXPOSURE_CATEGORIES},
        }

    return {
        "audit_version": AUDIT_VERSION,
        "taxonomy": list(RELATION_EXPOSURE_CATEGORIES),
        "all_folds": bucket(exposure_rows),
        "multi_object_folds": bucket([row for row in exposure_rows if row["signature"]["object_slice"] == "multi_object"]),
        "one_object_folds": bucket([row for row in exposure_rows if row["signature"]["object_slice"] == "one_object"]),
        "rows": exposure_rows,
    }


def surface_report(surface_rows: list[dict[str, Any]], hit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(field: str) -> float:
        values = [row[field] for row in surface_rows if row[field] is not None]
        return sum(values) / len(values) if values else 0.0

    return {
        "audit_version": AUDIT_VERSION,
        "fold_count": len(surface_rows),
        "aggregate_means": {
            "object_only_raw_candidate_count": avg("object_only_raw_candidate_count"),
            "relation_raw_candidate_count": avg("relation_raw_candidate_count"),
            "object_only_unique_output_count": avg("object_only_unique_output_count"),
            "relation_unique_output_count": avg("relation_unique_output_count"),
            "relation_only_unique_output_count": avg("relation_only_unique_output_count"),
            "duplicate_relation_output_count": avg("duplicate_relation_output_count"),
        },
        "rows": surface_rows,
        "did_either_exact_hit_originate_from_relation_conditioned_candidate_object_only_could_not_generate": any(
            row["relation_only_unique_target"] for row in hit_rows
        ),
    }


def hit_provenance_report(hit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "raw_target_candidate_record_count": len(hit_rows),
        "hit_fold_count": len({row["hit_fold_id"] for row in hit_rows}),
        "relation_conditioned_hit_record_count": sum(row["source_type"] == "relation-conditioned" for row in hit_rows),
        "relation_only_unique_hit_record_count": sum(row["relation_only_unique_target"] for row in hit_rows),
        "explicit_answer": "No"
        if not any(row["relation_only_unique_target"] for row in hit_rows)
        else "Yes",
        "rows": hit_rows,
    }


def constraint_efficacy_report(
    condition_scores: dict[str, list[dict[str, Any]]],
    random_scores: dict[str, list[dict[str, Any]]],
    hard_survival_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_summary = {}
    for condition, rows in condition_scores.items():
        ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
        condition_summary[condition] = {
            **_recall_summary(rows),
            **_rank_stats(ranks),
            "target_survival_rate_after_pruning": sum(row["target_rank"] is not None for row in rows) / len(rows) if rows else 0.0,
            "mean_candidate_count_before_pruning": sum(row["before_prune_count"] for row in rows) / len(rows) if rows else 0.0,
            "mean_candidate_count_after_pruning": sum(row["after_prune_count"] for row in rows) / len(rows) if rows else 0.0,
            "mean_unique_output_count_after_pruning": sum(row["candidate_count"] for row in rows) / len(rows) if rows else 0.0,
        }
    random_summary = {}
    for salt, rows in random_scores.items():
        ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
        random_summary[salt] = {
            **_recall_summary(rows),
            **_rank_stats(ranks),
            "target_survival_rate_after_pruning": sum(row["target_rank"] is not None for row in rows) / len(rows) if rows else 0.0,
            "mean_candidate_count_after_pruning": sum(row["after_prune_count"] for row in rows) / len(rows) if rows else 0.0,
        }
    random_mean = {
        f"Recall@{k}": statistics.mean(summary[f"Recall@{k}"]["rate"] for summary in random_summary.values())
        if random_summary
        else 0.0
        for k in (1, 4, 16)
    }
    hard = condition_summary["RELATION_HARD_CONSTRAINT"]
    no_prune = condition_summary["RELATION_NO_PRUNE"]
    deltas = {
        "delta_R@1_hard_vs_no_prune": hard["Recall@1"]["rate"] - no_prune["Recall@1"]["rate"],
        "delta_R@4_hard_vs_no_prune": hard["Recall@4"]["rate"] - no_prune["Recall@4"]["rate"],
        "delta_R@16_hard_vs_no_prune": hard["Recall@16"]["rate"] - no_prune["Recall@16"]["rate"],
        "delta_R@1_hard_vs_random": hard["Recall@1"]["rate"] - random_mean["Recall@1"],
        "delta_R@4_hard_vs_random": hard["Recall@4"]["rate"] - random_mean["Recall@4"],
        "delta_R@16_hard_vs_random": hard["Recall@16"]["rate"] - random_mean["Recall@16"],
    }
    return {
        "audit_version": AUDIT_VERSION,
        "primary_pruning_metric_note": "Raw candidate cap is <=32, so low-K recall and target survival are primary for pruning exposure.",
        "conditions": condition_summary,
        "matched_random_by_salt": random_summary,
        "deltas": deltas,
        "hard_prune_target_behavior_counts": dict(Counter(row["behavior"] for row in hard_survival_rows)),
        "hard_prune_target_behavior_rows": hard_survival_rows,
    }


def constraint_eligibility_report(
    eligibility_rows: list[dict[str, Any]],
    prune_counts: Counter[str],
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
        keys = [
            "candidate_independent_predicates_considered",
            "rejected_for_ambiguous_correspondence",
            "rejected_for_non_universality",
            "rejected_for_non_contrastiveness",
            "rejected_for_non_mechanical_detectability",
            "retained_as_soft_signals",
            "promoted_to_hard_constraints",
        ]
        return {key: sum(row[key] for row in rows) for key in keys}

    return {
        "audit_version": AUDIT_VERSION,
        "all_folds": aggregate(eligibility_rows),
        "multi_object_folds": aggregate([row for row in eligibility_rows if row["signature"]["object_slice"] == "multi_object"]),
        "one_object_folds": aggregate([row for row in eligibility_rows if row["signature"]["object_slice"] == "one_object"]),
        "candidates_pruned_by_hard_constraint": dict(sorted(prune_counts.items())),
        "readable_examples": examples,
        "rows": eligibility_rows,
    }


def parse_exposure_report(parse_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audit_version": AUDIT_VERSION,
        "fold_count": len(parse_rows),
        "mean_raw_parse_hypotheses": sum(row["number_of_raw_parse_hypotheses"] for row in parse_rows) / len(parse_rows) if parse_rows else 0.0,
        "mean_distinct_object_graphs": sum(row["number_of_distinct_object_graphs"] for row in parse_rows) / len(parse_rows) if parse_rows else 0.0,
        "hit_classification_counts": dict(Counter(row["hit_classification"] or "NO_HIT" for row in parse_rows)),
        "did_any_target_hit_require_noncanonical_parse_hypothesis": any(row["hit_classification"] == "NONCANONICAL_PARSE_HIT" for row in parse_rows),
        "rows": parse_rows,
    }


def runtime_exposure_report(runtime: dict[str, Any]) -> dict[str, Any]:
    stage = runtime["stage_stats"]
    pruning_mean = stage.get("pruning_dedup_hash_seconds", {}).get("mean") or 0.0
    graph_mean = stage.get("graph_construction_seconds", {}).get("mean") or 0.0
    constraint_mean = stage.get("constraint_acquisition_seconds", {}).get("mean") or 0.0
    return {
        "audit_version": AUDIT_VERSION,
        "source": "existing P0 runtime_profile.json; pilot not rerun",
        "stage_stats": stage,
        "marginal_runtime_costs_descriptive_only": {
            "relation_graph_versus_object_only_seconds_mean": graph_mean,
            "hard_constraints_versus_relation_no_prune_seconds_mean": constraint_mean + pruning_mean,
            "random_pruning_versus_hard_pruning_seconds_mean": pruning_mean,
        },
        "condition_specific_runtime_not_in_p0_ledger": True,
        "slowest_10_folds": runtime.get("slowest_10_folds", []),
    }


def representative_cards(exposure: dict[str, Any], hits: dict[str, Any], efficacy: dict[str, Any]) -> str:
    lines = ["# R5A-3B-P1 Representative Exposure Case Cards", ""]
    rows = exposure["rows"]
    for category in RELATION_EXPOSURE_CATEGORIES:
        match = next((row for row in rows if row["category"] == category), None)
        if not match:
            continue
        lines.append(f"## {category}")
        lines.append(f"- fold: {match['fold_key']}")
        lines.append(f"- slice: {match['signature']['object_slice']}")
        lines.append(f"- relation raw candidates: {match['relation_raw_candidate_count']}")
        lines.append(f"- relation-only unique outputs: {match['relation_only_unique_output_count']}")
        lines.append(f"- best relation rank: {match['best_relation_candidate_rank']}")
        lines.append("")
    lines.append("## Hit Summary")
    lines.append(f"- raw target candidate records: {hits['raw_target_candidate_record_count']}")
    lines.append(f"- exact hit folds: {hits['hit_fold_count']}")
    lines.append(f"- relation-only exact hits: {hits['relation_only_unique_hit_record_count']}")
    lines.append("")
    lines.append("## Low-K Hard-Prune Deltas")
    for key, value in efficacy["deltas"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def determinism_verification(artifact_root: Path) -> dict[str, Any]:
    names = [
        "ledger_completeness_report.json",
        "candidate_regeneration_verification.json",
        "relation_exposure_taxonomy.json",
        "object_vs_relation_surface_report.json",
        "hit_provenance_report.json",
        "constraint_efficacy_report.json",
        "constraint_eligibility_report.json",
        "parse_exposure_report.json",
        "runtime_exposure_report.json",
        "final_report.json",
    ]
    hashes = {name: sha256_file(artifact_root / name) for name in names if (artifact_root / name).exists()}
    return {
        "audit_version": AUDIT_VERSION,
        "artifact_hashes": hashes,
        "deterministic_reader_no_candidate_generation": True,
        "candidate_regeneration_performed": False,
    }


def final_report(
    taxonomy: dict[str, Any],
    surface: dict[str, Any],
    hits: dict[str, Any],
    efficacy: dict[str, Any],
    eligibility: dict[str, Any],
    parse: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    hard_removed = efficacy["hard_prune_target_behavior_counts"].get("HARD_PRUNE_REMOVED_ALL_TARGET_CANDIDATES", 0)
    exposed_no_target = taxonomy["all_folds"]["counts"].get("RELATION_DISTINCT_OUTPUT_EXPOSED_NO_TARGET", 0)
    relation_only_hits = hits["relation_only_unique_hit_record_count"]
    if hard_removed:
        action = "B. Repair a demonstrated hard-constraint eligibility bug, then rerun P0."
    elif relation_only_hits == 0 and exposed_no_target >= taxonomy["all_folds"]["fold_count"] / 2:
        action = "D. Stop this relation-constraint v0 because it was adequately exposed and added no target contribution."
    else:
        action = "A. Repair candidate allocation/order only; do not add grammar."
    return {
        "Measured facts": {
            "fold_count": taxonomy["all_folds"]["fold_count"],
            "relation_exposure_counts": taxonomy["all_folds"]["counts"],
            "mean_relation_raw_candidate_count": surface["aggregate_means"]["relation_raw_candidate_count"],
            "mean_relation_only_unique_output_count": surface["aggregate_means"]["relation_only_unique_output_count"],
            "raw_target_candidate_record_count": hits["raw_target_candidate_record_count"],
            "relation_only_unique_hit_record_count": relation_only_hits,
            "hard_prune_target_behavior_counts": efficacy["hard_prune_target_behavior_counts"],
            "low_k_constraint_deltas": efficacy["deltas"],
            "constraint_eligibility_totals": eligibility["all_folds"],
            "parse_hit_classification_counts": parse["hit_classification_counts"],
            "runtime_stage_stats": runtime["stage_stats"],
        },
        "What P1 establishes": [
            "Relation candidates were emitted and distinct relation outputs were exposed on a measurable subset of P0 folds."
            if taxonomy["all_folds"]["counts"].get("RELATION_DISTINCT_OUTPUT_EXPOSED_NO_TARGET", 0)
            else "Relation candidates had limited or absent exposure in the retained P0 candidate surface.",
            "No exact target hit was a relation-only unique output."
            if relation_only_hits == 0
            else "At least one exact target hit was a relation-only unique output.",
            "Hard pruning target survival is reported at low K and target-containing raw-universe granularity.",
            "Multiple parse hypotheses did not require a noncanonical parse hit."
            if not parse["did_any_target_hit_require_noncanonical_parse_hypothesis"]
            else "At least one target hit required a noncanonical parse hypothesis.",
        ],
        "What P1 does not establish": [
            "It does not prove relation graphs are globally useless.",
            "It does not prove relation graphs solve multi-object ARC.",
            "It is not a full-corpus result.",
            "It does not establish a need for TTT or LLM control.",
            "It does not justify adding a new operator.",
        ],
        "Narrowest justified next action": action,
    }


def run_audit_once(*, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    env = run_canonical_environment_reverification()
    integrity = p0_input_integrity()
    write_json(artifact_root / "p0_canonical_environment_reverification.json", env)
    write_json(artifact_root / "p0_input_integrity_verification.json", integrity)
    inputs = load_p0_inputs()
    completeness = ledger_completeness_report(inputs["raw_candidate_universe_ledger.json"], inputs["condition_candidate_ledgers.json"])
    write_json(artifact_root / "ledger_completeness_report.json", completeness)
    regeneration = {
        "audit_version": AUDIT_VERSION,
        "status": "NOT_REQUIRED",
        "reason": "P0 retained all candidate output hashes and provenance fields needed for candidate-surface analysis. P1 recomputed only parse/object-graph/relation-program-count metadata and did not regenerate candidate grids.",
        "candidate_output_hashes_compared": 0,
        "p0_artifacts_overwritten": False,
    }
    write_json(artifact_root / "candidate_regeneration_verification.json", regeneration)
    analysis = analyze_p0_ledgers(inputs)
    taxonomy = taxonomy_report(analysis["exposure_rows"])
    surface = surface_report(analysis["surface_rows"], analysis["hit_rows"])
    hits = hit_provenance_report(analysis["hit_rows"])
    efficacy = constraint_efficacy_report(analysis["condition_scores"], analysis["random_scores"], analysis["hard_survival_rows"])
    eligibility = constraint_eligibility_report(analysis["eligibility_rows"], analysis["constraint_prune_counts"], analysis["pruned_examples"])
    parse = parse_exposure_report(analysis["parse_rows"])
    runtime = runtime_exposure_report(analysis["runtime"])
    final = final_report(taxonomy, surface, hits, efficacy, eligibility, parse, runtime)
    write_json(artifact_root / "relation_exposure_taxonomy.json", taxonomy)
    write_json(artifact_root / "object_vs_relation_surface_report.json", surface)
    write_json(artifact_root / "hit_provenance_report.json", hits)
    write_json(artifact_root / "constraint_efficacy_report.json", efficacy)
    write_json(artifact_root / "constraint_eligibility_report.json", eligibility)
    write_json(artifact_root / "parse_exposure_report.json", parse)
    write_json(artifact_root / "runtime_exposure_report.json", runtime)
    write_text(artifact_root / "representative_exposure_case_cards.md", representative_cards(taxonomy, hits, efficacy))
    write_json(artifact_root / "final_report.json", final)
    write_json(artifact_root / "determinism_verification.json", determinism_verification(artifact_root))
    return final


def main() -> dict[str, Any]:
    return run_audit_once(artifact_root=ARTIFACT_ROOT)
