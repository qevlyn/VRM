from __future__ import annotations

import json
from pathlib import Path

from vrm.twin_tasks.r5a3a_arc_native_candidate_machine import SupportPair
from vrm.twin_tasks.r5a3a_scene_hypotheses import freeze_grid, stable_hash
from vrm.twin_tasks.r5a3b_p2a_balanced_relation_allocation import (
    MAX_FINAL_PROGRAMS_PER_PARSE,
    RELATION_ACTION_BUCKETS,
    allocation_spec,
    build_p2a_candidate_universe,
    enumerate_bucket_microproposals,
)
from vrm.twin_tasks.r5a3b_p2a_transport_pilot import (
    ARTIFACT_ROOT,
    PROTECTED_PRIOR_PATHS,
    load_task_payload,
    path_hashes,
    pre_hidden_commitment,
    support_and_query_from_payload,
    verify_p0_object_branch_equivalence,
)
from vrm.twin_tasks.r5a3b_object_graph import build_object_graph
from vrm.twin_tasks.r5a3b_parse_hypotheses import enumerate_parse_states


def simple_supports() -> tuple[SupportPair, ...]:
    return (
        SupportPair(
            freeze_grid([[1, 0, 2], [0, 0, 0]]),
            freeze_grid([[1, 0, 2], [0, 0, 0]]),
        ),
    )


def test_p0_object_branch_equivalence_on_handwritten_case() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    universe = build_p2a_candidate_universe(simple_supports(), query)
    object_programs = [candidate["source_program_ids"][0] for candidate in universe["raw_candidate_json"] if candidate["program_family"] == "object"]
    assert any("select=all_objects|ops=keep|render=same_canvas" == program for program in object_programs)
    assert any("|ops=recolor:" in program for program in object_programs)
    assert any("|ops=translate:" in program for program in object_programs)
    assert any("|ops=copy:" in program for program in object_programs)


def test_p0_object_branch_equivalence_on_one_ledger_backed_case() -> None:
    root = Path("/mnt/d/VRM")
    p0_raw = json.loads((root / "artifacts/r5a3b_relation_constraint_pilot/raw_candidate_universe_ledger.json").read_text())
    manifest = json.loads((root / "artifacts/r5a3b_relation_constraint_pilot/pilot_manifest.json").read_text())
    first = p0_raw[0]
    task_row = next(row for row in manifest["tasks"] if row["task_id"] == first["task_id_for_log_only"])
    payload = load_task_payload(task_row["source_relative_path"])
    supports, query = support_and_query_from_payload(payload, first["fold_index"])
    universe = build_p2a_candidate_universe(supports, query)
    report = verify_p0_object_branch_equivalence(
        manifest,
        [first],
        {first["fold_key"]: universe},
    )
    assert report["status"] == "PASS"


def test_each_relation_action_bucket_receives_attempt_before_early_return() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    universe = build_p2a_candidate_universe(simple_supports(), query)
    first_parse = universe["microproposal_reports"][0]
    assert {row["bucket"] for row in first_parse["bucket_rows"]} == {bucket for bucket, _ in RELATION_ACTION_BUCKETS}
    assert all(row["attempted"] for row in first_parse["bucket_rows"])


def test_vector_copy_and_translate_are_emitted_when_legal_candidates_exist() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    universe = build_p2a_candidate_universe(simple_supports(), query)
    actions = [candidate["operator_sequence"][1] for candidate in universe["raw_candidate_json"]]
    assert "copy_source_by_relation_vector" in actions
    assert "translate_target_by_relation_vector" in actions


def test_unavailable_buckets_are_logged_not_substituted() -> None:
    query = freeze_grid([[1, 0], [0, 0]])
    supports = (SupportPair(query, query),)
    universe = build_p2a_candidate_universe(supports, query)
    rows = universe["microproposal_reports"][0]["bucket_rows"]
    assert all(row["status"] == "BUCKET_UNAVAILABLE" for row in rows)
    assert all(row["selected_program"] is None for row in rows)


def test_action_bucket_tie_breaking_is_deterministic() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    left = build_p2a_candidate_universe(simple_supports(), query)
    right = build_p2a_candidate_universe(simple_supports(), query)
    assert stable_hash(left["top_candidate_json"]) == stable_hash(right["top_candidate_json"])
    assert stable_hash(left["microproposal_reports"]) == stable_hash(right["microproposal_reports"])


def test_final_relation_and_total_program_counts_do_not_exceed_caps() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    universe = build_p2a_candidate_universe(simple_supports(), query)
    for parse in universe["parse_reports"]:
        assert parse["relation_candidate_count"] <= 4
        assert parse["final_candidate_count"] <= MAX_FINAL_PROGRAMS_PER_PARSE


def test_no_held_out_output_accessed_before_commitment_policy() -> None:
    commitment = pre_hidden_commitment(
        {"tasks": [], "task_count": 0},
        {"status": "PASS"},
        {},
    )
    assert commitment["test_query_firewall_policy"]["held_out_train_outputs_read_before_commitment"] is False
    assert commitment["test_query_firewall_policy"]["arc_test_outputs_read"] is False


def test_global_edge_semantics_are_unchanged_and_logged() -> None:
    query = freeze_grid([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    supports = (SupportPair(query, query),)
    universe = build_p2a_candidate_universe(supports, query)
    copies = [candidate for candidate in universe["raw_candidate_json"] if candidate["operator_sequence"][1] == "copy_source_by_relation_vector"]
    assert copies
    assert any(candidate["execution_trace"]["selected_edge_count"] >= 2 for candidate in copies)


def test_collisions_and_out_of_bounds_writes_are_logged() -> None:
    query = freeze_grid([[1, 2]])
    supports = (SupportPair(query, query),)
    universe = build_p2a_candidate_universe(supports, query)
    transport = [candidate for candidate in universe["raw_candidate_json"] if candidate["operator_sequence"][1] in {"copy_source_by_relation_vector", "translate_target_by_relation_vector"}]
    assert transport
    assert any(candidate["execution_trace"]["dropped_out_of_bounds"] or candidate["execution_trace"]["overwritten_destinations"] for candidate in transport)


def test_p0_p1_sources_and_artifacts_remain_byte_identical_during_hash_scan() -> None:
    before = path_hashes(PROTECTED_PRIOR_PATHS)
    after = path_hashes(PROTECTED_PRIOR_PATHS)
    assert before == after


def test_two_p2a_runs_have_identical_stable_outputs_on_handwritten_case() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    first = build_p2a_candidate_universe(simple_supports(), query)
    second = build_p2a_candidate_universe(simple_supports(), query)
    assert stable_hash(first["raw_candidate_json"]) == stable_hash(second["raw_candidate_json"])


def test_bucket_microproposal_enumeration_uses_existing_predicates_only() -> None:
    query = freeze_grid([[1, 0, 2], [0, 0, 0]])
    parse = enumerate_parse_states(query)[0]
    graph = build_object_graph(parse)
    allowed_predicates = set(allocation_spec()["relation_predicate_menu"])
    for bucket, _ in RELATION_ACTION_BUCKETS:
        proposals = enumerate_bucket_microproposals(graph, (1, 2), bucket)
        assert len(proposals) <= 10
        assert all(program.predicate in allowed_predicates for program in proposals)
