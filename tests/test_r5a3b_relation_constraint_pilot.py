from __future__ import annotations

from pathlib import Path

from vrm.twin_tasks.r5a3a_arc_native_candidate_machine import SupportPair
from vrm.twin_tasks.r5a3a_scene_hypotheses import stable_hash
from vrm.twin_tasks.r5a3b_constraint_acquisition import acquire_constraints
from vrm.twin_tasks.r5a3b_object_graph import build_object_graph
from vrm.twin_tasks.r5a3b_parse_hypotheses import enumerate_parse_states
from vrm.twin_tasks.r5a3b_relation_constraint_pilot import (
    NEW_SOURCE_PATHS,
    PROTECTED_PRIOR_PATHS,
    mechanism_preflight_report,
    path_hashes,
    pilot_selection_spec,
    pre_hidden_pilot_commitment,
    property_test_report,
    select_pilot_manifest,
)
from vrm.twin_tasks.r5a3b_relation_programs import (
    MAX_RAW_CANDIDATES_PER_FOLD,
    apply_conditions,
    build_raw_candidate_universe,
    visible_input_hash,
)


class DummyTask:
    def __init__(self, task_id: str, train_inputs: tuple, train_outputs: tuple, split_name: str = "training") -> None:
        self.task_id = task_id
        self.source_relative_path = f"{task_id}.json"
        self.task_file_sha256 = stable_hash(task_id)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_inputs = tuple()
        self.split_name = split_name


def test_parse_hypotheses_include_connectivity_and_background_variants() -> None:
    diagonal = ((1, 0), (0, 1))
    states = enumerate_parse_states(diagonal)
    object_counts = sorted(state.object_count for state in states)
    assert object_counts == [1, 2]
    modal_not_zero = ((7, 7), (7, 0))
    backgrounds = {state.background_color for state in enumerate_parse_states(modal_not_zero)}
    assert backgrounds == {0, 7}


def test_object_graph_represents_nearest_ties_without_silent_choice() -> None:
    grid = ((0, 1, 0), (2, 0, 3), (0, 0, 0))
    graph = build_object_graph(enumerate_parse_states(grid)[0])
    assert graph.nearest_tie_count > 0
    tied_edges = [edge for edge in graph.edges if edge.predicates["nearest_is_tied"]]
    assert tied_edges
    assert all("nearest" in edge.predicates for edge in tied_edges)


def test_hard_constraints_require_universal_unambiguous_contrastive_support() -> None:
    supports = (
        SupportPair(((1, 0, 2),), ((1, 0, 2),)),
        SupportPair(((3, 0, 4),), ((3, 0, 4),)),
    )
    model = acquire_constraints(supports, background_color=0, connectivity=4)
    hard = {constraint.predicate for constraint in model.hard_constraints()}
    assert "color_unchanged" in hard
    accidental = acquire_constraints((SupportPair(((1, 0),), ((1, 0),)),), background_color=0, connectivity=4)
    assert "color_unchanged" not in {constraint.predicate for constraint in accidental.hard_constraints()}
    ambiguous = acquire_constraints((SupportPair(((1, 2),), ((1, 0),)),), background_color=0, connectivity=4)
    assert not ambiguous.hard_constraints()


def test_condition_ledgers_share_raw_universe_and_respect_budget() -> None:
    supports = (
        SupportPair(((1, 0, 2), (0, 0, 0)), ((1, 0, 1), (0, 0, 0))),
        SupportPair(((3, 0, 4), (0, 0, 0)), ((3, 0, 3), (0, 0, 0))),
    )
    query = ((5, 0, 6), (0, 0, 0))
    universe = build_raw_candidate_universe(supports, query)
    assert universe["raw_candidate_count"] <= MAX_RAW_CANDIDATES_PER_FOLD
    constraints = {
        state.parse_id: acquire_constraints(supports, background_color=state.background_color, connectivity=state.connectivity)
        for state in enumerate_parse_states(query)
    }
    conditions = apply_conditions(
        raw_candidates=universe["raw_candidates"],
        query=query,
        constraint_by_parse=constraints,
        visible_hash=visible_input_hash(supports, query),
    )
    raw_ids = {candidate.candidate_id for candidate in universe["raw_candidates"]}
    for name, payload in conditions.items():
        rows = payload["runs"] if name == "MATCHED_RANDOM_PRUNE" else [payload]
        for row in rows:
            assert set(row["candidate_ids"]).issubset(raw_ids)
    hard_count = conditions["RELATION_HARD_CONSTRAINT"]["unique_candidate_count"]
    assert all(run["unique_candidate_count"] <= hard_count for run in conditions["MATCHED_RANDOM_PRUNE"]["runs"])


def test_pre_hidden_commitment_records_forbidden_hidden_use_false() -> None:
    manifest = {
        "pilot_version": "test",
        "task_count": 0,
        "tasks": [],
        "pilot_manifest_sha256": "test",
    }
    spec = pilot_selection_spec()
    commitment = pre_hidden_pilot_commitment(manifest, spec)
    assert commitment["hidden_fold_outputs_used_before_this_commitment"] is False
    assert commitment["arc_test_outputs_used"] is False
    assert commitment["task_id_runtime_input"] is False


def test_selection_uses_input_hash_not_task_id_for_manifest_tie_breaks() -> None:
    tasks = []
    for index in range(30):
        one = index >= 20
        height = index + 1
        rows = [[0, 0, 0] for _ in range(height)]
        rows[0][0] = 1
        if not one:
            rows[-1][2] = 2
        grid = tuple(tuple(row) for row in rows)
        tasks.append(DummyTask(f"task-{index}", (grid, grid), (grid, grid)))
    manifest = select_pilot_manifest(tasks)
    assert manifest["task_count"] == 24
    assert manifest["task_ids_used_for_runtime_candidate_generation_or_ranking"] is False
    assert all("input_only_selection_hash" in row for row in manifest["tasks"])


def test_preflight_and_property_reports_pass() -> None:
    assert mechanism_preflight_report()["passed"] is True
    assert property_test_report()["passed"] is True


def test_protected_prior_hash_scan_is_read_only_and_new_source_paths_are_additive() -> None:
    before = path_hashes(PROTECTED_PRIOR_PATHS)
    after = path_hashes(PROTECTED_PRIOR_PATHS)
    assert before == after
    existing_new_sources = [path for path in NEW_SOURCE_PATHS if Path(path).exists()]
    assert existing_new_sources
