from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path("/mnt/d/VRM/r6a_compressarc/scripts")
sys.path.insert(0, str(SCRIPT_DIR))

from r6a_common import (  # noqa: E402
    REFERENCE_ROOT,
    challenge_for_condition,
    deterministic_selection,
    pairing_manifest_for_task,
    read_json,
    select_derangement,
)


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"


def test_protected_vrm_paths_are_byte_identical_before_and_after() -> None:
    before = read_json(R6 / "manifests" / "protected_paths_before.json")
    after = read_json(R6 / "manifests" / "protected_paths_after.json")
    assert before["file_count"] == after["file_count"]
    assert after["all_match_before"] is True
    assert all(row["matches_before"] for row in after["files"])


def test_solver_process_cannot_load_or_access_query_target_outputs() -> None:
    negative = read_json(R6 / "artifacts" / "target_firewall_negative_control.json")
    assert negative["status"] == "QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL_PASSED"
    for name in ["solver_a.json", "solver_b.json"]:
        solver = read_json(R6 / "artifacts" / "target_firewall_negative_control" / name)
        assert solver["solver_process_firewall"]["challenge_json_only"] is True
        assert solver["solver_process_firewall"]["solution_argument_present"] is False
        assert solver["solver_process_firewall"]["denied_solution_paths"] == []


def test_replacing_evaluator_targets_does_not_alter_solver_outputs() -> None:
    negative = read_json(R6 / "artifacts" / "target_firewall_negative_control.json")
    assert negative["solver_training_trajectory_identical"] is True
    assert negative["predicted_query_grid_identical"] is True
    assert negative["stopping_step_identical"] is True
    assert negative["solver_side_logs_identical"] is True
    assert negative["only_evaluator_metrics_may_differ"] is True


def test_full_b1_b2_planned_runs_share_seed_budget_optimizer_and_query_input() -> None:
    manifest = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    assert manifest["run_seed"] == 0
    assert manifest["optimization_budget"] == 1500
    for task_id, meta in manifest["task_metadata"].items():
        problem = challenges[task_id]
        full = challenge_for_condition(problem, "FULL")
        if meta["b1_selected_permutation"]:
            b1 = challenge_for_condition(problem, "ALL_PAIRING_SWAP", meta["b1_selected_permutation"])
            assert b1["test"] == full["test"]
            assert [pair["input"] for pair in b1["train"]] == [pair["input"] for pair in full["train"]]
            assert sorted(json.dumps(pair["output"], sort_keys=True) for pair in b1["train"]) == sorted(
                json.dumps(pair["output"], sort_keys=True) for pair in full["train"]
            )
        if meta["b2_selected_permutation"]:
            b2 = challenge_for_condition(problem, "CONTENT_PAIRING_SWAP", meta["b2_selected_permutation"])
            assert b2["test"] == full["test"]
            assert [pair["input"] for pair in b2["train"]] == [pair["input"] for pair in full["train"]]


def test_b1_derangement_is_valid_and_preserves_input_output_multisets() -> None:
    manifest = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    for task_id, meta in manifest["task_metadata"].items():
        perm = meta["b1_selected_permutation"]
        assert perm is not None
        assert sorted(perm) == list(range(len(perm)))
        assert all(src != idx for idx, src in enumerate(perm))
        full = challenges[task_id]
        swapped = challenge_for_condition(full, "ALL_PAIRING_SWAP", perm)
        assert [pair["input"] for pair in swapped["train"]] == [pair["input"] for pair in full["train"]]
        assert sorted(json.dumps(pair["output"], sort_keys=True) for pair in swapped["train"]) == sorted(
            json.dumps(pair["output"], sort_keys=True) for pair in full["train"]
        )


def test_b2_derangement_is_shape_preserving_when_eligible() -> None:
    manifest = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    for task_id, meta in manifest["task_metadata"].items():
        perm = meta["b2_selected_permutation"]
        if perm is None:
            assert meta["b2_status"].startswith("CONTENT_PAIRING_INELIGIBLE")
            continue
        shapes = [meta["train_output_shapes"][idx] for idx in range(len(perm))]
        reassigned = [meta["train_output_shapes"][src] for src in perm]
        assert reassigned == shapes
        swapped = challenge_for_condition(challenges[task_id], "CONTENT_PAIRING_SWAP", perm)
        assert [len(pair["output"]) for pair in swapped["train"]] == [shape[0] for shape in shapes]


def test_derangement_selection_is_deterministic_and_stable() -> None:
    outputs = [
        [[1, 1], [0, 0]],
        [[2, 2], [0, 0]],
        [[3, 0], [3, 0]],
    ]
    first = select_derangement(outputs, shape_preserving=True)
    second = select_derangement(outputs, shape_preserving=True)
    assert first == second
    assert first["selected"]["permutation"] == [1, 2, 0]
    assert deterministic_selection(["c", "a", "b"], 2, 123) == deterministic_selection(["b", "c", "a"], 2, 123)


def test_evaluator_metrics_are_computed_only_after_solver_inference() -> None:
    eval_path = R6 / "artifacts" / "phase0_smoke" / "903d1b4a" / "FULL" / "evaluator.json"
    solver_path = R6 / "artifacts" / "phase0_smoke" / "903d1b4a" / "FULL" / "solver.json"
    assert solver_path.exists()
    assert eval_path.exists()
    solver = read_json(solver_path)
    evaluator = read_json(eval_path)
    assert solver["solver_process_firewall"]["solution_argument_present"] is False
    assert evaluator["evaluator_process_loaded_solution"] is True
    assert evaluator["solver_process_loaded_solution"] is False


def test_wrapped_full_run_uses_reference_task_solution_none_path() -> None:
    solve_text = (R6 / "scripts" / "solve_challenge_only.py").read_text(encoding="utf-8")
    official_text = (REFERENCE_ROOT / "solve_task.py").read_text(encoding="utf-8")
    assert "preprocessing.Task(args.task_id, problem, None)" in solve_text
    assert "Task(task_name, problems[task_name], None)" in official_text
    solver = read_json(R6 / "artifacts" / "manual_two_step" / "solver.json")
    assert solver["actual_step_count"] == 2
    assert solver["solver_process_firewall"]["challenge_json_only"] is True


def test_arc_agi_1_manifests_and_pilot_selection_frozen_before_outcomes() -> None:
    smoke = read_json(R6 / "manifests" / "phase0_smoke_tasks.json")
    phase1 = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    assert smoke["split"] == "ARC-AGI-1 evaluation"
    assert phase1["split"] == "ARC-AGI-1 evaluation"
    assert smoke["outcomes_read_before_freeze"] is False
    assert phase1["outcomes_read_before_freeze"] is False
    assert smoke["selected_task_ids"] == ["903d1b4a", "e66aafb8", "67b4a34d"]
    assert phase1["selected_task_ids"] == ["604001fa", "79fb03f4", "981571dc", "3ed85e70", "aa4ec2a5", "929ab4e9"]

