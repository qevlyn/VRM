from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
SCRIPT_DIR = R6 / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from r6a_common import REFERENCE_ROOT, challenge_for_condition, read_json  # noqa: E402
from run_phase1_multirepeat_pairing_audit import CONDITIONS, SEEDS, padded_hamming, select_swap  # noqa: E402


MANIFEST = R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json"
INIT_MANIFEST = R6 / "manifests" / "r6a_phase1_initialization_artifacts.json"
PER_RUN = R6 / "artifacts" / "r6a_phase1_per_run_results.json"
FIREWALL_JSON = R6 / "artifacts" / "r6a_phase1_firewall_confirmation.json"


def phase1_manifest() -> dict:
    assert MANIFEST.exists()
    return read_json(MANIFEST)


def per_run_rows() -> list[dict]:
    if not PER_RUN.exists():
        return []
    return read_json(PER_RUN)["rows"]


def challenge_bank(split: str) -> dict:
    if split == "A":
        return read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    return read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")


def test_static_solver_evaluator_process_separation() -> None:
    firewall = read_json(FIREWALL_JSON)
    static = firewall["static_data_flow_isolation"]
    assert firewall["firewall_preflight_passed"] is True
    assert static["solution_json_path_absent_from_solver_inputs"] is True
    assert static["solution_json_path_absent_from_solver_commands"] is True
    assert static["solution_json_path_absent_from_data_path_env_vars"] is True
    assert all("solutions" not in path.lower() for path in static["solver_input_paths"])


def test_cpu_dummy_target_negative_control_is_solver_identical() -> None:
    negative = read_json(FIREWALL_JSON)["cpu_dummy_target_negative_control"]
    assert negative["fixed_prefix_steps"] == 20
    assert negative["solver_did_not_receive_dummy_targets"] is True
    assert negative["byte_identical_solver_traces"] is True
    assert negative["byte_identical_probability_tensors"] is True
    assert set(negative["evaluator_metrics"]) == {"dummy_a", "dummy_b"}


def test_protected_vrm_paths_byte_identical_before_and_after_phase1() -> None:
    before = read_json(R6 / "manifests" / "protected_paths_before_phase1.json")
    assert before["file_count"] > 0
    after_path = R6 / "manifests" / "protected_paths_after_phase1.json"
    if after_path.exists():
        after = read_json(after_path)
        assert after["all_match_before"] is True
        assert all(row["matches_before"] for row in after["files"])


def test_one_initialization_artifact_exists_per_task_seed() -> None:
    manifest = phase1_manifest()
    init = read_json(INIT_MANIFEST)
    expected = len(manifest["tasks"]) * len(SEEDS)
    assert len(init["rows"]) == expected
    assert init["all_artifacts_exist"] is True
    for row in init["rows"]:
        assert Path(row["path"]).exists()
        assert row["seed"] in SEEDS


def test_f1_s_f2_load_byte_identical_initialization_and_optimizer_states() -> None:
    rows = per_run_rows()
    if not rows:
        init = read_json(INIT_MANIFEST)
        assert all(row["model_state_hash"] and row["optimizer_state_hash"] for row in init["rows"])
        return
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["task_id"], row["seed"]), []).append(row)
    for group in grouped.values():
        completed = [row for row in group if row.get("status") == "RUN_COMPLETED"]
        if {row["condition"] for row in completed} == set(CONDITIONS):
            assert len({row["initialization_artifact"] for row in completed}) == 1
            assert len({row["initial_model_hash_loaded"] for row in completed}) == 1
            assert len({row["initial_optimizer_hash_loaded"] for row in completed}) == 1
            assert all(row["loaded_initialization_matches_saved"] is True for row in completed)


def test_f1_s_f2_reset_identical_saved_rng_states() -> None:
    rows = per_run_rows()
    if not rows:
        assert read_json(INIT_MANIFEST)["all_f1_s_f2_use_same_initialization_by_design"] is True
        return
    for row in rows:
        if row.get("status") == "RUN_COMPLETED":
            assert row["rng_state_reset_from_initialization_artifact"] is True


def test_swap_permutation_valid_derangement_and_preserves_multisets() -> None:
    manifest = phase1_manifest()
    for task in manifest["tasks"]:
        perm = task["recomputed"]["selected_permutation"]
        if not task["recomputed"]["eligible"]:
            continue
        assert sorted(perm) == list(range(len(perm)))
        assert all(idx != src for idx, src in enumerate(perm))
        problem = challenge_bank(task["stratum"])[task["task_id"]]
        swapped = challenge_for_condition(problem, "ALL_PAIRING_SWAP", perm)
        assert [pair["input"] for pair in swapped["train"]] == [pair["input"] for pair in problem["train"]]
        assert sorted(json.dumps(pair["output"], sort_keys=True) for pair in swapped["train"]) == sorted(
            json.dumps(pair["output"], sort_keys=True) for pair in problem["train"]
        )


def test_padded_hamming_uses_sentinel_minus_one_for_unequal_shapes() -> None:
    assert padded_hamming([[1, 2], [3, 4]], [[1]]) == 3
    assert padded_hamming([[0]], [[0, 0], [0, 0]]) == 3
    assert padded_hamming([], [[9]]) == 1


def test_max_hamming_and_lexicographic_tie_break_are_deterministic() -> None:
    problem = {
        "train": [
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
            {"input": [[2]], "output": [[2]]},
        ],
        "test": [{"input": [[0]]}],
    }
    first = select_swap(problem)
    second = select_swap(problem)
    assert first == second
    assert first["selected_permutation"] == [1, 2, 0]


def test_recomputed_stratum_a_permutations_equal_old_manifest_exactly() -> None:
    manifest = phase1_manifest()
    assert manifest["permutation_authority_all_stratum_a_match"] is True
    for task in manifest["tasks"]:
        if task["stratum"] == "A":
            assert task["permutation_authority_match"] is True
            assert task["old_manifest_permutation"] == task["recomputed"]["selected_permutation"]
            assert task["old_manifest_padded_hamming_score"] == task["recomputed"]["selected_total_padded_hamming"]


def test_query_probability_capture_contains_exactly_100_final_checkpoints() -> None:
    manifest = phase1_manifest()
    assert manifest["probability_capture_final_k"] == 100
    for row in per_run_rows():
        if row.get("status") != "RUN_COMPLETED" or row.get("condition") not in CONDITIONS:
            continue
        assert row["probability_capture"]["checkpoint_count"] == 100
        data = np.load(row["probability_npz"])
        assert data["probability"].shape[0] == 100
        assert data["checkpoint_steps"].tolist() == list(range(1401, 1501))


def test_solver_side_probability_capture_does_not_access_query_target_or_shape() -> None:
    source = (R6 / "scripts" / "phase1_condition_solver.py").read_text(encoding="utf-8")
    capture_section = source[source.index("def canonical_probability") : source.index("def run_condition")]
    assert "solution" not in capture_section.lower()
    assert "target" not in capture_section.lower()
    for row in per_run_rows():
        if row.get("status") == "RUN_COMPLETED":
            assert row["probability_capture"]["uses_hidden_target"] is False
            assert row["solver_process_firewall"]["solution_argument_present"] is False


def test_evaluator_computes_nll_only_after_solver_completion() -> None:
    source = (R6 / "scripts" / "run_phase1_multirepeat_pairing_audit.py").read_text(encoding="utf-8")
    assert source.index("solver_payload = read_json(solver_json)") < source.index('row["evaluator_metrics"] = evaluator_metrics')
    for row in per_run_rows():
        if row.get("status") == "RUN_COMPLETED":
            assert row["actual_step_count"] == 1500
            assert "target_cell_nll_from_final100_mean_probability" in row["evaluator_metrics"]


def test_frozen_manifest_written_before_first_gpu_run() -> None:
    assert MANIFEST.exists()
    solver_paths = [Path(row["solver_json"]) for row in per_run_rows() if row.get("solver_json")]
    existing_solver_paths = [path for path in solver_paths if path.exists()]
    if existing_solver_paths:
        assert MANIFEST.stat().st_mtime <= min(path.stat().st_mtime for path in existing_solver_paths)


def test_no_task_seed_run_or_ineligible_task_is_silently_dropped() -> None:
    manifest = phase1_manifest()
    assert len(manifest["tasks"]) == 9
    expected = {(row["task_id"], row["seed"], row["condition"]) for row in manifest["run_order"]}
    rows = per_run_rows()
    if rows:
        seen = {(row["task_id"], row["seed"], row["condition"]) for row in rows}
        assert expected <= seen
    assert all("eligibility_label" in row["recomputed"] for row in manifest["tasks"])


def test_selected_threshold_row_matches_number_of_eligible_tasks() -> None:
    manifest = phase1_manifest()
    eligible_a = sum(1 for row in manifest["tasks"] if row["stratum"] == "A" and row["recomputed"]["eligible"])
    eligible_b = sum(1 for row in manifest["tasks"] if row["stratum"] == "B" and row["recomputed"]["eligible"])
    threshold_a = manifest["verdict_thresholds"]["stratum_a"]
    threshold_b = manifest["verdict_thresholds"]["stratum_b"]
    assert threshold_a["eligible_tasks"] == eligible_a
    if eligible_a == 6:
        assert threshold_a["cells"] == 18
        assert threshold_a["positive_delta_threshold"] == 13
    elif eligible_a == 5:
        assert threshold_a["cells"] == 15
        assert threshold_a["positive_delta_threshold"] == 11
    else:
        assert threshold_a["insufficient_coverage"] is True
    assert threshold_b["eligible_tasks"] == eligible_b
    assert threshold_b["positive_control_valid"] is (eligible_b == 3)
