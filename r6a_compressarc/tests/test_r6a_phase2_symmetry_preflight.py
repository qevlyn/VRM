from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
CONTRACT = R6 / "manifests" / "r6a_phase2_symmetry_contract"
PREFLIGHT = R6 / "artifacts" / "r6a_phase2_symmetry_preflight"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_upstream_pin_and_source_claims_are_source_grounded() -> None:
    pin = read_json(CONTRACT / "upstream_source_pin_verification.json")
    contract = read_json(CONTRACT / "r6a_phase2_symmetry_contract.json")
    assert pin["source_pin_verified"] is True
    assert pin["git_rev_parse_head"] == PINNED_COMMIT
    assert contract["source_pin_verified"] is True
    for claim in contract["claims"]:
        assert claim["pinned_commit"] == PINNED_COMMIT
        assert claim["source_file"]
        assert claim["line_range"]
        assert claim["classification"] in {
            "DIRECT_SOURCE_FACT",
            "SOURCE_DERIVED_INFERENCE",
            "UNKNOWN_REQUIRES_RUNTIME_CHECK",
        }


def test_phase1_archive_and_immutability_passed() -> None:
    archive = read_json(CONTRACT / "phase1_archive_manifest.json")
    after = read_json(CONTRACT / "phase1_immutability_after.json")
    assert Path(archive["archive_path"]).exists()
    assert archive["integrity_test_passed"] is True
    assert archive["file_count"] == after["file_count"]
    assert after["all_match_before"] is True
    assert all(row["matches_before"] for row in after["files"])


def test_cpu_import_isolation_redirects_cuda_default_device() -> None:
    proof = read_json(PREFLIGHT / "cpu_execution_proof.json")
    assert proof["classification"] == "CPU_PREFLIGHT_ESTABLISHED"
    assert proof["effective_default_device"] == "cpu"
    assert proof["effective_default_dtype"] == "torch.float64"
    assert proof["cuda_memory_allocated_after_preflights"] == 0
    assert any(row["requested"].startswith("cuda") and row["effective"] == "cpu" and row["redirected"] for row in proof["import_intercept_log"])


def test_seed_reset_final_effective_seed_is_fixed() -> None:
    proof = read_json(PREFLIGHT / "cpu_execution_proof.json")
    assert proof["fixed_seed"] == FIXED_SEED
    assert proof["post_import_seed_probe"]["seed"] == FIXED_SEED
    assert proof["final_seed_probe_after_explicit_reset"] == proof["post_import_seed_probe"]


def test_c_raw_uses_within_active_set_derangements() -> None:
    c_raw = read_json(PREFLIGHT / "c_raw_color_preflight.json")
    for row in c_raw["tasks"]:
        if row["status"] != "C_RAW_VALID_PREFLIGHT":
            assert row["status"] == "C_RAW_NOT_ELIGIBLE"
            continue
        active = row["active_non_black_raw_colors"]
        mapping = {int(k): int(v) for k, v in row["raw_mapping"].items()}
        assert mapping[0] == 0
        assert sorted(mapping.keys()) == [0] + active
        for color in active:
            assert mapping[color] in active
            assert mapping[color] != color
        assert sorted(mapping[color] for color in active) == active


def test_c_raw_induced_internal_permutation_is_non_identity() -> None:
    c_raw = read_json(PREFLIGHT / "c_raw_color_preflight.json")
    valid = [row for row in c_raw["tasks"] if row["status"] == "C_RAW_VALID_PREFLIGHT"]
    assert valid
    for row in valid:
        induced = {int(k): int(v) for k, v in row["induced_internal_color_axis_mapping"].items()}
        assert any(key != value for key, value in induced.items())
        assert row["induced_internal_color_permutation_is_identity"] is False
        assert row["canonicalization_collapse_detected"] is False


def test_c_raw_canonicalization_collapse_detection_case() -> None:
    base_internal = [0, 1, 2]
    transformed_internal = [0, 3, 4]
    mapping = {0: 0, 1: 3, 2: 4}
    induced = {idx: transformed_internal.index(mapping[color]) for idx, color in enumerate(base_internal)}
    assert induced == {0: 0, 1: 1, 2: 2}
    assert all(key == value for key, value in induced.items())


def test_s_cycle_invariance_theorem_records_verified_and_nonverified_cases() -> None:
    structural = read_json(PREFLIGHT / "s_cycle_invariance_structural_reanalysis.json")
    rows = structural["tasks"]
    verified = [row for row in rows if row["lemma_assumption_status"] == "LEMMA_VERIFIED_FOR_TASK"]
    not_claimed = [row for row in rows if row["lemma_assumption_status"] == "LEMMA_NOT_CLAIMED"]
    assert verified
    assert not_claimed
    for row in verified:
        assert row["s_cycle_invariant"] is True
        assert row["task_shapes_equal_under_s"] is True
        assert row["task_masks_equal_under_s"] is True
        assert row["predicted_query_shape_equal_under_s"] is True
        assert row["grid_size_uncertain_branch_equal_under_s"] is True
        assert row["task_flag_flip_under_s"]["in_out_same_size"]["before"] == row["task_flag_flip_under_s"]["in_out_same_size"]["after"]
    assert any(row["task_masks_equal_under_s"] is False for row in not_claimed)
    assert any(row["predicted_query_shape_equal_under_s"] is False for row in not_claimed)
    assert any(
        row["task_flag_flip_under_s"]["in_out_same_size"]["before"]
        != row["task_flag_flip_under_s"]["in_out_same_size"]["after"]
        for row in not_claimed
    )


def test_p_strict_forward_tolerance_classification_matches_threshold() -> None:
    p = read_json(PREFLIGHT / "p_strict_preflight.json")
    tau = p["tau_forward"]["tau_forward_abs"]
    assert tau > 0
    for row in p["actual_p_strict_preflights"]:
        step0_max = max(float(value) for value in row["step0"].values())
        assert row["step0_within_tau_forward"] is (step0_max <= tau)
        if not row["step0_within_tau_forward"]:
            assert row["classification"] == "P_STRICT_STEP0_STRUCTURAL_FAIL"


def test_p_strict_composed_trajectory_calibration_is_20_step_real_path() -> None:
    p = read_json(PREFLIGHT / "p_strict_preflight.json")
    rows = p["tau_trajectory"]["calibration_rows"]
    assert rows
    assert p["tau_trajectory"]["tau_trajectory_abs"] > 0
    for row in rows:
        assert row["steps"] == 20
        assert len(row["trace"]) == 20
        for step in row["trace"]:
            for key in ["D_logits", "D_masks", "D_loss", "D_trainable_state", "D_optimizer_state", "D_prediction"]:
                assert key in step


def test_p_strict_optimizer_state_is_not_harness_mutated_during_training() -> None:
    p = read_json(PREFLIGHT / "p_strict_preflight.json")
    rule = p["optimizer_state_rule"]
    assert rule["adam_state_empty_at_step_zero_for_actuals"] is True
    assert rule["optimizer_state_mutated_during_training_by_harness"] is False
    assert rule["inverse_transform_for_comparison_only"] is True
    for row in p["actual_p_strict_preflights"]:
        assert row["optimizer_empty_at_step_zero"] is True
        assert row["state_inventory_summary"]["actions"]["UNKNOWN_REQUIRES_MANUAL_REVIEW"] == 0
        assert all("D_optimizer_state" in step for step in row["trace"])


def test_actual_cpu_preflight_tensors_are_cpu() -> None:
    proof = read_json(PREFLIGHT / "cpu_execution_proof.json")
    assert proof["all_tested_tensors_cpu"] is True
    for row in proof["tensor_proof"]["floating_tensors"] + proof["tensor_proof"]["integer_index_tensors"]:
        assert row["device"] == "cpu"
    assert proof["all_tested_floating_tensors_float64"] is True
    for row in proof["tensor_proof"]["floating_tensors"]:
        assert row["dtype"] == "torch.float64"
