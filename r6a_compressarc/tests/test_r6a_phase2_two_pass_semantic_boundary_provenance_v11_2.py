from __future__ import annotations

import csv
import json
from pathlib import Path

import torch


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
PREFIX = "r6a_phase2_two_pass_semantic_boundary_provenance_v11_2"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
SCRIPT = R6 / "scripts" / f"{PREFIX}_child.py"


CASES = {
    "604001fa_adjacent_0_1",
    "604001fa_adjacent_1_2",
    "604001fa_adjacent_2_3",
    "1f876c06_adjacent_0_1",
    "1f876c06_adjacent_1_2",
    "synthetic_uniform_shape_oracle_adjacent_0_1",
    "synthetic_uniform_shape_oracle_adjacent_1_2",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_write_guard_for_protected_phase1_v8_and_v10_paths() -> None:
    before = read_json(MANIFEST / "integrity_before.json")
    after = read_json(MANIFEST / "integrity_after.json")
    assert before["phase1"]["all_match_before"] is True
    assert after["phase1"]["all_match_before"] is True
    assert after["preexisting_phase2"]["pass"] is True
    assert after["v8_artifacts"]["pass"] is True
    assert after["v10_artifacts"]["pass"] is True
    assert after["pass"] is True


def test_imported_modules_exactly_match_pinned_git_blobs() -> None:
    provenance = read_json(MANIFEST / "imported_module_provenance.json")
    assert provenance["import_provenance_status"] == "IMPORT_PROVENANCE_CERTIFIED"
    assert {row["module_name"] for row in provenance["modules"]} == {"preprocessing", "arc_compressor", "layers", "initializers", "multitensor_systems", "train"}
    assert all(row["matches_pinned_blob"] for row in provenance["modules"])
    assert all(row["path_under_intended_source_root"] for row in provenance["modules"])


def test_cpu_only_no_gpu_execution() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["effective_default_device"] == "cpu"
    assert proof["effective_default_dtype"] == "torch.float64"
    assert proof["torch_num_threads"] == 1
    assert proof["cuda_allocation_count"] == 0
    assert proof["cuda_execution_count"] == 0
    assert proof["no_cuda_execution"] is True
    assert proof["runtime_tensors_audited_are_cpu"] is True


def test_v10_semantic_action_is_reused_without_shape_heuristics() -> None:
    state = read_json(ARTIFACT / "semantic_state_action_reuse_certificate.json")
    assert len(state["cases"]) == 7
    assert all(row["status"] == "SEMANTIC_STATE_ACTION_CERTIFIED" for row in state["cases"])
    assert all(row["v10_semantic_action_reused_exactly"] is True for row in state["cases"])
    source = SCRIPT.read_text(encoding="utf-8")
    assert "tensor.shape[0] == task.n_examples" not in source
    assert "leading_dim" not in source.split("def run_case", 1)[0]


def test_static_semantic_parent_schema_matches_actual_add_residual_dataflow() -> None:
    graph = read_json(ARTIFACT / "static_semantic_boundary_graph.json")
    assert graph["graph_type"] == "complete semantic-boundary provenance graph"
    assert graph["add_residual_prenorm_dataflow"]["down_affine_direct_parent"] == "raw_x"
    assert graph["add_residual_prenorm_dataflow"]["pre_norm_output_role"] == "observational_only"


def test_add_residual_prenorm_dataflow_verified() -> None:
    graph = read_json(ARTIFACT / "static_semantic_boundary_graph.json")
    assert graph["add_residual_prenorm_dataflow"]["status"] == "ADD_RESIDUAL_PRENORM_DATAFLOW_VERIFIED"


def test_event_alignment_uses_semantic_event_keys_not_raw_ordinal_only() -> None:
    with (ARTIFACT / "pass1_event_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all("::tensor=" in row["event_key"] for row in rows)
    assert any(row["parent_event_keys"] != "[]" for row in rows)


def test_trace_schedule_mismatch_blocks_later_value_comparison_logic_is_present() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TRACE_SCHEDULE_MISMATCH" in source
    assert "compare_event_schedule" in source


def test_pass1_hash_mismatch_is_not_labelled_e_bit_before_replay() -> None:
    pass1 = read_json(ARTIFACT / "pass1_candidate_markers.json")
    confirmed = read_json(ARTIFACT / "confirmed_markers.json")
    assert "E_hash" in pass1["cases"][0]
    assert all("E_bit" in row for row in confirmed["cases"])
    assert pass1 != confirmed


def test_targeted_replay_confirms_byte_mismatch_when_present() -> None:
    confirmed = read_json(ARTIFACT / "confirmed_markers.json")
    mismatched = [row for row in confirmed["cases"] if row.get("E_bit") is not None]
    assert mismatched
    assert all(row["status"] == "TARGETED_REPLAY_REPRODUCIBLE" for row in mismatched)


def test_failed_replay_status_is_available_for_nonreproducible_replay() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "PROVENANCE_REPLAY_NOT_REPRODUCIBLE" in source


def test_ulp_is_reported_only_as_severity_metric() -> None:
    replay = read_json(ARTIFACT / "targeted_replay_exact_comparison.json")
    comparisons = [item for row in replay["cases"] for item in row.get("comparisons", [])]
    assert comparisons
    assert any("max_ulp" in item for item in comparisons)
    assert all("final_classification" not in item for item in comparisons)


def test_signed_zero_nan_and_inf_are_structural_status() -> None:
    replay = read_json(ARTIFACT / "targeted_replay_exact_comparison.json")
    comparisons = [item for row in replay["cases"] for item in row.get("comparisons", [])]
    assert any({"signed_zero_count", "nan_count", "inf_count"}.issubset(item.keys()) for item in comparisons)


def test_cummax_microtrace_separates_cardinal_and_diagonal_branches() -> None:
    micro = read_json(ARTIFACT / "cummax_microtrace.json")
    assert micro["cases"]
    assert any(row["cardinal_branch_points"] > 0 for row in micro["cases"])
    assert any(row["diagonal_branch_points"] > 0 for row in micro["cases"])


def test_normalize_oracle_declares_three_reference_modes() -> None:
    oracle = read_json(ARTIFACT / "selective_numerical_oracle.json")
    assert oracle["oracles"]
    assert any(row.get("fixed_order_reference") == "FIXED_ORDER_HIGH_ACCURACY_REFERENCE" for row in oracle["oracles"])
    assert all("decimal_escalation" in row for row in oracle["oracles"])


def test_unstable_decimal_reference_status_is_represented() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "NUMERICAL_REFERENCE_UNRESOLVED" in source or "classification_rule_satisfied" in read_json(ARTIFACT / "selective_numerical_oracle.json")["oracles"][0]


def test_decoder_cannot_produce_final_grid_claim_without_certification() -> None:
    decoder = read_json(ARTIFACT / "decoder_scope_certificate.json")
    assert decoder["status"] == "AUDIT_DECODER_NOT_IN_FORWARD_SCOPE"
    assert all(row["E_discrete"] == "decoder-out-of-scope" for row in decoder["cases"])


def test_mutation_fixtures_compare_source_with_mutation_against_scanned_with_mutation() -> None:
    calibration = read_json(ARTIFACT / "scanner_mutation_calibration.json")
    assert calibration["pass_count"] == 4
    assert all(row["source_with_mutation_equals_scanned_with_mutation"] for row in calibration["fixtures"])


def test_mutation_calibration_locates_first_affected_semantic_trace_point() -> None:
    calibration = read_json(ARTIFACT / "scanner_mutation_calibration.json")
    assert all(row["localized_to_first_affected_semantic_trace_point"] for row in calibration["fixtures"])


def test_controls_are_included_in_every_scan() -> None:
    summary = read_json(ARTIFACT / "child_summary.json")
    assert set(summary["eligible_cases_scanned"]) == CASES


def test_no_loss_backward_optimizer_retraining_vrm_or_architecture_path_is_reachable() -> None:
    cert = read_json(ARTIFACT / "forbidden_path_certificate.json")
    assert cert["loss_run"] is False
    assert cert["backward_run"] is False
    assert cert["optimizer_or_multi_step_run"] is False
    assert cert["retraining_run"] is False
    assert cert["vrm_path_run"] is False
    assert cert["architecture_change_run"] is False
    assert cert["p_slot_c_internal_j_g_or_attribution_run"] is False
