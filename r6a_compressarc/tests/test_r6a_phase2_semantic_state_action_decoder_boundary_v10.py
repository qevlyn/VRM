from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
PREFIX = "r6a_phase2_semantic_state_action_decoder_boundary_v10"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
SCRIPT = R6 / "scripts" / f"{PREFIX}_child.py"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def action_rows() -> list[dict[str, str]]:
    with (ARTIFACT / "semantic_action_table.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_cell(value: str):
    return json.loads(value)


def test_protected_phase1_and_preexisting_phase2_receive_no_writes() -> None:
    before = read_json(MANIFEST / "integrity_before.json")
    after = read_json(MANIFEST / "integrity_after.json")
    assert before["phase1"]["all_match_before"] is True
    assert before["phase1"]["original_protected_file_count"] == 701
    assert after["phase1"]["all_match_before"] is True
    assert after["preexisting_phase2"]["pass"] is True
    assert after["v8_artifacts"]["pass"] is True
    assert after["pass"] is True


def test_imported_modules_match_pinned_git_blobs() -> None:
    provenance = read_json(MANIFEST / "imported_module_provenance.json")
    assert provenance["import_provenance_status"] == "IMPORT_PROVENANCE_CERTIFIED"
    assert {row["module_name"] for row in provenance["modules"]} == {"preprocessing", "arc_compressor", "layers", "initializers", "multitensor_systems", "train"}
    assert all(row["matches_pinned_blob"] for row in provenance["modules"])
    assert all(row["path_under_intended_source_root"] for row in provenance["modules"])


def test_cpu_only_and_no_gpu_execution() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["effective_default_device"] == "cpu"
    assert proof["effective_default_dtype"] == "torch.float64"
    assert proof["torch_num_threads"] == 1
    assert proof["cuda_memory_allocated"] == 0
    assert proof["no_cuda_execution"] is True
    assert proof["runtime_tensors_audited_are_cpu"] is True


def test_semantic_action_registry_does_not_use_leading_dimension_equality() -> None:
    rows = action_rows()
    accidental = [row for row in rows if row["accidental_leading_dim_equality"] == "True"]
    assert accidental
    assert all(row["action_label"] == "IDENTITY_COPY" for row in accidental)
    assert "shape[0] == task.n_examples" not in SCRIPT.read_text(encoding="utf-8").split("def semantic_action", 1)[1].split("def build_semantic_records", 1)[0]


def test_posterior_mean_dims0_one_is_permuted_on_axis_zero() -> None:
    rows = [row for row in action_rows() if row["constructor_role"] == "posterior_mean" and parse_json_cell(row["leaf_dims"])[0] == 1]
    assert rows
    assert all(row["action_label"] == "PERMUTE_EXAMPLE_AXIS_0" for row in rows)


def test_posterior_local_capacity_dims0_one_is_permuted_on_axis_zero() -> None:
    rows = [row for row in action_rows() if row["constructor_role"] == "posterior_local_capacity_adjustment" and parse_json_cell(row["leaf_dims"])[0] == 1]
    assert rows
    assert all(row["action_label"] == "PERMUTE_EXAMPLE_AXIS_0" for row in rows)


def test_posterior_tensors_dims0_zero_remain_identity() -> None:
    rows = [
        row
        for row in action_rows()
        if row["constructor_role"] in {"posterior_mean", "posterior_local_capacity_adjustment"}
        and parse_json_cell(row["leaf_dims"])[0] == 0
    ]
    assert rows
    assert all(row["action_label"] == "IDENTITY_COPY" for row in rows)


def test_decode_weight_shape_4_16_dims0_one_nexamples4_remains_bit_identical() -> None:
    rows = [
        row
        for row in action_rows()
        if row["constructor_role"] == "decode_weight_matrix"
        and parse_json_cell(row["physical_shape"]) == [4, 16]
        and parse_json_cell(row["leaf_dims"])[0] == 1
        and row["leading_dim_equals_task_n_examples"] == "True"
    ]
    assert rows
    assert all(row["action_label"] == "IDENTITY_COPY" for row in rows)
    assert all(row["checksum_before_action"] == row["checksum_after_action"] for row in rows)


def test_target_capacity_vector_shape_4_dims0_one_nexamples4_remains_bit_identical() -> None:
    rows = [
        row
        for row in action_rows()
        if row["constructor_role"] == "target_capacity"
        and parse_json_cell(row["physical_shape"]) == [4]
        and parse_json_cell(row["leaf_dims"])[0] == 1
        and row["leading_dim_equals_task_n_examples"] == "True"
    ]
    assert rows
    assert all(row["action_label"] == "IDENTITY_COPY" for row in rows)
    assert all(row["checksum_before_action"] == row["checksum_after_action"] for row in rows)


def test_every_actual_synthetic_collision_witness_is_identity() -> None:
    witnesses = read_json(ARTIFACT / "parameter_collision_witnesses.json")["witnesses"]
    synthetic = [row for row in witnesses if row["case_id"].startswith("synthetic_uniform_shape_oracle")]
    assert synthetic
    assert all(row["action_label"] == "IDENTITY_COPY" for row in synthetic)
    assert all(row["exact_equality_result"] is True for row in synthetic)


def test_alias_classes_cannot_mix_posterior_permuted_and_identity_roles() -> None:
    alias = read_json(ARTIFACT / "alias_semantic_action_certificate.json")
    state = read_json(ARTIFACT / "semantic_state_action_certificate.json")
    assert all(row["status"] == "ALIAS_SEMANTIC_ACTION_CERTIFIED" for row in alias["cases"])
    assert all(row["status"] == "SEMANTIC_STATE_ACTION_CERTIFIED" for row in state["cases"])


def test_direct_source_affine_commutes_with_fixed_weight_and_bias() -> None:
    cert = read_json(ARTIFACT / "affine_commutation_certificate.json")
    assert cert["pass_count"] == 7
    assert cert["fail_count"] == 0
    for case in cert["cases"]:
        assert case["production_leaf_pass"] is True
        assert all(row["status"] == "PASS" for row in case["production_leaf_rows"])


def test_intentional_weight_permutation_negative_control_mismatches() -> None:
    cert = read_json(ARTIFACT / "affine_commutation_certificate.json")
    assert all(case["negative_control"]["wrong_weight_action_mismatch_observed"] is True for case in cert["cases"])


def test_decoder_boundary_priority_labels_channel_before_affine_when_relevant() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "FIRST_DIVERGENCE_CHANNEL_LAYER" in source
    assert "AFFINE_OBSERVATION_CONTRADICTION" in source
    summary = read_json(ARTIFACT / "forward_trace_summary.json")
    labels = {row["label"] for row in summary["cases"]}
    assert "FIRST_DIVERGENCE_AFFINE" not in labels


def test_affine_output_cannot_receive_affine_label_when_direct_certificate_passes() -> None:
    summary = read_json(ARTIFACT / "summary.json")
    labels = set(summary["decoder_boundary_result_per_case"].values())
    assert "FIRST_DIVERGENCE_AFFINE" not in labels
    assert "AFFINE_OBSERVATION_CONTRADICTION" not in labels
    assert summary["affine_certificate_pass_fail"] == [7, 0]


def test_no_loss_backward_adam_retraining_or_gpu_path_is_reachable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = ["take_step(", ".backward(", "Adam(", "optimizer.step", "n_iterations", "VRM", "P-slot"]
    assert all(token not in source for token in forbidden)
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["cuda_memory_allocated"] == 0
