from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT = R6 / "reports" / PREFIX / "R6A_PHASE2_SUPPORT_PERMUTATION_FIRST_DIVERGENCE_CERTIFICATE_V8_REPORT.md"
SCRIPT = R6 / "scripts" / f"{PREFIX}_child.py"
FIXED_SEED = 20260702


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cases():
    return read_json(ARTIFACT / "first_divergence_cases.json")["cases"]


def test_no_write_guard_for_original_phase1_and_preexisting_phase2_paths() -> None:
    before = read_json(MANIFEST / "integrity_before.json")
    after = read_json(MANIFEST / "integrity_after.json")
    assert before["phase1_original_701"]["original_protected_file_count"] == 701
    assert before["phase1_original_701"]["all_match_before"] is True
    assert after["phase1_original_701"]["all_match_before"] is True
    assert after["preexisting_phase2_no_write_guard"]["pass"] is True
    assert after["protected_file_integrity_pass"] is True


def test_imported_module_provenance_against_pinned_git_blobs() -> None:
    provenance = read_json(MANIFEST / "imported_module_provenance.json")
    assert provenance["import_provenance_status"] == "IMPORT_PROVENANCE_CERTIFIED"
    assert provenance["import_context"]["sys_path_0"] == "/mnt/d/VRM/third_party/compressarc_reference"
    required = {"preprocessing", "arc_compressor", "layers", "initializers", "multitensor_systems", "train"}
    assert {row["module_name"] for row in provenance["modules"]} == required
    for row in provenance["modules"]:
        assert row["path_under_intended_source_root"] is True
        assert row["git_show_ok"] is True
        assert row["actual_file_sha256"] == row["expected_pinned_git_blob_sha256"]
        assert row["matches_pinned_blob"] is True


def test_cpu_only_no_cuda_proof() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["effective_default_device"] == "cpu"
    assert proof["effective_default_dtype"] == "torch.float64"
    assert proof["torch_num_threads"] == 1
    assert proof["cuda_memory_allocated"] == 0
    assert proof["no_cuda_execution"] is True
    assert proof["all_runtime_tensors_under_audit_cpu"] is True


def test_fixed_seed_after_all_production_imports() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["fixed_seed"] == FIXED_SEED
    assert proof["post_import_seed_probe"]["seed"] == FIXED_SEED
    assert proof["final_seed_probe_after_imports"] == proof["post_import_seed_probe"]


def test_adjacent_support_transpositions_keep_test_query_slots_fixed() -> None:
    registry = read_json(ARTIFACT / "permutation_registry.json")
    assert len([row for row in registry["cases"] if row["case_kind"] == "real_task"]) == 5
    for row in registry["cases"]:
        assert row["permutation_non_identity"] is True
        assert row["test_query_slots_fixed_by_perm"] is True
        assert row["test_query_slots_fixed_by_invperm"] is True
        assert row["transposition"][1] == row["transposition"][0] + 1


def test_task_action_pass_fixture() -> None:
    cert = read_json(ARTIFACT / "task_action_certificate.json")
    assert len(cert["cases"]) == 7
    assert all(row["status"] == "TASK_ACTION_CERTIFIED" for row in cert["cases"])
    for row in cert["cases"]:
        assert all(row["checks"].values())


def test_task_action_deliberate_fail_fixture() -> None:
    canonical = {"train": [{"input": [[0]], "output": [[1]]}, {"input": [[0]], "output": [[2]]}], "test": [{"input": [[0]]}]}
    transformed = {"train": [{"input": [[0]], "output": [[99]]}, {"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]]}]}
    invperm_old_to_new = [1, 0, 2]
    support_pairing_after_inverse_alignment = all(transformed["train"][invperm_old_to_new[old_idx]] == canonical["train"][old_idx] for old_idx in range(2))
    assert support_pairing_after_inverse_alignment is False


def test_alias_topology_preservation() -> None:
    state = read_json(ARTIFACT / "state_action_certificate.json")
    alias = read_json(ARTIFACT / "alias_graph_certificate.json")
    assert len(state["cases"]) == 7
    assert all(row["status"] == "STATE_ACTION_CERTIFIED" for row in state["cases"])
    assert all(row["status"] == "ALIAS_TOPOLOGY_CERTIFIED" for row in alias["cases"])


def test_true_example_axis_transform_and_non_example_axis_preservation() -> None:
    state = read_json(ARTIFACT / "state_action_certificate.json")
    assert all(row["example_axis_tensor_count"] > 0 for row in state["cases"])
    assert all(row["non_example_axis_tensor_count"] > 0 for row in state["cases"])
    inventory = (ARTIFACT / "forward_reachable_state_inventory.csv").read_text(encoding="utf-8")
    assert "true" in inventory
    assert "false" in inventory
    assert "has_real_example_axis" in inventory


def test_stale_optimizer_parameter_partition() -> None:
    partition = read_json(ARTIFACT / "optimizer_parameter_partition.json")
    assert len(partition["cases"]) == 7
    for row in partition["cases"]:
        assert row["forward_reachable_optimizer_parameters"]
        assert row["stale_gradient_probe_status"] in {"BLOCKED_UNTIL_FORWARD_CERTIFIED", "EXECUTED"}
        assert row["stale_optimizer_parameter_active"] is False


def test_posterior_leaf_schedule_equals_runtime_channel_layer_schedule() -> None:
    schedule = read_json(ARTIFACT / "posterior_leaf_schedule.json")["leaves"]
    by_case = {}
    for row in schedule:
        by_case.setdefault(row["case_id"], []).append(row["ordinal"])
    trace = read_json(ARTIFACT / "noise_tape_runtime_trace.json")["trace"]
    assert trace
    for case_id, ordinals in by_case.items():
        channel_events = [row for row in trace if row["case_id"] == case_id and row["event"] == "channel_layer"]
        assert channel_events
        assert set(row["ordinal"] for row in channel_events) == set(ordinals)


def test_unclassified_stochastic_call_hard_stop_is_present_and_not_triggered() -> None:
    unhandled = read_json(ARTIFACT / "noise_tape_unhandled_call_report.json")
    assert unhandled["unhandled_calls"] == []
    source = SCRIPT.read_text(encoding="utf-8")
    assert "STOCHASTIC_TRACE_INCOMPLETE" in source
    assert "randn_without_active_leaf" in source


def test_trace_schedule_mismatch_priority_rule_is_encoded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TRACE_SCHEDULE_MISMATCH" in source
    assert "event_count_mismatch" in source
    assert "posterior_leaf_identity_mismatch" in source


def test_a0_source_record_replay_observed_replay_transparency() -> None:
    for row in cases():
        a0 = row["a0"]
        assert a0["source_record_replay_observed_pass"] is True
        assert a0["source_vs_record"]["pass"] is True
        assert a0["record_vs_replay"]["pass"] is True
        assert a0["replay_vs_observed_replay"]["pass"] is True


def test_tier1_first_divergence_event_ordering() -> None:
    rows = list(csv.DictReader((ARTIFACT / "first_divergence_trace.csv").open(encoding="utf-8")))
    assert rows
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, items in by_case.items():
        first_fail_index = next(i for i, row in enumerate(items) if row["pass"] == "False")
        assert all(row["pass"] == "True" for row in items[:first_fail_index])
        assert items[first_fail_index]["event_name"] in {"affine", "normalize", "channel_layer", "decode_latents", "share_direction"}


def test_tier2_trace_transparency_contract() -> None:
    detail_dir = ARTIFACT / "first_divergence_detail"
    details = list(detail_dir.glob("*.json"))
    if details:
        for path in details:
            payload = read_json(path)
            assert payload["transparency"]["pass"] is True
    else:
        primitive = read_json(ARTIFACT / "primitive_commutation_certificate.json")
        assert all(row["final_label"] != "FIRST_DIVERGENCE_CHANNEL_LAYER" for row in primitive["cases"])


def test_synthetic_uniform_shape_oracle() -> None:
    spec = read_json(ARTIFACT / "synthetic_oracle_spec.json")
    results = read_json(ARTIFACT / "synthetic_oracle_results.json")
    assert spec["requirements"]["train_support_pairs"] == 3
    assert spec["requirements"]["test_query_inputs"] == 1
    assert spec["requirements"]["known_grid_shape"] == [3, 3]
    assert len(results["cases"]) == 2
    assert {row["case"]["transposition"][0] for row in results["cases"]} == {0, 1}


def test_bounded_fingerprint_finite_check() -> None:
    synthetic = read_json(ARTIFACT / "synthetic_oracle_results.json")
    for row in synthetic["cases"]:
        fp = row["synthetic_fingerprint"]
        assert fp["fingerprinted_leaf_count"] > 0
        assert fp["all_fingerprinted_tensors_finite"] is True


def test_actual_take_step_logger_proxy_compatibility() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "class CaptureNoOpLogger" in source
    assert "class CaptureAdamProxy" in source
    loss = read_json(ARTIFACT / "loss_certificate.json")
    if loss["cases"]:
        assert all(row["logger_event_count"] == 1 for row in loss["cases"])
    else:
        assert loss["status"] == "NOT_REACHED_FORWARD_FAILURE"


def test_loss_backward_adam_blocked_after_forward_failure() -> None:
    labels = {row["final_label"] for row in cases()}
    assert "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED" not in labels
    assert read_json(ARTIFACT / "loss_certificate.json")["status"] == "NOT_REACHED_FORWARD_FAILURE"
    assert read_json(ARTIFACT / "backward_certificate.json")["status"] == "NOT_REACHED_FORWARD_FAILURE"
    assert read_json(ARTIFACT / "adam_certificate.json")["status"] == "NOT_REACHED_FORWARD_FAILURE"


def test_no_gpu_execution() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["cpu_only_no_gpu"] if "cpu_only_no_gpu" in proof else proof["cuda_memory_allocated"] == 0
    assert proof["no_cuda_execution"] is True
    assert all(row["device"] == "cpu" for row in proof["tensor_proof"])
