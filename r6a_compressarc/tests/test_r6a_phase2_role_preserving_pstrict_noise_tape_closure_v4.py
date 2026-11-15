from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
PREFIX = "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT = R6 / "reports" / PREFIX
SCRIPT = R6 / "scripts" / f"{PREFIX}_child.py"
FIXED_SEED = 20260702

ALLOWED_LABELS = {
    "P_ROLE_PRESERVING_STRICT_GROUP_ACTION_CONFIRMED",
    "P_STRICT_DETERMINISTIC_STEP0_MISMATCH",
    "P_STRICT_STOCHASTIC_COUPLING_MISMATCH",
    "P_STRICT_BACKWARD_MISMATCH",
    "P_STRICT_ADAM_UPDATE_MISMATCH",
    "P_STRICT_TRAJECTORY_MISMATCH",
    "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE",
    "P_STRICT_RUNNER_INSTRUMENTATION_INVALID",
    "P_ROLE_PRESERVING_TASK_RECONSTRUCTION_MISMATCH",
    "P_ROLE_PERM_NOT_ELIGIBLE",
    "P_STRICT_REQUIRES_MANUAL_REVIEW",
}

EARLIER_LABELS = {
    "P_ROLE_PRESERVING_TASK_RECONSTRUCTION_MISMATCH",
    "P_STRICT_RUNNER_INSTRUMENTATION_INVALID",
    "P_STRICT_DETERMINISTIC_STEP0_MISMATCH",
    "P_STRICT_STOCHASTIC_COUPLING_MISMATCH",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def closure_tasks() -> list[dict]:
    return read_json(ARTIFACT / "closure_results.json")["tasks"]


def test_original_protected_file_no_write_guard() -> None:
    before = read_json(MANIFEST / "preflight_integrity_before.json")
    after = read_json(MANIFEST / "preflight_integrity_after.json")
    assert before["original_protected_file_count"] == 701
    assert after["original_protected_file_count"] == 701
    assert before["all_match_before"] is True
    assert after["all_match_before"] is True
    assert after["no_write_guard_after"]["pass"] is True


def test_701_versus_295_reconciliation() -> None:
    reconciliation = read_json(REPORT / "integrity_scope_reconciliation.json")
    assert reconciliation["count_701"] == 701
    assert reconciliation["count_295"] == 295
    assert isinstance(reconciliation["is_295_subset_of_701"], bool)
    if reconciliation["is_295_subset_of_701"]:
        assert reconciliation["set_difference_295_minus_701"] == []
    else:
        assert reconciliation["set_difference_295_minus_701"]
    assert isinstance(reconciliation["set_difference_701_minus_295"], list)
    assert reconciliation["phase1_fully_reverified_for_701"] is True
    assert reconciliation["all_archived_members_match_295_manifest"] is True
    assert all(row["matches_295_manifest"] for row in reconciliation["archive_checksum_status_by_archived_member"])


def test_cycle_invariant_frozen_ledger_aggregation() -> None:
    cycle = read_json(ARTIFACT / "cycle_invariant_effects.json")
    assert cycle["analysis_label"] == "EXPLORATORY_FROZEN_LEDGER_REANALYSIS"
    assert cycle["selected_task_count"] == 5
    assert len(cycle["selected_tasks"]) == 5
    assert 0 <= cycle["count_with_median_pairing_delta_gt_0"] <= 5
    assert 0 <= cycle["count_with_median_pairing_delta_gt_noise_delta"] <= 5
    for suffix in [".csv", ".json", ".md", ".png", ".pdf"]:
        assert (ARTIFACT / f"cycle_invariant_effects{suffix}").exists() or (
            suffix in {".png", ".pdf"} and (ARTIFACT / f"figure_cycle_invariant_pairing_vs_noise{suffix}").exists()
        )
    rows = list(csv.DictReader((ARTIFACT / "cycle_invariant_effects.csv").open(encoding="utf-8")))
    assert rows
    for task in cycle["selected_tasks"]:
        assert task["analysis_label"] == "EXPLORATORY_FROZEN_LEDGER_REANALYSIS"
        assert task["structural_lemma_assumptions_and_status"]["lemma_assumption_status"] == "LEMMA_VERIFIED_FOR_TASK"
        assert task["sign_of_task_level_median_pairing_delta"] in {"positive", "negative", "zero"}


def test_cpu_import_isolation() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["effective_default_device"] == "cpu"
    assert proof["effective_default_dtype"] == "torch.float64"
    assert proof["cuda_memory_allocated"] == 0
    assert proof["all_runtime_tensors_under_test_cpu"] is True
    assert proof["all_new_floating_preflight_tensors_float64"] is True
    assert any(row["requested"].startswith("cuda") and row["effective"] == "cpu" and row["redirected"] for row in proof["import_intercept_log"])


def test_fixed_seed_after_all_imports() -> None:
    proof = read_json(ARTIFACT / "cpu_execution_proof.json")
    assert proof["fixed_seed"] == FIXED_SEED
    assert proof["post_import_seed_probe"]["seed"] == FIXED_SEED
    assert proof["final_seed_probe_after_imports"] == proof["post_import_seed_probe"]


def test_support_only_permutation_preserves_train_test_membership() -> None:
    sys.path.insert(0, str(R6 / "scripts"))
    import r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4_child as child

    perm = child.support_perm({"train": [{}, {}, {}], "test": [{}, {}]})
    assert perm == [1, 2, 0, 3, 4]
    for task in closure_tasks():
        if task["final_label"] == "P_ROLE_PERM_NOT_ELIGIBLE":
            continue
        checks = task["task_reconstruction"]["checks"]
        assert checks["same_train_test_membership_at_every_index"] is True
        assert checks["same_n_train"] is True
        assert checks["same_n_test"] is True


def test_n_train_less_than_two_ineligibility_is_hard_coded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'len(problem["train"]) < 2' in source
    assert "P_ROLE_PERM_NOT_ELIGIBLE" in source


def test_alias_aware_snapshot_restore_preserves_alias_topology() -> None:
    audit = read_json(ARTIFACT / "snapshot_restore_audit.json")
    assert audit["tasks"]
    for task in audit["tasks"]:
        if "error" in task:
            continue
        assert task["alias_topology_preserved_by_path_counts"] is True
        assert task["restore_used_in_place_copy"] is True
        assert task["rebinding_used"] is False


def test_true_example_axis_transforms_and_non_example_state_remains() -> None:
    audit = read_json(ARTIFACT / "snapshot_restore_audit.json")
    non_error = [task for task in audit["tasks"] if "error" not in task]
    assert non_error
    assert any(task["example_axis_tensor_count"] > 0 for task in non_error)
    assert any(task["non_example_axis_tensor_count"] > 0 for task in non_error)
    inventory = (ARTIFACT / "forward_reachable_state_inventory.csv").read_text(encoding="utf-8")
    assert "true" in inventory
    assert "false" in inventory


def test_a0_unmodified_vs_instrumented_transparency() -> None:
    for task in closure_tasks():
        stages = task.get("stages", {})
        if "A0" in stages:
            if task["final_label"] == "P_STRICT_RUNNER_INSTRUMENTATION_INVALID":
                assert stages["A0"]["pass"] is False
            else:
                assert stages["A0"]["pass"] is True


def test_noise_tape_record_replay() -> None:
    manifest = read_json(ARTIFACT / "noise_tape_manifest.json")
    runtime = read_json(ARTIFACT / "noise_tape_runtime_trace.json")
    assert manifest["tasks"]
    assert runtime["trace"]
    assert any(row["mode"] == "record" for row in runtime["trace"])
    assert any(row["mode"] == "replay" for row in runtime["trace"])
    for task in manifest["tasks"]:
        assert task["runtime_trace_count"] >= 0


def test_coupled_support_axis_epsilon_replay() -> None:
    runtime = read_json(ARTIFACT / "noise_tape_runtime_trace.json")
    replay_rows = [row for row in runtime["trace"] if row["mode"] == "replay" and row["real_example_axis"]]
    if replay_rows:
        assert all(row["audit_stage"] in {"A0_replay", "A2_coupled", "B_backward", "C_adam", "C_next_forward", "D_trajectory"} for row in replay_rows)
    else:
        assert all(task["final_label"] in EARLIER_LABELS for task in closure_tasks())


def test_unhandled_stochastic_call_hard_stop() -> None:
    unhandled = read_json(ARTIFACT / "noise_tape_unhandled_call_report.json")
    labels = {task["final_label"] for task in closure_tasks()}
    assert labels <= ALLOWED_LABELS
    if "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE" not in labels:
        assert unhandled["unhandled_calls"] == []
    assert "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE" in SCRIPT.read_text(encoding="utf-8")


def test_zero_noise_a1_inverse_alignment() -> None:
    for task in closure_tasks():
        stages = task.get("stages", {})
        if task["final_label"] == "P_STRICT_RUNNER_INSTRUMENTATION_INVALID":
            assert "A1" not in stages
            continue
        if "A1" in stages:
            if task["final_label"] == "P_STRICT_DETERMINISTIC_STEP0_MISMATCH":
                assert stages["A1"]["pass"] is False
            else:
                assert stages["A1"]["pass"] is True


def test_gradient_inverse_alignment() -> None:
    for task in closure_tasks():
        stages = task.get("stages", {})
        if "B" in stages:
            if task["final_label"] == "P_STRICT_BACKWARD_MISMATCH":
                assert stages["B"]["pass"] is False
            else:
                assert stages["B"]["pass"] is True
        else:
            assert task["final_label"] in EARLIER_LABELS | {"P_ROLE_PERM_NOT_ELIGIBLE", "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE", "P_STRICT_REQUIRES_MANUAL_REVIEW"}


def test_adam_moment_inverse_alignment() -> None:
    for task in closure_tasks():
        stages = task.get("stages", {})
        if "C" in stages:
            if task["final_label"] == "P_STRICT_ADAM_UPDATE_MISMATCH":
                assert stages["C"]["pass"] is False
            else:
                assert stages["C"]["pass"] is True
            assert "optimizer_state" in stages["C"]
        else:
            assert task["final_label"] in EARLIER_LABELS | {"P_STRICT_BACKWARD_MISMATCH", "P_ROLE_PERM_NOT_ELIGIBLE", "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE", "P_STRICT_REQUIRES_MANUAL_REVIEW"}


def test_hard_stop_prevents_later_stages_after_earlier_failure() -> None:
    stage_order = ["A0", "A1", "A2", "B", "C", "D"]
    terminal_stage = {
        "P_STRICT_RUNNER_INSTRUMENTATION_INVALID": "A0",
        "P_STRICT_DETERMINISTIC_STEP0_MISMATCH": "A1",
        "P_STRICT_STOCHASTIC_COUPLING_MISMATCH": "A2",
        "P_STRICT_BACKWARD_MISMATCH": "B",
        "P_STRICT_ADAM_UPDATE_MISMATCH": "C",
        "P_STRICT_TRAJECTORY_MISMATCH": "D",
        "P_ROLE_PRESERVING_STRICT_GROUP_ACTION_CONFIRMED": "D",
    }
    for task in closure_tasks():
        label = task["final_label"]
        stages = task.get("stages", {})
        if label not in terminal_stage:
            continue
        last_allowed = stage_order.index(terminal_stage[label])
        assert all(stage_order.index(stage) <= last_allowed for stage in stages)
