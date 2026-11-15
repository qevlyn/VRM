from __future__ import annotations

import inspect
from pathlib import Path

import torch

from vrm.twin_tasks import transport_recolor_r5a0v2_init_reproduction_audit as audit
from vrm.twin_tasks.transport_recolor_r5a0v2_data import make_episode, partition_query_types


def test_fresh_path_cannot_call_checkpoint_loading():
    report = audit.fresh_path_structural_report()
    assert report["fresh_init_path_can_call_checkpoint_loading"] is False
    source = inspect.getsource(audit.fresh_model_for_seed)
    assert "torch.load" not in source
    assert "load_state_dict" not in source


def test_deterministic_audit_suite_generation_and_stable_hash():
    first = audit.build_random_suite_records()
    second = audit.build_random_suite_records()
    assert first["manifest"]["suite_hash"] == second["manifest"]["suite_hash"]
    assert first["records"] == second["records"]
    assert first["manifest"]["record_count"] == 13 * 128 * 2
    assert set(first["manifest"]["class_balance"].values()) == {256}
    assert first["manifest"]["coverage"]["primary_colors"] == [1, 2, 3, 4]
    assert first["manifest"]["coverage"]["distractor_counts"] == [1, 2]


def test_fresh_models_have_distinct_state_hashes_across_seeds():
    device = torch.device("cpu")
    hashes = [audit.model_state_hash(audit.fresh_model_for_seed(seed, device)) for seed in audit.FRESH_INIT_SEEDS]
    assert len(set(hashes)) == len(audit.FRESH_INIT_SEEDS)


def test_selected_checkpoint_differs_from_fresh_state_when_checkpoint_exists():
    records = audit.selected_checkpoint_records()
    assert records
    checked = 0
    for record in records:
        if not record["checkpoint_exists"]:
            continue
        payload = torch.load(record["checkpoint_path"], map_location="cpu")
        checkpoint_hash = audit.state_dict_hash(payload["model_state_dict"])
        fresh_hash = audit.model_state_hash(audit.fresh_model_for_seed(int(record["seed"]), torch.device("cpu")))
        assert checkpoint_hash != fresh_hash
        checked += 1
    assert checked >= 1


def test_no_optimizer_backward_or_checkpoint_write_terms():
    artifact = audit.no_training_artifact_audit()
    assert artifact["optimizer_backward_or_checkpoint_write_terms_found"] == []
    assert artifact["checkpoint_or_weight_files_created"] == []
    assert artifact["passed"] is True


def test_correct_per_class_reporting_on_small_fixture():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qtype = partition_query_types()["train"][0]
    episodes = [make_episode("unit", qtype, "C", 0), make_episode("unit", qtype, "K->C", 0)]
    model = audit.fresh_model_for_seed(12345, device)
    result = audit.evaluate_suite(model, episodes, device, batch_size=2)
    assert result["record_count"] == 2
    assert result["class_counts"] == {"C": 1, "K->C": 1}
    assert set(result["exact_by_class"]) == {"C", "K->C"}
    assert result["exact_by_group"]["geometry_only"] == result["exact_by_class"]["C"]
    assert result["exact_by_group"]["held_out_mixed"] == result["exact_by_class"]["K->C"]
    assert result["prediction_hash"]


def test_forward_input_provenance_excludes_forbidden_metadata():
    provenance = audit.forward_input_provenance()
    assert audit.provenance_invalid(provenance) is False
    by_field = {row["field_name"]: row for row in provenance}
    assert by_field["query_grids"]["used_in_model_forward"] is True
    assert by_field["support_input_grids"]["used_in_model_forward"] is True
    assert by_field["support_output_grids"]["used_in_model_forward"] is True
    assert by_field["target_grids"]["is_target_oracle_metadata"] is True
    assert by_field["target_grids"]["used_in_model_forward"] is False
    assert by_field["operation_class"]["is_operation_program_class_metadata"] is True
    assert by_field["operation_class"]["used_in_model_forward"] is False


def test_protected_v2_source_and_artifact_hashes_unchanged():
    before = audit.protected_hashes()
    _ = audit.build_random_suite_records()
    after = audit.protected_hashes()
    assert before == after
    protected = "\n".join(before)
    assert "transport_recolor_r5a0v2_model.py" in protected
    assert "run_transport_recolor_r5a0v2.py" in protected
    assert "dataset_manifest.json" in protected
