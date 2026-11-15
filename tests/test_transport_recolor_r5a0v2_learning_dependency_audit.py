from __future__ import annotations

import torch

from vrm.twin_tasks import transport_recolor_r5a0v2_learning_dependency_audit as audit
from vrm.twin_tasks.transport_recolor_r5a0v2_model import SupportConditionedTransportRecolorR5A0V2


def test_static_scan_separates_immutable_constants_from_oracle_helpers():
    scan = audit.static_source_scan()
    model_rows = [row for row in scan["rows"] if row["source_file"].endswith("transport_recolor_r5a0v2_model.py")]
    assert len(model_rows) == 1
    model_row = model_rows[0]
    assert model_row["imports_oracle_constants"] == ["COLOR_COUNT", "GRID_SIZE"]
    assert model_row["imports_oracle_helpers"] == []
    assert model_row["calls_oracle_symbol"] == []
    assert model_row["contains_target_access_in_forward_path"] is False
    assert model_row["contains_target_access_elsewhere"] is True
    assert scan["model_reused_immutable_oracle_constants"] == ["COLOR_COUNT", "GRID_SIZE"]
    assert scan["model_forward_imports_or_calls_oracle_helpers"] is False
    assert scan["model_contains_fixed_semantic_scaffold"] is True


def test_forward_provenance_keeps_targets_and_operation_metadata_out_of_forward():
    provenance = audit.forward_provenance()
    assert provenance["provenance_valid"] is True
    assert provenance["forbidden_forward_use"] == []
    by_field = {row["field_name"]: row for row in provenance["fields"]}
    assert by_field["query_grids"]["used_in_forward"] is True
    assert by_field["support_input_grids"]["used_in_forward"] is True
    assert by_field["support_output_grids"]["used_in_forward"] is True
    assert by_field["target_grids"]["target_oracle_metadata"] is True
    assert by_field["target_grids"]["used_in_forward"] is False
    assert by_field["operation_class"]["operation_or_program_metadata"] is True
    assert by_field["operation_class"]["used_in_forward"] is False


def test_dynamic_oracle_monkeypatch_does_not_break_model_forward():
    trace = audit.dynamic_trace_and_monkeypatch(torch.device("cpu"))
    assert trace["dynamic_trace"]["oracle_reached"] is False
    assert trace["dynamic_trace"]["oracle_functions_reached_during_model_forward"] == []
    assert trace["input_packing_trace"]["oracle_aliases_reached_during_tensor_packing"] == ["grid_to_list"]
    assert trace["all_monkeypatched_oracle_callables_forward_succeeded"] is True
    assert [row for row in trace["monkeypatch_execution_audit"] if not row["forward_succeeded"]] == []


def test_parameter_group_inventory_and_intervention_plan_cover_every_group():
    model = SupportConditionedTransportRecolorR5A0V2()
    inventory = audit.parameter_group_inventory(model)
    groups = set(inventory["groups"])
    assert groups == {
        "embedding_lookup_tables",
        "exact_pair_projector",
        "logit_scale_scalar",
        "pair_readout_latent_token",
        "relation_encoder_attention_projections",
        "relation_encoder_mlp_norms",
        "summary_pair_projector",
        "task_context_mlp",
    }
    assert "output_size_branch" in inventory["absent_requested_groups"]
    plan = audit.intervention_plan_for_checkpoint(model)
    assert len(plan) == 1 + len(audit.FULL_REINIT_SEEDS) + 4 * len(groups)
    for group in groups:
        suffixes = {row["condition_suffix"] for row in plan if row["groups"] == [group]}
        assert f"group_zeroed::{group}" in suffixes
        assert f"group_reinitialized::{group}" in suffixes
        assert f"group_perturbed_small::{group}" in suffixes
        assert f"group_perturbed_large::{group}" in suffixes


def test_gradient_read_only_probe_has_no_optimizer_step_and_identifies_output_path():
    probe = audit.gradient_read_only_probe(torch.device("cpu"))
    stats = probe["group_gradient_stats"]
    assert probe["optimizer_step_called"] is False
    assert stats["logit_scale_scalar"]["nonzero_grad_count"] == 1
    assert stats["logit_scale_scalar"]["nonzero_gradient_fraction"] == 1.0
    for group, row in stats.items():
        if group == "logit_scale_scalar":
            continue
        assert row["grad_norm"] == 0.0
        assert row["nonzero_grad_count"] == 0


def test_no_training_or_checkpoint_artifact_terms_are_introduced_by_audit_surface():
    artifact = audit.no_training_artifact_audit()
    assert artifact["forbidden_training_terms_found"] == []
    assert artifact["checkpoint_or_weight_files_created"] == []
    assert artifact["passed"] is True


def test_stable_report_hash_ignores_timestamp_and_report_hash_field():
    base = {"created_utc": "2026-01-01T00:00:00+00:00", "report_body_sha256_excluding_timestamp": "old", "value": 7}
    changed = {"created_utc": "2026-01-02T00:00:00+00:00", "report_body_sha256_excluding_timestamp": "new", "value": 7}
    assert audit.stable_report_hash(base) == audit.stable_report_hash(changed)


def test_reinitialization_scheme_is_documented_for_interventions():
    scheme = audit.reinitialization_scheme()
    assert scheme["scope"] == "in-memory audit copies only"
    assert "SupportConditionedTransportRecolorR5A0V2" in scheme["declared_initialization_path"]
    assert scheme["fallback_used"] is False
