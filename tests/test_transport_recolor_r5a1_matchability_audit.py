from __future__ import annotations

from pathlib import Path

import pytest

from vrm.twin_tasks import transport_recolor_r5a1_matchability_audit as audit
from vrm.twin_tasks import transport_recolor_r5a1_preflight as preflight


@pytest.fixture(scope="session")
def audit_bundle():
    return audit.build_all()


@pytest.fixture(scope="session")
def report(audit_bundle):
    return audit_bundle[3]


def test_canonical_profiling_batch_reuses_preflight_hash(report):
    assert report["hard_gates"]["canonical_batch_hash_reused"] is True
    assert report["canonical_profiling_batch"]["tensor_sha256"] == report["canonical_batch_hash_from_preflight"]
    assert report["canonical_profiling_batch"]["training_side_only"] is True
    assert report["canonical_profiling_batch"]["sealed_rows_read"] is False


def test_search_domain_generation_is_deterministic():
    first = audit.build_search_domain()
    second = audit.build_search_domain()
    assert first == second
    assert audit.json_hash(first) == audit.json_hash(second)
    assert first["cartesian_grid"]["hidden_width"][0] == 2
    assert first["cartesian_grid"]["hidden_width"][-1] == 256
    for width in range(32, 257, 8):
        assert width in first["cartesian_grid"]["hidden_width"]


def test_exhaustive_enumeration_count_matches_declared_domain(audit_bundle):
    domain, ledger, analysis, report = audit_bundle
    assert analysis["total_declared_cartesian_combinations"] == domain["declared_cartesian_combination_count"]
    assert ledger["candidate_count"] == analysis["total_valid_candidates"]
    assert ledger["candidate_count"] + ledger["invalid_candidate_count"] == domain["declared_cartesian_combination_count"]
    assert report["hard_gates"]["exhaustive_valid_candidate_count_matches_declared_domain"] is True
    assert ledger["invalid_reasons"] == {"num_heads must divide hidden_size": 120}


def test_candidate_ledger_schema(audit_bundle):
    _domain, ledger, _analysis, _report = audit_bundle
    row = ledger["rows"][0]
    required = {
        "candidate_id",
        "architecture_config",
        "hidden_width",
        "encoder_layers",
        "decoder_layers",
        "attention_heads",
        "mlp_expansion",
        "total_trainable_parameters",
        "embedding_parameters",
        "active_parameters_on_canonical_batch_if_measurable",
        "inactive_parameter_groups_if_any",
        "total_analytic_forward_flops",
        "encoder_forward_flops",
        "decoder_forward_flops",
        "output_size_forward_flops",
        "forward_flops_per_valid_output_cell",
        "activation_proxy_bytes",
        "parameter_ratio_to_v2",
        "flop_ratio_to_v2",
        "within_parameter_tolerance",
        "within_flop_tolerance",
        "jointly_feasible",
        "generic_interface_valid",
        "firewall_valid",
        "all_trainable_groups_reachable_or_explained",
    }
    assert required <= set(row)
    assert row["candidate_id"].startswith("h")
    assert row["inactive_parameter_groups_if_any"] == []
    assert row["generic_interface_valid"] is True
    assert row["firewall_valid"] is True


def test_parameter_and_flop_ratio_calculations(audit_bundle):
    _domain, ledger, _analysis, report = audit_bundle
    v2_params = report["v2_reference"]["trainable_parameters"]
    v2_flops_per_cell = report["v2_reference"]["flops"]["per_valid_output_cell_forward_flops"]
    for row in (ledger["rows"][0], ledger["rows"][len(ledger["rows"]) // 2], ledger["rows"][-1]):
        assert row["parameter_ratio_to_v2"] == row["total_trainable_parameters"] / v2_params
        assert row["flop_ratio_to_v2"] == row["forward_flops_per_valid_output_cell"] / v2_flops_per_cell
        assert row["jointly_feasible"] == (row["within_parameter_tolerance"] and row["within_flop_tolerance"])


def test_pareto_frontier_correctness_on_toy_points():
    toy = [
        {"candidate_id": "a", "parameter_error_abs": 0.1, "flop_error_abs": 0.5},
        {"candidate_id": "b", "parameter_error_abs": 0.2, "flop_error_abs": 0.4},
        {"candidate_id": "c", "parameter_error_abs": 0.1, "flop_error_abs": 0.6},
        {"candidate_id": "d", "parameter_error_abs": 0.3, "flop_error_abs": 0.3},
        {"candidate_id": "e", "parameter_error_abs": 0.2, "flop_error_abs": 0.4},
    ]
    frontier = audit.pareto_frontier(toy)
    assert [row["candidate_id"] for row in frontier] == ["a", "b", "e", "d"]


def test_jointly_feasible_rectangle_classification(audit_bundle):
    _domain, _ledger, analysis, report = audit_bundle
    assert analysis["feasible_rectangle_contains_candidate"] is False
    assert analysis["total_jointly_feasible_candidates"] == 0
    assert analysis["jointly_feasible_candidates"] == []
    assert report["status"] == "R5A1A_MATCHABILITY_CURRENT_FAMILY_INSUFFICIENT"
    assert "universal architectural impossibility" not in report["claim_boundary"].lower()


def test_forbidden_semantic_interface_absence(report):
    firewall = report["firewall_preservation"]
    assert firewall["search_introduced_new_semantic_mechanism"] is False
    assert firewall["source_scan"]["passed"] is True
    assert firewall["source_scan"]["forbidden_terms_found"] == []
    assert firewall["candidate_checks"]
    assert all(row["firewall_valid"] for row in firewall["candidate_checks"])


def test_no_dummy_or_unreachable_trainable_parameter_groups(report):
    findings = report["no_dummy_capacity_findings"]
    assert findings["dummy_or_unreachable_parameter_groups_found"] is False
    assert findings["all_trainable_groups_reachable_or_explained_for_all_candidates"] is True
    assert all(row["matches_instantiated_model"] for row in findings["parameter_formula_validation"])


def test_no_checkpoint_training_artifact_created(report):
    artifact = report["artifact_audit"]
    assert artifact["passed"] is True
    assert artifact["checkpoint_or_weight_files"] == []
    assert artifact["training_terms_found_in_new_sources"] == []
    assert artifact["sealed_eval_terms_found_in_new_sources"] == []
    artifact_dir = Path("/mnt/d/VRM/artifacts/transport_recolor_r5a1a_matchability_audit")
    if artifact_dir.exists():
        assert not list(artifact_dir.rglob("*.pt"))
        assert not list(artifact_dir.rglob("*.pth"))
        assert not list(artifact_dir.rglob("*.ckpt"))


def test_protected_v2_order_preflight_and_preregistration_hashes_unchanged(report):
    assert report["protected_hashes_before"] == report["protected_hashes_after"]
    assert report["hard_gates"]["protected_hashes_unchanged_during_report_build"] is True
    protected_paths = "\n".join(report["protected_hashes_before"])
    assert "transport_recolor_r5a0v2_order_audit.py" in protected_paths
    assert "transport_recolor_r5a1_preflight.py" in protected_paths
    assert "transport_recolor_r5a1a_preregistration.py" in protected_paths


def test_matchability_hashes_are_stable(audit_bundle):
    domain, ledger, analysis, report = audit_bundle
    second = audit.build_all()[3]
    assert audit.json_hash(domain) == report["search_domain_hashes"]["search_domain_body_sha256"]
    assert audit.json_hash(ledger) == report["search_domain_hashes"]["candidate_ledger_body_sha256"]
    assert audit.json_hash(analysis) == report["search_domain_hashes"]["pareto_frontier_body_sha256"]
    assert report["report_body_sha256_excluding_timestamp"] == second["report_body_sha256_excluding_timestamp"]
