from __future__ import annotations

import inspect
from pathlib import Path

from vrm.twin_tasks import transport_recolor_r5a0v2_order_audit as audit


def test_independent_checker_golden_cases_and_generated_agreement():
    report = audit.checker_agreement_report()
    assert report["checker_agreement_failures"] == []
    assert report["checker_agreement_count"] == len(audit.GOLDEN_CASES) + report["generated_validation_example_count"]
    assert report["generated_validation_example_count"] == 62


def test_probe_domain_is_exact_cell_exhaustive_domain():
    domain = audit.probe_domain()
    assert len(domain) == 7 * 7 * 9
    assert domain[0] == {"witness_id": "cell:x=0,y=0,color=0", "x": 0, "y": 0, "color": 0}
    assert domain[-1] == {"witness_id": "cell:x=6,y=6,color=8", "x": 6, "y": 6, "color": 8}


def test_cellwise_lift_validity_gate_controls_whole_grid_exhaustiveness(tmp_path: Path):
    pytest_result = tmp_path / "pytest.txt"
    pytest_result.write_text("270 passed\n", encoding="utf-8")
    report, _ = audit.build_report(pytest_result)
    cellwise = report["cellwise_lift_validity"]
    assert report["single_cell_domain_size"] == 441
    assert report["cellwise_lift_valid"] is True
    assert report["cellwise_lift_failure_reasons"] == []
    assert report["whole_grid_exhaustiveness_claim"] is True
    assert report["probe_domain_exhaustive"] is True
    assert cellwise["multicell_production_agreement_failures"] == []
    assert report["multicell_production_agreement_count"] == cellwise["multicell_fixture_count"] * cellwise["multicell_program_count"]


def test_coordinate_bijection_report_lists_no_drops_duplicates_or_overwrites():
    report = audit.cellwise_lift_validity_report()
    for primitive, row in report["coordinate_bijection_by_primitive"].items():
        assert row["mapping_total"], primitive
        assert row["destinations_in_bounds"], primitive
        assert row["output_colors_valid"], primitive
        assert row["foreground_coordinate_bijection"], primitive
        assert row["background_coordinate_bijection"], primitive
        assert row["dropped_destinations"] == []
        assert row["duplicated_destinations"] == []
        assert row["colliding_destinations"] == []
        assert row["overwritten_destinations"] == []


def test_multicell_fixture_catalog_covers_required_cases():
    fixture_ids = {row["fixture_id"] for row in audit.multicell_fixtures()}
    assert "empty_background_only" in fixture_ids
    assert "one_active_cell" in fixture_ids
    assert "two_cells_distinct_colors" in fixture_ids
    assert any(item.startswith("x_boundary_pair") for item in fixture_ids)
    assert any(item.startswith("y_boundary_pair") for item in fixture_ids)
    assert "quadrant_boundary_corners" in fixture_ids
    assert "dense_all_colors_all_branches" in fixture_ids
    assert "deterministic_random_dense_seed_20260628" in fixture_ids


def test_source_independence_gate_excludes_grid_level_reads():
    report = audit.source_grid_independence_report()
    assert report["no_neighbor_object_occupancy_or_global_state_reads"] is True
    assert report["forbidden_grid_state_terms_found"] == []


def test_all_current_program_fingerprints_are_semantically_unique():
    programs = audit.SEALED_PROGRAMS
    fingerprints = {program: audit.behavior_fingerprint(program) for program in programs}
    assert len(set(fingerprints.values())) == len(programs)


def test_heldout_programs_are_order_distinguishable_and_do_not_overlap_train(tmp_path: Path):
    pytest_result = tmp_path / "pytest.txt"
    pytest_result.write_text("263 passed\n", encoding="utf-8")
    report, _ = audit.build_report(pytest_result)
    assert report["verdict"] == audit.PASS_VERDICT
    assert report["heldout_vs_train"]["heldout_mixed_collision_with_train_count"] == 0
    rows = {row["program_id"]: row for row in report["per_program"]}
    for program in audit.HELDOUT_MIXED_PROGRAMS:
        assert rows[program]["order_distinguishable"]
        assert rows[program]["first_divergence_witness_id"] is not None


def test_report_hash_is_deterministic_with_same_pytest_gate(tmp_path: Path):
    pytest_result = tmp_path / "pytest.txt"
    pytest_result.write_text("270 passed\n", encoding="utf-8")
    first, first_samples = audit.build_report(pytest_result)
    second, second_samples = audit.build_report(pytest_result)
    assert audit.json_hash(first) == audit.json_hash(second)
    assert audit.json_hash(first_samples) == audit.json_hash(second_samples)


def test_independent_core_does_not_call_production_dispatcher():
    source = "\n".join(
        inspect.getsource(item)
        for item in (
            audit.independent_apply_primitive,
            audit.independent_apply_program_to_cell,
            audit.independent_apply_program_to_grid,
            audit.behavior_rows,
            audit.behavior_fingerprint,
        )
    )
    assert "apply_sequence_to_grid" not in source
    assert "apply_sequence_to_cell" not in source
    assert "transport_recolor_r5a0v2_oracles" not in source
    assert audit.REUSED_IMMUTABLE_CONSTANTS == ()


def test_known_reverse_witnesses_are_exposed():
    c_k = audit.first_divergence("C->K", "K->C")
    assert c_k is not None
    assert c_k["first_divergence_input"] == {"x": 0, "y": 0, "color": 1}
    assert c_k["original_output"] != c_k["reversed_output"]
    h_l = audit.first_divergence("H->L", "L->H")
    assert h_l is not None
    assert h_l["original_output"] != h_l["reversed_output"]
