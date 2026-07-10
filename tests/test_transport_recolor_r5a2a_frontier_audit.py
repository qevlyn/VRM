from __future__ import annotations

from pathlib import Path

from vrm.twin_tasks import transport_recolor_r5a2a_declared_closure as closure
from vrm.twin_tasks import transport_recolor_r5a2a_frontier_audit as audit
from vrm.twin_tasks import transport_recolor_r5a2a_frontier_probes as probes


def test_independent_closure_source_does_not_import_v2_model_or_oracle_helpers():
    source = Path("/mnt/d/VRM/vrm/twin_tasks/transport_recolor_r5a2a_declared_closure.py").read_text(encoding="utf-8")
    assert "transport_recolor_r5a0v2_model" not in source
    assert "transport_recolor_r5a0v2_oracles" not in source
    assert "SupportConditionedTransportRecolorR5A0V2" not in source
    assert "apply_sequence_to_grid" not in source
    report = audit.source_independence_report()
    assert report["passed"] is True


def test_closure_enumerator_correctness_on_handwritten_toy_programs():
    assert closure.apply_program_to_cell("C", 1, 2, 3) == (4, 1, 3)
    assert closure.apply_program_to_cell("H", 1, 2, 3) == (5, 2, 3)
    assert closure.apply_program_to_cell("V", 1, 2, 3) == (1, 4, 3)
    assert closure.apply_program_to_cell("L", 3, 2, 3) == (3, 2, 7)
    assert closure.apply_program_to_cell("K", 3, 3, 4) == (3, 3, 8)
    assert closure.apply_program_to_cell("C->L", 4, 1, 2) == (5, 4, 2)
    report = closure.closure_report()
    assert report["number_of_programs"] == 13
    assert report["whether_bounded_complete"] is True
    assert report["semantic_collision_count"] == 0


def test_probe_generation_is_deterministic_and_split_balanced():
    first, first_manifest = probes.generate_probe_panel(variants_per_combo=6)
    second, second_manifest = probes.generate_probe_panel(variants_per_combo=6)
    assert first_manifest["panel_hash"] == second_manifest["panel_hash"]
    assert [episode.to_record() for episode in first] == [episode.to_record() for episode in second]
    split = probes.split_audit(first)
    assert split["passed"] is True
    assert split["counts_by_split_group"] == {"train_visible": 48, "held_out": 48}


def test_per_family_golden_cases_selector_uniqueness_and_identifiability():
    episodes, _ = probes.generate_probe_panel(variants_per_combo=1)
    families = {episode.probe_family for episode in episodes}
    assert families == set(probes.FAMILY_ORDER)
    for episode in episodes:
        assert episode.validation["intended_rule_is_consistent"] is True
        assert episode.validation["selection_is_unique_in_every_support_and_query_grid"] is True
        assert episode.validation["at_least_one_alternative_rule_is_inconsistent"] is True
        assert episode.validation["transform_rule_is_identifiable_from_support_pairs"] is True


def test_closure_membership_for_generated_full_and_isolated_instances():
    episodes, _ = probes.generate_probe_panel(variants_per_combo=2)
    for episode in episodes:
        assert episode.validation["full_scene_representability"]["representable_by_declared_v2_closure"] is False
        assert episode.validation["isolated_target_representability"]["matching_programs"] == [episode.transform_program]


def test_shortcut_diagnostic_and_generator_validity():
    episodes, manifest = probes.generate_probe_panel(variants_per_combo=6)
    generator = probes.generator_audit(episodes, manifest)
    assert generator["by_family"]["largest_component_select_then_v2_transform"]["shortcut_validity"]["max_shortcut_exact"] == 1.0
    assert generator["by_family"]["nearest_marker_select_then_v2_transform"]["generator_valid_for_toolbox_expansion"] is True
    assert generator["by_family"]["unique_shape_select_then_v2_transform"]["generator_valid_for_toolbox_expansion"] is True
    assert generator["by_family"]["nearest_marker_select_then_v2_transform"]["shortcut_validity"]["max_shortcut_exact"] <= probes.SHORTCUT_THRESHOLD
    assert generator["by_family"]["unique_shape_select_then_v2_transform"]["shortcut_validity"]["max_shortcut_exact"] <= probes.SHORTCUT_THRESHOLD


def test_v2_forward_uses_only_raw_grid_fields_and_no_training_surface_terms():
    fields = audit.v2_forward_field_report()
    assert fields["fields_passed_to_v2_forward"] == ["query_grids", "support_input_grids", "support_output_grids"]
    assert fields["forbidden_metadata_fields_passed"] == []
    training = audit.no_training_artifact_audit()
    assert training["forbidden_training_or_checkpoint_terms_found"] == []
    assert training["checkpoint_or_weight_files_created"] == []
    assert training["passed"] is True


def test_protected_v2_and_prior_audit_hashes_unchanged_by_probe_generation():
    before = audit.protected_hashes()
    episodes, manifest = probes.generate_probe_panel(variants_per_combo=2)
    _ = probes.generator_audit(episodes, manifest)
    _ = closure.closure_report()
    after = audit.protected_hashes()
    assert before == after


def test_frontier_report_smoke_has_valid_global_verdict():
    report, *_ = audit.build_report()
    assert report["status"] in {
        audit.VERDICT_READY,
        audit.VERDICT_BROADER,
        audit.VERDICT_NO_CLEAN,
        audit.VERDICT_INCONCLUSIVE,
    }
    assert report["hard_gates"]["v2_forward_ordinary_raw_fields_only"] is True
    assert report["hard_gates"]["no_training_or_checkpoint_artifacts"] is True
