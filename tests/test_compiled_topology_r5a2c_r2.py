from __future__ import annotations

import inspect
from pathlib import Path

from vrm.twin_tasks import compiled_topology_r5a2c_r2_checker as checker
from vrm.twin_tasks import compiled_topology_r5a2c_r2_data as data
from vrm.twin_tasks import compiled_topology_r5a2c_r2_executor as executor
from vrm.twin_tasks import run_compiled_topology_r5a2c_r2_audit as audit


def test_executor_has_no_nn_parameter_or_module_surface():
    source = Path("/mnt/d/VRM/vrm/twin_tasks/compiled_topology_r5a2c_r2_executor.py").read_text(encoding="utf-8")
    assert "nn.Parameter" not in source
    assert "nn.Module" not in source
    assert "torch" not in source
    report = audit.parameter_free_report()
    assert report["no_nn_parameter_exists"] is True
    assert report["no_nn_module_with_trainable_parameters_exists"] is True


def test_no_optimizer_backward_or_checkpoint_io_code_exists():
    report = audit.parameter_free_report()
    assert report["no_optimizer_constructed"] is True
    assert report["no_backward_called"] is True
    assert report["no_checkpoint_path_read"] is True
    assert report["no_checkpoint_path_written"] is True
    assert report["forbidden_code_terms_found"] == []
    assert report["passed"] is True


def test_checker_is_independent_from_executor_and_v2():
    source = Path("/mnt/d/VRM/vrm/twin_tasks/compiled_topology_r5a2c_r2_checker.py").read_text(encoding="utf-8")
    assert "compiled_topology_r5a2c_r2_executor" not in source
    assert "transport_recolor_r5a0v2" not in source
    assert audit.checker_independence_report()["passed"] is True


def test_golden_case_per_convention_passes():
    report = checker.verify_golden_cases()
    assert report["golden_case_count"] == 3
    assert report["passed"] is True
    assert {row["convention"] for row in report["rows"]} == set(data.CONVENTIONS)


def test_normal_four_neighbor_fixed_point_reachability():
    mask = {(1, 1), (2, 1), (3, 2)}
    reachable = data.fixed_point_reachable({(1, 1)}, mask, adjacency="4_connected")
    assert reachable == {(1, 1), (2, 1)}
    reachable8 = data.fixed_point_reachable({(1, 1)}, mask, adjacency="8_connected")
    assert reachable8 == mask


def test_barrier_handling_changes_reachability_output():
    episode = data.make_episode("golden_suite", "border_reachable_recolor", 1, 7, None, "diagonal_bridge_a", 0)
    normal = data.apply_hypothesis_reference(episode.query_input_grid, episode.hypothesis)
    ignored = data.apply_hypothesis_reference(episode.query_input_grid, episode.hypothesis, barrier_ignored=True)
    assert normal == episode.query_output_grid
    assert ignored != episode.query_output_grid


def test_border_initialization_and_enclosed_complement():
    border = data.make_episode("golden_suite", "border_reachable_recolor", 2, 8, None, "diagonal_bridge_b", 1)
    enclosed = data.make_episode("golden_suite", "enclosed_region_recolor", 2, 8, None, "diagonal_bridge_b", 1)
    assert data.apply_hypothesis_reference(border.query_input_grid, border.hypothesis) == border.query_output_grid
    assert data.apply_hypothesis_reference(enclosed.query_input_grid, enclosed.hypothesis) == enclosed.query_output_grid
    assert border.query_output_grid != enclosed.query_output_grid


def test_complete_semantic_tuple_uniqueness_for_accepted_episode():
    episode = data.generate_suites()[0]["randomized_suite"][0]
    support_inputs, support_outputs = audit.support_inputs_outputs(episode)
    matches = executor.consistent_hypotheses(support_inputs, support_outputs)
    checker_matches = checker.checker_consistent_hypotheses(support_inputs, support_outputs)
    assert len(matches) == 1
    assert len(checker_matches) == 1
    assert matches[0].to_tuple() == episode.hypothesis.to_tuple()
    assert checker_matches[0].to_tuple() == episode.hypothesis.to_tuple()


def test_ambiguous_support_episode_is_rejected_not_tie_broken():
    rejected = data.make_ambiguous_rejected_episode(0)
    supports = [data.freeze_grid(pair["input"]) for pair in rejected["supports"]]
    outputs = [data.freeze_grid(pair["output"]) for pair in rejected["supports"]]
    result = executor.execute_from_raw_grids(supports, outputs, data.freeze_grid(rejected["query_input_grid"]))
    assert result["status"] == "ambiguous"
    assert len(result["consistent_hypotheses"]) > 1


def test_domain_containment_for_generated_suites():
    suites, _ = data.generate_suites()
    for episode in audit.all_accepted_episodes(suites):
        for pair in episode.supports:
            assert data.domain_check(pair.input, episode.hypothesis)[0]
            assert data.domain_check(pair.output, episode.hypothesis)[0]
        assert data.domain_check(episode.query_input_grid, episode.hypothesis)[0]
        assert data.domain_check(episode.query_output_grid, episode.hypothesis)[0]


def test_deterministic_generator_and_reports():
    first, first_rejected = data.generate_suites()
    second, second_rejected = data.generate_suites()
    assert [[episode.to_record() for episode in first[name]] for name in sorted(first)] == [[episode.to_record() for episode in second[name]] for name in sorted(second)]
    assert first_rejected == second_rejected


def test_accepted_rejected_distribution_reporting():
    suites, rejected = data.generate_suites()
    report = audit.generator_audit(suites, rejected)
    assert report["accepted_episode_count"] == 147
    assert report["rejected_episode_count"] == 8
    assert "reachable_region_size" in report["accepted_geometry_distributions"]
    assert "reachable_region_size" in report["rejected_geometry_distributions"]


def test_pairwise_counterbalancing_reporting_exists():
    suites, _ = data.generate_suites()
    report = data.counterbalance_report(audit.all_accepted_episodes(suites))
    assert report["marginal_contingency_summaries"]
    assert report["pairwise_contingency_summaries"]
    assert "convention__x__passable_color" in report["pairwise_contingency_summaries"]


def test_fixed_shortcut_catalog_counts_and_ordering():
    catalog = data.enumerate_shortcut_catalog()
    assert catalog["single_rule_count"] == 11
    assert catalog["pairwise_conjunction_count"] == 45
    assert catalog["lexicographic_rule_count"] == 90
    assert catalog["total_rule_count"] == 146
    names = [rule["name"] for rule in catalog["rules"]]
    assert names == sorted(names)


def test_shortcut_catalog_hash_is_deterministic_and_not_mutated():
    first = data.enumerate_shortcut_catalog()
    second = data.enumerate_shortcut_catalog()
    assert first["shortcut_catalog_sha256"] == second["shortcut_catalog_sha256"]
    suites, _ = data.generate_suites()
    before = first["shortcut_catalog_sha256"]
    _ = audit.shortcut_audit(audit.all_accepted_episodes(suites), first)
    after = data.enumerate_shortcut_catalog()["shortcut_catalog_sha256"]
    assert before == after


def test_wrong_adjacency_is_exactly_eight_connected():
    assert data.ADJACENCY_8 == ("north", "north_east", "east", "south_east", "south", "south_west", "west", "north_west")
    assert set(data.neighbors((3, 3), adjacency="8_connected")) == {
        (3, 2),
        (4, 2),
        (4, 3),
        (4, 4),
        (3, 4),
        (2, 4),
        (2, 3),
        (2, 2),
    }


def test_adjacency_discrimination_coverage_is_at_least_25_percent():
    suites, _ = data.generate_suites()
    report = audit.ablation_report(suites)
    adj = report["adjacency_discrimination"]
    assert adj["adjacency_discrimination_episode_fraction"] >= 0.25
    assert adj["absolute_drop_on_adjacency_slice"] >= 0.30


def test_every_required_ablation_changes_expected_behavior():
    suites, _ = data.generate_suites()
    report = audit.ablation_report(suites)
    for name in (
        "no_propagation",
        "one_step_only",
        "wrong_adjacency_8_connected",
        "barrier_ignored",
        "border_seed_disabled",
        "support_consistency_disabled",
    ):
        assert report["ablations"][name]["passes_relevant_drop_threshold"], name


def test_normal_executor_exact_and_shortcut_threshold():
    suites, _ = data.generate_suites()
    catalog = data.enumerate_shortcut_catalog()
    evaluation = audit.evaluation_report(suites)
    shortcut = audit.shortcut_audit(audit.all_accepted_episodes(suites), catalog)
    assert evaluation["randomized_suite_exact"] == 1.0
    assert evaluation["generator_stress_suite_exact"] == 1.0
    assert shortcut["best_shortcut_exact"] <= data.SHORTCUT_THRESHOLD


def test_protected_prior_hashes_remain_unchanged_during_build_report():
    before = audit.protected_hashes()
    report, *_ = audit.build_report()
    after = audit.protected_hashes()
    assert before == after
    assert report["hard_gates"]["protected_hashes_unchanged"] is True
