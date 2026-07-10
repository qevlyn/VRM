from __future__ import annotations

import inspect
import json

from vrm.twin_tasks import transport_recolor_r5a1a_preregistration as prereg


def test_preregistration_is_control_design_only_and_neutral():
    report = prereg.build_report()
    assert report["status"] == "CONTROL_DESIGN_AND_PREFLIGHT_ONLY"
    assert report["prohibited_actions_in_this_task"] == {
        "training_started": False,
        "model_architecture_changed": False,
        "v2_grammar_or_split_changed": False,
        "v2_checkpoints_changed": False,
        "historical_v2_verdict_changed": False,
    }
    neutral = report["neutrality_statement"].lower()
    assert "does not state or assume" in neutral
    assert "expected to pass or fail" in neutral


def test_shared_renderer_allowance_and_forbidden_interfaces_are_explicit():
    report = prereg.build_report()
    allowed = " ".join(report["shared_renderer_allowed_interfaces"]).lower()
    forbidden = " ".join(report["shared_renderer_forbidden_interfaces"]).lower()
    assert "generic cross-attention" in allowed
    assert "generic attention" in report["generic_attention_policy"].lower()
    assert "explicit source-coordinate prediction" in forbidden
    assert "explicit pointer distribution" in forbidden
    assert "copy gate" in forbidden
    assert "explicit transport representation" in forbidden
    assert "explicit color-transition representation" in forbidden
    assert "factor-labelled latent channels" in forbidden
    assert "auxiliary geometry/recolor losses" in forbidden


def test_pre_registered_outcome_categories_are_exact():
    assert prereg.OUTCOME_CATEGORIES == (
        "CONTROL_UNDERFIT_OR_INVALID",
        "PRACTICAL_FACTORIZATION_ADVANTAGE_SUPPORTED",
        "EXPLICIT_FACTORIZATION_NOT_NECESSARY_ON_CURRENT_GRAMMAR",
        "CAPACITY_SENSITIVE_OR_INCONCLUSIVE",
    )
    report = prereg.build_report()
    assert tuple(report["pre_registered_outcome_categories"]) == prereg.OUTCOME_CATEGORIES
    outcomes = [row["outcome"] for row in report["future_outcome_classification_rules"]["evaluation_order"]]
    assert set(outcomes) == set(prereg.OUTCOME_CATEGORIES)
    assert outcomes[0] == "CONTROL_UNDERFIT_OR_INVALID"


def test_thresholds_are_fixed_and_justified_by_dataset_granularity():
    report = prereg.build_report()
    thresholds = report["pre_registered_thresholds"]
    train_gate = thresholds["train_fit_adequacy_gate"]["requirements"]
    assert train_gate["train_visible_overall_exact_min"] == 0.995
    assert train_gate["train_visible_seen_mixed_macro_exact_min"] == 0.995
    assert train_gate["each_train_visible_operation_exact_min"] == 0.99
    assert thresholds["practically_meaningful_heldout_mixed_effect_floor"]["absolute_exact_gap_vs_v2_min"] == 0.03
    assert thresholds["equivalence_margin"]["absolute_exact_gap_vs_v2_max"] == 0.01
    assert thresholds["seed_consistency_criterion"]["within_variant_max_minus_min_heldout_mixed_exact_max"] == 0.02
    granularity = report["fixed_benchmark_anchor"]["metric_granularity"]
    assert granularity["train_overall_exact"] == 1 / 43_200
    assert granularity["sealed_heldout_mixed_macro_episode_granularity"] == 1 / 5_760


def test_sealed_data_discipline_forbids_adaptive_design_changes():
    report = prereg.build_report()
    forbidden = report["sealed_data_discipline"]["forbidden_uses"]
    for item in (
        "resize a shared variant",
        "add a missing decoder component",
        "alter optimizer or schedule",
        "change train-fit threshold",
        "change effect floor",
        "change equivalence margin",
        "select between architecture variants",
    ):
        assert item in forbidden
    assert "new experiment version" in report["sealed_data_discipline"]["post_sealed_modification_rule"]


def test_planned_variants_are_capacity_controls_not_factor_interfaces():
    report = prereg.build_report()
    matched = report["planned_shared_variants"]["shared_matched"]
    large = report["planned_shared_variants"]["shared_large"]
    assert matched["semantic_factor_interface_allowed"] is False
    assert large["semantic_factor_interface_allowed"] is False
    assert matched["v2_parameter_anchor"] == report["fixed_benchmark_anchor"]["v2_trainable_parameter_count"]
    assert large["allowed_capacity_range_inclusive"][0] > matched["allowed_capacity_range_inclusive"][1]


def test_preregistration_module_does_not_implement_training_or_model():
    source = inspect.getsource(prereg)
    lowered = source.lower()
    assert "torch.optim" not in lowered
    assert "loss.backward" not in lowered
    assert "class shared" not in lowered
    assert "sealed" in lowered
    assert "preregister" in lowered or "preregistration" in lowered


def test_report_is_json_deterministic():
    first = prereg.build_report()
    second = prereg.build_report()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
