from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from vrm.twin_tasks.transport_recolor_r5a0v2_data import (
    EPISODES_PER_QUERY_TYPE_CLASS,
    RECOMBINATION_CONTEXT_TO_OPERATION,
    REQUIRED_QUERY_TYPE_COUNTS,
    R5A0V2Episode,
    all_query_types,
    factor_complete,
    generate_counterfactual_suite,
    generate_recombination_suite,
    partition_query_types,
    query_type_key,
    query_type_preflight_catalog,
    render_scene,
    tensors_from_episodes,
    validate_scene,
)
from vrm.twin_tasks.transport_recolor_r5a0v2_model import (
    SupportConditionedTransportRecolorR5A0V2,
    forced_identity_color_probs,
    forced_identity_transport_probs,
    output_loss,
)
from vrm.twin_tasks.transport_recolor_r5a0v2_oracles import (
    EXPECTED_MIXED_PREDICATES,
    GRID_SIZE,
    HELD_OUT_MIXED_CLASSES,
    MIXED_CLASSES,
    SEALED_CLASSES,
    SEEN_MIXED_CLASSES,
    SINGLE_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    apply_sequence_to_grid,
    operation_function_audit,
)
from vrm.twin_tasks.run_transport_recolor_r5a0v2 import (
    cuda_environment_report,
    evaluate_model,
    recombine_probs,
    selection_key,
    train_one_seed,
)


def test_v1_r5a0_files_and_artifacts_are_present_and_still_collision_stopped():
    root = Path("/mnt/d/VRM")
    for relative in (
        "vrm/twin_tasks/transport_recolor_r5a0_data.py",
        "vrm/twin_tasks/transport_recolor_r5a0_oracles.py",
        "vrm/twin_tasks/transport_recolor_r5a0_model.py",
        "vrm/twin_tasks/run_transport_recolor_r5a0.py",
        "tests/test_transport_recolor_r5a0.py",
        "docs/TRANSPORT_RECOLOR_R5A0.md",
    ):
        assert (root / relative).exists()
    aggregate = json.loads((root / "artifacts/transport_recolor_r5a0_v1/aggregate_seed_summary.json").read_text(encoding="utf-8"))
    assert aggregate["verdict"] == "R5A0_FUNCTION_COLLISION_DETECTED"


def test_cuda_required_and_no_cpu_fallback():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    source = inspect.getsource(cuda_environment_report)
    assert "CUDA_REQUIRED_NOT_AVAILABLE" in source
    assert '"cuda"' in source
    assert "cpu" not in source.lower()


def test_all_13_signatures_are_exact_and_collision_free():
    audit = operation_function_audit()
    assert audit["proof"]["passed"]
    assert not audit["collisions"]
    assert len(audit["operation_entries"]) == len(SINGLE_CLASSES) + len(MIXED_CLASSES)
    assert audit["mixed_signatures_all_unique"]
    assert audit["single_signatures_all_unique"]


def test_expected_analytic_predicates_match_independent_signatures():
    audit = operation_function_audit()
    rows = {entry["class"]: entry for entry in audit["operation_entries"]}
    for operation, predicate in EXPECTED_MIXED_PREDICATES.items():
        assert rows[operation]["original_coordinate_recolor_predicate"] == predicate
        assert rows[operation]["expected_predicate_matches"]


def test_no_heldout_mixed_class_equals_seen_or_single_behavior():
    audit = operation_function_audit()
    assert audit["no_heldout_mixed_signature_equals_seen_mixed_signature"]
    assert audit["no_mixed_signature_equals_single_signature"]


def test_all_128_query_types_are_feasible_across_all_13_classes():
    catalog = query_type_preflight_catalog()
    assert catalog["declared_query_type_count"] == 128
    assert catalog["feasible_query_type_count"] == 128
    assert catalog["all_128_feasible"]
    assert all(set(entry["valid_scene_counts_by_operation_class"]) == set(SEALED_CLASSES) for entry in catalog["entries"])


def test_exact_split_episode_counts_are_declared():
    split = partition_query_types()
    assert {key: len(value) for key, value in split.items()} == REQUIRED_QUERY_TYPE_COUNTS
    assert len(split["train"]) * len(TRAIN_VISIBLE_CLASSES) * EPISODES_PER_QUERY_TYPE_CLASS == 43_200
    assert len(split["dev"]) * len(TRAIN_VISIBLE_CLASSES) * EPISODES_PER_QUERY_TYPE_CLASS == 12_960
    assert len(split["sealed"]) * len(SEALED_CLASSES) * EPISODES_PER_QUERY_TYPE_CLASS == 18_720


def test_split_query_types_are_disjoint():
    split = partition_query_types()
    sets = {name: {query_type_key(item) for item in rows} for name, rows in split.items()}
    assert not sets["train"] & sets["dev"]
    assert not sets["train"] & sets["sealed"]
    assert not sets["dev"] & sets["sealed"]
    assert len(set().union(*sets.values())) == len(all_query_types())


def test_every_visual_factor_appears_in_every_split():
    split = partition_query_types()
    assert all(factor_complete(rows) for rows in split.values())


def test_scene_generator_rejects_empty_or_full_recolor_subsets():
    empty = tuple(tuple(0 for _ in range(GRID_SIZE)) for _ in range(GRID_SIZE))
    ok, reasons = validate_scene(empty)
    assert not ok
    assert "empty_foreground" in reasons
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for x, y in ((2, 2), (3, 2), (2, 3), (3, 3)):
        grid[y][x] = 1
    ok, reasons = validate_scene(tuple(tuple(row) for row in grid))
    assert not ok
    assert any("recolor_not_proper_subset" in reason for reason in reasons)


def test_scene_generator_rejects_mixed_output_collision():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[3][3] = 1
    ok, reasons = validate_scene(tuple(tuple(row) for row in grid))
    assert not ok
    assert "mixed_output_collision" in reasons


def test_transport_head_cannot_read_query_source_color():
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0V2.encode_pairs)
    assert "query_grids" not in source
    assert "support_input_grids" in source
    assert "support_output_grids" in source


def test_color_transition_head_cannot_read_target_or_support_outputs():
    signature = inspect.signature(SupportConditionedTransportRecolorR5A0V2.color_transition_from_flags)
    assert set(signature.parameters) == {"self", "query_source_colors", "source_target_flags"}
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0V2.color_transition_from_flags)
    assert "support_output" not in source
    assert "target_grids" not in source
    assert "operation_class" not in source


def test_final_output_probabilities_equal_sum_v_a_times_b():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    split = partition_query_types()
    episode = R5A0V2Episode(
        episode_id="unit",
        split="unit",
        operation_class="C->K",
        query_type=split["train"][0],
        counterfactual_group_id=None,
        query=render_scene(split["train"][0], 7, "unit"),
        supports=tuple(),
        target=tuple(),
    )
    from vrm.twin_tasks.transport_recolor_r5a0v2_data import make_episode

    episode = make_episode("unit", split["train"][0], "C->K", 0)
    batch = tensors_from_episodes([episode], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    recomputed = recombine_probs(out["transport_attention"], out["color_transition_probs"])
    assert torch.allclose(out["output_probs"], recomputed, atol=1e-7)


def test_no_generic_paint_bypass_exists():
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0V2.forward)
    assert "torch.einsum" in source
    assert "output_layer" not in source
    assert "paint" not in source.lower()


def test_training_loss_is_final_output_cross_entropy_only():
    source = inspect.getsource(output_loss)
    for forbidden in ("transport", "transition", "sequence", "operation", "aux", "sparsity", "connection"):
        assert forbidden not in source


def test_interventions_alter_only_requested_factor_shapes():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    from vrm.twin_tasks.transport_recolor_r5a0v2_data import make_episode

    episode = make_episode("unit", partition_query_types()["train"][0], "C->L", 0)
    batch = tensors_from_episodes([episode], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    identity_color = forced_identity_color_probs(batch["query_grids"], out["transport_attention"])
    identity_transport = forced_identity_transport_probs(out["color_transition_probs"])
    assert identity_color.shape == out["output_probs"].shape
    assert identity_transport.shape == out["output_probs"].shape
    assert not torch.equal(identity_color.argmax(dim=-1), batch["target_grids"])
    assert not torch.equal(identity_transport.argmax(dim=-1), batch["target_grids"])


def test_counterfactual_groups_preserve_query_and_support_inputs():
    split = partition_query_types()
    rows = generate_counterfactual_suite(split, group_count=3)
    by_group: dict[str, list] = {}
    for row in rows:
        by_group.setdefault(str(row.counterfactual_group_id), []).append(row)
    for group in by_group.values():
        assert len(group) == len(MIXED_CLASSES)
        assert len({json.dumps(row.query) for row in group}) == 1
        assert len({json.dumps([pair.input for pair in row.supports]) for row in group}) == 1
        assert len({json.dumps(row.target) for row in group}) == len(MIXED_CLASSES)


def test_recombination_groups_preserve_query_and_support_inputs():
    split = partition_query_types()
    rows = generate_recombination_suite(split, group_count=3)
    by_group: dict[str, list] = {}
    for row in rows:
        by_group.setdefault(str(row.counterfactual_group_id), []).append(row)
    for group in by_group.values():
        assert len(group) == len(RECOMBINATION_CONTEXT_TO_OPERATION)
        assert len({json.dumps(row.query) for row in group}) == 1
        assert len({json.dumps([pair.input for pair in row.supports]) for row in group}) == 1


def test_recombined_output_is_computed_exactly_from_selected_a_and_b():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    from vrm.twin_tasks.transport_recolor_r5a0v2_data import make_episode

    qtype = partition_query_types()["train"][0]
    source = make_episode("unit", qtype, "C", 0)
    transition = make_episode("unit", qtype, "L->H", 0)
    batch = tensors_from_episodes([source, transition], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    recombined = recombine_probs(out["transport_attention"][:1], out["color_transition_probs"][1:2])
    manual = torch.einsum(
        "buv,buvc->buc",
        out["transport_attention"][:1].reshape(1, 49, 49),
        out["color_transition_probs"][1:2].reshape(1, 49, 49, 9),
    ).reshape(1, 7, 7, 9)
    assert torch.allclose(recombined, manual, atol=1e-7)


def test_recombination_has_no_effect_on_loss_selection_or_core_verdict():
    runner_source = inspect.getsource(selection_key)
    train_source = inspect.getsource(train_one_seed)
    assert "head_recombination" not in runner_source
    assert "head_recombination" not in train_source


def test_sealed_metrics_remain_unread_inside_checkpoint_selection():
    source = inspect.getsource(train_one_seed)
    assert "episodes_by_split" not in source
    assert "final_sealed_metrics" not in source


def test_reported_metrics_match_hand_built_oracle_fixture():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    from vrm.twin_tasks.transport_recolor_r5a0v2_data import make_episode

    episode = make_episode("unit", partition_query_types()["train"][0], "K->C", 0)
    metrics = evaluate_model(model, [episode], device, 1)
    row = metrics["by_operation_class"]["K->C"]
    assert row["exact"] == 1.0
    assert row["foreground_accuracy"] == 1.0
    assert row["transport_exact_source_coordinate_accuracy"] == 1.0
    assert row["color_transition_accuracy_given_oracle_source"] == 1.0
    assert apply_sequence_to_grid(episode.query, "K->C") == episode.target
