from __future__ import annotations

import inspect
from pathlib import Path

import torch

from vrm.twin_tasks.transport_recolor_r5a0_data import (
    R5A0Episode,
    all_query_types,
    derange_episode,
    make_supports,
    query_type_key,
    reorder_episode,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
)
from vrm.twin_tasks.transport_recolor_r5a0_model import (
    SupportConditionedTransportRecolorR5A0,
    forced_identity_color_probs,
    forced_identity_transport_probs,
    output_loss,
)
from vrm.twin_tasks.transport_recolor_r5a0_oracles import (
    HELD_OUT_MIXED_CLASSES,
    INPUT_COLORS,
    MIXED_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    apply_sequence_to_cell,
    apply_sequence_to_grid,
    operation_function_audit,
)
from vrm.twin_tasks.run_transport_recolor_r5a0 import cuda_environment_report, protected_hashes


def test_cuda_required_and_no_cpu_fallback():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    source = inspect.getsource(cuda_environment_report)
    assert "CUDA_REQUIRED_NOT_AVAILABLE" in source
    assert '"cuda"' in source


def test_declared_mixed_function_audit_detects_exact_collisions():
    audit = operation_function_audit()
    assert not audit["proof"]["passed"]
    collisions = {tuple(sorted((row["left"], row["right"]))) for row in audit["collisions"]}
    assert tuple(sorted(("F->U", "U->F"))) in collisions
    assert tuple(sorted(("L->C", "C->U"))) in collisions


def test_no_heldout_collision_can_be_ignored():
    audit = operation_function_audit()
    signatures = {key: str(value) for key, value in audit["mapping_signatures"].items()}
    assert signatures["F->U"] == signatures["U->F"]
    assert signatures["L->C"] == signatures["C->U"]
    assert any(row["left"] in HELD_OUT_MIXED_CLASSES or row["right"] in HELD_OUT_MIXED_CLASSES for row in audit["collisions"])


def test_all_query_type_catalog_is_declared_128_even_before_collision_stop():
    assert len(all_query_types()) == 128
    assert len({query_type_key(item) for item in all_query_types()}) == 128


def test_recolor_primitive_changes_proper_nonempty_subset_on_support_scene():
    support = make_supports("L")[0].input
    out = apply_sequence_to_grid(support, "L")
    changed = sum(1 for y, row in enumerate(support) for x, value in enumerate(row) if value != 0 and out[y][x] == value + 4)
    foreground = sum(1 for row in support for value in row if value != 0)
    assert 0 < changed < foreground


def test_input_foreground_colors_restricted_to_one_through_four():
    support = make_supports("C")[0].input
    assert {value for row in support for value in row if value != 0}.issubset(set(INPUT_COLORS))


def test_recolor_outputs_obey_private_oracle():
    assert apply_sequence_to_cell("L", 2, 4, 3) == (2, 4, 7)
    assert apply_sequence_to_cell("L", 4, 4, 3) == (4, 4, 3)
    assert apply_sequence_to_cell("U", 4, 2, 1) == (4, 2, 5)
    assert apply_sequence_to_cell("U", 4, 4, 1) == (4, 4, 1)


def test_no_forbidden_metadata_enters_model_forward():
    signature = inspect.signature(SupportConditionedTransportRecolorR5A0.forward)
    forbidden = {"operation_class", "sequence_class", "query_type", "split", "episode_id", "target", "target_grids", "counterfactual_group_id"}
    assert not (forbidden & set(signature.parameters))


def test_transport_head_does_not_receive_query_source_color_in_code_path():
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0.encode_pairs)
    assert "query_grids" not in source
    assert "support_input_grids" in source
    assert "support_output_grids" in source


def test_color_transition_head_does_not_receive_target_or_operation_label():
    signature = inspect.signature(SupportConditionedTransportRecolorR5A0.color_transition_from_flags)
    assert set(signature.parameters) == {"self", "query_grids", "recolor_flags"}


def test_final_output_probabilities_equal_transport_times_transition():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0().to(device).eval()
    supports = make_supports("C->L")
    query = supports[0].input
    episode_like = type("E", (), {"query": query, "supports": supports, "target": apply_sequence_to_grid(query, "C->L")})
    batch = tensors_from_episodes([episode_like], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    a = out["transport_attention"].reshape(1, 49, 49)
    b = out["color_transition_probs"].reshape(1, 49, 49, 9)
    recomputed = torch.einsum("buv,buvc->buc", a, b).reshape(1, 7, 7, 9)
    assert torch.allclose(out["output_probs"], recomputed, atol=1e-7)


def test_no_generic_paint_bypass_exists_and_background_constraint_is_enforced():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0().to(device).eval()
    supports = make_supports("L")
    query = supports[0].input
    grid = [list(row) for row in query]
    grid[2][2] = 0
    episode_like = type("E", (), {"query": tuple(tuple(row) for row in grid), "supports": supports, "target": apply_sequence_to_grid(tuple(tuple(row) for row in grid), "L")})
    batch = tensors_from_episodes([episode_like], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    b = out["color_transition_probs"].reshape(1, 49, 49, 9)
    bg_source = 2 * 7 + 2
    assert torch.all(b[0, :, bg_source, 0] == 1.0)
    assert torch.all(b[0, :, bg_source, 1:] == 0.0)


def test_training_loss_is_output_only_cross_entropy():
    source = inspect.getsource(output_loss)
    forbidden = ["sequence", "operation", "primitive", "transport", "transition", "contrastive", "modularity", "sparsity"]
    assert all(item not in source for item in forbidden)


def test_pair_permutation_derangement_helpers_are_coherent():
    supports = make_supports("C->L")
    episode = R5A0Episode(
        episode_id="unit",
        split="unit",
        operation_class="C->L",
        query_type=all_query_types()[0],
        counterfactual_group_id=None,
        query=supports[0].input,
        supports=supports,
        target=apply_sequence_to_grid(supports[0].input, "C->L"),
    )
    original = support_signature(episode)
    viewed = support_signature(reorder_episode(episode, (2, 0, 1)))
    assert viewed == [original[2], original[0], original[1]]
    assert sorted(viewed) == sorted(original)
    deranged = derange_episode(episode, 1)
    assert support_input_bag(episode) == support_input_bag(deranged)
    assert support_output_bag(episode) == support_output_bag(deranged)
    assert support_signature(episode) != support_signature(deranged)


def test_forced_identity_transport_and_color_are_exact_on_fixture():
    device = torch.device("cuda")
    model = SupportConditionedTransportRecolorR5A0().to(device).eval()
    supports = make_supports("L")
    query = supports[0].input
    episode_like = type("E", (), {"query": query, "supports": supports, "target": apply_sequence_to_grid(query, "L")})
    batch = tensors_from_episodes([episode_like], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    identity_color = forced_identity_color_probs(batch["query_grids"], out["transport_attention"])
    identity_transport = forced_identity_transport_probs(out["color_transition_probs"])
    assert identity_color.shape == out["output_probs"].shape
    assert identity_transport.shape == out["output_probs"].shape
    assert torch.equal(identity_color.argmax(dim=-1), batch["query_grids"])
    assert torch.equal(identity_transport.argmax(dim=-1), batch["target_grids"])


def test_negative_recombination_controls_cannot_influence_training_source():
    import vrm.twin_tasks.run_transport_recolor_r5a0 as runner

    source = inspect.getsource(runner)
    assert "R5A0_FUNCTION_COLLISION_DETECTED" in source
    assert "optimizer" not in source.lower()


def test_protected_hash_snapshot_stable_for_read_only_call():
    data_out = Path("/mnt/d/VRM/artifacts/transport_recolor_r5a0_v1")
    checkpoint_out = Path("/mnt/d/VRM/checkpoints/transport_recolor_r5a0_v1")
    before = protected_hashes(data_out, checkpoint_out)
    after = protected_hashes(data_out, checkpoint_out)
    assert before == after


def test_mixed_class_list_contains_requested_classes():
    assert set(MIXED_CLASSES) == {"C->L", "F->U", "L->F", "U->C", "L->C", "U->F", "F->L", "C->U"}
    assert set(HELD_OUT_MIXED_CLASSES).isdisjoint({"C->L", "F->U", "L->F", "U->C"} - set(TRAIN_VISIBLE_CLASSES))
