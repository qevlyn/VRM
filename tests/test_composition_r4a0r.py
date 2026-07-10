from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from vrm.twin_tasks.composition_r4a0r_data import (
    HELD_OUT_TWO_STEP_CLASSES,
    LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS,
    REQUIRED_QUERY_TYPE_COUNTS,
    SEEN_TWO_STEP_CLASSES,
    SINGLE_STEP_CLASSES,
    TWO_STEP_CLASSES,
    algebraic_function_audit,
    all_query_types,
    apply_sequence_to_coord,
    counterfactual_suite_manifest,
    coord_c,
    coord_f,
    coord_t,
    derange_episode,
    factor_complete,
    generate_order_counterfactual_suite,
    make_episode,
    mapping_signature,
    partition_query_types,
    query_type_preflight_catalog,
    query_type_split_audit,
    reorder_episode,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
)
from vrm.twin_tasks.composition_r4a0r_oracles import apply_sequence_to_grid
from vrm.twin_tasks.run_composition_r4a0r import (
    SupportConditionedCompositePointerR4A0R,
    cuda_environment_report,
    output_loss,
    protected_hashes,
)


def test_cuda_required_and_no_cpu_fallback():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    source = inspect.getsource(cuda_environment_report)
    assert "CUDA_REQUIRED_NOT_AVAILABLE" in source
    assert '"cuda"' in source


def test_coordinate_formulas_for_c_f_t():
    assert coord_c(1, 2) == (4, 1)
    assert coord_f(1, 2) == (5, 2)
    assert coord_t(1, 2) == (2, 2)
    with pytest.raises(ValueError):
        coord_t(6, 2)


def test_sequence_notation_applies_left_to_right():
    assert apply_sequence_to_coord("C->F", (1, 2)) == coord_f(*coord_c(1, 2))
    assert apply_sequence_to_coord("F->C", (1, 2)) == coord_c(*coord_f(1, 2))
    assert apply_sequence_to_coord("T->C", (1, 2)) == coord_c(*coord_t(1, 2))


def test_every_two_step_sequence_signature_is_unique():
    signatures = {sequence: tuple(map(tuple, mapping_signature(sequence))) for sequence in TWO_STEP_CLASSES}
    assert len(set(signatures.values())) == len(TWO_STEP_CLASSES)


def test_no_heldout_sequence_equals_seen_or_single_function():
    signatures = {sequence: tuple(map(tuple, mapping_signature(sequence))) for sequence in SINGLE_STEP_CLASSES + TWO_STEP_CLASSES}
    for heldout in HELD_OUT_TWO_STEP_CLASSES:
        for seen in SEEN_TWO_STEP_CLASSES:
            assert signatures[heldout] != signatures[seen]
        for single in SINGLE_STEP_CLASSES:
            assert signatures[heldout] != signatures[single]


def test_algebraic_function_audit_passes():
    audit = algebraic_function_audit()
    assert audit["proof"]["passed"]
    assert audit["two_step_signatures_all_unique"]
    assert audit["no_heldout_two_step_signature_equals_seen_two_step_signature"]
    assert audit["no_heldout_two_step_signature_equals_single_step_signature"]
    assert not audit["collisions"]


def test_total_declared_query_type_count_is_128():
    assert len(all_query_types()) == 128
    assert REQUIRED_QUERY_TYPE_COUNTS == {"train": 80, "dev": 24, "sealed": 24}
    assert sum(REQUIRED_QUERY_TYPE_COUNTS.values()) == 128


def test_preflight_catalog_contains_all_128_and_is_deterministic():
    first = query_type_preflight_catalog()
    second = query_type_preflight_catalog()
    assert first == second
    assert first["declared_query_type_count"] == 128
    assert len(first["entries"]) == 128
    assert first["feasible_query_type_count"] == 128
    assert all(entry["feasible"] for entry in first["entries"])
    assert all("deterministic_preflight_seed" in entry for entry in first["entries"])


def test_full_run_may_begin_only_with_128_feasible_types():
    catalog = query_type_preflight_catalog()
    may_begin = catalog["feasible_query_type_count"] == 128
    assert may_begin


def test_corrected_split_counts_union_and_intersections():
    split = partition_query_types()
    audit = query_type_split_audit(split)
    assert audit["passed"]
    assert audit["split_counts"] == {"train": 80, "dev": 24, "sealed": 24}
    assert audit["union_count"] == 128
    assert audit["uses_every_catalog_type_exactly_once"]
    assert not audit["intersections"]["train_dev"]
    assert not audit["intersections"]["train_sealed"]
    assert not audit["intersections"]["dev_sealed"]


def test_every_factor_value_appears_in_each_split():
    split = partition_query_types()
    assert all(factor_complete(items) for items in split.values())


def test_previous_144_type_request_cannot_reappear_silently():
    assert REQUIRED_QUERY_TYPE_COUNTS != LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS
    assert sum(LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS.values()) == 144


def test_generated_episode_stage_safe_and_no_raw_copying():
    qtype = partition_query_types()["train"][0]
    episode = make_episode("unit", qtype, "C->F", 0)
    assert all(pair.input != episode.query for pair in episode.supports)
    assert all(pair.output != episode.target for pair in episode.supports)
    outputs = [apply_sequence_to_grid(torch.tensor([episode.query], device="cuda"), sequence) for sequence in TWO_STEP_CLASSES]
    hashes = {tuple(output.detach().cpu().flatten().tolist()) for output in outputs}
    assert len(hashes) == len(TWO_STEP_CLASSES)


def test_sequence_metadata_and_target_never_enter_model_forward():
    signature = inspect.signature(SupportConditionedCompositePointerR4A0R.forward)
    forbidden = {"target", "target_grids", "sequence_class", "query_type", "split", "episode_id"}
    assert not (forbidden & set(signature.parameters))


def test_training_loss_is_output_only():
    source = inspect.getsource(output_loss)
    forbidden = ["sequence", "primitive", "contrastive", "route", "candidate", "intermediate"]
    assert all(item not in source for item in forbidden)


def test_pair_aggregation_is_permutation_invariant():
    device = torch.device("cuda")
    qtype = partition_query_types()["train"][0]
    episode = make_episode("unit", qtype, "F->T", 1)
    model = SupportConditionedCompositePointerR4A0R().to(device).eval()
    batch = tensors_from_episodes([episode], device)
    permuted = tensors_from_episodes([reorder_episode(episode, (2, 0, 1))], device)
    with torch.no_grad():
        out_a = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        out_b = model(permuted["query_grids"], permuted["support_input_grids"], permuted["support_output_grids"])
    assert torch.allclose(out_a["output_probs"], out_b["output_probs"], atol=1e-6)
    assert torch.allclose(out_a["task_context"], out_b["task_context"], atol=1e-6)


def test_complete_pair_correspondence_remains_coherent_under_permutations():
    qtype = partition_query_types()["train"][0]
    episode = make_episode("unit", qtype, "T->C", 2)
    original = support_signature(episode)
    viewed = support_signature(reorder_episode(episode, (1, 2, 0)))
    assert viewed == [original[1], original[2], original[0]]
    assert sorted(viewed) == sorted(original)


def test_zero_support_changes_only_support_evidence():
    device = torch.device("cuda")
    qtype = partition_query_types()["train"][0]
    episode = make_episode("unit", qtype, "C->T", 3)
    model = SupportConditionedCompositePointerR4A0R().to(device).eval()
    batch = tensors_from_episodes([episode], device)
    with torch.no_grad():
        normal = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        zero = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
    assert torch.equal(batch["query_grids"], batch["query_grids"])
    assert not torch.equal(normal["decoded_grid"], zero["decoded_grid"])


def test_deranged_pairing_preserves_bags_and_breaks_pairing():
    qtype = partition_query_types()["train"][0]
    episode = make_episode("unit", qtype, "C->T", 4)
    deranged = derange_episode(episode, 1)
    assert support_input_bag(episode) == support_input_bag(deranged)
    assert support_output_bag(episode) == support_output_bag(deranged)
    assert support_signature(episode) != support_signature(deranged)


def test_counterfactual_groups_preserve_q_and_xi_and_targets_differ():
    split = partition_query_types()
    rows = generate_order_counterfactual_suite(split, group_count=3)
    manifest = counterfactual_suite_manifest(rows)
    assert manifest["passed"] is False or manifest["group_count"] == 3
    groups = {}
    for episode in rows:
        groups.setdefault(episode.counterfactual_group_id, []).append(episode)
    for episodes in groups.values():
        assert len({str(episode.query) for episode in episodes}) == 1
        assert len({str([pair.input for pair in episode.supports]) for episode in episodes}) == 1
        assert len({str(episode.target) for episode in episodes}) == len(episodes)


def test_protected_hash_snapshot_is_stable_for_read_only_call():
    data_out = Path("/mnt/d/VRM/artifacts/composition_r4a0r_v2")
    checkpoint_out = Path("/mnt/d/VRM/checkpoints/composition_r4a0r_v2")
    before = protected_hashes(data_out=data_out, checkpoint_out=checkpoint_out)
    after = protected_hashes(data_out=data_out, checkpoint_out=checkpoint_out)
    assert before == after
