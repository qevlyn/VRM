from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from vrm.twin_tasks.mixed_router_r3b2_data import (
    CORE_FAMILIES,
    FAMILY_HARD_NEGATIVE,
    FAMILY_RETRIEVAL,
    FAMILY_TRANSFORM,
    boundary_candidate_pool_manifest,
    counterfactual_group_manifest,
    family_balance_report,
    generate_boundary_pool,
    generate_split,
    make_counterfactual_group,
    partition_query_types,
    query_type_key,
    reorder_episode,
    support_signature,
    tensors_from_episodes,
)
from vrm.twin_tasks.mixed_router_r3b2_gate import (
    EvidenceGateR3B2,
    FORBIDDEN_GATE_FORWARD_KEYS,
    MixedRouterR3B2,
    nll_per_episode,
    router_output_loss,
)
from vrm.twin_tasks.run_mixed_router_r3b2 import (
    DEFAULT_CHECKPOINT_OUT,
    DEFAULT_OUT,
    boundary_audit_report,
    boundary_partition,
    endpoint_equivalence,
    failed_gates_for_seed,
    protected_hashes,
    reader_checkpoint_for_seed,
    transform_checkpoint_for_seed,
)


SEED = 20260628


def cuda_device() -> torch.device:
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    return torch.device("cuda")


def small_group():
    qtype = partition_query_types(SEED)["dev"][0]
    return make_counterfactual_group("unit", qtype, 12345)


def small_router() -> MixedRouterR3B2:
    device = cuda_device()
    return MixedRouterR3B2(reader_checkpoint_for_seed(SEED), transform_checkpoint_for_seed(SEED), device).to(device)


def test_cuda_is_required_and_no_cpu_fallback_exists():
    assert torch.cuda.is_available()
    source = inspect.getsource(cuda_device)
    assert "torch.device(\"cuda\")" in source


def test_protected_prior_hashes_are_stable_during_inspection():
    before = protected_hashes()
    after = protected_hashes()
    assert before == after
    assert str(DEFAULT_OUT) not in before
    assert str(DEFAULT_CHECKPOINT_OUT) not in before


def test_frozen_wrappers_reproduce_underlying_expert_outputs():
    router = small_router()
    device = cuda_device()
    batch = tensors_from_episodes(small_group(), device)
    with torch.no_grad():
        wrapped_reader = router.reader_expert(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        direct_reader = router.reader_expert.reader(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        wrapped_transform = router.transform_expert(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        direct_transform = router.transform_expert.transform(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    assert torch.equal(wrapped_reader["reader_probs"], direct_reader["retrieved_color_probabilities"])
    assert torch.equal(wrapped_transform["transform_probs"], direct_transform["output_probs"])


def test_expert_parameters_get_zero_gradients_and_gate_gets_signal():
    router = small_router()
    router.train()
    device = cuda_device()
    batch = tensors_from_episodes(small_group(), device)
    out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss = router_output_loss(out["soft_probs"], batch["target_grids"])
    loss.backward()
    assert all(parameter.grad is None for parameter in router.expert_parameters())
    gate_norm = sum(float(parameter.grad.detach().abs().sum().item()) for parameter in router.gate_parameters() if parameter.grad is not None)
    assert gate_norm > 0.0


def test_endpoint_equivalence_with_forced_gate():
    router = small_router()
    report = endpoint_equivalence(router, small_group(), cuda_device(), batch_size=3)
    assert report["force_g_1_reader_max_abs_difference"] <= 1e-6
    assert report["force_g_0_transform_max_abs_difference"] <= 1e-6


def test_gate_initialization_mean_is_half():
    gate = EvidenceGateR3B2().to(cuda_device())
    evidence = torch.randn(17, gate.config.evidence_dim, device=cuda_device())
    with torch.no_grad():
        values = gate(evidence)
    assert torch.allclose(values, torch.full_like(values, 0.5), atol=1e-7)


def test_gate_forward_accepts_only_evidence_not_metadata_or_targets():
    signature = inspect.signature(EvidenceGateR3B2.forward)
    assert list(signature.parameters) == ["self", "evidence"]
    assert not (FORBIDDEN_GATE_FORWARD_KEYS & set(signature.parameters))


def test_router_loss_is_only_soft_mixture_output_loss():
    source = inspect.getsource(router_output_loss)
    forbidden = ["route_family", "true_route", "reader_nll", "transform_nll", "entropy", "contrastive", "candidate"]
    assert all(item not in source for item in forbidden)


def test_query_type_splits_are_disjoint_and_factor_complete():
    splits = partition_query_types(SEED)
    sets = {split: {query_type_key(item) for item in values} for split, values in splits.items()}
    assert sets["train"].isdisjoint(sets["dev"])
    assert sets["train"].isdisjoint(sets["sealed"])
    assert sets["dev"].isdisjoint(sets["sealed"])
    assert {split: len(values) for split, values in splits.items()} == {"train": 120, "dev": 30, "sealed": 30}


def test_splits_are_family_balanced_and_all_types_have_core_families():
    splits = partition_query_types(SEED)
    episodes = {"unit": generate_split("unit", splits["dev"][:3], groups_per_query_type=2)}
    report = family_balance_report(episodes)["unit"]
    assert report["passed"]
    manifest = counterfactual_group_manifest(episodes)["unit"]
    assert manifest["passed"]
    assert {episode.route_family for episode in episodes["unit"]} == set(CORE_FAMILIES)


def test_support_pair_permutation_remains_coherent():
    episode = small_group()[0]
    reordered = reorder_episode(episode, (2, 0, 1))
    original = support_signature(episode)
    viewed = support_signature(reordered)
    assert viewed == [original[2], original[0], original[1]]
    assert sorted(viewed) == sorted(original)


def test_all_six_permutations_preserve_logical_correspondence():
    episode = small_group()[0]
    original = sorted(support_signature(episode))
    seen = set()
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        viewed = reorder_episode(episode, order)
        assert sorted(support_signature(viewed)) == original
        seen.add(tuple(support_signature(viewed)))
    assert len(seen) == 6


def test_reader_disabled_changes_reader_probabilities_for_mix_only():
    router = small_router()
    device = cuda_device()
    batch = tensors_from_episodes(small_group(), device)
    with torch.no_grad():
        normal = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        disabled = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], reader_disabled=True)
    assert torch.equal(normal["reader_probs"], disabled["reader_probs"])
    assert torch.equal(normal["transform_probs"], disabled["transform_probs"])
    assert torch.allclose(normal["gate"], disabled["gate"])
    assert not torch.equal(normal["reader_probs_for_mix"], disabled["reader_probs_for_mix"])


def test_transform_disabled_changes_transform_probabilities_for_mix_only():
    router = small_router()
    device = cuda_device()
    batch = tensors_from_episodes(small_group(), device)
    with torch.no_grad():
        normal = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        disabled = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_disabled=True)
    assert torch.equal(normal["reader_probs"], disabled["reader_probs"])
    assert torch.equal(normal["transform_probs"], disabled["transform_probs"])
    assert torch.allclose(normal["gate"], disabled["gate"])
    assert not torch.equal(normal["transform_probs_for_mix"], disabled["transform_probs_for_mix"])


def test_counterfactual_groups_keep_query_identical_while_support_changes():
    rows = small_group()
    assert {row.route_family for row in rows} == {FAMILY_RETRIEVAL, FAMILY_TRANSFORM, FAMILY_HARD_NEGATIVE}
    assert len({str(row.query) for row in rows}) == 1
    assert len({str([(pair.input, pair.output) for pair in row.supports]) for row in rows}) == 3


def test_boundary_pool_is_audit_only_and_not_training_data():
    pool = generate_boundary_pool(SEED, count=120)
    manifest = boundary_candidate_pool_manifest(pool)
    assert manifest["audit_only"]
    assert manifest["contributes_to_training"] is False
    assert len(pool) == 120


def test_boundary_partition_uses_post_inference_branch_outcomes():
    details = [
        {"reader_exact": True, "transform_exact": False, "reader_nll": 0.01, "transform_nll": 1.0, "selected_branch": "reader", "hard_exact": True, "route_utility_agreement": True, "routing_regret": 0.0, "gate": 0.9, "soft_hard_disagree": False},
        {"reader_exact": False, "transform_exact": True, "reader_nll": 1.0, "transform_nll": 0.01, "selected_branch": "transform", "hard_exact": True, "route_utility_agreement": True, "routing_regret": 0.0, "gate": 0.1, "soft_hard_disagree": False},
        {"reader_exact": True, "transform_exact": True, "reader_nll": 0.01, "transform_nll": 0.01, "selected_branch": "reader", "hard_exact": True, "route_utility_agreement": None, "routing_regret": 0.0, "gate": 0.6, "soft_hard_disagree": False},
        {"reader_exact": False, "transform_exact": False, "reader_nll": 1.0, "transform_nll": 1.2, "selected_branch": "reader", "hard_exact": False, "route_utility_agreement": True, "routing_regret": 0.0, "gate": 0.7, "soft_hard_disagree": False},
    ]
    partitions = boundary_partition(details)
    assert len(partitions["BOTH_GOOD"]) == 1
    assert len(partitions["BOTH_BAD"]) == 1
    report = boundary_audit_report(details)
    assert report["partitions_contribute_to_training"] is False


def test_metric_fixture_for_route_utility_and_regret():
    reader = torch.zeros(2, 1, 1, 10, device=cuda_device())
    transform = torch.zeros_like(reader)
    target = torch.tensor([[[1]], [[2]]], device=cuda_device())
    reader[0, 0, 0, 1] = 0.9
    reader[0, 0, 0, 2] = 0.1
    transform[0, 0, 0, 1] = 0.2
    transform[0, 0, 0, 2] = 0.8
    reader[1, 0, 0, 2] = 0.2
    reader[1, 0, 0, 1] = 0.8
    transform[1, 0, 0, 2] = 0.9
    transform[1, 0, 0, 1] = 0.1
    rnll = nll_per_episode(reader, target)
    tnll = nll_per_episode(transform, target)
    assert rnll[0] < tnll[0]
    assert tnll[1] < rnll[1]
    selected = torch.stack([rnll[0], tnll[1]])
    regret = selected - torch.minimum(rnll, tnll)
    assert torch.allclose(regret, torch.zeros_like(regret))


def test_failed_gate_classifier_names_hard_negative_failure():
    base_family = {
        "HARD_ROUTE": {"direct_grid_exact_accuracy": 1.0},
        "reader_route_recall": 1.0,
        "transform_route_recall": 1.0,
        "mean_gate": 0.0,
    }
    sealed = {
        "macro_hard_route_exact": 1.0,
        "overall": {"route_utility_agreement": 1.0, "mean_routing_regret": 0.0},
        "families": {
            FAMILY_RETRIEVAL: base_family | {"reader_route_recall": 1.0, "mean_gate": 0.99},
            FAMILY_TRANSFORM: base_family | {"transform_route_recall": 1.0, "mean_gate": 0.01},
            FAMILY_HARD_NEGATIVE: base_family | {"transform_route_recall": 0.0, "mean_gate": 0.99},
        },
    }
    heldout = {"macro_hard_route_exact": 1.0, "macro_route_utility_agreement": 1.0, "hard_negative_transform_route_recall": 1.0}
    endpoint = {"passed": True}
    permutation = {"passed": True}
    ablation = {"families": {family: {"normal_to_disabled_hard_route_grid_change_rate": 1.0 if family == FAMILY_RETRIEVAL else 0.0} for family in CORE_FAMILIES}}
    transform_ablation = {"families": {family: {"normal_to_disabled_hard_route_grid_change_rate": 0.0 if family == FAMILY_RETRIEVAL else 1.0} for family in CORE_FAMILIES}}
    same_query = {"passed": True}
    failed = failed_gates_for_seed(sealed, heldout, endpoint, permutation, ablation, transform_ablation, same_query)
    assert "hard_negative_core" in failed
