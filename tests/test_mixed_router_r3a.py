from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.twin_tasks.mixed_router_r3a_data import (
    ROUTE_HARD_NEGATIVE,
    ROUTE_RETRIEVAL,
    ROUTE_TRANSFORM,
    ROUTE_FAMILIES,
    COLORS,
    ANCHORS,
    DISTRACTOR_COUNTS,
    SHAPES,
    make_counterfactual_group,
    partition_query_types,
    query_type_key,
    reorder_episode,
    route_balance_report,
    support_signature,
    tensors_from_episodes,
)
from vrm.twin_tasks.mixed_router_r3a_oracles import certify_transform_oracle, transform_oracle_probs
from vrm.twin_tasks.mixed_router_r3a_reader import PairKeyValueReaderR3A, reader_output_loss, reader_training_batch_keys
from vrm.twin_tasks.mixed_router_r3a_router import MixedRouteRouterR3A, grad_norm, router_output_loss
from vrm.twin_tasks.run_mixed_router_r3a import protected_hashes


def _episodes():
    qtype = ("marker_0", 1, "upper_left", 1)
    return make_counterfactual_group("unit", qtype, 12345)


def test_cuda_required_and_cpu_fallback_forbidden() -> None:
    assert torch.cuda.is_available() is True


def test_protected_prior_hashes_unchanged_by_imports() -> None:
    before = protected_hashes()
    _ = _episodes()
    assert protected_hashes() == before


def test_query_type_splits_are_disjoint_and_factor_complete() -> None:
    split_types = partition_query_types()
    sets = {split: {query_type_key(item) for item in values} for split, values in split_types.items()}
    assert sets["train"].isdisjoint(sets["dev"])
    assert sets["train"].isdisjoint(sets["sealed"])
    assert sets["dev"].isdisjoint(sets["sealed"])
    for values in split_types.values():
        assert {item[0] for item in values} == set(SHAPES)
        assert {item[1] for item in values} == set(COLORS)
        assert {item[2] for item in values} == set(ANCHORS)
        assert {item[3] for item in values} == set(DISTRACTOR_COUNTS)


def test_counterfactual_group_has_all_families_and_same_query() -> None:
    episodes = _episodes()
    assert {episode.route_family for episode in episodes} == set(ROUTE_FAMILIES)
    assert len({episode.query for episode in episodes}) == 1
    report = route_balance_report({"unit": episodes})
    assert report["unit"]["query_types_all_have_three_families"]
    assert report["unit"]["query_types_balanced_by_family"]


def test_retrieval_transform_and_hard_negative_input_match_rules() -> None:
    episodes = {episode.route_family: episode for episode in _episodes()}
    retrieval = episodes[ROUTE_RETRIEVAL]
    transform = episodes[ROUTE_TRANSFORM]
    hard = episodes[ROUTE_HARD_NEGATIVE]
    assert sum(pair.input == retrieval.query for pair in retrieval.supports) == 1
    assert sum(pair.input == transform.query for pair in transform.supports) == 0
    assert sum(pair.input == hard.query for pair in hard.supports) == 0
    shape, color, anchor, _ = hard.query_type
    hard_inputs = [pair.input for pair in hard.supports]
    assert len(hard_inputs) == 3
    assert hard.supports[0].input != hard.query
    assert hard.supports[1].input != hard.query
    assert hard.supports[2].input != hard.query
    assert shape and color and anchor


def test_targets_and_transform_oracle_contracts() -> None:
    episodes = _episodes()
    by_family = {episode.route_family: episode for episode in episodes}
    retrieval = by_family[ROUTE_RETRIEVAL]
    assert sum(pair.output == retrieval.target for pair in retrieval.supports) == 1
    for family in (ROUTE_TRANSFORM, ROUTE_HARD_NEGATIVE):
        episode = by_family[family]
        assert all(pair.output != episode.target for pair in episode.supports)
    report = certify_transform_oracle({"unit": episodes})
    assert report["passed"]
    assert report["exact_accuracy"][ROUTE_RETRIEVAL] == 0.0
    assert report["exact_accuracy"][ROUTE_TRANSFORM] == 1.0
    assert report["exact_accuracy"][ROUTE_HARD_NEGATIVE] == 1.0


def test_reader_and_router_forward_boundaries() -> None:
    assert list(inspect.signature(PairKeyValueReaderR3A.forward).parameters) == [
        "self",
        "query_grids",
        "support_input_grids",
        "support_output_grids",
    ]
    router_params = inspect.signature(MixedRouteRouterR3A.forward).parameters
    forbidden = {"target", "route_family", "transform_spec", "candidate", "query_type", "counterfactual_group_id", "episode_id"}
    assert forbidden.isdisjoint(router_params)
    assert reader_training_batch_keys() == ["query_grids", "support_input_grids", "support_output_grids", "target_grids"]


def test_losses_are_output_only_and_batch_has_no_route_labels() -> None:
    batch = tensors_from_episodes(_episodes(), torch.device("cuda"))
    assert set(batch) == {"query_grids", "support_input_grids", "support_output_grids", "target_grids"}
    reader = PairKeyValueReaderR3A().cuda()
    reader_out = reader(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    assert reader_output_loss(reader_out["reader_probs"], batch["target_grids"]).requires_grad
    router = MixedRouteRouterR3A(reader).cuda()
    probs = transform_oracle_probs(batch["query_grids"], [episode.transform_spec for episode in _episodes()])
    router_out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs)
    assert router_output_loss(router_out["final_probs"], batch["target_grids"]).requires_grad


def test_reader_frozen_and_only_gate_gets_router_gradient() -> None:
    episodes = _episodes()
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    reader = PairKeyValueReaderR3A().cuda()
    router = MixedRouteRouterR3A(reader).cuda()
    probs = transform_oracle_probs(batch["query_grids"], [episode.transform_spec for episode in episodes])
    out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs)
    loss = router_output_loss(out["final_probs"], batch["target_grids"])
    loss.backward()
    assert grad_norm(router.reader.parameters()) == 0.0
    assert grad_norm(router.gate_head.parameters()) > 0.0


def test_initial_gate_is_neutral() -> None:
    episodes = _episodes()
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    router = MixedRouteRouterR3A(PairKeyValueReaderR3A().cuda()).cuda()
    probs = transform_oracle_probs(batch["query_grids"], [episode.transform_spec for episode in episodes])
    with torch.no_grad():
        out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs)
    assert float(out["gate"].mean().item()) == pytest.approx(0.5, abs=1e-6)


def test_pair_permutations_preserve_correspondence_and_logical_restoration() -> None:
    episode = _episodes()[0]
    original = sorted(support_signature(episode))
    for order in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1)):
        view = reorder_episode(episode, order)
        assert sorted(support_signature(view)) == original


def test_reader_and_transform_disabled_change_only_their_probs() -> None:
    episodes = _episodes()
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    router = MixedRouteRouterR3A(PairKeyValueReaderR3A().cuda()).cuda()
    probs = transform_oracle_probs(batch["query_grids"], [episode.transform_spec for episode in episodes])
    with torch.no_grad():
        base = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs)
        reader_disabled = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs, reader_disabled=True)
        transform_disabled = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], probs, transform_disabled=True)
    assert torch.equal(base["gate"], reader_disabled["gate"])
    assert torch.equal(base["reader_alpha"], reader_disabled["reader_alpha"])
    assert torch.equal(base["transform_oracle_probs"], reader_disabled["transform_oracle_probs"])
    assert not torch.equal(base["reader_probs_for_mix"], reader_disabled["reader_probs_for_mix"])
    assert torch.equal(base["gate"], transform_disabled["gate"])
    assert torch.equal(base["reader_alpha"], transform_disabled["reader_alpha"])
    assert torch.equal(base["reader_probs"], transform_disabled["reader_probs"])
    assert not torch.equal(base["transform_probs_for_mix"], transform_disabled["transform_probs_for_mix"])


def test_selection_artifacts_precede_sealed_eval_when_present() -> None:
    out = Path("/mnt/d/VRM/artifacts/mixed_router_r3a_v1")
    selection = out / "router_best_checkpoint_selection_seed_20260628.json"
    sealed = out / "router_final_sealed_metrics_seed_20260628.json"
    if selection.exists() and sealed.exists():
        assert selection.stat().st_mtime <= sealed.stat().st_mtime

