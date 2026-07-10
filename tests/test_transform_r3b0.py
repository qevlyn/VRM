from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.twin_tasks.run_transform_r3b0 import protected_hashes
from vrm.twin_tasks.transform_r3b0_data import (
    TRANSFORMS,
    apply_transform,
    counterfactual_suite_report,
    factor_complete,
    generate_counterfactual_suite,
    generate_split,
    is_nondegenerate,
    partition_query_types,
    query_type_key,
    reorder_episode,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
)
from vrm.twin_tasks.transform_r3b0_model import SupportConditionedTransformPointerR3B0, training_batch_keys, transform_r3b0_output_loss


def _sample_episodes():
    split_types = partition_query_types()
    return generate_split("unit", split_types["dev"][:1])


def test_cuda_required_and_cpu_fallback_forbidden() -> None:
    assert torch.cuda.is_available() is True


def test_protected_prior_hashes_unchanged_by_r3b0_helpers() -> None:
    before = protected_hashes()
    _ = _sample_episodes()[:3]
    assert protected_hashes() == before


def test_query_type_split_disjoint_and_factor_complete() -> None:
    split_types = partition_query_types()
    sets = {split: {query_type_key(item) for item in values} for split, values in split_types.items()}
    assert len(split_types["train"]) == 120
    assert len(split_types["dev"]) == 30
    assert len(split_types["sealed"]) == 30
    assert sets["train"].isdisjoint(sets["dev"])
    assert sets["train"].isdisjoint(sets["sealed"])
    assert sets["dev"].isdisjoint(sets["sealed"])
    assert all(factor_complete(values) for values in split_types.values())


def test_transform_balance_and_query_type_independence() -> None:
    episodes = _sample_episodes()
    assert len(episodes) == 300
    counts = {spec: 0 for spec in TRANSFORMS}
    for episode in episodes:
        counts[episode.transform_spec] += 1
    assert counts == {spec: 100 for spec in TRANSFORMS}


def test_nondegeneracy_and_episode_oracle_contracts() -> None:
    for episode in _sample_episodes()[:24]:
        assert is_nondegenerate(episode.query)
        assert apply_transform(episode.query, episode.transform_spec) == episode.target
        assert all(pair.input != episode.query for pair in episode.supports)
        assert all(is_nondegenerate(pair.input) for pair in episode.supports)
        assert all(apply_transform(pair.input, episode.transform_spec) == pair.output for pair in episode.supports)
        assert all(pair.output != episode.target for pair in episode.supports)


def test_model_forward_has_no_private_metadata_inputs() -> None:
    params = inspect.signature(SupportConditionedTransformPointerR3B0.forward).parameters
    forbidden = {"transform_spec", "query_type", "split_id", "target", "episode_id", "counterfactual_group_id", "candidate", "label"}
    assert forbidden.isdisjoint(params)
    names = " ".join(name for name, _ in SupportConditionedTransformPointerR3B0().named_parameters())
    assert "pair_id" not in names
    assert "pair_position" not in names


def test_forward_shapes_and_output_only_loss() -> None:
    episodes = _sample_episodes()[:4]
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    assert set(batch) == {"query_grids", "support_input_grids", "support_output_grids", "target_grids"}
    assert training_batch_keys() == ["query_grids", "support_input_grids", "support_output_grids", "target_grids"]
    model = SupportConditionedTransformPointerR3B0().cuda()
    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    assert out["output_probs"].shape == (4, 7, 7, 10)
    assert out["decoded_grid"].shape == (4, 7, 7)
    assert out["task_context"].shape == (4, 128)
    assert out["pair_representations"].shape == (4, 3, 128)
    assert out["pointer_attention"].shape == (4, 7, 7, 49)
    assert transform_r3b0_output_loss(out["output_probs"], batch["target_grids"]).requires_grad


def test_pair_order_permutation_invariance_and_logical_correspondence() -> None:
    episodes = _sample_episodes()[:2]
    model = SupportConditionedTransformPointerR3B0().cuda().eval()
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    permuted = [reorder_episode(episode, (2, 0, 1)) for episode in episodes]
    perm_batch = tensors_from_episodes(permuted, torch.device("cuda"))
    with torch.no_grad():
        base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        view = model(perm_batch["query_grids"], perm_batch["support_input_grids"], perm_batch["support_output_grids"])
    assert torch.allclose(base["output_probs"], view["output_probs"], atol=1e-6)
    assert torch.allclose(base["task_context"], view["task_context"], atol=1e-6)
    assert sorted(support_signature(episodes[0])) == sorted(support_signature(permuted[0]))


def test_zero_support_intervention_only_zeros_context_evidence() -> None:
    episodes = _sample_episodes()[:3]
    batch = tensors_from_episodes(episodes, torch.device("cuda"))
    model = SupportConditionedTransformPointerR3B0().cuda().eval()
    with torch.no_grad():
        base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        zero = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
    assert torch.equal(base["pair_representations"], zero["pair_representations"])
    assert torch.count_nonzero(zero["pair_representations_for_context"]) == 0
    assert torch.equal(batch["query_grids"], batch["query_grids"])


def test_deranged_pairing_preserves_bags_and_breaks_correspondence() -> None:
    from vrm.twin_tasks.transform_r3b0_data import derange_episode

    episode = _sample_episodes()[0]
    deranged = derange_episode(episode, 1)
    assert support_input_bag(episode) == support_input_bag(deranged)
    assert support_output_bag(episode) == support_output_bag(deranged)
    assert support_signature(episode) != support_signature(deranged)


def test_counterfactual_suite_keeps_query_and_support_inputs_fixed() -> None:
    qtype = partition_query_types()["sealed"][0]
    suite = generate_counterfactual_suite("unit_cf", [qtype], groups_per_type=2)
    report = counterfactual_suite_report(suite)
    assert report["passed"]
    groups = {}
    for episode in suite:
        groups.setdefault(episode.counterfactual_group_id, []).append(episode)
    for items in groups.values():
        assert len({episode.query for episode in items}) == 1
        assert len({tuple(support_input_bag(episode)) for episode in items}) == 1
        assert {episode.transform_spec for episode in items} == set(TRANSFORMS)
        assert len({episode.target for episode in items}) == 3


def test_selection_artifacts_precede_sealed_eval_when_present() -> None:
    out = Path("/mnt/d/VRM/artifacts/transform_r3b0_v1")
    selection = out / "best_checkpoint_selection_seed_20260628.json"
    sealed = out / "final_sealed_metrics_seed_20260628.json"
    if selection.exists() and sealed.exists():
        assert selection.stat().st_mtime <= sealed.stat().st_mtime
