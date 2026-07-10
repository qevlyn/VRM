from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.data import ArcPair
from vrm.model import VRMConfig
from vrm.twin_tasks.epoch_sampler import (
    coverage_row,
    epoch_permutation_indices,
    epoch_training_view,
    support_pair_order_report,
)
from vrm.twin_tasks.run_vrm_epoch_baseline import (
    derived_pair_proxy,
    foreground_metric,
    load_split_episodes,
    protected_hashes,
    sha256_file,
)
from vrm.twin_tasks.vrm_probe_data import (
    collate_twin_episodes,
    load_twin_episodes,
    model_inputs_for_twin_batch,
    pair_order_shuffled_view,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
    twin_swap_supports_view,
)
from vrm.twin_tasks.vrm_probe_train import initialize_fresh_model

DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
SEED = 20260628


def _episodes():
    return load_twin_episodes(DATA)


def _first_pair():
    episodes = _episodes()["train"]
    base_id = episodes[0].base_id
    twins = {episode.twin: episode for episode in episodes if episode.base_id == base_id}
    return twins["A"], twins["B"]


def _tiny_config() -> VRMConfig:
    return VRMConfig(hidden_size=16, num_heads=4, pair_encoder_layers=1, rewrite_layers=1, rewrite_steps=1, rule_slots=2)


def test_live_model_hash_stable_during_sampler_checks() -> None:
    path = Path("/mnt/d/VRM/vrm/model.py")
    before = sha256_file(path)
    _ = epoch_permutation_indices(100, base_seed=SEED, epoch_index=0)
    after = sha256_file(path)
    assert before == after


def test_protected_twin_probe_v0_files_remain_unchanged_during_loads() -> None:
    before = protected_hashes()
    _ = _episodes()["train"][:4]
    after = protected_hashes()
    assert before == after


def test_fresh_initialization_occurs_without_loading_old_checkpoint(monkeypatch) -> None:
    def forbidden_load(*args, **kwargs):
        raise AssertionError("torch.load must not be used for fresh initialization")

    monkeypatch.setattr(torch, "load", forbidden_load)
    model, report = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED)
    assert model is not None
    assert report["fresh_random_initialization"]
    assert not report["loaded_checkpoint"]


def test_each_epoch_contains_every_episode_once_with_zero_duplicates() -> None:
    episodes = _episodes()["train"][:100]
    ids = [episode.episode_id for episode in episodes]
    for epoch_index in range(3):
        order = epoch_permutation_indices(len(episodes), base_seed=SEED, epoch_index=epoch_index)
        row = coverage_row(epoch_index=epoch_index, order=order, observed=order, episode_ids=ids, base_seed=SEED)
        assert row["observed_episode_count"] == len(episodes)
        assert row["unique_episode_count"] == len(episodes)
        assert row["duplicate_episode_count"] == 0
        assert row["missing_episode_count"] == 0
        assert row["passed"]


def test_different_epochs_use_different_deterministic_permutations() -> None:
    first = epoch_permutation_indices(100, base_seed=SEED, epoch_index=0)
    second = epoch_permutation_indices(100, base_seed=SEED, epoch_index=1)
    again = epoch_permutation_indices(100, base_seed=SEED, epoch_index=0)
    assert first != second
    assert first == again


def test_resume_remaining_order_is_exact_suffix() -> None:
    order = epoch_permutation_indices(25, base_seed=SEED, epoch_index=0)
    next_offset = 12
    remaining_before = order[next_offset:]
    checkpoint_payload = {"current_epoch_permutation": order, "next_batch_offset": next_offset}
    remaining_after = checkpoint_payload["current_epoch_permutation"][checkpoint_payload["next_batch_offset"] :]
    assert remaining_after == remaining_before


def test_training_support_pair_shuffle_preserves_coherence() -> None:
    twin_a, _ = _first_pair()
    report = support_pair_order_report([twin_a], base_seed=SEED, epochs=3)
    assert report["all_pairs_remained_coherent"]
    assert report["orders_vary_across_epochs"]
    for epoch_index in range(3):
        view, _ = epoch_training_view(twin_a, base_seed=SEED, epoch_index=epoch_index)
        assert sorted(support_signature(view)) == sorted(support_signature(twin_a))


def test_candidates_labels_and_target_do_not_enter_forward_inputs() -> None:
    twin_a, _ = _first_pair()
    batch = collate_twin_episodes([twin_a])
    inputs = model_inputs_for_twin_batch(batch)
    forbidden = {"candidates", "candidate_0", "candidate_1", "label", "query_targets", "height_targets", "width_targets", "base_id", "twin"}
    assert forbidden.isdisjoint(inputs)


def test_sealed_split_read_is_guarded_before_best_selection() -> None:
    with pytest.raises(RuntimeError):
        load_split_episodes(DATA, "sealed", sealed_allowed=False)


def test_foreground_metric_ignores_background_cells() -> None:
    target = torch.tensor([[0, 1, 0], [2, 0, 3]])
    pred = torch.tensor([[9, 1, 8], [0, 0, 3]])
    row = foreground_metric(pred, target)
    assert row["foreground_color_total"] == 3
    assert row["foreground_color_correct"] == 2
    assert row["foreground_color_accuracy"] == 2 / 3


def test_derived_relevant_pair_proxy_cases() -> None:
    twin_a, _ = _first_pair()
    query_idx = next(idx for idx, pair in enumerate(twin_a.supports) if pair.input == twin_a.query_input)
    wrong_idx = next(idx for idx, pair in enumerate(twin_a.supports) if idx != query_idx)
    good = torch.tensor(twin_a.supports[query_idx].output.to_list())
    wrong = torch.tensor(twin_a.supports[wrong_idx].output.to_list())
    mixed = torch.zeros_like(good)
    assert derived_pair_proxy(good, twin_a)["derived_relevant_pair_output"]
    wrong_row = derived_pair_proxy(wrong, twin_a)
    assert wrong_row["matches_any_support_output"]
    assert not wrong_row["derived_relevant_pair_output"]
    mixed_row = derived_pair_proxy(mixed, twin_a)
    assert not mixed_row["matches_any_support_output"]
    duplicate_supports = list(twin_a.supports)
    duplicate_supports[wrong_idx] = ArcPair(
        input=duplicate_supports[wrong_idx].input,
        output=duplicate_supports[query_idx].output,
        pair_index=duplicate_supports[wrong_idx].pair_index,
    )
    tied = replace(twin_a, supports=tuple(duplicate_supports))
    tie_row = derived_pair_proxy(good, tied)
    assert tie_row["support_output_match_tie"]
    assert not tie_row["derived_relevant_pair_output"]


def test_twin_swap_preserves_query_and_unpaired_bags() -> None:
    twin_a, twin_b = _first_pair()
    swapped = twin_swap_supports_view(twin_a, twin_b)
    assert swapped.query_input == twin_a.query_input
    assert swapped.candidates == twin_a.candidates
    assert support_input_bag_signature(swapped) == support_input_bag_signature(twin_a)
    assert support_output_bag_signature(swapped) == support_output_bag_signature(twin_a)
    assert support_signature(swapped) != support_signature(twin_a)


def test_pair_order_shuffle_preserves_coherent_pairs() -> None:
    twin_a, _ = _first_pair()
    shuffled = pair_order_shuffled_view(twin_a, seed=SEED)
    assert sorted(support_signature(shuffled)) == sorted(support_signature(twin_a))
