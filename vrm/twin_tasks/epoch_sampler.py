from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import replace
from typing import Any

from vrm.data import ArcPair

from .vrm_probe_data import TwinEpisode, stable_int, support_signature


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def epoch_seed(base_seed: int, epoch_index: int) -> int:
    return int(base_seed) + int(epoch_index)


def epoch_permutation_indices(episode_count: int, *, base_seed: int, epoch_index: int) -> list[int]:
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    order = list(range(episode_count))
    rng = random.Random(epoch_seed(base_seed, epoch_index))
    rng.shuffle(order)
    return order


def epoch_batches(order: list[int], batch_size: int, start_offset: int = 0) -> list[list[int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if start_offset < 0 or start_offset > len(order):
        raise ValueError("start_offset outside permutation")
    return [order[start : start + batch_size] for start in range(start_offset, len(order), batch_size)]


def coverage_row(
    *,
    epoch_index: int,
    order: list[int],
    observed: list[int],
    episode_ids: list[str],
    base_seed: int,
) -> dict[str, Any]:
    expected = len(order)
    observed_count = len(observed)
    observed_set = set(observed)
    duplicates = observed_count - len(observed_set)
    missing = sorted(set(order) - observed_set)
    extra = sorted(observed_set - set(order))
    ordered_episode_ids = [episode_ids[idx] for idx in order]
    return {
        "epoch_index": epoch_index,
        "epoch_seed": epoch_seed(base_seed, epoch_index),
        "expected_episode_count": expected,
        "observed_episode_count": observed_count,
        "unique_episode_count": len(observed_set),
        "duplicate_episode_count": duplicates,
        "missing_episode_count": len(missing),
        "extra_episode_count": len(extra),
        "missing_episode_indices": missing[:50],
        "extra_episode_indices": extra[:50],
        "permutation_sha256": sha256_json(order),
        "ordered_episode_ids_sha256": sha256_json(ordered_episode_ids),
        "ordered_episode_indices": order,
        "ordered_episode_ids": ordered_episode_ids,
        "passed": expected == observed_count and expected == len(observed_set) and duplicates == 0 and not missing and not extra,
    }


def support_order_indices(episode: TwinEpisode, *, base_seed: int, epoch_index: int) -> list[int]:
    permutations = list(itertools.permutations(range(len(episode.supports))))
    if not permutations:
        raise ValueError("episode has no supports")
    start = stable_int(f"{base_seed}|{episode.episode_id}|support_pair_order")
    return list(permutations[(start + epoch_index) % len(permutations)])


def epoch_training_view(
    episode: TwinEpisode,
    *,
    base_seed: int,
    epoch_index: int,
) -> tuple[TwinEpisode, list[int]]:
    order = support_order_indices(episode, base_seed=base_seed, epoch_index=epoch_index)
    reordered = tuple(
        ArcPair(
            input=episode.supports[source_idx].input,
            output=episode.supports[source_idx].output,
            pair_index=new_idx,
        )
        for new_idx, source_idx in enumerate(order)
    )
    return replace(episode, supports=reordered), order


def support_pair_order_report(episodes: list[TwinEpisode], *, base_seed: int, epochs: int) -> dict[str, Any]:
    incoherent: list[dict[str, Any]] = []
    varied_count = 0
    samples: list[dict[str, Any]] = []
    coherent_presentations = 0
    for idx, episode in enumerate(episodes):
        original_sorted = sorted(support_signature(episode))
        orders: list[list[int]] = []
        signatures: list[Any] = []
        for epoch_index in range(epochs):
            view, order = epoch_training_view(episode, base_seed=base_seed, epoch_index=epoch_index)
            orders.append(order)
            signatures.append(support_signature(view))
            coherent_presentations += 1
            if sorted(support_signature(view)) != original_sorted:
                incoherent.append(
                    {
                        "episode_index": idx,
                        "episode_id": episode.episode_id,
                        "epoch_index": epoch_index,
                        "order": order,
                    }
                )
        if len({tuple(order) for order in orders}) > 1:
            varied_count += 1
        if len(samples) < 20:
            samples.append(
                {
                    "episode_index": idx,
                    "episode_id": episode.episode_id,
                    "orders_by_epoch": orders,
                    "signature_hashes_by_epoch": [sha256_json(sig) for sig in signatures],
                }
            )
    return {
        "episode_count": len(episodes),
        "epochs": epochs,
        "coherent_presentations_checked": coherent_presentations,
        "incoherent_presentation_count": len(incoherent),
        "incoherent_examples": incoherent[:20],
        "episodes_with_order_variation": varied_count,
        "all_pairs_remained_coherent": not incoherent,
        "orders_vary_across_epochs": varied_count == len(episodes),
        "sample_orders": samples,
        "passed": not incoherent and varied_count == len(episodes),
    }
