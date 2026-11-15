from __future__ import annotations

import hashlib
import random
from typing import Any

import torch

from vrm.data import (
    PAD_ID,
    ROLE_SUPPORT_OUTPUT,
    ArcEpisode,
    ArcPair,
    ArcTask,
    collate_episodes,
    grid_to_text,
    model_inputs_from_batch,
)


def tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        f"{tuple(array.shape)}|{array.dtype}".encode("utf-8") + b"\0" + array.numpy().tobytes()
    ).hexdigest()


def model_input_digest(inputs: dict[str, torch.Tensor]) -> dict[str, str]:
    return {key: tensor_digest(value) for key, value in sorted(inputs.items())}


def collate_model_inputs(episodes: list[ArcEpisode] | tuple[ArcEpisode, ...]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    batch = collate_episodes(episodes)
    return batch, model_inputs_from_batch(batch)


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def zero_support_output_colors(batch: dict[str, Any]) -> dict[str, Any]:
    ablated = clone_batch(batch)
    output_token_mask = (
        ablated["support_evidence_mask"].to(torch.bool)
        & (ablated["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
    )
    ablated["support_evidence_colors"][output_token_mask] = 0
    if "support_outputs" in ablated:
        nonpad = ablated["support_outputs"] != PAD_ID
        ablated["support_outputs"][nonpad] = 0
    return ablated


def query_only_masked_batch(batch: dict[str, Any]) -> dict[str, Any]:
    masked = clone_batch(batch)
    masked["support_evidence_mask"] = torch.zeros_like(masked["support_evidence_mask"], dtype=torch.bool)
    if "support_mask" in masked:
        masked["support_mask"] = torch.zeros_like(masked["support_mask"], dtype=torch.bool)
    return masked


def deterministic_permutation(length: int, *, task_id: str, query_pair_index: int, salt: str) -> list[int]:
    values = list(range(length))
    if length < 2:
        return values
    digest = hashlib.sha256(f"{salt}:{task_id}:{query_pair_index}:{length}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rng.shuffle(values)
    if values == list(range(length)):
        values = values[1:] + values[:1]
    return values


def permute_support_pair_order(episode: ArcEpisode) -> tuple[ArcEpisode, dict[str, Any]]:
    supports = list(episode.supports)
    permutation = deterministic_permutation(
        len(supports),
        task_id=episode.task_id,
        query_pair_index=episode.query_pair_index,
        salt="R5A4A_SUPPORT_PAIR_ORDER",
    )
    new_supports = tuple(
        ArcPair(input=supports[old_index].input, output=supports[old_index].output, pair_index=new_slot)
        for new_slot, old_index in enumerate(permutation)
    )
    return (
        ArcEpisode(
            task_id=episode.task_id,
            source_split=episode.source_split,
            supports=new_supports,
            query=episode.query,
            query_pair_index=episode.query_pair_index,
            source_file=episode.source_file,
        ),
        {
            "support_pair_permutation_old_index_by_new_slot": permutation,
            "non_identity": permutation != list(range(len(supports))),
        },
    )


def make_training_episode(task: ArcTask, query_pair_index: int, *, source_split: str = "training") -> ArcEpisode:
    query = task.train[query_pair_index]
    supports = tuple(pair for idx, pair in enumerate(task.train) if idx != query_pair_index)
    if not supports:
        raise ValueError("R5A-4A training episode requires at least one visible support pair")
    return ArcEpisode(
        task_id=task.task_id,
        source_split=source_split,
        supports=supports,
        query=query,
        query_pair_index=query.pair_index,
        source_file=task.source_file,
    )


def render_episode_grids(episode: ArcEpisode) -> str:
    lines: list[str] = []
    for support in episode.supports:
        lines.extend(
            [
                f"support pair {support.pair_index} input ({support.input.height}x{support.input.width})",
                grid_to_text(support.input),
                f"support pair {support.pair_index} output ({support.output.height}x{support.output.width})",
                grid_to_text(support.output),
            ]
        )
    lines.extend(
        [
            f"held-out query input ({episode.query.input.height}x{episode.query.input.width})",
            grid_to_text(episode.query.input),
            f"hidden target output ({episode.query.output.height}x{episode.query.output.width})",
            grid_to_text(episode.query.output),
        ]
    )
    return "\n".join(lines)


def changed_metadata_episode(episode: ArcEpisode, *, suffix: str = "_changed_metadata") -> ArcEpisode:
    return ArcEpisode(
        task_id=episode.task_id + suffix,
        source_split=episode.source_split + suffix,
        supports=episode.supports,
        query=episode.query,
        query_pair_index=episode.query_pair_index,
        source_file=episode.source_file + suffix,
    )
