from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT

from .transform_r3b0_data import R3B0Episode, apply_transform


def transform_oracle_grid(query_grid: tuple[tuple[int, ...], ...], transform_spec: str) -> tuple[tuple[int, ...], ...]:
    return apply_transform(query_grid, transform_spec)


def transform_oracle_probs(query_grids: torch.Tensor, transform_specs: list[str]) -> torch.Tensor:
    if query_grids.shape[0] != len(transform_specs):
        raise ValueError("query batch and transform spec count differ")
    outputs = []
    for idx, spec in enumerate(transform_specs):
        grid = tuple(tuple(int(value) for value in row) for row in query_grids[idx].detach().cpu().tolist())
        outputs.append(apply_transform(grid, spec))
    decoded = torch.tensor(outputs, dtype=torch.long, device=query_grids.device)
    return F.one_hot(decoded, num_classes=ARC_COLOR_COUNT).to(torch.float32)


def certify_dataset_transforms(episodes_by_split: dict[str, list[R3B0Episode]]) -> dict[str, Any]:
    counts = {"total": 0, "target_exact": 0, "support_exact": 0, "target_equals_support_output": 0}
    bad_examples = []
    for split, episodes in episodes_by_split.items():
        for episode in episodes:
            counts["total"] += 1
            target_exact = apply_transform(episode.query, episode.transform_spec) == episode.target
            support_exact = all(apply_transform(pair.input, episode.transform_spec) == pair.output for pair in episode.supports)
            target_equals_support = any(pair.output == episode.target for pair in episode.supports)
            counts["target_exact"] += int(target_exact)
            counts["support_exact"] += int(support_exact)
            counts["target_equals_support_output"] += int(target_equals_support)
            if (not target_exact or not support_exact or target_equals_support) and len(bad_examples) < 20:
                bad_examples.append({"split": split, "episode_id": episode.episode_id})
    return {
        "counts": counts,
        "target_exact_accuracy": counts["target_exact"] / max(counts["total"], 1),
        "support_pair_exact_accuracy": counts["support_exact"] / max(counts["total"], 1),
        "target_equals_support_output_count": counts["target_equals_support_output"],
        "bad_examples": bad_examples,
        "passed": counts["target_exact"] == counts["total"] and counts["support_exact"] == counts["total"] and counts["target_equals_support_output"] == 0,
    }
