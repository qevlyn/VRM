from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT

from .mixed_router_r3a_data import (
    ROUTE_HARD_NEGATIVE,
    ROUTE_RETRIEVAL,
    ROUTE_TRANSFORM,
    R3AEpisode,
    apply_transform,
    grid_to_list,
)


def transform_grid_tensor(query_grids: torch.Tensor, transform_specs: list[str]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for idx, spec in enumerate(transform_specs):
        grid = query_grids[idx]
        if spec == "HFLIP":
            rows.append(torch.flip(grid, dims=[1]))
        elif spec == "VFLIP":
            rows.append(torch.flip(grid, dims=[0]))
        elif spec == "ROT180":
            rows.append(torch.flip(grid, dims=[0, 1]))
        else:
            raise ValueError(f"unknown transform {spec}")
    return torch.stack(rows, dim=0)


def transform_oracle_probs(query_grids: torch.Tensor, transform_specs: list[str]) -> torch.Tensor:
    transformed = transform_grid_tensor(query_grids, transform_specs)
    return F.one_hot(transformed.long(), num_classes=ARC_COLOR_COUNT).to(torch.float32)


def certify_transform_oracle(episodes_by_split: dict[str, list[R3AEpisode]]) -> dict[str, Any]:
    family_counts = {ROUTE_RETRIEVAL: 0, ROUTE_TRANSFORM: 0, ROUTE_HARD_NEGATIVE: 0}
    exact_counts = {ROUTE_RETRIEVAL: 0, ROUTE_TRANSFORM: 0, ROUTE_HARD_NEGATIVE: 0}
    bad: list[dict[str, Any]] = []
    for split, episodes in episodes_by_split.items():
        for episode in episodes:
            family_counts[episode.route_family] += 1
            prediction = apply_transform(episode.query, episode.transform_spec)
            exact = prediction == episode.target
            exact_counts[episode.route_family] += int(exact)
            expected = episode.route_family in {ROUTE_TRANSFORM, ROUTE_HARD_NEGATIVE}
            if exact != expected and len(bad) < 20:
                bad.append(
                    {
                        "episode_id": episode.episode_id,
                        "split": split,
                        "route_family": episode.route_family,
                        "transform_spec": episode.transform_spec,
                        "prediction": grid_to_list(prediction),
                        "target": grid_to_list(episode.target),
                    }
                )
    exact_accuracy = {
        family: exact_counts[family] / max(family_counts[family], 1)
        for family in family_counts
    }
    passed = (
        exact_accuracy[ROUTE_TRANSFORM] == 1.0
        and exact_accuracy[ROUTE_HARD_NEGATIVE] == 1.0
        and exact_accuracy[ROUTE_RETRIEVAL] == 0.0
        and not bad
    )
    return {
        "family_counts": family_counts,
        "exact_counts": exact_counts,
        "exact_accuracy": exact_accuracy,
        "bad_examples": bad,
        "passed": passed,
    }

