from __future__ import annotations

from typing import Any

from . import compiled_topology_r5a2c_r2_data as data


def execute_hypothesis(
    grid: tuple[tuple[int, ...], ...],
    hypothesis: data.TopologyHypothesis,
    *,
    ablation: str | None = None,
) -> tuple[tuple[int, ...], ...]:
    if ablation == "no_propagation":
        return data.apply_hypothesis_reference(grid, hypothesis, no_propagation=True)
    if ablation == "one_step_only":
        return data.apply_hypothesis_reference(grid, hypothesis, one_step_only=True)
    if ablation == "wrong_adjacency_8_connected":
        return data.apply_hypothesis_reference(grid, hypothesis, adjacency="8_connected")
    if ablation == "barrier_ignored":
        return data.apply_hypothesis_reference(grid, hypothesis, barrier_ignored=True)
    if ablation == "border_seed_disabled":
        return data.apply_hypothesis_reference(grid, hypothesis, border_seed_disabled=True)
    return data.apply_hypothesis_reference(grid, hypothesis)


def consistent_hypotheses(
    support_input_grids: list[tuple[tuple[int, ...], ...]],
    support_output_grids: list[tuple[tuple[int, ...], ...]],
    *,
    adjacency: str = "4_connected",
    ablation: str | None = None,
) -> list[data.TopologyHypothesis]:
    matches = []
    for hypothesis in data.candidate_hypotheses(adjacency=adjacency):
        if all(execute_hypothesis(before, hypothesis, ablation=ablation) == after for before, after in zip(support_input_grids, support_output_grids)):
            matches.append(hypothesis)
    return matches


def execute_from_raw_grids(
    support_input_grids: list[tuple[tuple[int, ...], ...]],
    support_output_grids: list[tuple[tuple[int, ...], ...]],
    query_input_grid: tuple[tuple[int, ...], ...],
    *,
    ablation: str | None = None,
) -> dict[str, Any]:
    if len(support_input_grids) != data.SUPPORT_PAIR_COUNT or len(support_output_grids) != data.SUPPORT_PAIR_COUNT:
        return {"status": "invalid_support_count", "consistent_hypotheses": [], "output_grid": None}
    if ablation == "support_consistency_disabled":
        selected = data.candidate_hypotheses()[0]
        return {
            "status": "support_consistency_disabled",
            "consistent_hypotheses": [selected.to_json()],
            "selected_hypothesis": selected.to_json(),
            "output_grid": execute_hypothesis(query_input_grid, selected),
        }
    matches = consistent_hypotheses(support_input_grids, support_output_grids, ablation=ablation)
    if len(matches) != 1:
        return {
            "status": "ambiguous" if len(matches) > 1 else "no_consistent_hypothesis",
            "consistent_hypotheses": [item.to_json() for item in matches],
            "selected_hypothesis": None,
            "output_grid": None,
        }
    selected = matches[0]
    return {
        "status": "ok",
        "consistent_hypotheses": [selected.to_json()],
        "selected_hypothesis": selected.to_json(),
        "output_grid": execute_hypothesis(query_input_grid, selected, ablation=ablation),
    }


def executor_runtime_contract() -> dict[str, Any]:
    return {
        "runtime_inputs": ["support_input_grids", "support_output_grids", "query_input_grid"],
        "forbidden_runtime_inputs": [
            "task_family_id",
            "class_id",
            "target_grid",
            "target_mask",
            "oracle_label",
            "split_id",
            "episode_id",
            "seed_id",
        ],
        "parameter_free": True,
        "trainable_parameters": 0,
        "uses_complete_tuple_support_consistency": True,
    }
