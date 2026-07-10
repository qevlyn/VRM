from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch

from vrm.data import MAX_GRID_HEIGHT, MAX_GRID_WIDTH, move_tensor_batch
from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM

from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_PAIR_ORDER_SHUFFLED,
    CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    collate_twin_episodes,
    condition_view,
    group_by_base,
    model_inputs_for_twin_batch,
)


def _grid_tensor(grid: Any) -> torch.Tensor:
    return torch.tensor(grid.to_list(), dtype=torch.long)


def _target_rect_from_prediction(prediction: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return prediction.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)[:height, :width].detach().cpu().to(torch.long)


def candidate_decision(
    pred_rect: torch.Tensor,
    candidate_0: torch.Tensor,
    candidate_1: torch.Tensor,
    correct_label: int,
) -> dict[str, Any]:
    if pred_rect.shape != candidate_0.shape or pred_rect.shape != candidate_1.shape:
        raise ValueError("candidate distance scoring requires same target rectangle shape")
    distance_0 = int((pred_rect != candidate_0).sum().item())
    distance_1 = int((pred_rect != candidate_1).sum().item())
    if distance_0 < distance_1:
        decision: int | None = 0
        decision_name = "0"
    elif distance_1 < distance_0:
        decision = 1
        decision_name = "1"
    else:
        decision = None
        decision_name = "TIE"
    if decision is None:
        score = 0.5
    else:
        score = 1.0 if decision == correct_label else 0.0
    return {
        "distance_to_candidate_0": distance_0,
        "distance_to_candidate_1": distance_1,
        "candidate_decision": decision,
        "candidate_decision_name": decision_name,
        "candidate_nearest_score": score,
        "candidate_tie": decision is None,
    }


def evaluate_condition(
    *,
    model: VRM,
    episodes: list[TwinEpisode],
    split: str,
    condition: str,
    device: torch.device,
    seed: int,
    batch_size: int,
    shape_loss_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    opposite_lookup = build_opposite_lookup(episodes)
    per_episode: list[dict[str, Any]] = []
    totals = {
        "total_loss": 0.0,
        "color_loss": 0.0,
        "height_loss": 0.0,
        "width_loss": 0.0,
    }
    color_correct = 0
    color_total = 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            view_batch = [
                condition_view(
                    episode,
                    condition=condition,
                    opposite_by_episode=opposite_lookup,
                    seed=seed,
                )
                for episode in source_batch
            ]
            batch = move_tensor_batch(collate_twin_episodes(view_batch), device)
            outputs = model(
                **model_inputs_for_twin_batch(batch),
                disable_support_evidence=condition == CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED,
            )
            loss, components = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
            prediction = outputs["color_log_probs"].argmax(dim=-1).detach().cpu().to(torch.long)
            predicted_heights = (outputs["height_logits"].argmax(dim=-1) + 1).detach().cpu().to(torch.long)
            predicted_widths = (outputs["width_logits"].argmax(dim=-1) + 1).detach().cpu().to(torch.long)
            for key in totals:
                totals[key] += float(components[key].detach().cpu().item()) * len(source_batch)
            for local_idx, episode in enumerate(source_batch):
                target_h = episode.target_output.height
                target_w = episode.target_output.width
                pred_rect = _target_rect_from_prediction(prediction[local_idx], target_h, target_w)
                target_rect = _grid_tensor(episode.target_output)
                candidate_0 = _grid_tensor(episode.candidates[0])
                candidate_1 = _grid_tensor(episode.candidates[1])
                cand = candidate_decision(pred_rect, candidate_0, candidate_1, episode.label)
                correct = int((pred_rect == target_rect).sum().item())
                total = int(target_rect.numel())
                pred_h = int(predicted_heights[local_idx].item())
                pred_w = int(predicted_widths[local_idx].item())
                shape_match = pred_h == target_h and pred_w == target_w
                hamming = total - correct
                direct_exact = bool(shape_match and hamming == 0)
                candidate_exact = bool(
                    shape_match
                    and (
                        torch.equal(pred_rect, candidate_0)
                        or torch.equal(pred_rect, candidate_1)
                    )
                )
                alternate_score = None
                if condition == CONDITION_TWIN_SWAP_SUPPORTS:
                    if cand["candidate_decision"] is None:
                        alternate_score = 0.5
                    elif cand["candidate_decision"] == episode.alternate_label:
                        alternate_score = 1.0
                    elif cand["candidate_decision"] == episode.label:
                        alternate_score = 0.0
                    else:
                        alternate_score = 0.0
                row = {
                    "split": split,
                    "condition": condition,
                    "episode_index": start + local_idx,
                    "episode_id": episode.episode_id,
                    "base_id": episode.base_id,
                    "twin": episode.twin,
                    "label": episode.label,
                    "alternate_label": episode.alternate_label,
                    "target_shape": [target_h, target_w],
                    "predicted_shape": [pred_h, pred_w],
                    "distance_to_candidate_0": cand["distance_to_candidate_0"],
                    "distance_to_candidate_1": cand["distance_to_candidate_1"],
                    "candidate_decision": cand["candidate_decision"],
                    "candidate_decision_name": cand["candidate_decision_name"],
                    "candidate_nearest_score": cand["candidate_nearest_score"],
                    "candidate_tie": cand["candidate_tie"],
                    "swapped_support_alternate_choice_score": alternate_score,
                    "candidate_exact_match": candidate_exact,
                    "direct_grid_exact": direct_exact,
                    "target_rectangle_hamming_error": hamming,
                    "target_rectangle_color_correct": correct,
                    "target_rectangle_color_total": total,
                    "target_rectangle_color_accuracy": correct / max(total, 1),
                    "shape_match": shape_match,
                    "decoded_prediction": pred_rect.tolist(),
                    "target_grid": target_rect.tolist(),
                    "candidate_0": candidate_0.tolist(),
                    "candidate_1": candidate_1.tolist(),
                    "total_loss": float(components["total_loss"].detach().cpu().item()),
                    "color_loss": float(components["color_loss"].detach().cpu().item()),
                    "height_loss": float(components["height_loss"].detach().cpu().item()),
                    "width_loss": float(components["width_loss"].detach().cpu().item()),
                }
                per_episode.append(row)
                color_correct += correct
                color_total += total
    count = len(per_episode)
    hamming_values = [row["target_rectangle_hamming_error"] for row in per_episode]
    aggregate = {
        "split": split,
        "condition": condition,
        "episode_count": count,
        "candidate_nearest_score": statistics.fmean(row["candidate_nearest_score"] for row in per_episode) if per_episode else 0.0,
        "candidate_tie_rate": sum(1 for row in per_episode if row["candidate_tie"]) / max(count, 1),
        "candidate_exact_match_rate": sum(1 for row in per_episode if row["candidate_exact_match"]) / max(count, 1),
        "direct_grid_exact_accuracy": sum(1 for row in per_episode if row["direct_grid_exact"]) / max(count, 1),
        "target_rectangle_color_accuracy": color_correct / max(color_total, 1),
        "mean_target_rectangle_hamming_error": statistics.fmean(hamming_values) if hamming_values else 0.0,
        "median_target_rectangle_hamming_error": statistics.median(hamming_values) if hamming_values else 0.0,
        "shape_match_rate": sum(1 for row in per_episode if row["shape_match"]) / max(count, 1),
        "total_loss": totals["total_loss"] / max(count, 1),
        "color_loss": totals["color_loss"] / max(count, 1),
        "height_loss": totals["height_loss"] / max(count, 1),
        "width_loss": totals["width_loss"] / max(count, 1),
    }
    if condition == CONDITION_TWIN_SWAP_SUPPORTS:
        aggregate["swapped_support_alternate_choice_score"] = statistics.fmean(
            row["swapped_support_alternate_choice_score"] for row in per_episode
        ) if per_episode else 0.0
    return aggregate, per_episode


def evaluate_all_conditions(
    *,
    model: VRM,
    episodes: list[TwinEpisode],
    split: str,
    device: torch.device,
    seed: int,
    batch_size: int,
    shape_loss_weight: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    per_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in (
        CONDITION_NORMAL,
        CONDITION_TWIN_SWAP_SUPPORTS,
        CONDITION_PAIR_ORDER_SHUFFLED,
        CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED,
    ):
        aggregate, rows = evaluate_condition(
            model=model,
            episodes=episodes,
            split=split,
            condition=condition,
            device=device,
            seed=seed,
            batch_size=batch_size,
            shape_loss_weight=shape_loss_weight,
        )
        aggregates[condition] = aggregate
        per_condition[condition] = rows
    pair_metrics = pair_level_metrics(episodes, per_condition)
    return aggregates, per_condition, pair_metrics


def _by_episode_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in rows}


def pair_level_metrics(episodes: list[TwinEpisode], per_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    normal = _by_episode_id(per_condition[CONDITION_NORMAL])
    swapped = _by_episode_id(per_condition[CONDITION_TWIN_SWAP_SUPPORTS])
    by_base = group_by_base(episodes)
    pair_count = len(by_base)
    joint_correct = 0
    mapping_flip = 0
    episode_switches = 0
    episode_correct_mapping_switches = 0
    for episode in episodes:
        normal_row = normal[episode.episode_id]
        swapped_row = swapped[episode.episode_id]
        if (
            normal_row["candidate_decision"] is not None
            and swapped_row["candidate_decision"] is not None
            and normal_row["candidate_decision"] != swapped_row["candidate_decision"]
        ):
            episode_switches += 1
        if (
            normal_row["candidate_decision"] == episode.label
            and swapped_row["candidate_decision"] == episode.alternate_label
        ):
            episode_correct_mapping_switches += 1
    for base_id, twins in by_base.items():
        if set(twins) != {"A", "B"}:
            continue
        ep_a = twins["A"]
        ep_b = twins["B"]
        normal_a = normal[ep_a.episode_id]
        normal_b = normal[ep_b.episode_id]
        swapped_a = swapped[ep_a.episode_id]
        swapped_b = swapped[ep_b.episode_id]
        normal_pair_correct = (
            normal_a["candidate_decision"] == ep_a.label
            and normal_b["candidate_decision"] == ep_b.label
        )
        if normal_pair_correct:
            joint_correct += 1
        if (
            normal_pair_correct
            and swapped_a["candidate_decision"] == ep_a.alternate_label
            and swapped_b["candidate_decision"] == ep_b.alternate_label
        ):
            mapping_flip += 1
    return {
        "base_pair_count": pair_count,
        "twin_pair_joint_normal_accuracy": joint_correct / max(pair_count, 1),
        "counterfactual_switch_rate": episode_switches / max(len(episodes), 1),
        "correct_mapping_switch_rate": episode_correct_mapping_switches / max(len(episodes), 1),
        "twin_pair_mapping_flip_consistency": mapping_flip / max(pair_count, 1),
    }


def twin_pair_joint_normal_accuracy(episodes: list[TwinEpisode], normal_rows: list[dict[str, Any]]) -> float:
    normal = _by_episode_id(normal_rows)
    by_base = group_by_base(episodes)
    correct = 0
    for twins in by_base.values():
        if set(twins) != {"A", "B"}:
            continue
        ep_a = twins["A"]
        ep_b = twins["B"]
        if (
            normal[ep_a.episode_id]["candidate_decision"] == ep_a.label
            and normal[ep_b.episode_id]["candidate_decision"] == ep_b.label
        ):
            correct += 1
    return correct / max(len(by_base), 1)


def condition_deltas(aggregates: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for split, split_rows in aggregates.items():
        normal = split_rows[CONDITION_NORMAL]
        deltas[split] = {}
        for condition, row in split_rows.items():
            if condition == CONDITION_NORMAL:
                continue
            deltas[split][condition] = {
                "candidate_score_delta_vs_normal": row["candidate_nearest_score"] - normal["candidate_nearest_score"],
                "direct_exact_delta_vs_normal": row["direct_grid_exact_accuracy"] - normal["direct_grid_exact_accuracy"],
                "mean_hamming_delta_vs_normal": row["mean_target_rectangle_hamming_error"] - normal["mean_target_rectangle_hamming_error"],
                "shape_match_delta_vs_normal": row["shape_match_rate"] - normal["shape_match_rate"],
            }
    return deltas


def save_per_episode(path: Path, per_condition: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(per_condition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
