from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from vrm.data import (
    MAX_GRID_HEIGHT,
    MAX_GRID_WIDTH,
    ArcEpisode,
    collate_episodes,
    grid_to_text,
    model_inputs_from_batch,
    move_tensor_batch,
)
from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM
from vrm.twin_tasks.r5a4a_neural_vrm_data import (
    permute_support_pair_order,
    query_only_masked_batch,
    tensor_digest,
    zero_support_output_colors,
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def grid_from_prediction(tokens: torch.Tensor, height: int, width: int) -> list[list[int]]:
    grid = tokens.detach().cpu().to(torch.long).view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    return [[int(cell) for cell in row] for row in grid[:height, :width].tolist()]


def per_fold_metrics(
    *,
    condition: str,
    episode_index: int,
    episode: ArcEpisode,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    components: dict[str, torch.Tensor],
    inference_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = batch["query_targets"][0].detach()
    mask = batch["query_target_mask"][0].detach().to(torch.bool)
    prediction = outputs["color_log_probs"][0].argmax(dim=-1).detach()
    pred_h = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    pred_w = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    target_h = int(batch["query_output_heights"][0].detach().cpu().item())
    target_w = int(batch["query_output_widths"][0].detach().cpu().item())
    matches = prediction == target
    correct = int(matches[mask].sum().detach().cpu().item())
    total = int(mask.sum().detach().cpu().item())
    exact_shape = pred_h == target_h and pred_w == target_w
    row = {
        "condition": condition,
        "episode_index": episode_index,
        "task_id": episode.task_id,
        "query_pair_index": episode.query_pair_index,
        "support_pair_indices": [pair.pair_index for pair in episode.supports],
        "target_shape": [target_h, target_w],
        "predicted_shape": [pred_h, pred_w],
        "exact_full_grid_correct": bool(exact_shape and correct == total),
        "exact_output_shape_correct": bool(exact_shape),
        "target_rectangle_color_correct": correct,
        "target_rectangle_cell_count": total,
        "target_rectangle_cell_accuracy": correct / max(total, 1),
        "color_nll": float(components["color_loss"].detach().cpu().item()),
        "height_nll": float(F.cross_entropy(outputs["height_logits"], batch["height_targets"]).detach().cpu().item()),
        "width_nll": float(F.cross_entropy(outputs["width_logits"], batch["width_targets"]).detach().cpu().item()),
        "total_loss": float(components["total_loss"].detach().cpu().item()),
        "inference_seconds": inference_seconds,
        "prediction_digest": tensor_digest(prediction.detach().cpu()),
        "target_digest": tensor_digest(target.detach().cpu()),
        "decoded_grid": grid_from_prediction(prediction, pred_h, pred_w),
        "_target_color_probabilities": outputs["color_probabilities"][0, mask].detach().cpu(),
    }
    if metadata:
        row.update(metadata)
    return row


def strip_internal_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def aggregate_records(condition: str, records: list[dict[str, Any]], *, task_count: int | None = None) -> dict[str, Any]:
    if not records:
        return {
            "condition": condition,
            "supported": False,
            "fold_count": 0,
            "task_count": task_count or 0,
            "failure_reason": "no_records",
        }
    fold_count = len(records)
    cell_correct = sum(int(row["target_rectangle_color_correct"]) for row in records)
    cell_total = sum(int(row["target_rectangle_cell_count"]) for row in records)
    return {
        "condition": condition,
        "supported": True,
        "fold_count": fold_count,
        "task_count": task_count if task_count is not None else len({row["task_id"] for row in records}),
        "exact_full_grid_accuracy": sum(1 for row in records if row["exact_full_grid_correct"]) / fold_count,
        "exact_output_shape_accuracy": sum(1 for row in records if row["exact_output_shape_correct"]) / fold_count,
        "cell_accuracy_inside_true_target_rectangle": cell_correct / max(cell_total, 1),
        "mean_fold_cell_accuracy_inside_true_target_rectangle": statistics.fmean(
            row["target_rectangle_cell_accuracy"] for row in records
        ),
        "mean_color_negative_log_likelihood_inside_true_target_rectangle": statistics.fmean(
            row["color_nll"] for row in records
        ),
        "mean_height_negative_log_likelihood": statistics.fmean(row["height_nll"] for row in records),
        "mean_width_negative_log_likelihood": statistics.fmean(row["width_nll"] for row in records),
        "mean_total_existing_model_loss": statistics.fmean(row["total_loss"] for row in records),
        "mean_inference_runtime_per_fold_seconds": statistics.fmean(row["inference_seconds"] for row in records),
        "total_target_rectangle_cells": cell_total,
    }


def evaluate_condition(
    *,
    model: VRM,
    episodes: list[ArcEpisode],
    condition: str,
    device: torch.device,
    shape_loss_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    task_count = len({episode.task_id for episode in episodes})
    unsupported_reason: str | None = None
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            eval_episode = episode
            metadata: dict[str, Any] = {}
            if condition == "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER":
                eval_episode, metadata = permute_support_pair_order(episode)
            batch = collate_episodes([eval_episode])
            if condition == "TRAINED_ZEROED_SUPPORT_OUTPUTS":
                batch = zero_support_output_colors(batch)
            elif condition == "TRAINED_QUERY_ONLY":
                batch = query_only_masked_batch(batch)
            batch = move_tensor_batch(batch, device)
            inputs = model_inputs_from_batch(batch)
            start = time.perf_counter()
            try:
                outputs = model(**inputs)
            except ValueError as exc:
                if condition == "TRAINED_QUERY_ONLY" and (
                    "zero valid keys" in str(exc)
                    or "at least one packed support evidence token" in str(exc)
                ):
                    unsupported_reason = "QUERY_ONLY_ABLATION_UNSUPPORTED_BY_CURRENT_ARCHITECTURE"
                    break
                raise
            inference_seconds = time.perf_counter() - start
            loss, components = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
            components = dict(components)
            components["total_loss"] = loss
            records.append(
                per_fold_metrics(
                    condition=condition,
                    episode_index=episode_index,
                    episode=episode,
                    outputs=outputs,
                    batch=batch,
                    components=components,
                    inference_seconds=inference_seconds,
                    metadata=metadata,
                )
            )
    if unsupported_reason:
        return (
            {
                "condition": condition,
                "supported": False,
                "fold_count": 0,
                "task_count": task_count,
                "failure_reason": unsupported_reason,
            },
            [],
        )
    return aggregate_records(condition, records, task_count=task_count), records


def paired_bootstrap_ci(values: list[float], *, seed: int, replicates: int = 10000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "count": 0, "bootstrap_replicates": replicates}
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(replicates):
        total = 0.0
        for _sample in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "ci95_low": means[int(0.025 * (replicates - 1))],
        "ci95_high": means[int(0.975 * (replicates - 1))],
        "count": count,
        "bootstrap_replicates": replicates,
        "confidence_level": 0.95,
    }


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.to(torch.float64).clamp_min(1e-12)
    q = q.to(torch.float64).clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    return float((0.5 * ((p * (p.log() - m.log())).sum(dim=-1) + (q * (q.log() - m.log())).sum(dim=-1))).mean().cpu().item())


def records_by_fold(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["task_id"], int(row["query_pair_index"])): row for row in records}


def paired_trained_vs_fresh_intervals(
    fresh_records: list[dict[str, Any]],
    trained_records: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    fresh = records_by_fold(fresh_records)
    trained = records_by_fold(trained_records)
    keys = sorted(set(fresh) & set(trained))
    specs = {
        "total_loss_improvement_fresh_minus_trained": lambda f, t: f["total_loss"] - t["total_loss"],
        "color_nll_improvement_fresh_minus_trained": lambda f, t: f["color_nll"] - t["color_nll"],
        "cell_accuracy_improvement_trained_minus_fresh": lambda f, t: t["target_rectangle_cell_accuracy"] - f["target_rectangle_cell_accuracy"],
        "shape_accuracy_improvement_trained_minus_fresh": lambda f, t: float(t["exact_output_shape_correct"]) - float(f["exact_output_shape_correct"]),
        "exact_grid_improvement_trained_minus_fresh": lambda f, t: float(t["exact_full_grid_correct"]) - float(f["exact_full_grid_correct"]),
    }
    intervals = {
        name: paired_bootstrap_ci([float(fn(fresh[key], trained[key])) for key in keys], seed=seed + idx * 1009, replicates=replicates)
        for idx, (name, fn) in enumerate(specs.items())
    }
    intervals["paired_fold_count"] = len(keys)
    intervals["positive_values_favor"] = "TRAINED_FULL_SUPPORT over FRESH_INIT"
    return intervals


def support_ablation_intervals(
    full_records: list[dict[str, Any]],
    condition_records: list[dict[str, Any]],
    *,
    condition: str,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    full = records_by_fold(full_records)
    cond = records_by_fold(condition_records)
    keys = sorted(set(full) & set(cond))
    specs = {
        "delta_exact_grid_accuracy_full_minus_condition": lambda f, c: float(f["exact_full_grid_correct"]) - float(c["exact_full_grid_correct"]),
        "delta_cell_accuracy_full_minus_condition": lambda f, c: f["target_rectangle_cell_accuracy"] - c["target_rectangle_cell_accuracy"],
        "delta_shape_accuracy_full_minus_condition": lambda f, c: float(f["exact_output_shape_correct"]) - float(c["exact_output_shape_correct"]),
        "delta_color_nll_condition_minus_full": lambda f, c: c["color_nll"] - f["color_nll"],
        "delta_total_loss_condition_minus_full": lambda f, c: c["total_loss"] - f["total_loss"],
        "decoded_grid_change_rate": lambda f, c: float(f["prediction_digest"] != c["prediction_digest"] or f["predicted_shape"] != c["predicted_shape"]),
        "shape_prediction_change_rate": lambda f, c: float(f["predicted_shape"] != c["predicted_shape"]),
        "mean_js_divergence_color_distributions": lambda f, c: js_divergence(f["_target_color_probabilities"], c["_target_color_probabilities"]),
    }
    intervals = {
        name: paired_bootstrap_ci([float(fn(full[key], cond[key])) for key in keys], seed=seed + idx * 1009, replicates=replicates)
        for idx, (name, fn) in enumerate(specs.items())
    }
    intervals["condition"] = condition
    intervals["paired_fold_count"] = len(keys)
    intervals["positive_values_favor"] = "TRAINED_FULL_SUPPORT"
    return intervals


def condition_payload(aggregate: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"aggregate": aggregate, "per_fold": [strip_internal_record(row) for row in records]}


def stable_payload_without_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(to_jsonable(payload), sort_keys=True))

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items() if "runtime" not in key and key != "inference_seconds"}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(clean)


def choose_representative_cases(
    episodes: list[ArcEpisode],
    fresh_records: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
    ablation_records: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, tuple[str, int]]]:
    fresh = records_by_fold(fresh_records)
    full = records_by_fold(full_records)
    keys = sorted(set(fresh) & set(full))
    selected: list[tuple[str, tuple[str, int]]] = []
    used: set[tuple[str, int]] = set()

    def add(label: str, ordered: list[tuple[str, int]], count: int) -> None:
        for key in ordered:
            if key in used:
                continue
            selected.append((label, key))
            used.add(key)
            if sum(1 for item_label, _ in selected if item_label == label) >= count:
                return

    add("largest_trained_vs_fresh_improvement", sorted(keys, key=lambda key: (-(fresh[key]["total_loss"] - full[key]["total_loss"]), key[0], key[1])), 3)
    ablation_loss_increase = []
    for key in keys:
        best = -math.inf
        for rows in ablation_records.values():
            lookup = records_by_fold(rows)
            if key in lookup:
                best = max(best, lookup[key]["total_loss"] - full[key]["total_loss"])
        ablation_loss_increase.append((best, key))
    add("largest_support_ablation_loss_increase", [key for _score, key in sorted(ablation_loss_increase, key=lambda item: (-item[0], item[1][0], item[1][1]))], 3)
    add("trained_and_fresh_nearly_identical", sorted(keys, key=lambda key: (abs(fresh[key]["total_loss"] - full[key]["total_loss"]) + abs(fresh[key]["target_rectangle_cell_accuracy"] - full[key]["target_rectangle_cell_accuracy"]), key[0], key[1])), 3)
    add("deterministic_hash_selection", sorted(keys, key=lambda key: hashlib.sha256(f"R5A4A_CASE:{key[0]}:{key[1]}".encode("utf-8")).hexdigest()), 3)
    if len(selected) < 12:
        add("deterministic_hash_selection", sorted(keys, key=lambda key: hashlib.sha256(f"R5A4A_CASE_FILL:{key[0]}:{key[1]}".encode("utf-8")).hexdigest()), 12 - len(selected))
    return selected[:12]


def render_case_cards(
    episodes: list[ArcEpisode],
    fresh_records: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
    condition_records: dict[str, list[dict[str, Any]]],
) -> str:
    episode_by_key = {(episode.task_id, episode.query_pair_index): episode for episode in episodes}
    lookups = {
        "FRESH_INIT": records_by_fold(fresh_records),
        "TRAINED_FULL_SUPPORT": records_by_fold(full_records),
        **{condition: records_by_fold(rows) for condition, rows in condition_records.items()},
    }
    selected = choose_representative_cases(episodes, fresh_records, full_records, condition_records)
    lines = [
        "# R5A-4A representative diagnostic case cards",
        "",
        "These 12 folds are selected by predeclared deterministic rules for inspection only, not as demonstrations.",
        "",
    ]
    for index, (label, key) in enumerate(selected, start=1):
        episode = episode_by_key[key]
        lines.extend(
            [
                f"## Case {index}: {label}",
                "",
                f"- task_id: `{episode.task_id}`",
                f"- held_out_pair: `{episode.query_pair_index}`",
                "",
                "```text",
                render_episode_grids_inline(episode),
                "```",
                "",
            ]
        )
        for condition, lookup in lookups.items():
            row = lookup.get(key)
            lines.append(f"### {condition}")
            if row is None:
                lines.extend(["", "```text", "unsupported or not evaluated", "```", ""])
                continue
            lines.extend(
                [
                    "",
                    f"- predicted_shape: `{row['predicted_shape'][0]}x{row['predicted_shape'][1]}`",
                    f"- total_loss: `{row['total_loss']:.6f}`",
                    f"- cell_accuracy: `{row['target_rectangle_cell_accuracy']:.6f}`",
                    "",
                    "```text",
                    grid_text(row["decoded_grid"]),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_episode_grids_inline(episode: ArcEpisode) -> str:
    sections = []
    for support in episode.supports:
        sections.append(f"support {support.pair_index} input ({support.input.height}x{support.input.width})\n{grid_to_text(support.input)}")
        sections.append(f"support {support.pair_index} output ({support.output.height}x{support.output.width})\n{grid_to_text(support.output)}")
    sections.append(f"held-out query input ({episode.query.input.height}x{episode.query.input.width})\n{grid_to_text(episode.query.input)}")
    sections.append(f"hidden target output ({episode.query.output.height}x{episode.query.output.width})\n{grid_to_text(episode.query.output)}")
    return "\n\n".join(sections)


def grid_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)
