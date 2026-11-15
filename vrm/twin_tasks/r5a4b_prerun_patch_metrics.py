from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from vrm.data import (
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    MAX_GRID_WIDTH,
    ROLE_SUPPORT_OUTPUT,
    ArcEpisode,
    collate_episodes,
    model_inputs_from_batch,
    move_tensor_batch,
)
from vrm.metrics import factorized_shape_color_loss
from vrm.twin_tasks.r5a4a_arc_manifest import build_r5a4a_splits, episodes_for_task_ids
from vrm.twin_tasks.r5a4a_neural_vrm_data import (
    permute_support_pair_order,
    query_only_masked_batch,
    tensor_digest,
    zero_support_output_colors,
)
from vrm.twin_tasks.r5a4a_neural_vrm_train import DEFAULT_SHAPE_LOSS_WEIGHT, load_state, resolve_device

PROJECT_ROOT = Path("/mnt/d/VRM")
R5A4A_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "r5a4a_neural_vrm_arc_grounded_preflight"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "r5a4b_neural_vrm_prerun_patch"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260701

CONDITIONS = (
    "FRESH_INIT",
    "TRAINED_FULL_SUPPORT_REPLAY",
    "TRAINED_ZEROED_SUPPORT_OUTPUTS",
    "TRAINED_OUTPUT_PAIRING_BROKEN",
    "TRAINED_SUPPORT_PAIR_ORDER_PERMUTED",
    "TRAINED_QUERY_ONLY",
)

LIMITATION_SENTENCE = (
    "Changed-cell color metrics are evaluated at target coordinates after inference and therefore isolate "
    "cell-color quality from shape prediction. They must be read together with exact shape accuracy and exact "
    "full-grid accuracy; they are not a substitute for either."
)
INTERPRETATION_SENTENCE = (
    "A low macro changed-cell score, even with a favorable pooled score, indicates that useful transformed-cell "
    "prediction is concentrated in a small subset of folds rather than broadly distributed across the sentinel."
)
DENSE_SCORING_NOTE = (
    "Dense target-coordinate color scoring measures whether the model assigned useful colors to target positions. "
    "It does not repair an incorrect predicted shape and does not convert color accuracy into exact-grid success."
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def support_output_profile(batch: dict[str, Any], pair_id: int) -> dict[str, Any]:
    pair_indices = batch["support_evidence_pair_indices"][0]
    roles = batch["support_evidence_roles"][0]
    mask = batch["support_evidence_mask"][0].to(torch.bool)
    output_mask = mask & (roles == ROLE_SUPPORT_OUTPUT) & (pair_indices == pair_id)
    rows = batch["support_evidence_rows"][0, output_mask].detach().cpu().tolist()
    cols = batch["support_evidence_cols"][0, output_mask].detach().cpu().tolist()
    colors = batch["support_evidence_colors"][0, output_mask].detach().cpu().tolist()
    if not colors:
        return {
            "source_pair_index": pair_id,
            "source_output_shape": [0, 0],
            "source_output_cell_count": 0,
            "source_output_color_histogram": {},
        }
    return {
        "source_pair_index": pair_id,
        "source_output_shape": [max(rows) + 1, max(cols) + 1],
        "source_output_cell_count": len(colors),
        "source_output_color_histogram": {str(k): v for k, v in sorted(Counter(colors).items())},
    }


def rotate_output_pairing_over_visible_slots(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rotated = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    original_pair_indices = batch["support_evidence_pair_indices"].clone()
    mask = batch["support_evidence_mask"][0].to(torch.bool)
    visible_pair_ids = sorted(int(item) for item in torch.unique(original_pair_indices[0, mask]).detach().cpu().tolist())
    rotation_rows = []
    if len(visible_pair_ids) < 2:
        return rotated, {
            "visible_pair_ids": visible_pair_ids,
            "rotation_applicable": False,
            "source_to_destination_rotation": {},
            "rotation_rows": rotation_rows,
        }
    destination_for_source = {
        source: visible_pair_ids[(idx + 1) % len(visible_pair_ids)]
        for idx, source in enumerate(visible_pair_ids)
    }
    new_pair_indices = rotated["support_evidence_pair_indices"]
    roles = batch["support_evidence_roles"]
    evidence_mask = batch["support_evidence_mask"].to(torch.bool)
    for source, destination in destination_for_source.items():
        source_mask = evidence_mask & (roles == ROLE_SUPPORT_OUTPUT) & (original_pair_indices == source)
        new_pair_indices[source_mask] = destination
        profile = support_output_profile(batch, source)
        profile["destination_pair_index"] = destination
        rotation_rows.append(profile)
    return rotated, {
        "visible_pair_ids": visible_pair_ids,
        "rotation_applicable": True,
        "source_to_destination_rotation": {str(source): destination for source, destination in destination_for_source.items()},
        "rotation_rows": rotation_rows,
    }


def target_coordinate_scoring(
    *,
    episode: ArcEpisode,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> dict[str, Any]:
    target_h = int(batch["query_output_heights"][0].detach().cpu().item())
    target_w = int(batch["query_output_widths"][0].detach().cpu().item())
    query_h = int(batch["query_input_heights"][0].detach().cpu().item())
    query_w = int(batch["query_input_widths"][0].detach().cpu().item())
    pred_h = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    pred_w = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1

    target = batch["query_targets"][0].detach().cpu().view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    query = batch["query_inputs"][0].detach().cpu().view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    color_probs = outputs["color_probabilities"][0].detach().cpu()
    dense_argmax = color_probs.argmax(dim=-1).to(torch.long)
    dense_grid = dense_argmax.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)

    target_entries = []
    changed_entries = []
    unchanged_entries = []
    predicted_change_count = 0
    true_positive_change_count = 0
    correct_changed_color_count = 0
    changed_nll_sum = 0.0
    all_target_correct = 0
    all_target_nll_sum = 0.0

    for row in range(target_h):
        for col in range(target_w):
            flat_idx = row * MAX_GRID_WIDTH + col
            target_color = int(target[row, col].item())
            pred_color = int(dense_grid[row, col].item())
            query_valid = row < query_h and col < query_w
            query_color = int(query[row, col].item()) if query_valid else None
            changed = (not query_valid) or query_color != target_color
            predicted_change = (not query_valid) or pred_color != query_color
            target_prob = float(color_probs[flat_idx, target_color].clamp_min(1e-12).item())
            target_nll = -math.log(target_prob)
            correct = pred_color == target_color
            all_target_correct += int(correct)
            all_target_nll_sum += target_nll
            if predicted_change:
                predicted_change_count += 1
            if changed and predicted_change:
                true_positive_change_count += 1
            entry = {
                "row": row,
                "col": col,
                "target_color": target_color,
                "query_color": query_color,
                "query_valid_at_target_coordinate": query_valid,
                "dense_predicted_color": pred_color,
                "target_color_probability": target_prob,
                "target_color_nll": target_nll,
                "changed_cell": changed,
                "predicted_change": predicted_change,
                "correct_color": correct,
            }
            target_entries.append(entry)
            if changed:
                changed_entries.append(entry)
                changed_nll_sum += target_nll
                correct_changed_color_count += int(correct)
            else:
                unchanged_entries.append(entry)

    changed_count = len(changed_entries)
    unchanged_count = len(unchanged_entries)
    precision = true_positive_change_count / predicted_change_count if predicted_change_count else 0.0
    recall = true_positive_change_count / changed_count if changed_count else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0
        else (0.0 if changed_count else None)
    )
    exact_shape = pred_h == target_h and pred_w == target_w
    dense_digest = tensor_digest(dense_argmax)
    return {
        "query_shape": [query_h, query_w],
        "target_shape": [target_h, target_w],
        "predicted_shape": [pred_h, pred_w],
        "same_shape_regime": query_h == target_h and query_w == target_w,
        "exact_output_shape_correct": exact_shape,
        "exact_full_grid_correct": bool(exact_shape and all_target_correct == len(target_entries)),
        "dense_argmax_canvas": dense_grid.tolist(),
        "dense_argmax_canvas_sha256": dense_digest,
        "target_coordinate_scoring_entries": target_entries,
        "dense_target_coordinate_color_scoring_note": DENSE_SCORING_NOTE,
        "target_rectangle_cell_count": len(target_entries),
        "target_rectangle_dense_color_accuracy": all_target_correct / max(1, len(target_entries)),
        "target_rectangle_dense_color_nll": all_target_nll_sum / max(1, len(target_entries)),
        "changed_cell_count": changed_count,
        "unchanged_cell_count": unchanged_count,
        "changed_cell_color_correct": correct_changed_color_count,
        "changed_cell_color_accuracy": correct_changed_color_count / changed_count if changed_count else None,
        "changed_cell_color_nll": changed_nll_sum / changed_count if changed_count else None,
        "predicted_change_count": predicted_change_count,
        "true_positive_predicted_change_count": true_positive_change_count,
        "predicted_change_precision": precision,
        "predicted_change_recall": recall,
        "predicted_change_f1": f1,
        "correct_changed_color_rate": correct_changed_color_count / changed_count if changed_count else None,
        "target_changed_colors": [entry["target_color"] for entry in changed_entries],
    }


def evaluate_condition_records(
    *,
    condition: str,
    model: Any,
    episodes: list[ArcEpisode],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    pairing_ledger = []
    unsupported = False
    model.eval()
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            eval_episode = episode
            metadata: dict[str, Any] = {}
            if condition == "TRAINED_SUPPORT_PAIR_ORDER_PERMUTED":
                eval_episode, metadata = permute_support_pair_order(episode)
            batch = collate_episodes([eval_episode])
            if condition == "TRAINED_ZEROED_SUPPORT_OUTPUTS":
                batch = zero_support_output_colors(batch)
            elif condition == "TRAINED_OUTPUT_PAIRING_BROKEN":
                batch, metadata = rotate_output_pairing_over_visible_slots(batch)
                pairing_ledger.append(
                    {
                        "condition": condition,
                        "episode_index": episode_index,
                        "task_id": episode.task_id,
                        "query_pair_index": episode.query_pair_index,
                        **metadata,
                    }
                )
            elif condition == "TRAINED_QUERY_ONLY":
                batch = query_only_masked_batch(batch)
            batch = move_tensor_batch(batch, device)
            start = time.perf_counter()
            try:
                outputs = model(**model_inputs_from_batch(batch))
            except ValueError as exc:
                if condition == "TRAINED_QUERY_ONLY" and (
                    "zero valid keys" in str(exc)
                    or "at least one packed support evidence token" in str(exc)
                ):
                    unsupported = True
                    break
                raise
            inference_seconds = time.perf_counter() - start
            loss, components = factorized_shape_color_loss(outputs, batch, DEFAULT_SHAPE_LOSS_WEIGHT)
            score = target_coordinate_scoring(episode=episode, outputs=outputs, batch=batch)
            records.append(
                {
                    "condition": condition,
                    "episode_index": episode_index,
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    "support_pair_indices": [pair.pair_index for pair in episode.supports],
                    "inference_seconds": inference_seconds,
                    "total_existing_model_loss": float(loss.detach().cpu().item()),
                    "color_loss_inside_true_target_rectangle": float(components["color_loss"].detach().cpu().item()),
                    "height_loss": float(components["height_loss"].detach().cpu().item()),
                    "width_loss": float(components["width_loss"].detach().cpu().item()),
                    **metadata,
                    **score,
                }
            )
    if unsupported:
        records.append(
            {
                "condition": condition,
                "supported": False,
                "failure_reason": "QUERY_ONLY_ABLATION_UNSUPPORTED_BY_CURRENT_ARCHITECTURE",
            }
        )
    return records, pairing_ledger


def paired_bootstrap(values: list[float], *, seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None, "bootstrap_replicates": replicates}
    rng = torch.Generator().manual_seed(seed)
    data = torch.tensor(values, dtype=torch.float64)
    count = data.numel()
    indices = torch.randint(0, count, (replicates, count), generator=rng)
    means = data[indices].mean(dim=1).sort().values
    return {
        "count": int(count),
        "mean": float(data.mean().item()),
        "ci95_low": float(means[int(0.025 * (replicates - 1))].item()),
        "ci95_high": float(means[int(0.975 * (replicates - 1))].item()),
        "bootstrap_replicates": replicates,
        "confidence_level": 0.95,
    }


def fold_key(row: dict[str, Any]) -> tuple[str, int] | None:
    if row.get("supported") is False:
        return None
    return (row["task_id"], int(row["query_pair_index"]))


def shape_regime_metrics(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    tables = {}
    for condition, records in condition_records.items():
        valid = [row for row in records if row.get("supported", True)]
        same = [row for row in valid if row["same_shape_regime"]]
        changing = [row for row in valid if not row["same_shape_regime"]]
        correct = sum(1 for row in valid if row["exact_output_shape_correct"])
        tables[condition] = {
            "condition": condition,
            "same_shape_fold_count": len(same),
            "same_shape_correct_shape_count": sum(1 for row in same if row["exact_output_shape_correct"]),
            "same_shape_shape_accuracy": sum(1 for row in same if row["exact_output_shape_correct"]) / len(same) if same else None,
            "shape_changing_fold_count": len(changing),
            "shape_changing_correct_shape_count": sum(1 for row in changing if row["exact_output_shape_correct"]),
            "shape_changing_shape_accuracy": sum(1 for row in changing if row["exact_output_shape_correct"]) / len(changing) if changing else None,
            "overall_correct_shape_count": correct,
            "overall_shape_accuracy": correct / len(valid) if valid else None,
        }
    fresh = {fold_key(row): row for row in condition_records["FRESH_INIT"] if fold_key(row)}
    trained = {fold_key(row): row for row in condition_records["TRAINED_FULL_SUPPORT_REPLAY"] if fold_key(row)}
    intervals = {}
    for regime, predicate in {
        "same_shape": lambda row: row["same_shape_regime"],
        "shape_changing": lambda row: not row["same_shape_regime"],
    }.items():
        deltas = []
        for key in sorted(set(fresh) & set(trained)):
            if predicate(trained[key]):
                deltas.append(float(trained[key]["exact_output_shape_correct"]) - float(fresh[key]["exact_output_shape_correct"]))
        intervals[f"trained_minus_fresh_{regime}_shape_accuracy"] = paired_bootstrap(
            deltas,
            seed=BOOTSTRAP_SEED + (11 if regime == "same_shape" else 17),
        )
    return {
        "metric_note": "Overall shape accuracy is descriptive; same-shape and shape-changing regimes must be read separately.",
        "by_condition": tables,
        "paired_bootstrap_intervals": intervals,
    }


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def changed_cell_metrics(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metric_names = (
        "changed_cell_color_accuracy",
        "changed_cell_color_nll",
        "predicted_change_precision",
        "predicted_change_recall",
        "predicted_change_f1",
        "correct_changed_color_rate",
    )
    by_condition = {}
    for condition, records in condition_records.items():
        valid = [row for row in records if row.get("supported", True)]
        eligible = [row for row in valid if row["changed_cell_count"] > 0]
        changed_counts = [row["changed_cell_count"] for row in eligible]
        total_changed = sum(changed_counts)
        pooled_correct = sum(row["changed_cell_color_correct"] for row in eligible)
        pooled_nll_sum = sum(row["changed_cell_color_nll"] * row["changed_cell_count"] for row in eligible)
        pooled_predicted_change = sum(row["predicted_change_count"] for row in eligible)
        pooled_true_positive = sum(row["true_positive_predicted_change_count"] for row in eligible)
        pooled_precision = pooled_true_positive / pooled_predicted_change if pooled_predicted_change else 0.0
        pooled_recall = pooled_true_positive / total_changed if total_changed else None
        pooled = {
            "changed_cell_color_accuracy": pooled_correct / total_changed if total_changed else None,
            "changed_cell_color_nll": pooled_nll_sum / total_changed if total_changed else None,
            "predicted_change_precision": pooled_precision,
            "predicted_change_recall": pooled_recall,
            "predicted_change_f1": f1(pooled_precision, pooled_recall) if pooled_recall is not None else None,
            "correct_changed_color_rate": pooled_correct / total_changed if total_changed else None,
            "descriptive_only": True,
        }
        macro = {}
        for metric in metric_names:
            values = [row[metric] for row in eligible if row[metric] is not None]
            macro[metric] = {
                "mean": safe_mean(values),
                "median": safe_median(values),
                "eligible_fold_count": len(values),
                "primary_inferential_statistic": True,
            }
        by_condition[condition] = {
            "eligible_changed_cell_fold_count": len(eligible),
            "total_changed_cell_count": total_changed,
            "median_changed_cells_per_eligible_fold": safe_median(changed_counts),
            "minimum_changed_cells_per_eligible_fold": min(changed_counts) if changed_counts else None,
            "maximum_changed_cells_per_eligible_fold": max(changed_counts) if changed_counts else None,
            "pooled": pooled,
            "macro_per_fold": macro,
        }

    comparisons = {}
    baseline_name = "TRAINED_FULL_SUPPORT_REPLAY"
    baseline = {fold_key(row): row for row in condition_records[baseline_name] if fold_key(row) and row["changed_cell_count"] > 0}
    for condition, records in condition_records.items():
        if condition == baseline_name:
            continue
        other = {fold_key(row): row for row in records if fold_key(row) and row.get("changed_cell_count", 0) > 0}
        shared = sorted(set(baseline) & set(other))
        metric_deltas = {}
        for metric in metric_names:
            values = []
            for key in shared:
                left = baseline[key][metric]
                right = other[key][metric]
                if left is None or right is None:
                    continue
                values.append(float(left) - float(right))
            metric_deltas[f"{baseline_name}_minus_{condition}_{metric}"] = paired_bootstrap(
                values,
                seed=BOOTSTRAP_SEED + int(hashlib.sha256(f"{condition}:{metric}".encode()).hexdigest()[:6], 16),
            )
        comparisons[condition] = {
            "paired_fold_count": len(shared),
            "delta_definition": f"{baseline_name} macro per-fold metric minus {condition} macro per-fold metric",
            "metric_deltas": metric_deltas,
        }

    fresh = {fold_key(row): row for row in condition_records["FRESH_INIT"] if fold_key(row) and row.get("changed_cell_count", 0) > 0}
    trained = baseline
    shared = sorted(set(fresh) & set(trained))
    trained_minus_fresh = {}
    for metric in metric_names:
        values = []
        for key in shared:
            left = trained[key][metric]
            right = fresh[key][metric]
            if left is None or right is None:
                continue
            values.append(float(left) - float(right))
        trained_minus_fresh[f"trained_minus_fresh_{metric}"] = paired_bootstrap(
            values,
            seed=BOOTSTRAP_SEED + int(hashlib.sha256(f"trained:fresh:{metric}".encode()).hexdigest()[:6], 16),
        )

    return {
        "changed_cell_metric_note": "Pooled values are descriptive only; macro per-fold values are primary for fold-paired inference.",
        "by_condition": by_condition,
        "paired_bootstrap_comparisons": comparisons,
        "trained_minus_fresh_macro_per_fold": {
            "paired_fold_count": len(shared),
            "metric_deltas": trained_minus_fresh,
        },
    }


def target_changed_region_difficulty(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base_records = [
        row for row in condition_records["TRAINED_FULL_SUPPORT_REPLAY"]
        if row.get("supported", True) and row["changed_cell_count"] > 0
    ]
    rows = []
    total_changed = 0
    single_color = 0
    modal_weighted = 0.0
    distinct_values = []
    changed_counts = []
    for row in base_records:
        colors = row["target_changed_colors"]
        counts = Counter(colors)
        changed_count = len(colors)
        distinct = len(counts)
        modal_fraction = max(counts.values()) / changed_count if changed_count else None
        all_one = distinct == 1 if changed_count else None
        rows.append(
            {
                "task_id": row["task_id"],
                "query_pair_index": row["query_pair_index"],
                "number_of_changed_cells": changed_count,
                "number_of_distinct_colors_among_target_changed_cells": distinct,
                "modal_target_changed_color_fraction": modal_fraction,
                "all_target_changed_cells_share_one_color": all_one,
            }
        )
        total_changed += changed_count
        modal_weighted += (modal_fraction or 0.0) * changed_count
        single_color += int(bool(all_one))
        distinct_values.append(distinct)
        changed_counts.append(changed_count)
    return {
        "analysis_name": "TARGET_CHANGED_REGION_DIFFICULTY_DIAGNOSTICS",
        "diagnostic_only": True,
        "not_a_baseline": True,
        "not_used_in_verdict": True,
        "not_used_for_checkpoint_or_case_selection": True,
        "eligible_changed_cell_fold_count": len(rows),
        "pooled": {
            "total_changed_cell_count": total_changed,
            "weighted_modal_target_changed_color_fraction": modal_weighted / total_changed if total_changed else None,
            "single_color_changed_region_fold_rate": single_color / len(rows) if rows else None,
        },
        "macro": {
            "mean_changed_cells_per_eligible_fold": safe_mean(changed_counts),
            "median_changed_cells_per_eligible_fold": safe_median(changed_counts),
            "mean_distinct_changed_colors_per_eligible_fold": safe_mean(distinct_values),
            "median_distinct_changed_colors_per_eligible_fold": safe_median(distinct_values),
            "mean_modal_target_changed_color_fraction": safe_mean(
                [row["modal_target_changed_color_fraction"] for row in rows if row["modal_target_changed_color_fraction"] is not None]
            ),
        },
        "per_fold": rows,
    }


def compact_dense_ledger(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for condition, records in condition_records.items():
        for row in records:
            if row.get("supported") is False:
                rows.append(row)
                continue
            rows.append(
                {
                    "condition": condition,
                    "episode_index": row["episode_index"],
                    "task_id": row["task_id"],
                    "query_pair_index": row["query_pair_index"],
                    "query_shape": row["query_shape"],
                    "target_shape": row["target_shape"],
                    "predicted_height": row["predicted_shape"][0],
                    "predicted_width": row["predicted_shape"][1],
                    "dense_argmax_canvas": row["dense_argmax_canvas"],
                    "dense_argmax_canvas_sha256": row["dense_argmax_canvas_sha256"],
                    "target_coordinate_scoring_entries": row["target_coordinate_scoring_entries"],
                    "dense_target_coordinate_color_scoring_note": DENSE_SCORING_NOTE,
                }
            )
    return {
        "ledger_note": DENSE_SCORING_NOTE,
        "fixed_canvas_shape": [MAX_GRID_HEIGHT, MAX_GRID_WIDTH],
        "rows": rows,
    }


def final_report_payload(
    *,
    shape_metrics: dict[str, Any],
    changed_metrics: dict[str, Any],
    difficulty: dict[str, Any],
    r5a4a_integrity: dict[str, str],
) -> dict[str, Any]:
    return {
        "experiment": "R5A-4B Pre-Run Patch - Shape, Changed-Cell, and Pairing Integrity",
        "no_training": True,
        "model_modified": False,
        "selected_r5a4a_checkpoint_modified": False,
        "sentinel_modified": False,
        "conditions": list(CONDITIONS),
        "shape_regime_metrics_path": str(ARTIFACT_DIR / "shape_regime_metrics.json"),
        "changed_cell_pooled_and_macro_metrics_path": str(ARTIFACT_DIR / "changed_cell_pooled_and_macro_metrics.json"),
        "dense_prediction_scoring_ledger_path": str(ARTIFACT_DIR / "dense_prediction_scoring_ledger.json"),
        "pairing_rotation_integrity_ledger_path": str(ARTIFACT_DIR / "pairing_rotation_integrity_ledger.json"),
        "target_changed_region_difficulty_diagnostics_path": str(ARTIFACT_DIR / "target_changed_region_difficulty_diagnostics.json"),
        "limitations": [LIMITATION_SENTENCE],
        "interpretation": [INTERPRETATION_SENTENCE],
        "shape_regime_summary": shape_metrics["by_condition"],
        "changed_cell_summary": changed_metrics["by_condition"],
        "target_changed_region_difficulty_diagnostics_summary": {
            key: difficulty[key] for key in ("analysis_name", "diagnostic_only", "not_a_baseline", "not_used_in_verdict", "pooled", "macro")
        },
        "r5a4a_artifact_integrity": r5a4a_integrity,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# R5A-4B Pre-Run Patch",
        "",
        "No model weights were trained or modified. The evaluator replays the frozen R5A-4A initial and selected checkpoints on the frozen sentinel folds.",
        "",
        "## Limitations",
        "",
        LIMITATION_SENTENCE,
        "",
        "## Interpretation",
        "",
        INTERPRETATION_SENTENCE,
        "",
        "## Artifacts",
        "",
    ]
    for key in (
        "shape_regime_metrics_path",
        "changed_cell_pooled_and_macro_metrics_path",
        "dense_prediction_scoring_ledger_path",
        "pairing_rotation_integrity_ledger_path",
        "target_changed_region_difficulty_diagnostics_path",
    ):
        lines.append(f"- {key}: `{report[key]}`")
    lines.extend(["", "## Conditions", "", *[f"- `{condition}`" for condition in report["conditions"]], ""])
    return "\n".join(lines)


def run_r5a4b_prerun_patch(*, device_arg: str = "auto") -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    r5a4a_paths = {
        "initial": R5A4A_ARTIFACT_DIR / "initial_state_before_any_optimizer_step.pt",
        "selected": R5A4A_ARTIFACT_DIR / "selected_trained_state.pt",
        "sentinel_manifest": R5A4A_ARTIFACT_DIR / "arc_sentinel_manifest.json",
    }
    r5a4a_integrity_before = {name: file_sha256(path) for name, path in r5a4a_paths.items()}
    device = resolve_device(device_arg)
    fresh_model, _ = load_state(r5a4a_paths["initial"], device)
    trained_model, _ = load_state(r5a4a_paths["selected"], device)
    splits = build_r5a4a_splits()
    sentinel_episodes = episodes_for_task_ids(splits.tasks_by_id, splits.sentinel_task_ids, source_split="sentinel")

    condition_records: dict[str, list[dict[str, Any]]] = {}
    pairing_ledger: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        model = fresh_model if condition == "FRESH_INIT" else trained_model
        records, condition_pairing = evaluate_condition_records(
            condition=condition,
            model=model,
            episodes=sentinel_episodes,
            device=device,
        )
        condition_records[condition] = records
        pairing_ledger.extend(condition_pairing)

    shape_metrics = shape_regime_metrics(condition_records)
    dense_ledger = compact_dense_ledger(condition_records)
    changed_metrics = changed_cell_metrics(condition_records)
    difficulty = target_changed_region_difficulty(condition_records)
    r5a4a_integrity_after = {name: file_sha256(path) for name, path in r5a4a_paths.items()}
    integrity = {
        "before": r5a4a_integrity_before,
        "after": r5a4a_integrity_after,
        "unchanged": r5a4a_integrity_before == r5a4a_integrity_after,
    }
    report = final_report_payload(
        shape_metrics=shape_metrics,
        changed_metrics=changed_metrics,
        difficulty=difficulty,
        r5a4a_integrity=integrity,
    )

    write_json(ARTIFACT_DIR / "shape_regime_metrics.json", shape_metrics)
    write_json(ARTIFACT_DIR / "dense_prediction_scoring_ledger.json", dense_ledger)
    write_json(ARTIFACT_DIR / "changed_cell_pooled_and_macro_metrics.json", changed_metrics)
    write_json(
        ARTIFACT_DIR / "pairing_rotation_integrity_ledger.json",
        {
            "rotation_rule": "cyclic rotation over actual visible original support-pair IDs only",
            "rows": pairing_ledger,
        },
    )
    write_json(ARTIFACT_DIR / "target_changed_region_difficulty_diagnostics.json", difficulty)
    write_json(ARTIFACT_DIR / "final_report.json", report)
    (PROJECT_ROOT / "docs" / "R5A4B_PRERUN_PATCH.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run R5A-4B pre-run patch metrics.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(to_jsonable(run_r5a4b_prerun_patch(device_arg=args.device)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
