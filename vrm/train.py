from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .data import (
    ARC_COLOR_COUNT,
    ArcEpisode,
    ArcPair,
    DEFAULT_ARC_DATA_ROOT,
    Grid,
    LOCAL_ARC_DATA_ROOT,
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    MAX_GRID_WIDTH,
    PAD_ID,
    ROLE_SUPPORT_INPUT,
    ROLE_SUPPORT_OUTPUT,
    build_dataset,
    collate_episodes,
    grid_to_text,
    model_inputs_from_batch,
    move_tensor_batch,
)
from .metrics import factorized_shape_color_loss
from .model import VRM, VRMConfig

ARTIFACTS_DIR = Path("/mnt/d/VRM/artifacts")
CHECKPOINTS_DIR = Path("/mnt/d/VRM/checkpoints")
V1_ARTIFACTS_DIR = ARTIFACTS_DIR / "vrm_v1_capacity50"
V1_TASK_MANIFEST = V1_ARTIFACTS_DIR / "vrm_v1_task_manifest.json"
V1_1_ARTIFACTS_DIR = ARTIFACTS_DIR / "vrm_v1_1_capacity50"
V1_1_SOURCE_ZIP = Path("/mnt/d/VRM/VRM_V1_1_SOURCE_SNAPSHOT.zip")
V1_1_SOURCE_MANIFEST = Path("/mnt/d/VRM/VRM_V1_1_SOURCE_SNAPSHOT_manifest.txt")
V1_1_SUPPORT_ABLATION_DIR = ARTIFACTS_DIR / "vrm_v1_1_support_ablation"
V1_1_BEST_CHECKPOINT = CHECKPOINTS_DIR / "vrm_v1_1_best_train_hamming.pt"
V1_1_TASK_MANIFEST = V1_1_ARTIFACTS_DIR / "vrm_v1_1_task_manifest.json"
V1_1_METRICS = V1_1_ARTIFACTS_DIR / "vrm_v1_1_metrics.json"
V1_1_DEFAULT_STEPS = 1500
V1_1_EVAL_STEPS = (0, 50, 100, 250, 500, 1000, 1500)
V1_1_PREDICTION_STEPS = (0, 250, 1500)
V1_1_ARCHITECTURE = "VRM V1.1 - Masked Query Spatial Evidence Rewrite"
V1_1_ARCHITECTURE_DISPLAY = "VRM V1.1 — Masked Query Spatial Evidence Rewrite"
SUPPORT_ABLATION_CONDITIONS = (
    "NORMAL",
    "OUTPUT_DERANGED_WITHIN_TASK",
    "PAIR_ORDER_SHUFFLED",
    "OUTPUT_COLOR_PERMUTED",
    "SUPPORT_EVIDENCE_PATH_ZEROED",
)
BANNED_FORWARD_NAMES = {
    "task_id",
    "task_ids",
    "puzzle_id",
    "puzzle_identifier",
    "puzzle_identifiers",
    "query_target",
    "query_targets",
    "labels",
}
BANNED_STATE_TERMS = ("puzzle_emb", "puzzle_identifier", "sparse_embedding", "task_id")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> dict[str, Any]:
    set_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic_algorithms = "enabled_warn_only"
    except TypeError:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = "enabled"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "python_random_seed": seed,
        "torch_manual_seed": seed,
        "torch_cuda_manual_seed_all": seed if torch.cuda.is_available() else None,
        "torch_deterministic_algorithms": deterministic_algorithms,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False))
        if hasattr(torch.backends, "cudnn")
        else None,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False))
        if hasattr(torch.backends, "cudnn")
        else None,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def save_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(args: argparse.Namespace) -> VRMConfig:
    return VRMConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        pair_encoder_layers=args.pair_encoder_layers,
        rewrite_layers=args.rewrite_layers,
        rewrite_steps=args.rewrite_steps,
        rule_slots=args.rule_slots,
        dropout=args.dropout,
        output_color_count=ARC_COLOR_COUNT,
        shape_classes=MAX_GRID_HEIGHT,
    )


def parameter_count(model: VRM) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def optimizer_description(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    group = optimizer.param_groups[0]
    return {
        "name": optimizer.__class__.__name__,
        "lr": group.get("lr"),
        "betas": group.get("betas"),
        "eps": group.get("eps"),
        "weight_decay": group.get("weight_decay"),
    }


def select_deterministic_subset(task_ids: list[str], count: int, seed: int, salt: int) -> list[str]:
    if len(task_ids) < count:
        raise RuntimeError(f"need at least {count} task IDs, found {len(task_ids)}")
    shuffled = sorted(task_ids)
    rng = random.Random(seed + salt)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count])


def load_prior_v1_task_ids(manifest_path: Path = V1_TASK_MANIFEST) -> tuple[list[str], list[str], str]:
    if not manifest_path.exists():
        return [], [], "regenerated_prior_manifest_missing"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], [], "regenerated_prior_manifest_invalid_json"
    train_ids = payload.get("train_50_task_ids")
    unseen_ids = payload.get("unseen_50_task_ids")
    if (
        isinstance(train_ids, list)
        and isinstance(unseen_ids, list)
        and len(train_ids) == 50
        and len(unseen_ids) == 50
        and len(set(train_ids)) == 50
        and len(set(unseen_ids)) == 50
        and set(train_ids).isdisjoint(unseen_ids)
    ):
        return sorted(str(task_id) for task_id in train_ids), sorted(str(task_id) for task_id in unseen_ids), "reused_prior_v1_manifest"
    return [], [], "regenerated_prior_manifest_invalid_contents"


def select_v1_1_capacity50_episode_sets(data: dict[str, Any], seed: int) -> dict[str, Any]:
    prior_train_ids, prior_unseen_ids, selection_status = load_prior_v1_task_ids()
    available_ids = {task.task_id for task in data["tasks"]}
    if prior_train_ids and prior_unseen_ids and set(prior_train_ids).issubset(available_ids) and set(prior_unseen_ids).issubset(available_ids):
        train_task_ids = prior_train_ids
        unseen_task_ids = prior_unseen_ids
    else:
        selection_status = selection_status if selection_status.startswith("regenerated") else "regenerated_prior_manifest_task_missing"
        train_candidates = [task_id for task_id, split in data["split_assignments"].items() if split == "train"]
        unseen_candidates = [task_id for task_id, split in data["split_assignments"].items() if split == "validation"]
        train_task_ids = select_deterministic_subset(train_candidates, 50, seed, salt=50_000)
        unseen_task_ids = select_deterministic_subset(unseen_candidates, 50, seed, salt=75_000)
    if set(train_task_ids).intersection(unseen_task_ids):
        raise RuntimeError("train and unseen task IDs overlap")

    train_set = set(train_task_ids)
    unseen_set = set(unseen_task_ids)
    order = {task_id: idx for idx, task_id in enumerate(train_task_ids + unseen_task_ids)}
    train_episodes = [ep for ep in data["episodes"] if ep.task_id in train_set]
    unseen_episodes = [ep for ep in data["episodes"] if ep.task_id in unseen_set]
    train_episodes.sort(key=lambda ep: (order[ep.task_id], ep.query_pair_index))
    unseen_episodes.sort(key=lambda ep: (order[ep.task_id], ep.query_pair_index))
    task_by_id = {task.task_id: task for task in data["tasks"]}
    train_counts = {task_id: len(task_by_id[task_id].train) for task_id in train_task_ids}
    unseen_counts = {task_id: len(task_by_id[task_id].train) for task_id in unseen_task_ids}
    if len(train_episodes) != sum(train_counts.values()):
        raise RuntimeError("train episode count mismatch")
    if len(unseen_episodes) != sum(unseen_counts.values()):
        raise RuntimeError("unseen episode count mismatch")
    return {
        "selection_status": selection_status,
        "prior_manifest_path": str(V1_TASK_MANIFEST),
        "train_task_ids": train_task_ids,
        "unseen_task_ids": unseen_task_ids,
        "train_episodes": train_episodes,
        "unseen_episodes": unseen_episodes,
        "train_episode_counts": train_counts,
        "unseen_episode_counts": unseen_counts,
    }


class DeterministicCyclingSampler:
    def __init__(self, episode_count: int, seed: int) -> None:
        if episode_count < 1:
            raise ValueError("episode_count must be positive")
        self.episode_count = episode_count
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self.epoch = -1
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(self.episode_count))
        self.rng.shuffle(self.order)
        self.position = 0
        self.epoch += 1

    def next_indices(self, batch_size: int) -> list[int]:
        indices: list[int] = []
        while len(indices) < batch_size:
            if self.position >= self.episode_count:
                self._reshuffle()
            indices.append(self.order[self.position])
            self.position += 1
        return indices


def initialize_fresh_v1_1_model(config: VRMConfig, device: torch.device) -> tuple[VRM, dict[str, Any]]:
    model = VRM(config).to(device)
    report = {
        "architecture": V1_1_ARCHITECTURE_DISPLAY,
        "fresh_random_initialization": True,
        "loaded_checkpoint": False,
        "loaded_checkpoint_path": None,
        "forbidden_checkpoint_paths_not_loaded": [
            str(CHECKPOINTS_DIR / "vrm_v1_capacity50_step1500.pt"),
            str(CHECKPOINTS_DIR / "vrm_v1_1_step1500.pt"),
            str(CHECKPOINTS_DIR / "vrm_v1_1_best_train_hamming.pt"),
        ],
        "parameter_count": parameter_count(model),
        "config": asdict(config),
    }
    return model, report


def initialize_fresh_v1_model(config: VRMConfig, device: torch.device) -> tuple[VRM, dict[str, Any]]:
    return initialize_fresh_v1_1_model(config, device)


def compute_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    if step <= 0:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def compute_grad_norm(model: VRM) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        norm = parameter.grad.detach().norm(2).item()
        total += norm * norm
    return math.sqrt(total)


def max_abs_parameter(model: VRM) -> float:
    return max(float(parameter.detach().abs().max().cpu().item()) for parameter in model.parameters())


def ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite tensor detected: {name}")


def ensure_finite_components(components: dict[str, torch.Tensor]) -> None:
    for name, value in components.items():
        ensure_finite_tensor(name, value)


def ensure_finite_gradients(model: VRM) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient detected: {name}")


def ensure_finite_parameters(model: VRM) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"non-finite parameter detected: {name}")


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached).all().item()),
        "min": float(detached.min().cpu().item()) if detached.numel() else None,
        "max": float(detached.max().cpu().item()) if detached.numel() else None,
        "mean": float(detached.float().mean().cpu().item()) if detached.numel() else None,
    }


def save_diagnostic_failure(
    *,
    step: int,
    reason: str,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any] | None,
    outputs: dict[str, torch.Tensor] | None,
) -> None:
    diagnostic_dir = V1_1_ARTIFACTS_DIR / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "reason": reason,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        diagnostic_dir / f"vrm_v1_1_diagnostic_step{step:04d}.pt",
    )
    metadata = {
        "step": step,
        "reason": reason,
        "batch_task_ids": batch.get("task_ids") if batch else None,
        "query_pair_indices": batch.get("query_pair_indices") if batch else None,
        "source_splits": batch.get("source_splits") if batch else None,
    }
    save_json(diagnostic_dir / f"vrm_v1_1_diagnostic_batch_step{step:04d}.json", metadata)
    stats: dict[str, Any] = {}
    if batch:
        stats["batch"] = {key: tensor_stats(value) for key, value in batch.items() if torch.is_tensor(value)}
    if outputs:
        stats["outputs"] = {key: tensor_stats(value) for key, value in outputs.items() if torch.is_tensor(value)}
    save_json(diagnostic_dir / f"vrm_v1_1_diagnostic_tensor_stats_step{step:04d}.json", stats)


def forward_loss(
    model: VRM,
    batch: dict[str, Any],
    shape_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = model(**model_inputs_from_batch(batch))
    loss, components = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
    return loss, outputs, components


def instrumentation_from_outputs(
    *,
    step: int,
    loss: torch.Tensor,
    components: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    learning_rate: float,
    grad_norm_before_clip: float | None,
    grad_norm_after_clip: float | None,
    model: VRM,
) -> dict[str, Any]:
    with torch.no_grad():
        height_pred = outputs["height_logits"].argmax(dim=-1) + 1
        width_pred = outputs["width_logits"].argmax(dim=-1) + 1
        height_hist = torch.bincount(height_pred.detach().cpu() - 1, minlength=MAX_GRID_HEIGHT).tolist()
        width_hist = torch.bincount(width_pred.detach().cpu() - 1, minlength=MAX_GRID_WIDTH).tolist()
        canvas = outputs["final_canvas"].detach()
        canvas_rms = torch.sqrt(canvas.pow(2).mean(dim=-1)).mean()
        gate = outputs["copy_gate"].detach()
        return {
            "event": "train_step",
            "global_step": step,
            "total_loss": float(loss.detach().cpu().item()),
            "color_loss": float(components["color_loss"].detach().cpu().item()),
            "height_loss": float(components["height_loss"].detach().cpu().item()),
            "width_loss": float(components["width_loss"].detach().cpu().item()),
            "learning_rate": learning_rate,
            "grad_norm_before_clip": grad_norm_before_clip,
            "grad_norm_after_clip": grad_norm_after_clip,
            "maximum_absolute_parameter_value": max_abs_parameter(model),
            "maximum_absolute_canvas_value": float(canvas.abs().max().cpu().item()),
            "mean_canvas_rms_norm": float(canvas_rms.cpu().item()),
            "maximum_absolute_color_log_probability": float(outputs["color_log_probs"].detach().abs().max().cpu().item()),
            "copy_gate_mean": float(gate.mean().cpu().item()),
            "copy_gate_min": float(gate.min().cpu().item()),
            "copy_gate_max": float(gate.max().cpu().item()),
            "copy_entropy_mean": float(outputs["copy_entropy"].detach().mean().cpu().item()),
            "paint_entropy_mean": float(outputs["paint_entropy"].detach().mean().cpu().item()),
            "predicted_height_mean": float(height_pred.float().mean().cpu().item()),
            "predicted_width_mean": float(width_pred.float().mean().cpu().item()),
            "predicted_height_histogram": height_hist,
            "predicted_width_histogram": width_hist,
        }


def episode_metrics(
    *,
    step: int,
    split: str,
    episode_index: int,
    episode: Any,
    components: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    color_prediction = outputs["color_log_probs"].argmax(dim=-1)[0].detach().cpu().to(torch.long)
    target = batch["query_targets"][0].detach().cpu().to(torch.long)
    target_mask = batch["query_target_mask"][0].detach().cpu().to(torch.bool)
    target_h = int(batch["query_output_heights"][0].detach().cpu().item())
    target_w = int(batch["query_output_widths"][0].detach().cpu().item())
    pred_h = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    pred_w = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    matches = color_prediction == target
    correct = int(matches[target_mask].sum().item())
    total = int(target_mask.sum().item())
    hamming = total - correct
    height_correct = pred_h == target_h
    width_correct = pred_w == target_w
    exact = bool(height_correct and width_correct and hamming == 0)
    record = {
        "step": step,
        "split": split,
        "episode_index": episode_index,
        "task_id": episode.task_id,
        "query_pair_index": episode.query_pair_index,
        "support_pair_indices": [pair.pair_index for pair in episode.supports],
        "target_shape": [target_h, target_w],
        "predicted_shape": [pred_h, pred_w],
        "target_rectangle_hamming_error": hamming,
        "target_rectangle_color_correct": correct,
        "target_rectangle_color_total": total,
        "target_rectangle_color_accuracy": correct / max(total, 1),
        "shape_height_correct": height_correct,
        "shape_width_correct": width_correct,
        "output_shape_match": bool(height_correct and width_correct),
        "exact_decoded_grid_correct": exact,
        "exact_padded_grid_correct": None,
        "invalid_padding_pattern": None,
        "total_loss": float(components["total_loss"].detach().cpu().item()),
        "color_loss": float(components["color_loss"].detach().cpu().item()),
        "height_loss": float(components["height_loss"].detach().cpu().item()),
        "width_loss": float(components["width_loss"].detach().cpu().item()),
    }
    prediction_record = {
        "episode_index": episode_index,
        "episode": episode,
        "prediction_colors": color_prediction,
        "predicted_height": pred_h,
        "predicted_width": pred_w,
        "metrics": record,
    }
    return record, prediction_record


def evaluate_split(
    *,
    model: VRM,
    episodes: list[Any],
    split: str,
    device: torch.device,
    shape_loss_weight: float,
    step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    per_episode: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    totals = {"total_loss": 0.0, "color_loss": 0.0, "height_loss": 0.0, "width_loss": 0.0}
    color_correct = 0
    color_total = 0
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            batch = move_tensor_batch(collate_episodes([episode]), device)
            _, outputs, components = forward_loss(model, batch, shape_loss_weight)
            record, prediction_record = episode_metrics(
                step=step,
                split=split,
                episode_index=episode_index,
                episode=episode,
                components=components,
                outputs=outputs,
                batch=batch,
            )
            per_episode.append(record)
            prediction_records.append(prediction_record)
            for key in totals:
                totals[key] += record[key]
            color_correct += record["target_rectangle_color_correct"]
            color_total += record["target_rectangle_color_total"]
    count = len(per_episode)
    hamming_values = [record["target_rectangle_hamming_error"] for record in per_episode]
    aggregate = {
        "step": step,
        "split": split,
        "episode_count": count,
        "total_loss": totals["total_loss"] / max(count, 1),
        "color_loss": totals["color_loss"] / max(count, 1),
        "height_loss": totals["height_loss"] / max(count, 1),
        "width_loss": totals["width_loss"] / max(count, 1),
        "shape_height_accuracy": sum(1 for r in per_episode if r["shape_height_correct"]) / max(count, 1),
        "shape_width_accuracy": sum(1 for r in per_episode if r["shape_width_correct"]) / max(count, 1),
        "output_shape_match_rate": sum(1 for r in per_episode if r["output_shape_match"]) / max(count, 1),
        "target_rectangle_color_accuracy": color_correct / max(color_total, 1),
        "exact_decoded_grid_accuracy": sum(1 for r in per_episode if r["exact_decoded_grid_correct"]) / max(count, 1),
        "mean_target_rectangle_hamming_error": statistics.fmean(hamming_values) if hamming_values else 0.0,
        "median_target_rectangle_hamming_error": statistics.median(hamming_values) if hamming_values else 0.0,
        "minimum_target_rectangle_hamming_error": min(hamming_values) if hamming_values else 0,
        "exact_padded_grid_accuracy": None,
        "invalid_padding_pattern_rate": None,
        "legacy_metric_note": "V1.1 predicts output shape explicitly; padding-pattern metrics are not primary.",
    }
    return aggregate, per_episode, prediction_records


def aggregate_per_task(per_episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in per_episode:
        grouped.setdefault(record["task_id"], []).append(record)
    rows = []
    for task_id in sorted(grouped):
        records = grouped[task_id]
        hamming = [record["target_rectangle_hamming_error"] for record in records]
        rows.append(
            {
                "task_id": task_id,
                "episode_count": len(records),
                "mean_total_loss": statistics.fmean(record["total_loss"] for record in records),
                "mean_color_loss": statistics.fmean(record["color_loss"] for record in records),
                "mean_target_rectangle_hamming_error": statistics.fmean(hamming),
                "minimum_target_rectangle_hamming_error": min(hamming),
                "target_rectangle_color_accuracy": sum(record["target_rectangle_color_correct"] for record in records)
                / max(1, sum(record["target_rectangle_color_total"] for record in records)),
                "exact_decoded_grid_count": sum(1 for record in records if record["exact_decoded_grid_correct"]),
                "exact_decoded_grid_rate": sum(1 for record in records if record["exact_decoded_grid_correct"]) / len(records),
                "shape_match_rate": sum(1 for record in records if record["output_shape_match"]) / len(records),
            }
        )
    return rows


def compact_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "query_pair_index": record["query_pair_index"],
        "support_pair_indices": record["support_pair_indices"],
        "target_shape": record["target_shape"],
        "predicted_shape": record["predicted_shape"],
        "target_rectangle_hamming_error": record["target_rectangle_hamming_error"],
        "target_rectangle_color_accuracy": record["target_rectangle_color_accuracy"],
        "shape_height_correct": record["shape_height_correct"],
        "shape_width_correct": record["shape_width_correct"],
        "output_shape_match": record["output_shape_match"],
        "exact_decoded_grid_correct": record["exact_decoded_grid_correct"],
        "total_loss": record["total_loss"],
        "color_loss": record["color_loss"],
        "height_loss": record["height_loss"],
        "width_loss": record["width_loss"],
    }


def color_grid_to_text(tokens: torch.Tensor, height: int, width: int) -> str:
    grid = tokens.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    rows = []
    for row in grid[:height, :width].tolist():
        rows.append(" ".join(str(int(cell)) for cell in row))
    return "\n".join(rows)


def select_prediction_cases(
    train_predictions: list[dict[str, Any]],
    unseen_predictions: list[dict[str, Any]],
    final_step: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    best_train = sorted(
        train_predictions,
        key=lambda item: (
            item["metrics"]["target_rectangle_hamming_error"],
            item["metrics"]["task_id"],
            item["metrics"]["query_pair_index"],
        ),
    )[:10 if final_step else 5]
    cases.extend({"label": "best_train", **item} for item in best_train)
    if final_step:
        worst_train = sorted(
            train_predictions,
            key=lambda item: (
                -item["metrics"]["target_rectangle_hamming_error"],
                item["metrics"]["task_id"],
                item["metrics"]["query_pair_index"],
            ),
        )[:10]
        cases.extend({"label": "worst_train", **item} for item in worst_train)
    best_unseen = sorted(
        unseen_predictions,
        key=lambda item: (
            item["metrics"]["target_rectangle_hamming_error"],
            item["metrics"]["task_id"],
            item["metrics"]["query_pair_index"],
        ),
    )[:10 if final_step else 5]
    cases.extend({"label": "best_unseen", **item} for item in best_unseen)
    return cases


def render_predictions(
    *,
    step: int,
    train_predictions: list[dict[str, Any]],
    unseen_predictions: list[dict[str, Any]],
    final_step: bool,
) -> str:
    lines = [
        f"# VRM V1.1 predictions at step {step}",
        "",
        "Predicted outputs are cropped to the model-predicted height and width.",
        "",
    ]
    for record in select_prediction_cases(train_predictions, unseen_predictions, final_step):
        episode = record["episode"]
        metrics = record["metrics"]
        pred_h = record["predicted_height"]
        pred_w = record["predicted_width"]
        lines.extend(
            [
                f"## {record['label']}: task {episode.task_id}, query pair {episode.query_pair_index}",
                "",
                f"support_pair_indices: {metrics['support_pair_indices']}",
                f"target H/W: {metrics['target_shape'][0]} x {metrics['target_shape'][1]}",
                f"predicted H/W: {pred_h} x {pred_w}",
                f"target-rectangle Hamming error: {metrics['target_rectangle_hamming_error']}",
                f"target-rectangle color accuracy: {metrics['target_rectangle_color_accuracy']:.6f}",
                f"exact decoded result: {metrics['exact_decoded_grid_correct']}",
                "",
                "support input/output pairs:",
            ]
        )
        for support in episode.supports:
            lines.extend(
                [
                    f"- support pair {support.pair_index} input ({support.input.height}x{support.input.width})",
                    grid_to_text(support.input),
                    f"- support pair {support.pair_index} output ({support.output.height}x{support.output.width})",
                    grid_to_text(support.output),
                ]
            )
        lines.extend(
            [
                f"query input ({episode.query.input.height}x{episode.query.input.width}):",
                grid_to_text(episode.query.input),
                f"target output ({episode.query.output.height}x{episode.query.output.width}):",
                grid_to_text(episode.query.output),
                "predicted output cropped to predicted H x W:",
                color_grid_to_text(record["prediction_colors"], pred_h, pred_w),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def verify_leave_one_out(episodes: list[Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in data["tasks"]}
    failures = []
    for episode in episodes:
        expected = {
            pair.pair_index for pair in task_by_id[episode.task_id].train if pair.pair_index != episode.query_pair_index
        }
        actual = {pair.pair_index for pair in episode.supports}
        if expected != actual:
            failures.append(
                {
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    "expected_support_pair_indices": sorted(expected),
                    "actual_support_pair_indices": sorted(actual),
                }
            )
    return failures


def integrity_checks(data: dict[str, Any], selection: dict[str, Any], model: VRM, data_root: Path) -> dict[str, Any]:
    train_ids = set(selection["train_task_ids"])
    unseen_ids = set(selection["unseen_task_ids"])
    example_batch = collate_episodes(selection["train_episodes"][:1])
    model_input_keys = set(model_inputs_from_batch(example_batch))
    forward_params = set(inspect.signature(VRM.forward).parameters)
    state_keys = [key.lower() for key in model.state_dict()]
    checks = {
        "local_vrm_data_root_used": Path(data_root) == LOCAL_ARC_DATA_ROOT,
        "exactly_50_train_tasks": len(train_ids) == 50,
        "exactly_50_unseen_tasks": len(unseen_ids) == 50,
        "train_unseen_disjoint": train_ids.isdisjoint(unseen_ids),
        "selection_status": selection["selection_status"],
        "prior_manifest_reused": selection["selection_status"] == "reused_prior_v1_manifest",
        "leave_one_out_correct": not verify_leave_one_out(selection["train_episodes"] + selection["unseen_episodes"], data),
        "query_target_not_in_model_input": "query_targets" not in model_input_keys and "query_target" not in model_input_keys,
        "task_ids_not_in_forward_or_tensors": BANNED_FORWARD_NAMES.isdisjoint(model_input_keys)
        and BANNED_FORWARD_NAMES.isdisjoint(forward_params),
        "query_input_mask_present": "query_input_mask" in model_input_keys,
        "color_head_has_10_classes": model.config.output_color_count == ARC_COLOR_COUNT,
        "state_dict_free_of_puzzle_or_task_lookup": not any(
            any(term in key for term in BANNED_STATE_TERMS) for key in state_keys
        ),
        "unseen_not_in_optimizer_pool": not any(ep.task_id in unseen_ids for ep in selection["train_episodes"]),
    }
    checks["failed_checks"] = sorted(key for key, value in checks.items() if value is False)
    checks["all_passed"] = not checks["failed_checks"]
    return checks


def build_task_manifest(args: argparse.Namespace, data: dict[str, Any], selection: dict[str, Any], checks: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": V1_1_ARCHITECTURE_DISPLAY,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "prior_v1_task_manifest_path": selection["prior_manifest_path"],
        "task_selection_status": selection["selection_status"],
        "train_50_task_ids": selection["train_task_ids"],
        "unseen_50_task_ids": selection["unseen_task_ids"],
        "train_unseen_overlap": sorted(set(selection["train_task_ids"]).intersection(selection["unseen_task_ids"])),
        "episode_counts_per_train_task": selection["train_episode_counts"],
        "episode_counts_per_unseen_task": selection["unseen_episode_counts"],
        "total_train_episodes": len(selection["train_episodes"]),
        "total_unseen_episodes": len(selection["unseen_episodes"]),
        "raw_challenge_json_filename": data["sources"]["training_challenges"],
        "training_solutions_discovered_not_used": data["sources"]["training_solutions"],
        "integrity_checks": checks,
    }


def save_checkpoint(
    path: Path,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    config: VRMConfig,
    step: int,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "extra": report,
        },
        path,
    )


def checkpoint_path_for_step(step: int) -> Path:
    return CHECKPOINTS_DIR / f"vrm_v1_1_step{step:04d}.pt"


def load_prior_v1_final_train_hamming() -> float | None:
    metrics_path = V1_ARTIFACTS_DIR / "vrm_v1_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    final_train = payload.get("final_train") or {}
    value = final_train.get("mean_nonpad_hamming_error")
    return float(value) if isinstance(value, (int, float)) else None


def choose_stability_verdict(metrics: list[dict[str, Any]], finite_guard_triggered: bool) -> str:
    if finite_guard_triggered:
        return "RUN_FAILED"
    train_rows = [row for row in metrics if row["split"] == "train_50"]
    if not train_rows:
        return "RUN_FAILED"
    by_step = {row["step"]: row for row in train_rows}
    final = train_rows[-1]
    if 250 in by_step:
        step250 = by_step[250]
        if (
            final["mean_target_rectangle_hamming_error"] > step250["mean_target_rectangle_hamming_error"] * 1.25
            and final["target_rectangle_color_accuracy"] + 0.05 < step250["target_rectangle_color_accuracy"]
        ):
            return "COLLAPSED"
    if final["target_rectangle_color_accuracy"] >= max(row["target_rectangle_color_accuracy"] for row in train_rows) - 0.05:
        return "STABLE"
    return "PARTIAL"


def choose_capacity_verdict(metrics: list[dict[str, Any]]) -> str:
    train_rows = [row for row in metrics if row["split"] == "train_50"]
    if not train_rows:
        return "INCONCLUSIVE"
    final = train_rows[-1]
    prior_hamming = load_prior_v1_final_train_hamming()
    if prior_hamming is None:
        return "INCONCLUSIVE"
    if final["mean_target_rectangle_hamming_error"] < prior_hamming * 0.95:
        return "IMPROVED"
    if final["mean_target_rectangle_hamming_error"] > prior_hamming * 1.05:
        return "WORSE"
    return "UNCHANGED"


def choose_unseen_verdict(metrics: list[dict[str, Any]]) -> str:
    unseen_rows = [row for row in metrics if row["split"] == "unseen_50"]
    if not unseen_rows:
        return "EVALUATION_FAILED"
    initial = unseen_rows[0]
    final = unseen_rows[-1]
    if final["exact_decoded_grid_accuracy"] > 0:
        return "SIGNAL_PRESENT"
    color_gain = final["target_rectangle_color_accuracy"] > initial["target_rectangle_color_accuracy"] + 0.05
    hamming_gain = final["mean_target_rectangle_hamming_error"] < initial["mean_target_rectangle_hamming_error"] * 0.90
    shape_gain = final["output_shape_match_rate"] > initial["output_shape_match_rate"] + 0.05
    return "SIGNAL_PRESENT" if color_gain or hamming_gain or shape_gain else "SIGNAL_ABSENT"


def render_summary(
    report: dict[str, Any],
    metrics: list[dict[str, Any]],
    train_tasks: list[dict[str, Any]],
    unseen_tasks: list[dict[str, Any]],
) -> str:
    headers = [
        "step",
        "split",
        "total_loss",
        "color_loss",
        "shape_height_accuracy",
        "shape_width_accuracy",
        "output_shape_match_rate",
        "target_rectangle_color_accuracy",
        "exact_decoded_grid_accuracy",
        "mean_target_rectangle_hamming_error",
        "median_target_rectangle_hamming_error",
    ]
    lines = [
        "# VRM V1.1 - Masked Query Spatial Evidence Rewrite summary",
        "",
        "This is an experimental ARC-2 capacity probe, not a claim that VRM solves ARC.",
        "",
        f"- command: `{report['command']}`",
        f"- data_root: `{report['data_root']}`",
        f"- device: `{report['device']}`",
        f"- task_selection_status: `{report['task_selection_status']}`",
        f"- wall_clock_seconds: {report['wall_clock_seconds']:.3f}",
        f"- average_seconds_per_optimizer_step: {report['average_seconds_per_optimizer_step']:.6f}",
        f"- train_episodes: {report['train_episode_count']}",
        f"- unseen_episodes: {report['unseen_episode_count']}",
        f"- best_train_hamming_checkpoint_step: {report['best_train_hamming_step']}",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in metrics:
        rendered = []
        for header in headers:
            value = row[header]
            rendered.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("|" + "|".join(rendered) + "|")
    lines.extend(["", "## Best train tasks", "", "|task_id|min_hamming|exact_count|shape_rate|", "|---|---:|---:|---:|"])
    for row in sorted(train_tasks, key=lambda item: (item["minimum_target_rectangle_hamming_error"], item["task_id"]))[:5]:
        lines.append(
            f"|{row['task_id']}|{row['minimum_target_rectangle_hamming_error']}|"
            f"{row['exact_decoded_grid_count']}|{row['shape_match_rate']:.3f}|"
        )
    lines.extend(["", "## Best unseen tasks", "", "|task_id|min_hamming|exact_count|shape_rate|", "|---|---:|---:|---:|"])
    for row in sorted(unseen_tasks, key=lambda item: (item["minimum_target_rectangle_hamming_error"], item["task_id"]))[:5]:
        lines.append(
            f"|{row['task_id']}|{row['minimum_target_rectangle_hamming_error']}|"
            f"{row['exact_decoded_grid_count']}|{row['shape_match_rate']:.3f}|"
        )
    lines.extend(
        [
            "",
            f"V1_1_STABILITY_VERDICT: {report['stability_verdict']}",
            f"V1_1_CAPACITY_VERDICT: {report['capacity_verdict']}",
            f"V1_1_UNSEEN_VERDICT: {report['unseen_verdict']}",
            "",
        ]
    )
    return "\n".join(lines)


def create_source_snapshot(zip_path: Path = V1_1_SOURCE_ZIP, manifest_path: Path = V1_1_SOURCE_MANIFEST, pytest_result: str | None = None) -> dict[str, Any]:
    root = Path("/mnt/d/VRM")
    report_sources = [
        V1_1_ARTIFACTS_DIR / "vrm_v1_1_config.json",
        V1_1_ARTIFACTS_DIR / "vrm_v1_1_task_manifest.json",
        V1_1_ARTIFACTS_DIR / "vrm_v1_1_initialization_report.json",
        V1_1_ARTIFACTS_DIR / "vrm_v1_1_metrics.json",
        V1_1_ARTIFACTS_DIR / "vrm_v1_1_summary.md",
    ]
    excluded_parts = {
        "arc-prize-2026-arc-agi-2",
        "checkpoints",
        "artifacts",
        ".git",
        ".claude",
        ".pytest_cache",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
    }
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".zip"}
    entries: list[tuple[Path, str]] = []
    for rel in ("README.md", "requirements.txt"):
        path = root / rel
        if path.exists():
            entries.append((path, rel))
    for folder in ("vrm", "tests"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if any(part in excluded_parts for part in rel.parts):
                continue
            if path.suffix.lower() in forbidden_suffixes:
                continue
            entries.append((path, rel.as_posix()))
    for source in report_sources:
        if source.exists() and source.is_file() and source.stat().st_size <= 1_000_000:
            entries.append((source, f"reports/{source.name}"))
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for source, arcname in entries:
            info = zipfile.ZipInfo(arcname)
            info.date_time = (2026, 6, 28, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, source.read_bytes())
    with zipfile.ZipFile(zip_path) as handle:
        names = handle.namelist()
        bad_crc = handle.testzip()
    if bad_crc is not None:
        raise RuntimeError(f"source ZIP CRC check failed at {bad_crc}")
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    size = zip_path.stat().st_size
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest_lines = [
        f"archive path: {zip_path}",
        f"timestamp: {timestamp}",
        f"ZIP size: {size} bytes",
        f"SHA-256: {sha256}",
        f"pytest result: {pytest_result or 'not run by snapshot helper'}",
        f"active architecture name: {V1_1_ARCHITECTURE_DISPLAY}",
        "excluded major directories: arc-prize-2026-arc-agi-2/, checkpoints/, full artifacts/, .git/, .claude/, virtual environments, cache files, other ZIPs",
        "file list:",
    ]
    manifest_lines.extend(f"- {name}" for name in names)
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {
        "archive_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "size_bytes": size,
        "sha256": sha256,
        "file_list": names,
    }


def create_source_zip(zip_path: Path = V1_1_SOURCE_ZIP) -> Path:
    return Path(create_source_snapshot(zip_path=zip_path)["archive_path"])


def run_v1_1_capacity50(args: argparse.Namespace) -> dict[str, Any]:
    steps = args.steps if args.steps is not None else V1_1_DEFAULT_STEPS
    if steps < 1:
        raise ValueError("--steps must be positive")
    run_kind = "full" if steps == V1_1_DEFAULT_STEPS else "dry"
    determinism = configure_determinism(args.seed)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    device = resolve_device(args.device)
    data = build_dataset(args.data_root, seed=args.seed, manifest_path=None)
    selection = select_v1_1_capacity50_episode_sets(data, args.seed)
    config = make_config(args)
    model, initialization_report = initialize_fresh_v1_1_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    checks = integrity_checks(data, selection, model, args.data_root)
    if not checks["all_passed"]:
        raise RuntimeError(f"integrity checks failed: {checks['failed_checks']}")

    command = " ".join(["python", "-m", "vrm.train", *sys.argv[1:]])
    eval_steps = sorted({0, steps, *[step for step in V1_1_EVAL_STEPS if step <= steps]})
    prediction_steps = sorted({step for step in V1_1_PREDICTION_STEPS if step <= steps} | {0, steps})
    checkpoint_steps = sorted({0, steps, *[step for step in V1_1_EVAL_STEPS if step <= steps]})
    V1_1_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = V1_1_ARTIFACTS_DIR / ("vrm_v1_1_training_log.jsonl" if run_kind == "full" else "vrm_v1_1_dry_training_log.jsonl")
    if log_path.exists():
        log_path.unlink()

    raw_hash_before = hash_file(Path(data["sources"]["training_challenges"]))
    config_payload = {
        "architecture": V1_1_ARCHITECTURE_DISPLAY,
        "command": command,
        "mode": "v1_1_capacity50",
        "run_kind": run_kind,
        "steps": steps,
        "eval_steps": eval_steps,
        "prediction_steps": prediction_steps,
        "checkpoint_steps": checkpoint_steps,
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "shape_loss_weight": args.shape_loss_weight,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "optimizer": optimizer_description(optimizer),
        "seed": args.seed,
        "determinism": determinism,
        "torch_threads": torch.get_num_threads(),
        "model_config": asdict(config),
        "parameter_count": parameter_count(model),
        "data_root": str(args.data_root),
    }
    save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_config.json", config_payload)
    save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_initialization_report.json", initialization_report)
    manifest = build_task_manifest(args, data, selection, checks)
    save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_task_manifest.json", manifest)

    train_episodes = selection["train_episodes"]
    unseen_episodes = selection["unseen_episodes"]
    unseen_ids = set(selection["unseen_task_ids"])
    sampler = DeterministicCyclingSampler(len(train_episodes), args.seed + 90_000)
    aggregate_metrics: list[dict[str, Any]] = []
    batch_history: list[dict[str, Any]] = []
    training_stability: list[dict[str, Any]] = []
    coverage = [0 for _ in train_episodes]
    final_train_episode_metrics: list[dict[str, Any]] = []
    final_unseen_episode_metrics: list[dict[str, Any]] = []
    best_train_hamming = math.inf
    best_train_hamming_step: int | None = None
    finite_guard_triggered = False
    run_start = time.perf_counter()
    train_start = time.perf_counter()

    def evaluate_both(step: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nonlocal best_train_hamming, best_train_hamming_step
        train_agg, train_per_episode, train_predictions = evaluate_split(
            model=model,
            episodes=train_episodes,
            split="train_50",
            device=device,
            shape_loss_weight=args.shape_loss_weight,
            step=step,
        )
        unseen_agg, unseen_per_episode, unseen_predictions = evaluate_split(
            model=model,
            episodes=unseen_episodes,
            split="unseen_50",
            device=device,
            shape_loss_weight=args.shape_loss_weight,
            step=step,
        )
        aggregate_metrics.extend([train_agg, unseen_agg])
        event = {
            "event": "eval",
            "mode": "v1_1_capacity50",
            "step": step,
            "train_total_loss": train_agg["total_loss"],
            "train_color_loss": train_agg["color_loss"],
            "train_shape_match": train_agg["output_shape_match_rate"],
            "train_target_color_accuracy": train_agg["target_rectangle_color_accuracy"],
            "train_mean_hamming": train_agg["mean_target_rectangle_hamming_error"],
            "unseen_total_loss": unseen_agg["total_loss"],
            "unseen_color_loss": unseen_agg["color_loss"],
            "unseen_shape_match": unseen_agg["output_shape_match_rate"],
            "unseen_target_color_accuracy": unseen_agg["target_rectangle_color_accuracy"],
            "unseen_mean_hamming": unseen_agg["mean_target_rectangle_hamming_error"],
        }
        append_jsonl(log_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        if step in prediction_steps:
            save_text(
                V1_1_ARTIFACTS_DIR / f"vrm_v1_1_predictions_step_{step}.txt",
                render_predictions(
                    step=step,
                    train_predictions=train_predictions,
                    unseen_predictions=unseen_predictions,
                    final_step=run_kind == "full" and step == steps,
                ),
            )
        if run_kind == "full" and step in checkpoint_steps:
            checkpoint_report = {"step": step, "train_aggregate": train_agg, "unseen_aggregate": unseen_agg}
            save_checkpoint(checkpoint_path_for_step(step), model, optimizer, config, step, checkpoint_report)
        if train_agg["mean_target_rectangle_hamming_error"] < best_train_hamming:
            best_train_hamming = train_agg["mean_target_rectangle_hamming_error"]
            best_train_hamming_step = step
            if run_kind == "full":
                save_checkpoint(
                    CHECKPOINTS_DIR / "vrm_v1_1_best_train_hamming.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    {"step": step, "train_aggregate": train_agg, "unseen_aggregate_recorded_but_not_used": unseen_agg},
                )
        return train_per_episode, unseen_per_episode

    try:
        model.eval()
        evaluate_both(0)
        model.train()
        for step in range(1, steps + 1):
            lr = compute_lr(step, steps, args.learning_rate, args.warmup_steps, args.min_learning_rate)
            set_optimizer_lr(optimizer, lr)
            indices = sampler.next_indices(args.batch_size)
            episodes = [train_episodes[idx] for idx in indices]
            unseen_hits = sorted({episode.task_id for episode in episodes if episode.task_id in unseen_ids})
            if unseen_hits:
                raise RuntimeError(f"unseen tasks entered optimizer batch: {unseen_hits}")
            for idx in indices:
                coverage[idx] += 1
            batch = move_tensor_batch(collate_episodes(episodes), device)
            optimizer.zero_grad(set_to_none=True)
            loss, outputs, components = forward_loss(model, batch, args.shape_loss_weight)
            ensure_finite_components(components)
            loss.backward()
            ensure_finite_gradients(model)
            grad_norm_before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm))
            grad_norm_after = compute_grad_norm(model)
            optimizer.step()
            ensure_finite_parameters(model)
            if step % 10 == 0 or step in eval_steps:
                row = instrumentation_from_outputs(
                    step=step,
                    loss=loss,
                    components=components,
                    outputs=outputs,
                    learning_rate=lr,
                    grad_norm_before_clip=grad_norm_before,
                    grad_norm_after_clip=grad_norm_after,
                    model=model,
                )
                training_stability.append(row)
                append_jsonl(log_path, row)
            batch_history.append(
                {
                    "step": step,
                    "episode_indices": indices,
                    "task_ids": [episode.task_id for episode in episodes],
                    "query_pair_indices": [episode.query_pair_index for episode in episodes],
                    "learning_rate": lr,
                    "total_loss": float(loss.detach().cpu().item()),
                    "color_loss": float(components["color_loss"].detach().cpu().item()),
                    "height_loss": float(components["height_loss"].detach().cpu().item()),
                    "width_loss": float(components["width_loss"].detach().cpu().item()),
                    "grad_norm_before_clip": grad_norm_before,
                    "grad_norm_after_clip": grad_norm_after,
                }
            )
            if step in eval_steps and step != 0:
                model.eval()
                train_metrics, unseen_metrics = evaluate_both(step)
                if step == steps:
                    final_train_episode_metrics = train_metrics
                    final_unseen_episode_metrics = unseen_metrics
                model.train()
    except Exception as exc:
        finite_guard_triggered = True
        try:
            save_diagnostic_failure(
                step=locals().get("step", -1),
                reason=str(exc),
                model=model,
                optimizer=optimizer,
                batch=locals().get("batch"),
                outputs=locals().get("outputs"),
            )
        finally:
            raise

    training_seconds = time.perf_counter() - train_start
    wall_clock_seconds = time.perf_counter() - run_start
    raw_hash_after = hash_file(Path(data["sources"]["training_challenges"]))
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("raw ARC training challenge JSON changed during run")
    if not final_train_episode_metrics or not final_unseen_episode_metrics:
        model.eval()
        final_train_episode_metrics, final_unseen_episode_metrics = evaluate_both(steps)
    train_per_task = aggregate_per_task(final_train_episode_metrics)
    unseen_per_task = aggregate_per_task(final_unseen_episode_metrics)
    final_train_aggregate = [row for row in aggregate_metrics if row["split"] == "train_50"][-1]
    final_unseen_aggregate = [row for row in aggregate_metrics if row["split"] == "unseen_50"][-1]
    stability_verdict = choose_stability_verdict(aggregate_metrics, finite_guard_triggered)
    capacity_verdict = choose_capacity_verdict(aggregate_metrics)
    unseen_verdict = choose_unseen_verdict(aggregate_metrics)
    optimizer_seen_task_ids = sorted({task_id for row in batch_history for task_id in row["task_ids"]})
    unseen_used = sorted(set(optimizer_seen_task_ids).intersection(unseen_ids))
    if unseen_used:
        raise RuntimeError(f"unseen tasks used for optimization: {unseen_used}")

    report = {
        "architecture": V1_1_ARCHITECTURE_DISPLAY,
        "mode": "v1_1_capacity50",
        "run_kind": run_kind,
        "command": command,
        "data_root": str(args.data_root),
        "task_selection_status": selection["selection_status"],
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_clock_seconds": wall_clock_seconds,
        "training_seconds": training_seconds,
        "average_seconds_per_optimizer_step": training_seconds / steps,
        "steps": steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "shape_loss_weight": args.shape_loss_weight,
        "optimizer": optimizer_description(optimizer),
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "train_task_ids": selection["train_task_ids"],
        "unseen_task_ids": selection["unseen_task_ids"],
        "train_episode_count": len(train_episodes),
        "unseen_episode_count": len(unseen_episodes),
        "aggregate_metrics": aggregate_metrics,
        "final_train_aggregate": final_train_aggregate,
        "final_unseen_aggregate": final_unseen_aggregate,
        "best_train_hamming": best_train_hamming,
        "best_train_hamming_step": best_train_hamming_step,
        "stability_verdict": stability_verdict,
        "capacity_verdict": capacity_verdict,
        "unseen_verdict": unseen_verdict,
        "finite_guard_triggered": finite_guard_triggered,
        "artifact_dir": str(V1_1_ARTIFACTS_DIR),
        "checkpoint_paths": [str(checkpoint_path_for_step(step)) for step in checkpoint_steps if run_kind == "full"],
        "best_train_hamming_checkpoint": str(CHECKPOINTS_DIR / "vrm_v1_1_best_train_hamming.pt") if run_kind == "full" else None,
        "integrity_checks": checks,
        "raw_arc_files_unchanged": raw_hash_before == raw_hash_after,
        "raw_challenge_hash_before": raw_hash_before,
        "raw_challenge_hash_after": raw_hash_after,
        "training_stability": training_stability,
        "optimizer_seen_task_ids": optimizer_seen_task_ids,
        "unseen_task_ids_passed_to_optimizer": unseen_used,
        "coverage_counts_by_train_episode_index": coverage,
        "config": asdict(config),
    }
    config_payload.update(
        {
            "wall_clock_seconds": wall_clock_seconds,
            "training_seconds": training_seconds,
            "average_seconds_per_optimizer_step": training_seconds / steps,
            "train_episode_count": len(train_episodes),
            "unseen_episode_count": len(unseen_episodes),
            "best_train_hamming_step": best_train_hamming_step,
            "stability_verdict": stability_verdict,
            "capacity_verdict": capacity_verdict,
            "unseen_verdict": unseen_verdict,
        }
    )
    manifest["raw_challenge_hash_before"] = raw_hash_before
    manifest["raw_challenge_hash_after"] = raw_hash_after
    manifest["raw_arc_files_unchanged"] = raw_hash_before == raw_hash_after
    save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_config.json", config_payload)
    save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_task_manifest.json", manifest)

    if run_kind == "dry":
        dry_payload = {
            "report": report,
            "final_train": final_train_aggregate,
            "final_unseen": final_unseen_aggregate,
            "dry_run_passed": True,
        }
        save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_dry_run_report.json", dry_payload)
    else:
        save_json(
            V1_1_ARTIFACTS_DIR / "vrm_v1_1_metrics.json",
            {
                "mode": "v1_1_capacity50",
                "aggregate_metrics": aggregate_metrics,
                "final_train": final_train_aggregate,
                "final_unseen": final_unseen_aggregate,
                "best_train_hamming_step": best_train_hamming_step,
                "stability_verdict": stability_verdict,
                "capacity_verdict": capacity_verdict,
                "unseen_verdict": unseen_verdict,
                "finite_guard_triggered": finite_guard_triggered,
                "training_stability": training_stability,
            },
        )
        save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_per_task_train.json", {"tasks": train_per_task})
        save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_per_task_unseen.json", {"tasks": unseen_per_task})
        save_json(
            V1_1_ARTIFACTS_DIR / "vrm_v1_1_per_episode_train_step1500.json",
            {"step": steps, "episodes": [compact_episode_record(row) for row in final_train_episode_metrics]},
        )
        save_json(
            V1_1_ARTIFACTS_DIR / "vrm_v1_1_per_episode_unseen_step1500.json",
            {"step": steps, "episodes": [compact_episode_record(row) for row in final_unseen_episode_metrics]},
        )
        save_text(
            V1_1_ARTIFACTS_DIR / "vrm_v1_1_summary.md",
            render_summary(report, aggregate_metrics, train_per_task, unseen_per_task),
        )
        snapshot = create_source_snapshot(pytest_result=args.pytest_result or "pytest result not supplied to training command")
        report["source_snapshot"] = snapshot
        save_json(V1_1_ARTIFACTS_DIR / "vrm_v1_1_metrics.json", {
            "mode": "v1_1_capacity50",
            "aggregate_metrics": aggregate_metrics,
            "final_train": final_train_aggregate,
            "final_unseen": final_unseen_aggregate,
            "best_train_hamming_step": best_train_hamming_step,
            "stability_verdict": stability_verdict,
            "capacity_verdict": capacity_verdict,
            "unseen_verdict": unseen_verdict,
            "finite_guard_triggered": finite_guard_triggered,
            "training_stability": training_stability,
            "source_snapshot": snapshot,
        })

    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def stable_seed(*items: Any, seed: int) -> int:
    payload = "|".join(str(item) for item in (*items, seed)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def grid_with_color_permutation(grid: Grid, permutation: list[int]) -> Grid:
    return Grid(tuple(tuple(permutation[cell] for cell in row) for row in grid.cells))


def deterministic_derangement(length: int, *, task_id: str, query_pair_index: int, seed: int) -> list[int]:
    if length < 2:
        raise ValueError("cannot derange fewer than two support pairs")
    if length == 2:
        return [1, 0]
    rng = random.Random(stable_seed("derange", task_id, query_pair_index, seed=seed))
    values = list(range(length))
    for _ in range(1000):
        rng.shuffle(values)
        if all(values[idx] != idx for idx in range(length)):
            return values[:]
    return values[1:] + values[:1]


def deterministic_order_permutation(length: int, *, task_id: str, query_pair_index: int, seed: int) -> list[int]:
    values = list(range(length))
    if length < 2:
        return values
    rng = random.Random(stable_seed("order", task_id, query_pair_index, seed=seed))
    rng.shuffle(values)
    if values == list(range(length)):
        values = values[1:] + values[:1]
    return values


def deterministic_color_permutation(*, task_id: str, query_pair_index: int, seed: int) -> list[int]:
    values = list(range(ARC_COLOR_COUNT))
    rng = random.Random(stable_seed("color", task_id, query_pair_index, seed=seed))
    rng.shuffle(values)
    if values == list(range(ARC_COLOR_COUNT)):
        values = values[1:] + values[:1]
    return values


def episode_for_condition(
    episode: ArcEpisode,
    condition: str,
    seed: int,
) -> tuple[ArcEpisode | None, dict[str, Any]]:
    supports = list(episode.supports)
    metadata: dict[str, Any] = {"condition": condition, "applicable": True}
    if condition == "NORMAL" or condition == "SUPPORT_EVIDENCE_PATH_ZEROED":
        return episode, metadata
    if condition == "OUTPUT_DERANGED_WITHIN_TASK":
        if len(supports) < 2:
            metadata["applicable"] = False
            metadata["output_derangement_not_applicable"] = True
            return None, metadata
        perm = deterministic_derangement(len(supports), task_id=episode.task_id, query_pair_index=episode.query_pair_index, seed=seed)
        metadata["output_derangement_permutation"] = perm
        new_supports = tuple(
            ArcPair(input=supports[idx].input, output=supports[perm[idx]].output, pair_index=supports[idx].pair_index)
            for idx in range(len(supports))
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
            metadata,
        )
    if condition == "PAIR_ORDER_SHUFFLED":
        perm = deterministic_order_permutation(len(supports), task_id=episode.task_id, query_pair_index=episode.query_pair_index, seed=seed)
        metadata["pair_order_permutation"] = perm
        new_supports = tuple(
            ArcPair(input=supports[old_idx].input, output=supports[old_idx].output, pair_index=new_idx)
            for new_idx, old_idx in enumerate(perm)
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
            metadata,
        )
    if condition == "OUTPUT_COLOR_PERMUTED":
        permutation = deterministic_color_permutation(task_id=episode.task_id, query_pair_index=episode.query_pair_index, seed=seed)
        metadata["color_permutation"] = permutation
        new_supports = tuple(
            ArcPair(
                input=support.input,
                output=grid_with_color_permutation(support.output, permutation),
                pair_index=support.pair_index,
            )
            for support in supports
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
            metadata,
        )
    raise ValueError(f"unknown ablation condition {condition}")


def identity_copy_baseline_prediction(query_input: Grid) -> dict[str, Any]:
    tokens = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.long)
    for row_idx, row in enumerate(query_input.cells):
        for col_idx, color in enumerate(row):
            tokens[row_idx, col_idx] = color
    return {
        "predicted_height": query_input.height,
        "predicted_width": query_input.width,
        "prediction_colors": tokens.reshape(MAX_GRID_TOKENS),
    }


def support_shape_copy_baseline_prediction(query_input: Grid, supports: tuple[ArcPair, ...]) -> dict[str, Any]:
    shape_counts: dict[tuple[int, int], int] = {}
    for support in supports:
        shape = support.output.shape
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    best_shape = sorted(shape_counts, key=lambda shape: (-shape_counts[shape], shape[0], shape[1]))[0]
    tokens = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.long)
    for row_idx, row in enumerate(query_input.cells):
        for col_idx, color in enumerate(row):
            if row_idx < MAX_GRID_HEIGHT and col_idx < MAX_GRID_WIDTH:
                tokens[row_idx, col_idx] = color
    return {
        "predicted_height": best_shape[0],
        "predicted_width": best_shape[1],
        "prediction_colors": tokens.reshape(MAX_GRID_TOKENS),
    }


def transformation_cell_mask(episode: ArcEpisode) -> torch.Tensor:
    query = torch.full((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), PAD_ID, dtype=torch.long)
    target = torch.full((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), PAD_ID, dtype=torch.long)
    query_mask = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.bool)
    target_mask = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.bool)
    for row_idx, row in enumerate(episode.query.input.cells):
        for col_idx, color in enumerate(row):
            query[row_idx, col_idx] = color
            query_mask[row_idx, col_idx] = True
    for row_idx, row in enumerate(episode.query.output.cells):
        for col_idx, color in enumerate(row):
            target[row_idx, col_idx] = color
            target_mask[row_idx, col_idx] = True
    outside_query = target_mask & ~query_mask
    changed_overlap = target_mask & query_mask & (target != query)
    return (outside_query | changed_overlap).reshape(MAX_GRID_TOKENS)


def prediction_metrics_for_episode(
    *,
    episode: ArcEpisode,
    prediction_colors: torch.Tensor,
    predicted_height: int,
    predicted_width: int,
) -> dict[str, Any]:
    prediction = prediction_colors.detach().cpu().to(torch.long)
    target = torch.full((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), PAD_ID, dtype=torch.long)
    target_mask_grid = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.bool)
    for row_idx, row in enumerate(episode.query.output.cells):
        for col_idx, color in enumerate(row):
            target[row_idx, col_idx] = color
            target_mask_grid[row_idx, col_idx] = True
    target_flat = target.reshape(MAX_GRID_TOKENS)
    target_mask = target_mask_grid.reshape(MAX_GRID_TOKENS)
    trans_mask = transformation_cell_mask(episode)
    copy_mask = target_mask & ~trans_mask
    matches = prediction == target_flat
    target_correct = int(matches[target_mask].sum().item())
    target_total = int(target_mask.sum().item())
    trans_total = int(trans_mask.sum().item())
    trans_correct = int(matches[trans_mask].sum().item()) if trans_total else 0
    copy_total = int(copy_mask.sum().item())
    copy_correct = int(matches[copy_mask].sum().item()) if copy_total else 0
    target_h = episode.query.output.height
    target_w = episode.query.output.width
    shape_match = predicted_height == target_h and predicted_width == target_w
    exact = bool(shape_match and target_correct == target_total)
    return {
        "target_shape": [target_h, target_w],
        "predicted_shape": [predicted_height, predicted_width],
        "shape_match": bool(shape_match),
        "target_rectangle_color_correct": target_correct,
        "target_rectangle_color_total": target_total,
        "target_rectangle_color_accuracy": target_correct / max(target_total, 1),
        "target_rectangle_hamming_error": target_total - target_correct,
        "exact_decoded_grid_correct": exact,
        "transformation_cell_count": trans_total,
        "zero_transformation_cells": trans_total == 0,
        "transformation_cell_correct": trans_correct,
        "transformation_cell_accuracy": trans_correct / trans_total if trans_total else None,
        "transformation_cell_hamming_error": trans_total - trans_correct if trans_total else None,
        "copy_preserved_cell_count": copy_total,
        "copy_preserved_cell_correct": copy_correct,
        "copy_preserved_cell_accuracy": copy_correct / copy_total if copy_total else None,
    }


def aggregate_ablation_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [record for record in records if record.get("applicable", True)]
    count = len(applicable)
    hamming = [record["target_rectangle_hamming_error"] for record in applicable]
    trans_records = [record for record in applicable if record["transformation_cell_count"] > 0]
    trans_total = sum(record["transformation_cell_count"] for record in trans_records)
    trans_correct = sum(record["transformation_cell_correct"] for record in trans_records)
    copy_total = sum(record["copy_preserved_cell_count"] for record in applicable)
    copy_correct = sum(record["copy_preserved_cell_correct"] for record in applicable)
    return {
        "episode_count": count,
        "inapplicable_episode_count": len(records) - count,
        "zero_transformation_cell_episode_count": sum(1 for record in applicable if record["transformation_cell_count"] == 0),
        "shape_match_rate": sum(1 for record in applicable if record["shape_match"]) / max(count, 1),
        "target_rectangle_color_accuracy": sum(record["target_rectangle_color_correct"] for record in applicable)
        / max(1, sum(record["target_rectangle_color_total"] for record in applicable)),
        "exact_decoded_grid_accuracy": sum(1 for record in applicable if record["exact_decoded_grid_correct"]) / max(count, 1),
        "mean_target_rectangle_hamming_error": statistics.fmean(hamming) if hamming else None,
        "median_target_rectangle_hamming_error": statistics.median(hamming) if hamming else None,
        "mean_predicted_height": statistics.fmean(record["predicted_shape"][0] for record in applicable) if applicable else None,
        "mean_predicted_width": statistics.fmean(record["predicted_shape"][1] for record in applicable) if applicable else None,
        "transformation_cell_accuracy": trans_correct / trans_total if trans_total else None,
        "mean_transformation_cell_hamming_error": statistics.fmean(
            record["transformation_cell_hamming_error"] for record in trans_records
        )
        if trans_records
        else None,
        "transformation_cell_count": trans_total,
        "copy_preserved_cell_accuracy": copy_correct / copy_total if copy_total else None,
        "copy_preserved_cell_count": copy_total,
    }


def load_v1_1_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[VRM, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = VRMConfig(**checkpoint["config"])
    model = VRM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_model_condition(
    *,
    model: VRM,
    episodes: list[ArcEpisode],
    split: str,
    condition: str,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    disable_support = condition == "SUPPORT_EVIDENCE_PATH_ZEROED"
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            conditioned_episode, metadata = episode_for_condition(episode, condition, seed)
            if conditioned_episode is None:
                records.append(
                    {
                        "split": split,
                        "condition": condition,
                        "episode_index": episode_index,
                        "task_id": episode.task_id,
                        "query_pair_index": episode.query_pair_index,
                        "applicable": False,
                        **metadata,
                    }
                )
                continue
            batch = move_tensor_batch(collate_episodes([conditioned_episode]), device)
            outputs = model(**model_inputs_from_batch(batch), disable_support_evidence=disable_support)
            if not all(torch.isfinite(value).all() for value in outputs.values() if torch.is_tensor(value)):
                raise RuntimeError(f"non-finite output in condition {condition}")
            prediction_colors = outputs["color_log_probs"].argmax(dim=-1)[0].detach().cpu()
            predicted_height = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
            predicted_width = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
            metrics = prediction_metrics_for_episode(
                episode=episode,
                prediction_colors=prediction_colors,
                predicted_height=predicted_height,
                predicted_width=predicted_width,
            )
            record = {
                "split": split,
                "condition": condition,
                "episode_index": episode_index,
                "task_id": episode.task_id,
                "query_pair_index": episode.query_pair_index,
                "support_count": len(episode.supports),
                "applicable": True,
                **metadata,
                **metrics,
            }
            records.append(record)
            predictions.append(
                {
                    "split": split,
                    "condition": condition,
                    "episode_index": episode_index,
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    "prediction_colors": prediction_colors,
                    "predicted_height": predicted_height,
                    "predicted_width": predicted_width,
                    "metrics": record,
                }
            )
    aggregate = aggregate_ablation_records(records)
    aggregate.update({"split": split, "condition": condition})
    return aggregate, records, predictions


def evaluate_baseline(
    *,
    episodes: list[ArcEpisode],
    split: str,
    baseline: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        if baseline == "IDENTITY_COPY":
            prediction = identity_copy_baseline_prediction(episode.query.input)
        elif baseline == "SUPPORT_SHAPE_COPY":
            prediction = support_shape_copy_baseline_prediction(episode.query.input, episode.supports)
        else:
            raise ValueError(f"unknown baseline {baseline}")
        metrics = prediction_metrics_for_episode(
            episode=episode,
            prediction_colors=prediction["prediction_colors"],
            predicted_height=prediction["predicted_height"],
            predicted_width=prediction["predicted_width"],
        )
        record = {
            "split": split,
            "baseline": baseline,
            "episode_index": episode_index,
            "task_id": episode.task_id,
            "query_pair_index": episode.query_pair_index,
            "support_count": len(episode.supports),
            "applicable": True,
            **metrics,
        }
        records.append(record)
        predictions.append(
            {
                "split": split,
                "baseline": baseline,
                "episode_index": episode_index,
                "task_id": episode.task_id,
                "query_pair_index": episode.query_pair_index,
                "prediction_colors": prediction["prediction_colors"],
                "predicted_height": prediction["predicted_height"],
                "predicted_width": prediction["predicted_width"],
                "metrics": record,
            }
        )
    aggregate = aggregate_ablation_records(records)
    aggregate.update({"split": split, "baseline": baseline})
    return aggregate, records, predictions


def paired_bootstrap_ci(values: list[float], *, seed: int, resamples: int = 2000) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "count": 0, "resamples": resamples}
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(count)] for _ in range(count)) / count)
    means.sort()
    low_index = int(0.025 * (resamples - 1))
    high_index = int(0.975 * (resamples - 1))
    return {
        "mean": statistics.fmean(values),
        "ci95_low": means[low_index],
        "ci95_high": means[high_index],
        "count": count,
        "resamples": resamples,
    }


def paired_condition_deltas(
    normal_records: list[dict[str, Any]],
    condition_records: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normal_by_key = {
        (record["task_id"], record["query_pair_index"]): record
        for record in normal_records
        if record.get("applicable", True)
    }
    condition_by_key = {
        (record["task_id"], record["query_pair_index"]): record
        for record in condition_records
        if record.get("applicable", True)
    }
    shared_keys = sorted(set(normal_by_key).intersection(condition_by_key))
    metric_specs = {
        "delta_shape_match": lambda normal, cond: float(normal["shape_match"]) - float(cond["shape_match"]),
        "delta_target_rectangle_color_accuracy": lambda normal, cond: normal["target_rectangle_color_accuracy"]
        - cond["target_rectangle_color_accuracy"],
        "delta_transformation_cell_accuracy": lambda normal, cond: None
        if normal["transformation_cell_count"] == 0 or cond["transformation_cell_count"] == 0
        else normal["transformation_cell_accuracy"] - cond["transformation_cell_accuracy"],
        "delta_target_rectangle_hamming_error": lambda normal, cond: normal["target_rectangle_hamming_error"]
        - cond["target_rectangle_hamming_error"],
        "delta_transformation_cell_hamming_error": lambda normal, cond: None
        if normal["transformation_cell_count"] == 0 or cond["transformation_cell_count"] == 0
        else normal["transformation_cell_hamming_error"] - cond["transformation_cell_hamming_error"],
    }
    deltas: dict[str, list[float]] = {name: [] for name in metric_specs}
    per_episode = []
    for key in shared_keys:
        normal = normal_by_key[key]
        cond = condition_by_key[key]
        row = {"task_id": key[0], "query_pair_index": key[1]}
        for name, fn in metric_specs.items():
            value = fn(normal, cond)
            row[name] = value
            if value is not None:
                deltas[name].append(float(value))
        per_episode.append(row)
    intervals = {
        name: paired_bootstrap_ci(
            values,
            seed=stable_seed("bootstrap", name, seed=seed),
            resamples=resamples,
        )
        for name, values in deltas.items()
    }
    return {"paired_episode_count": len(shared_keys), "per_episode": per_episode}, intervals


def material_accuracy_drop(interval: dict[str, Any], threshold: float = 0.02) -> bool:
    return (
        interval.get("mean") is not None
        and interval["mean"] > threshold
        and interval["ci95_low"] is not None
        and interval["ci95_low"] > 0
    )


def material_hamming_improvement(interval: dict[str, Any], threshold: float = 1.0) -> bool:
    return (
        interval.get("mean") is not None
        and interval["mean"] < -threshold
        and interval["ci95_high"] is not None
        and interval["ci95_high"] < 0
    )


def choose_support_use_verdict(split_intervals: dict[str, dict[str, Any]]) -> str:
    deranged = split_intervals.get("OUTPUT_DERANGED_WITHIN_TASK", {})
    order = split_intervals.get("PAIR_ORDER_SHUFFLED", {})
    zeroed = split_intervals.get("SUPPORT_EVIDENCE_PATH_ZEROED", {})
    deranged_acc = deranged.get("delta_transformation_cell_accuracy", {})
    deranged_ham = deranged.get("delta_transformation_cell_hamming_error", {})
    order_acc = order.get("delta_transformation_cell_accuracy", {})
    order_ham = order.get("delta_transformation_cell_hamming_error", {})
    zero_acc = zeroed.get("delta_transformation_cell_accuracy", {})
    zero_ham = zeroed.get("delta_transformation_cell_hamming_error", {})
    deranged_hurts = material_accuracy_drop(deranged_acc) or material_hamming_improvement(deranged_ham)
    order_hurts = material_accuracy_drop(order_acc) or material_hamming_improvement(order_ham)
    zero_hurts = material_accuracy_drop(zero_acc) or material_hamming_improvement(zero_ham)
    deranged_acc_mean = deranged_acc.get("mean") or 0.0
    order_acc_mean = order_acc.get("mean") or 0.0
    deranged_ham_hurt = -(deranged_ham.get("mean") or 0.0)
    order_ham_hurt = -(order_ham.get("mean") or 0.0)
    if order_hurts and (order_acc_mean >= deranged_acc_mean or order_ham_hurt >= deranged_ham_hurt):
        return "ORDER_ARTIFACT_DOMINANT"
    if deranged_hurts:
        return "SUPPORT_MAPPING_USE_SUPPORTED"
    if zero_hurts:
        return "SUPPORT_PATH_USE_ONLY"
    similar_acc = abs(deranged_acc_mean) < 0.01 and abs(zero_acc.get("mean") or 0.0) < 0.01
    similar_ham = abs(deranged_ham.get("mean") or 0.0) < 1.0 and abs(zero_ham.get("mean") or 0.0) < 1.0
    if similar_acc and similar_ham:
        return "SUPPORT_MAPPING_USE_NOT_DETECTED"
    return "SUPPORT_USE_INCONCLUSIVE"


def grid_text_from_tokens(tokens: torch.Tensor, height: int, width: int) -> str:
    grid = tokens.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid[:height, :width].tolist())


def transformation_mask_text(episode: ArcEpisode) -> str:
    mask = transformation_cell_mask(episode).view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    rows = []
    for row in mask[: episode.query.output.height, : episode.query.output.width].tolist():
        rows.append(" ".join("T" if cell else "." for cell in row))
    return "\n".join(rows)


def render_ablation_examples(
    *,
    episodes_by_split: dict[str, list[ArcEpisode]],
    condition_predictions: dict[str, dict[str, list[dict[str, Any]]]],
    baseline_predictions: dict[str, dict[str, list[dict[str, Any]]]],
    paired_deltas: dict[str, dict[str, Any]],
) -> str:
    normal_lookup = {}
    condition_lookup = {}
    baseline_lookup = {}
    for split, condition_map in condition_predictions.items():
        for condition, rows in condition_map.items():
            for row in rows:
                key = (split, row["task_id"], row["query_pair_index"])
                condition_lookup[(condition, *key)] = row
                if condition == "NORMAL":
                    normal_lookup[key] = row
    for split, baseline_map in baseline_predictions.items():
        for baseline, rows in baseline_map.items():
            for row in rows:
                baseline_lookup[(baseline, split, row["task_id"], row["query_pair_index"])] = row
    candidates = []
    for split, delta_payload in paired_deltas.items():
        for row in delta_payload["OUTPUT_DERANGED_WITHIN_TASK"]["per_episode"]:
            key = (split, row["task_id"], row["query_pair_index"])
            if key in normal_lookup:
                candidates.append((split, row))
    best_deranged = sorted(
        candidates,
        key=lambda item: (
            -(item[1]["delta_transformation_cell_accuracy"] or 0.0),
            item[1]["delta_transformation_cell_hamming_error"] or 0.0,
        ),
    )[:4]
    similar = sorted(
        candidates,
        key=lambda item: (
            abs(item[1]["delta_transformation_cell_accuracy"] or 0.0),
            abs(item[1]["delta_transformation_cell_hamming_error"] or 0.0),
        ),
    )[:4]
    zero_candidates = []
    for split, delta_payload in paired_deltas.items():
        for row in delta_payload["SUPPORT_EVIDENCE_PATH_ZEROED"]["per_episode"]:
            key = (split, row["task_id"], row["query_pair_index"])
            if key in normal_lookup:
                zero_candidates.append((split, row))
    zero_hurts = sorted(
        zero_candidates,
        key=lambda item: (
            -(item[1]["delta_transformation_cell_accuracy"] or 0.0),
            item[1]["delta_transformation_cell_hamming_error"] or 0.0,
        ),
    )[:4]
    selected = [("normal_beats_deranged", *item) for item in best_deranged]
    selected += [("normal_deranged_similar", *item) for item in similar]
    selected += [("support_zeroed_hurts", *item) for item in zero_hurts]
    seen = set()
    lines = ["# VRM V1.1 support ablation examples", ""]
    for label, split, delta_row in selected:
        key = (split, delta_row["task_id"], delta_row["query_pair_index"])
        if (label, key) in seen:
            continue
        seen.add((label, key))
        episode = next(ep for ep in episodes_by_split[split] if ep.task_id == key[1] and ep.query_pair_index == key[2])
        lines.extend(
            [
                f"## {label}: {split} task {episode.task_id}, held-out pair {episode.query_pair_index}",
                "",
                f"target H/W: {episode.query.output.height} x {episode.query.output.width}",
                "normal support pairs:",
            ]
        )
        for support in episode.supports:
            lines.extend(
                [
                    f"- support pair {support.pair_index} input ({support.input.height}x{support.input.width})",
                    grid_to_text(support.input),
                    f"- support pair {support.pair_index} output ({support.output.height}x{support.output.width})",
                    grid_to_text(support.output),
                ]
            )
        lines.extend(
            [
                f"query input ({episode.query.input.height}x{episode.query.input.width}):",
                grid_to_text(episode.query.input),
                f"target output ({episode.query.output.height}x{episode.query.output.width}):",
                grid_to_text(episode.query.output),
                "transformation-cell mask:",
                transformation_mask_text(episode),
                "",
            ]
        )
        for condition in SUPPORT_ABLATION_CONDITIONS:
            pred = condition_lookup.get((condition, *key))
            lines.append(f"condition {condition}:")
            if pred is None:
                lines.append("not applicable")
            else:
                metrics = pred["metrics"]
                lines.extend(
                    [
                        f"predicted H/W: {pred['predicted_height']} x {pred['predicted_width']}",
                        f"transformation accuracy: {metrics['transformation_cell_accuracy']}",
                        f"transformation hamming: {metrics['transformation_cell_hamming_error']}",
                        grid_text_from_tokens(pred["prediction_colors"], pred["predicted_height"], pred["predicted_width"]),
                    ]
                )
            lines.append("")
        for baseline in ("IDENTITY_COPY", "SUPPORT_SHAPE_COPY"):
            pred = baseline_lookup[(baseline, *key)]
            metrics = pred["metrics"]
            lines.extend(
                [
                    f"baseline {baseline}:",
                    f"predicted H/W: {pred['predicted_height']} x {pred['predicted_width']}",
                    f"transformation accuracy: {metrics['transformation_cell_accuracy']}",
                    f"transformation hamming: {metrics['transformation_cell_hamming_error']}",
                    grid_text_from_tokens(pred["prediction_colors"], pred["predicted_height"], pred["predicted_width"]),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def render_ablation_summary(report: dict[str, Any]) -> str:
    lines = [
        "# VRM V1.1 - Support Mapping Counterfactual Ablation",
        "",
        "Evaluation-only diagnostic. No optimizer, gradient update, or checkpoint modification was used.",
        "",
        f"- command: `{report['command']}`",
        f"- checkpoint: `{report['checkpoint_path']}`",
        f"- checkpoint_sha256_before: `{report['checkpoint_sha256_before']}`",
        f"- checkpoint_sha256_after: `{report['checkpoint_sha256_after']}`",
        f"- task_manifest: `{report['task_manifest_path']}`",
        f"- train_episodes: {report['train_episode_count']}",
        f"- unseen_episodes: {report['unseen_episode_count']}",
        f"- train_output_deranged_eligible: {report['output_deranged_eligible_counts']['train_50']}",
        f"- unseen_output_deranged_eligible: {report['output_deranged_eligible_counts']['unseen_50']}",
        "",
        "## Counterfactual metrics",
        "",
        "|split|condition|shape_match|target_color_acc|transformation_acc|exact|mean_hamming|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["condition_aggregates"]:
        trans_acc = row["transformation_cell_accuracy"]
        trans_text = f"{trans_acc:.6f}" if trans_acc is not None else "null"
        lines.append(
            f"|{row['split']}|{row['condition']}|{row['shape_match_rate']:.6f}|"
            f"{row['target_rectangle_color_accuracy']:.6f}|{trans_text}|"
            f"{row['exact_decoded_grid_accuracy']:.6f}|{row['mean_target_rectangle_hamming_error']:.6f}|"
        )
    lines.extend(
        [
            "",
            "## Verdicts",
            "",
            f"- train_50: {report['support_use_verdicts']['train_50']}",
            f"- unseen_50: {report['support_use_verdicts']['unseen_50']}",
            "",
            "Support-path zeroing is an out-of-distribution stress test and is supplementary to output derangement.",
            "",
        ]
    )
    return "\n".join(lines)


def load_v1_1_manifest_selection(data: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_task_ids = sorted(str(task_id) for task_id in payload["train_50_task_ids"])
    unseen_task_ids = sorted(str(task_id) for task_id in payload["unseen_50_task_ids"])
    if len(train_task_ids) != 50 or len(unseen_task_ids) != 50:
        raise RuntimeError("V1.1 task manifest must contain exactly 50 train and 50 unseen task IDs")
    if set(train_task_ids).intersection(unseen_task_ids):
        raise RuntimeError("V1.1 task manifest train/unseen overlap")
    order = {task_id: idx for idx, task_id in enumerate(train_task_ids + unseen_task_ids)}
    train_set = set(train_task_ids)
    unseen_set = set(unseen_task_ids)
    train_episodes = [ep for ep in data["episodes"] if ep.task_id in train_set]
    unseen_episodes = [ep for ep in data["episodes"] if ep.task_id in unseen_set]
    train_episodes.sort(key=lambda ep: (order[ep.task_id], ep.query_pair_index))
    unseen_episodes.sort(key=lambda ep: (order[ep.task_id], ep.query_pair_index))
    return {
        "manifest_payload": payload,
        "train_task_ids": train_task_ids,
        "unseen_task_ids": unseen_task_ids,
        "train_episodes": train_episodes,
        "unseen_episodes": unseen_episodes,
    }


def compare_normal_to_prior(
    normal_aggregates: dict[str, dict[str, Any]],
    prior_metrics_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    prior = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    key_map = {
        "shape_match_rate": "output_shape_match_rate",
        "target_rectangle_color_accuracy": "target_rectangle_color_accuracy",
        "exact_decoded_grid_accuracy": "exact_decoded_grid_accuracy",
        "mean_target_rectangle_hamming_error": "mean_target_rectangle_hamming_error",
        "median_target_rectangle_hamming_error": "median_target_rectangle_hamming_error",
    }
    for split, prior_key in (("train_50", "final_train"), ("unseen_50", "final_unseen")):
        current = normal_aggregates[split]
        previous = prior[prior_key]
        rows = {}
        material = []
        for current_key, old_key in key_map.items():
            diff = abs(float(current[current_key]) - float(previous[old_key]))
            rows[current_key] = {"current": current[current_key], "prior": previous[old_key], "absolute_difference": diff}
            if diff > tolerance:
                material.append(current_key)
        comparisons[split] = {"metrics": rows, "material_differences": material}
    comparisons["tolerance"] = tolerance
    comparisons["matches_within_tolerance"] = all(not payload["material_differences"] for payload in comparisons.values() if isinstance(payload, dict) and "material_differences" in payload)
    return comparisons


def run_v1_1_support_ablation(args: argparse.Namespace) -> dict[str, Any]:
    command = " ".join(["python", "-m", "vrm.train", *sys.argv[1:]])
    determinism = configure_determinism(args.seed)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path != V1_1_BEST_CHECKPOINT:
        raise RuntimeError(f"support ablation must use {V1_1_BEST_CHECKPOINT}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not V1_1_TASK_MANIFEST.exists():
        raise FileNotFoundError(V1_1_TASK_MANIFEST)
    if not V1_1_METRICS.exists():
        raise FileNotFoundError(V1_1_METRICS)
    checkpoint_sha_before = hash_file(checkpoint_path)
    manifest_sha_before = hash_file(V1_1_TASK_MANIFEST)
    source_zip_sha_before = hash_file(V1_1_SOURCE_ZIP) if V1_1_SOURCE_ZIP.exists() else None
    data = build_dataset(args.data_root, seed=args.seed, manifest_path=None)
    selection = load_v1_1_manifest_selection(data, V1_1_TASK_MANIFEST)
    model, checkpoint = load_v1_1_checkpoint(checkpoint_path, device)
    model.eval()

    episodes_by_split = {
        "train_50": selection["train_episodes"],
        "unseen_50": selection["unseen_episodes"],
    }
    condition_aggregates: list[dict[str, Any]] = []
    condition_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    condition_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}

    normal_aggregates_by_split = {}
    for split, episodes in episodes_by_split.items():
        aggregate, records, predictions = evaluate_model_condition(
            model=model,
            episodes=episodes,
            split=split,
            condition="NORMAL",
            device=device,
            seed=args.seed,
        )
        normal_aggregates_by_split[split] = aggregate
        condition_aggregates.append(aggregate)
        condition_records.setdefault(split, {})["NORMAL"] = records
        condition_predictions.setdefault(split, {})["NORMAL"] = predictions
    normal_sanity = compare_normal_to_prior(normal_aggregates_by_split, V1_1_METRICS, args.sanity_tolerance)
    if not normal_sanity["matches_within_tolerance"]:
        save_json(V1_1_SUPPORT_ABLATION_DIR / "ablation_config.json", {"command": command, "normal_sanity": normal_sanity})
        raise RuntimeError(f"normal checkpoint metrics differ from previous V1.1 metrics: {normal_sanity}")

    for condition in SUPPORT_ABLATION_CONDITIONS:
        if condition == "NORMAL":
            continue
        for split, episodes in episodes_by_split.items():
            aggregate, records, predictions = evaluate_model_condition(
                model=model,
                episodes=episodes,
                split=split,
                condition=condition,
                device=device,
                seed=args.seed,
            )
            condition_aggregates.append(aggregate)
            condition_records.setdefault(split, {})[condition] = records
            condition_predictions.setdefault(split, {})[condition] = predictions

    baseline_aggregates: list[dict[str, Any]] = []
    baseline_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    baseline_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for baseline in ("IDENTITY_COPY", "SUPPORT_SHAPE_COPY"):
        for split, episodes in episodes_by_split.items():
            aggregate, records, predictions = evaluate_baseline(episodes=episodes, split=split, baseline=baseline)
            baseline_aggregates.append(aggregate)
            baseline_records.setdefault(split, {})[baseline] = records
            baseline_predictions.setdefault(split, {})[baseline] = predictions

    paired_deltas: dict[str, dict[str, Any]] = {}
    bootstrap_intervals: dict[str, dict[str, Any]] = {}
    for split in episodes_by_split:
        paired_deltas[split] = {}
        bootstrap_intervals[split] = {}
        normal_records = condition_records[split]["NORMAL"]
        for condition in SUPPORT_ABLATION_CONDITIONS:
            if condition == "NORMAL":
                continue
            deltas, intervals = paired_condition_deltas(
                normal_records,
                condition_records[split][condition],
                seed=stable_seed(split, condition, seed=args.seed),
                resamples=args.bootstrap_resamples,
            )
            paired_deltas[split][condition] = deltas
            bootstrap_intervals[split][condition] = intervals

    support_use_verdicts = {
        split: choose_support_use_verdict(bootstrap_intervals[split])
        for split in episodes_by_split
    }
    output_deranged_eligible_counts = {
        split: sum(1 for record in condition_records[split]["OUTPUT_DERANGED_WITHIN_TASK"] if record.get("applicable", True))
        for split in episodes_by_split
    }
    checkpoint_sha_after = hash_file(checkpoint_path)
    manifest_sha_after = hash_file(V1_1_TASK_MANIFEST)
    source_zip_sha_after = hash_file(V1_1_SOURCE_ZIP) if V1_1_SOURCE_ZIP.exists() else None
    report = {
        "architecture": "VRM V1.1 - Support Mapping Counterfactual Ablation",
        "command": command,
        "mode": "v1_1_support_ablation",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_weights_unchanged": checkpoint_sha_before == checkpoint_sha_after,
        "task_manifest_path": str(V1_1_TASK_MANIFEST),
        "task_manifest_sha256_before": manifest_sha_before,
        "task_manifest_sha256_after": manifest_sha_after,
        "task_manifest_unchanged": manifest_sha_before == manifest_sha_after,
        "source_zip_sha256_before": source_zip_sha_before,
        "source_zip_sha256_after": source_zip_sha_after,
        "source_zip_unchanged": source_zip_sha_before == source_zip_sha_after,
        "data_root": str(args.data_root),
        "seed": args.seed,
        "determinism": determinism,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "train_episode_count": len(selection["train_episodes"]),
        "unseen_episode_count": len(selection["unseen_episodes"]),
        "output_deranged_eligible_counts": output_deranged_eligible_counts,
        "normal_sanity_comparison": normal_sanity,
        "condition_aggregates": condition_aggregates,
        "baseline_aggregates": baseline_aggregates,
        "support_use_verdicts": support_use_verdicts,
        "bootstrap_resamples": args.bootstrap_resamples,
    }
    V1_1_SUPPORT_ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        V1_1_SUPPORT_ABLATION_DIR / "ablation_config.json",
        {
            "command": command,
            "mode": "v1_1_support_ablation",
            "checkpoint": str(checkpoint_path),
            "seed": args.seed,
            "bootstrap_resamples": args.bootstrap_resamples,
            "sanity_tolerance": args.sanity_tolerance,
            "conditions": SUPPORT_ABLATION_CONDITIONS,
        },
    )
    save_json(V1_1_SUPPORT_ABLATION_DIR / "reused_task_manifest.json", selection["manifest_payload"])
    save_json(
        V1_1_SUPPORT_ABLATION_DIR / "checkpoint_manifest.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "checkpoint_weights_unchanged": checkpoint_sha_before == checkpoint_sha_after,
            "checkpoint_step": checkpoint.get("step"),
        },
    )
    save_json(V1_1_SUPPORT_ABLATION_DIR / "ablation_metrics.json", report)
    save_json(V1_1_SUPPORT_ABLATION_DIR / "ablation_paired_deltas.json", paired_deltas)
    save_json(V1_1_SUPPORT_ABLATION_DIR / "ablation_bootstrap_intervals.json", bootstrap_intervals)
    per_episode_payload = {
        split: {
            condition: records
            for condition, records in condition_records[split].items()
        }
        for split in condition_records
    }
    save_json(V1_1_SUPPORT_ABLATION_DIR / "ablation_per_episode.json", per_episode_payload)
    save_json(
        V1_1_SUPPORT_ABLATION_DIR / "ablation_baselines.json",
        {"aggregates": baseline_aggregates, "per_episode": baseline_records},
    )
    save_text(V1_1_SUPPORT_ABLATION_DIR / "ablation_summary.md", render_ablation_summary(report))
    save_text(
        V1_1_SUPPORT_ABLATION_DIR / "ablation_examples.txt",
        render_ablation_examples(
            episodes_by_split=episodes_by_split,
            condition_predictions=condition_predictions,
            baseline_predictions=baseline_predictions,
            paired_deltas=paired_deltas,
        ),
    )
    save_text(
        V1_1_SUPPORT_ABLATION_DIR / "pytest_result.txt",
        (args.pytest_result or "pytest result not supplied") + "\n",
    )
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VRM V1.1 on ARC leave-one-out episodes.")
    parser.add_argument("--mode", choices=["v1_1_capacity50", "v1_1_support_ablation"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ARC_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=V1_1_BEST_CHECKPOINT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--sanity-tolerance", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=V1_1_DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--shape-loss-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pair-encoder-layers", type=int, default=1)
    parser.add_argument("--rewrite-layers", type=int, default=2)
    parser.add_argument("--rewrite-steps", type=int, default=4)
    parser.add_argument("--rule-slots", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--pytest-result", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "v1_1_capacity50":
        run_v1_1_capacity50(args)
    elif args.mode == "v1_1_support_ablation":
        run_v1_1_support_ablation(args)


if __name__ == "__main__":
    main()
