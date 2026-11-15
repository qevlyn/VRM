from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrm.data import ArcEpisode, move_tensor_batch, model_inputs_from_batch
from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM, VRMConfig
from vrm.twin_tasks.r5a4a_arc_manifest import (
    ARTIFACT_DIR,
    INITIALIZATION_SEED,
    PROJECT_ROOT,
    R5A4ASplits,
    assert_disjoint_splits,
    episodes_for_task_ids,
    file_sha256,
    manifest_hashes_for_commitment,
    render_source_manifest_text,
    source_file_manifest,
    split_counts,
    write_json,
    write_split_artifacts,
    write_text,
)
from vrm.twin_tasks.r5a4a_neural_vrm_data import make_training_episode

MAX_OPTIMIZER_STEPS = 12000
CHECKPOINT_INTERVAL = 1000
VALIDATION_INTERVAL = 1000
DEFAULT_BATCH_SIZE = 1
DEFAULT_SHAPE_LOSS_WEIGHT = 0.5
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MIN_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_STEPS = 100
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_GRAD_CLIP_NORM = 1.0


class DeterministicTaskSampler:
    def __init__(self, splits: R5A4ASplits, *, seed: int) -> None:
        self.tasks_by_id = splits.tasks_by_id
        self.task_ids = list(splits.training_task_ids)
        self.rng = random.Random(seed)
        if not self.task_ids:
            raise ValueError("R5A-4A training pool is empty")

    def next_episode(self) -> ArcEpisode:
        task_id = self.task_ids[self.rng.randrange(len(self.task_ids))]
        task = self.tasks_by_id[task_id]
        query_pair_index = self.rng.randrange(len(task.train))
        return make_training_episode(task, query_pair_index, source_split="training")

    def next_batch(self, batch_size: int) -> list[ArcEpisode]:
        return [self.next_episode() for _ in range(batch_size)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int, torch_threads: int) -> dict[str, Any]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(torch_threads, os.cpu_count() or 1)))
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
        "seed": seed,
        "torch_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": deterministic_algorithms,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch.backends, "cudnn") else None,
    }


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_lr(step: int, total_steps: int) -> float:
    if step <= 0:
        return 0.0
    if step <= DEFAULT_WARMUP_STEPS:
        return DEFAULT_LEARNING_RATE * step / DEFAULT_WARMUP_STEPS
    progress = (step - DEFAULT_WARMUP_STEPS) / max(1, total_steps - DEFAULT_WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return DEFAULT_MIN_LEARNING_RATE + (DEFAULT_LEARNING_RATE - DEFAULT_MIN_LEARNING_RATE) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def parameter_count(model: VRM) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def optimizer_config() -> dict[str, Any]:
    return {
        "name": "AdamW",
        "learning_rate": DEFAULT_LEARNING_RATE,
        "minimum_learning_rate": DEFAULT_MIN_LEARNING_RATE,
        "warmup_steps": DEFAULT_WARMUP_STEPS,
        "betas": list(DEFAULT_BETAS),
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "grad_clip_norm": DEFAULT_GRAD_CLIP_NORM,
        "source": "existing VRM V1.1 optimizer/loss conventions",
    }


def model_config() -> VRMConfig:
    return VRMConfig()


def build_optimizer(model: VRM) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE, betas=DEFAULT_BETAS, weight_decay=DEFAULT_WEIGHT_DECAY)


def save_state(path: Path, *, model: VRM, optimizer: torch.optim.Optimizer | None, config: VRMConfig, step: int, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "config": asdict(config),
            "step": step,
            "extra": extra,
        },
        path,
    )


def load_state(path: Path, device: torch.device) -> tuple[VRM, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = VRMConfig(**payload["config"])
    model = VRM(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def evaluate_validation_loss(*, model: VRM, episodes: list[ArcEpisode], device: torch.device, shape_loss_weight: float) -> dict[str, Any]:
    from vrm.data import collate_episodes

    model.eval()
    totals = {"total": 0.0, "color": 0.0, "height": 0.0, "width": 0.0}
    cell_correct = 0
    cell_total = 0
    exact = 0
    with torch.no_grad():
        for episode in episodes:
            batch = move_tensor_batch(collate_episodes([episode]), device)
            outputs = model(**model_inputs_from_batch(batch))
            loss, components = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
            totals["total"] += float(loss.detach().cpu().item())
            totals["color"] += float(components["color_loss"].detach().cpu().item())
            totals["height"] += float(components["height_loss"].detach().cpu().item())
            totals["width"] += float(components["width_loss"].detach().cpu().item())
            prediction = outputs["color_log_probs"].argmax(dim=-1)[0]
            target = batch["query_targets"][0]
            mask = batch["query_target_mask"][0].to(torch.bool)
            correct = int((prediction[mask] == target[mask]).sum().detach().cpu().item())
            total = int(mask.sum().detach().cpu().item())
            cell_correct += correct
            cell_total += total
            pred_h = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
            pred_w = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
            target_h = int(batch["query_output_heights"][0].detach().cpu().item())
            target_w = int(batch["query_output_widths"][0].detach().cpu().item())
            exact += int(pred_h == target_h and pred_w == target_w and correct == total)
    count = len(episodes)
    return {
        "fold_count": count,
        "mean_total_loss": totals["total"] / max(count, 1),
        "mean_color_loss": totals["color"] / max(count, 1),
        "mean_height_loss": totals["height"] / max(count, 1),
        "mean_width_loss": totals["width"] / max(count, 1),
        "cell_accuracy": cell_correct / max(cell_total, 1),
        "exact_full_grid_accuracy": exact / max(count, 1),
    }


def select_best_checkpoint_step(validation_curve: list[dict[str, Any]]) -> int:
    if not validation_curve:
        raise ValueError("validation curve is empty")
    return int(min(validation_curve, key=lambda row: (row["mean_total_loss"], row["step"]))["step"])


def source_paths_used_by_r5a4a() -> list[Path]:
    return [
        PROJECT_ROOT / "vrm" / "model.py",
        PROJECT_ROOT / "vrm" / "data.py",
        PROJECT_ROOT / "vrm" / "metrics.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4a_arc_manifest.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4a_neural_vrm_data.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4a_neural_vrm_train.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4a_neural_vrm_eval.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4a_neural_vrm_noid_audit.py",
        PROJECT_ROOT / "vrm" / "twin_tasks" / "run_r5a4a_neural_vrm_arc_preflight.py",
    ]


def build_pre_training_commitment(*, splits: R5A4ASplits, config: VRMConfig, max_steps: int, batch_size: int, device: torch.device, command: str, determinism: dict[str, Any]) -> dict[str, Any]:
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_paths_used_by_r5a4a() if path.exists()}
    return {
        "command": command,
        "commitment_created_before_training": True,
        "initialization_seed": INITIALIZATION_SEED,
        "model_config": asdict(config),
        "optimizer_configuration": optimizer_config(),
        "learning_rate_schedule": "100-step warmup then cosine decay to 2e-5 over the declared maximum optimizer steps",
        "batch_size": batch_size,
        "maximum_optimizer_steps": max_steps,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_checkpoint_selection_rule": "minimum validation mean_total_loss; sentinel metrics are not computed until selected checkpoint is frozen",
        "metric_definitions": {
            "exact_full_grid_accuracy": "predicted shape equals hidden target shape and all cells inside the hidden target rectangle match",
            "exact_output_shape_accuracy": "predicted height and width equal hidden target height and width",
            "cell_accuracy_inside_true_target_rectangle": "target-rectangle correct cells divided by target-rectangle cell count",
            "color_nll": "negative log likelihood of true colors inside hidden target rectangle",
            "height_nll": "cross entropy against hidden target height class after inference",
            "width_nll": "cross entropy against hidden target width class after inference",
            "total_existing_model_loss": "factorized_shape_color_loss with shape_loss_weight 0.5",
        },
        "support_ablation_definitions": {
            "TRAINED_ZEROED_SUPPORT_OUTPUTS": "support output color tokens set to color 0 while masks, roles, coordinates, pair indices, support inputs, and query input remain unchanged",
            "TRAINED_QUERY_ONLY": "attempt to set all support evidence mask entries false; if VRM rejects zero valid support keys it is reported unsupported",
            "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER": "complete support input/output pairs are moved together into deterministic support-pair slots",
        },
        "no_id_audit_protocol": "changed external task metadata must leave model input tensors and loss-free outputs byte-identical",
        "no_query_target_firewall_policy": "hidden target tensors are not included in model_inputs_from_batch and are used only after model inference for loss/metrics",
        "manifest_hashes": manifest_hashes_for_commitment(splits),
        "source_hashes": source_hashes,
        "split_counts": split_counts(splits),
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "determinism": determinism,
    }


def write_training_config(*, path: Path, config: VRMConfig, max_steps: int, batch_size: int, device: torch.device, command: str) -> dict[str, Any]:
    payload = {
        "experiment": "R5A-4A ARC-Grounded Neural VRM Preflight",
        "architecture": "VRM V1.1 - Masked Query Spatial Evidence Rewrite",
        "command": command,
        "model_config": asdict(config),
        "initialization_seed": INITIALIZATION_SEED,
        "maximum_optimizer_steps": max_steps,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "validation_interval": VALIDATION_INTERVAL,
        "batch_size": batch_size,
        "shape_loss_weight": DEFAULT_SHAPE_LOSS_WEIGHT,
        "optimizer": optimizer_config(),
        "device": str(device),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    write_json(path, payload)
    return payload


def run_training(*, splits: R5A4ASplits, artifact_dir: Path = ARTIFACT_DIR, max_steps: int = MAX_OPTIMIZER_STEPS, batch_size: int = DEFAULT_BATCH_SIZE, device_arg: str = "auto", torch_threads: int = 4, command: str = "") -> dict[str, Any]:
    if max_steps != MAX_OPTIMIZER_STEPS:
        raise ValueError("R5A-4A final training must use exactly 12000 optimizer steps")
    assert_disjoint_splits(splits)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    determinism = configure_determinism(INITIALIZATION_SEED, torch_threads)
    device = resolve_device(device_arg)
    config = model_config()
    source_before = source_file_manifest()
    write_text(artifact_dir / "source_manifest_before.txt", render_source_manifest_text(source_before))
    write_split_artifacts(splits, artifact_dir)
    training_config = write_training_config(path=artifact_dir / "training_config.json", config=config, max_steps=max_steps, batch_size=batch_size, device=device, command=command)
    commitment = build_pre_training_commitment(splits=splits, config=config, max_steps=max_steps, batch_size=batch_size, device=device, command=command, determinism=determinism)
    write_json(artifact_dir / "pre_training_commitment.json", commitment)

    set_seed(INITIALIZATION_SEED)
    model = VRM(config).to(device)
    optimizer = build_optimizer(model)
    initial_path = artifact_dir / "initial_state_before_any_optimizer_step.pt"
    save_state(initial_path, model=model, optimizer=None, config=config, step=0, extra={"saved_before_any_optimizer_step": True, "initialization_seed": INITIALIZATION_SEED, "parameter_count": parameter_count(model)})

    validation_episodes = episodes_for_task_ids(splits.tasks_by_id, splits.validation_task_ids, source_split="validation")
    sampler = DeterministicTaskSampler(splits, seed=INITIALIZATION_SEED + 4242)
    log_path = artifact_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    validation_curve: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    best_loss = math.inf
    best_step: int | None = None
    selected_path = artifact_dir / "selected_trained_state.pt"
    start_time = time.perf_counter()

    from vrm.data import collate_episodes

    for step in range(1, max_steps + 1):
        model.train()
        lr = compute_lr(step, max_steps)
        set_optimizer_lr(optimizer, lr)
        episodes = sampler.next_batch(batch_size)
        batch = move_tensor_batch(collate_episodes(episodes), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_inputs_from_batch(batch))
        loss, components = factorized_shape_color_loss(outputs, batch, DEFAULT_SHAPE_LOSS_WEIGHT)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP_NORM))
        optimizer.step()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "train_step",
                        "step": step,
                        "task_ids": [episode.task_id for episode in episodes],
                        "query_pair_indices": [episode.query_pair_index for episode in episodes],
                        "learning_rate": lr,
                        "total_loss": float(loss.detach().cpu().item()),
                        "color_loss": float(components["color_loss"].detach().cpu().item()),
                        "height_loss": float(components["height_loss"].detach().cpu().item()),
                        "width_loss": float(components["width_loss"].detach().cpu().item()),
                        "grad_norm_before_clip": grad_norm,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if step % VALIDATION_INTERVAL == 0:
            validation = evaluate_validation_loss(model=model, episodes=validation_episodes, device=device, shape_loss_weight=DEFAULT_SHAPE_LOSS_WEIGHT)
            validation["step"] = step
            validation_curve.append(validation)
            checkpoint_path = artifact_dir / f"checkpoint_step_{step:05d}.pt"
            save_state(checkpoint_path, model=model, optimizer=optimizer, config=config, step=step, extra={"validation": validation, "checkpoint_selection_metric": "validation_mean_total_loss"})
            checkpoint_rows.append({"step": step, "checkpoint_path": str(checkpoint_path), "validation_mean_total_loss": validation["mean_total_loss"]})
            if validation["mean_total_loss"] < best_loss:
                best_loss = validation["mean_total_loss"]
                best_step = step
                save_state(selected_path, model=model, optimizer=optimizer, config=config, step=step, extra={"selection_rule": "minimum validation mean_total_loss", "selected_by_validation_only": True, "sentinel_metrics_available_at_selection_time": False, "validation": validation})

    if best_step is None:
        raise RuntimeError("no validation checkpoint was selected")
    write_json(artifact_dir / "validation_curve.json", {"selection_metric": "validation_mean_total_loss", "validation_interval": VALIDATION_INTERVAL, "curve": validation_curve})
    checkpoint_selection = {
        "selected_checkpoint_step": best_step,
        "selected_checkpoint_path": str(selected_path),
        "selection_rule": "minimum validation mean_total_loss",
        "selected_by_validation_only": True,
        "sentinel_metrics_available_at_selection_time": False,
        "checkpoint_candidates": checkpoint_rows,
    }
    write_json(artifact_dir / "checkpoint_selection.json", checkpoint_selection)
    source_after = source_file_manifest()
    write_text(artifact_dir / "source_manifest_after.txt", render_source_manifest_text(source_after))
    return {
        "artifact_dir": str(artifact_dir),
        "device": str(device),
        "config": asdict(config),
        "training_config": training_config,
        "initial_state_path": str(initial_path),
        "selected_state_path": str(selected_path),
        "selected_checkpoint_step": best_step,
        "validation_curve": validation_curve,
        "checkpoint_selection": checkpoint_selection,
        "wall_clock_training_seconds": time.perf_counter() - start_time,
        "source_manifest_before": source_before,
        "source_manifest_after": source_after,
    }
