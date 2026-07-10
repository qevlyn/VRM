from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM, VRMConfig

from .vrm_probe_data import TwinEpisode, collate_twin_episodes, model_inputs_for_twin_batch, training_view
from .vrm_probe_metrics import CONDITION_NORMAL, evaluate_condition, twin_pair_joint_normal_accuracy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(seed: int) -> dict[str, Any]:
    set_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic_mode = "enabled_warn_only"
    except TypeError:
        torch.use_deterministic_algorithms(True)
        deterministic_mode = "enabled"
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "torch_deterministic_algorithms": deterministic_mode,
        "cuda_available": torch.cuda.is_available(),
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parameter_count(model: VRM) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def initialize_fresh_model(config: VRMConfig, device: torch.device, seed: int) -> tuple[VRM, dict[str, Any]]:
    set_seed(seed)
    model = VRM(config).to(device)
    report = {
        "architecture": "VRM V1.1 - Masked Query Spatial Evidence Rewrite",
        "fresh_random_initialization": True,
        "loaded_checkpoint": False,
        "loaded_checkpoint_path": None,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "config": asdict(config),
    }
    return model, report


def compute_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    if step <= 0:
        return 0.0
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def grad_norm(model: VRM) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm(2).item())
        total += value * value
    return math.sqrt(total)


def finite_parameters(model: VRM) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())


def finite_gradients(model: VRM) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def next_train_indices(step: int, episode_count: int, batch_size: int, seed: int) -> list[int]:
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    rng = random.Random(seed + step * 1_000_003)
    return [rng.randrange(episode_count) for _ in range(batch_size)]


def train_step(
    *,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    train_episodes: list[TwinEpisode],
    step: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    shape_loss_weight: float,
    grad_clip_norm: float,
    learning_rate: float,
) -> dict[str, Any]:
    model.train()
    set_optimizer_lr(optimizer, learning_rate)
    indices = next_train_indices(step, len(train_episodes), batch_size, seed)
    batch_episodes = [
        training_view(train_episodes[idx], seed=seed, step=step * 10_000 + pos)
        for pos, idx in enumerate(indices)
    ]
    batch = collate_twin_episodes(batch_episodes)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**model_inputs_for_twin_batch(batch))
    loss, components = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    loss.backward()
    if not finite_gradients(model):
        raise RuntimeError("non-finite gradient")
    grad_before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm))
    grad_after = grad_norm(model)
    optimizer.step()
    if not finite_parameters(model):
        raise RuntimeError("non-finite parameter")
    return {
        "event": "train_step",
        "step": step,
        "episode_indices": indices,
        "learning_rate": learning_rate,
        "train_loss": float(loss.detach().cpu().item()),
        "color_loss": float(components["color_loss"].detach().cpu().item()),
        "height_loss": float(components["height_loss"].detach().cpu().item()),
        "width_loss": float(components["width_loss"].detach().cpu().item()),
        "grad_norm_before_clip": grad_before,
        "grad_norm_after_clip": grad_after,
        "finite_loss": True,
        "finite_gradients": True,
        "finite_parameters": True,
        "support_pair_orders": [[pair.pair_index for pair in episode.supports] for episode in batch_episodes],
    }


def evaluate_dev_normal_for_selection(
    *,
    model: VRM,
    dev_episodes: list[TwinEpisode],
    device: torch.device,
    seed: int,
    batch_size: int,
    shape_loss_weight: float,
    step: int,
) -> dict[str, Any]:
    aggregate, rows = evaluate_condition(
        model=model,
        episodes=dev_episodes,
        split="dev",
        condition=CONDITION_NORMAL,
        device=device,
        seed=seed,
        batch_size=batch_size,
        shape_loss_weight=shape_loss_weight,
    )
    aggregate = dict(aggregate)
    aggregate["event"] = "dev_eval"
    aggregate["step"] = step
    aggregate["twin_pair_joint_normal_accuracy"] = twin_pair_joint_normal_accuracy(dev_episodes, rows)
    return aggregate


def dev_selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["candidate_nearest_score"]),
        float(row["twin_pair_joint_normal_accuracy"]),
        float(row["direct_grid_exact_accuracy"]),
        -float(row["mean_target_rectangle_hamming_error"]),
    )


def is_better_dev(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    return dev_selection_key(candidate) > dev_selection_key(incumbent)


def save_checkpoint(
    path: Path,
    *,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    config: VRMConfig,
    step: int,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "extra": extra,
        },
        path,
    )


def state_dict_copy(model: VRM) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_state_dict_copy(model: VRM, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in copy.deepcopy(state).items()})
