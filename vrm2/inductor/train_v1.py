from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as functional

from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.episodes import load_query_episodes, load_support_episodes
from vrm2.inductor.evaluate_v1 import evaluate_model_v1
from vrm2.inductor.hashing import module_state_hashes
from vrm2.inductor.model_v1 import (
    SequentialResidualInductorV1,
    component_parameter_counts,
    model_config_from_mapping,
)
from vrm2.inductor.train import batch_tensors, loo_views
from vrm2.inductor.types import SupportEpisode
from vrm2.oracle.train import load_checkpoint


def load_inductor_v1_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for section in ("model", "data", "episodes", "training", "evaluation", "outputs"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"V1 config missing mapping: {section}")
    if config["model_version"] != "sequential-residual-inductor-v1":
        raise ValueError("unexpected V1 model version")
    if int(config["model_seed"]) != 20260713:
        raise ValueError("V1 has exactly one frozen seed")
    return config


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shuffled_instruction_targets(
    programs: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Frozen non-identity permutation for every two-instruction target."""

    shuffled = programs.clone()
    length_two = lengths.eq(2)
    shuffled[length_two, 0] = programs[length_two, 1]
    shuffled[length_two, 1] = programs[length_two, 0]
    return shuffled


def sequential_program_loss(
    logits: Mapping[str, torch.Tensor], targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {}
    total = torch.zeros((), device=targets.device)
    for field_index, name in enumerate(
        ("operator", "selector", "argument_type", "argument_value")
    ):
        loss = functional.cross_entropy(
            logits[name].reshape(-1, logits[name].shape[-1]),
            targets[:, :, field_index].reshape(-1),
        )
        losses[name] = loss
        total = total + loss
    return total, {
        "total": float(total.detach().item()),
        **{name: float(value.detach().item()) for name, value in losses.items()},
    }


def save_checkpoint_v1(
    path: Path,
    model: SequentialResidualInductorV1,
    config: Mapping[str, Any],
    seed: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "model_version": model.model_version,
            "seed": seed,
            "step": step,
            "inductor_v1_config": dict(config),
        },
        path,
    )


def load_checkpoint_v1(
    path: Path, device: torch.device
) -> tuple[SequentialResidualInductorV1, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SequentialResidualInductorV1(
        model_config_from_mapping(payload["model_config"])
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def _length_audit(episodes: Sequence[SupportEpisode]) -> dict[str, Any]:
    counts = {
        str(length): sum(item.target_program_length == length for item in episodes)
        for length in (1, 2, 3)
    }
    maximum = max(item.target_program_length for item in episodes)
    if maximum != 2 or counts["3"] != 0:
        raise RuntimeError("V1 optimizer received a prohibited target length")
    return {
        "length_1_episodes": counts["1"],
        "length_2_episodes": counts["2"],
        "length_3_episodes_reaching_optimizer": counts["3"],
        "maximum_active_target_length_observed_by_optimizer": maximum,
    }


def train_one_seed_v1(
    config: Mapping[str, Any],
    seed: int,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if seed != int(config["model_seed"]):
        raise ValueError("V1 permits only the frozen model seed")
    if config["training"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_INDUCTOR_V1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_start = time.perf_counter()
    episode_dir = Path(config["episodes"]["runtime_dir"])
    opened_paths = [
        episode_dir / "train_supports.jsonl",
        episode_dir / "train_eval_supports.jsonl",
        episode_dir / "train_eval_queries.jsonl",
        episode_dir / "dev_supports.jsonl",
        episode_dir / "dev_queries.jsonl",
    ]
    data_start = time.perf_counter()
    train_episodes = load_support_episodes(opened_paths[0])
    train_eval_supports = load_support_episodes(opened_paths[1])
    train_eval_queries = load_query_episodes(opened_paths[2])
    dev_supports = load_support_episodes(opened_paths[3])
    dev_queries = load_query_episodes(opened_paths[4])
    data_seconds = time.perf_counter() - data_start
    target_audit = _length_audit(train_episodes)

    set_deterministic_seed(seed)
    model = SequentialResidualInductorV1(
        model_config_from_mapping(config["model"])
    ).to(device)
    components = component_parameter_counts(model)
    if components["total"] > int(config["training"]["max_parameters"]):
        raise RuntimeError("V1 parameter budget exceeded")
    interpreter, interpreter_metadata = load_checkpoint(
        Path(config["data"]["interpreter_checkpoint"]), device
    )
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    interpreter_hashes_before = module_state_hashes(interpreter)
    if (
        interpreter_hashes_before["model_state_sha256"]
        != config["data"]["interpreter_model_state_sha256"]
    ):
        raise RuntimeError("frozen interpreter hash mismatch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=tuple(config["training"]["betas"]),
    )
    maximum_steps = int(config["training"]["max_steps"])
    warmup = int(config["training"]["warmup_steps"])

    def factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        minimum = float(config["training"]["minimum_lr_ratio"])
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    mixed = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    autocast = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if mixed
        else (lambda: nullcontext())
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    epoch = 0
    seen_views = 0
    best_score: tuple[float, ...] | None = None
    best_step = 0
    no_improvement = 0
    loss_history = []
    ranking = []
    train_start = time.perf_counter()
    deadline = float(config["training"]["max_wall_clock_hours"]) * 3600.0
    stop = False
    shuffle_order = bool(config["training"]["shuffle_instruction_order"])
    while step < maximum_steps and not stop:
        indices = list(range(len(train_episodes)))
        random.Random(seed + epoch).shuffle(indices)
        batch_size = int(config["training"]["batch_size"])
        for offset in range(0, len(indices), batch_size):
            if time.perf_counter() - train_start > deadline:
                raise RuntimeError("V1 wall-clock budget exceeded")
            selected = indices[offset : offset + batch_size]
            inputs, outputs, lengths, programs = batch_tensors(
                train_episodes, selected, device
            )
            view_inputs, view_outputs = loo_views(inputs, outputs)
            repeated_lengths = lengths.repeat_interleave(3)
            repeated_programs = programs.repeat_interleave(3, dim=0)
            targets = (
                shuffled_instruction_targets(repeated_programs, repeated_lengths)
                if shuffle_order
                else repeated_programs
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(view_inputs, view_outputs, targets)
                loss, parts = sequential_program_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            step += 1
            seen_views += view_inputs.shape[0]
            if step == 1 or step % int(config["training"]["log_interval_steps"]) == 0:
                loss_history.append(
                    {"step": step, "learning_rate": optimizer.param_groups[0]["lr"], **parts}
                )
            if step % int(config["training"]["eval_interval_steps"]) == 0 or step == maximum_steps:
                dev = evaluate_model_v1(
                    model,
                    interpreter,
                    dev_supports,
                    dev_queries,
                    int(config["evaluation"]["batch_size"]),
                    device,
                    include_loo=False,
                )
                train_eval = evaluate_model_v1(
                    model,
                    interpreter,
                    train_eval_supports,
                    train_eval_queries,
                    int(config["evaluation"]["batch_size"]),
                    device,
                    include_loo=False,
                )
                score = (
                    float(dev["query_exact"]),
                    float(dev["behavioral_program_exact"]),
                    float(dev["autoregressive_program_exact"]),
                    float(train_eval["query_exact"]),
                    -float(step),
                )
                entry = {
                    "step": step,
                    "score": list(score),
                    "dev": dev,
                    "train_eval": train_eval,
                }
                ranking.append(entry)
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = step
                    no_improvement = 0
                    save_checkpoint_v1(checkpoint_path, model, config, seed, step)
                else:
                    no_improvement += 1
                if (
                    step >= int(config["training"]["minimum_steps"])
                    and no_improvement >= int(config["training"]["early_stopping_patience"])
                ):
                    stop = True
                    break
            if step >= maximum_steps:
                break
        epoch += 1

    training_seconds = time.perf_counter() - train_start
    model, checkpoint_metadata = load_checkpoint_v1(checkpoint_path, device)
    dev = evaluate_model_v1(
        model,
        interpreter,
        dev_supports,
        dev_queries,
        int(config["evaluation"]["batch_size"]),
        device,
        include_loo=True,
    )
    train_eval = evaluate_model_v1(
        model,
        interpreter,
        train_eval_supports,
        train_eval_queries,
        int(config["evaluation"]["batch_size"]),
        device,
        include_loo=True,
    )
    interpreter_hashes_after = module_state_hashes(interpreter)
    for entry in ranking:
        entry["selected"] = int(entry["step"]) == best_step
    ranking = sorted(ranking, key=lambda item: tuple(item["score"]), reverse=True)
    report = {
        "seed": seed,
        "ablation": config["ablation"],
        "component_parameter_counts": components,
        "steps_executed": step,
        "best_step": best_step,
        "checkpoint_step": checkpoint_metadata["step"],
        "best_score": list(best_score) if best_score is not None else None,
        "seen_loo_views": seen_views,
        "target_length_audit": target_audit,
        "opened_training_and_selection_paths": [str(path) for path in opened_paths],
        "data_preparation_seconds": data_seconds,
        "training_seconds": training_seconds,
        "total_wall_seconds": time.perf_counter() - total_start,
        "training_views_per_second": seen_views / training_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "interpreter_checkpoint_step": interpreter_metadata["step"],
        "interpreter_hashes_before": interpreter_hashes_before,
        "interpreter_hashes_after": interpreter_hashes_after,
        "interpreter_unchanged": interpreter_hashes_before == interpreter_hashes_after,
        "interpreter_parameters_frozen": all(
            not parameter.requires_grad for parameter in interpreter.parameters()
        ),
        "interpreter_eval_mode": not interpreter.training,
        "loss_history": loss_history,
        "checkpoint_ranking": ranking,
        "train_eval": train_eval,
        "dev": dev,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report
