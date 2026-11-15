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
from vrm2.inductor.evaluate import evaluate_model_on_episodes
from vrm2.inductor.hashing import module_state_hashes
from vrm2.inductor.model import (
    SupervisedTypedProgramInductorV0,
    inductor_parameter_count,
    model_config_from_mapping,
)
from vrm2.inductor.types import SupportEpisode
from vrm2.oracle.train import load_checkpoint


def load_inductor_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for section in ("model", "data", "episodes", "training", "evaluation", "gates", "outputs"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"inductor config missing mapping: {section}")
    if config["model_version"] != "supervised-typed-program-inductor-v0":
        raise ValueError("unexpected inductor model version")
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


def batch_tensors(
    episodes: Sequence[SupportEpisode], indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [episodes[index] for index in indices]
    inputs = torch.tensor([item.support_input_grids for item in selected], dtype=torch.long, device=device)
    outputs = torch.tensor([item.support_output_grids for item in selected], dtype=torch.long, device=device)
    lengths = torch.tensor([item.target_program_length for item in selected], dtype=torch.long, device=device)
    programs = torch.tensor([item.target_typed_program_ids for item in selected], dtype=torch.long, device=device)
    return inputs, outputs, lengths, programs


def loo_views(inputs: torch.Tensor, outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_views = []
    output_views = []
    for omitted in range(3):
        keep = [index for index in range(3) if index != omitted]
        input_views.append(inputs[:, keep])
        output_views.append(outputs[:, keep])
    return torch.stack(input_views, dim=1).flatten(0, 1), torch.stack(output_views, dim=1).flatten(0, 1)


def program_loss(
    logits: Mapping[str, torch.Tensor],
    lengths: torch.Tensor,
    programs: torch.Tensor,
    loss_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = loss_weights or {
        "length": 1.0,
        "operator": 1.0,
        "selector": 1.0,
        "argument_type": 1.0,
        "argument_value": 1.0,
    }
    if any(float(weights[name]) != 1.0 for name in weights):
        raise ValueError("inductor V0 requires equal unit loss weights")
    length_targets = lengths - 2
    length_loss = functional.cross_entropy(logits["length"], length_targets)
    active = torch.arange(3, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    field_losses = {}
    total = float(weights["length"]) * length_loss
    for field_index, name in enumerate(("operator", "selector", "argument_type", "argument_value")):
        per_step_losses = []
        for step in range(3):
            step_active = active[:, step]
            if step_active.any():
                per_step_losses.append(
                    functional.cross_entropy(
                        logits[name][step_active, step],
                        programs[step_active, step, field_index],
                    )
                )
        field_loss = torch.stack(per_step_losses).sum()
        field_losses[name] = field_loss
        total = total + float(weights[name]) * field_loss
    return total, {
        "total": float(total.detach().item()),
        "length": float(length_loss.detach().item()),
        **{name: float(value.detach().item()) for name, value in field_losses.items()},
    }


def save_checkpoint(
    path: Path,
    model: SupervisedTypedProgramInductorV0,
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
            "inductor_config": dict(config),
        },
        path,
    )


def load_inductor_checkpoint(
    path: Path, device: torch.device
) -> tuple[SupervisedTypedProgramInductorV0, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SupervisedTypedProgramInductorV0(model_config_from_mapping(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def train_one_seed(
    config: Mapping[str, Any],
    seed: int,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if config["training"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_INDUCTOR_V0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_start = time.perf_counter()
    episode_dir = Path(config["episodes"]["runtime_dir"])
    data_start = time.perf_counter()
    train_episodes = load_support_episodes(episode_dir / "train_supports.jsonl")
    train_eval_supports = load_support_episodes(episode_dir / "train_eval_supports.jsonl")
    train_eval_queries = load_query_episodes(episode_dir / "train_eval_queries.jsonl")
    dev_supports = load_support_episodes(episode_dir / "dev_supports.jsonl")
    dev_queries = load_query_episodes(episode_dir / "dev_queries.jsonl")
    data_seconds = time.perf_counter() - data_start

    set_deterministic_seed(seed)
    model = SupervisedTypedProgramInductorV0(model_config_from_mapping(config["model"])).to(device)
    parameter_count = inductor_parameter_count(model)
    if parameter_count > int(config["training"]["max_parameters"]):
        raise RuntimeError("inductor parameter budget exceeded")
    interpreter, interpreter_metadata = load_checkpoint(
        Path(config["data"]["interpreter_checkpoint"]), device
    )
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    interpreter_hashes_before = module_state_hashes(interpreter)
    expected_interpreter_hash = str(config["data"]["interpreter_model_state_sha256"])
    if interpreter_hashes_before["model_state_sha256"] != expected_interpreter_hash:
        raise RuntimeError("frozen V1 model-state hash mismatch before inductor training")
    interpreter_before = {
        key: value.detach().cpu().clone() for key, value in interpreter.state_dict().items()
    }

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
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if mixed else (lambda: nullcontext())
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
    deadline_seconds = float(config["training"]["max_wall_clock_hours"]) * 3600.0
    stop = False
    while step < maximum_steps and not stop:
        indices = list(range(len(train_episodes)))
        random.Random(seed + epoch).shuffle(indices)
        batch_size = int(config["training"]["batch_size"])
        for offset in range(0, len(indices), batch_size):
            if time.perf_counter() - train_start > deadline_seconds:
                raise RuntimeError("inductor V0 wall-clock budget exceeded")
            model.train()
            batch_indices = indices[offset : offset + batch_size]
            inputs, outputs, lengths, programs = batch_tensors(train_episodes, batch_indices, device)
            view_inputs, view_outputs = loo_views(inputs, outputs)
            repeated_lengths = lengths.repeat_interleave(3)
            repeated_programs = programs.repeat_interleave(3, dim=0)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(view_inputs, view_outputs)
                loss, parts = program_loss(
                    logits,
                    repeated_lengths,
                    repeated_programs,
                    config["training"]["loss_weights"],
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            step += 1
            seen_views += view_inputs.shape[0]
            if step == 1 or step % int(config["training"]["log_interval_steps"]) == 0:
                loss_history.append({"step": step, "learning_rate": optimizer.param_groups[0]["lr"], **parts})
            if step % int(config["training"]["eval_interval_steps"]) == 0 or step == maximum_steps:
                dev = evaluate_model_on_episodes(
                    model, interpreter, dev_supports, dev_queries,
                    int(config["evaluation"]["batch_size"]), device, include_loo=False
                )
                train_eval = evaluate_model_on_episodes(
                    model, interpreter, train_eval_supports, train_eval_queries,
                    int(config["evaluation"]["batch_size"]), device, include_loo=False
                )
                score = (
                    float(dev["query_exact"]),
                    float(dev["behavioral_program_exact"]),
                    float(dev["program_exact"]),
                    float(train_eval["query_exact"]),
                    -float(step),
                )
                entry = {"step": step, "score": list(score), "dev": dev, "train_eval": train_eval}
                ranking.append(entry)
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = step
                    no_improvement = 0
                    save_checkpoint(checkpoint_path, model, config, seed, step)
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
    model, checkpoint_metadata = load_inductor_checkpoint(checkpoint_path, device)
    dev = evaluate_model_on_episodes(
        model, interpreter, dev_supports, dev_queries,
        int(config["evaluation"]["batch_size"]), device, include_loo=True
    )
    train_eval = evaluate_model_on_episodes(
        model, interpreter, train_eval_supports, train_eval_queries,
        int(config["evaluation"]["batch_size"]), device, include_loo=True
    )
    interpreter_unchanged = all(
        torch.equal(interpreter_before[key], value.detach().cpu())
        for key, value in interpreter.state_dict().items()
    )
    interpreter_hashes_after = module_state_hashes(interpreter)
    interpreter_parameters_frozen = all(
        not parameter.requires_grad for parameter in interpreter.parameters()
    )
    interpreter_eval_mode = not interpreter.training
    for entry in ranking:
        entry["selected"] = int(entry["step"]) == best_step
    ranking = sorted(ranking, key=lambda item: tuple(item["score"]), reverse=True)
    report = {
        "seed": seed,
        "parameter_count": parameter_count,
        "steps_executed": step,
        "best_step": best_step,
        "checkpoint_step": checkpoint_metadata["step"],
        "best_score": list(best_score) if best_score is not None else None,
        "seen_loo_views": seen_views,
        "data_preparation_seconds": data_seconds,
        "training_seconds": training_seconds,
        "total_wall_seconds": time.perf_counter() - total_start,
        "training_views_per_second": seen_views / training_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "interpreter_checkpoint_step": interpreter_metadata["step"],
        "interpreter_unchanged": interpreter_unchanged,
        "interpreter_parameters_frozen": interpreter_parameters_frozen,
        "interpreter_eval_mode": interpreter_eval_mode,
        "interpreter_hashes_before": interpreter_hashes_before,
        "interpreter_hashes_after": interpreter_hashes_after,
        "loss_history": loss_history,
        "checkpoint_ranking": ranking,
        "train_eval": train_eval,
        "dev": dev,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report
