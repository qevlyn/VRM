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

from vrm2.data.microworld import load_config as load_stage0_config
from vrm2.data.serialization import canonical_json_bytes
from vrm2.oracle.dataset import (
    assert_disjoint_examples,
    build_oracle_examples,
    example_fingerprint,
    iter_oracle_batches,
    load_oracle_corpus,
)
from vrm2.oracle.evaluate import evaluate_model
from vrm2.oracle.model import OracleInterpreterV0, model_config_from_mapping, parameter_count
from vrm2.oracle.types import OracleBatch, OracleExample


def load_oracle_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("oracle config must use the dependency-free JSON subset of YAML 1.2") from exc
    for section in ("model", "data", "training", "evaluation", "outputs"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"oracle config is missing mapping: {section}")
    if config["model"]["maximum_program_length"] != 3:
        raise ValueError("oracle maximum program length must remain 3")
    if config["training"]["max_parameters"] > 5_000_000:
        raise ValueError("oracle run policy forbids more than 5,000,000 parameters")
    if config.get("model_version", "v0") not in ("v0", "v1"):
        raise ValueError("model_version must be v0 or v1")
    return config


def build_oracle_model(oracle_config: Mapping[str, Any]) -> torch.nn.Module:
    model_config = model_config_from_mapping(oracle_config["model"])
    version = str(oracle_config.get("model_version", "v0"))
    if version == "v0":
        return OracleInterpreterV0(model_config)
    if version == "v1":
        from vrm2.oracle.model_v1 import OracleInterpreterV1

        return OracleInterpreterV1(model_config)
    raise ValueError(f"unsupported oracle model version: {version}")


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def oracle_loss(
    step_logits: torch.Tensor,
    batch: OracleBatch,
    lambda_intermediate: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_logits = step_logits[:, -1].permute(0, 2, 3, 1).reshape(-1, 10)
    final_loss = functional.cross_entropy(final_logits, batch.target_output_grid.reshape(-1))
    batch_size, program_length = step_logits.shape[:2]
    intermediate_logits = step_logits.permute(0, 1, 3, 4, 2).reshape(-1, 10)
    intermediate_loss = functional.cross_entropy(
        intermediate_logits,
        batch.intermediate_target_grids.reshape(batch_size * program_length * 8 * 8),
    )
    total = final_loss + lambda_intermediate * intermediate_loss
    return total, {
        "total": float(total.detach().item()),
        "final": float(final_loss.detach().item()),
        "intermediate": float(intermediate_loss.detach().item()),
    }


def build_run_datasets(
    oracle_config: Mapping[str, Any],
    stage0_config: Mapping[str, Any],
    sealed_examples: Sequence[OracleExample],
) -> dict[str, tuple[OracleExample, ...]]:
    data = oracle_config["data"]
    train_examples = build_oracle_examples(
        stage0_config, "train", int(data["train_examples"]), int(data["train_seed"])
    )
    train_eval = _build_disjoint_evaluation_examples(
        stage0_config,
        "train",
        int(data["train_eval_examples"]),
        int(data["train_eval_seed"]),
        train_examples,
    )
    datasets = {
        "train": train_examples,
        "train_eval": train_eval,
        "dev": build_oracle_examples(
            stage0_config, "dev", int(data["dev_examples"]), int(data["dev_seed"])
        ),
        "sealed": tuple(sealed_examples),
    }
    assert_disjoint_examples(datasets["train"], datasets["train_eval"])
    assert_disjoint_examples(datasets["train"], datasets["dev"])
    assert_disjoint_examples(datasets["train"], datasets["sealed"])
    return datasets


def _build_disjoint_evaluation_examples(
    stage0_config: Mapping[str, Any],
    split: str,
    count: int,
    dataset_seed: int,
    forbidden_examples: Sequence[OracleExample],
) -> tuple[OracleExample, ...]:
    forbidden = {example_fingerprint(example) for example in forbidden_examples}
    selected: list[OracleExample] = []
    selected_hashes: set[str] = set()
    attempt = 0
    while len(selected) < count:
        remaining = count - len(selected)
        candidate_count = remaining + max(64, remaining // 10)
        candidates = build_oracle_examples(
            stage0_config,
            split,
            candidate_count,
            dataset_seed + attempt * 1_000_000,
        )
        for example in candidates:
            fingerprint = example_fingerprint(example)
            if fingerprint in forbidden or fingerprint in selected_hashes:
                continue
            selected.append(example)
            selected_hashes.add(fingerprint)
            if len(selected) == count:
                break
        attempt += 1
        if attempt > 10:
            raise RuntimeError("could not build a disjoint oracle evaluation corpus")
    return tuple(selected)


def train_one_seed(
    oracle_config: Mapping[str, Any],
    stage0_config: Mapping[str, Any],
    sealed_examples: Sequence[OracleExample],
    seed: int,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if oracle_config["training"].get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_ORACLE_INTERPRETER")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_start = time.perf_counter()
    data_start = time.perf_counter()
    datasets = build_run_datasets(oracle_config, stage0_config, sealed_examples)
    data_seconds = time.perf_counter() - data_start

    set_deterministic_seed(seed)
    model = build_oracle_model(oracle_config).to(device)
    parameters = parameter_count(model)
    if parameters > int(oracle_config["training"]["max_parameters"]):
        raise RuntimeError(f"parameter budget exceeded: {parameters}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(oracle_config["training"]["learning_rate"]),
        weight_decay=float(oracle_config["training"]["weight_decay"]),
        betas=tuple(oracle_config["training"].get("betas", [0.9, 0.95])),
    )
    max_steps = int(oracle_config["training"]["max_steps"])
    warmup_steps = int(oracle_config["training"]["warmup_steps"])

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return max(1, step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        minimum = float(oracle_config["training"].get("minimum_lr_ratio", 0.05))
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    mixed_precision = bool(oracle_config["training"]["mixed_precision"] and device.type == "cuda")
    precision_name = str(oracle_config["training"].get("mixed_precision_dtype", "bfloat16"))
    precision_dtype = torch.bfloat16 if precision_name == "bfloat16" else torch.float16
    use_scaler = mixed_precision and precision_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=precision_dtype)
        if mixed_precision
        else nullcontext()
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    global_step = 0
    epoch = 0
    seen_examples = 0
    best_dev = -1.0
    best_train = -1.0
    best_step = 0
    evaluations_without_improvement = 0
    loss_history: list[dict[str, float | int]] = []
    evaluation_history: list[dict[str, Any]] = []
    stop_training = False
    train_start = time.perf_counter()
    deadline_seconds = float(oracle_config["training"]["max_wall_clock_hours"]) * 3600.0

    while global_step < max_steps and not stop_training:
        for cpu_batch, _ in iter_oracle_batches(
            datasets["train"],
            int(oracle_config["training"]["batch_size"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            if time.perf_counter() - train_start > deadline_seconds:
                raise RuntimeError("oracle interpreter wall-clock budget exceeded")
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(*batch.model_inputs())
                loss, loss_parts = oracle_loss(
                    logits, batch, float(oracle_config["training"]["lambda_intermediate"])
                )
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(oracle_config["training"]["gradient_clip_norm"])
            )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1
            seen_examples += batch.input_grid.shape[0]
            if global_step == 1 or global_step % int(oracle_config["training"]["log_interval_steps"]) == 0:
                loss_history.append(
                    {
                        "step": global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        **loss_parts,
                    }
                )

            should_evaluate = (
                global_step % int(oracle_config["training"]["eval_interval_steps"]) == 0
                or global_step == max_steps
            )
            if should_evaluate:
                train_metrics = evaluate_model(
                    model,
                    datasets["train_eval"],
                    int(oracle_config["evaluation"]["batch_size"]),
                    device,
                )
                dev_metrics = evaluate_model(
                    model,
                    datasets["dev"],
                    int(oracle_config["evaluation"]["batch_size"]),
                    device,
                )
                evaluation_history.append(
                    {
                        "step": global_step,
                        "train_signature_exact": train_metrics["exact"],
                        "dev_len2_exact": dev_metrics["exact"],
                    }
                )
                score = (float(dev_metrics["exact"]), float(train_metrics["exact"]))
                if score > (best_dev, best_train):
                    best_dev = float(dev_metrics["exact"])
                    best_train = float(train_metrics["exact"])
                    best_step = global_step
                    evaluations_without_improvement = 0
                    save_checkpoint(checkpoint_path, model, oracle_config, seed, global_step)
                else:
                    evaluations_without_improvement += 1
                minimum_steps = int(oracle_config["training"].get("minimum_steps", max_steps))
                patience = int(oracle_config["training"].get("early_stopping_patience", 0))
                target = float(oracle_config["training"].get("early_stopping_target_exact", 0.95))
                if (
                    global_step >= minimum_steps
                    and patience
                    and best_dev >= target
                    and evaluations_without_improvement >= patience
                ):
                    stop_training = True
                    break
            if global_step >= max_steps:
                break
        epoch += 1

    train_seconds = time.perf_counter() - train_start
    model, checkpoint_metadata = load_checkpoint(checkpoint_path, device)
    train_metrics = evaluate_model(
        model, datasets["train_eval"], int(oracle_config["evaluation"]["batch_size"]), device
    )
    dev_metrics = evaluate_model(
        model, datasets["dev"], int(oracle_config["evaluation"]["batch_size"]), device
    )
    sealed_len2 = tuple(example for example in datasets["sealed"] if example.program_length == 2)
    sealed_len3 = tuple(example for example in datasets["sealed"] if example.program_length == 3)
    sealed_len2_metrics = evaluate_model(
        model, sealed_len2, int(oracle_config["evaluation"]["batch_size"]), device
    )
    sealed_len3_metrics = evaluate_model(
        model, sealed_len3, int(oracle_config["evaluation"]["batch_size"]), device
    )
    sealed_all_metrics = evaluate_model(
        model, datasets["sealed"], int(oracle_config["evaluation"]["batch_size"]), device
    )
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    total_seconds = time.perf_counter() - total_start
    report = {
        "seed": seed,
        "parameter_count": parameters,
        "best_step": best_step,
        "checkpoint_step": checkpoint_metadata["step"],
        "steps_executed": global_step,
        "seen_examples": seen_examples,
        "data_generation_seconds": data_seconds,
        "training_seconds": train_seconds,
        "total_wall_seconds": total_seconds,
        "training_examples_per_second": seen_examples / train_seconds,
        "peak_vram_bytes": peak_vram,
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "flops_per_grid": None,
        "train_signature": train_metrics,
        "dev_len2": dev_metrics,
        "sealed_len2": sealed_len2_metrics,
        "sealed_len3": sealed_len3_metrics,
        "sealed_all": sealed_all_metrics,
        "loss_history": loss_history,
        "evaluation_history": evaluation_history,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Mapping[str, Any],
    seed: int,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "model_version": getattr(model, "model_version", "v0"),
            "seed": seed,
            "step": step,
            "oracle_config": dict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = {
        "model": payload["model_config"],
        "model_version": payload.get("model_version", "v0"),
    }
    model = build_oracle_model(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def load_stage0_for_oracle(oracle_config: Mapping[str, Any], repository_root: Path) -> dict[str, Any]:
    return load_stage0_config(repository_root / oracle_config["stage0_config"])
