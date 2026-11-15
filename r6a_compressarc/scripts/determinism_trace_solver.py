from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


def install_solution_firewall() -> list[str]:
    original_open = builtins.open
    denied: list[str] = []

    def firewall_open(file, *args, **kwargs):
        path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else str(file)
        lowered = path.lower()
        if "solutions" in lowered or lowered.endswith("submission.json"):
            denied.append(path)
            raise RuntimeError(f"R6A target firewall denied solver-side access to {path}")
        return original_open(file, *args, **kwargs)

    builtins.open = firewall_open
    return denied


def json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    item = tensor.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(item.shape)}|{item.dtype}".encode() + b"\0" + item.numpy().tobytes()).hexdigest()


def tensor_list_hash(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for index, tensor in enumerate(tensors):
        item = tensor.detach().cpu().contiguous()
        digest.update(f"{index}|{tuple(item.shape)}|{item.dtype}".encode())
        digest.update(b"\0")
        digest.update(item.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    state = optimizer.state_dict()
    digest = hashlib.sha256()
    for group in state["param_groups"]:
        comparable = {key: value for key, value in group.items() if key != "params"}
        comparable["param_count"] = len(group["params"])
        digest.update(json.dumps(comparable, sort_keys=True, default=str).encode())
    for param_id in sorted(state["state"]):
        digest.update(str(param_id).encode())
        param_state = state["state"][param_id]
        for key in sorted(param_state):
            value = param_state[key]
            digest.update(str(key).encode())
            if torch.is_tensor(value):
                digest.update(tensor_hash(value).encode())
            else:
                digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def rng_state_payload() -> dict[str, Any]:
    payload = {
        "python_random_state_hash": json_hash(repr(random.getstate())),
        "numpy_random_state_hash": json_hash(repr(np.random.get_state())),
        "torch_cpu_rng_state_hash": tensor_hash(torch.random.get_rng_state()),
        "torch_cuda_rng_state_hash": None,
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_hash"] = tensor_hash(torch.cuda.get_rng_state())
    return payload


def fixed_best_crop(task: Any, grid_logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor, shape: list[int]) -> list[list[int]]:
    colors = torch.argmax(grid_logits, dim=0)

    def best_start(mask: torch.Tensor, length: int) -> int:
        scores = []
        for offset in range(mask.shape[0] - length + 1):
            scores.append(-torch.sum(mask[:offset]) + torch.sum(mask[offset : offset + length]) - torch.sum(mask[offset + length :]))
        return int(torch.argmax(torch.stack(scores)).item())

    height, width = int(shape[0]), int(shape[1])
    row0 = best_start(x_mask, height)
    col0 = best_start(y_mask, width)
    crop = colors[row0 : row0 + height, col0 : col0 + width].detach().cpu().numpy().tolist()
    return [[task.colors[int(value)] for value in row] for row in crop]


def current_predictions(task: Any, logger: Any) -> list[dict[str, Any]]:
    rows = []
    for example_num in range(task.n_test):
        most = logger.solution_most_frequent[example_num] if logger.solution_most_frequent is not None else tuple()
        second = logger.solution_second_most_frequent[example_num] if logger.solution_second_most_frequent is not None else most
        rows.append({"attempt_1": [list(row) for row in most], "attempt_2": [list(row) for row in second]})
    return rows


def compute_loss_components(task: Any, model: Any, train_step: int) -> dict[str, Any]:
    import train

    with torch.no_grad():
        logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
        logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
        total_KL = sum(torch.sum(item) for item in KL_amounts)
        reconstruction_error = 0
        for example_num in range(task.n_examples):
            for in_out_mode in range(2):
                if example_num >= task.n_train and in_out_mode == 1:
                    continue
                grid_size_uncertain = not (
                    task.in_out_same_size
                    or task.all_out_same_size
                    and in_out_mode == 1
                    or task.all_in_same_size
                    and in_out_mode == 0
                )
                coefficient = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
                logits_slice = logits[example_num, :, :, :, in_out_mode]
                problem_slice = task.problem[example_num, :, :, in_out_mode]
                output_shape = task.shapes[example_num][in_out_mode]
                x_log_partition, x_logprobs = train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], output_shape[0])
                y_log_partition, y_logprobs = train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], output_shape[1])
                if grid_size_uncertain:
                    x_log_partitions = []
                    y_log_partitions = []
                    for length in range(1, x_mask.shape[1] + 1):
                        x_log_partitions.append(train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], length)[0])
                    for length in range(1, y_mask.shape[1] + 1):
                        y_log_partitions.append(train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], length)[0])
                    x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                    y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)
                logprobs = [[] for _ in range(x_logprobs.shape[0])]
                for x_offset in range(x_logprobs.shape[0]):
                    for y_offset in range(y_logprobs.shape[0]):
                        logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                        logits_crop = logits_slice[:, x_offset : x_offset + output_shape[0], y_offset : y_offset + output_shape[1]]
                        target_crop = problem_slice[: output_shape[0], : output_shape[1]]
                        logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None, ...], target_crop[None, ...], reduction="sum")
                        logprobs[x_offset].append(logprob)
                logprobs = torch.stack([torch.stack(items, dim=0) for items in logprobs], dim=0)
                coefficient = 0.1 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
                logprob = torch.logsumexp(coefficient * logprobs, dim=(0, 1)) / coefficient
                reconstruction_error = reconstruction_error - logprob
        loss = total_KL + 10 * reconstruction_error
    return {
        "loss": float(loss.detach().cpu().item()),
        "total_KL": float(total_KL.detach().cpu().item()),
        "reconstruction_error": float(reconstruction_error.detach().cpu().item()),
        "KL_names": list(KL_names),
        "KL_amount_hashes": [tensor_hash(item) for item in KL_amounts],
    }


def apply_strict_config() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    if args.strict_deterministic:
        apply_strict_config()
    denied_paths = install_solution_firewall()
    reference_root = Path(args.compressarc_root).resolve()
    os.chdir(reference_root)
    sys.path.insert(0, str(reference_root))

    import preprocessing
    import arc_compressor
    import solution_selection
    import train

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(args.device)

    challenge_payload = json.loads(Path(args.challenge_json).read_text(encoding="utf-8"))
    problem = challenge_payload[args.task_id] if args.task_id in challenge_payload else challenge_payload
    rng_before_task_model = rng_state_payload()

    task = preprocessing.Task(args.task_id, problem, None)
    model = arc_compressor.ARCCompressor(task)
    model_initial_hash = tensor_list_hash(model.weights_list)
    optimizer = torch.optim.Adam(model.weights_list, lr=args.learning_rate, betas=(args.beta1, args.beta2))
    optimizer_initial_hash = optimizer_state_hash(optimizer)
    logger = solution_selection.Logger(task)
    pre_update_step0 = compute_loss_components(task, model, 0)
    checkpoints = [
        {
            "checkpoint": "pre_update_step_0",
            "step": 0,
            "loss": pre_update_step0["loss"],
            "total_KL": pre_update_step0["total_KL"],
            "reconstruction_error": pre_update_step0["reconstruction_error"],
            "model_parameter_state_hash": model_initial_hash,
            "optimizer_state_hash": optimizer_initial_hash,
        }
    ]
    start = time.perf_counter()
    failure: dict[str, Any] | None = None
    try:
        for train_step in range(args.steps):
            train.take_step(task, model, optimizer, train_step, logger)
            step_count = train_step + 1
            if step_count == 1 or step_count % args.trace_interval == 0 or step_count == args.steps:
                checkpoints.append(
                    {
                        "checkpoint": f"after_optimizer_step_{step_count}",
                        "step": step_count,
                        "loss": logger.loss_curve[-1],
                        "total_KL": logger.total_KL_curve[-1],
                        "reconstruction_error": logger.reconstruction_error_curve[-1],
                        "model_parameter_state_hash": tensor_list_hash(model.weights_list),
                        "optimizer_state_hash": optimizer_state_hash(optimizer),
                        "rng_state": rng_state_payload() if step_count == 1 else None,
                    }
                )
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "occurred_after_completed_steps": len(logger.loss_curve),
        }

    predictions = current_predictions(task, logger) if logger.solution_most_frequent is not None else []
    stable_log = {
        "task_id": args.task_id,
        "condition": args.condition,
        "seed": args.seed,
        "steps_requested": args.steps,
        "actual_step_count": len(logger.loss_curve),
        "failure": failure,
        "checkpoints": checkpoints,
        "predictions": predictions,
    }
    payload = {
        "experiment": "R6A_0C_DETERMINISM_TRIAGE",
        "run_label": args.run_label,
        "task_id": args.task_id,
        "condition": args.condition,
        "seed": args.seed,
        "steps_requested": args.steps,
        "actual_step_count": len(logger.loss_curve),
        "trace_interval": args.trace_interval,
        "device": args.device,
        "strict_deterministic_requested": args.strict_deterministic,
        "torch_are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_backends_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_backends_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_backends_cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_backends_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "env": {key: os.environ.get(key) for key in ["PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"]},
        "solver_process_firewall": {
            "challenge_json_only": True,
            "solution_argument_present": False,
            "denied_solution_paths": denied_paths,
        },
        "rng_state_immediately_before_task_model_construction": rng_before_task_model,
        "model_parameter_state_hash_after_initialization": model_initial_hash,
        "optimizer_state_hash_after_construction": optimizer_initial_hash,
        "pre_update_forward_loss_step_0": pre_update_step0,
        "checkpoints": checkpoints,
        "final_parameter_state_hash": tensor_list_hash(model.weights_list),
        "final_predicted_query_grid_hash": json_hash(predictions),
        "predictions": predictions,
        "structured_log_hash_excluding_wall_clock": json_hash(stable_log),
        "runtime_seconds": time.perf_counter() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() and args.device == "cuda" else None,
        "failure": failure,
        "status": "TRACE_COMPLETED" if failure is None else "TRACE_FAILED",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace CompressARC determinism checkpoints without target access.")
    parser.add_argument("--compressarc-root", required=True)
    parser.add_argument("--challenge-json", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--condition", default="FULL")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--trace-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        payload = run_trace(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "TRACE_COMPLETED" else 2
    except Exception as exc:
        payload = {
            "experiment": "R6A_0C_DETERMINISM_TRIAGE",
            "run_label": args.run_label,
            "status": "TRACE_FAILED",
            "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "occurred_before_trace_payload": True},
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

