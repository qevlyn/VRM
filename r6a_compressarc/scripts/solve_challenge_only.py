from __future__ import annotations

import argparse
import builtins
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def install_solution_firewall() -> dict[str, Any]:
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
    return {"denied_paths": denied, "original_open": original_open}


def fixed_best_crop(task: Any, grid_logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor, shape: list[int]) -> list[list[int]]:
    colors = torch.argmax(grid_logits, dim=0)

    def best_start(mask: torch.Tensor, length: int) -> int:
        scores = []
        for offset in range(mask.shape[0] - length + 1):
            scores.append(-torch.sum(mask[:offset]) + torch.sum(mask[offset : offset + length]) - torch.sum(mask[offset + length :]))
        return int(torch.argmax(torch.stack(scores)).item())

    h, w = int(shape[0]), int(shape[1])
    row0 = best_start(x_mask, h)
    col0 = best_start(y_mask, w)
    crop = colors[row0 : row0 + h, col0 : col0 + w].detach().cpu().numpy().tolist()
    return [[task.colors[int(value)] for value in row] for row in crop]


def support_reconstruction_metrics(task: Any, model: Any) -> dict[str, Any]:
    import train

    with torch.no_grad():
        logits, x_mask, y_mask, _, _ = model.forward()
        logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    rows = []
    total_correct = 0
    total_cells = 0
    exact_count = 0
    grid_count = 0
    for example_idx, pair in enumerate(task.unprocessed_problem["train"]):
        for mode_idx, mode_name in enumerate(("input", "output")):
            target = pair[mode_name]
            pred = fixed_best_crop(
                task,
                logits[example_idx, :, :, :, mode_idx],
                x_mask[example_idx, :, mode_idx],
                y_mask[example_idx, :, mode_idx],
                [len(target), len(target[0])],
            )
            correct = sum(
                int(pred[row][col] == target[row][col])
                for row in range(len(target))
                for col in range(len(target[0]))
            )
            cells = len(target) * len(target[0])
            rows.append(
                {
                    "train_pair_index": example_idx,
                    "mode": mode_name,
                    "exact_match": pred == target,
                    "cell_accuracy": correct / cells if cells else None,
                }
            )
            exact_count += int(pred == target)
            grid_count += 1
            total_correct += correct
            total_cells += cells
    return {
        "support_grid_exact_match_rate": exact_count / grid_count if grid_count else None,
        "support_cell_accuracy": total_correct / total_cells if total_cells else None,
        "per_grid": rows,
    }


def run_solver(args: argparse.Namespace) -> dict[str, Any]:
    firewall = install_solution_firewall()
    reference_root = Path(args.compressarc_root).resolve()
    os.chdir(reference_root)
    sys.path.insert(0, str(reference_root))

    import preprocessing
    import arc_compressor
    import solution_selection
    import train

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    challenge_payload = json.loads(Path(args.challenge_json).read_text(encoding="utf-8"))
    problem = challenge_payload[args.task_id] if args.task_id in challenge_payload else challenge_payload
    task = preprocessing.Task(args.task_id, problem, None)
    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=args.learning_rate, betas=(args.beta1, args.beta2))
    logger = solution_selection.Logger(task)

    trajectory = []
    start = time.perf_counter()
    stopping_reason = "budget_exhausted"
    for train_step in range(args.steps):
        train.take_step(task, model, optimizer, train_step, logger)
        if train_step == 0 or (train_step + 1) % args.log_interval == 0 or train_step + 1 == args.steps:
            trajectory.append(
                {
                    "step": train_step + 1,
                    "total_KL": logger.total_KL_curve[-1],
                    "reconstruction_error": logger.reconstruction_error_curve[-1],
                    "loss": logger.loss_curve[-1],
                }
            )
        if args.time_limit_seconds and time.perf_counter() - start > args.time_limit_seconds:
            stopping_reason = "time_limit"
            break
    runtime = time.perf_counter() - start

    predictions = []
    for example_num in range(task.n_test):
        most = logger.solution_most_frequent[example_num] if logger.solution_most_frequent is not None else tuple()
        second = logger.solution_second_most_frequent[example_num] if logger.solution_second_most_frequent is not None else most
        predictions.append(
            {
                "attempt_1": [list(row) for row in most],
                "attempt_2": [list(row) for row in second],
            }
        )

    support_metrics = support_reconstruction_metrics(task, model)
    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    payload = {
        "experiment": "R6A_COMPRESSARC_REFERENCE_SEMANTICS_AUDIT",
        "solver_process_firewall": {
            "challenge_json_only": True,
            "solution_argument_present": False,
            "denied_solution_paths": firewall["denied_paths"],
        },
        "task_id": args.task_id,
        "condition": args.condition,
        "seed": args.seed,
        "steps_requested": args.steps,
        "actual_step_count": len(logger.loss_curve),
        "stopping_reason": stopping_reason,
        "learning_rate": args.learning_rate,
        "optimizer": {"name": "Adam", "betas": [args.beta1, args.beta2]},
        "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": peak_gpu,
        "final_objective": logger.loss_curve[-1] if logger.loss_curve else None,
        "final_KL_term": logger.total_KL_curve[-1] if logger.total_KL_curve else None,
        "final_reconstruction_term": logger.reconstruction_error_curve[-1] if logger.reconstruction_error_curve else None,
        "final_MDL_or_code_length_term": logger.total_KL_curve[-1] if logger.total_KL_curve else None,
        "final_regularization_terms": {},
        "trajectory": trajectory,
        "KL_curves": logger.KL_curves,
        "support_reconstruction": support_metrics,
        "predictions": predictions,
        "solver_side_log_digest": None,
    }
    stable = json.dumps({k: v for k, v in payload.items() if k != "runtime_seconds"}, sort_keys=True).encode()
    import hashlib

    payload["solver_side_log_digest"] = hashlib.sha256(stable).hexdigest()
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official CompressARC on challenge JSON only.")
    parser.add_argument("--compressarc-root", required=True)
    parser.add_argument("--challenge-json", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", default="FULL")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--time-limit-seconds", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run_solver(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

