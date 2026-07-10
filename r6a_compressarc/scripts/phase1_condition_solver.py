from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAX_CANVAS = 30
ARC_COLORS = 10
PROBABILITY_CAPTURE_K = 100


def install_solution_firewall() -> list[str]:
    original_open = builtins.open
    denied: list[str] = []

    def firewall_open(file, *args, **kwargs):
        path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else str(file)
        lowered = path.lower()
        if "solutions" in lowered or lowered.endswith("submission.json"):
            denied.append(path)
            raise RuntimeError(f"R6A Phase1 target firewall denied solver-side access to {path}")
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
    for idx, tensor in enumerate(tensors):
        item = tensor.detach().cpu().contiguous()
        digest.update(f"{idx}|{tuple(item.shape)}|{item.dtype}".encode())
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
    for key in sorted(state["state"]):
        digest.update(str(key).encode())
        for subkey, value in sorted(state["state"][key].items()):
            digest.update(str(subkey).encode())
            digest.update(tensor_hash(value).encode() if torch.is_tensor(value) else json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def set_rng_states(payload: dict[str, Any], device: str) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(tuple(payload["numpy_random_state"]))
    torch.random.set_rng_state(payload["torch_cpu_rng_state"])
    if device == "cuda" and torch.cuda.is_available() and payload.get("torch_cuda_rng_state") is not None:
        torch.cuda.set_rng_state(payload["torch_cuda_rng_state"])


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


def support_reconstruction_metrics(task: Any, model: Any) -> dict[str, Any]:
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
            pred = fixed_best_crop(task, logits[example_idx, :, :, :, mode_idx], x_mask[example_idx, :, mode_idx], y_mask[example_idx, :, mode_idx], [len(target), len(target[0])])
            cells = len(target) * len(target[0])
            correct = sum(int(pred[row][col] == target[row][col]) for row in range(len(target)) for col in range(len(target[0])))
            rows.append({"train_pair_index": example_idx, "mode": mode_name, "exact_match": pred == target, "cell_accuracy": correct / cells if cells else None})
            exact_count += int(pred == target)
            grid_count += 1
            total_correct += correct
            total_cells += cells
    return {"support_grid_exact_match_rate": exact_count / grid_count if grid_count else None, "support_cell_accuracy": total_correct / total_cells if total_cells else None, "per_grid": rows}


def gpu_metadata(device: str) -> dict[str, Any] | None:
    if device != "cuda" or not torch.cuda.is_available():
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
        return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout.strip()}
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}


def predictions_from_logger(task: Any, logger: Any) -> list[dict[str, Any]]:
    rows = []
    for example_num in range(task.n_test):
        most = logger.solution_most_frequent[example_num] if logger.solution_most_frequent is not None else tuple()
        second = logger.solution_second_most_frequent[example_num] if logger.solution_second_most_frequent is not None else most
        rows.append({"attempt_1": [list(row) for row in most], "attempt_2": [list(row) for row in second]})
    return rows


def canonical_probability(task: Any, current_logits: torch.Tensor) -> np.ndarray:
    # current_logits: n_test, task_color, x, y. R6A phase1 tasks have one test query.
    probs = torch.softmax(current_logits[0], dim=0).detach().cpu().numpy()
    canvas = np.zeros((MAX_CANVAS, MAX_CANVAS, ARC_COLORS), dtype=np.float32)
    for task_color_index, arc_color in enumerate(task.colors):
        if 0 <= arc_color < ARC_COLORS:
            height = min(task.n_x, MAX_CANVAS)
            width = min(task.n_y, MAX_CANVAS)
            canvas[:height, :width, arc_color] = probs[task_color_index, :height, :width]
    return canvas


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    denied_paths = install_solution_firewall()
    reference_root = Path(args.compressarc_root).resolve()
    os.chdir(reference_root)
    sys.path.insert(0, str(reference_root))

    import preprocessing
    import arc_compressor
    import solution_selection
    import train

    torch.set_default_dtype(torch.float32)
    torch.set_default_device(args.device)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    challenge_payload = json.loads(Path(args.challenge_json).read_text(encoding="utf-8"))
    problem = challenge_payload[args.task_id] if args.task_id in challenge_payload else challenge_payload
    init = torch.load(args.initialization_artifact, map_location="cpu", weights_only=False)

    task = preprocessing.Task(args.task_id, problem, None)
    model = arc_compressor.ARCCompressor(task)
    for dest, src in zip(model.weights_list, init["model_weights"]):
        dest.data.copy_(src.to(dest.device))
    optimizer = torch.optim.Adam(model.weights_list, lr=init["config"]["learning_rate"], betas=tuple(init["config"]["betas"]))
    optimizer.load_state_dict(init["optimizer_state_dict"])
    set_rng_states(init["rng_state_after_initialization"], args.device)
    logger = solution_selection.Logger(task)

    init_model_hash = tensor_list_hash(model.weights_list)
    init_optimizer_hash = optimizer_state_hash(optimizer)
    probability_snapshots = []
    probability_steps = []
    trajectory = []
    start = time.perf_counter()
    failure = None
    try:
        for train_step in range(args.steps):
            train.take_step(task, model, optimizer, train_step, logger)
            step_count = train_step + 1
            if step_count > args.steps - PROBABILITY_CAPTURE_K:
                probability_steps.append(step_count)
                probability_snapshots.append(canonical_probability(task, logger.current_logits))
            if step_count == 1 or step_count % args.log_interval == 0 or step_count == args.steps:
                trajectory.append({"step": step_count, "loss": logger.loss_curve[-1], "total_KL": logger.total_KL_curve[-1], "reconstruction_error": logger.reconstruction_error_curve[-1], "model_parameter_state_hash": tensor_list_hash(model.weights_list)})
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "completed_steps": len(logger.loss_curve)}

    probability_tensor = np.stack(probability_snapshots, axis=0) if probability_snapshots else np.zeros((0, MAX_CANVAS, MAX_CANVAS, ARC_COLORS), dtype=np.float32)
    Path(args.probability_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probability_npz,
        probability=probability_tensor,
        checkpoint_steps=np.array(probability_steps, dtype=np.int64),
        active_mask=np.pad(np.ones((min(task.n_x, MAX_CANVAS), min(task.n_y, MAX_CANVAS)), dtype=np.bool_), ((0, MAX_CANVAS - min(task.n_x, MAX_CANVAS)), (0, MAX_CANVAS - min(task.n_y, MAX_CANVAS))), constant_values=False),
        task_colors=np.array(task.colors, dtype=np.int64),
        task_n_x=np.array(task.n_x, dtype=np.int64),
        task_n_y=np.array(task.n_y, dtype=np.int64),
    )
    predictions = predictions_from_logger(task, logger)
    stable_log = {"task_id": args.task_id, "condition": args.condition, "seed": init["seed"], "actual_step_count": len(logger.loss_curve), "trajectory": trajectory, "predictions": predictions, "failure": failure}
    payload = {
        "experiment": "R6A_PHASE1_MULTIREPEAT_PAIRING_AUDIT",
        "task_id": args.task_id,
        "stratum": args.stratum,
        "condition": args.condition,
        "seed": init["seed"],
        "steps_requested": args.steps,
        "actual_step_count": len(logger.loss_curve),
        "initialization_artifact": str(args.initialization_artifact),
        "initial_model_hash_loaded": init_model_hash,
        "initial_optimizer_hash_loaded": init_optimizer_hash,
        "saved_initial_model_hash": init["model_state_hash"],
        "saved_initial_optimizer_hash": init["optimizer_state_hash"],
        "loaded_initialization_matches_saved": init_model_hash == init["model_state_hash"] and init_optimizer_hash == init["optimizer_state_hash"],
        "rng_state_reset_from_initialization_artifact": True,
        "learning_rate": init["config"]["learning_rate"],
        "optimizer": {"name": "Adam", "betas": init["config"]["betas"]},
        "runtime_seconds": time.perf_counter() - start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" and torch.cuda.is_available() else None,
        "final_objective": logger.loss_curve[-1] if logger.loss_curve else None,
        "final_KL_term": logger.total_KL_curve[-1] if logger.total_KL_curve else None,
        "final_reconstruction_term": logger.reconstruction_error_curve[-1] if logger.reconstruction_error_curve else None,
        "final_regularization_terms": {},
        "support_reconstruction": None,
        "support_reconstruction_note": "Not recomputed in Phase1 to preserve the no-extra-forward-pass probability-capture discipline; official training-step logs remain unchanged.",
        "trajectory": trajectory,
        "predictions": predictions,
        "final_prediction_hash": json_hash(predictions),
        "probability_npz": str(args.probability_npz),
        "probability_capture": {"checkpoint_count": int(probability_tensor.shape[0]), "checkpoint_steps": probability_steps, "shape": list(probability_tensor.shape), "uses_hidden_target": False},
        "gpu_metadata": gpu_metadata(args.device),
        "solver_process_firewall": {"challenge_json_only": True, "solution_argument_present": False, "denied_solution_paths": denied_paths},
        "structured_log_hash_excluding_wall_clock": json_hash(stable_log),
        "failure": failure,
        "status": "RUN_COMPLETED" if failure is None and len(logger.loss_curve) == args.steps else "RUN_FAILED",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one R6A phase1 condition with frozen initialization and probability capture.")
    parser.add_argument("--compressarc-root", required=True)
    parser.add_argument("--challenge-json", required=True)
    parser.add_argument("--initialization-artifact", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--stratum", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--probability-npz", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    try:
        payload = run_condition(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "RUN_COMPLETED" else 2
    except Exception as exc:
        payload = {"experiment": "R6A_PHASE1_MULTIREPEAT_PAIRING_AUDIT", "condition": args.condition, "task_id": args.task_id, "status": "RUN_FAILED", "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
