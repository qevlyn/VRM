from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, SMOKE_BUDGET, challenge_for_condition, read_json, write_json, write_text

R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts"
SEEDS = [20260702, 20260703, 20260704]
STRATUM_B_TASKS = ["0dfd9992", "1c786137", "1f876c06"]
CONDITIONS = ["F1", "S", "F2"]
MANIFEST_PATH = R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def padded_hamming(a: list[list[int]], b: list[list[int]]) -> int:
    height = max(len(a), len(b))
    width = max(len(a[0]) if a else 0, len(b[0]) if b else 0)
    total = 0
    for r in range(height):
        for c in range(width):
            av = a[r][c] if r < len(a) and c < len(a[0]) else -1
            bv = b[r][c] if r < len(b) and c < len(b[0]) else -1
            total += int(av != bv)
    return total


def derangements(n: int) -> list[tuple[int, ...]]:
    import itertools

    return [perm for perm in itertools.permutations(range(n)) if all(perm[i] != i for i in range(n))]


def select_swap(problem: dict[str, Any]) -> dict[str, Any]:
    outputs = [pair["output"] for pair in problem["train"]]
    candidates = []
    for perm in derangements(len(outputs)):
        score = sum(padded_hamming(outputs[i], outputs[perm[i]]) for i in range(len(outputs)))
        candidates.append({"permutation": list(perm), "total_padded_hamming": score})
    candidates.sort(key=lambda row: (-row["total_padded_hamming"], row["permutation"]))
    selected = candidates[0] if candidates else None
    eligible = selected is not None and selected["total_padded_hamming"] > 0
    return {
        "eligible": eligible,
        "eligibility_label": "PAIRING_AUDIT_TASK_ELIGIBLE" if eligible else "PAIRING_AUDIT_TASK_INELIGIBLE",
        "selected_permutation": selected["permutation"] if selected else None,
        "selected_total_padded_hamming": selected["total_padded_hamming"] if selected else None,
        "candidates": candidates,
        "output_shapes": [[len(grid), len(grid[0]) if grid else 0] for grid in outputs],
        "reason": None if eligible else "no output-changing derangement",
    }


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True).strip()


def tensor_list_hash(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for idx, tensor in enumerate(tensors):
        item = tensor.detach().cpu().contiguous()
        h.update(f"{idx}|{tuple(item.shape)}|{item.dtype}".encode())
        h.update(b"\0")
        h.update(item.numpy().tobytes())
        h.update(b"\0")
    return h.hexdigest()


def optimizer_state_hash(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for group in state["param_groups"]:
        comparable = {key: value for key, value in group.items() if key != "params"}
        comparable["param_count"] = len(group["params"])
        digest.update(json.dumps(comparable, sort_keys=True, default=str).encode())
    for key in sorted(state["state"]):
        digest.update(str(key).encode())
        for subkey, value in sorted(state["state"][key].items()):
            digest.update(str(subkey).encode())
            if torch.is_tensor(value):
                item = value.detach().cpu().contiguous()
                digest.update(f"{tuple(item.shape)}|{item.dtype}".encode())
                digest.update(b"\0")
                digest.update(item.numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def create_initialization_artifact(task_id: str, stratum: str, seed: int, problem: dict[str, Any], output: Path, *, device: str = "cuda") -> dict[str, Any]:
    if output.exists():
        meta = torch.load(output, map_location="cpu", weights_only=False)
        return {"path": str(output), "model_state_hash": meta["model_state_hash"], "optimizer_state_hash": meta["optimizer_state_hash"], "rng_state_hash": meta["rng_state_hash"], "reused": True}
    sys.path.insert(0, str(REFERENCE_ROOT))
    import preprocessing
    import arc_compressor

    os.chdir(REFERENCE_ROOT)
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    actual_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(actual_device)
    task = preprocessing.Task(task_id, problem, None)
    model = arc_compressor.ARCCompressor(task)
    optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
    rng_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    model_weights = [tensor.detach().cpu().clone() for tensor in model.weights_list]
    optimizer_state = optimizer.state_dict()
    model_hash = tensor_list_hash(model.weights_list)
    optimizer_hash = optimizer_state_hash(optimizer_state)
    rng_hash = hashlib.sha256(json.dumps({"python": repr(rng_state["python_random_state"]), "numpy": repr(rng_state["numpy_random_state"]), "torch_cpu": str(rng_state["torch_cpu_rng_state"].tolist()[:32]), "torch_cuda": str(rng_state["torch_cuda_rng_state"].tolist()[:32]) if rng_state["torch_cuda_rng_state"] is not None else None}, sort_keys=True).encode()).hexdigest()
    payload = {"experiment": "R6A_PHASE1_MULTIREPEAT_PAIRING_AUDIT", "task_id": task_id, "stratum": stratum, "seed": seed, "model_weights": model_weights, "optimizer_state_dict": optimizer_state, "rng_state_after_initialization": rng_state, "model_state_hash": model_hash, "optimizer_state_hash": optimizer_hash, "rng_state_hash": rng_hash, "config": {"learning_rate": 0.01, "betas": [0.5, 0.9], "budget": SMOKE_BUDGET, "initialization_device": actual_device}}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {"path": str(output), "model_state_hash": model_hash, "optimizer_state_hash": optimizer_hash, "rng_state_hash": rng_hash, "reused": False}


def build_phase1_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        existing = read_json(MANIFEST_PATH)
        if existing.get("experiment") == "R6A_PHASE1_MULTIREPEAT_PAIRING_AUDIT" and existing.get("seeds") == SEEDS:
            return existing
    old = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    if len(old.get("selected_task_ids", [])) != 6:
        raise RuntimeError("old phase1 manifest does not contain exactly six selected task IDs")
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    env = read_json(R6 / "manifests" / "determinism_environment.json")
    tasks = []
    authority_ok = True
    for task_id in old["selected_task_ids"]:
        recomputed = select_swap(eval_challenges[task_id])
        old_meta = old["task_metadata"][task_id]
        old_perm = old_meta.get("b1_selected_permutation")
        old_score = None
        if old_perm is not None:
            outputs = [pair["output"] for pair in eval_challenges[task_id]["train"]]
            old_score = sum(padded_hamming(outputs[i], outputs[old_perm[i]]) for i in range(len(outputs)))
        match = old_perm == recomputed["selected_permutation"] and old_score == recomputed["selected_total_padded_hamming"]
        authority_ok = authority_ok and match
        tasks.append({"task_id": task_id, "split": "ARC_AGI_1_EVALUATION_SPLIT", "stratum": "A", "role": "PRIMARY_EVALUATION_PILOT", "old_manifest_permutation": old_perm, "old_manifest_padded_hamming_score": old_score, "recomputed": recomputed, "permutation_authority_match": match})
    for task_id in STRATUM_B_TASKS:
        tasks.append({"task_id": task_id, "split": "ARC_AGI_1_TRAINING_SPLIT", "stratum": "B", "role": "AUTHOR_CAPABILITY_POSITIVE_CONTROL;NOT_GENERALIZATION_EVIDENCE;ARC_AGI_1_TRAINING_SPLIT", "old_manifest_permutation": None, "old_manifest_padded_hamming_score": None, "recomputed": select_swap(train_challenges[task_id]), "permutation_authority_match": None})

    eligible_a = sum(1 for row in tasks if row["stratum"] == "A" and row["recomputed"]["eligible"])
    eligible_b = sum(1 for row in tasks if row["stratum"] == "B" and row["recomputed"]["eligible"])
    if eligible_a == 6:
        threshold_a = {"eligible_tasks": 6, "cells": 18, "positive_delta_threshold": 13, "delta_gt_nu_threshold": 7, "positive_task_mean_threshold": 5}
    elif eligible_a == 5:
        threshold_a = {"eligible_tasks": 5, "cells": 15, "positive_delta_threshold": 11, "delta_gt_nu_threshold": 6, "positive_task_mean_threshold": 4}
    else:
        threshold_a = {"eligible_tasks": eligible_a, "cells": eligible_a * 3, "insufficient_coverage": True}
    threshold_b = {"eligible_tasks": eligible_b, "cells": eligible_b * 3, "positive_delta_threshold": 7, "delta_gt_nu_threshold": 4, "positive_task_mean_threshold": 3, "positive_control_valid": eligible_b == 3}
    run_order = []
    for row in tasks:
        if not row["recomputed"]["eligible"]:
            continue
        for seed in SEEDS:
            for condition in CONDITIONS:
                run_order.append({"task_id": row["task_id"], "stratum": row["stratum"], "seed": seed, "condition": condition})
    init_root = R6 / "artifacts" / "phase1_initializations"
    init_paths = {f"{row['task_id']}:{seed}": str(init_root / row["task_id"] / str(seed) / "initialization.pt") for row in tasks for seed in SEEDS}
    manifest = {"experiment": "R6A_PHASE1_MULTIREPEAT_PAIRING_AUDIT", "created_utc": now(), "source_commit": git_commit(), "environment_label": env["environment_label"], "seeds": SEEDS, "optimization_budget": SMOKE_BUDGET, "probability_capture_final_k": 100, "primary_metric": "evaluator-side target-cell NLL from average of final 100 query probability tensors; delta = ell_S - (ell_F1 + ell_F2)/2; nu = abs(ell_F1 - ell_F2)", "verdict_thresholds": {"stratum_a": threshold_a, "stratum_b": threshold_b}, "old_manifest_path": str(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json"), "tasks": tasks, "permutation_authority_all_stratum_a_match": authority_ok, "run_order": run_order, "initialization_artifact_paths": init_paths, "manifest_becomes_authoritative": authority_ok}
    write_json(MANIFEST_PATH, manifest)
    return manifest


def write_challenge(task_id: str, problem: dict[str, Any], condition: str, permutation: list[int] | None, path: Path) -> Path:
    phase_condition = "FULL" if condition in {"F1", "F2"} else "ALL_PAIRING_SWAP"
    payload = {task_id: challenge_for_condition(problem, phase_condition, permutation)}
    write_json(path, payload)
    return path


def solver_command(task_id: str, stratum: str, condition: str, challenge: Path, init_artifact: Path, out_json: Path, prob_npz: Path, device: str, *, steps: int = SMOKE_BUDGET) -> list[str]:
    return [sys.executable, str(R6 / "scripts" / "phase1_condition_solver.py"), "--compressarc-root", str(REFERENCE_ROOT), "--challenge-json", str(challenge), "--initialization-artifact", str(init_artifact), "--task-id", task_id, "--stratum", stratum, "--condition", condition, "--steps", str(steps), "--output-json", str(out_json), "--probability-npz", str(prob_npz), "--device", device]


def run_command(cmd: list[str], timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=merged, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-6000:], "runtime_seconds": time.perf_counter() - start}


def evaluator_metrics(task_id: str, split: str, condition_payload: dict[str, Any], challenge_problem: dict[str, Any], solution_grids: list[list[list[int]]], prob_npz: Path) -> dict[str, Any]:
    from r6a_common import evaluate_prediction

    prediction_grids = [row["attempt_1"] for row in condition_payload["predictions"]]
    metrics = evaluate_prediction(challenge_problem=challenge_problem, solution_grids=solution_grids, prediction_grids=prediction_grids)
    data = np.load(prob_npz)
    probability = data["probability"]
    mean_prob = probability.mean(axis=0)
    nll_sum = 0.0
    count = 0
    for target in solution_grids:
        for r, row in enumerate(target):
            for c, color in enumerate(row):
                p = float(mean_prob[r, c, color]) if r < mean_prob.shape[0] and c < mean_prob.shape[1] else 0.0
                nll_sum += -float(np.log(max(p, 1e-12)))
                count += 1
    metrics["target_cell_nll_from_final100_mean_probability"] = nll_sum / count if count else None
    metrics["probability_checkpoint_count"] = int(probability.shape[0])
    metrics["probability_note"] = "For cells lying inside the evaluator-known target shape but outside a model-active or semantically meaningful padded region, the stored padded probability may not carry an interpretable spatial meaning. This does not alter the paired comparison because F1, S, and F2 use the identical canonical probability representation and evaluator rule."
    return metrics


def confirm_firewall(manifest: dict[str, Any]) -> dict[str, Any]:
    task_id = "903d1b4a"
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    neg_root = R6 / "artifacts" / "phase1_firewall_cpu_negative"
    cpu_init = neg_root / "initialization.pt"
    create_initialization_artifact(task_id, "FIREWALL_CPU_NEGATIVE", SEEDS[0], eval_challenges[task_id], cpu_init, device="cpu")
    challenge_path = write_challenge(task_id, eval_challenges[task_id], "F1", None, neg_root / "challenge.json")
    solution_path = REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_solutions.json"
    static = {
        "solver_working_directory": str(PROJECT_ROOT),
        "mounted_or_copied_files": [str(challenge_path), str(cpu_init)],
        "solver_input_paths": [str(challenge_path), str(cpu_init)],
        "evaluator_only_solution_paths": [str(solution_path)],
        "solution_json_path_absent_from_solver_inputs": str(solution_path) not in {str(challenge_path), str(cpu_init)},
        "data_path_environment_variables": {k: v for k, v in os.environ.items() if "ARC" in k or "DATA" in k or "SOLUTION" in k or "CHALLENGE" in k},
    }
    static["solution_json_path_absent_from_data_path_env_vars"] = all(str(solution_path) not in str(value) for value in static["data_path_environment_variables"].values())
    test_input = eval_challenges[task_id]["test"][0]["input"]
    dummy_a = [[0 for _ in row] for row in test_input]
    dummy_b = [[1 for _ in row] for row in test_input]
    write_json(neg_root / "dummy_targets_evaluator_only.json", {"dummy_a": [dummy_a], "dummy_b": [dummy_b]})
    runs = {}
    for label in ["dummy_a", "dummy_b"]:
        out_json = neg_root / f"{label}_solver.json"
        prob = neg_root / f"{label}_probability.npz"
        cmd = solver_command(task_id, "FIREWALL_CPU_NEGATIVE", "FIREWALL_CPU_PREFIX", challenge_path, cpu_init, out_json, prob, "cpu", steps=20)
        if not out_json.exists():
            runs[label] = run_command(cmd, timeout=3600)
        else:
            runs[label] = {"cmd": cmd, "returncode": 0, "stdout": "reused", "runtime_seconds": 0.0}
    a = read_json(neg_root / "dummy_a_solver.json")
    b = read_json(neg_root / "dummy_b_solver.json")
    comparable = ["trajectory", "predictions", "actual_step_count", "structured_log_hash_excluding_wall_clock", "final_prediction_hash"]
    solver_commands = [runs[label].get("cmd") for label in ["dummy_a", "dummy_b"]]
    static["solver_commands"] = solver_commands
    static["solution_json_path_absent_from_solver_commands"] = all(str(solution_path) not in " ".join(cmd or []) for cmd in solver_commands)
    byte_identical = all(a.get(key) == b.get(key) for key in comparable)
    probability_a = np.load(neg_root / "dummy_a_probability.npz")["probability"]
    probability_b = np.load(neg_root / "dummy_b_probability.npz")["probability"]
    probability_identical = bool(np.array_equal(probability_a, probability_b))
    challenge_problem = read_json(challenge_path)[task_id]
    evaluator = {
        "dummy_a": evaluator_metrics(task_id, "A", a, challenge_problem, [dummy_a], neg_root / "dummy_a_probability.npz"),
        "dummy_b": evaluator_metrics(task_id, "A", b, challenge_problem, [dummy_b], neg_root / "dummy_b_probability.npz"),
    }
    evaluator_metrics_differ = evaluator["dummy_a"]["target_cell_nll_from_final100_mean_probability"] != evaluator["dummy_b"]["target_cell_nll_from_final100_mean_probability"]
    payload = {
        "static_data_flow_isolation": static,
        "cpu_dummy_target_negative_control": {
            "runs": runs,
            "fixed_prefix_steps": 20,
            "byte_identical_solver_traces": byte_identical,
            "byte_identical_probability_tensors": probability_identical,
            "compared_fields": comparable,
            "solver_did_not_receive_dummy_targets": True,
            "evaluator_metrics": evaluator,
            "evaluator_metrics_differ": evaluator_metrics_differ,
        },
        "firewall_preflight_passed": static["solution_json_path_absent_from_solver_inputs"] and static["solution_json_path_absent_from_solver_commands"] and static["solution_json_path_absent_from_data_path_env_vars"] and byte_identical and probability_identical,
    }
    lines = ["# R6A Phase1 Firewall Confirmation", "", f"- firewall preflight passed: {payload['firewall_preflight_passed']}", f"- solver working directory: {static['solver_working_directory']}", f"- solver input paths: {static['solver_input_paths']}", f"- solver commands: `{json.dumps(solver_commands)}`", f"- solution JSON path absent from solver inputs: {static['solution_json_path_absent_from_solver_inputs']}", f"- solution JSON path absent from solver commands: {static['solution_json_path_absent_from_solver_commands']}", f"- CPU dummy-target negative control byte-identical: {byte_identical}", f"- CPU dummy probability tensors byte-identical: {probability_identical}", f"- evaluator metrics differ: {evaluator_metrics_differ}", "", "## Static Proof", "", "```json", json.dumps(static, indent=2, sort_keys=True), "```"]
    write_json(ARTIFACT / "r6a_phase1_firewall_confirmation.json", payload)
    write_text(R6 / "reports" / "r6a_phase1_firewall_confirmation.md", "\n".join(lines) + "\n")
    return payload


def aggregate_results(per_run: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in per_run:
        if row["status"] != "RUN_COMPLETED":
            continue
        key = (row["task_id"], row["seed"])
        grouped.setdefault(key, {})[row["condition"]] = row
    cells = []
    for (task_id, seed), conds in grouped.items():
        if all(cond in conds for cond in CONDITIONS):
            f1 = conds["F1"]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            s = conds["S"]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            f2 = conds["F2"]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            nu = abs(f1 - f2)
            delta = s - ((f1 + f2) / 2.0)
            cells.append({"task_id": task_id, "stratum": conds["F1"]["stratum"], "seed": seed, "ell_F1": f1, "ell_S": s, "ell_F2": f2, "nu": nu, "delta": delta, "delta_gt_0": delta > 0, "delta_gt_nu": delta > nu})
    per_task = []
    for task_id in sorted({cell["task_id"] for cell in cells}):
        task_cells = [cell for cell in cells if cell["task_id"] == task_id]
        per_task.append({"task_id": task_id, "stratum": task_cells[0]["stratum"], "seed_count": len(task_cells), "mean_delta": sum(cell["delta"] for cell in task_cells) / len(task_cells), "positive_seed_count": sum(cell["delta"] > 0 for cell in task_cells), "delta_gt_nu_seed_count": sum(cell["delta"] > cell["nu"] for cell in task_cells), "cells": task_cells})
    def stratum_summary(stratum: str) -> dict[str, Any]:
        scells = [cell for cell in cells if cell["stratum"] == stratum]
        stasks = [task for task in per_task if task["stratum"] == stratum]
        ltoo = []
        for task in stasks:
            rem = [cell for cell in scells if cell["task_id"] != task["task_id"]]
            ltoo.append({"removed_task_id": task["task_id"], "mean_delta": sum(cell["delta"] for cell in rem) / len(rem) if rem else None, "positive": (sum(cell["delta"] for cell in rem) / len(rem) > 0) if rem else None})
        thresholds = manifest["verdict_thresholds"]["stratum_a" if stratum == "A" else "stratum_b"]
        passes = False
        if stratum == "A" and not thresholds.get("insufficient_coverage"):
            passes = sum(cell["delta_gt_0"] for cell in scells) >= thresholds["positive_delta_threshold"] and sum(cell["delta_gt_nu"] for cell in scells) >= thresholds["delta_gt_nu_threshold"] and sum(task["mean_delta"] > 0 for task in stasks) >= thresholds["positive_task_mean_threshold"] and all(row["positive"] for row in ltoo)
        if stratum == "B" and thresholds.get("positive_control_valid"):
            passes = sum(cell["delta_gt_0"] for cell in scells) >= thresholds["positive_delta_threshold"] and sum(cell["delta_gt_nu"] for cell in scells) >= thresholds["delta_gt_nu_threshold"] and sum(task["mean_delta"] > 0 for task in stasks) >= thresholds["positive_task_mean_threshold"]
        return {"stratum": stratum, "cell_count": len(scells), "task_count": len(stasks), "mean_delta": sum(cell["delta"] for cell in scells) / len(scells) if scells else None, "positive_delta_cells": sum(cell["delta_gt_0"] for cell in scells), "delta_gt_nu_cells": sum(cell["delta_gt_nu"] for cell in scells), "positive_task_mean_count": sum(task["mean_delta"] > 0 for task in stasks), "leave_one_task_out": ltoo, "thresholds": thresholds, "passes": passes}
    ineligible_tasks = [row for row in manifest["tasks"] if not row["recomputed"]["eligible"]]
    expected_keys = {(row["task_id"], row["seed"], row["condition"]) for row in manifest["run_order"]}
    completed_keys = {(row["task_id"], row["seed"], row["condition"]) for row in per_run if row.get("status") == "RUN_COMPLETED"}
    missing_runs = [dict(task_id=task_id, seed=seed, condition=condition) for task_id, seed, condition in sorted(expected_keys - completed_keys)]
    historical_failed_attempt_rows = [row for row in per_run if row.get("status") != "RUN_COMPLETED"]
    unresolved_failed_runs = [
        row
        for row in historical_failed_attempt_rows
        if (row.get("task_id"), row.get("seed"), row.get("condition")) not in completed_keys
    ]
    aggregate = {
        "cells": cells,
        "stratum_a": stratum_summary("A"),
        "stratum_b": stratum_summary("B"),
        "ineligible_tasks": ineligible_tasks,
        "run_completion": {
            "expected_condition_runs": len(expected_keys),
            "completed_condition_runs": len(completed_keys),
            "missing_runs": missing_runs,
            "unresolved_failed_runs": unresolved_failed_runs,
            "historical_failed_attempt_rows": historical_failed_attempt_rows,
            "all_frozen_runs_completed": not missing_runs and not unresolved_failed_runs,
        },
    }
    a = aggregate["stratum_a"]["passes"]
    b = aggregate["stratum_b"]["passes"]
    if not aggregate["run_completion"]["all_frozen_runs_completed"]:
        verdict = "EXECUTION_OR_PROTOCOL_BLOCKED"
    elif any(row["stratum"] == "B" for row in ineligible_tasks):
        verdict = "INCONCLUSIVE"
    elif a and b:
        verdict = "PAIRING_SENSITIVE_QUERY_PILOT"
    elif len(cells) == 0:
        verdict = "EXECUTION_OR_PROTOCOL_BLOCKED"
    else:
        verdict = "INCONCLUSIVE"
    aggregate["final_verdict"] = verdict
    return {"tasks": per_task}, aggregate


def write_reports(manifest: dict[str, Any], firewall: dict[str, Any], per_run: list[dict[str, Any]], per_task: dict[str, Any], aggregate: dict[str, Any], pytest_pre: str | None, pytest_post: str | None, runtime_seconds: float) -> None:
    report = ["# R6A Phase1 Multi-Repeat Pairing Audit", "", f"- source commit: {manifest['source_commit']}", f"- environment label: {manifest['environment_label']}", f"- firewall preflight passed: {firewall['firewall_preflight_passed']}", f"- runtime seconds: {runtime_seconds}", "- runtime note: 16.7 hours is only a lower-bound estimate based on the 12:22 smoke runtime; actual runtime may be approximately 25-27 hours because evaluation tasks may be larger or slower.", f"- final verdict: {aggregate['final_verdict']}", "", "## Stratum Results", "", json.dumps({"stratum_a": aggregate["stratum_a"], "stratum_b": aggregate["stratum_b"]}, indent=2, sort_keys=True), "", "## Tests", "", f"- pre-run focused tests: {pytest_pre}", f"- post-run focused tests: {pytest_post}"]
    write_text(R6 / "reports" / "r6a_phase1_multirepeat_pairing_report.md", "\n".join(report) + "\n")
    packet = ["# R6A Phase1 Decision Packet", "", f"- final verdict: {aggregate['final_verdict']}", f"- firewall preflight passed: {firewall['firewall_preflight_passed']}", f"- protected VRM integrity: pending/final manifest", "", "## Aggregate", "", json.dumps(aggregate, indent=2, sort_keys=True)]
    write_text(R6 / "reports" / "r6a_phase1_decision_packet.md", "\n".join(packet) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R6A Phase1 multi-repeat ALL_PAIRING_SWAP audit.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stop-after-preflight", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = build_phase1_manifest()
    if not manifest["permutation_authority_all_stratum_a_match"]:
        result = {"status": "EXECUTION_OR_PROTOCOL_BLOCKED", "reason": "Stratum A recomputed permutation does not match old manifest", "manifest": manifest}
        write_json(ARTIFACT / "r6a_phase1_aggregate_results.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    if manifest["verdict_thresholds"]["stratum_a"].get("insufficient_coverage"):
        result = {"status": "INCONCLUSIVE", "reason": "fewer than 5 eligible Stratum A tasks", "manifest": manifest}
        write_json(ARTIFACT / "r6a_phase1_aggregate_results.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    init_rows = []
    for task in manifest["tasks"]:
        problem = eval_challenges[task["task_id"]] if task["stratum"] == "A" else train_challenges[task["task_id"]]
        for seed in SEEDS:
            init_rows.append({"task_id": task["task_id"], "stratum": task["stratum"], "seed": seed, **create_initialization_artifact(task["task_id"], task["stratum"], seed, problem, Path(manifest["initialization_artifact_paths"][f"{task['task_id']}:{seed}"]))})
    init_manifest = {"created_utc": now(), "rows": init_rows, "all_artifacts_exist": all(Path(row["path"]).exists() for row in init_rows), "all_f1_s_f2_use_same_initialization_by_design": True}
    write_json(R6 / "manifests" / "r6a_phase1_initialization_artifacts.json", init_manifest)
    firewall = confirm_firewall(manifest)
    if not firewall["firewall_preflight_passed"]:
        result = {"status": "EXECUTION_OR_PROTOCOL_BLOCKED", "reason": "firewall preflight failed", "firewall": firewall}
        write_json(ARTIFACT / "r6a_phase1_aggregate_results.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    if args.stop_after_preflight:
        print(json.dumps({"status": "PREFLIGHT_COMPLETE", "manifest": str(R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json")}, indent=2))
        return 0
    per_run_path = ARTIFACT / "r6a_phase1_per_run_results.json"
    per_run = read_json(per_run_path)["rows"] if per_run_path.exists() else []
    completed_keys = {(row["task_id"], row["seed"], row["condition"]) for row in per_run if row.get("status") == "RUN_COMPLETED"}
    solution_files = {"A": REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_solutions.json", "B": REFERENCE_ROOT / "dataset" / "arc-agi_training_solutions.json"}
    solutions = {key: read_json(path) for key, path in solution_files.items()}
    for item in manifest["run_order"]:
        key = (item["task_id"], item["seed"], item["condition"])
        if key in completed_keys:
            continue
        task_meta = next(row for row in manifest["tasks"] if row["task_id"] == item["task_id"])
        problem = eval_challenges[item["task_id"]] if item["stratum"] == "A" else train_challenges[item["task_id"]]
        permutation = task_meta["recomputed"]["selected_permutation"] if item["condition"] == "S" else None
        run_dir = ARTIFACT / "phase1_runs" / item["stratum"] / item["task_id"] / str(item["seed"]) / item["condition"]
        challenge = write_challenge(item["task_id"], problem, item["condition"], permutation, run_dir / "challenge.json")
        solver_json = run_dir / "solver.json"
        prob_npz = run_dir / "probability.npz"
        cmd = solver_command(item["task_id"], item["stratum"], item["condition"], challenge, Path(manifest["initialization_artifact_paths"][f"{item['task_id']}:{item['seed']}"]), solver_json, prob_npz, args.device)
        attempt = run_command(cmd, timeout=max(7200, SMOKE_BUDGET * 6))
        row = {**item, "solver_command": cmd, "attempts": [attempt], "solver_json": str(solver_json), "probability_npz": str(prob_npz)}
        if attempt["returncode"] != 0:
            retry = run_command(cmd, timeout=max(7200, SMOKE_BUDGET * 6))
            row["attempts"].append(retry)
        if solver_json.exists():
            solver_payload = read_json(solver_json)
            row.update(solver_payload)
            if solver_payload.get("status") == "RUN_COMPLETED":
                row["evaluator_metrics"] = evaluator_metrics(item["task_id"], item["stratum"], solver_payload, read_json(challenge)[item["task_id"]], solutions[item["stratum"]][item["task_id"]], prob_npz)
        per_run.append(row)
        write_json(per_run_path, {"rows": per_run, "updated_utc": now()})
    per_task, aggregate = aggregate_results(per_run, manifest)
    aggregate["firewall"] = firewall
    aggregate["runtime_seconds"] = time.perf_counter() - start
    aggregate["failed_or_retried_runs"] = [row for row in per_run if len(row.get("attempts", [])) > 1 or row.get("status") != "RUN_COMPLETED"]
    write_json(ARTIFACT / "r6a_phase1_per_run_results.json", {"rows": per_run, "updated_utc": now()})
    write_json(ARTIFACT / "r6a_phase1_per_task_results.json", per_task)
    write_json(ARTIFACT / "r6a_phase1_aggregate_results.json", aggregate)
    pytest_pre = (ARTIFACT / "r6a_phase1_pytest_pre.txt").read_text(encoding="utf-8").strip() if (ARTIFACT / "r6a_phase1_pytest_pre.txt").exists() else None
    pytest_post = (ARTIFACT / "r6a_phase1_pytest_post.txt").read_text(encoding="utf-8").strip() if (ARTIFACT / "r6a_phase1_pytest_post.txt").exists() else None
    write_reports(manifest, firewall, per_run, per_task, aggregate, pytest_pre, pytest_post, aggregate["runtime_seconds"])
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
