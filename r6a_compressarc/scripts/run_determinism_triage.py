from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, SMOKE_BUDGET, RUN_SEED, read_json, write_json, write_text, challenge_for_condition


R6 = PROJECT_ROOT / "r6a_compressarc"
TRIAGE_DIR = R6 / "artifacts" / "determinism_triage"
TASK_ID = "903d1b4a"


def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=merged, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-6000:], "runtime_seconds": time.perf_counter() - start}


def trace_cmd(label: str, output: Path, *, strict: bool = False, steps: int = SMOKE_BUDGET, device: str = "cuda") -> list[str]:
    challenge = R6 / "artifacts" / "phase0_smoke" / TASK_ID / "FULL" / "challenge.json"
    cmd = [
        sys.executable,
        str(R6 / "scripts" / "determinism_trace_solver.py"),
        "--compressarc-root",
        str(REFERENCE_ROOT),
        "--challenge-json",
        str(challenge),
        "--task-id",
        TASK_ID,
        "--condition",
        "FULL",
        "--run-label",
        label,
        "--steps",
        str(steps),
        "--trace-interval",
        "25",
        "--seed",
        str(RUN_SEED),
        "--device",
        device,
        "--output-json",
        str(output),
    ]
    if strict:
        cmd.append("--strict-deterministic")
    return cmd


def load_trace(path: Path) -> dict[str, Any]:
    return read_json(path)


def compare_trace(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    checks = [
        ("python_random_state_immediately_before_task_model_construction", left.get("rng_state_immediately_before_task_model_construction", {}).get("python_random_state_hash"), right.get("rng_state_immediately_before_task_model_construction", {}).get("python_random_state_hash"), "RNG_STATE_DRIFT"),
        ("numpy_random_state_immediately_before_task_model_construction", left.get("rng_state_immediately_before_task_model_construction", {}).get("numpy_random_state_hash"), right.get("rng_state_immediately_before_task_model_construction", {}).get("numpy_random_state_hash"), "RNG_STATE_DRIFT"),
        ("torch_cpu_rng_state_immediately_before_task_model_construction", left.get("rng_state_immediately_before_task_model_construction", {}).get("torch_cpu_rng_state_hash"), right.get("rng_state_immediately_before_task_model_construction", {}).get("torch_cpu_rng_state_hash"), "RNG_STATE_DRIFT"),
        ("torch_cuda_rng_state_immediately_before_task_model_construction", left.get("rng_state_immediately_before_task_model_construction", {}).get("torch_cuda_rng_state_hash"), right.get("rng_state_immediately_before_task_model_construction", {}).get("torch_cuda_rng_state_hash"), "RNG_STATE_DRIFT"),
        ("model_parameter_state_hash_after_initialization", left.get("model_parameter_state_hash_after_initialization"), right.get("model_parameter_state_hash_after_initialization"), "INITIALIZATION_DIVERGENCE"),
        ("optimizer_state_hash_after_construction", left.get("optimizer_state_hash_after_construction"), right.get("optimizer_state_hash_after_construction"), "INITIALIZATION_DIVERGENCE"),
        ("pre_update_forward_loss_step_0", left.get("pre_update_forward_loss_step_0"), right.get("pre_update_forward_loss_step_0"), "FORWARD_DIVERGENCE_BEFORE_FIRST_UPDATE"),
    ]
    for name, a, b, classification in checks:
        if a != b:
            return {"first_divergence_checkpoint": name, "classification": classification, "left": a, "right": b}
    left_by_step = {row["step"]: row for row in left.get("checkpoints", [])}
    right_by_step = {row["step"]: row for row in right.get("checkpoints", [])}
    for step in sorted(set(left_by_step) & set(right_by_step)):
        a = left_by_step[step]
        b = right_by_step[step]
        fields = ["loss", "total_KL", "reconstruction_error", "model_parameter_state_hash", "optimizer_state_hash"]
        for field in fields:
            if a.get(field) != b.get(field):
                cls = "BACKWARD_OR_OPTIMIZER_DIVERGENCE" if step >= 1 else "FORWARD_DIVERGENCE_BEFORE_FIRST_UPDATE"
                return {"first_divergence_checkpoint": f"after_optimizer_step_{step}:{field}", "classification": cls, "left": a.get(field), "right": b.get(field)}
    if left.get("final_predicted_query_grid_hash") != right.get("final_predicted_query_grid_hash"):
        return {"first_divergence_checkpoint": "final_predicted_query_grid_hash", "classification": "BACKWARD_OR_OPTIMIZER_DIVERGENCE", "left": left.get("final_predicted_query_grid_hash"), "right": right.get("final_predicted_query_grid_hash")}
    if left.get("structured_log_hash_excluding_wall_clock") != right.get("structured_log_hash_excluding_wall_clock"):
        return {"first_divergence_checkpoint": "structured_log_hash_excluding_wall_clock", "classification": "LOG_ONLY_SERIALIZATION_DIVERGENCE", "left": left.get("structured_log_hash_excluding_wall_clock"), "right": right.get("structured_log_hash_excluding_wall_clock")}
    return {"first_divergence_checkpoint": None, "classification": "NO_DIVERGENCE_FOUND"}


def randomness_audit() -> dict[str, Any]:
    terms = ["random", "np.random", "torch.manual_seed", "torch.cuda.manual_seed_all", "torch.Generator", "torch.randn", "torch.rand", "randn_like", "set(", "dict("]
    rows = []
    for path in sorted(REFERENCE_ROOT.glob("*.py")) + sorted((R6 / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(term in line for term in terms):
                hits.append({"line": lineno, "text": line.strip()})
        if hits:
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)), "hits": hits})
    return {"terms": terms, "rows": rows}


def create_initialization_artifacts() -> dict[str, Any]:
    import random
    import numpy as np
    import torch

    sys.path.insert(0, str(REFERENCE_ROOT))
    import preprocessing
    import arc_compressor

    phase1 = read_json(R6 / "manifests" / "phase1_pilot_tasks_and_permutations.json")
    challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    root = TRIAGE_DIR / "future_pairing_initialization_artifacts"
    proofs = []
    for task_id, meta in phase1["task_metadata"].items():
        problem = challenges[task_id]
        task_dir = root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        random.seed(RUN_SEED)
        np.random.seed(RUN_SEED)
        torch.manual_seed(RUN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RUN_SEED)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("cuda")
        full_task = preprocessing.Task(task_id, challenge_for_condition(problem, "FULL"), None)
        model = arc_compressor.ARCCompressor(full_task)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        init_state = [tensor.detach().cpu().clone() for tensor in model.weights_list]
        torch.save(init_state, task_dir / "initial_model_weights.pt")
        torch.save(optimizer.state_dict(), task_dir / "initial_optimizer_state.pt")
        conditions = {"FULL": None}
        if meta["b1_selected_permutation"] is not None:
            conditions["ALL_PAIRING_SWAP"] = meta["b1_selected_permutation"]
        if meta["b2_selected_permutation"] is not None:
            conditions["CONTENT_PAIRING_SWAP"] = meta["b2_selected_permutation"]
        loaded_hashes = {}
        for condition, permutation in conditions.items():
            task = preprocessing.Task(task_id, challenge_for_condition(problem, condition, permutation), None)
            condition_model = arc_compressor.ARCCompressor(task)
            for dest, src in zip(condition_model.weights_list, init_state):
                dest.data.copy_(src.to(dest.device))
            condition_optimizer = torch.optim.Adam(condition_model.weights_list, lr=0.01, betas=(0.5, 0.9))
            from determinism_trace_solver import tensor_list_hash, optimizer_state_hash

            loaded_hashes[condition] = {
                "model_hash_after_loading_shared_initialization": tensor_list_hash(condition_model.weights_list),
                "optimizer_hash_after_construction": optimizer_state_hash(condition_optimizer),
            }
        proofs.append(
            {
                "task_id": task_id,
                "conditions": sorted(conditions),
                "loaded_hashes": loaded_hashes,
                "all_model_hashes_identical": len({row["model_hash_after_loading_shared_initialization"] for row in loaded_hashes.values()}) == 1,
                "all_optimizer_hashes_identical": len({row["optimizer_hash_after_construction"] for row in loaded_hashes.values()}) == 1,
            }
        )
    payload = {"artifact_root": str(root), "proofs": proofs, "all_conditions_begin_byte_identical": all(row["all_model_hashes_identical"] and row["all_optimizer_hashes_identical"] for row in proofs)}
    write_json(TRIAGE_DIR / "future_pairing_initialization_proof.json", payload)
    return payload


def classify_root(baseline: dict[str, Any], strict: dict[str, Any], cpu: dict[str, Any] | None) -> dict[str, Any]:
    if strict.get("unsupported_operation"):
        strict_label = "UNSUPPORTED_STRICT_DETERMINISM_OPERATION"
    elif strict["comparison"]["classification"] == "NO_DIVERGENCE_FOUND":
        strict_label = "DETERMINISTIC_AFTER_KERNEL_CONFIGURATION"
    elif strict["comparison"]["classification"] == "LOG_ONLY_SERIALIZATION_DIVERGENCE":
        strict_label = "LOG_ONLY_NONDETERMINISM"
    else:
        strict_label = "PERSISTENT_GPU_NONDETERMINISM"
    layer = "hardware/kernel-level" if strict_label in {"DETERMINISTIC_AFTER_KERNEL_CONFIGURATION", "PERSISTENT_GPU_NONDETERMINISM"} else "source-level"
    if strict_label == "UNSUPPORTED_STRICT_DETERMINISM_OPERATION" and cpu and cpu["comparison"]["classification"] == "NO_DIVERGENCE_FOUND":
        layer = "hardware/kernel-level"
    if baseline["comparison"]["classification"] == "LOG_ONLY_SERIALIZATION_DIVERGENCE":
        layer = "logging-only"
    if baseline["comparison"]["classification"] in {"INITIALIZATION_DIVERGENCE", "RNG_STATE_DRIFT"}:
        layer = "wrapper-level"
    recommendation = "A. Resume Phase 1 with exact deterministic paired runs." if strict_label == "DETERMINISTIC_AFTER_KERNEL_CONFIGURATION" else "C. Repair a concrete source/wrapper seed or logging defect, then rerun Phase 0B."
    if strict_label in {"PERSISTENT_GPU_NONDETERMINISM", "UNSUPPORTED_STRICT_DETERMINISM_OPERATION"} and cpu and cpu["comparison"]["classification"] == "NO_DIVERGENCE_FOUND":
        recommendation = "B. Resume Phase 1 with a predeclared multi-repeat stochastic paired protocol, only if nondeterminism is proven to be GPU-kernel-level and cannot be removed."
    return {"strict_result_classification": strict_label, "root_cause_layer": layer, "recommendation": recommendation}


def render_report(payload: dict[str, Any]) -> str:
    env = payload["environment"]
    baseline = payload["baseline_trace_comparison"]["comparison"]
    strict = payload["strict_trace_comparison"]["comparison"]
    root = payload["root_cause"]
    lines = [
        "# R6A 0C Determinism Triage Report",
        "",
        f"- environment label: {env['environment_label']}",
        f"- first baseline divergence checkpoint: {baseline['first_divergence_checkpoint']}",
        f"- preliminary baseline classification: {baseline['classification']}",
        f"- strict deterministic classification: {root['strict_result_classification']}",
        f"- root-cause layer: {root['root_cause_layer']}",
        f"- recommendation: {root['recommendation']}",
        "",
        "## Notes",
        "",
        "The solver process receives challenge JSON only. Hidden query targets are not loaded by trace runs.",
        "Phase 1 was not run in this subphase.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R6A 0C determinism triage.")
    parser.add_argument("--steps", type=int, default=SMOKE_BUDGET)
    parser.add_argument("--strict-steps", type=int, default=SMOKE_BUDGET)
    parser.add_argument("--cpu-prefix-steps", type=int, default=20)
    args = parser.parse_args()
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    env_manifest = read_json(R6 / "manifests" / "determinism_environment.json")

    runs = {}
    for label in ["baseline_a", "baseline_b"]:
        path = TRIAGE_DIR / f"{label}.json"
        if not path.exists():
            runs[label] = run(trace_cmd(label, path, steps=args.steps), timeout=max(7200, args.steps * 6))
        else:
            runs[label] = {"returncode": 0, "stdout": "reused existing trace", "runtime_seconds": 0.0}
    baseline_a = load_trace(TRIAGE_DIR / "baseline_a.json")
    baseline_b = load_trace(TRIAGE_DIR / "baseline_b.json")
    baseline_comparison = {"runs": runs, "comparison": compare_trace(baseline_a, baseline_b)}

    strict_env = {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    strict_runs = {}
    for label in ["strict_a", "strict_b"]:
        path = TRIAGE_DIR / f"{label}.json"
        if not path.exists():
            strict_runs[label] = run(trace_cmd(label, path, strict=True, steps=args.strict_steps), env=strict_env, timeout=max(7200, args.strict_steps * 6))
        else:
            strict_runs[label] = {"returncode": 0, "stdout": "reused existing trace", "runtime_seconds": 0.0}
    strict_a = load_trace(TRIAGE_DIR / "strict_a.json")
    strict_b = load_trace(TRIAGE_DIR / "strict_b.json")
    unsupported = next((row for row in [strict_a, strict_b] if row.get("status") == "TRACE_FAILED"), None)
    strict_comparison = {"runs": strict_runs, "comparison": compare_trace(strict_a, strict_b), "unsupported_operation": unsupported.get("failure") if unsupported else None}

    cpu_comparison = None
    if strict_comparison["unsupported_operation"] or strict_comparison["comparison"]["classification"] != "NO_DIVERGENCE_FOUND":
        cpu_runs = {}
        for label in ["cpu_prefix_a", "cpu_prefix_b"]:
            path = TRIAGE_DIR / f"{label}.json"
            if not path.exists():
                cpu_runs[label] = run(trace_cmd(label, path, steps=args.cpu_prefix_steps, device="cpu"), timeout=3600)
            else:
                cpu_runs[label] = {"returncode": 0, "stdout": "reused existing trace", "runtime_seconds": 0.0}
        cpu_comparison = {
            "runs": cpu_runs,
            "comparison": compare_trace(load_trace(TRIAGE_DIR / "cpu_prefix_a.json"), load_trace(TRIAGE_DIR / "cpu_prefix_b.json")),
        }

    init_proof = create_initialization_artifacts()
    payload = {
        "experiment": "R6A_0C_DETERMINISM_TRIAGE",
        "task_id": TASK_ID,
        "smoke_budget": SMOKE_BUDGET,
        "environment": env_manifest,
        "randomness_audit": randomness_audit(),
        "baseline_trace_comparison": baseline_comparison,
        "strict_trace_comparison": strict_comparison,
        "cpu_prefix_comparison": cpu_comparison,
        "future_pairing_initialization_proof": init_proof,
    }
    payload["root_cause"] = classify_root(baseline_comparison, strict_comparison, cpu_comparison)
    write_json(R6 / "artifacts" / "determinism_triage.json", payload)
    write_text(R6 / "reports" / "determinism_triage_report.md", render_report(payload))

    packet = R6 / "reports" / "decision_packet.md"
    text = packet.read_text(encoding="utf-8") if packet.exists() else "# R6A Decision Packet\n"
    section = "\n## R6A 0C Determinism Triage\n\n" + render_report(payload)
    if "## R6A 0C Determinism Triage" not in text:
        packet.write_text(text.rstrip() + section, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
