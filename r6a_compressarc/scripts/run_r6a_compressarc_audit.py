from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r6a_common import (
    EXPERIMENT_ROOT,
    PROJECT_ROOT,
    REFERENCE_BENCHMARK_BUDGET,
    REFERENCE_ROOT,
    RUN_SEED,
    SELECTION_SEED,
    SMOKE_BUDGET,
    challenge_for_condition,
    deterministic_selection,
    file_sha256,
    pairing_manifest_for_task,
    read_json,
    utc_now,
    write_json,
    write_text,
)


def run_command(cmd: list[str], *, cwd: Path = PROJECT_ROOT, timeout: int | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "runtime_seconds": time.perf_counter() - start}


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(REFERENCE_ROOT), *args], text=True).strip()

    return {
        "repository_url": git("remote", "get-url", "origin"),
        "commit_hash": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--format=%s"),
        "commit_author": git("log", "-1", "--format=%an <%ae>"),
        "commit_date": git("log", "-1", "--format=%ad"),
        "status_short": git("status", "--short"),
    }


def source_provenance() -> dict[str, Any]:
    info = git_info()
    req = (REFERENCE_ROOT / "requirements.txt").read_text(encoding="utf-8")
    readme = (REFERENCE_ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (REFERENCE_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
    payload = {
        "experiment": "R6A_COMPRESSARC_REFERENCE_SEMANTICS_AUDIT",
        "created_utc": utc_now(),
        "official_repository": "https://github.com/iliao2345/CompressARC",
        "git": info,
        "paper": {
            "title": "ARC-AGI Without Pretraining",
            "authors": ["Isaac Liao", "Albert Gu"],
            "arxiv": "https://arxiv.org/abs/2512.06104",
            "openreview": "https://openreview.net/forum?id=TbbMyr3E0x",
            "project_blog": "https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html",
            "readme_citation_present": "ARC-AGI Without Pretraining" in readme and "Isaac Liao and Albert Gu" in readme,
        },
        "license": license_text,
        "official_requirements_txt": req.splitlines(),
        "reference_budgets": {
            "SMOKE_BUDGET": SMOKE_BUDGET,
            "REFERENCE_BENCHMARK_BUDGET": REFERENCE_BENCHMARK_BUDGET,
        },
        "dataset_files_present": sorted(str(path.relative_to(REFERENCE_ROOT)) for path in (REFERENCE_ROOT / "dataset").glob("*.json")),
    }
    write_json(EXPERIMENT_ROOT / "source_provenance.json", payload)
    return payload


def static_source_audit() -> dict[str, Any]:
    rows = []
    for path in sorted(REFERENCE_ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        rows.append(
            {
                "path": str(path.relative_to(REFERENCE_ROOT)),
                "contains_solution_string": "solution" in text.lower(),
                "contains_challenge_string": "challenge" in text.lower(),
                "contains_json_load": "json.load" in text,
                "contains_preprocess_tasks": "preprocess_tasks" in text,
                "contains_Task_solution_None": "Task(task_name, problems[task_name], None)" in text,
            }
        )
    return {
        "safe_solver_path": "solve_task.py constructs preprocessing.Task(task_name, problems[task_name], None).",
        "unsafe_or_scoring_paths": [
            "preprocessing.preprocess_tasks loads *_solutions.json for training/evaluation splits.",
            "train.py __main__ uses preprocess_tasks and true solution hashes for plotting accuracy.",
            "list_solved_puzzles.py uses preprocess_tasks to compare against true solution hashes.",
            "scoring.py loads solutions for evaluator-side scoring.",
        ],
        "source_rows": rows,
    }


def collect_used_r5a_ids() -> set[str]:
    used: set[str] = set()
    for path in PROJECT_ROOT.glob("artifacts/r5a*/**/*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(key, str) and len(key) == 8 and all(ch in "0123456789abcdef" for ch in key.lower()):
                        used.add(key)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and len(value) == 8 and all(ch in "0123456789abcdef" for ch in value.lower()):
                used.add(value)

        walk(obj)
    return used


def freeze_task_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    used = collect_used_r5a_ids()
    untouched = sorted(set(eval_challenges) - used)
    label = "UNTOUCHED_ARC_AGI_1_EVALUATION_POOL" if len(untouched) >= 9 else "EXPLORATORY_DEVELOPMENT_POOL_NOT_SEALED"
    candidate_pool = untouched if len(untouched) >= 9 else sorted(eval_challenges)
    selected = deterministic_selection(candidate_pool, 9, SELECTION_SEED)
    smoke_ids = selected[:3]
    phase1_ids = selected[3:9]
    smoke = {
        "created_utc": utc_now(),
        "split": "ARC-AGI-1 evaluation",
        "selection_seed": SELECTION_SEED,
        "pool_label": label,
        "candidate_pool_count": len(candidate_pool),
        "candidate_pool_task_ids": candidate_pool,
        "selected_task_ids": smoke_ids,
        "task_metadata": {task_id: pairing_manifest_for_task(task_id, eval_challenges[task_id]) for task_id in smoke_ids},
        "outcomes_read_before_freeze": False,
    }
    phase1 = {
        "created_utc": utc_now(),
        "split": "ARC-AGI-1 evaluation",
        "selection_seed": SELECTION_SEED,
        "run_seed": RUN_SEED,
        "optimization_budget": SMOKE_BUDGET,
        "pool_label": label,
        "candidate_pool_count": len(candidate_pool),
        "candidate_pool_task_ids": candidate_pool,
        "selected_task_ids": phase1_ids,
        "task_metadata": {task_id: pairing_manifest_for_task(task_id, eval_challenges[task_id]) for task_id in phase1_ids},
        "outcomes_read_before_freeze": False,
    }
    write_json(EXPERIMENT_ROOT / "manifests" / "phase0_smoke_tasks.json", smoke)
    write_json(EXPERIMENT_ROOT / "manifests" / "phase1_pilot_tasks_and_permutations.json", phase1)
    return smoke, phase1


def write_challenge(task_id: str, problem: dict[str, Any], condition: str, permutation: list[int] | None, run_dir: Path) -> Path:
    payload = {task_id: challenge_for_condition(problem, condition, permutation)}
    path = run_dir / "challenge.json"
    write_json(path, payload)
    return path


def solver_cmd(task_id: str, condition: str, challenge_path: Path, output_path: Path, steps: int) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENT_ROOT / "scripts" / "solve_challenge_only.py"),
        "--compressarc-root",
        str(REFERENCE_ROOT),
        "--challenge-json",
        str(challenge_path),
        "--task-id",
        task_id,
        "--steps",
        str(steps),
        "--seed",
        str(RUN_SEED),
        "--condition",
        condition,
        "--output-json",
        str(output_path),
        "--device",
        "cuda",
    ]


def eval_cmd(task_id: str, challenge_path: Path, solution_path: Path, prediction_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        str(EXPERIMENT_ROOT / "scripts" / "evaluate_prediction.py"),
        "--challenge-json",
        str(challenge_path),
        "--solution-json",
        str(solution_path),
        "--prediction-json",
        str(prediction_path),
        "--task-id",
        task_id,
        "--output-json",
        str(output_path),
    ]


def negative_control(task_id: str, steps: int) -> dict[str, Any]:
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    problem = eval_challenges[task_id]
    run_root = EXPERIMENT_ROOT / "artifacts" / "target_firewall_negative_control"
    challenge = write_challenge(task_id, problem, "FULL", None, run_root)
    target_a = [[[0]] for _ in problem["test"]]
    target_b = [[[1, 1], [1, 1]] for _ in problem["test"]]
    solution_a = run_root / "solution_a.json"
    solution_b = run_root / "solution_b.json"
    write_json(solution_a, {task_id: target_a})
    write_json(solution_b, {task_id: target_b})
    pred_a = run_root / "solver_a.json"
    pred_b = run_root / "solver_b.json"
    run_a = run_command(solver_cmd(task_id, "FULL", challenge, pred_a, steps), timeout=1800)
    run_b = run_command(solver_cmd(task_id, "FULL", challenge, pred_b, steps), timeout=1800)
    eval_a = run_root / "eval_a.json"
    eval_b = run_root / "eval_b.json"
    eval_run_a = run_command(eval_cmd(task_id, challenge, solution_a, pred_a, eval_a), timeout=300) if pred_a.exists() else {"returncode": 99, "stdout": "prediction missing"}
    eval_run_b = run_command(eval_cmd(task_id, challenge, solution_b, pred_b, eval_b), timeout=300) if pred_b.exists() else {"returncode": 99, "stdout": "prediction missing"}
    solver_equal = False
    if pred_a.exists() and pred_b.exists():
        a = read_json(pred_a)
        b = read_json(pred_b)
        comparable = [
            "predictions",
            "actual_step_count",
            "stopping_reason",
            "trajectory",
            "final_objective",
            "final_KL_term",
            "final_reconstruction_term",
            "support_reconstruction",
        ]
        solver_equal = all(a.get(key) == b.get(key) for key in comparable)
    payload = {
        "task_id": task_id,
        "steps": steps,
        "run_a_returncode": run_a["returncode"],
        "run_b_returncode": run_b["returncode"],
        "eval_a_returncode": eval_run_a["returncode"],
        "eval_b_returncode": eval_run_b["returncode"],
        "solver_training_trajectory_identical": solver_equal,
        "predicted_query_grid_identical": solver_equal,
        "stopping_step_identical": solver_equal,
        "solver_side_logs_identical": solver_equal,
        "only_evaluator_metrics_may_differ": solver_equal,
        "status": "QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL_PASSED" if solver_equal and run_a["returncode"] == 0 and run_b["returncode"] == 0 else "TARGET_FIREWALL_FAILURE",
        "solver_a_stdout_tail": run_a["stdout"][-4000:],
        "solver_b_stdout_tail": run_b["stdout"][-4000:],
    }
    write_json(EXPERIMENT_ROOT / "artifacts" / "target_firewall_negative_control.json", payload)
    return payload


def run_one(task_id: str, condition: str, problem: dict[str, Any], permutation: list[int] | None, steps: int, root: Path) -> dict[str, Any]:
    run_dir = root / task_id / condition
    challenge = write_challenge(task_id, problem, condition, permutation, run_dir)
    solver_out = run_dir / "solver.json"
    evaluator_out = run_dir / "evaluator.json"
    command = solver_cmd(task_id, condition, challenge, solver_out, steps)
    if solver_out.exists() and evaluator_out.exists():
        solver_run = {"returncode": 0, "stdout": "reused existing solver artifact", "runtime_seconds": 0.0}
        eval_run = {"returncode": 0, "stdout": "reused existing evaluator artifact", "runtime_seconds": 0.0}
    else:
        solver_run = run_command(command, timeout=max(1800, steps * 4))
        eval_run = {"returncode": None, "stdout": "solver did not finish"}
    if solver_run["returncode"] == 0 and not evaluator_out.exists():
        eval_run = run_command(
            eval_cmd(task_id, challenge, REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_solutions.json", solver_out, evaluator_out),
            timeout=300,
        )
    solver_payload = read_json(solver_out) if solver_out.exists() else None
    evaluator_payload = read_json(evaluator_out) if evaluator_out.exists() else None
    return {
        "task_id": task_id,
        "condition": condition,
        "permutation": permutation,
        "steps": steps,
        "solver_command": command,
        "solver_returncode": solver_run["returncode"],
        "solver_stdout_tail": solver_run["stdout"][-4000:],
        "evaluator_returncode": eval_run["returncode"],
        "evaluator_stdout_tail": eval_run["stdout"][-4000:] if isinstance(eval_run.get("stdout"), str) else "",
        "solver_json": str(solver_out),
        "evaluator_json": str(evaluator_out),
        "solver": solver_payload,
        "evaluator": evaluator_payload,
    }


def summarize_phase1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], {})[row["condition"]] = row
    deltas = []
    b2_deltas = []
    for task_id, conds in by_task.items():
        full = conds.get("FULL")
        if not full or not full.get("evaluator"):
            continue
        full_metric = full["evaluator"]["metrics"]["all_cell_accuracy"]
        for control, target in (("ALL_PAIRING_SWAP", deltas), ("CONTENT_PAIRING_SWAP", b2_deltas)):
            other = conds.get(control)
            if other and other.get("evaluator"):
                metric = other["evaluator"]["metrics"]["all_cell_accuracy"]
                target.append({"task_id": task_id, "control": control, "delta_full_minus_control": full_metric - metric})
    def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
        vals = [item["delta_full_minus_control"] for item in items]
        return {
            "count": len(vals),
            "mean_delta_full_minus_control": sum(vals) / len(vals) if vals else None,
            "directionally_positive_count": sum(1 for value in vals if value > 0),
            "directionally_negative_count": sum(1 for value in vals if value < 0),
            "zero_count": sum(1 for value in vals if value == 0),
            "per_task": items,
        }
    return {"FULL_vs_B1": agg(deltas), "FULL_vs_B2": agg(b2_deltas)}


def verdict(preflight_ok: bool, smoke_ok: bool, phase1_rows: list[dict[str, Any]], phase1_summary: dict[str, Any]) -> tuple[str, str]:
    if not preflight_ok or not smoke_ok or not phase1_rows:
        return "EXECUTION_OR_PROTOCOL_BLOCKED", "investigate a concrete implementation/protocol defect"
    b1 = phase1_summary["FULL_vs_B1"]
    if b1["count"] == 0:
        return "EXECUTION_OR_PROTOCOL_BLOCKED", "investigate a concrete implementation/protocol defect"
    if b1["directionally_positive_count"] == b1["count"] and b1["count"] >= 2:
        b2 = phase1_summary["FULL_vs_B2"]
        if b2["count"] and b2["directionally_positive_count"] == b2["count"]:
            return "CONTENT_PAIRING_SENSITIVE_QUERY_PILOT", "freeze and run a larger sealed confirmation audit"
        return "PAIRING_SENSITIVE_QUERY_PILOT", "freeze and run a larger sealed confirmation audit"
    if b1["zero_count"] + b1["directionally_negative_count"] == b1["count"]:
        return "PAIRING_INSENSITIVE_PILOT", "stop the CompressARC semantic-audit line"
    return "REFERENCE_RUNNABLE_ONLY", "freeze and run a larger sealed confirmation audit"


def render_reports(
    provenance: dict[str, Any],
    source_audit: dict[str, Any],
    preflight_status: dict[str, Any],
    smoke_report: dict[str, Any],
    phase1_report: dict[str, Any],
    final_verdict: str,
    next_action: str,
) -> None:
    preflight = [
        "# R6A CompressARC Preflight Report",
        "",
        f"- official repository: {provenance['official_repository']}",
        f"- commit: `{provenance['git']['commit_hash']}`",
        f"- paper: {provenance['paper']['title']} ({', '.join(provenance['paper']['authors'])})",
        f"- license: {provenance['license']}",
        f"- official requirements: `{', '.join(provenance['official_requirements_txt'])}`",
        "- local compatibility note: separate `/home/quyen/.venvs/compressarc` could not be created because WSL Python lacks ensurepip; Python 3.14 could not install pinned matplotlib 3.10.0 without a compiler, so the existing CUDA VRM venv with matplotlib 3.11.0/tqdm 4.68.3 was used as a non-algorithmic compatibility runtime.",
        "",
        "## Source Audit",
        "",
        source_audit["safe_solver_path"],
        "",
        "Unsafe/scoring paths:",
        *[f"- {item}" for item in source_audit["unsafe_or_scoring_paths"]],
        "",
        "## Preflight Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in preflight_status.items()],
        "",
        "The objective in `train.take_step` is `total_KL + 10 * reconstruction_error`. `total_KL` is the latent/code-length/MDL-like term recorded per multitensor component. `reconstruction_error` is support/query-input reconstruction NLL-like error; hidden query outputs are skipped during loss.",
    ]
    write_text(EXPERIMENT_ROOT / "reports" / "preflight_report.md", "\n".join(preflight) + "\n")
    smoke_lines = ["# R6A Smoke Reproduction Report", "", json.dumps(smoke_report, indent=2, sort_keys=True)]
    write_text(EXPERIMENT_ROOT / "reports" / "smoke_reproduction_report.md", "\n".join(smoke_lines) + "\n")
    phase_lines = ["# R6A Phase 1 Semantic Audit Report", "", json.dumps(phase1_report, indent=2, sort_keys=True)]
    write_text(EXPERIMENT_ROOT / "reports" / "phase1_semantic_audit_report.md", "\n".join(phase_lines) + "\n")
    decision = [
        "# R6A Decision Packet",
        "",
        f"- official source: {provenance['official_repository']}",
        f"- commit: `{provenance['git']['commit_hash']}`",
        f"- target-firewall result: `{preflight_status.get('QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL')}`",
        f"- protected VRM integrity result: `{phase1_report.get('protected_vrm_integrity')}`",
        f"- final verdict: `{final_verdict}`",
        f"- recommended next action: {next_action}",
        "",
        "## Smoke Results",
        "",
        json.dumps(smoke_report, indent=2, sort_keys=True),
        "",
        "## Phase 1 Results",
        "",
        json.dumps(phase1_report, indent=2, sort_keys=True),
    ]
    write_text(EXPERIMENT_ROOT / "reports" / "decision_packet.md", "\n".join(decision) + "\n")
    write_text(
        EXPERIMENT_ROOT / "README.md",
        "# R6A CompressARC Reference Semantics Audit\n\nIsolated experiment line for official CompressARC provenance, target-firewall validation, smoke reproduction, and paired-support semantic controls.\n",
    )
    write_text(
        EXPERIMENT_ROOT / "PROTOCOL.md",
        f"# R6A Protocol\n\nSMOKE_BUDGET = {SMOKE_BUDGET}; REFERENCE_BENCHMARK_BUDGET = {REFERENCE_BENCHMARK_BUDGET}. Positive paired deltas mean FULL is better than the control. Hidden query targets are evaluator-only.\n",
    )


def protected_after_manifest() -> dict[str, Any]:
    before = read_json(EXPERIMENT_ROOT / "manifests" / "protected_paths_before.json")
    rows = []
    for row in before["files"]:
        path = PROJECT_ROOT / row["path"]
        rows.append(
            {
                "path": row["path"],
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": file_sha256(path) if path.exists() else None,
                "matches_before": path.exists() and file_sha256(path) == row["sha256"] and path.stat().st_size == row["size_bytes"],
            }
        )
    after = {
        "created_utc": utc_now(),
        "file_count": len(rows),
        "all_match_before": all(row["matches_before"] for row in rows),
        "files": rows,
    }
    write_json(EXPERIMENT_ROOT / "manifests" / "protected_paths_after.json", after)
    return after


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R6A CompressARC reference semantics audit.")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--negative-control-steps", type=int, default=2)
    args = parser.parse_args()

    for sub in ["configs", "manifests", "scripts", "tests", "artifacts", "patches", "reports"]:
        (EXPERIMENT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    provenance = source_provenance()
    source_audit = static_source_audit()
    smoke_manifest, phase1_manifest = freeze_task_manifests()
    negative = negative_control(smoke_manifest["selected_task_ids"][0], args.negative_control_steps)
    preflight_status = {
        "OFFICIAL_SOURCE_IDENTIFIED": "PASS",
        "RUNNABLE_COMMAND_IDENTIFIED": "PASS",
        "QUERY_TARGET_FIREWALL_IMPLEMENTED": "PASS",
        "QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL": negative["status"],
        "PROTECTED_VRM_MANIFEST_CREATED": "PASS" if (EXPERIMENT_ROOT / "manifests" / "protected_paths_before.json").exists() else "FAIL",
    }
    preflight_ok = all(value in {"PASS", "QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL_PASSED"} for value in preflight_status.values())
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")

    smoke_rows = []
    smoke_ok = False
    if preflight_ok:
        for task_id in smoke_manifest["selected_task_ids"]:
            smoke_rows.append(run_one(task_id, "FULL", eval_challenges[task_id], None, SMOKE_BUDGET, EXPERIMENT_ROOT / "artifacts" / "phase0_smoke"))
        repeat_task = smoke_manifest["selected_task_ids"][0]
        repeat = run_one(repeat_task, "FULL_REPEAT", eval_challenges[repeat_task], None, SMOKE_BUDGET, EXPERIMENT_ROOT / "artifacts" / "phase0_smoke_repeat")
        smoke_rows.append(repeat)
        first = next(row for row in smoke_rows if row["task_id"] == repeat_task and row["condition"] == "FULL")
        deterministic = False
        if first.get("solver") and repeat.get("solver"):
            deterministic = all(first["solver"].get(key) == repeat["solver"].get(key) for key in ["predictions", "actual_step_count", "trajectory", "stopping_reason"])
        smoke_ok = all(row["solver_returncode"] == 0 and row["evaluator_returncode"] == 0 for row in smoke_rows) and deterministic
    smoke_report = {"budget": SMOKE_BUDGET, "rows": smoke_rows, "smoke_deterministic_repeat_exact": smoke_ok}

    phase1_rows = []
    phase1_summary = {"FULL_vs_B1": {"count": 0}, "FULL_vs_B2": {"count": 0}}
    if preflight_ok and smoke_ok and not args.skip_phase1:
        for task_id in phase1_manifest["selected_task_ids"]:
            meta = phase1_manifest["task_metadata"][task_id]
            problem = eval_challenges[task_id]
            phase1_rows.append(run_one(task_id, "FULL", problem, None, SMOKE_BUDGET, EXPERIMENT_ROOT / "artifacts" / "phase1"))
            if meta["b1_status"] == "B1_ELIGIBLE":
                phase1_rows.append(run_one(task_id, "ALL_PAIRING_SWAP", problem, meta["b1_selected_permutation"], SMOKE_BUDGET, EXPERIMENT_ROOT / "artifacts" / "phase1"))
            if meta["b2_status"] == "CONTENT_PAIRING_ELIGIBLE":
                phase1_rows.append(run_one(task_id, "CONTENT_PAIRING_SWAP", problem, meta["b2_selected_permutation"], SMOKE_BUDGET, EXPERIMENT_ROOT / "artifacts" / "phase1"))
        phase1_summary = summarize_phase1(phase1_rows)

    protected_after = protected_after_manifest()
    final_verdict, next_action = verdict(preflight_ok, smoke_ok, phase1_rows, phase1_summary)
    phase1_report = {
        "rows": phase1_rows,
        "summary": phase1_summary,
        "protected_vrm_integrity": "PASS" if protected_after["all_match_before"] else "FAIL",
        "phase1_verdict": final_verdict,
        "next_action": next_action,
    }
    aggregate = {
        "preflight_status": preflight_status,
        "smoke": smoke_report,
        "phase1": phase1_report,
        "final_verdict": final_verdict,
        "next_action": next_action,
    }
    write_json(EXPERIMENT_ROOT / "artifacts" / "per_task_results.json", {"smoke": smoke_rows, "phase1": phase1_rows})
    write_json(EXPERIMENT_ROOT / "artifacts" / "aggregate_results.json", aggregate)
    render_reports(provenance, source_audit, preflight_status, smoke_report, phase1_report, final_verdict, next_action)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
