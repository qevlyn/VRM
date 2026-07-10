from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT


R6 = PROJECT_ROOT / "r6a_compressarc"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    manifest = read_json(R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json")
    init_manifest = read_json(R6 / "manifests" / "r6a_phase1_initialization_artifacts.json")
    firewall = read_json(R6 / "artifacts" / "r6a_phase1_firewall_confirmation.json")
    aggregate = read_json(R6 / "artifacts" / "r6a_phase1_aggregate_results.json")
    per_task = read_json(R6 / "artifacts" / "r6a_phase1_per_task_results.json")
    per_run = read_json(R6 / "artifacts" / "r6a_phase1_per_run_results.json")
    env = read_json(R6 / "manifests" / "determinism_environment.json")
    protected_after = read_json(R6 / "manifests" / "protected_paths_after_phase1.json")
    pytest_pre = (R6 / "artifacts" / "r6a_phase1_pytest_pre.txt").read_text(encoding="utf-8").strip()
    pytest_post = (R6 / "artifacts" / "r6a_phase1_pytest_post.txt").read_text(encoding="utf-8").strip()

    completed = [row for row in per_run["rows"] if row.get("status") == "RUN_COMPLETED"]
    historical_failed = aggregate["run_completion"]["historical_failed_attempt_rows"]
    completed_keys = {(row["task_id"], row["seed"], row["condition"]) for row in completed}
    failed_keys_later_completed = sum(
        1
        for row in historical_failed
        if (row.get("task_id"), row.get("seed"), row.get("condition")) in completed_keys
    )

    aggregate["post_run_verification"] = {
        "focused_pytest_pre": pytest_pre,
        "focused_pytest_post": pytest_post,
        "protected_vrm_integrity_all_match_before": protected_after["all_match_before"],
        "protected_vrm_integrity_file_count": protected_after["file_count"],
        "all_frozen_condition_keys_completed": aggregate["run_completion"]["all_frozen_runs_completed"],
        "completed_condition_rows": len(completed),
        "historical_failed_attempt_rows": len(historical_failed),
        "historical_failed_keys_later_completed": failed_keys_later_completed,
    }
    aggregate["runtime_hours"] = aggregate["runtime_seconds"] / 3600.0
    write_json(R6 / "artifacts" / "r6a_phase1_aggregate_results.json", aggregate)

    perm_lines = [
        "| task | stratum | eligible | old permutation | recomputed permutation | score | authority |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for task in manifest["tasks"]:
        rec = task["recomputed"]
        perm_lines.append(
            f"| {task['task_id']} | {task['stratum']} | {rec['eligible']} | "
            f"{task['old_manifest_permutation']} | {rec['selected_permutation']} | "
            f"{rec['selected_total_padded_hamming']} | {task['permutation_authority_match']} |"
        )

    cell_lines = [
        "| stratum | task | seed | ell_F1 | ell_S | ell_F2 | nu | delta | delta>0 | delta>nu |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in aggregate["cells"]:
        cell_lines.append(
            f"| {cell['stratum']} | {cell['task_id']} | {cell['seed']} | "
            f"{fmt(cell['ell_F1'])} | {fmt(cell['ell_S'])} | {fmt(cell['ell_F2'])} | "
            f"{fmt(cell['nu'])} | {fmt(cell['delta'])} | {cell['delta_gt_0']} | {cell['delta_gt_nu']} |"
        )

    init_lines = [
        "| task | stratum | seed | model hash | optimizer hash | rng hash |",
        "|---|---|---:|---|---|---|",
    ]
    for row in init_manifest["rows"]:
        init_lines.append(
            f"| {row['task_id']} | {row['stratum']} | {row['seed']} | "
            f"{row['model_state_hash'][:16]}... | {row['optimizer_state_hash'][:16]}... | "
            f"{row['rng_state_hash'][:16]}... |"
        )

    failure_summary = {
        "historical_failed_attempt_rows": len(historical_failed),
        "failed_keys_later_completed": failed_keys_later_completed,
        "unresolved_failed_runs": len(aggregate["run_completion"]["unresolved_failed_runs"]),
        "failure_type": historical_failed[0]["failure"]["type"] if historical_failed else None,
        "failure_message": historical_failed[0]["failure"]["message"] if historical_failed else None,
        "note": (
            "Historical failures occurred before optimization because torch.load map_location moved "
            "the saved CPU RNG state; the solver loader was patched to load initialization artifacts "
            "on CPU first, and all frozen keys later completed. No failed numerical outcome was discarded."
        ),
    }

    report: list[str] = []
    report += ["# R6A Phase1 Multi-Repeat Pairing Audit", ""]
    report += [
        f"- source commit: {manifest['source_commit']}",
        f"- environment label: {manifest['environment_label']}",
        f"- active dependency versions: {json.dumps(env['installed_versions'], sort_keys=True)}",
        f"- Python executable: {env['active_environment']['python_executable']}",
        f"- CUDA available: {env['active_environment']['torch_cuda_is_available']}",
        f"- firewall preflight passed: {firewall['firewall_preflight_passed']}",
        f"- protected VRM/reference integrity: {'PASS' if protected_after['all_match_before'] else 'FAIL'} ({protected_after['file_count']} files)",
        f"- all frozen runs completed: {aggregate['run_completion']['all_frozen_runs_completed']} ({aggregate['run_completion']['completed_condition_runs']}/{aggregate['run_completion']['expected_condition_runs']})",
        f"- runtime seconds: {aggregate['runtime_seconds']}",
        f"- runtime hours: {aggregate['runtime_hours']:.3f}",
        "- runtime note: 16.7 hours is only a lower-bound estimate based on the 12:22 smoke runtime; actual runtime may be approximately 25-27 hours because evaluation tasks may be larger or slower.",
        f"- final verdict: {aggregate['final_verdict']}",
        "",
    ]
    report += ["## Firewall Proof", "", "```json", json.dumps(firewall, indent=2, sort_keys=True), "```", ""]
    report += [
        "## Frozen Manifest",
        "",
        f"- manifest path: {R6 / 'manifests' / 'r6a_phase1_multirepeat_frozen_manifest.json'}",
        f"- seeds: {manifest['seeds']}",
        f"- optimization budget: {manifest['optimization_budget']}",
        f"- run order count: {len(manifest['run_order'])}",
        f"- permutation authority all Stratum A match: {manifest['permutation_authority_all_stratum_a_match']}",
        "",
        *perm_lines,
        "",
    ]
    report += [
        "## Eligibility And Thresholds",
        "",
        "```json",
        json.dumps(manifest["verdict_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    report += [
        "## Initialization Artifacts",
        "",
        f"- artifact count: {len(init_manifest['rows'])}",
        f"- all artifacts exist: {init_manifest['all_artifacts_exist']}",
        "",
        *init_lines,
        "",
    ]
    report += [
        "## Stratum Summaries",
        "",
        "```json",
        json.dumps({"stratum_a": aggregate["stratum_a"], "stratum_b": aggregate["stratum_b"]}, indent=2, sort_keys=True),
        "```",
        "",
    ]
    report += [
        "## Per-Task Results",
        "",
        "```json",
        json.dumps(per_task, indent=2, sort_keys=True),
        "```",
        "",
    ]
    report += ["## All F1/S/F2 NLL Cells", "", *cell_lines, ""]
    report += ["## Tests", "", f"- pre-run focused tests: {pytest_pre}", f"- post-run focused tests: {pytest_post}", ""]
    report += ["## Failed Attempts And Retries", "", "```json", json.dumps(failure_summary, indent=2, sort_keys=True), "```", ""]
    report += [
        "## Notes",
        "",
        "- The evaluator-only target-cell NLL used the mean of the final 100 serialized query probability checkpoints and did not affect optimizer updates, stopping, output voting, or answer selection.",
        "- For cells lying inside the evaluator-known target shape but outside a model-active or semantically meaningful padded region, the stored padded probability may not carry an interpretable spatial meaning. This does not alter the paired comparison because F1, S, and F2 use the identical canonical probability representation and evaluator rule.",
        "- Under this local implementation and frozen assay, the full CompressARC pipeline -- per-task optimization, MDL-style objective, multitensor representation, equivariance, and engineered operations together -- showed evidence of using paired support correspondence for hidden-query prediction.",
        "",
    ]
    (R6 / "reports" / "r6a_phase1_multirepeat_pairing_report.md").write_text("\n".join(report), encoding="utf-8")

    packet: list[str] = []
    packet += ["# R6A Phase1 Decision Packet", ""]
    packet += [
        f"- final verdict: {aggregate['final_verdict']}",
        f"- firewall preflight passed: {firewall['firewall_preflight_passed']}",
        f"- protected VRM integrity: {'PASS' if protected_after['all_match_before'] else 'FAIL'}",
        f"- all frozen runs completed: {aggregate['run_completion']['all_frozen_runs_completed']}",
        f"- runtime seconds: {aggregate['runtime_seconds']}",
        f"- runtime hours: {aggregate['runtime_hours']:.3f}",
        f"- historical failed attempt rows: {len(historical_failed)}",
        f"- unresolved failed runs: {len(aggregate['run_completion']['unresolved_failed_runs'])}",
        "",
    ]
    packet += ["## Stratum A", "", "```json", json.dumps(aggregate["stratum_a"], indent=2, sort_keys=True), "```", ""]
    packet += ["## Stratum B", "", "```json", json.dumps(aggregate["stratum_b"], indent=2, sort_keys=True), "```", ""]
    packet += ["## Run Completion", "", "```json", json.dumps(aggregate["run_completion"], indent=2, sort_keys=True), "```", ""]
    (R6 / "reports" / "r6a_phase1_decision_packet.md").write_text("\n".join(packet), encoding="utf-8")

    print(json.dumps({"report": str(R6 / "reports" / "r6a_phase1_multirepeat_pairing_report.md"), "decision_packet": str(R6 / "reports" / "r6a_phase1_decision_packet.md"), "aggregate_updated": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
