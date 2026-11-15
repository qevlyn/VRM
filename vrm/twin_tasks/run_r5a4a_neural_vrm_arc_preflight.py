from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from vrm.twin_tasks.r5a4a_arc_manifest import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    build_r5a4a_splits,
    episodes_for_task_ids,
    render_source_manifest_text,
    source_file_manifest,
    split_counts,
    write_json,
    write_text,
)
from vrm.twin_tasks.r5a4a_neural_vrm_eval import (
    condition_payload,
    evaluate_condition,
    paired_trained_vs_fresh_intervals,
    render_case_cards,
    stable_payload_without_runtime,
    support_ablation_intervals,
    write_json as write_eval_json,
)
from vrm.twin_tasks.r5a4a_neural_vrm_noid_audit import run_no_id_audit
from vrm.twin_tasks.r5a4a_neural_vrm_train import (
    DEFAULT_SHAPE_LOSS_WEIGHT,
    INITIALIZATION_SEED,
    MAX_OPTIMIZER_STEPS,
    load_state,
    resolve_device,
    run_training,
    source_paths_used_by_r5a4a,
)

METRIC_FILES = {
    "FRESH_INIT": "fresh_init_metrics.json",
    "TRAINED_FULL_SUPPORT": "trained_full_support_metrics.json",
    "TRAINED_ZEROED_SUPPORT_OUTPUTS": "trained_zeroed_support_outputs_metrics.json",
    "TRAINED_QUERY_ONLY": "trained_query_only_metrics.json",
    "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER": "trained_permuted_support_pair_order_metrics.json",
}


def interval_excludes_zero_positive(interval: dict[str, Any]) -> bool:
    return interval.get("ci95_low") is not None and interval["ci95_low"] > 0


def determine_verdict(*, trained_vs_fresh: dict[str, Any], support_summary: dict[str, Any], no_id_pass: bool, query_only_supported: bool) -> str:
    if not query_only_supported:
        return "INCONCLUSIVE"
    loss_signal = interval_excludes_zero_positive(trained_vs_fresh["total_loss_improvement_fresh_minus_trained"]) or interval_excludes_zero_positive(trained_vs_fresh["color_nll_improvement_fresh_minus_trained"])
    accuracy_signal = interval_excludes_zero_positive(trained_vs_fresh["cell_accuracy_improvement_trained_minus_fresh"])
    if not loss_signal and not accuracy_signal:
        return "NO_TRAINING_SIGNAL"
    zero = support_summary["ablations"].get("TRAINED_ZEROED_SUPPORT_OUTPUTS", {})
    query = support_summary["ablations"].get("TRAINED_QUERY_ONLY", {})
    zero_support_loss = interval_excludes_zero_positive(zero.get("delta_color_nll_condition_minus_full", {})) or interval_excludes_zero_positive(zero.get("delta_total_loss_condition_minus_full", {}))
    query_support_loss = interval_excludes_zero_positive(query.get("delta_color_nll_condition_minus_full", {})) or interval_excludes_zero_positive(query.get("delta_total_loss_condition_minus_full", {}))
    zero_change = (zero.get("decoded_grid_change_rate", {}) or {}).get("mean") or 0.0
    query_change = (query.get("decoded_grid_change_rate", {}) or {}).get("mean") or 0.0
    if loss_signal and accuracy_signal and zero_support_loss and query_support_loss and max(zero_change, query_change) >= 0.10 and no_id_pass:
        return "ARC_GROUNDED_LEARNABLE_PREFLIGHT_SIGNAL"
    if loss_signal and not (zero_support_loss and query_support_loss):
        return "TRAINING_BUT_NO_SUPPORT_SIGNAL"
    if max(zero_change, query_change) >= 0.10 and not (loss_signal and accuracy_signal):
        return "SUPPORT_SENSITIVE_BUT_NO_CLEAR_GENERALIZATION_SIGNAL"
    return "INCONCLUSIVE"


def choose_next_action(verdict: str) -> str:
    if verdict == "NO_TRAINING_SIGNAL":
        return "A. Stop neural VRM because there is no clean training signal."
    if verdict == "TRAINING_BUT_NO_SUPPORT_SIGNAL":
        return "B. Audit training/data plumbing because weights change behavior but support outputs do not matter."
    if verdict == "ARC_GROUNDED_LEARNABLE_PREFLIGHT_SIGNAL":
        return "C. Replicate the exact preflight with two additional fixed initialization seeds because an ARC-grounded learnable preflight signal exists."
    return "D. Investigate why query-only or zeroed-support behavior dominates despite no-ID compliance."


def protected_integrity_status(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def protected(row: dict[str, Any]) -> bool:
        path = row["path"]
        return "/r5a2" in path or "/r5a3" in path or path.startswith("artifacts/r5a2") or path.startswith("artifacts/r5a3")

    before_map = {row["path"]: row["sha256"] for row in before["files"] if protected(row)}
    after_map = {row["path"]: row["sha256"] for row in after["files"] if protected(row)}
    changed = sorted(path for path, digest in before_map.items() if after_map.get(path) != digest)
    missing = sorted(path for path in before_map if path not in after_map)
    added = sorted(path for path in after_map if path not in before_map)
    return {
        "protected_r5a2_r5a3_unchanged": not changed and not missing and not added,
        "changed": changed,
        "missing": missing,
        "added": added,
        "protected_file_count": len(before_map),
    }


def write_condition_artifacts(*, artifact_dir: Path, condition_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]) -> None:
    for condition, (aggregate, records) in condition_results.items():
        write_eval_json(artifact_dir / METRIC_FILES[condition], condition_payload(aggregate, records))


def build_support_summary(*, full_records: list[dict[str, Any]], condition_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]], bootstrap_replicates: int) -> dict[str, Any]:
    ablations: dict[str, Any] = {}
    for index, condition in enumerate(["TRAINED_ZEROED_SUPPORT_OUTPUTS", "TRAINED_QUERY_ONLY", "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER"]):
        aggregate, records = condition_results[condition]
        if not aggregate.get("supported", False):
            ablations[condition] = {"condition": condition, "supported": False, "failure_reason": aggregate.get("failure_reason")}
            continue
        ablations[condition] = support_ablation_intervals(full_records, records, condition=condition, seed=INITIALIZATION_SEED + 10_000 + index * 100, replicates=bootstrap_replicates)
    return {"baseline_condition": "TRAINED_FULL_SUPPORT", "positive_values_favor": "TRAINED_FULL_SUPPORT", "bootstrap_replicates": bootstrap_replicates, "ablations": ablations}


def write_markdown_summary(path: Path, final_report: dict[str, Any]) -> None:
    lines = ["# R5A-4A ARC-Grounded Neural VRM Preflight", ""]
    for section in [
        "1. Frozen setup",
        "2. Measured trained-vs-fresh results",
        "3. Measured support-ablation results",
        "4. No-ID causal audit",
        "5. Exact verdict category",
        "6. What this establishes",
        "7. What this does not establish",
        "8. Narrowest justified next action",
    ]:
        lines.extend([f"## {section}", "", "```json", json.dumps(final_report[section], indent=2, sort_keys=True), "```", ""])
    write_text(path, "\n".join(lines).rstrip() + "\n")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, "-m", "vrm.twin_tasks.run_r5a4a_neural_vrm_arc_preflight", *sys.argv[1:]])
    interface_audit_path = artifact_dir / "model_interface_audit.json"
    if interface_audit_path.exists():
        interface_audit = json.loads(interface_audit_path.read_text(encoding="utf-8"))
        if interface_audit.get("audit_status") == "ID_FREE_AUDIT_FAIL":
            blocker = {"blocker": "ID_FREE_AUDIT_FAIL", "model_interface_audit": interface_audit}
            write_json(artifact_dir / "final_report.json", blocker)
            return blocker

    splits = build_r5a4a_splits(Path(args.data_root))
    if args.reuse_completed_training:
        checkpoint_selection = json.loads((artifact_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
        training_config = json.loads((artifact_dir / "training_config.json").read_text(encoding="utf-8"))
        training = {
            "artifact_dir": str(artifact_dir),
            "device": training_config["device"],
            "training_config": training_config,
            "initial_state_path": str(artifact_dir / "initial_state_before_any_optimizer_step.pt"),
            "selected_state_path": str(artifact_dir / "selected_trained_state.pt"),
            "selected_checkpoint_step": checkpoint_selection["selected_checkpoint_step"],
            "checkpoint_selection": checkpoint_selection,
            "wall_clock_training_seconds": None,
            "source_manifest_before": source_file_manifest(),
            "reused_completed_training": True,
        }
    else:
        training = run_training(splits=splits, artifact_dir=artifact_dir, max_steps=args.max_steps, batch_size=args.batch_size, device_arg=args.device, torch_threads=args.torch_threads, command=command)
    device = resolve_device(training["device"])
    sentinel_episodes = episodes_for_task_ids(splits.tasks_by_id, splits.sentinel_task_ids, source_split="sentinel")
    fresh_model, _fresh_state = load_state(Path(training["initial_state_path"]), device)
    trained_model, _trained_state = load_state(Path(training["selected_state_path"]), device)

    condition_results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    condition_results["FRESH_INIT"] = evaluate_condition(model=fresh_model, episodes=sentinel_episodes, condition="FRESH_INIT", device=device, shape_loss_weight=DEFAULT_SHAPE_LOSS_WEIGHT)
    for condition in ["TRAINED_FULL_SUPPORT", "TRAINED_ZEROED_SUPPORT_OUTPUTS", "TRAINED_QUERY_ONLY", "TRAINED_PERMUTED_SUPPORT_PAIR_ORDER"]:
        condition_results[condition] = evaluate_condition(model=trained_model, episodes=sentinel_episodes, condition=condition, device=device, shape_loss_weight=DEFAULT_SHAPE_LOSS_WEIGHT)
    write_condition_artifacts(artifact_dir=artifact_dir, condition_results=condition_results)

    fresh_aggregate, fresh_records = condition_results["FRESH_INIT"]
    full_aggregate, full_records = condition_results["TRAINED_FULL_SUPPORT"]
    trained_vs_fresh = paired_trained_vs_fresh_intervals(fresh_records, full_records, seed=INITIALIZATION_SEED, replicates=args.bootstrap_replicates)
    support_summary = build_support_summary(full_records=full_records, condition_results=condition_results, bootstrap_replicates=args.bootstrap_replicates)
    write_json(artifact_dir / "paired_bootstrap_intervals.json", {"trained_vs_fresh": trained_vs_fresh, "support_ablations": support_summary})
    write_json(artifact_dir / "support_ablation_summary.json", support_summary)

    no_id = run_no_id_audit(model=trained_model, episodes=sentinel_episodes, device=device, artifact_dir=artifact_dir, source_paths=source_paths_used_by_r5a4a())
    rerun_aggregate, rerun_records = evaluate_condition(model=trained_model, episodes=sentinel_episodes, condition="TRAINED_FULL_SUPPORT", device=device, shape_loss_weight=DEFAULT_SHAPE_LOSS_WEIGHT)
    determinism = {
        "rerun_condition": "TRAINED_FULL_SUPPORT",
        "stable_artifacts_match_excluding_runtime": stable_payload_without_runtime(condition_payload(full_aggregate, full_records)) == stable_payload_without_runtime(condition_payload(rerun_aggregate, rerun_records)),
        "original_fold_count": full_aggregate.get("fold_count"),
        "rerun_fold_count": rerun_aggregate.get("fold_count"),
    }
    write_json(artifact_dir / "determinism_verification.json", determinism)

    case_cards = render_case_cards(
        sentinel_episodes,
        fresh_records,
        full_records,
        {condition: records for condition, (_aggregate, records) in condition_results.items() if condition not in {"FRESH_INIT", "TRAINED_FULL_SUPPORT"}},
    )
    write_text(artifact_dir / "representative_case_cards.md", case_cards)
    runtime_profile = {
        "wall_clock_total_seconds": time.perf_counter() - start,
        "wall_clock_training_seconds": training["wall_clock_training_seconds"],
        "reused_completed_training": bool(training.get("reused_completed_training", False)),
        "condition_mean_inference_seconds": {condition: aggregate.get("mean_inference_runtime_per_fold_seconds") for condition, (aggregate, _records) in condition_results.items()},
    }
    write_json(artifact_dir / "runtime_profile.json", runtime_profile)

    query_supported = condition_results["TRAINED_QUERY_ONLY"][0].get("supported", False)
    verdict = determine_verdict(trained_vs_fresh=trained_vs_fresh, support_summary=support_summary, no_id_pass=no_id["no_id_causal_audit_pass"], query_only_supported=query_supported)
    source_after = source_file_manifest()
    write_text(artifact_dir / "source_manifest_after.txt", render_source_manifest_text(source_after))
    integrity = protected_integrity_status(training["source_manifest_before"], source_after)

    final_report = {
        "1. Frozen setup": {
            "architecture": "VRM V1.1 - Masked Query Spatial Evidence Rewrite",
            "experiment": "R5A-4A ARC-Grounded Neural VRM Preflight",
            "data_root": str(args.data_root),
            "initialization_seed": INITIALIZATION_SEED,
            "split_counts": split_counts(splits),
            "selected_checkpoint_step": training["selected_checkpoint_step"],
            "initial_state_path": training["initial_state_path"],
            "selected_trained_state_path": training["selected_state_path"],
            "pre_training_commitment": str(artifact_dir / "pre_training_commitment.json"),
            "reused_completed_training": bool(training.get("reused_completed_training", False)),
            "model_interface_audit_passed": interface_audit_path.exists() and json.loads(interface_audit_path.read_text(encoding="utf-8")).get("audit_status") == "ID_FREE_AUDIT_PASS",
        },
        "2. Measured trained-vs-fresh results": {"fresh_init": fresh_aggregate, "trained_full_support": full_aggregate, "paired_bootstrap_intervals": trained_vs_fresh},
        "3. Measured support-ablation results": {
            "condition_aggregates": {condition: aggregate for condition, (aggregate, _records) in condition_results.items() if condition.startswith("TRAINED_")},
            "support_ablation_summary": support_summary,
        },
        "4. No-ID causal audit": no_id,
        "5. Exact verdict category": {"verdict": verdict, "query_only_supported": query_supported},
        "6. What this establishes": {
            "statement": "This preflight measures trained-vs-fresh behavior and support-output sensitivity on a fixed real-ARC sentinel leave-one-out set under a no-ID input audit.",
            "determinism_verification": determinism,
            "source_artifact_integrity": integrity,
        },
        "7. What this does not establish": {"not_claimed": ["ARC solving", "reasoning", "abstraction", "hidden tools", "generalization beyond this frozen sentinel preflight", "synthetic task performance"]},
        "8. Narrowest justified next action": {"next_action": choose_next_action(verdict), "do_not_automatically_start": True},
    }
    write_json(artifact_dir / "final_report.json", final_report)
    write_markdown_summary(PROJECT_ROOT / "docs" / "R5A4A_NEURAL_VRM_ARC_GROUNDED_PREFLIGHT.md", final_report)
    write_markdown_summary(artifact_dir / "R5A4A_NEURAL_VRM_ARC_GROUNDED_PREFLIGHT_summary.md", final_report)
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R5A-4A ARC-grounded neural VRM preflight")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/VRM/arc-prize-2026-arc-agi-2"))
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--max-steps", type=int, default=MAX_OPTIMIZER_STEPS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--reuse-completed-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run_preflight(args), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
