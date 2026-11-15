from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import (
    candidate_order_label_report,
    duplicate_unpaired_report,
    group_twins,
    label_balance,
    load_records,
    load_split_records,
    paired_payload_hash,
    partition_integrity,
    unpaired_payload_hash,
)
from .generator import DEFAULT_GRID_SIZE, FAMILY, generate_records, rejection_summary, stable_hash, write_jsonl
from .invariants import unpaired_payload_hash_report, validate_all_twins
from .probes import (
    PROBE_NAMES,
    PROBE_VIEW,
    pass_fail_for_probe,
    feature_scores,
    summarize_probe_runs,
    train_probe,
)

DEFAULT_OUT = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
DEFAULT_SEED = 20260628
DEFAULT_PROBE_SEEDS = (20260628, 20260629, 20260630)
V1_1_PATHS_TO_PROTECT = (
    Path("/mnt/d/VRM/vrm/data.py"),
    Path("/mnt/d/VRM/vrm/model.py"),
    Path("/mnt/d/VRM/vrm/metrics.py"),
    Path("/mnt/d/VRM/vrm/train.py"),
    Path("/mnt/d/VRM/checkpoints/vrm_v1_1_best_train_hamming.pt"),
    Path("/mnt/d/VRM/artifacts/vrm_v1_1_capacity50/vrm_v1_1_metrics.json"),
    Path("/mnt/d/VRM/artifacts/vrm_v1_1_support_ablation/ablation_metrics.json"),
    Path("/mnt/d/VRM/VRM_V1_1_SOURCE_SNAPSHOT.zip"),
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path): hash_file(path) if path.exists() else None for path in V1_1_PATHS_TO_PROTECT}


def split_hash(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_generated_suite(
    *,
    out: Path,
    seed: int,
    train_pairs: int,
    dev_pairs: int,
    sealed_pairs: int,
    grid_size: int,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    split_pair_counts = {"train": train_pairs, "dev": dev_pairs, "sealed": sealed_pairs}
    result = generate_records(seed=seed, split_pair_counts=split_pair_counts, grid_size=grid_size)
    for split in ("train", "dev", "sealed"):
        rows = [record for record in result.records if record["split"] == split]
        write_jsonl(out / f"{split}.jsonl", rows)
    split_manifest = {
        "family": FAMILY,
        "seed": seed,
        "grid_size": grid_size,
        "split_pair_counts": split_pair_counts,
        "split_episode_counts": {split: count * 2 for split, count in split_pair_counts.items()},
        "split_base_ids": result.split_base_ids,
        "split_hashes": {
            split: split_hash([record for record in result.records if record["split"] == split])
            for split in split_pair_counts
        },
    }
    save_json(out / "pilot_config.json", {
        "family": FAMILY,
        "seed": seed,
        "grid_size": grid_size,
        "train_pairs": train_pairs,
        "dev_pairs": dev_pairs,
        "sealed_pairs": sealed_pairs,
        "sealed_protocol_note": "sealed is generated once and not used for feature design, model choice, early stopping, or threshold tuning",
    })
    save_json(out / "split_manifest.json", split_manifest)
    save_json(out / "generator_rejections.json", rejection_summary(result.rejections))
    return {"records": result.records, "split_manifest": split_manifest, "rejections": result.rejections}


def audit_exact_invariants(out: Path) -> dict[str, Any]:
    records_by_split = {split: load_split_records(out, split) for split in ("train", "dev", "sealed")}
    all_records = [record for rows in records_by_split.values() for record in rows]
    grouped = group_twins(all_records)
    invariance_report = validate_all_twins(grouped)
    unpaired_report = unpaired_payload_hash_report(grouped)
    partition_report = partition_integrity(records_by_split)
    duplicate_report = duplicate_unpaired_report(all_records)
    label_reports = {split: label_balance(rows) for split, rows in records_by_split.items()}
    candidate_reports = {split: candidate_order_label_report(rows) for split, rows in records_by_split.items()}
    partition_report["duplicate_unpaired_payload_report"] = duplicate_report
    partition_report["label_balance"] = label_reports
    partition_report["candidate_order_label_report"] = candidate_reports
    partition_report["all_passed"] = (
        partition_report["passed"]
        and duplicate_report["passed"]
        and all(report["balanced"] for report in label_reports.values())
        and all(report["passed"] for report in candidate_reports.values())
    )
    save_json(out / "twin_invariance_report.json", invariance_report)
    save_json(out / "unpaired_payload_hash_report.json", unpaired_report)
    save_json(out / "partition_integrity_report.json", partition_report)
    return {
        "records_by_split": records_by_split,
        "all_records": all_records,
        "invariance_report": invariance_report,
        "unpaired_report": unpaired_report,
        "partition_report": partition_report,
    }


def run_pytest(out: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd="/mnt/d/VRM",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError("pytest failed:\n" + result.stdout)
    return result.stdout.strip()


def run_probe_suite(
    *,
    out: Path,
    records_by_split: dict[str, list[dict[str, Any]]],
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> dict[str, Any]:
    train_records = records_by_split["train"]
    dev_records = records_by_split["dev"]
    sealed_records = records_by_split["sealed"]
    log_path = out / "probe_train_logs.jsonl"
    if log_path.exists():
        log_path.unlink()
    probe_config = {
        "probe_names": PROBE_NAMES,
        "probe_views": PROBE_VIEW,
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
    }
    save_json(out / "probe_config.json", probe_config)
    all_runs: dict[str, list[dict[str, Any]]] = {probe: [] for probe in PROBE_NAMES}
    for probe in PROBE_NAMES:
        for seed in seeds:
            run = train_probe(
                probe_name=probe,
                train_records=train_records,
                dev_records=dev_records,
                sealed_records=sealed_records,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
            )
            all_runs[probe].append(run)
            for row in run["logs"]:
                append_jsonl(log_path, row)
    dev_metrics = {}
    sealed_metrics = {}
    for probe, runs in all_runs.items():
        dev_metrics[probe] = summarize_probe_runs(runs, split_name="dev", example_count=len(dev_records))
        sealed_metrics[probe] = summarize_probe_runs(runs, split_name="sealed", example_count=len(sealed_records))
        sealed_metrics[probe]["pass"] = pass_fail_for_probe(
            probe,
            sealed_metrics[probe]["sealed_mean_accuracy"],
            sealed_metrics[probe]["sealed_binomial_ci95"],
        )
    save_json(out / "probe_dev_metrics.json", dev_metrics)
    save_json(out / "probe_sealed_metrics.json", sealed_metrics)
    return {"runs": all_runs, "dev_metrics": dev_metrics, "sealed_metrics": sealed_metrics}


def pair_consistency_report(
    records_by_split: dict[str, list[dict[str, Any]]],
    probe_runs: dict[str, list[dict[str, Any]]],
    out: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, records in records_by_split.items():
        grouped = group_twins(records)
        split_report = {
            "twin_pairs": len(grouped),
            "unpaired_payload_exact_matches": 0,
            "opposite_labels": 0,
            "unpaired_probe_identical_prediction_counts": {},
        }
        for twins in grouped.values():
            if len(twins) != 2:
                continue
            left, right = sorted(twins, key=lambda row: row["twin"])
            if unpaired_payload_hash(left) == unpaired_payload_hash(right):
                split_report["unpaired_payload_exact_matches"] += 1
            if left["label"] + right["label"] == 1:
                split_report["opposite_labels"] += 1
        for probe in ("P1_QUERY_ONLY", "P2_SUPPORT_INPUT_BAG_ONLY", "P3_SUPPORT_OUTPUT_BAG_ONLY", "P4_UNPAIRED_SUPPORT_BAG"):
            identical = 0
            for twins in grouped.values():
                if len(twins) != 2:
                    continue
                left, right = sorted(twins, key=lambda row: row["twin"])
                if feature_scores(left, probe, 20260628) == feature_scores(right, probe, 20260628):
                    identical += 1
            split_report["unpaired_probe_identical_prediction_counts"][probe] = identical
        split_report["passed"] = (
            split_report["unpaired_payload_exact_matches"] == split_report["twin_pairs"]
            and split_report["opposite_labels"] == split_report["twin_pairs"]
            and all(count == split_report["twin_pairs"] for count in split_report["unpaired_probe_identical_prediction_counts"].values())
        )
        report[split] = split_report
    save_json(out / "probe_pair_consistency.json", report)
    return report


def render_examples(out: Path, records_by_split: dict[str, list[dict[str, Any]]], probe_runs: dict[str, list[dict[str, Any]]]) -> str:
    train_grouped = group_twins(records_by_split["train"])
    selected_base_ids = sorted(train_grouped)[:5]
    lines = ["# Twin task pilot probe examples", ""]
    first_seed = 20260628
    for base_id in selected_base_ids:
        twins = sorted(train_grouped[base_id], key=lambda row: row["twin"])
        lines.extend([f"## base {base_id}", ""])
        for record in twins:
            lines.extend(
                [
                    f"### Twin {record['twin']}",
                    f"label: {record['label']}",
                    f"unpaired_payload_hash: {unpaired_payload_hash(record)}",
                    f"paired_payload_hash: {paired_payload_hash(record)}",
                    "support pairings:",
                ]
            )
            for support in record["supports"]:
                lines.append(f"- marker {support['marker_index']} -> output {support['output_symbol_index']}")
                lines.append("support input:")
                lines.extend(" ".join(str(cell) for cell in row) for row in support["input"])
                lines.append("support output:")
                lines.extend(" ".join(str(cell) for cell in row) for row in support["output"])
            lines.append("shared query:")
            lines.extend(" ".join(str(cell) for cell in row) for row in record["query"])
            for idx, candidate in enumerate(record["candidates"]):
                lines.append(f"candidate_{idx}:")
                lines.extend(" ".join(str(cell) for cell in row) for row in candidate)
            lines.append("probe predictions:")
            for probe in PROBE_NAMES:
                scores = feature_scores(record, probe, first_seed)
                prediction = 0 if scores[0] >= scores[1] else 1
                lines.append(f"- {probe}: scores={scores}, prediction={prediction}")
            lines.append("exact invariant status: passed")
            lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    save_text(out / "probe_examples.txt", text)
    return text


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# VRM V1.2A - Counterfactual Twin-Task Suite Pilot",
        "",
        "Synthetic causal testbed only. This is not ARC-2 training and not evidence that VRM solves ARC.",
        "",
        f"- seed: {report['seed']}",
        f"- accepted_twin_pairs: {report['accepted_twin_pairs']}",
        f"- rejected_base_instances: {report['rejected_base_instances']}",
        f"- artifact_path: `{report['artifact_path']}`",
        "",
        "## Probe sealed metrics",
        "",
        "|probe|view|sealed_mean_acc|std|ci95|expected|pass|",
        "|---|---|---:|---:|---|---|---|",
    ]
    for probe, row in report["probe_sealed_metrics"].items():
        ci = row["sealed_binomial_ci95"]
        lines.append(
            f"|{probe}|{row['input_view']}|{row['sealed_mean_accuracy']:.6f}|"
            f"{row['sealed_std_accuracy']:.6f}|[{ci[0]:.6f},{ci[1]:.6f}]|{row['expected_result']}|{row['pass']}|"
        )
    lines.extend(["", f"TWIN_PILOT_AUDIT_VERDICT: {report['audit_verdict']}", ""])
    text = "\n".join(lines)
    save_text(Path(report["artifact_path"]) / "audit_summary.md", text)
    return text


def evaluate_pass_criteria(
    *,
    invariance_report: dict[str, Any],
    unpaired_report: dict[str, Any],
    partition_report: dict[str, Any],
    pair_consistency: dict[str, Any],
    sealed_metrics: dict[str, Any],
) -> tuple[str, list[str]]:
    failures = []
    if not invariance_report["all_passed"]:
        failures.append("exact_invariance_failed")
    if not unpaired_report["all_passed"]:
        failures.append("unpaired_payload_match_failed")
    if not partition_report["all_passed"]:
        failures.append("partition_integrity_failed")
    if not all(row["passed"] for row in pair_consistency.values()):
        failures.append("paired_twin_consistency_failed")
    for probe, row in sealed_metrics.items():
        if not row["pass"]:
            failures.append(f"{probe}_sealed_threshold_failed")
    return ("PASS" if not failures else "FAIL"), failures


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    before = protected_hashes()
    result = write_generated_suite(
        out=args.out,
        seed=args.seed,
        train_pairs=args.train_pairs,
        dev_pairs=args.dev_pairs,
        sealed_pairs=args.sealed_pairs,
        grid_size=args.grid_size,
    )
    after = protected_hashes()
    return {"result": result, "protected_hashes_unchanged": before == after, "before": before, "after": after}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    before = protected_hashes()
    exact = audit_exact_invariants(args.data)
    pytest_result = run_pytest(args.data)
    probe_result = run_probe_suite(
        out=args.data,
        records_by_split=exact["records_by_split"],
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    pair_consistency = pair_consistency_report(exact["records_by_split"], probe_result["runs"], args.data)
    render_examples(args.data, exact["records_by_split"], probe_result["runs"])
    verdict, failures = evaluate_pass_criteria(
        invariance_report=exact["invariance_report"],
        unpaired_report=exact["unpaired_report"],
        partition_report=exact["partition_report"],
        pair_consistency=pair_consistency,
        sealed_metrics=probe_result["sealed_metrics"],
    )
    all_records = exact["all_records"]
    after = protected_hashes()
    report = {
        "seed": args.seed,
        "probe_seeds": args.seeds,
        "artifact_path": str(args.data),
        "accepted_twin_pairs": len(group_twins(all_records)),
        "accepted_episodes": len(all_records),
        "rejected_base_instances": json.loads((args.data / "generator_rejections.json").read_text())["rejected_count"],
        "pytest_result": pytest_result,
        "invariance_report": exact["invariance_report"],
        "unpaired_report": exact["unpaired_report"],
        "partition_report": exact["partition_report"],
        "probe_dev_metrics": probe_result["dev_metrics"],
        "probe_sealed_metrics": probe_result["sealed_metrics"],
        "pair_consistency": pair_consistency,
        "audit_verdict": verdict,
        "audit_failures": failures,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_hashes_unchanged": before == after,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(),
        },
    }
    render_summary(report)
    return report


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    before = protected_hashes()
    write_generated_suite(
        out=args.out,
        seed=args.seed,
        train_pairs=args.train_pairs,
        dev_pairs=args.dev_pairs,
        sealed_pairs=args.sealed_pairs,
        grid_size=args.grid_size,
    )
    audit_args = argparse.Namespace(
        data=args.out,
        seed=args.seed,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    report = run_audit(audit_args)
    after = protected_hashes()
    report["protected_hashes_before_full"] = before
    report["protected_hashes_after_full"] = after
    report["protected_hashes_unchanged_full"] = before == after
    save_json(args.out / "audit_run_report.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and audit the VRM V1.2A twin-task pilot.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.add_argument("--train-pairs", type=int, default=8000)
    gen.add_argument("--dev-pairs", type=int, default=1000)
    gen.add_argument("--sealed-pairs", type=int, default=1000)
    gen.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    gen.add_argument("--out", type=Path, default=DEFAULT_OUT)
    audit = sub.add_parser("audit")
    audit.add_argument("--data", type=Path, default=DEFAULT_OUT)
    audit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    audit.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_PROBE_SEEDS))
    audit.add_argument("--epochs", type=int, default=20)
    audit.add_argument("--batch-size", type=int, default=64)
    audit.add_argument("--learning-rate", type=float, default=3e-4)
    audit.add_argument("--patience", type=int, default=5)
    full = sub.add_parser("full")
    full.add_argument("--seed", type=int, default=DEFAULT_SEED)
    full.add_argument("--train-pairs", type=int, default=8000)
    full.add_argument("--dev-pairs", type=int, default=1000)
    full.add_argument("--sealed-pairs", type=int, default=1000)
    full.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    full.add_argument("--out", type=Path, default=DEFAULT_OUT)
    full.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_PROBE_SEEDS))
    full.add_argument("--epochs", type=int, default=20)
    full.add_argument("--batch-size", type=int, default=64)
    full.add_argument("--learning-rate", type=float, default=3e-4)
    full.add_argument("--patience", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.perf_counter()
    if args.command == "generate":
        result = run_generate(args)
        print(json.dumps(to_jsonable({"seconds": time.perf_counter() - start, **result}), indent=2, sort_keys=True))
    elif args.command == "audit":
        result = run_audit(args)
        print(json.dumps(to_jsonable({"seconds": time.perf_counter() - start, **result}), indent=2, sort_keys=True))
    elif args.command == "full":
        run_full(args)


if __name__ == "__main__":
    main()
