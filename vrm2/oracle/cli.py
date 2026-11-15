from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.program_schema import TYPED_PROGRAM_SCHEMA_VERSION
from vrm2.data.serialization import canonical_json_bytes
from vrm2.oracle.controls import run_controls
from vrm2.oracle.dataset import build_oracle_examples, load_oracle_corpus, write_oracle_corpus
from vrm2.oracle.diagnostics import run_rollout_diagnostics
from vrm2.oracle.evaluate import evaluate_model
from vrm2.oracle.train import (
    load_checkpoint,
    load_oracle_config,
    load_stage0_for_oracle,
    train_one_seed,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _manifest_path(corpus_path: Path) -> Path:
    return corpus_path.with_name(f"{corpus_path.stem}.manifest.json")


def _load_pinned_corpus(path: Path) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    examples = load_oracle_corpus(path, manifest["dataset_sha256"])
    if len(examples) != int(manifest["num_examples"]):
        raise ValueError("pinned corpus count differs from its manifest")
    return examples, manifest


def command_generate_sealed(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    stage0 = load_stage0_for_oracle(config, REPOSITORY_ROOT)
    examples = build_oracle_examples(
        stage0,
        "sealed",
        int(config["data"]["sealed_examples"]),
        int(config["data"]["sealed_seed"]),
    )
    length_counts = {
        str(length): sum(example.program_length == length for example in examples) for length in (2, 3)
    }
    expected = config["data"].get("sealed_length_counts")
    if expected is not None and length_counts != {str(key): value for key, value in expected.items()}:
        raise ValueError(f"sealed length balance mismatch: {length_counts} != {expected}")
    versions = {
        "generator_version": stage0["generator_version"],
        "operation_registry_version": stage0["operation_registry_version"],
        "split_version": stage0["split_version"],
    }
    return write_oracle_corpus(
        args.output,
        examples,
        dataset_seed=int(config["data"]["sealed_seed"]),
        versions=versions,
    )


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    stage0 = load_stage0_for_oracle(config, REPOSITORY_ROOT)
    sealed, manifest = _load_pinned_corpus(args.sealed_corpus)
    report = train_one_seed(
        config,
        stage0,
        sealed,
        args.seed,
        args.checkpoint,
        args.report,
    )
    report["sealed_corpus_sha256"] = manifest["dataset_sha256"]
    report["typed_program_schema_version"] = TYPED_PROGRAM_SCHEMA_VERSION
    args.report.write_bytes(canonical_json_bytes(report) + b"\n")
    return report


def _evaluation_bundle(
    config: Mapping[str, Any],
    checkpoint: Path,
    sealed_corpus: Path,
) -> tuple[dict[str, Any], torch.nn.Module, tuple[Any, ...], torch.device]:
    stage0 = load_stage0_for_oracle(config, REPOSITORY_ROOT)
    sealed, _ = _load_pinned_corpus(sealed_corpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    batch_size = int(config["evaluation"]["batch_size"])
    train_eval = build_oracle_examples(
        stage0,
        "train",
        int(config["data"]["train_eval_examples"]),
        int(config["data"]["train_eval_seed"]),
    )
    dev = build_oracle_examples(
        stage0,
        "dev",
        int(config["data"]["dev_examples"]),
        int(config["data"]["dev_seed"]),
    )
    sealed_len2 = tuple(example for example in sealed if example.program_length == 2)
    sealed_len3 = tuple(example for example in sealed if example.program_length == 3)
    metrics = {
        "train_signature": evaluate_model(model, train_eval, batch_size, device),
        "dev_len2": evaluate_model(model, dev, batch_size, device),
        "sealed_len2": evaluate_model(model, sealed_len2, batch_size, device),
        "sealed_len3": evaluate_model(model, sealed_len3, batch_size, device),
        "sealed_all": evaluate_model(model, sealed, batch_size, device),
    }
    return metrics, model, sealed, device


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    metrics, _, _, _ = _evaluation_bundle(config, args.checkpoint, args.sealed_corpus)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json_bytes(metrics) + b"\n")
    return metrics


def command_controls(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    _, model, sealed, device = _evaluation_bundle(config, args.checkpoint, args.sealed_corpus)
    maximum = int(config["evaluation"].get("control_examples", len(sealed)))
    examples = tuple(sealed[:maximum])
    controls = run_controls(
        model,
        examples,
        int(config["evaluation"]["batch_size"]),
        device,
        args.seed,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json_bytes(controls) + b"\n")
    return controls


def command_rollout_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    _, model, sealed, device = _evaluation_bundle(
        config, args.checkpoint, args.sealed_corpus
    )
    diagnostics = run_rollout_diagnostics(
        model,
        sealed,
        int(config["evaluation"]["batch_size"]),
        device,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json_bytes(diagnostics) + b"\n")
    return diagnostics


def pilot_allows_certification(report: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[bool, str]:
    len2 = float(report["sealed_len2"]["exact"])
    len3 = float(report["sealed_len3"]["exact"])
    train = float(report["train_signature"]["exact"])
    if len2 < 0.70 or len3 < 0.50:
        return False, "hard pilot stop threshold triggered"
    thresholds = config["certification"]["pilot_proceed_thresholds"]
    if (
        train < float(thresholds["train_signature"])
        or len2 < float(thresholds["sealed_len2"])
        or len3 < float(thresholds["sealed_len3"])
    ):
        return False, "pilot is not plausibly approaching the frozen 0.95 gate"
    return True, "pilot satisfies the preregistered continuation policy"


def command_certify(args: argparse.Namespace) -> dict[str, Any]:
    config = load_oracle_config(args.config)
    pilot_report = json.loads(args.pilot_report.read_text(encoding="utf-8"))
    allowed, reason = pilot_allows_certification(pilot_report, config)
    if not allowed:
        result = {"certification_ran": False, "reason": reason, "verdict": "REDESIGN_REQUIRED"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "certification_summary.json").write_bytes(canonical_json_bytes(result) + b"\n")
        return result

    stage0 = load_stage0_for_oracle(config, REPOSITORY_ROOT)
    sealed, manifest = _load_pinned_corpus(args.sealed_corpus)
    seed_reports = []
    control_reports = []
    diagnostic_reports = []
    for seed in config["certification"]["seeds"]:
        checkpoint = args.output_dir / f"oracle_interpreter_seed_{seed}.pt"
        report_path = args.output_dir / f"oracle_interpreter_seed_{seed}.json"
        report = train_one_seed(config, stage0, sealed, int(seed), checkpoint, report_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(checkpoint, device)
        control_limit = int(config["evaluation"].get("control_examples", len(sealed)))
        controls = run_controls(
            model,
            sealed[:control_limit],
            int(config["evaluation"]["batch_size"]),
            device,
            int(seed),
        )
        controls_path = args.output_dir / f"oracle_interpreter_controls_seed_{seed}.json"
        controls_path.write_bytes(canonical_json_bytes(controls) + b"\n")
        seed_reports.append(report)
        control_reports.append(controls)
        if config.get("model_version", "v0") == "v1":
            diagnostics = run_rollout_diagnostics(
                model,
                sealed,
                int(config["evaluation"]["batch_size"]),
                device,
            )
            diagnostics_path = args.output_dir / f"oracle_interpreter_diagnostics_seed_{seed}.json"
            diagnostics_path.write_bytes(canonical_json_bytes(diagnostics) + b"\n")
            diagnostic_reports.append(diagnostics)

    metric_paths = {
        "train_signature": ("train_signature", "exact"),
        "dev_len2": ("dev_len2", "exact"),
        "sealed_len2": ("sealed_len2", "exact"),
        "sealed_len3": ("sealed_len3", "exact"),
        "sealed_all": ("sealed_all", "exact"),
    }
    aggregate = {}
    for label, (section, metric) in metric_paths.items():
        values = [float(report[section][metric]) for report in seed_reports]
        aggregate[label] = {
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    gate = config["certification"]
    gate_evaluation = []
    for name in ("sealed_len2", "sealed_len3", "sealed_all"):
        observed = float(aggregate[name]["minimum"])
        threshold = float(gate["exact_threshold"])
        gate_evaluation.append(
            {
                "metric": f"{name}.minimum",
                "observed": observed,
                "operator": ">=",
                "threshold": threshold,
                "pass": observed >= threshold,
            }
        )
    accuracy_pass = all(item["pass"] for item in gate_evaluation)
    control_pass = True
    for report, controls in zip(seed_reports, control_reports, strict=True):
        correct = float(controls["correct"]["exact"])
        for control_name in ("random", "swap", "reverse", "null"):
            control = controls[control_name]
            if control["num_examples"] == 0:
                control_pass = False
                continue
            exact_drop = correct - float(control["exact"])
            exact_drop_threshold = float(gate["minimum_control_exact_drop"])
            output_change = float(control["output_change_rate"])
            output_change_threshold = float(gate["minimum_control_output_change_rate"])
            exact_drop_pass = exact_drop >= exact_drop_threshold
            output_change_pass = output_change >= output_change_threshold
            control_pass = control_pass and exact_drop_pass and output_change_pass
            gate_evaluation.extend(
                [
                    {
                        "metric": f"seed_{report['seed']}.{control_name}.exact_drop",
                        "observed": exact_drop,
                        "operator": ">=",
                        "threshold": exact_drop_threshold,
                        "pass": exact_drop_pass,
                    },
                    {
                        "metric": f"seed_{report['seed']}.{control_name}.output_change_rate",
                        "observed": output_change,
                        "operator": ">=",
                        "threshold": output_change_threshold,
                        "pass": output_change_pass,
                    },
                ]
            )
    verdict = "ORACLE_INTERPRETER_PASS" if accuracy_pass and control_pass else "ORACLE_INTERPRETER_FAIL"
    result = {
        "certification_ran": True,
        "pilot_gate_reason": reason,
        "sealed_corpus_sha256": manifest["dataset_sha256"],
        "seeds": list(config["certification"]["seeds"]),
        "aggregate": aggregate,
        "seed_reports": seed_reports,
        "controls": control_reports,
        "diagnostics": diagnostic_reports,
        "gate_evaluation": gate_evaluation,
        "accuracy_gate_pass": accuracy_pass,
        "control_gate_pass": control_pass,
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "certification_summary.json").write_bytes(canonical_json_bytes(result) + b"\n")
    return result


def _summary(command: str, result: Mapping[str, Any]) -> dict[str, Any]:
    if command == "generate-sealed":
        return dict(result)
    if command == "train":
        return {
            "seed": result["seed"],
            "parameter_count": result["parameter_count"],
            "train_signature_exact": result["train_signature"]["exact"],
            "dev_len2_exact": result["dev_len2"]["exact"],
            "sealed_len2_exact": result["sealed_len2"]["exact"],
            "sealed_len3_exact": result["sealed_len3"]["exact"],
            "sealed_all_exact": result["sealed_all"]["exact"],
            "training_seconds": result["training_seconds"],
            "peak_vram_bytes": result["peak_vram_bytes"],
        }
    return dict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM v2 oracle typed-program interpreter diagnostic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-sealed")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--sealed-corpus", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--sealed-corpus", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    controls = subparsers.add_parser("controls")
    controls.add_argument("--config", type=Path, required=True)
    controls.add_argument("--checkpoint", type=Path, required=True)
    controls.add_argument("--sealed-corpus", type=Path, required=True)
    controls.add_argument("--seed", type=int, required=True)
    controls.add_argument("--output", type=Path)

    diagnostics = subparsers.add_parser("rollout-diagnostics")
    diagnostics.add_argument("--config", type=Path, required=True)
    diagnostics.add_argument("--checkpoint", type=Path, required=True)
    diagnostics.add_argument("--sealed-corpus", type=Path, required=True)
    diagnostics.add_argument("--output", type=Path)

    certify = subparsers.add_parser("certify")
    certify.add_argument("--config", type=Path, required=True)
    certify.add_argument("--sealed-corpus", type=Path, required=True)
    certify.add_argument("--pilot-report", type=Path, required=True)
    certify.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "generate-sealed": command_generate_sealed,
        "train": command_train,
        "evaluate": command_evaluate,
        "controls": command_controls,
        "rollout-diagnostics": command_rollout_diagnostics,
        "certify": command_certify,
    }
    result = commands[args.command](args)
    print(json.dumps(_summary(args.command, result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
