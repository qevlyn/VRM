from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.microworld import load_config as load_stage0_config
from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.controls_v1 import run_controls_v1
from vrm2.inductor.episodes import load_query_episodes, load_support_episodes
from vrm2.inductor.episodes_v1 import prepare_v1_episode_corpora
from vrm2.inductor.evaluate import evaluate_model_on_episodes as evaluate_v0
from vrm2.inductor.evaluate_v1 import (
    evaluate_model_v1,
    predict_programs_v1,
    write_predicted_programs_v1,
)
from vrm2.inductor.execution_v1 import evaluate_induced_program_file_v1
from vrm2.inductor.l3_coverage import (
    build_coverage_corpus,
    deterministic_corpus_bytes,
    write_coverage_corpus,
)
from vrm2.inductor.train import load_inductor_checkpoint
from vrm2.inductor.train_v1 import (
    load_checkpoint_v1,
    load_inductor_v1_config,
    train_one_seed_v1,
)
from vrm2.oracle.dataset import load_oracle_corpus
from vrm2.oracle.train import build_run_datasets, load_checkpoint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _build_datasets(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stage0 = load_stage0_config(REPOSITORY_ROOT / config["stage0_config"])
    sealed_path = Path(config["data"]["sealed_corpus"])
    sealed_manifest = json.loads(
        sealed_path.with_name(f"{sealed_path.stem}.manifest.json").read_text(encoding="utf-8")
    )
    sealed = load_oracle_corpus(sealed_path, config["data"]["sealed_corpus_sha256"])
    return stage0, build_run_datasets(config, stage0, sealed)


def command_prepare_episodes(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    _, datasets = _build_datasets(config)
    return prepare_v1_episode_corpora(
        datasets,
        Path(config["episodes"]["runtime_dir"]),
        Path(config["episodes"]["frozen_v0_dir"]),
        config["episodes"]["assembly_seeds"],
    )


def command_generate_coverage(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    stage0, datasets = _build_datasets(config)
    options = {
        "seed": int(config["l3_coverage"]["seed"]),
        "episode_count": int(config["l3_coverage"]["episode_count"]),
    }
    first = build_coverage_corpus(stage0, datasets, **options)
    second = build_coverage_corpus(stage0, datasets, **options)
    identical = (
        deterministic_corpus_bytes(first[2]) == deterministic_corpus_bytes(second[2])
        and first[3] == second[3]
        and first[4] == second[4]
    )
    if not identical or len(first[0]) != 512:
        raise RuntimeError("INDUCTOR_V1_INVALID_L3_CORPUS")
    return write_coverage_corpus(
        Path(config["l3_coverage"]["runtime_dir"]),
        first[0],
        first[1],
        first[2],
        first[3],
        first[4],
        deterministic_regeneration_identical=identical,
    )


def _runtime(config: Mapping[str, Any]) -> Path:
    return Path(config["outputs"]["runtime_dir"])


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    runtime = _runtime(config)
    checkpoint = args.checkpoint or runtime / f"inductor_v1_seed_{args.seed}.pt"
    report = args.report or runtime / f"inductor_v1_seed_{args.seed}.json"
    return train_one_seed_v1(config, args.seed, checkpoint, report)


def _phase_a_record(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["gates_runtime_dir"]) / "phase_a_verdict.json" if "gates_runtime_dir" in config else Path("artifacts/vrm2/inductor/runtime/v1/pilot_seed_20260713/phase_a_verdict.json")
    if not path.is_file():
        raise RuntimeError("coverage L3 is sealed until Phase A has a recorded result")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record["phase_a_pass"]:
        raise RuntimeError("coverage L3 must remain unopened because Phase A failed")
    return record


def _record_coverage_access(config: Mapping[str, Any], label: str) -> None:
    audit_path = Path(config["l3_coverage"]["runtime_dir"]) / "access_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["phase_b_evaluation_count"] = int(audit["phase_b_evaluation_count"]) + 1
    audit.setdefault("phase_b_access_labels", []).append(label)
    _write_json(audit_path, audit)


def _split_paths(config: Mapping[str, Any], split: str) -> tuple[Path, Path]:
    if split == "coverage_l3":
        _phase_a_record(config)
        directory = Path(config["l3_coverage"]["runtime_dir"])
        return directory / "coverage_l3_supports.jsonl", directory / "coverage_l3_queries.jsonl"
    directory = Path(config["episodes"]["runtime_dir"])
    return directory / f"{split}_supports.jsonl", directory / f"{split}_queries.jsonl"


def command_induce(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    support_path, _ = _split_paths(config, args.split)
    if args.split == "coverage_l3":
        _record_coverage_access(config, f"induce:{config['ablation']}")
    supports = load_support_episodes(args.episodes or support_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_v1(args.checkpoint, device)
    lengths, programs, confidence, seconds = predict_programs_v1(
        model, supports, int(config["evaluation"]["batch_size"]), device
    )
    output = args.output or _runtime(config) / f"predicted_programs_{args.split}.jsonl"
    result = write_predicted_programs_v1(
        output, supports, lengths, programs, confidence, args.checkpoint
    )
    _write_json(output.with_suffix(".manifest.json"), result)
    _write_json(
        output.with_suffix(".runtime.json"),
        {
            "induction_seconds": seconds,
            "induction_ms_per_episode": seconds * 1000.0 / len(supports),
        },
    )
    return result


def command_evaluate_induced(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    _, query_path = _split_paths(config, args.split)
    if args.split == "coverage_l3":
        _record_coverage_access(config, f"execute:{config['ablation']}")
    program_path = args.programs or _runtime(config) / f"predicted_programs_{args.split}.jsonl"
    measured = evaluate_induced_program_file_v1(
        args.queries or query_path,
        program_path,
        Path(config["data"]["interpreter_checkpoint"]),
        int(config["evaluation"]["batch_size"]),
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    output = args.output or _runtime(config) / f"execution_{args.split}.json"
    semantic = {
        key: value
        for key, value in measured.items()
        if key not in ("execution_seconds", "execution_ms_per_grid")
    }
    _write_json(output, semantic)
    _write_json(
        output.with_suffix(".runtime.json"),
        {
            "execution_seconds": measured["execution_seconds"],
            "execution_ms_per_grid": measured["execution_ms_per_grid"],
        },
    )
    return semantic


def _load_frozen_interpreter(config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    interpreter, _ = load_checkpoint(Path(config["data"]["interpreter_checkpoint"]), device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    return interpreter


def command_evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    support_path, query_path = _split_paths(config, args.split)
    if args.split == "coverage_l3":
        _record_coverage_access(config, f"evaluate-model:{config['ablation']}")
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_v1(args.checkpoint, device)
    result = evaluate_model_v1(
        model,
        _load_frozen_interpreter(config, device),
        supports,
        queries,
        int(config["evaluation"]["batch_size"]),
        device,
        include_loo=True,
    )
    output = args.output or _runtime(config) / f"evaluation_{args.split}.json"
    _write_json(output, result)
    return result


def command_controls(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    support_path, query_path = _split_paths(config, args.split)
    if args.split == "coverage_l3":
        _record_coverage_access(config, "controls:main")
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    training_supports = load_support_episodes(
        Path(config["episodes"]["runtime_dir"]) / "train_supports.jsonl"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint_v1(args.checkpoint, device)
    result = run_controls_v1(
        model,
        _load_frozen_interpreter(config, device),
        supports,
        queries,
        training_supports,
        int(config["evaluation"]["batch_size"]),
        device,
    )
    output = args.output or _runtime(config) / f"controls_{args.split}.json"
    _write_json(output, result)
    return result


def command_evaluate_v0_coverage(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    _phase_a_record(config)
    support_path, query_path = _split_paths(config, "coverage_l3")
    _record_coverage_access(config, "evaluate-model:frozen-v0")
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_inductor_checkpoint(Path(config["data"]["v0_checkpoint"]), device)
    result = evaluate_v0(
        model,
        _load_frozen_interpreter(config, device),
        supports,
        queries,
        int(config["evaluation"]["batch_size"]),
        device,
        include_loo=True,
    )
    output = args.output or Path(config["l3_coverage"]["runtime_dir"]) / "evaluation_frozen_v0.json"
    _write_json(output, result)
    return result


def command_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    base = Path("artifacts/vrm2/inductor/runtime/v1")
    main = json.loads((base / "pilot_seed_20260713/evaluation_sealed_l2.json").read_text())
    no_residual = json.loads((base / "ablation_no_residual/evaluation_sealed_l2.json").read_text())
    shuffled = json.loads((base / "ablation_shuffled_order/evaluation_sealed_l2.json").read_text())
    controls = json.loads((base / "pilot_seed_20260713/controls_sealed_l2.json").read_text())
    structural = json.loads((base / "pilot_seed_20260713/structural_validity.json").read_text())
    v0_exact = 0.5129533410072327
    observed = {
        "sealed_l2_query_exact": main["query_exact"],
        "improvement_over_v0_l2": main["query_exact"] - v0_exact,
        "residual_margin": main["query_exact"] - no_residual["query_exact"],
        "shuffled_order_drop": main["query_exact"] - shuffled["query_exact"],
        "support_permutation_identity": controls["support_permutation"]["predicted_program_identity_rate"],
        "structural_validity": structural["all_pass"],
    }
    thresholds = config["gates"]["phase_a"]
    checks = {
        "sealed_l2_query_exact": observed["sealed_l2_query_exact"] >= thresholds["sealed_l2_query_exact"],
        "improvement_over_v0_l2": observed["improvement_over_v0_l2"] >= thresholds["improvement_over_v0_l2"],
        "residual_margin": observed["residual_margin"] >= thresholds["residual_margin"],
        "shuffled_order_drop": observed["shuffled_order_drop"] >= thresholds["shuffled_order_drop"],
        "support_permutation_identity": observed["support_permutation_identity"] == thresholds["support_permutation_identity"],
        "structural_validity": bool(observed["structural_validity"]),
    }
    passed = all(checks.values())
    result = {
        "observed": observed,
        "checks": checks,
        "phase_a_pass": passed,
        "verdict": None if passed else "INDUCTOR_V1_REDESIGN_REQUIRED",
    }
    _write_json(base / "pilot_seed_20260713/phase_a_verdict.json", result)
    return result


def command_phase_b(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_v1_config(args.config)
    _phase_a_record(config)
    base = Path("artifacts/vrm2/inductor/runtime/v1")
    main = json.loads((base / "pilot_seed_20260713/evaluation_coverage_l3.json").read_text())
    v0 = json.loads((base / "l3_coverage/evaluation_frozen_v0.json").read_text())
    controls = json.loads((base / "pilot_seed_20260713/controls_coverage_l3.json").read_text())
    thresholds = config["gates"]["phase_b"]
    correct = controls["correct"]["query_exact"]
    observed = {
        "coverage_l3_query_exact": main["query_exact"],
        "improvement_over_v0_coverage_l3": main["query_exact"] - v0["query_exact"],
        "coverage_l3_oracle_exact": main["oracle_query_exact"],
        "support_swap_exact_drop": correct - controls["support_swap"]["query_exact"],
        "output_shuffle_exact_drop": correct - controls["output_shuffle"]["query_exact"],
        "null_support_exact_drop": correct - controls["null_support"]["query_exact"],
        "support_swap_output_change": controls["support_swap"]["output_change_rate"],
        "output_shuffle_output_change": controls["output_shuffle"]["output_change_rate"],
        "null_support_output_change": controls["null_support"]["output_change_rate"],
        "support_permutation_identity": controls["support_permutation"]["predicted_program_identity_rate"],
    }
    checks = {
        "coverage_l3_query_exact": observed["coverage_l3_query_exact"] >= thresholds["coverage_l3_query_exact"],
        "improvement_over_v0_coverage_l3": observed["improvement_over_v0_coverage_l3"] >= thresholds["improvement_over_v0_coverage_l3"],
        "coverage_l3_oracle_exact": observed["coverage_l3_oracle_exact"] >= thresholds["coverage_l3_oracle_exact"],
        "support_swap_exact_drop": observed["support_swap_exact_drop"] >= thresholds["control_exact_drop"],
        "output_shuffle_exact_drop": observed["output_shuffle_exact_drop"] >= thresholds["control_exact_drop"],
        "null_support_exact_drop": observed["null_support_exact_drop"] >= thresholds["control_exact_drop"],
        "support_swap_output_change": observed["support_swap_output_change"] >= thresholds["control_output_change"],
        "output_shuffle_output_change": observed["output_shuffle_output_change"] >= thresholds["control_output_change"],
        "null_support_output_change": observed["null_support_output_change"] >= thresholds["control_output_change"],
        "support_permutation_identity": observed["support_permutation_identity"] == thresholds["support_permutation_identity"],
    }
    passed = all(checks.values())
    result = {
        "observed": observed,
        "checks": checks,
        "phase_b_pass": passed,
        "verdict": "INDUCTOR_V1_PILOT_CONTINUATION_GATE_PASSED" if passed else "INDUCTOR_V1_L3_EXTRAPOLATION_FAILED",
    }
    _write_json(base / "pilot_seed_20260713/final_verdict.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SequentialResidualInductorV1")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("prepare-episodes", command_prepare_episodes), ("generate-coverage-l3", command_generate_coverage)):
        item = sub.add_parser(name); item.add_argument("--config", type=Path, required=True); item.set_defaults(function=function)
    train = sub.add_parser("train"); train.add_argument("--config", type=Path, required=True); train.add_argument("--seed", type=int, required=True); train.add_argument("--checkpoint", type=Path); train.add_argument("--report", type=Path); train.set_defaults(function=command_train)
    induce = sub.add_parser("induce-programs"); induce.add_argument("--config", type=Path, required=True); induce.add_argument("--checkpoint", type=Path, required=True); induce.add_argument("--split", required=True); induce.add_argument("--episodes", type=Path); induce.add_argument("--output", type=Path); induce.set_defaults(function=command_induce)
    execute = sub.add_parser("evaluate-induced"); execute.add_argument("--config", type=Path, required=True); execute.add_argument("--split", required=True); execute.add_argument("--queries", type=Path); execute.add_argument("--programs", type=Path); execute.add_argument("--output", type=Path); execute.set_defaults(function=command_evaluate_induced)
    evaluate = sub.add_parser("evaluate-model"); evaluate.add_argument("--config", type=Path, required=True); evaluate.add_argument("--checkpoint", type=Path, required=True); evaluate.add_argument("--split", required=True); evaluate.add_argument("--output", type=Path); evaluate.set_defaults(function=command_evaluate_model)
    controls = sub.add_parser("controls"); controls.add_argument("--config", type=Path, required=True); controls.add_argument("--checkpoint", type=Path, required=True); controls.add_argument("--split", required=True); controls.add_argument("--output", type=Path); controls.set_defaults(function=command_controls)
    v0 = sub.add_parser("evaluate-v0-coverage"); v0.add_argument("--config", type=Path, required=True); v0.add_argument("--output", type=Path); v0.set_defaults(function=command_evaluate_v0_coverage)
    phase_a = sub.add_parser("phase-a-verdict"); phase_a.add_argument("--config", type=Path, required=True); phase_a.set_defaults(function=command_phase_a)
    phase_b = sub.add_parser("phase-b-verdict"); phase_b.add_argument("--config", type=Path, required=True); phase_b.set_defaults(function=command_phase_b)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
