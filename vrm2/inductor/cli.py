from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.microworld import load_config as load_stage0_config
from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.controls import run_inductor_controls
from vrm2.inductor.episodes import (
    assemble_episode_records,
    load_query_episodes,
    load_support_episodes,
    write_episode_corpus,
)
from vrm2.inductor.evaluate import (
    evaluate_model_on_episodes,
    predict_programs,
    write_predicted_programs,
)
from vrm2.inductor.execution import evaluate_induced_program_file
from vrm2.inductor.hashing import file_sha256
from vrm2.inductor.schema import validate_typed_program
from vrm2.inductor.train import load_inductor_checkpoint, load_inductor_config, train_one_seed
from vrm2.oracle.dataset import load_oracle_corpus
from vrm2.oracle.train import build_run_datasets, load_checkpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def command_assemble(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    stage0 = load_stage0_config(REPOSITORY_ROOT / config["stage0_config"])
    sealed_path = Path(config["data"]["sealed_corpus"])
    sealed_manifest = json.loads(
        sealed_path.with_name(f"{sealed_path.stem}.manifest.json").read_text(encoding="utf-8")
    )
    sealed = load_oracle_corpus(sealed_path, sealed_manifest["dataset_sha256"])
    datasets = build_run_datasets(config, stage0, sealed)
    output_dir = Path(config["episodes"]["runtime_dir"])
    eligible = config["episodes"]["eligible_program_lengths"]
    split_specs = {
        "train": (datasets["train"], int(config["episodes"]["assembly_seeds"]["train"]), eligible["train"]),
        "train_eval": (datasets["train_eval"], int(config["episodes"]["assembly_seeds"]["train_eval"]), eligible["train_eval"]),
        "dev": (datasets["dev"], int(config["episodes"]["assembly_seeds"]["dev"]), eligible["dev"]),
        "sealed_l2": (tuple(item for item in datasets["sealed"] if item.program_length == 2), int(config["episodes"]["assembly_seeds"]["sealed"]), eligible["sealed_l2"]),
        "sealed_l3": (tuple(item for item in datasets["sealed"] if item.program_length == 3), int(config["episodes"]["assembly_seeds"]["sealed"]), eligible["sealed_l3"]),
    }
    manifests = {}
    fingerprint_sets = {}
    for name, (examples, seed, lengths) in split_specs.items():
        supports, queries, manifest = assemble_episode_records(examples, name, seed, lengths)
        if not supports:
            raise RuntimeError("INDUCTOR_INVALID_INSUFFICIENT_EPISODE_SUPPORT")
        manifests[name] = write_episode_corpus(output_dir, name, supports, queries, manifest)
        fingerprint_sets[name] = set(manifest["consumed_example_fingerprints"])
    overlaps = {}
    names = list(split_specs)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(fingerprint_sets[left] & fingerprint_sets[right])
    if any(overlaps.values()):
        raise ValueError(f"episode splits overlap: {overlaps}")
    summary = {
        "episode_corpora": manifests,
        "cross_split_overlap_counts": overlaps,
        "sealed_source_sha256": sealed_manifest["dataset_sha256"],
    }
    _write_json(output_dir / "assembly_summary.json", summary)
    return summary


def _runtime_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["outputs"]["runtime_dir"])


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    if int(args.seed) != int(config["model_seed"]):
        raise ValueError("inductor V0 permits exactly the frozen model seed")
    runtime = _runtime_dir(config)
    checkpoint = args.checkpoint or runtime / f"inductor_seed_{args.seed}.pt"
    report = args.report or runtime / f"inductor_seed_{args.seed}.json"
    return train_one_seed(config, args.seed, checkpoint, report)


def _split_paths(config: Mapping[str, Any], split: str) -> tuple[Path, Path]:
    episode_dir = Path(config["episodes"]["runtime_dir"])
    return episode_dir / f"{split}_supports.jsonl", episode_dir / f"{split}_queries.jsonl"


def command_induce(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    support_path, _ = _split_paths(config, args.split)
    supports = load_support_episodes(args.episodes or support_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_inductor_checkpoint(args.checkpoint, device)
    lengths, programs, confidence, seconds = predict_programs(
        model, supports, int(config["evaluation"]["batch_size"]), device
    )
    output = args.output or _runtime_dir(config) / f"predicted_programs_{args.split}.jsonl"
    manifest = write_predicted_programs(
        output, supports, lengths, programs, confidence, args.checkpoint
    )
    _write_json(output.with_suffix(".manifest.json"), manifest)
    _write_json(
        output.with_suffix(".runtime.json"),
        {"induction_seconds": seconds, "induction_ms_per_episode": seconds * 1000 / len(supports)},
    )
    return manifest


def command_evaluate_induced(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    _, query_path = _split_paths(config, args.split)
    program_path = args.programs or _runtime_dir(config) / f"predicted_programs_{args.split}.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measured_result = evaluate_induced_program_file(
        args.queries or query_path,
        program_path,
        Path(config["data"]["interpreter_checkpoint"]),
        int(config["evaluation"]["batch_size"]),
        device,
    )
    output = args.output or _runtime_dir(config) / f"execution_{args.split}.json"
    result = {
        key: value for key, value in measured_result.items()
        if key not in ("execution_seconds", "execution_ms_per_grid")
    }
    _write_json(output, result)
    _write_json(
        output.with_suffix(".runtime.json"),
        {
            "execution_seconds": measured_result["execution_seconds"],
            "execution_ms_per_grid": measured_result["execution_ms_per_grid"],
        },
    )
    return result


def command_structural_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    runtime = _runtime_dir(config)
    source = json.loads(args.source_verification.read_text(encoding="utf-8"))
    episodes = json.loads(args.episode_verification.read_text(encoding="utf-8"))
    checkpoint_before = json.loads(args.checkpoint_before.read_text(encoding="utf-8"))
    checkpoint_after = json.loads(args.checkpoint_after.read_text(encoding="utf-8"))
    tests = json.loads(args.test_summary.read_text(encoding="utf-8"))
    assembly = json.loads(
        (Path(config["episodes"]["runtime_dir"]) / "assembly_summary.json").read_text(encoding="utf-8")
    )
    overlap_pass = not any(assembly["cross_split_overlap_counts"].values())
    integrity_pass = overlap_pass
    for manifest in assembly["episode_corpora"].values():
        records = manifest["support_query_fingerprints"]
        integrity_pass = integrity_pass and len(records) == manifest["episode_count"]
        integrity_pass = integrity_pass and manifest["examples_consumed"] == 4 * manifest["episode_count"]
        for record in records:
            example_ids = record["support_example_fingerprints"] + [record["query_example_fingerprint"]]
            input_ids = record["support_input_fingerprints"] + [record["query_input_fingerprint"]]
            integrity_pass = integrity_pass and len(set(example_ids)) == 4 and len(set(input_ids)) == 4
    schema_pass = True
    for path in (args.induction_first, args.induction_second):
        for line in path.read_bytes().splitlines():
            record = json.loads(line)
            try:
                validate_typed_program(
                    int(record["predicted_program_length"]),
                    record["predicted_typed_program_ids"],
                )
            except ValueError:
                schema_pass = False
    controls = [
        json.loads((runtime / f"controls_{split}.json").read_text(encoding="utf-8"))
        for split in ("sealed_l2", "sealed_l3")
    ]
    permutation_pass = all(
        report["support_permutation"]["predicted_program_identity_rate"] == 1.0
        and report["support_permutation"]["query_prediction_identity_rate"] == 1.0
        for report in controls
    )
    expected_state = str(config["data"]["interpreter_model_state_sha256"])
    checkpoint_pass = (
        checkpoint_before["model_state_sha256"] == expected_state
        and checkpoint_after["model_state_sha256"] == expected_state
        and checkpoint_before["archive_sha256"] == checkpoint_after["archive_sha256"]
        and checkpoint_before["parameter_state_sha256"] == checkpoint_after["parameter_state_sha256"]
        and checkpoint_before["buffer_state_sha256"] == checkpoint_after["buffer_state_sha256"]
    )
    checks = {
        "all_tests_pass": bool(tests["all_pass"]),
        "episode_integrity_pass": bool(integrity_pass),
        "no_bypass_tests_pass": bool(tests["no_bypass_pass"]),
        "source_unchanged": bool(source["source_unchanged"]),
        "episode_freeze_unchanged": bool(episodes["episode_artifacts_unchanged"]),
        "frozen_interpreter_unchanged": bool(checkpoint_pass),
        "support_permutation_identity": bool(permutation_pass),
        "schema_valid": bool(schema_pass),
        "induction_artifacts_identical": file_sha256(args.induction_first) == file_sha256(args.induction_second),
        "execution_artifacts_identical": file_sha256(args.execution_first) == file_sha256(args.execution_second),
    }
    result = {"checks": checks, "all_pass": all(checks.values())}
    _write_json(runtime / "structural_validity.json", result)
    return result


def command_evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    support_path, query_path = _split_paths(config, args.split)
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_inductor_checkpoint(args.checkpoint, device)
    interpreter, _ = load_checkpoint(Path(config["data"]["interpreter_checkpoint"]), device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    result = evaluate_model_on_episodes(
        model, interpreter, supports, queries,
        int(config["evaluation"]["batch_size"]), device, include_loo=True
    )
    output = args.output or _runtime_dir(config) / f"evaluation_{args.split}.json"
    _write_json(output, result)
    return result


def command_controls(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    support_path, query_path = _split_paths(config, args.split)
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    train_support_path, _ = _split_paths(config, "train")
    training_supports = load_support_episodes(train_support_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_inductor_checkpoint(args.checkpoint, device)
    interpreter, _ = load_checkpoint(Path(config["data"]["interpreter_checkpoint"]), device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    result = run_inductor_controls(
        model, interpreter, supports, queries, training_supports,
        int(config["evaluation"]["batch_size"]), device
    )
    output = args.output or _runtime_dir(config) / f"controls_{args.split}.json"
    _write_json(output, result)
    return result


def command_verdict(args: argparse.Namespace) -> dict[str, Any]:
    config = load_inductor_config(args.config)
    runtime = _runtime_dir(config)
    l2 = json.loads((runtime / "execution_sealed_l2.json").read_text())
    l3 = json.loads((runtime / "execution_sealed_l3.json").read_text())
    controls_l2 = json.loads((runtime / "controls_sealed_l2.json").read_text())
    controls_l3 = json.loads((runtime / "controls_sealed_l3.json").read_text())
    n2, n3 = int(l2["num_episodes"]), int(l3["num_episodes"])
    aggregate = (l2["query_exact"] * n2 + l3["query_exact"] * n3) / (n2 + n3)
    def combined(condition: str, metric: str) -> float:
        return (
            controls_l2[condition][metric] * controls_l2[condition]["eligible"]
            + controls_l3[condition][metric] * controls_l3[condition]["eligible"]
        ) / (controls_l2[condition]["eligible"] + controls_l3[condition]["eligible"])
    correct = combined("correct", "query_exact")
    prior = combined("most_frequent_prior", "query_exact")
    observed = {
        "sealed_l2_query_exact": l2["query_exact"],
        "sealed_l3_query_exact": l3["query_exact"],
        "sealed_aggregate_query_exact": aggregate,
        "improvement_over_prior": aggregate - prior,
        "support_swap_exact_drop": correct - combined("support_swap", "query_exact"),
        "output_shuffle_exact_drop": correct - combined("output_shuffle", "query_exact"),
        "null_support_exact_drop": correct - combined("null_support", "query_exact"),
        "support_swap_output_change": combined("support_swap", "output_change_rate"),
        "output_shuffle_output_change": combined("output_shuffle", "output_change_rate"),
        "null_support_output_change": combined("null_support", "output_change_rate"),
    }
    thresholds = config["gates"]["continuation"]
    gate_evaluation = [
        {"gate": name, "observed": value, "operator": ">=", "threshold": float(thresholds[name]), "pass": value >= float(thresholds[name])}
        for name, value in observed.items()
    ]
    floors = config["gates"]["hard_stop"]
    hard_stop = (
        l2["query_exact"] < float(floors["sealed_l2_query_exact"])
        or l3["query_exact"] < float(floors["sealed_l3_query_exact"])
        or aggregate < float(floors["sealed_aggregate_query_exact"])
        or aggregate - prior < float(floors["improvement_over_prior"])
    )
    structural = json.loads((runtime / "structural_validity.json").read_text())
    if not structural["all_pass"]:
        verdict = "INDUCTOR_V0_INVALID"
    elif hard_stop or not all(item["pass"] for item in gate_evaluation):
        verdict = "INDUCTOR_V0_REDENIGN_REQUIRED"
    else:
        verdict = "INDUCTOR_V0_PILOT_CONTINUATION_GATE_PASSED"
    result = {"observed": observed, "gate_evaluation": gate_evaluation, "hard_stop": hard_stop, "verdict": verdict}
    _write_json(runtime / "pilot_verdict.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM supervised typed-program inductor V0")
    sub = parser.add_subparsers(dest="command", required=True)
    assemble = sub.add_parser("assemble-episodes"); assemble.add_argument("--config", type=Path, required=True)
    train = sub.add_parser("train"); train.add_argument("--config", type=Path, required=True); train.add_argument("--seed", type=int, required=True); train.add_argument("--checkpoint", type=Path); train.add_argument("--report", type=Path)
    induce = sub.add_parser("induce-programs"); induce.add_argument("--config", type=Path, required=True); induce.add_argument("--checkpoint", type=Path, required=True); induce.add_argument("--split", required=True); induce.add_argument("--episodes", type=Path); induce.add_argument("--output", type=Path)
    execute = sub.add_parser("evaluate-induced"); execute.add_argument("--config", type=Path, required=True); execute.add_argument("--split", required=True); execute.add_argument("--queries", type=Path); execute.add_argument("--programs", type=Path); execute.add_argument("--output", type=Path)
    evaluate = sub.add_parser("evaluate-model"); evaluate.add_argument("--config", type=Path, required=True); evaluate.add_argument("--checkpoint", type=Path, required=True); evaluate.add_argument("--split", required=True); evaluate.add_argument("--output", type=Path)
    controls = sub.add_parser("controls"); controls.add_argument("--config", type=Path, required=True); controls.add_argument("--checkpoint", type=Path, required=True); controls.add_argument("--split", required=True); controls.add_argument("--output", type=Path)
    audit = sub.add_parser("structural-audit"); audit.add_argument("--config", type=Path, required=True); audit.add_argument("--source-verification", type=Path, required=True); audit.add_argument("--episode-verification", type=Path, required=True); audit.add_argument("--checkpoint-before", type=Path, required=True); audit.add_argument("--checkpoint-after", type=Path, required=True); audit.add_argument("--test-summary", type=Path, required=True); audit.add_argument("--induction-first", type=Path, required=True); audit.add_argument("--induction-second", type=Path, required=True); audit.add_argument("--execution-first", type=Path, required=True); audit.add_argument("--execution-second", type=Path, required=True)
    verdict = sub.add_parser("pilot-verdict"); verdict.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"assemble-episodes": command_assemble, "train": command_train, "induce-programs": command_induce, "evaluate-induced": command_evaluate_induced, "evaluate-model": command_evaluate_model, "controls": command_controls, "structural-audit": command_structural_audit, "pilot-verdict": command_verdict}
    result = commands[args.command](args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
