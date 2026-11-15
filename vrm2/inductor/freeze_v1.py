from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.freeze import checkpoint_audit as oracle_checkpoint_audit
from vrm2.inductor.hashing import file_sha256, module_state_hashes
from vrm2.inductor.model_v1 import (
    SequentialResidualInductorV1,
    component_parameter_counts,
)
from vrm2.inductor.train import load_inductor_checkpoint
from vrm2.inductor.train_v1 import load_checkpoint_v1


SOURCE_MANIFEST_VERSION = "vrm2-inductor-v1-source-v1"
TRAINING_EPISODE_FREEZE_VERSION = "vrm2-inductor-v1-training-episodes-v1"
COVERAGE_FREEZE_VERSION = "vrm2-inductor-v1-coverage-l3-freeze-v1"

EXPECTED_ORACLE_ARCHIVE_SHA256 = (
    "52c05ee52badf8bf3244d15d146b81a0f7a372a049473756771d05a90ee02e48"
)
EXPECTED_ORACLE_MODEL_STATE_SHA256 = (
    "4cf0b68d675c87fcbff9ac3613801017e9cba47acebe9e86f77137037ff6aae0"
)
EXPECTED_V0_ARCHIVE_SHA256 = (
    "f0fd045bb2b08ef8a716b6b76fb47de8e20b4163881d94d5f14f4d81a1dccd09"
)

# The V1 generator depends on all data modules and the evaluator depends on the
# frozen oracle implementation.  Hashing the entire scientific Python surface
# avoids a selectively incomplete source freeze.
FROZEN_SOURCE_PATHS = (
    "vrm2/data/__init__.py",
    "vrm2/data/composition_split.py",
    "vrm2/data/microworld.py",
    "vrm2/data/operations.py",
    "vrm2/data/program_schema.py",
    "vrm2/data/serialization.py",
    "vrm2/data/types.py",
    "vrm2/inductor/__init__.py",
    "vrm2/inductor/types.py",
    "vrm2/inductor/episodes.py",
    "vrm2/inductor/model.py",
    "vrm2/inductor/schema.py",
    "vrm2/inductor/evaluate.py",
    "vrm2/inductor/execution.py",
    "vrm2/inductor/controls.py",
    "vrm2/inductor/hashing.py",
    "vrm2/inductor/train.py",
    "vrm2/inductor/freeze.py",
    "vrm2/inductor/cli.py",
    "vrm2/inductor/episodes_v1.py",
    "vrm2/inductor/sequential_decoder.py",
    "vrm2/inductor/model_v1.py",
    "vrm2/inductor/schema_v1.py",
    "vrm2/inductor/evaluate_v1.py",
    "vrm2/inductor/execution_v1.py",
    "vrm2/inductor/controls_v1.py",
    "vrm2/inductor/l3_coverage.py",
    "vrm2/inductor/train_v1.py",
    "vrm2/inductor/freeze_v1.py",
    "vrm2/inductor/cli_v1.py",
    "configs/vrm2/inductor_v1_smoke.yaml",
    "configs/vrm2/inductor_v1_pilot.yaml",
    "configs/vrm2/inductor_v1_ablation_no_residual.yaml",
    "configs/vrm2/inductor_v1_ablation_shuffled_order.yaml",
    "docs/vrm2/INDUCTOR_V1_PROTOCOL.md",
    "tests/vrm2/test_inductor_v1_decoder.py",
    "tests/vrm2/test_inductor_v1_training_split.py",
    "tests/vrm2/test_inductor_v1_l3_coverage.py",
    "tests/vrm2/test_inductor_v1_protocol.py",
)

TRAINING_EPISODE_FILES = (
    "assembly_summary.json",
    "training_data_audit.json",
    "train_length1_manifest.json",
    "train_length1_supports.jsonl",
    "train_length1_queries.jsonl",
    "train_manifest.json",
    "train_supports.jsonl",
    "train_queries.jsonl",
    "train_eval_length1_manifest.json",
    "train_eval_length1_supports.jsonl",
    "train_eval_length1_queries.jsonl",
    "train_eval_manifest.json",
    "train_eval_supports.jsonl",
    "train_eval_queries.jsonl",
    "dev_manifest.json",
    "dev_supports.jsonl",
    "dev_queries.jsonl",
    "sealed_l2_manifest.json",
    "sealed_l2_supports.jsonl",
    "sealed_l2_queries.jsonl",
    "original_l3_manifest.json",
    "original_l3_supports.jsonl",
    "original_l3_queries.jsonl",
)

COVERAGE_FILES = (
    "episodes.jsonl",
    "coverage_l3_supports.jsonl",
    "coverage_l3_queries.jsonl",
    "coverage_l3_manifest.json",
    "manifest.json",
    "eligibility_audit.json",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _hash_files(root: Path, names: Sequence[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {path}")
        files[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return files


def build_source_manifest(repository_root: Path) -> dict[str, Any]:
    files = _hash_files(repository_root, FROZEN_SOURCE_PATHS)
    oracle_manifest = (
        repository_root
        / "artifacts/vrm2/oracle/runtime/v1/certification/source_manifest.json"
    )
    v0_manifest = (
        repository_root
        / "artifacts/vrm2/inductor/runtime/v0/pilot_seed_20260713/source_manifest.json"
    )
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "files": files,
        "frozen_oracle_source_manifest": {
            "path": str(oracle_manifest),
            "sha256": file_sha256(oracle_manifest),
        },
        "frozen_v0_source_manifest": {
            "path": str(v0_manifest),
            "sha256": file_sha256(v0_manifest),
        },
    }


def _verify_manifest(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = expected["files"]
    actual_files = actual["files"]
    changed = {
        name: {"expected": expected_files.get(name), "actual": details}
        for name, details in actual_files.items()
        if expected_files.get(name) != details
    }
    missing = sorted(set(expected_files) - set(actual_files))
    metadata_changed = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in set(expected) | set(actual)
        if key != "files" and expected.get(key) != actual.get(key)
    }
    return {
        "unchanged": not changed and not missing and not metadata_changed,
        "changed": changed,
        "missing": missing,
        "metadata_changed": metadata_changed,
        "actual": actual,
    }


def verify_source_manifest(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = _verify_manifest(expected, build_source_manifest(repository_root))
    result["manifest_version"] = SOURCE_MANIFEST_VERSION
    result["source_unchanged"] = result.pop("unchanged")
    return result


def build_artifact_manifest(
    root: Path, names: Sequence[str], version: str
) -> dict[str, Any]:
    return {"manifest_version": version, "files": _hash_files(root, names)}


def verify_artifact_manifest(
    root: Path, names: Sequence[str], version: str, manifest_path: Path
) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = _verify_manifest(expected, build_artifact_manifest(root, names, version))
    result["manifest_version"] = version
    result["artifacts_unchanged"] = result.pop("unchanged")
    return result


def v0_checkpoint_audit(path: Path) -> dict[str, Any]:
    model, metadata = load_inductor_checkpoint(path, torch.device("cpu"))
    model.eval()
    hashes = module_state_hashes(model)
    return {
        "checkpoint_path": str(path),
        "archive_sha256": file_sha256(path),
        **hashes,
        "model_version": getattr(model, "model_version", metadata.get("model_version")),
        "checkpoint_step": int(metadata["step"]),
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "eval_mode": not model.training,
    }


def v1_checkpoint_audit(path: Path) -> dict[str, Any]:
    model, metadata = load_checkpoint_v1(path, torch.device("cpu"))
    model.eval()
    return {
        "checkpoint_path": str(path),
        "archive_sha256": file_sha256(path),
        **module_state_hashes(model),
        "model_version": model.model_version,
        "checkpoint_step": int(metadata["step"]),
        "component_parameter_counts": component_parameter_counts(model),
        "eval_mode": not model.training,
    }


def build_structural_validity(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.source_verification.read_text(encoding="utf-8"))
    episodes = json.loads(args.episode_verification.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage_preflight_verification.read_text(encoding="utf-8"))
    tests = json.loads(args.test_summary.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    main = json.loads(args.main_report.read_text(encoding="utf-8"))
    no_residual = json.loads(args.no_residual_report.read_text(encoding="utf-8"))
    shuffled = json.loads(args.shuffled_report.read_text(encoding="utf-8"))
    oracle_before = json.loads(args.oracle_before.read_text(encoding="utf-8"))
    oracle_after = json.loads(args.oracle_after.read_text(encoding="utf-8"))
    v0 = json.loads(args.v0_audit.read_text(encoding="utf-8"))
    controls = json.loads(args.controls.read_text(encoding="utf-8"))
    access = json.loads(args.coverage_access_audit.read_text(encoding="utf-8"))
    first_programs = file_sha256(args.first_programs)
    second_programs = file_sha256(args.second_programs)
    first_execution = json.loads(args.first_execution.read_text(encoding="utf-8"))
    second_execution = json.loads(args.second_execution.read_text(encoding="utf-8"))

    expected_oracle_fields = {
        "archive_sha256": EXPECTED_ORACLE_ARCHIVE_SHA256,
        "model_state_sha256": EXPECTED_ORACLE_MODEL_STATE_SHA256,
        "parameter_count": 741930,
        "checkpoint_step": 7000,
        "parameters_frozen": True,
        "eval_mode": True,
    }
    checks = {
        "source_unchanged": source["source_unchanged"],
        "training_episode_artifacts_unchanged": episodes["artifacts_unchanged"],
        "coverage_corpus_preflight_frozen": coverage["artifacts_unchanged"],
        "tests_passed": tests["passed"] == 87 and tests["failed"] == 0,
        "smoke_completed": smoke["steps_executed"] == 100 and smoke["best_step"] > 0,
        "main_parameter_budget": main["component_parameter_counts"]["total"] <= 1350000,
        "single_seed_main": main["seed"] == 20260713,
        "single_seed_no_residual": no_residual["seed"] == 20260713,
        "single_seed_shuffled": shuffled["seed"] == 20260713,
        "main_optimizer_maximum_target_length_two": main["target_length_audit"][
            "maximum_active_target_length_observed_by_optimizer"
        ] == 2,
        "no_residual_optimizer_maximum_target_length_two": no_residual[
            "target_length_audit"
        ]["maximum_active_target_length_observed_by_optimizer"] == 2,
        "shuffled_optimizer_maximum_target_length_two": shuffled[
            "target_length_audit"
        ]["maximum_active_target_length_observed_by_optimizer"] == 2,
        "no_length_three_optimizer_targets": all(
            report["target_length_audit"]["length_3_episodes_reaching_optimizer"] == 0
            for report in (main, no_residual, shuffled)
        ),
        "training_and_selection_never_opened_coverage": all(
            "coverage" not in path.lower()
            for report in (main, no_residual, shuffled)
            for path in report["opened_training_and_selection_paths"]
        ),
        "training_and_selection_never_opened_sealed": all(
            "sealed" not in path.lower()
            for report in (main, no_residual, shuffled)
            for path in report["opened_training_and_selection_paths"]
        ),
        "coverage_access_audit_sealed": (
            access["phase_b_evaluation_count"] == 0
            and not access["opened_during_training"]
            and not access["opened_during_checkpoint_selection"]
        ),
        "interpreter_unchanged_in_all_training_runs": all(
            report["interpreter_unchanged"]
            and report["interpreter_parameters_frozen"]
            and report["interpreter_eval_mode"]
            for report in (smoke, main, no_residual, shuffled)
        ),
        "oracle_before_expected": all(
            oracle_before.get(key) == value for key, value in expected_oracle_fields.items()
        ),
        "oracle_after_expected": all(
            oracle_after.get(key) == value for key, value in expected_oracle_fields.items()
        ),
        "oracle_hashes_unchanged": all(
            oracle_before.get(key) == oracle_after.get(key)
            for key in (
                "archive_sha256",
                "model_state_sha256",
                "parameter_state_sha256",
                "buffer_state_sha256",
            )
        ),
        "v0_checkpoint_expected": v0["archive_sha256"] == EXPECTED_V0_ARCHIVE_SHA256,
        "repeated_hard_program_artifacts_identical": first_programs == second_programs,
        "repeated_fresh_process_execution_identical": first_execution == second_execution,
        "support_permutation_program_identity": controls["support_permutation"][
            "predicted_program_identity_rate"
        ] == 1.0,
        "support_permutation_query_identity": controls["support_permutation"][
            "query_prediction_identity_rate"
        ] == 1.0,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "repeated_program_sha256": first_programs,
        "oracle_before": oracle_before,
        "oracle_after": oracle_after,
        "v0_checkpoint": v0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SequentialResidualInductorV1 freezes")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_source = sub.add_parser("freeze-source")
    freeze_source.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_source.add_argument("--output", type=Path, required=True)
    verify_source = sub.add_parser("verify-source")
    verify_source.add_argument("--repository-root", type=Path, default=Path("."))
    verify_source.add_argument("--manifest", type=Path, required=True)
    verify_source.add_argument("--output", type=Path, required=True)
    for prefix in ("training-episodes", "coverage"):
        freeze = sub.add_parser(f"freeze-{prefix}")
        freeze.add_argument("--root", type=Path, required=True)
        freeze.add_argument("--output", type=Path, required=True)
        verify = sub.add_parser(f"verify-{prefix}")
        verify.add_argument("--root", type=Path, required=True)
        verify.add_argument("--manifest", type=Path, required=True)
        verify.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("checkpoint-audit")
    audit.add_argument("--kind", choices=("oracle", "v0", "v1"), required=True)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    structural = sub.add_parser("structural-validity")
    for name in (
        "source-verification",
        "episode-verification",
        "coverage-preflight-verification",
        "test-summary",
        "smoke-report",
        "main-report",
        "no-residual-report",
        "shuffled-report",
        "oracle-before",
        "oracle-after",
        "v0-audit",
        "controls",
        "coverage-access-audit",
        "first-programs",
        "second-programs",
        "first-execution",
        "second-execution",
        "output",
    ):
        structural.add_argument(f"--{name}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-source":
        result = build_source_manifest(args.repository_root.resolve())
    elif args.command == "verify-source":
        result = verify_source_manifest(args.repository_root.resolve(), args.manifest)
    elif args.command in ("freeze-training-episodes", "verify-training-episodes"):
        if args.command.startswith("freeze"):
            result = build_artifact_manifest(
                args.root, TRAINING_EPISODE_FILES, TRAINING_EPISODE_FREEZE_VERSION
            )
        else:
            result = verify_artifact_manifest(
                args.root,
                TRAINING_EPISODE_FILES,
                TRAINING_EPISODE_FREEZE_VERSION,
                args.manifest,
            )
    elif args.command in ("freeze-coverage", "verify-coverage"):
        if args.command.startswith("freeze"):
            result = build_artifact_manifest(
                args.root, COVERAGE_FILES, COVERAGE_FREEZE_VERSION
            )
        else:
            result = verify_artifact_manifest(
                args.root, COVERAGE_FILES, COVERAGE_FREEZE_VERSION, args.manifest
            )
    elif args.command == "checkpoint-audit":
        function = {
            "oracle": oracle_checkpoint_audit,
            "v0": v0_checkpoint_audit,
            "v1": v1_checkpoint_audit,
        }[args.kind]
        result = function(args.checkpoint)
    else:
        result = build_structural_validity(args)
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    valid = result.get(
        "source_unchanged", result.get("artifacts_unchanged", result.get("all_pass", True))
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
