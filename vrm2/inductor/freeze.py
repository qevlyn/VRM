from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.hashing import file_sha256, module_state_hashes
from vrm2.oracle.train import load_checkpoint


SOURCE_MANIFEST_VERSION = "vrm2-inductor-v0-source-v1"
EPISODE_FREEZE_VERSION = "vrm2-inductor-v0-episodes-freeze-v1"
FROZEN_SOURCE_PATHS = (
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
    "configs/vrm2/inductor_v0_smoke.yaml",
    "configs/vrm2/inductor_v0_pilot.yaml",
    "docs/vrm2/INDUCTOR_V0_PROTOCOL.md",
    "tests/vrm2/test_inductor_v0_episodes.py",
    "tests/vrm2/test_inductor_v0_model.py",
    "tests/vrm2/test_inductor_v0_no_bypass.py",
    "tests/vrm2/test_inductor_v0_evaluation.py",
    "tests/vrm2/test_inductor_v0_protocol.py",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def build_source_manifest(repository_root: Path) -> dict[str, Any]:
    files = {}
    for relative in FROZEN_SOURCE_PATHS:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {relative}")
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    oracle_manifest = repository_root / "artifacts/vrm2/oracle/runtime/v1/certification/source_manifest.json"
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "files": files,
        "frozen_oracle_source_manifest": {
            "path": str(oracle_manifest),
            "sha256": file_sha256(oracle_manifest),
        },
    }


def verify_source_manifest(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_source_manifest(repository_root)
    changed = {
        path: {"expected": expected["files"].get(path), "actual": details}
        for path, details in actual["files"].items()
        if expected["files"].get(path) != details
    }
    missing = sorted(set(expected["files"]) - set(actual["files"]))
    oracle_changed = expected.get("frozen_oracle_source_manifest") != actual.get(
        "frozen_oracle_source_manifest"
    )
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "source_unchanged": not changed and not missing and not oracle_changed,
        "changed": changed,
        "missing": missing,
        "frozen_oracle_source_manifest_changed": oracle_changed,
        "actual": actual,
    }


def build_episode_manifest(episode_dir: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(episode_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and (
            path.name == "assembly_summary.json"
            or path.name.endswith("_manifest.json")
            or path.suffix == ".jsonl"
        ):
            files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    if len(files) != 16:
        raise ValueError(f"expected 16 frozen episode artifacts, found {len(files)}")
    return {"manifest_version": EPISODE_FREEZE_VERSION, "files": files}


def verify_episode_manifest(episode_dir: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_episode_manifest(episode_dir)
    changed = {
        name: {"expected": expected["files"].get(name), "actual": details}
        for name, details in actual["files"].items()
        if expected["files"].get(name) != details
    }
    missing = sorted(set(expected["files"]) - set(actual["files"]))
    return {
        "manifest_version": EPISODE_FREEZE_VERSION,
        "episode_artifacts_unchanged": not changed and not missing,
        "changed": changed,
        "missing": missing,
        "actual": actual,
    }


def checkpoint_audit(path: Path) -> dict[str, Any]:
    device = torch.device("cpu")
    model, metadata = load_checkpoint(path, device)
    total_parameters = sum(item.numel() for item in model.parameters())
    for item in model.parameters():
        item.requires_grad_(False)
    model.eval()
    hashes = module_state_hashes(model)
    return {
        "checkpoint_path": str(path),
        "archive_sha256": file_sha256(path),
        **hashes,
        "model_version": getattr(model, "model_version", metadata.get("model_version")),
        "checkpoint_step": int(metadata["step"]),
        "parameter_count": total_parameters,
        "parameters_frozen": all(not item.requires_grad for item in model.parameters()),
        "eval_mode": not model.training,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM inductor V0 freeze utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_source = sub.add_parser("freeze-source")
    freeze_source.add_argument("--repository-root", type=Path, default=Path("."))
    freeze_source.add_argument("--output", type=Path, required=True)
    verify_source = sub.add_parser("verify-source")
    verify_source.add_argument("--repository-root", type=Path, default=Path("."))
    verify_source.add_argument("--manifest", type=Path, required=True)
    verify_source.add_argument("--output", type=Path, required=True)
    freeze_episodes = sub.add_parser("freeze-episodes")
    freeze_episodes.add_argument("--episode-dir", type=Path, required=True)
    freeze_episodes.add_argument("--output", type=Path, required=True)
    verify_episodes = sub.add_parser("verify-episodes")
    verify_episodes.add_argument("--episode-dir", type=Path, required=True)
    verify_episodes.add_argument("--manifest", type=Path, required=True)
    verify_episodes.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("checkpoint-audit")
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-source":
        result = build_source_manifest(args.repository_root.resolve())
    elif args.command == "verify-source":
        result = verify_source_manifest(args.repository_root.resolve(), args.manifest)
    elif args.command == "freeze-episodes":
        result = build_episode_manifest(args.episode_dir)
    elif args.command == "verify-episodes":
        result = verify_episode_manifest(args.episode_dir, args.manifest)
    else:
        result = checkpoint_audit(args.checkpoint)
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    valid = result.get("source_unchanged", result.get("episode_artifacts_unchanged", True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
