from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vrm2.data.serialization import canonical_json_bytes


SOURCE_MANIFEST_VERSION = "vrm2-oracle-v1-certification-source-v1"
FROZEN_SOURCE_PATHS = (
    "vrm2/oracle/model_v1.py",
    "vrm2/oracle/model.py",
    "vrm2/oracle/train.py",
    "vrm2/oracle/evaluate.py",
    "vrm2/oracle/controls.py",
    "vrm2/oracle/diagnostics.py",
    "vrm2/oracle/cli.py",
    "vrm2/oracle/dataset.py",
    "vrm2/oracle/types.py",
    "vrm2/oracle/certification.py",
    "vrm2/data/program_schema.py",
    "configs/vrm2/oracle_interpreter_v1_pilot.yaml",
    "configs/vrm2/oracle_interpreter_v1_cert.yaml",
    "configs/vrm2/oracle_interpreter_cert.yaml",
    "docs/vrm2/ORACLE_INTERPRETER_V1_PROTOCOL.md",
    "docs/vrm2/ORACLE_INTERPRETER_PROTOCOL.md",
    "tests/vrm2/test_oracle_v1_certification.py",
)
ALLOWED_CONFIG_DIFFERENCES = ("run_level", "outputs.runtime_dir")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_manifest(repository_root: Path) -> dict[str, Any]:
    files = {}
    for relative_path in FROZEN_SOURCE_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {relative_path}")
        files[relative_path] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return {"manifest_version": SOURCE_MANIFEST_VERSION, "files": files}


def write_source_manifest(repository_root: Path, output: Path) -> dict[str, Any]:
    manifest = build_source_manifest(repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def verify_source_manifest(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_source_manifest(repository_root)
    changed = {
        path: {"expected": expected["files"].get(path), "actual": details}
        for path, details in actual["files"].items()
        if expected["files"].get(path) != details
    }
    missing = sorted(set(expected["files"]) - set(actual["files"]))
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "source_unchanged": not changed and not missing,
        "changed": changed,
        "missing": missing,
        "actual": actual,
    }


def _differences(left: Any, right: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences = {}
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences[path] = {"pilot": left.get(key), "certification": right.get(key)}
            else:
                differences.update(_differences(left[key], right[key], path))
        return differences
    if left != right:
        return {prefix: {"pilot": left, "certification": right}}
    return {}


def audit_configs(pilot_path: Path, certification_path: Path) -> dict[str, Any]:
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    differences = _differences(pilot, certification)
    permitted = {
        path: values for path, values in differences.items() if path in ALLOWED_CONFIG_DIFFERENCES
    }
    unexpected = {
        path: values for path, values in differences.items() if path not in ALLOWED_CONFIG_DIFFERENCES
    }
    return {
        "pilot_config_sha256": _sha256(pilot_path),
        "certification_config_sha256": _sha256(certification_path),
        "allowed_difference_paths": list(ALLOWED_CONFIG_DIFFERENCES),
        "permitted_differences": permitted,
        "unexpected_differences": unexpected,
        "scientific_fields_equal": not unexpected,
    }


def _write_json(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload) + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM V1 certification reproducibility utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-source")
    freeze.add_argument("--repository-root", type=Path, default=Path("."))
    freeze.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-source")
    verify.add_argument("--repository-root", type=Path, default=Path("."))
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit-config")
    audit.add_argument("--pilot", type=Path, required=True)
    audit.add_argument("--certification", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-source":
        result = write_source_manifest(args.repository_root.resolve(), args.output)
    elif args.command == "verify-source":
        result = verify_source_manifest(args.repository_root.resolve(), args.manifest)
        _write_json(args.output, result)
    else:
        result = audit_configs(args.pilot, args.certification)
        _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("source_unchanged", result.get("scientific_fields_equal", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
