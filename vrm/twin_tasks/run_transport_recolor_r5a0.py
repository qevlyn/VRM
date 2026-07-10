from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .transport_recolor_r5a0_oracles import operation_function_audit

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v1"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/transport_recolor_r5a0_v1"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_new_output_path(path: Path, data_out: Path, checkpoint_out: Path) -> bool:
    resolved = path.resolve()
    return (
        resolved == data_out.resolve()
        or data_out.resolve() in resolved.parents
        or resolved == checkpoint_out.resolve()
        or checkpoint_out.resolve() in resolved.parents
    )


def protected_hashes(data_out: Path, checkpoint_out: Path) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    roots = (
        PROJECT_ROOT / "vrm",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "archives",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if is_new_output_path(path, data_out, checkpoint_out):
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            if path.name in {
                "transport_recolor_r5a0_data.py",
                "transport_recolor_r5a0_oracles.py",
                "transport_recolor_r5a0_model.py",
                "run_transport_recolor_r5a0.py",
                "test_transport_recolor_r5a0.py",
                "TRANSPORT_RECOLOR_R5A0.md",
            }:
                continue
            hashes[str(path)] = sha256_file(path)
    return hashes


def hash_status(before: dict[str, str | None], after: dict[str, str | None]) -> dict[str, Any]:
    changed = sorted(key for key, value in before.items() if after.get(key) != value)
    added = sorted(key for key in after if key not in before)
    removed = sorted(key for key in before if key not in after)
    return {
        "unchanged": not changed and not added and not removed,
        "changed_count": len(changed),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_examples": changed[:20],
        "added_examples": added[:20],
        "removed_examples": removed[:20],
    }


def cuda_environment_report() -> dict[str, Any]:
    available = torch.cuda.is_available()
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_is_available": available,
        "device": "cuda" if available else None,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if available else None,
        "torch_threads": torch.get_num_threads(),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not available:
        print("CUDA_REQUIRED_NOT_AVAILABLE", flush=True)
        raise SystemExit("CUDA_REQUIRED_NOT_AVAILABLE")
    return report


def run_pytest(data_out: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_transport_recolor_r5a0.py"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    save_text(data_out / "pytest_result.txt", result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:]}


def write_summary(data_out: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# Transport-Recolor R5A-0", ""]
    lines.append(f"Verdict: `{aggregate['verdict']}`")
    lines.append("")
    lines.append("The requested primitive split contains exact functional collisions over the required `(x,y,c)->(x_final,y_final,c_final)` domain.")
    for collision in aggregate["operation_function_audit"]["collisions"]:
        lines.append(f"- `{collision['left']}` equals `{collision['right']}`")
    lines.append("")
    lines.append("Per the R5A-0 protocol, dataset generation, model construction, training, checkpointing, and sealed evaluation were not started.")
    save_text(data_out / "summary.md", "\n".join(lines) + "\n")


def stop_with_collision(data_out: Path, checkpoint_out: Path, before: dict[str, str | None], environment: dict[str, Any], audit: dict[str, Any], pytest_result: dict[str, Any] | None) -> int:
    after = protected_hashes(data_out, checkpoint_out)
    aggregate = {
        "verdict": "R5A0_FUNCTION_COLLISION_DETECTED",
        "environment": environment,
        "operation_function_audit": audit,
        "pytest_result": pytest_result,
        "dataset_generation_started": False,
        "model_construction_started": False,
        "training_started": False,
        "checkpoint_write_started": False,
        "source_protection_hash_status": hash_status(before, after),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(torch.device("cuda")),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(torch.device("cuda")),
    }
    save_json(data_out / "source_protection_hashes_after.json", after)
    save_json(data_out / "dry_run_report.json", aggregate)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    write_summary(data_out, aggregate)
    print("R5A0_FUNCTION_COLLISION_DETECTED", flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    args = parser.parse_args(argv)

    data_out = Path(args.data_out)
    checkpoint_out = Path(args.checkpoint_out)
    data_out.mkdir(parents=True, exist_ok=True)
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    save_json(data_out / "run_config.json", vars(args) | {"seeds": seeds})
    before = protected_hashes(data_out, checkpoint_out)
    save_json(data_out / "source_protection_hashes_before.json", before)
    environment = cuda_environment_report()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    save_json(data_out / "environment_report.json", environment)

    audit = operation_function_audit()
    save_json(data_out / "operation_function_audit.json", audit)
    pytest_result = run_pytest(data_out)

    if not audit["proof"]["passed"]:
        return stop_with_collision(data_out, checkpoint_out, before, environment, audit, pytest_result)

    # The current declared grammar cannot reach this branch because the exact
    # function audit detects collisions before dataset generation.
    after = protected_hashes(data_out, checkpoint_out)
    save_json(data_out / "source_protection_hashes_after.json", after)
    aggregate = {
        "verdict": "INCONCLUSIVE",
        "reason": "operation function audit unexpectedly passed but full R5A-0 training path was not reached in this guard runner",
        "operation_function_audit": audit,
        "pytest_result": pytest_result,
        "source_protection_hash_status": hash_status(before, after),
    }
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    print(json.dumps({"verdict": "INCONCLUSIVE"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
