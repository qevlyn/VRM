from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .r5a3a_arc_audit import ARTIFACT_ROOT, run_audit_twice


def _default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R5A-3A ARC-native structural candidate machine audit.")
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT))
    parser.add_argument("--run-pytest", action="store_true")
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root)
    result = run_audit_twice(artifact_root=artifact_root)
    if args.run_pytest:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd="/mnt/d/VRM",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "pytest_result.txt").write_text(completed.stdout, encoding="utf-8")
        result["pytest_returncode"] = completed.returncode
    print(json.dumps(result, indent=2, sort_keys=True, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
