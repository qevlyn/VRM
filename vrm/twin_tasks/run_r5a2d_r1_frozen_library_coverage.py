from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .r5a2d_r1_coverage_audit import ARTIFACT_ROOT, run_complete_audit_twice


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R5A-2D-R1 frozen two-specialist ARC-native coverage audit.")
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT), help="Artifact directory for R5A-2D-R1 outputs.")
    parser.add_argument("--run-pytest", action="store_true", help="Run pytest -q after the audit and store pytest_result.txt.")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    result = run_complete_audit_twice(artifact_root=artifact_root)

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

    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("final_verdict") == "R5A2D_R1_FROZEN_LIBRARY_COVERAGE_MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
