from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((REPOSITORY_ROOT / "configs" / "vrm2" / name).read_text(encoding="utf-8"))


def test_v1_pilot_changes_only_version_and_runtime_namespace_from_v0():
    v0 = _load("oracle_interpreter_pilot.yaml")
    v1 = _load("oracle_interpreter_v1_pilot.yaml")
    assert v1.pop("model_version") == "v1"
    assert v1.pop("protocol_version") == "vrm2-oracle-interpreter-v1"
    assert v0.pop("protocol_version") == "vrm2-oracle-interpreter-v0"
    assert v1["outputs"].pop("runtime_dir") == "artifacts/vrm2/oracle/runtime/v1/pilot_seed_20260713"
    assert v0["outputs"].pop("runtime_dir") == "artifacts/vrm2/oracle/runtime/pilot"
    assert v1 == v0
