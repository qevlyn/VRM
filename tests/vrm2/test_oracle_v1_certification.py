from __future__ import annotations

import inspect
import json
from pathlib import Path

from vrm2.oracle.certification import audit_configs, build_source_manifest
from vrm2.oracle.model import parameter_count
from vrm2.oracle.model_v1 import OracleInterpreterV1
import vrm2.oracle.cli as oracle_cli
from vrm2.oracle.train import build_oracle_model, train_one_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads(
        (REPOSITORY_ROOT / "configs" / "vrm2" / name).read_text(encoding="utf-8")
    )


def test_v1_certification_config_preserves_all_scientific_pilot_fields():
    pilot = _load("oracle_interpreter_v1_pilot.yaml")
    certification = _load("oracle_interpreter_v1_cert.yaml")
    allowed_differences = {
        "run_level": ("pilot", "certification"),
        "outputs.runtime_dir": (
            "artifacts/vrm2/oracle/runtime/v1/pilot_seed_20260713",
            "artifacts/vrm2/oracle/runtime/v1/certification",
        ),
    }
    actual_differences = {}
    if pilot["run_level"] != certification["run_level"]:
        actual_differences["run_level"] = (pilot.pop("run_level"), certification.pop("run_level"))
    pilot_output = pilot["outputs"].pop("runtime_dir")
    certification_output = certification["outputs"].pop("runtime_dir")
    if pilot_output != certification_output:
        actual_differences["outputs.runtime_dir"] = (pilot_output, certification_output)
    assert actual_differences == allowed_differences
    assert pilot == certification


def test_v1_certification_seed_list_and_gates_are_exactly_frozen():
    config = _load("oracle_interpreter_v1_cert.yaml")
    assert config["certification"] == {
        "seeds": [20260713, 20260714, 20260715],
        "pilot_proceed_thresholds": {
            "train_signature": 0.95,
            "sealed_len2": 0.9,
            "sealed_len3": 0.85,
        },
        "exact_threshold": 0.95,
        "minimum_control_exact_drop": 0.2,
        "minimum_control_output_change_rate": 0.5,
    }


def test_v1_certification_runner_selects_the_frozen_v1_model():
    model = build_oracle_model(_load("oracle_interpreter_v1_cert.yaml"))
    assert isinstance(model, OracleInterpreterV1)
    assert parameter_count(model) == 741_930


def test_checkpoint_selection_region_does_not_use_sealed_metrics():
    source = inspect.getsource(train_one_seed)
    selection = source[source.index("score =") : source.index("minimum_steps =")]
    assert 'score = (float(dev_metrics["exact"]), float(train_metrics["exact"]))' in selection
    assert "sealed" not in selection


def test_certification_config_audit_allows_only_orchestration_fields():
    config_dir = REPOSITORY_ROOT / "configs" / "vrm2"
    audit = audit_configs(
        config_dir / "oracle_interpreter_v1_pilot.yaml",
        config_dir / "oracle_interpreter_v1_cert.yaml",
    )
    assert audit["scientific_fields_equal"] is True
    assert set(audit["permitted_differences"]) == {"run_level", "outputs.runtime_dir"}
    assert audit["unexpected_differences"] == {}


def test_source_manifest_covers_required_frozen_files():
    manifest = build_source_manifest(REPOSITORY_ROOT)
    assert manifest["manifest_version"] == "vrm2-oracle-v1-certification-source-v1"
    assert "vrm2/oracle/model_v1.py" in manifest["files"]
    assert "configs/vrm2/oracle_interpreter_v1_cert.yaml" in manifest["files"]
    assert "docs/vrm2/ORACLE_INTERPRETER_PROTOCOL.md" in manifest["files"]


def test_certification_control_drop_uses_same_sample_correct_baseline():
    source = inspect.getsource(oracle_cli.command_certify)
    assert 'correct = float(controls["correct"]["exact"])' in source
    assert 'correct = float(report["sealed_all"]["exact"])' not in source
