from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs" / "vrm2" / name).read_text())


def test_pilot_freezes_seed_data_model_budget_and_gates():
    config = _load("inductor_v0_pilot.yaml")
    assert config["data"]["train_seed"] == 610000
    assert config["model_seed"] == 20260713
    assert config["typed_program_schema_version"] == "vrm2-typed-program-v1"
    assert config["data"]["sealed_seed"] == 640000
    assert config["training"]["max_steps"] == 10000
    assert config["training"]["max_parameters"] == 2_000_000
    assert config["gates"]["continuation"]["sealed_l3_query_exact"] == 0.85
    assert config["data"]["interpreter_model_state_sha256"] == "4cf0b68d675c87fcbff9ac3613801017e9cba47acebe9e86f77137037ff6aae0"
    assert config["episodes"]["eligible_program_lengths"] == {
        "train": [2], "train_eval": [2], "dev": [2], "sealed_l2": [2], "sealed_l3": [3]
    }


def test_smoke_and_pilot_differ_only_in_run_bounded_training_and_output_fields():
    pilot = _load("inductor_v0_pilot.yaml")
    smoke = _load("inductor_v0_smoke.yaml")
    assert pilot.pop("run_level") == "pilot" and smoke.pop("run_level") == "smoke"
    pilot_training = pilot.pop("training")
    smoke_training = smoke.pop("training")
    assert pilot.pop("outputs") != smoke.pop("outputs")
    assert pilot == smoke
    for field in ("learning_rate", "weight_decay", "betas", "loss_weights", "gradient_clip_norm", "mixed_precision", "max_parameters", "checkpoint_selection"):
        assert pilot_training[field] == smoke_training[field]
