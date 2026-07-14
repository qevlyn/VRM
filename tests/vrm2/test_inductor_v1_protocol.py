from __future__ import annotations

import json
from pathlib import Path

import torch

from vrm2.inductor.model_v1 import InductorV1ModelConfig, SequentialResidualInductorV1
from vrm2.inductor.train_v1 import load_checkpoint_v1, save_checkpoint_v1


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return json.loads((ROOT / "configs/vrm2" / name).read_text())


def test_configs_freeze_seed_budget_training_and_ablation_interventions():
    main = _load("inductor_v1_pilot.yaml")
    no_residual = _load("inductor_v1_ablation_no_residual.yaml")
    shuffled = _load("inductor_v1_ablation_shuffled_order.yaml")
    assert main["model_seed"] == 20260713
    assert main["training"]["max_parameters"] == 1_350_000
    assert main["training"]["max_steps"] == 10_000
    assert main["l3_coverage"] == {
        "runtime_dir": "artifacts/vrm2/inductor/runtime/v1/l3_coverage",
        "seed": 830000,
        "episode_count": 512,
    }
    assert no_residual["model"]["disable_residual_update"] is True
    assert shuffled["training"]["shuffle_instruction_order"] is True


def test_phase_gates_and_frozen_hashes_are_preregistered():
    config = _load("inductor_v1_pilot.yaml")
    assert config["gates"]["phase_a"]["sealed_l2_query_exact"] == 0.70
    assert config["gates"]["phase_b"]["coverage_l3_query_exact"] == 0.50
    assert config["data"]["v0_checkpoint_sha256"] == "f0fd045bb2b08ef8a716b6b76fb47de8e20b4163881d94d5f14f4d81a1dccd09"
    assert config["data"]["interpreter_model_state_sha256"] == "4cf0b68d675c87fcbff9ac3613801017e9cba47acebe9e86f77137037ff6aae0"


def test_v1_checkpoint_round_trip_preserves_teacher_and_autoregressive_outputs(tmp_path):
    torch.manual_seed(302)
    model = SequentialResidualInductorV1(
        InductorV1ModelConfig(
            hidden_dim=32,
            color_embedding_dim=8,
            pair_encoder_blocks=1,
            pair_summary_slots=2,
            instruction_embedding_dim=4,
        )
    ).eval()
    inputs = torch.randint(0, 10, (2, 3, 8, 8))
    outputs = torch.randint(0, 10, (2, 3, 8, 8))
    targets = torch.tensor(
        [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]] * 2
    )
    teacher = model(inputs, outputs, targets)
    autoregressive = model.forward_autoregressive(inputs, outputs)
    path = tmp_path / "v1.pt"
    save_checkpoint_v1(path, model, {"test": True}, 302, 7)
    loaded, metadata = load_checkpoint_v1(path, torch.device("cpu"))
    assert metadata["step"] == 7
    actual_teacher = loaded(inputs, outputs, targets)
    actual_autoregressive = loaded.forward_autoregressive(inputs, outputs)
    assert all(torch.equal(teacher[key], actual_teacher[key]) for key in teacher)
    assert torch.equal(autoregressive[1], actual_autoregressive[1])
    assert torch.equal(autoregressive[2], actual_autoregressive[2])
