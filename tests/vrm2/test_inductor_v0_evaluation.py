from __future__ import annotations

import torch

from vrm2.inductor.model import InductorModelConfig, SupervisedTypedProgramInductorV0
from vrm2.inductor.train import load_inductor_checkpoint, save_checkpoint


def test_inductor_checkpoint_round_trip_preserves_logits_and_hard_state(tmp_path):
    torch.manual_seed(203)
    model = SupervisedTypedProgramInductorV0(
        InductorModelConfig(hidden_dim=32, color_embedding_dim=8, pair_encoder_blocks=1, pair_summary_slots=2)
    ).eval()
    inputs = torch.randint(0, 10, (2, 3, 8, 8))
    outputs = torch.randint(0, 10, (2, 3, 8, 8))
    expected = model(inputs, outputs)
    path = tmp_path / "inductor.pt"
    save_checkpoint(path, model, {"test": True}, seed=203, step=9)
    loaded, metadata = load_inductor_checkpoint(path, torch.device("cpu"))
    actual = loaded(inputs, outputs)
    assert metadata["step"] == 9
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
