from __future__ import annotations

import torch

from vrm2.oracle.controls import run_controls
from vrm2.oracle.dataset import make_oracle_batch
from vrm2.oracle.model import OracleInterpreterV0, OracleModelConfig
from vrm2.oracle.train import load_checkpoint, oracle_loss, save_checkpoint, set_deterministic_seed


def _tiny_model() -> OracleInterpreterV0:
    return OracleInterpreterV0(
        OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    )


def test_random_swap_reverse_and_null_controls_run(small_sealed_oracle_examples):
    model = _tiny_model().eval()
    report = run_controls(model, small_sealed_oracle_examples, 16, torch.device("cpu"), seed=44)
    assert set(report) == {"correct", "random", "swap", "reverse", "null"}
    assert all(report[name]["num_examples"] > 0 for name in report)


def test_checkpoint_reload_gives_identical_predictions(tmp_path, small_train_oracle_examples):
    set_deterministic_seed(91)
    model = _tiny_model().eval()
    example = small_train_oracle_examples[0]
    grid = torch.tensor([example.input_grid], dtype=torch.long)
    program = torch.tensor([example.typed_program_ids], dtype=torch.long)
    expected = model(grid, program)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, {"test": True}, seed=91, step=3)
    loaded, metadata = load_checkpoint(path, torch.device("cpu"))
    actual = loaded(grid, program)
    assert metadata["step"] == 3
    assert torch.equal(expected, actual)


def test_tiny_overfit_loss_reduces(small_train_oracle_examples):
    set_deterministic_seed(77)
    examples = [item for item in small_train_oracle_examples if item.program_length == 1][:4]
    batch = make_oracle_batch(examples)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = None
    final = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*batch.model_inputs())
        loss, _ = oracle_loss(logits, batch, lambda_intermediate=0.5)
        if initial is None:
            initial = float(loss.item())
        loss.backward()
        optimizer.step()
        final = float(loss.item())
    assert final is not None and initial is not None and final < initial * 0.7
