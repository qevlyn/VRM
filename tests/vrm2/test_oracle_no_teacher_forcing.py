from __future__ import annotations

import inspect

import torch

from vrm2.oracle.model import OracleInterpreterV0, OracleModelConfig
from vrm2.oracle.train import oracle_loss


def test_forward_interface_has_no_target_or_intermediate_input():
    parameters = tuple(inspect.signature(OracleInterpreterV0.forward).parameters)
    assert parameters == ("self", "input_grid", "typed_program_ids")


def test_targets_change_loss_but_never_prediction_runtime_input(small_train_oracle_examples):
    model = OracleInterpreterV0(
        OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    ).eval()
    example = next(item for item in small_train_oracle_examples if item.program_length == 1)
    input_grid = torch.tensor([example.input_grid], dtype=torch.long)
    program = torch.tensor([example.typed_program_ids], dtype=torch.long)
    first = model(input_grid, program)
    second = model(input_grid, program)
    assert torch.equal(first, second)


def test_zero_intermediate_ablation_removes_auxiliary_loss(small_train_oracle_examples):
    from vrm2.oracle.dataset import make_oracle_batch

    examples = [item for item in small_train_oracle_examples if item.program_length == 1][:2]
    batch = make_oracle_batch(examples)
    logits = torch.randn(len(examples), 1, 10, 8, 8)
    total, parts = oracle_loss(logits, batch, lambda_intermediate=0.0)
    assert torch.isclose(total, torch.tensor(parts["final"]))
