from __future__ import annotations

import torch

from vrm2.oracle.model import (
    OracleInterpreterV0,
    OracleModelConfig,
    SharedExecutionBlock,
    parameter_count,
)


def test_every_instruction_reuses_one_execution_block_instance():
    model = OracleInterpreterV0(
        OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    )
    blocks = [module for module in model.modules() if isinstance(module, SharedExecutionBlock)]
    assert blocks == [model.shared_execution_block]
    calls = []
    handle = model.shared_execution_block.register_forward_hook(lambda module, _inputs, _output: calls.append(id(module)))
    grids = torch.zeros(2, 8, 8, dtype=torch.long)
    programs = torch.tensor([[[1, 1, 1, 1], [2, 2, 2, 4], [3, 3, 3, 2]]] * 2)
    logits = model(grids, programs)
    handle.remove()
    assert logits.shape == (2, 3, 10, 8, 8)
    assert calls == [id(model.shared_execution_block)] * 3


def test_default_model_is_within_parameter_budget():
    model = OracleInterpreterV0()
    assert parameter_count(model) == 735_787
    assert parameter_count(model) < 5_000_000
