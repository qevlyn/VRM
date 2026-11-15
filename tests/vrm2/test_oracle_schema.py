from __future__ import annotations

import torch

from vrm2.data.microworld import sample_hidden_program
from vrm2.data.program_schema import (
    ArgumentTypeId,
    AxisValue,
    DirectionValue,
    OperatorId,
    SelectorId,
    factorize_instruction,
)
from vrm2.oracle.model import OracleInterpreterV0, OracleModelConfig
from vrm2.oracle.train import set_deterministic_seed


def test_typed_metadata_factorization():
    assert factorize_instruction("translate_leftmost", {"direction": "up"}).as_ids() == (
        OperatorId.TRANSLATE,
        SelectorId.LEFTMOST_OBJECT,
        ArgumentTypeId.CARDINAL_DIRECTION,
        DirectionValue.UP,
    )
    assert factorize_instruction("recolor_topmost", {"color": 7}).as_ids() == (
        OperatorId.RECOLOR,
        SelectorId.TOPMOST_OBJECT,
        ArgumentTypeId.TARGET_COLOR,
        7,
    )
    assert factorize_instruction("reflect_rows", {}).as_ids() == (
        OperatorId.REFLECT,
        SelectorId.ALL_CONTENT,
        ArgumentTypeId.AXIS,
        AxisValue.ROW,
    )


def test_generated_hidden_program_has_only_integer_typed_factors():
    program = sample_hidden_program("translate_leftmost->recolor_topmost->reflect_columns", 123)
    ids = tuple(instruction.as_ids() for instruction in program.typed_instructions)
    assert len(ids) == 3
    assert all(len(instruction) == 4 for instruction in ids)
    assert all(isinstance(value, int) for instruction in ids for value in instruction)


def test_deterministic_seed_reproduces_model_initialization():
    config = OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    set_deterministic_seed(42)
    first = OracleInterpreterV0(config)
    set_deterministic_seed(42)
    second = OracleInterpreterV0(config)
    assert all(torch.equal(left, right) for left, right in zip(first.state_dict().values(), second.state_dict().values()))


def test_hard_program_order_is_forwarded_in_order():
    model = OracleInterpreterV0(
        OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    ).eval()
    seen = []

    def capture(_module, inputs, _output):
        seen.append(inputs[0].detach().clone())

    handle = model.instruction_encoder.register_forward_hook(capture)
    program = torch.tensor([[[1, 1, 1, 4], [2, 2, 2, 7], [3, 3, 3, 1]]], dtype=torch.long)
    model(torch.zeros(1, 8, 8, dtype=torch.long), program)
    handle.remove()
    assert len(seen) == 3
    assert torch.equal(torch.stack(seen, dim=1), program)
