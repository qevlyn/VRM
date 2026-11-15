from __future__ import annotations

import inspect

import torch

from vrm2.inductor.model import (
    InductorModelConfig,
    SupervisedTypedProgramInductorV0,
    inductor_parameter_count,
    materialize_hard_program,
)
from vrm2.inductor.schema import validate_typed_program


def _model() -> SupervisedTypedProgramInductorV0:
    return SupervisedTypedProgramInductorV0(
        InductorModelConfig(hidden_dim=32, color_embedding_dim=8, pair_encoder_blocks=1, pair_summary_slots=2)
    ).eval()


def test_inductor_api_has_no_query_grid_parameter():
    assert tuple(inspect.signature(SupervisedTypedProgramInductorV0.forward).parameters) == (
        "self", "support_input_grids", "support_output_grids", "support_mask"
    )


def test_support_permutations_preserve_logits_and_hard_programs():
    torch.manual_seed(201)
    model = _model()
    inputs = torch.randint(0, 10, (2, 3, 8, 8))
    outputs = torch.randint(0, 10, (2, 3, 8, 8))
    first = model(inputs, outputs)
    second = model(inputs[:, [2, 0, 1]], outputs[:, [2, 0, 1]])
    for key in first:
        assert torch.allclose(first[key], second[key], atol=1e-6, rtol=1e-6)
    first_hard = materialize_hard_program(first)[:2]
    second_hard = materialize_hard_program(second)[:2]
    assert all(torch.equal(left, right) for left, right in zip(first_hard, second_hard, strict=True))


def test_every_materialized_program_is_schema_valid_and_deterministic():
    torch.manual_seed(202)
    model = _model()
    inputs = torch.randint(0, 10, (4, 3, 8, 8))
    outputs = torch.randint(0, 10, (4, 3, 8, 8))
    first = materialize_hard_program(model(inputs, outputs))
    second = materialize_hard_program(model(inputs, outputs))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    for length, program in zip(first[0].tolist(), first[1].tolist(), strict=True):
        validate_typed_program(length, program)


def test_default_inductor_parameter_count_is_frozen_below_budget():
    assert inductor_parameter_count(SupervisedTypedProgramInductorV0()) == 1_148_676
    assert inductor_parameter_count(SupervisedTypedProgramInductorV0()) < 2_000_000
