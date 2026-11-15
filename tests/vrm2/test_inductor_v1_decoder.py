from __future__ import annotations

import inspect

import torch

from vrm2.inductor.model_v1 import (
    InductorV1ModelConfig,
    SequentialResidualInductorV1,
    component_parameter_counts,
)
from vrm2.inductor.schema_v1 import validate_terminated_program
from vrm2.inductor.sequential_decoder import (
    SequentialDecoderConfig,
    SequentialResidualProgramDecoder,
    materialize_program,
)


def _small(disable: bool = False) -> SequentialResidualInductorV1:
    return SequentialResidualInductorV1(
        InductorV1ModelConfig(
            hidden_dim=32,
            color_embedding_dim=8,
            pair_encoder_blocks=1,
            pair_summary_slots=2,
            instruction_embedding_dim=4,
            disable_residual_update=disable,
        )
    ).eval()


def test_decoder_uses_one_cell_update_and_shared_head_set_without_position_modules():
    decoder = SequentialResidualProgramDecoder(SequentialDecoderConfig())
    assert isinstance(decoder.decoder_cell, torch.nn.GRUCell)
    assert isinstance(decoder.residual_update, torch.nn.Linear)
    for name in ("operator_head", "selector_head", "argument_type_head", "argument_value_head"):
        assert isinstance(getattr(decoder, name), torch.nn.Linear)
    names = tuple(name for name, _ in decoder.named_modules())
    assert not any("position" in name or "step_3" in name or "length_head" in name for name in names)
    assert not hasattr(decoder, "length_head")


def test_null_termination_forces_suffix_and_derives_length_from_first_null():
    logits = {
        "operator": torch.tensor([[[0.0, 9.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]]]),
        "selector": torch.randn(1, 3, 4),
        "argument_type": torch.randn(1, 3, 4),
        "argument_value": torch.randn(1, 3, 10),
    }
    lengths, programs, _ = materialize_program(logits)
    assert lengths.tolist() == [1]
    assert programs[0, 1:].eq(0).all()
    validate_terminated_program(1, programs[0].tolist())


def test_teacher_forced_path_uses_targets_and_autoregressive_path_is_hard():
    model = _small()
    inputs = torch.randint(0, 10, (2, 3, 8, 8))
    outputs = torch.randint(0, 10, (2, 3, 8, 8))
    targets = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 2, 3], [3, 3, 3, 1], [0, 0, 0, 0]],
        ]
    )
    teacher = model(inputs, outputs, targets)
    autoregressive, lengths, programs, _ = model.forward_autoregressive(inputs, outputs)
    assert all(value.shape[:2] == (2, 3) for value in teacher.values())
    assert all(value.shape[:2] == (2, 3) for value in autoregressive.values())
    assert lengths.dtype == torch.long and programs.dtype == torch.long
    for length, program in zip(lengths.tolist(), programs.tolist(), strict=True):
        validate_terminated_program(length, program)


def test_residual_update_receives_emitted_instruction_and_ablation_disables_update():
    task = torch.randn(2, 32)
    targets = torch.tensor(
        [
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[2, 2, 2, 9], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    decoder = SequentialResidualProgramDecoder(
        SequentialDecoderConfig(hidden_dim=32, instruction_embedding_dim=4)
    )
    captured = []
    handle = decoder.residual_update.register_forward_pre_hook(
        lambda _module, values: captured.append(values[0].detach().clone())
    )
    decoder.teacher_forced(task, targets)
    handle.remove()
    assert captured and not torch.equal(captured[0][0], captured[0][1])

    disabled = SequentialResidualProgramDecoder(
        SequentialDecoderConfig(
            hidden_dim=32, instruction_embedding_dim=4, disable_residual_update=True
        )
    )
    calls = []
    hook = disabled.residual_update.register_forward_hook(
        lambda *_args: calls.append(True)
    )
    disabled.teacher_forced(task, targets)
    hook.remove()
    assert calls == []


def test_support_permutation_invariance_and_no_query_api():
    torch.manual_seed(301)
    model = _small()
    inputs = torch.randint(0, 10, (2, 3, 8, 8))
    outputs = torch.randint(0, 10, (2, 3, 8, 8))
    first = model.forward_autoregressive(inputs, outputs)
    second = model.forward_autoregressive(inputs[:, [2, 0, 1]], outputs[:, [2, 0, 1]])
    assert torch.equal(first[1], second[1]) and torch.equal(first[2], second[2])
    parameters = tuple(inspect.signature(SequentialResidualInductorV1.forward).parameters)
    assert "query" not in " ".join(parameters)


def test_frozen_component_budget_is_exact():
    counts = component_parameter_counts(SequentialResidualInductorV1())
    assert counts == {
        "pair_encoder": 1_007_552,
        "set_aggregator": 132_352,
        "residual_initializer": 16_512,
        "shared_decoder_cell": 131_968,
        "shared_residual_update": 49_536,
        "instruction_embeddings": 8_800,
        "shared_field_heads": 2_838,
        "total": 1_349_558,
    }
    assert counts["total"] <= 1_350_000
