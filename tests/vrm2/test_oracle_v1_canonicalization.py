from __future__ import annotations

import inspect
import subprocess
import sys

import torch

from vrm2.oracle.model import OracleModelConfig, SharedExecutionBlock, parameter_count
from vrm2.oracle.model_v1 import OracleInterpreterV1
from vrm2.oracle.train import load_checkpoint, save_checkpoint


def _small_model() -> OracleInterpreterV1:
    return OracleInterpreterV1(
        OracleModelConfig(
            hidden_dim=32,
            color_embedding_dim=16,
            workspace_tokens=2,
            shared_execution_layers=1,
        )
    )


def _program(batch_size: int, length: int = 3) -> torch.Tensor:
    instructions = [[1, 1, 1, 1], [2, 2, 2, 4], [3, 3, 3, 2]]
    return torch.tensor([instructions[:length]] * batch_size, dtype=torch.long)


def test_rollout_forwards_hard_one_hot_grid_into_next_encoder():
    torch.manual_seed(100)
    model = _small_model().train()
    received = []
    original = model.encode_canonical_grid

    def record(canonical_grid: torch.Tensor, instruction: torch.Tensor) -> torch.Tensor:
        received.append(canonical_grid.detach().clone())
        return original(canonical_grid, instruction)

    model.encode_canonical_grid = record  # type: ignore[method-assign]
    model(torch.randint(0, 10, (2, 8, 8)), _program(2, length=2))
    assert len(received) == 2
    boundary = received[1]
    assert torch.allclose(boundary.sum(dim=-1), torch.ones_like(boundary[..., 0]), atol=1e-6)
    assert torch.equal(
        (boundary.abs() > 1e-6).sum(dim=-1),
        torch.ones_like(boundary[..., 0], dtype=torch.long),
    )


def test_later_step_loss_has_finite_nonzero_gradient_through_earlier_logits():
    torch.manual_seed(101)
    model = _small_model().train()
    head_outputs = []

    def retain_head_output(_module, _inputs, output):
        output.retain_grad()
        head_outputs.append(output)

    handle = model.output_head.register_forward_hook(retain_head_output)
    logits = model(torch.randint(0, 10, (2, 8, 8)), _program(2, length=2))
    later_loss = logits[:, 1].square().mean()
    later_loss.backward()
    handle.remove()
    earlier_gradient = head_outputs[0].grad
    assert earlier_gradient is not None
    assert torch.isfinite(earlier_gradient).all()
    assert torch.count_nonzero(earlier_gradient).item() > 0


def test_evaluation_is_invariant_to_non_argmax_logits_across_boundary():
    torch.manual_seed(102)
    model = _small_model().eval()
    winner = torch.randint(0, 10, (2, 8, 8))
    first = torch.full((2, 10, 8, 8), -4.0)
    second = torch.randn(2, 10, 8, 8) * 0.5 - 8.0
    first.scatter_(1, winner.unsqueeze(1), 4.0)
    second.scatter_(1, winner.unsqueeze(1), 7.0)
    canonical_first = model.canonicalize_predicted_logits(first)
    canonical_second = model.canonicalize_predicted_logits(second)
    assert torch.equal(canonical_first, canonical_second)
    instruction = _program(2, length=1)[:, 0]
    assert torch.equal(
        model.execute_canonical_step(canonical_first, instruction),
        model.execute_canonical_step(canonical_second, instruction),
    )


def test_rollout_canonical_state_equals_fresh_integer_grid_encoding():
    torch.manual_seed(103)
    model = _small_model().eval()
    logits = torch.randn(2, 10, 8, 8)
    predicted_grid = logits.argmax(dim=1)
    instruction_ids = _program(2, length=1)[:, 0]
    instruction = model.instruction_encoder(instruction_ids)
    rollout_state = model.encode_canonical_grid(
        model.canonicalize_predicted_logits(logits), instruction
    )
    fresh_state = model.encode_canonical_grid(
        model.canonicalize_input_grid(predicted_grid), instruction
    )
    assert torch.equal(rollout_state, fresh_state)


def test_next_transition_interface_has_no_prior_hidden_or_workspace_argument():
    parameters = tuple(inspect.signature(OracleInterpreterV1.execute_canonical_step).parameters)
    assert parameters == ("self", "canonical_grid", "instruction_ids")
    source = inspect.getsource(OracleInterpreterV1.forward)
    assert "next_state" not in source
    assert "workspace" not in source


def test_all_instruction_positions_reuse_one_execution_block_instance():
    model = _small_model().eval()
    blocks = [module for module in model.modules() if isinstance(module, SharedExecutionBlock)]
    assert blocks == [model.shared_execution_block]
    calls = []
    handle = model.shared_execution_block.register_forward_hook(
        lambda module, _inputs, _output: calls.append(id(module))
    )
    model(torch.zeros(2, 8, 8, dtype=torch.long), _program(2))
    handle.remove()
    assert calls == [id(model.shared_execution_block)] * 3


def test_v1_forward_is_deterministic_and_within_parameter_budget():
    torch.manual_seed(104)
    model = OracleInterpreterV1().eval()
    grids = torch.randint(0, 10, (2, 8, 8))
    programs = _program(2)
    first = model(grids, programs)
    second = model(grids, programs)
    assert torch.equal(first, second)
    assert parameter_count(model) == 741_930
    assert parameter_count(model) < 5_000_000


def test_v1_checkpoint_round_trip_preserves_predictions(tmp_path):
    torch.manual_seed(105)
    model = _small_model().eval()
    grids = torch.randint(0, 10, (2, 8, 8))
    programs = _program(2)
    expected = model(grids, programs)
    checkpoint = tmp_path / "v1.pt"
    save_checkpoint(checkpoint, model, {"model_version": "v1"}, seed=105, step=7)
    loaded, metadata = load_checkpoint(checkpoint, torch.device("cpu"))
    actual = loaded(grids, programs)
    assert isinstance(loaded, OracleInterpreterV1)
    assert metadata["model_version"] == "v1"
    assert metadata["step"] == 7
    assert torch.equal(expected, actual)


def test_v1_model_source_has_no_generator_executor_or_semantic_branch():
    source = inspect.getsource(inspect.getmodule(OracleInterpreterV1))
    forbidden = (
        "vrm2.data.operations",
        "execute_operation",
        "execute_program",
        "if operator",
        "if selector",
        "if step_index == 2",
    )
    assert all(term not in source for term in forbidden)


def test_v1_runtime_import_does_not_load_generator_execution_module():
    code = (
        "import sys; import vrm2.oracle.model_v1; "
        "assert 'vrm2.data.operations' not in sys.modules; "
        "assert 'vrm2.data.microworld' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
