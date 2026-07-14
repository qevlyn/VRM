from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import torch

from vrm2.inductor.evaluate import execute_programs
from vrm2.inductor.execution import evaluate_induced_program_file
from vrm2.inductor.freeze import checkpoint_audit
from vrm2.inductor.hashing import module_state_hashes
from vrm2.inductor.model import SupervisedTypedProgramInductorV0
from vrm2.inductor.train import train_one_seed
from vrm2.oracle.train import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def test_executor_stage_api_has_no_support_or_inductor_parameter():
    parameters = tuple(inspect.signature(evaluate_induced_program_file).parameters)
    assert parameters == (
        "queries_path", "programs_path", "interpreter_checkpoint", "batch_size", "device"
    )


def test_executor_module_import_does_not_load_episode_support_module():
    code = (
        "import sys; import vrm2.inductor.execution; "
        "assert 'vrm2.inductor.episodes' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_hard_executor_boundary_receives_long_integer_ids():
    class RecordingInterpreter(torch.nn.Module):
        def forward(self, grids, typed_program_ids):
            assert typed_program_ids.dtype == torch.long
            assert typed_program_ids.ndim == 3
            return torch.zeros(len(grids), typed_program_ids.shape[1], 10, 8, 8)

    execute_programs(
        RecordingInterpreter(),
        torch.zeros(2, 8, 8, dtype=torch.long),
        torch.tensor([2, 3]),
        torch.zeros(2, 3, 4, dtype=torch.long),
        2,
        torch.device("cpu"),
    )


def test_model_source_has_no_metadata_or_generator_dependency():
    source = inspect.getsource(inspect.getmodule(SupervisedTypedProgramInductorV0))
    forbidden = (
        "composition_signature", "episode_seed", "fingerprint", "split", "query_target",
        "vrm2.data.operations", "execute_operation", "operation_name", "generator_parameter"
    )
    assert all(term not in source for term in forbidden)


def test_training_source_never_opens_or_selects_on_sealed_data():
    source = inspect.getsource(train_one_seed)
    assert "sealed" not in source


def test_certified_interpreter_is_frozen_and_hash_stable():
    config = json.loads((ROOT / "configs/vrm2/inductor_v0_pilot.yaml").read_text())
    checkpoint = ROOT / config["data"]["interpreter_checkpoint"]
    interpreter, _ = load_checkpoint(checkpoint, torch.device("cpu"))
    before = module_state_hashes(interpreter)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    after = module_state_hashes(interpreter)
    assert before == after
    assert after["model_state_sha256"] == config["data"]["interpreter_model_state_sha256"]
    assert all(not parameter.requires_grad for parameter in interpreter.parameters())
    assert not interpreter.training
    audit = checkpoint_audit(checkpoint)
    assert audit["parameter_count"] == 741_930
    assert audit["checkpoint_step"] == 7_000
