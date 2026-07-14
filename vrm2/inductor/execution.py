from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import torch

from vrm2.data.types import make_grid
from vrm2.inductor.evaluate import execute_programs
from vrm2.inductor.hashing import module_state_hashes
from vrm2.inductor.schema import validate_typed_program
from vrm2.oracle.train import load_checkpoint


@torch.inference_mode()
def evaluate_induced_program_file(
    queries_path: Path,
    programs_path: Path,
    interpreter_checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Execute query grids from hard program records without accepting support data."""

    queries = [json.loads(line) for line in queries_path.read_bytes().splitlines() if line]
    programs = [json.loads(line) for line in programs_path.read_bytes().splitlines() if line]
    if [item["episode_id"] for item in queries] != [item["episode_id"] for item in programs]:
        raise ValueError("query/program episode IDs are not aligned")
    lengths = torch.tensor(
        [item["predicted_program_length"] for item in programs], dtype=torch.long
    )
    typed = torch.tensor(
        [item["predicted_typed_program_ids"] for item in programs], dtype=torch.long
    )
    for length, program in zip(lengths.tolist(), typed.tolist(), strict=True):
        validate_typed_program(length, program)
    inputs = torch.tensor(
        [make_grid(item["query_input_grid"], expected_size=8) for item in queries],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [make_grid(item["query_target_grid"], expected_size=8) for item in queries],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [item["evaluator_private"]["program_length"] for item in queries], dtype=torch.long
    )
    target_programs = torch.tensor(
        [item["evaluator_private"]["typed_program_ids"] for item in queries], dtype=torch.long
    )
    interpreter, metadata = load_checkpoint(interpreter_checkpoint, device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    interpreter_hashes_before = module_state_hashes(interpreter)
    predictions, execution_seconds = execute_programs(
        interpreter, inputs, lengths, typed, batch_size, device
    )
    exact = predictions.eq(targets).flatten(1).all(dim=1)
    active = torch.arange(3).unsqueeze(0) < target_lengths.unsqueeze(1)
    instruction_exact = typed.eq(target_programs).all(dim=2)
    program_exact = lengths.eq(target_lengths) & (instruction_exact | ~active).all(dim=1)
    interpreter_hashes_after = module_state_hashes(interpreter)
    return {
        "num_episodes": len(queries),
        "query_exact": float(exact.float().mean().item()),
        "program_exact": float(program_exact.float().mean().item()),
        "execution_seconds": execution_seconds,
        "execution_ms_per_grid": execution_seconds * 1000.0 / len(queries),
        "interpreter_checkpoint_step": metadata["step"],
        "query_file_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "program_file_sha256": hashlib.sha256(programs_path.read_bytes()).hexdigest(),
        "interpreter_hashes_before": interpreter_hashes_before,
        "interpreter_hashes_after": interpreter_hashes_after,
        "interpreter_parameters_frozen": all(
            not parameter.requires_grad for parameter in interpreter.parameters()
        ),
        "interpreter_eval_mode": not interpreter.training,
        "support_file_loaded": False,
        "hard_integer_program_dtype": str(typed.dtype),
    }
