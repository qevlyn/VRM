from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes
from vrm2.data.types import make_grid
from vrm2.inductor.evaluate_v1 import execute_programs_v1
from vrm2.inductor.hashing import module_state_hashes
from vrm2.inductor.order_selector import reverse_predicted_program
from vrm2.inductor.schema_v1 import validate_terminated_program
from vrm2.oracle.train import load_checkpoint


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records) + b"\n")


@torch.inference_mode()
def execute_stage_b(
    query_path: Path,
    prediction_path: Path,
    interpreter_checkpoint: Path,
    output_path: Path,
    runtime_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Fresh Stage B. It reads query/program files and has no support-file API."""

    queries = [json.loads(line) for line in query_path.read_bytes().splitlines() if line]
    predictions = [
        json.loads(line) for line in prediction_path.read_bytes().splitlines() if line
    ]
    if [item["episode_id"] for item in queries] != [
        item["episode_id"] for item in predictions
    ]:
        raise ValueError("query and prediction episode IDs are not aligned")
    lengths = torch.tensor(
        [item["predicted_program_length"] for item in predictions], dtype=torch.long
    )
    programs = torch.tensor(
        [item["predicted_typed_program_ids"] for item in predictions], dtype=torch.long
    )
    reversed_programs = []
    reversal_eligible = []
    for length, program in zip(lengths.tolist(), programs.tolist(), strict=True):
        validate_terminated_program(length, program)
        reversed_program, eligible = reverse_predicted_program(length, program)
        reversed_programs.append(reversed_program)
        reversal_eligible.append(eligible)
    reversed_tensor = torch.tensor(reversed_programs, dtype=torch.long)
    inputs = torch.tensor(
        [make_grid(item["query_input_grid"], expected_size=8) for item in queries],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [make_grid(item["query_target_grid"], expected_size=8) for item in queries],
        dtype=torch.long,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    interpreter, metadata = load_checkpoint(interpreter_checkpoint, device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    hashes_before = module_state_hashes(interpreter)
    native, native_seconds = execute_programs_v1(
        interpreter, inputs, lengths, programs, batch_size, device
    )
    reversed_outputs, reversed_seconds = execute_programs_v1(
        interpreter, inputs, lengths, reversed_tensor, batch_size, device
    )
    native_exact = native.eq(targets).flatten(1).all(dim=1)
    reversed_exact = reversed_outputs.eq(targets).flatten(1).all(dim=1)
    changed = native.ne(reversed_outputs).flatten(1).any(dim=1)
    records = []
    for index, prediction in enumerate(predictions):
        records.append(
            {
                "episode_id": prediction["episode_id"],
                "checkpoint_role": prediction["checkpoint_role"],
                "native_query_prediction": native[index].tolist(),
                "reversed_query_prediction": reversed_outputs[index].tolist(),
                "native_query_exact": bool(native_exact[index].item()),
                "reversed_query_exact": bool(reversed_exact[index].item()),
                "output_changed_under_reversal": bool(changed[index].item()),
                "reversal_eligible": bool(reversal_eligible[index]),
            }
        )
    _write_jsonl(output_path, records)
    result = {
        "num_episodes": len(records),
        "checkpoint_role": predictions[0]["checkpoint_role"] if predictions else None,
        "query_file_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "prediction_file_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "output_file_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "interpreter_checkpoint_step": int(metadata["step"]),
        "interpreter_hashes_before": hashes_before,
        "interpreter_hashes_after": module_state_hashes(interpreter),
        "interpreter_parameters_frozen": all(
            not parameter.requires_grad for parameter in interpreter.parameters()
        ),
        "interpreter_eval_mode": not interpreter.training,
        "native_execution_seconds": native_seconds,
        "reversed_execution_seconds": reversed_seconds,
        "total_execution_seconds": native_seconds + reversed_seconds,
        "hard_integer_program_dtype": str(programs.dtype),
        "support_file_loaded": False,
    }
    _write_json(runtime_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order diagnostic fresh Stage B")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--interpreter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute_stage_b(
        args.queries,
        args.predictions,
        args.interpreter_checkpoint,
        args.output,
        args.runtime,
        args.batch_size,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
