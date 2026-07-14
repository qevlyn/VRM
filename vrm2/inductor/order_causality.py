from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from vrm2.data.operations import SceneValidationError, execute_program
from vrm2.data.program_schema import ProgramInstruction
from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.data.types import Grid, OperationStep, grid_as_lists
from vrm2.inductor.evaluate_v1 import execute_programs_v1
from vrm2.inductor.schema_v1 import validate_terminated_program
from vrm2.inductor.types import PaddedTypedProgramIds, QueryEpisode, SupportEpisode


NULL_INSTRUCTION = (0, 0, 0, 0)
_DIRECTION_NAMES = {1: "up", 2: "down", 3: "left", 4: "right"}


def reverse_two_instruction_program(
    program: Sequence[Sequence[int]],
) -> PaddedTypedProgramIds:
    normalized = tuple(tuple(int(value) for value in item) for item in program)
    validate_terminated_program(2, normalized)
    return (normalized[1], normalized[0], NULL_INSTRUCTION)  # type: ignore[return-value]


def instructions_distinct(program: Sequence[Sequence[int]]) -> bool:
    normalized = tuple(tuple(int(value) for value in item) for item in program)
    validate_terminated_program(2, normalized)
    return normalized[0] != normalized[1]


def instruction_to_operation_step(instruction: Sequence[int]) -> OperationStep:
    operator, selector, argument_type, value = (int(item) for item in instruction)
    typed = ProgramInstruction(operator, selector, argument_type, value)
    if operator == 1 and selector in (1, 2) and argument_type == 1 and value in _DIRECTION_NAMES:
        name = "translate_leftmost" if selector == 1 else "translate_topmost"
        return OperationStep(name, {"direction": _DIRECTION_NAMES[value]}, typed)
    if operator == 2 and selector in (1, 2) and argument_type == 2 and 1 <= value <= 9:
        name = "recolor_leftmost" if selector == 1 else "recolor_topmost"
        return OperationStep(name, {"color": value}, typed)
    if operator == 3 and selector == 3 and argument_type == 3 and value in (1, 2):
        return OperationStep("reflect_rows" if value == 1 else "reflect_columns", {}, typed)
    raise ValueError(f"instruction is not legal under vrm2-typed-program-v1: {instruction}")


def execute_generator_program(grid: Grid, program: Sequence[Sequence[int]]) -> Grid:
    normalized = tuple(tuple(int(value) for value in item) for item in program)
    validate_terminated_program(2, normalized)
    steps = tuple(instruction_to_operation_step(item) for item in normalized[:2])
    return execute_program(grid, steps)


def generator_valid_orientations(
    inputs: Sequence[Grid],
    forward: Sequence[Sequence[int]],
    reverse: Sequence[Sequence[int]],
    *,
    executor: Callable[[Grid, Sequence[Sequence[int]]], Grid] = execute_generator_program,
) -> tuple[bool, tuple[Grid, ...], tuple[Grid, ...], str | None]:
    try:
        validate_terminated_program(2, forward)
        validate_terminated_program(2, reverse)
    except ValueError as error:
        return False, (), (), f"invalid_reverse:{error}"
    try:
        forward_outputs = tuple(executor(grid, forward) for grid in inputs)
        reverse_outputs = tuple(executor(grid, reverse) for grid in inputs)
    except (SceneValidationError, ValueError) as error:
        return False, (), (), f"execution_failure:{error}"
    return True, forward_outputs, reverse_outputs, None


def query_is_order_sensitive(
    forward_query: Grid, reverse_query: Grid, query_target: Grid
) -> bool:
    return forward_query == query_target and forward_query != reverse_query


def support_disambiguation_count(
    reverse_support_outputs: Sequence[Grid], support_targets: Sequence[Grid]
) -> int:
    if len(reverse_support_outputs) != 3 or len(support_targets) != 3:
        raise ValueError("order identifiability requires exactly three supports")
    return sum(
        reversed_output != target
        for reversed_output, target in zip(
            reverse_support_outputs, support_targets, strict=True
        )
    )


def interpreter_orientations_faithful(
    generator_forward: Sequence[Grid],
    generator_reverse: Sequence[Grid],
    interpreter_forward: Sequence[Grid],
    interpreter_reverse: Sequence[Grid],
) -> bool:
    return tuple(generator_forward) == tuple(interpreter_forward) and tuple(
        generator_reverse
    ) == tuple(interpreter_reverse)


def grid_fingerprint(grid: Grid) -> str:
    return sha256_hex(canonical_json_bytes(grid_as_lists(grid)))


def pair_fingerprint(input_grid: Grid, output_grid: Grid) -> str:
    return sha256_hex(
        canonical_json_bytes([grid_as_lists(input_grid), grid_as_lists(output_grid)])
    )


def subset_manifest_has_predictions(payload: Mapping[str, Any]) -> bool:
    prohibited = ("predicted_", "checkpoint_id", "query_prediction", "model_output")

    def visit(value: Any) -> bool:
        if isinstance(value, Mapping):
            if any(any(token in str(key).lower() for token in prohibited) for key in value):
                return True
            return any(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(visit(item) for item in value)
        return False

    return visit(payload)


@torch.inference_mode()
def build_order_causal_subset(
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    interpreter: torch.nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    start = time.perf_counter()
    if len(supports) != 193 or len(queries) != 193:
        raise ValueError("frozen sealed-L2 diagnostic requires exactly 193 episodes")
    if [item.episode_id for item in supports] != [item.episode_id for item in queries]:
        raise ValueError("support and query episode IDs are not aligned")

    records: list[dict[str, Any]] = []
    reverse_valid_records: list[dict[str, Any]] = []
    invalid_reverse = 0
    execution_failure = 0
    for support, query in zip(supports, queries, strict=True):
        if support.target_program_length != 2 or query.evaluator_program_length != 2:
            raise ValueError("sealed-L2 artifact contains a non-L2 episode")
        forward = tuple(tuple(item) for item in support.target_typed_program_ids)
        if forward != tuple(tuple(item) for item in query.evaluator_typed_program_ids):
            raise ValueError("support/query ground-truth programs disagree")
        distinct = instructions_distinct(forward)
        reverse = reverse_two_instruction_program(forward)
        base: dict[str, Any] = {
            "episode_id": support.episode_id,
            "forward_program": [list(item) for item in forward],
            "reverse_program": [list(item) for item in reverse],
            "distinct_instructions": distinct,
            "reverse_valid": False,
            "query_order_sensitive": False,
            "support_identifiable": False,
            "interpreter_faithful": False,
            "final_causal_subset": False,
            "disambiguating_supports": 0,
        }
        records.append(base)
        if not distinct:
            continue
        inputs = (*support.support_input_grids, query.query_input_grid)
        valid, forward_outputs, reverse_outputs, error = generator_valid_orientations(
            inputs, forward, reverse
        )
        if not valid:
            base["reverse_failure"] = error
            if error and error.startswith("invalid_reverse"):
                invalid_reverse += 1
            else:
                execution_failure += 1
            continue
        base["reverse_valid"] = True
        base["generator_forward_outputs"] = [grid_as_lists(item) for item in forward_outputs]
        base["generator_reverse_outputs"] = [grid_as_lists(item) for item in reverse_outputs]
        base["forward_query_matches_target"] = forward_outputs[3] == query.query_target_grid
        base["query_order_sensitive"] = query_is_order_sensitive(
            forward_outputs[3], reverse_outputs[3], query.query_target_grid
        )
        disambiguating = support_disambiguation_count(
            reverse_outputs[:3], support.support_output_grids
        )
        base["disambiguating_supports"] = disambiguating
        base["support_identifiable"] = base["query_order_sensitive"] and disambiguating > 0
        base["support_input_fingerprints"] = [
            grid_fingerprint(item) for item in support.support_input_grids
        ]
        base["support_pair_fingerprints"] = [
            pair_fingerprint(input_grid, output_grid)
            for input_grid, output_grid in zip(
                support.support_input_grids, support.support_output_grids, strict=True
            )
        ]
        base["query_input_fingerprint"] = grid_fingerprint(query.query_input_grid)
        base["query_pair_fingerprint"] = pair_fingerprint(
            query.query_input_grid, query.query_target_grid
        )
        reverse_valid_records.append(base)

    flat_inputs: list[Grid] = []
    flat_lengths: list[int] = []
    flat_programs: list[list[list[int]]] = []
    for record in reverse_valid_records:
        support = supports[records.index(record)]
        query = queries[records.index(record)]
        inputs = (*support.support_input_grids, query.query_input_grid)
        for program_key in ("forward_program", "reverse_program"):
            for grid in inputs:
                flat_inputs.append(grid)
                flat_lengths.append(2)
                flat_programs.append(record[program_key])
    interpreter_seconds = 0.0
    interpreter_outputs: list[Grid] = []
    if flat_inputs:
        predictions, interpreter_seconds = execute_programs_v1(
            interpreter,
            torch.tensor(flat_inputs, dtype=torch.long),
            torch.tensor(flat_lengths, dtype=torch.long),
            torch.tensor(flat_programs, dtype=torch.long),
            batch_size,
            device,
        )
        interpreter_outputs = [
            tuple(tuple(int(value) for value in row) for row in grid.tolist())
            for grid in predictions
        ]

    for index, record in enumerate(reverse_valid_records):
        block = interpreter_outputs[index * 8 : (index + 1) * 8]
        forward_outputs = tuple(
            tuple(tuple(int(value) for value in row) for row in grid)
            for grid in record["generator_forward_outputs"]
        )
        reverse_outputs = tuple(
            tuple(tuple(int(value) for value in row) for row in grid)
            for grid in record["generator_reverse_outputs"]
        )
        faithful = interpreter_orientations_faithful(
            forward_outputs, reverse_outputs, block[:4], block[4:]
        )
        record["interpreter_faithful"] = faithful
        record["interpreter_agreement_audit"] = {
            "forward_by_input": [a == b for a, b in zip(forward_outputs, block[:4], strict=True)],
            "reverse_by_input": [a == b for a, b in zip(reverse_outputs, block[4:], strict=True)],
            "all_inputs_both_orientations": faithful,
        }
        record["final_causal_subset"] = bool(record["support_identifiable"] and faithful)

    final_records = [record for record in records if record["final_causal_subset"]]
    # Keep generator outputs only in the persisted final subset. Other records retain
    # stage flags and programs but cannot leak model-dependent selection.
    episode_index = []
    for record in records:
        compact = {
            key: value
            for key, value in record.items()
            if key not in ("generator_forward_outputs", "generator_reverse_outputs")
        }
        episode_index.append(compact)
    disambiguation_distribution = {
        str(count): sum(
            record["query_order_sensitive"] and record["disambiguating_supports"] == count
            for record in records
        )
        for count in (0, 1, 2, 3)
    }
    funnel = {
        "all_sealed_l2": len(records),
        "distinct_instructions": sum(record["distinct_instructions"] for record in records),
        "reverse_valid": sum(record["reverse_valid"] for record in records),
        "query_order_sensitive": sum(record["query_order_sensitive"] for record in records),
        "support_identifiable": sum(record["support_identifiable"] for record in records),
        "interpreter_faithful": sum(
            record["support_identifiable"] and record["interpreter_faithful"]
            for record in records
        ),
        "final_causal_subset": len(final_records),
    }
    result = {
        "manifest_version": "vrm2-order-causality-subset-v1",
        "model_independent_selection": True,
        "funnel": funnel,
        "reverse_validity_audit": {
            "eligible": funnel["reverse_valid"],
            "invalid_reverse": invalid_reverse,
            "execution_failure": execution_failure,
        },
        "disambiguating_support_distribution": disambiguation_distribution,
        "stage_episode_ids": {
            "query_order_sensitive": [
                record["episode_id"] for record in records if record["query_order_sensitive"]
            ],
            "support_identifiable": [
                record["episode_id"] for record in records if record["support_identifiable"]
            ],
            "final_causal_subset": [record["episode_id"] for record in final_records],
        },
        "episode_index": episode_index,
        "final_records": final_records,
        "compute": {
            "subset_construction_seconds": time.perf_counter() - start,
            "interpreter_fidelity_execution_seconds": interpreter_seconds,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        },
    }
    if subset_manifest_has_predictions(result):
        raise AssertionError("subset manifest contains model predictions")
    return result


def combined_episode_corpus_sha256(support_path: Path, query_path: Path) -> str:
    return sha256_hex(support_path.read_bytes() + query_path.read_bytes())
