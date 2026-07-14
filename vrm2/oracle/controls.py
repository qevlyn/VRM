from __future__ import annotations

import collections
import random
from dataclasses import replace
from typing import Any, Sequence

import torch

from vrm2.data.microworld import sample_hidden_program
from vrm2.data.operations import SceneValidationError, execute_program
from vrm2.oracle.evaluate import predict_examples
from vrm2.oracle.types import OracleExample, TypedProgramIds


def _program_pool(examples: Sequence[OracleExample]) -> dict[int, list[int]]:
    pool: dict[int, list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        pool[example.program_length].append(index)
    return pool


def random_program_examples(
    examples: Sequence[OracleExample], seed: int
) -> tuple[OracleExample, ...]:
    rng = random.Random(seed)
    pool = _program_pool(examples)
    modified = []
    for example in examples:
        candidates = [
            index
            for index in pool[example.program_length]
            if examples[index].composition_signature != example.composition_signature
        ]
        if not candidates:
            raise ValueError("random-program control needs another signature of equal length")
        donor = examples[rng.choice(candidates)]
        modified.append(replace(example, typed_program_ids=donor.typed_program_ids))
    return tuple(modified)


def swapped_program_examples(
    examples: Sequence[OracleExample], seed: int
) -> tuple[OracleExample, ...]:
    rng = random.Random(seed)
    pool = _program_pool(examples)
    replacement: dict[int, TypedProgramIds] = {}
    for indices in pool.values():
        ordered = list(indices)
        rng.shuffle(ordered)
        for position, target_index in enumerate(ordered):
            donor = None
            for offset in range(1, len(ordered)):
                candidate = examples[ordered[(position + offset) % len(ordered)]]
                if candidate.composition_signature != examples[target_index].composition_signature:
                    donor = candidate
                    break
            if donor is None:
                raise ValueError("program-swap control needs multiple composition signatures")
            replacement[target_index] = donor.typed_program_ids
    return tuple(replace(example, typed_program_ids=replacement[index]) for index, example in enumerate(examples))


def reversed_program_examples(
    examples: Sequence[OracleExample],
) -> tuple[tuple[OracleExample, ...], tuple[int, ...]]:
    eligible = []
    original_indices = []
    for index, example in enumerate(examples):
        if example.program_length < 2:
            continue
        program = sample_hidden_program(example.composition_signature, example.episode_seed)
        reversed_steps = tuple(reversed(program.steps))
        try:
            reversed_output = execute_program(example.input_grid, reversed_steps)
        except SceneValidationError:
            continue
        if reversed_output == example.target_output_grid:
            continue
        reversed_ids = tuple(
            instruction.as_ids()
            for instruction in (step.typed_instruction for step in reversed_steps)
            if instruction is not None
        )
        if len(reversed_ids) != example.program_length:
            raise ValueError("reversed generator program is missing typed metadata")
        eligible.append(replace(example, typed_program_ids=reversed_ids))
        original_indices.append(index)
    return tuple(eligible), tuple(original_indices)


def null_program_examples(examples: Sequence[OracleExample]) -> tuple[OracleExample, ...]:
    return tuple(
        replace(example, typed_program_ids=tuple((0, 0, 0, 0) for _ in example.typed_program_ids))
        for example in examples
    )


def _control_report(
    predictions: torch.Tensor,
    correct_predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    exact = predictions.eq(targets).flatten(1).all(dim=1)
    changed = predictions.ne(correct_predictions).flatten(1).any(dim=1)
    return {
        "num_examples": predictions.shape[0],
        "exact": float(exact.float().mean().item()),
        "output_change_rate": float(changed.float().mean().item()),
    }


def run_controls(
    model: torch.nn.Module,
    examples: Sequence[OracleExample],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    correct_predictions = predict_examples(model, examples, batch_size, device)
    targets = torch.tensor([example.target_output_grid for example in examples], dtype=torch.long)
    correct = _control_report(correct_predictions, correct_predictions, targets)

    random_examples = random_program_examples(examples, seed)
    random_predictions = predict_examples(model, random_examples, batch_size, device)
    swapped_examples = swapped_program_examples(examples, seed + 1)
    swapped_predictions = predict_examples(model, swapped_examples, batch_size, device)
    null_examples = null_program_examples(examples)
    null_predictions = predict_examples(model, null_examples, batch_size, device)

    reversed_examples, reversed_indices = reversed_program_examples(examples)
    if reversed_examples:
        reversed_predictions = predict_examples(model, reversed_examples, batch_size, device)
        reversed_correct = correct_predictions[list(reversed_indices)]
        reversed_targets = targets[list(reversed_indices)]
        reverse_report = _control_report(reversed_predictions, reversed_correct, reversed_targets)
    else:
        reverse_report = {"num_examples": 0, "exact": 0.0, "output_change_rate": 0.0}

    return {
        "correct": correct,
        "random": _control_report(random_predictions, correct_predictions, targets),
        "swap": _control_report(swapped_predictions, correct_predictions, targets),
        "reverse": reverse_report,
        "null": _control_report(null_predictions, correct_predictions, targets),
    }
