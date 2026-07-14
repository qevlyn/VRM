from __future__ import annotations

import collections
import time
from typing import Any, Sequence

import torch

from vrm2.oracle.dataset import iter_oracle_batches
from vrm2.oracle.types import OracleExample

_OPERATOR_LABELS = {0: "NULL", 1: "TRANSLATE", 2: "RECOLOR", 3: "REFLECT"}
_SELECTOR_LABELS = {0: "NULL", 1: "LEFTMOST_OBJECT", 2: "TOPMOST_OBJECT", 3: "ALL_CONTENT"}
_ARGUMENT_TYPE_LABELS = {0: "NULL", 1: "CARDINAL_DIRECTION", 2: "TARGET_COLOR", 3: "AXIS"}


def _rate(correct: int, total: int) -> float:
    return correct / total if total else 0.0


@torch.inference_mode()
def predict_examples(
    model: torch.nn.Module,
    examples: Sequence[OracleExample],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    predictions: list[torch.Tensor | None] = [None] * len(examples)
    model.eval()
    for batch, indices in iter_oracle_batches(examples, batch_size, shuffle=False, seed=0):
        batch = batch.to(device)
        logits = model(*batch.model_inputs())
        final = logits[:, -1].argmax(dim=1).cpu()
        for local_index, original_index in enumerate(indices):
            predictions[original_index] = final[local_index]
    if any(item is None for item in predictions):
        raise AssertionError("evaluation did not produce every prediction")
    return torch.stack([item for item in predictions if item is not None])


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    examples: Sequence[OracleExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot evaluate an empty oracle corpus")
    model.eval()
    final_correct = 0
    cell_correct = 0
    cell_total = 0
    intermediate_correct: collections.Counter[int] = collections.Counter()
    intermediate_total: collections.Counter[int] = collections.Counter()
    length_counts: collections.Counter[int] = collections.Counter()
    length_correct: collections.Counter[int] = collections.Counter()
    signature_counts: collections.Counter[str] = collections.Counter()
    signature_correct: collections.Counter[str] = collections.Counter()
    factor_counts: dict[str, collections.Counter[str]] = {
        "operator": collections.Counter(),
        "selector": collections.Counter(),
        "argument_type": collections.Counter(),
    }
    factor_correct: dict[str, collections.Counter[str]] = {
        "operator": collections.Counter(),
        "selector": collections.Counter(),
        "argument_type": collections.Counter(),
    }
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = time.perf_counter()

    for cpu_batch, indices in iter_oracle_batches(examples, batch_size, shuffle=False, seed=0):
        batch = cpu_batch.to(device)
        logits = model(*batch.model_inputs())
        predictions = logits.argmax(dim=2)
        final_predictions = predictions[:, -1]
        final_exact = final_predictions.eq(batch.target_output_grid).flatten(1).all(dim=1)
        final_correct += int(final_exact.sum().item())
        cell_correct += int(final_predictions.eq(batch.target_output_grid).sum().item())
        cell_total += batch.target_output_grid.numel()

        intermediate_exact = predictions.eq(batch.intermediate_target_grids).flatten(2).all(dim=2)
        for step_index in range(intermediate_exact.shape[1]):
            intermediate_correct[step_index + 1] += int(intermediate_exact[:, step_index].sum().item())
            intermediate_total[step_index + 1] += intermediate_exact.shape[0]

        for local_index, original_index in enumerate(indices):
            example = examples[original_index]
            is_correct = int(final_exact[local_index].item())
            length_counts[example.program_length] += 1
            length_correct[example.program_length] += is_correct
            signature_counts[example.composition_signature] += 1
            signature_correct[example.composition_signature] += is_correct
            factors = {
                "operator": {_OPERATOR_LABELS[instruction[0]] for instruction in example.typed_program_ids},
                "selector": {_SELECTOR_LABELS[instruction[1]] for instruction in example.typed_program_ids},
                "argument_type": {
                    _ARGUMENT_TYPE_LABELS[instruction[2]] for instruction in example.typed_program_ids
                },
            }
            for factor_name, labels in factors.items():
                for label in labels:
                    factor_counts[factor_name][label] += 1
                    factor_correct[factor_name][label] += is_correct

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "num_examples": len(examples),
        "exact": _rate(final_correct, len(examples)),
        "cell_accuracy": _rate(cell_correct, cell_total),
        "intermediate_exact_by_step": {
            str(step): _rate(intermediate_correct[step], intermediate_total[step])
            for step in sorted(intermediate_total)
        },
        "by_program_length": {
            str(length): {
                "count": length_counts[length],
                "exact": _rate(length_correct[length], length_counts[length]),
            }
            for length in sorted(length_counts)
        },
        "by_composition": {
            signature: {
                "count": signature_counts[signature],
                "exact": _rate(signature_correct[signature], signature_counts[signature]),
            }
            for signature in sorted(signature_counts)
        },
        "by_operator": _factor_report(factor_counts["operator"], factor_correct["operator"]),
        "by_selector": _factor_report(factor_counts["selector"], factor_correct["selector"]),
        "by_argument_type": _factor_report(
            factor_counts["argument_type"], factor_correct["argument_type"]
        ),
        "wall_seconds": elapsed,
        "examples_per_second": len(examples) / elapsed,
        "inference_ms_per_grid": elapsed * 1000.0 / len(examples),
    }


def _factor_report(
    counts: collections.Counter[str], correct: collections.Counter[str]
) -> dict[str, dict[str, float | int]]:
    return {
        label: {"count": counts[label], "exact": _rate(correct[label], counts[label])}
        for label in sorted(counts)
    }
