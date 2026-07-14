from __future__ import annotations

from typing import Any, Sequence

import torch

from vrm2.oracle.dataset import iter_oracle_batches
from vrm2.oracle.model_v1 import OracleInterpreterV1
from vrm2.oracle.types import OracleExample


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@torch.inference_mode()
def run_rollout_diagnostics(
    model: torch.nn.Module,
    examples: Sequence[OracleExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run bounded evaluation-only diagnostics on sealed length-3 examples."""

    if not isinstance(model, OracleInterpreterV1):
        raise TypeError("rollout diagnostics require OracleInterpreterV1")
    length3 = tuple(example for example in examples if example.program_length == 3)
    if not length3:
        raise ValueError("rollout diagnostics require length-3 examples")
    model.eval()

    free_correct = [0, 0, 0]
    oracle_reset_correct = [0, 0, 0]
    entropy_sums = [0.0, 0.0, 0.0]
    confidence_sums = [0.0, 0.0, 0.0]
    confidence_cells = [0, 0, 0]
    reset_logit_equal = 0
    reset_prediction_equal = 0
    reset_transitions = 0
    maximum_reset_logit_difference = 0.0

    for cpu_batch, _ in iter_oracle_batches(
        length3, batch_size, shuffle=False, seed=0
    ):
        batch = cpu_batch.to(device)
        logits = model(*batch.model_inputs())
        predictions = logits.argmax(dim=2)
        exact = predictions.eq(batch.intermediate_target_grids).flatten(2).all(dim=2)
        probabilities = logits.softmax(dim=2)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=2)
        confidence = probabilities.max(dim=2).values

        for step_index in range(3):
            free_correct[step_index] += int(exact[:, step_index].sum().item())
            entropy_sums[step_index] += float(entropy[:, step_index].sum().item())
            confidence_sums[step_index] += float(confidence[:, step_index].sum().item())
            confidence_cells[step_index] += confidence[:, step_index].numel()

            current_grid = (
                batch.input_grid
                if step_index == 0
                else batch.intermediate_target_grids[:, step_index - 1]
            )
            reset_logits = model.execute_integer_step(
                current_grid, batch.typed_program_ids[:, step_index]
            )
            reset_prediction = reset_logits.argmax(dim=1)
            reset_exact = reset_prediction.eq(
                batch.intermediate_target_grids[:, step_index]
            ).flatten(1).all(dim=1)
            oracle_reset_correct[step_index] += int(reset_exact.sum().item())

        for transition_index in range(2):
            predicted_grid = predictions[:, transition_index]
            reset_logits = model.execute_integer_step(
                predicted_grid, batch.typed_program_ids[:, transition_index + 1]
            )
            normal_logits = logits[:, transition_index + 1]
            differences = (reset_logits - normal_logits).abs().flatten(1).max(dim=1).values
            maximum_reset_logit_difference = max(
                maximum_reset_logit_difference, float(differences.max().item())
            )
            reset_logit_equal += int(differences.eq(0).sum().item())
            reset_prediction_equal += int(
                reset_logits.argmax(dim=1)
                .eq(normal_logits.argmax(dim=1))
                .flatten(1)
                .all(dim=1)
                .sum()
                .item()
            )
            reset_transitions += batch.input_grid.shape[0]

    example_count = len(length3)
    reset_total = sum(oracle_reset_correct)
    return {
        "num_sealed_length3_examples": example_count,
        "free_rollout_exact_by_step": {
            str(index + 1): _rate(correct, example_count)
            for index, correct in enumerate(free_correct)
        },
        "oracle_reset_exact": {
            "overall": _rate(reset_total, example_count * 3),
            "by_step": {
                str(index + 1): _rate(correct, example_count)
                for index, correct in enumerate(oracle_reset_correct)
            },
        },
        "prediction_reset_equivalence": {
            "transitions": reset_transitions,
            "exact_logit_rate": _rate(reset_logit_equal, reset_transitions),
            "exact_prediction_rate": _rate(reset_prediction_equal, reset_transitions),
            "maximum_absolute_logit_difference": maximum_reset_logit_difference,
        },
        "mean_cell_entropy_by_step": {
            str(index + 1): entropy_sums[index] / confidence_cells[index]
            for index in range(3)
        },
        "mean_winning_probability_by_step": {
            str(index + 1): confidence_sums[index] / confidence_cells[index]
            for index in range(3)
        },
    }
