from __future__ import annotations

import collections
from typing import Any, Sequence

import torch

from vrm2.inductor.evaluate import execute_programs, predict_programs, support_tensors, target_tensors
from vrm2.inductor.model import SupervisedTypedProgramInductorV0
from vrm2.inductor.types import QueryEpisode, SupportEpisode


def _derangement(target_programs: torch.Tensor) -> torch.Tensor:
    count = len(target_programs)
    for shift in range(1, count):
        indices = (torch.arange(count) + shift) % count
        if target_programs[indices].ne(target_programs).flatten(1).any(dim=1).all():
            return indices
    indices = []
    for index in range(count):
        replacement = next(
            other for other in range(count)
            if not torch.equal(target_programs[index], target_programs[other])
        )
        indices.append(replacement)
    return torch.tensor(indices)


def _report(
    predictions: torch.Tensor,
    lengths: torch.Tensor,
    programs: torch.Tensor,
    query_targets: torch.Tensor,
    correct_predictions: torch.Tensor,
    correct_lengths: torch.Tensor,
    correct_programs: torch.Tensor,
) -> dict[str, Any]:
    exact = predictions.eq(query_targets).flatten(1).all(dim=1)
    output_change = predictions.ne(correct_predictions).flatten(1).any(dim=1)
    program_change = lengths.ne(correct_lengths) | programs.ne(correct_programs).flatten(1).any(dim=1)
    return {
        "eligible": len(predictions),
        "query_exact": float(exact.float().mean().item()),
        "output_change_rate": float(output_change.float().mean().item()),
        "predicted_program_change_rate": float(program_change.float().mean().item()),
    }


@torch.inference_mode()
def run_inductor_controls(
    model: SupervisedTypedProgramInductorV0,
    interpreter: torch.nn.Module,
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    training_supports: Sequence[SupportEpisode],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    target_lengths, target_programs, query_inputs, query_targets = target_tensors(supports, queries)
    correct_lengths, correct_programs, _, _ = predict_programs(model, supports, batch_size, device)
    correct_predictions, _ = execute_programs(
        interpreter, query_inputs, correct_lengths, correct_programs, batch_size, device
    )
    correct = _report(
        correct_predictions, correct_lengths, correct_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )

    permuted_lengths, permuted_programs, _, _ = predict_programs(
        model, supports, batch_size, device, support_permutation=(2, 0, 1)
    )
    permuted_predictions, _ = execute_programs(
        interpreter, query_inputs, permuted_lengths, permuted_programs, batch_size, device
    )
    support_permutation = _report(
        permuted_predictions, permuted_lengths, permuted_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )
    support_permutation["predicted_program_identity_rate"] = float(
        (permuted_lengths.eq(correct_lengths) & permuted_programs.eq(correct_programs).flatten(1).all(dim=1))
        .float().mean().item()
    )
    support_permutation["query_prediction_identity_rate"] = float(
        permuted_predictions.eq(correct_predictions).flatten(1).all(dim=1).float().mean().item()
    )

    support_inputs, support_outputs = support_tensors(supports, torch.device("cpu"))
    swap_indices = _derangement(target_programs)
    swap_lengths, swap_programs, _, _ = predict_programs(
        model, supports, batch_size, device,
        replacement_inputs=support_inputs[swap_indices],
        replacement_outputs=support_outputs[swap_indices],
    )
    swap_predictions, _ = execute_programs(
        interpreter, query_inputs, swap_lengths, swap_programs, batch_size, device
    )
    support_swap = _report(
        swap_predictions, swap_lengths, swap_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )

    shuffled_outputs = support_outputs.roll(shifts=1, dims=1)
    eligible_mask = shuffled_outputs.ne(support_outputs).flatten(1).any(dim=1)
    eligible_indices = eligible_mask.nonzero(as_tuple=False).flatten()
    eligible_supports = [supports[index] for index in eligible_indices.tolist()]
    shuffle_lengths, shuffle_programs, _, _ = predict_programs(
        model, eligible_supports, batch_size, device,
        replacement_inputs=support_inputs[eligible_indices],
        replacement_outputs=shuffled_outputs[eligible_indices],
    )
    shuffle_predictions, _ = execute_programs(
        interpreter, query_inputs[eligible_indices], shuffle_lengths, shuffle_programs, batch_size, device
    )
    output_shuffle = _report(
        shuffle_predictions, shuffle_lengths, shuffle_programs, query_targets[eligible_indices],
        correct_predictions[eligible_indices], correct_lengths[eligible_indices], correct_programs[eligible_indices]
    )

    null_inputs = torch.zeros_like(support_inputs)
    null_outputs = torch.zeros_like(support_outputs)
    null_lengths, null_programs, _, _ = predict_programs(
        model, supports, batch_size, device,
        replacement_inputs=null_inputs, replacement_outputs=null_outputs
    )
    null_predictions, _ = execute_programs(
        interpreter, query_inputs, null_lengths, null_programs, batch_size, device
    )
    null_support = _report(
        null_predictions, null_lengths, null_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )

    frequencies = collections.Counter(
        (item.target_program_length, item.target_typed_program_ids) for item in training_supports
    )
    prior_length, prior_program = min(
        frequencies,
        key=lambda item: (-frequencies[item], item),
    )
    prior_lengths = torch.full_like(correct_lengths, prior_length)
    prior_programs = torch.tensor([prior_program] * len(supports), dtype=torch.long)
    prior_predictions, _ = execute_programs(
        interpreter, query_inputs, prior_lengths, prior_programs, batch_size, device
    )
    most_frequent_prior = _report(
        prior_predictions, prior_lengths, prior_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )
    most_frequent_prior["training_frequency"] = frequencies[(prior_length, prior_program)]

    program_swap_indices = _derangement(target_programs)
    swapped_induced_lengths = correct_lengths[program_swap_indices]
    swapped_induced_programs = correct_programs[program_swap_indices]
    program_swap_predictions, _ = execute_programs(
        interpreter, query_inputs, swapped_induced_lengths, swapped_induced_programs, batch_size, device
    )
    program_swap = _report(
        program_swap_predictions, swapped_induced_lengths, swapped_induced_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )

    oracle_predictions, _ = execute_programs(
        interpreter, query_inputs, target_lengths, target_programs, batch_size, device
    )
    oracle_program = _report(
        oracle_predictions, target_lengths, target_programs, query_targets,
        correct_predictions, correct_lengths, correct_programs
    )
    for report in (
        support_permutation, support_swap, output_shuffle, null_support,
        most_frequent_prior, program_swap, oracle_program
    ):
        report["exact_drop_from_correct"] = correct["query_exact"] - report["query_exact"]
    correct["exact_drop_from_correct"] = 0.0
    return {
        "correct": correct,
        "support_permutation": support_permutation,
        "support_swap": support_swap,
        "output_shuffle": output_shuffle,
        "null_support": null_support,
        "most_frequent_prior": most_frequent_prior,
        "program_swap": program_swap,
        "oracle_program": oracle_program,
    }
