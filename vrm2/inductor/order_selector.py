from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.types import Grid
from vrm2.inductor.evaluate_v1 import execute_programs_v1
from vrm2.inductor.order_causality import NULL_INSTRUCTION
from vrm2.inductor.schema_v1 import validate_terminated_program
from vrm2.inductor.types import PaddedTypedProgramIds


CORRECT_ORDERED = "correct_ordered_program"
EXACT_REVERSED = "exact_reversed_program"
CORRECT_UNORDERED_OTHER = "correct_unordered_multiset_other_serialization"
WRONG_MULTISET = "wrong_instruction_multiset"
WRONG_LENGTH = "wrong_length"
ORIENTATION_AMBIGUOUS = "ORIENTATION_AMBIGUOUS"


def normalized_program(program: Sequence[Sequence[int]]) -> PaddedTypedProgramIds:
    result = tuple(tuple(int(value) for value in item) for item in program)
    if len(result) != 3 or any(len(item) != 4 for item in result):
        raise ValueError("programs require three four-field positions")
    return result  # type: ignore[return-value]


def active_instruction_multiset(
    length: int, program: Sequence[Sequence[int]]
) -> tuple[tuple[int, int, int, int], ...]:
    normalized = normalized_program(program)
    return tuple(sorted(normalized[: int(length)]))


def reverse_predicted_program(
    length: int, program: Sequence[Sequence[int]]
) -> tuple[PaddedTypedProgramIds, bool]:
    normalized = normalized_program(program)
    validate_terminated_program(int(length), normalized)
    if int(length) != 2 or normalized[0][0] == 0 or normalized[1][0] == 0:
        return normalized, False
    reversed_program: PaddedTypedProgramIds = (
        normalized[1],
        normalized[0],
        NULL_INSTRUCTION,
    )
    return reversed_program, True


def classify_predicted_program(
    predicted_length: int,
    predicted_program: Sequence[Sequence[int]],
    forward_program: Sequence[Sequence[int]],
    reverse_program: Sequence[Sequence[int]],
) -> str:
    predicted = normalized_program(predicted_program)
    forward = normalized_program(forward_program)
    reverse = normalized_program(reverse_program)
    if int(predicted_length) != 2:
        return WRONG_LENGTH
    if predicted == forward:
        return CORRECT_ORDERED
    if predicted == reverse:
        return EXACT_REVERSED
    if active_instruction_multiset(2, predicted) == active_instruction_multiset(2, forward):
        return CORRECT_UNORDERED_OTHER
    return WRONG_MULTISET


def candidate_score(outputs: Sequence[Grid], targets: Sequence[Grid]) -> tuple[int, int]:
    if len(outputs) != 3 or len(targets) != 3:
        raise ValueError("selector scoring requires exactly three supports")
    exact = sum(output == target for output, target in zip(outputs, targets, strict=True))
    cells = sum(
        predicted == expected
        for output, target in zip(outputs, targets, strict=True)
        for output_row, target_row in zip(output, target, strict=True)
        for predicted, expected in zip(output_row, target_row, strict=True)
    )
    return exact, cells


def choose_orientation_from_scores(
    score_a: tuple[int, int], score_b: tuple[int, int]
) -> str:
    if score_a == score_b:
        return ORIENTATION_AMBIGUOUS
    return "A" if score_a > score_b else "B"


@torch.inference_mode()
def support_only_orientation_selector(
    interpreter: torch.nn.Module,
    support_inputs: Sequence[Grid],
    support_targets: Sequence[Grid],
    program: Sequence[Sequence[int]],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Two-candidate orientation choice. This API intentionally has no query argument."""

    candidate_a = normalized_program(program)
    validate_terminated_program(2, candidate_a)
    candidate_b, reversible = reverse_predicted_program(2, candidate_a)
    if not reversible:
        raise ValueError("orientation selector requires two active instructions")
    inputs = tuple(support_inputs) + tuple(support_inputs)
    lengths = torch.full((6,), 2, dtype=torch.long)
    programs = torch.tensor(
        [candidate_a] * 3 + [candidate_b] * 3, dtype=torch.long
    )
    outputs, seconds = execute_programs_v1(
        interpreter,
        torch.tensor(inputs, dtype=torch.long),
        lengths,
        programs,
        batch_size,
        device,
    )
    grids = [
        tuple(tuple(int(value) for value in row) for row in grid.tolist())
        for grid in outputs
    ]
    score_a = candidate_score(grids[:3], support_targets)
    score_b = candidate_score(grids[3:], support_targets)
    choice = choose_orientation_from_scores(score_a, score_b)
    selected = None
    if choice == "A":
        selected = candidate_a
    elif choice == "B":
        selected = candidate_b
    return {
        "choice": choice,
        "score_a": list(score_a),
        "score_b": list(score_b),
        "candidate_a": [list(item) for item in candidate_a],
        "candidate_b": [list(item) for item in candidate_b],
        "selected_program": None
        if selected is None
        else [list(item) for item in selected],
        "execution_seconds": seconds,
        "hard_integer_program_dtype": str(programs.dtype),
        "query_data_accessed": False,
    }


def build_checkpoint_analysis_record(
    episode: Mapping[str, Any],
    prediction: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    category = classify_predicted_program(
        int(prediction["predicted_program_length"]),
        prediction["predicted_typed_program_ids"],
        episode["forward_program"],
        episode["reverse_program"],
    )
    multiset_correct = category in (
        CORRECT_ORDERED,
        EXACT_REVERSED,
        CORRECT_UNORDERED_OTHER,
    )
    native = bool(execution["native_query_exact"])
    reversed_exact = bool(execution["reversed_query_exact"])
    return {
        "episode_id": episode["episode_id"],
        "checkpoint_role": prediction["checkpoint_role"],
        "classification": category,
        "correct_instruction_multiset": multiset_correct,
        "native_query_exact": native,
        "reversed_query_exact": reversed_exact,
        "output_changed_under_reversal": bool(execution["output_changed_under_reversal"]),
        "reversal_eligible": bool(execution["reversal_eligible"]),
        "reverse_rescue": not native and reversed_exact,
        "reverse_destruction": native and not reversed_exact,
    }


def aggregate_checkpoint_metrics(
    records: Sequence[Mapping[str, Any]], episode_ids: Sequence[str]
) -> dict[str, Any]:
    selected_ids = set(episode_ids)
    selected = [record for record in records if record["episode_id"] in selected_ids]
    if len(selected) != len(selected_ids):
        raise ValueError("analysis records do not cover the requested subset")
    count = len(selected)
    if count == 0:
        return {"episodes": 0}

    def rate(key: str) -> float:
        return sum(bool(record[key]) for record in selected) / count

    category_counts = {
        category: sum(record["classification"] == category for record in selected)
        for category in (
            CORRECT_ORDERED,
            EXACT_REVERSED,
            CORRECT_UNORDERED_OTHER,
            WRONG_MULTISET,
            WRONG_LENGTH,
        )
    }
    multiset_count = sum(category_counts[key] for key in (
        CORRECT_ORDERED, EXACT_REVERSED, CORRECT_UNORDERED_OTHER
    ))
    native_correct_reverse_wrong = sum(
        record["native_query_exact"] and not record["reversed_query_exact"]
        for record in selected
    )
    native_wrong_reverse_correct = sum(
        not record["native_query_exact"] and record["reversed_query_exact"]
        for record in selected
    )
    both_correct = sum(
        record["native_query_exact"] and record["reversed_query_exact"]
        for record in selected
    )
    both_wrong = count - native_correct_reverse_wrong - native_wrong_reverse_correct - both_correct
    ordered_rate = category_counts[CORRECT_ORDERED] / count
    unordered_rate = multiset_count / count
    native_rate = rate("native_query_exact")
    reversed_rate = rate("reversed_query_exact")
    return {
        "episodes": count,
        "category_counts": category_counts,
        "category_rates": {key: value / count for key, value in category_counts.items()},
        "ordered_program_exact": ordered_rate,
        "reversed_program_exact": category_counts[EXACT_REVERSED] / count,
        "unordered_instruction_multiset_exact": unordered_rate,
        "wrong_multiset_rate": category_counts[WRONG_MULTISET] / count,
        "wrong_length_rate": category_counts[WRONG_LENGTH] / count,
        "native_query_exact": native_rate,
        "reversed_query_exact": reversed_rate,
        "native_minus_reversed_exact": native_rate - reversed_rate,
        "output_change_rate_under_reversal": rate("output_changed_under_reversal"),
        "reverse_rescue_rate": rate("reverse_rescue"),
        "reverse_destruction_rate": rate("reverse_destruction"),
        "order_specific_outcomes": {
            "native_correct_reversed_incorrect": native_correct_reverse_wrong,
            "native_incorrect_reversed_correct": native_wrong_reverse_correct,
            "both_correct": both_correct,
            "both_incorrect": both_wrong,
        },
        "correct_multiset_count": multiset_count,
        "correct_orientation_conditional_count": category_counts[CORRECT_ORDERED],
        "reversed_orientation_conditional_count": category_counts[EXACT_REVERSED],
        "correct_orientation_conditional_rate": (
            category_counts[CORRECT_ORDERED] / multiset_count if multiset_count else 0.0
        ),
        "reversed_orientation_conditional_rate": (
            category_counts[EXACT_REVERSED] / multiset_count if multiset_count else 0.0
        ),
        "identity_ceiling": unordered_rate,
        "ordered_realization": ordered_rate,
        "order_loss": unordered_rate - ordered_rate,
    }


def summarize_selector(
    selector_records: Sequence[Mapping[str, Any]],
    analysis_records: Sequence[Mapping[str, Any]],
    episode_ids: Sequence[str],
    *,
    ground_truth: bool = False,
) -> dict[str, Any]:
    ids = set(episode_ids)
    selectors = [record for record in selector_records if record["episode_id"] in ids]
    analyses = {record["episode_id"]: record for record in analysis_records if record["episode_id"] in ids}
    total = len(ids)
    eligible = [record for record in selectors if record["status"] != "INELIGIBLE"]
    non_tie = [record for record in eligible if record["choice"] != ORIENTATION_AMBIGUOUS]
    ambiguous = [record for record in eligible if record["choice"] == ORIENTATION_AMBIGUOUS]
    selected_exact = 0
    orientation_correct = 0
    orientation_denominator = 0
    for record in selectors:
        analysis = analyses.get(record["episode_id"])
        if record["status"] == "INELIGIBLE" or record["choice"] == ORIENTATION_AMBIGUOUS:
            if analysis is not None:
                selected_exact += bool(analysis["native_query_exact"])
            continue
        orientation_denominator += 1
        orientation_correct += bool(record["selected_is_ground_truth_forward"])
        if ground_truth:
            selected_exact += bool(record["selected_is_ground_truth_forward"])
        elif analysis is not None:
            selected_exact += bool(
                analysis["native_query_exact"]
                if record["choice"] == "A"
                else analysis["reversed_query_exact"]
            )
    native_exact = (
        sum(bool(record["native_query_exact"]) for record in analyses.values()) / total
        if analyses
        else 1.0
    )
    query_exact = selected_exact / total if total else 0.0
    return {
        "episodes": total,
        "eligible_count": len(eligible),
        "non_tie_count": len(non_tie),
        "non_tie_coverage": len(non_tie) / total if total else 0.0,
        "orientation_accuracy_denominator": orientation_denominator,
        "orientation_selection_accuracy": (
            orientation_correct / orientation_denominator if orientation_denominator else 0.0
        ),
        "selected_orientation_query_exact": query_exact,
        "query_exact_gain_over_native": None if ground_truth else query_exact - native_exact,
        "ambiguous_count": len(ambiguous),
        "ambiguous_rate": len(ambiguous) / len(eligible) if eligible else 0.0,
        "ineligible_count": total - len(eligible),
    }


def paired_query_counts(
    main_records: Sequence[Mapping[str, Any]],
    shuffled_records: Sequence[Mapping[str, Any]],
    episode_ids: Sequence[str],
) -> dict[str, int]:
    main = {record["episode_id"]: bool(record["native_query_exact"]) for record in main_records}
    shuffled = {
        record["episode_id"]: bool(record["native_query_exact"])
        for record in shuffled_records
    }
    result = {
        "main_correct_shuffled_wrong": 0,
        "main_wrong_shuffled_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for episode_id in episode_ids:
        pair = main[episode_id], shuffled[episode_id]
        if pair == (True, False):
            result["main_correct_shuffled_wrong"] += 1
        elif pair == (False, True):
            result["main_wrong_shuffled_correct"] += 1
        elif pair == (True, True):
            result["both_correct"] += 1
        else:
            result["both_wrong"] += 1
    return result
