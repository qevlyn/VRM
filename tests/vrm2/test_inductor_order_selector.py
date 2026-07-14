from __future__ import annotations

import inspect

import torch

import vrm2.inductor.order_selector as selector_module
from vrm2.inductor.order_selector import (
    CORRECT_ORDERED,
    EXACT_REVERSED,
    ORIENTATION_AMBIGUOUS,
    WRONG_LENGTH,
    active_instruction_multiset,
    choose_orientation_from_scores,
    classify_predicted_program,
    reverse_predicted_program,
    support_only_orientation_selector,
)


NULL = (0, 0, 0, 0)
A = (1, 1, 1, 4)
B = (2, 2, 2, 7)
FORWARD = (A, B, NULL)
REVERSE = (B, A, NULL)
GRID_A = ((1, 0), (0, 0))
GRID_B = ((0, 1), (0, 0))


def test_predicted_program_multiset_comparison_is_order_invariant():
    assert active_instruction_multiset(2, FORWARD) == active_instruction_multiset(2, REVERSE)


def test_ordered_versus_reversed_classification():
    assert classify_predicted_program(2, FORWARD, FORWARD, REVERSE) == CORRECT_ORDERED
    assert classify_predicted_program(2, REVERSE, FORWARD, REVERSE) == EXACT_REVERSED


def test_wrong_length_classification_precedes_instruction_identity():
    assert classify_predicted_program(1, FORWARD, FORWARD, REVERSE) == WRONG_LENGTH


def test_support_only_selector_api_has_no_query_argument():
    parameters = inspect.signature(support_only_orientation_selector).parameters
    assert not any("query" in name for name in parameters)


def test_selector_tie_is_explicitly_ambiguous():
    assert choose_orientation_from_scores((2, 100), (2, 100)) == ORIENTATION_AMBIGUOUS


def test_ground_truth_multiset_selector_chooses_orientation_from_supports(monkeypatch):
    def fake_execute(interpreter, inputs, lengths, programs, batch_size, device):
        assert programs.dtype == torch.long
        outputs = torch.tensor(
            [GRID_A, GRID_A, GRID_A, GRID_B, GRID_B, GRID_B], dtype=torch.long
        )
        return outputs, 0.0

    monkeypatch.setattr(selector_module, "execute_programs_v1", fake_execute)
    result = support_only_orientation_selector(
        torch.nn.Identity(),
        (GRID_A, GRID_A, GRID_A),
        (GRID_B, GRID_B, GRID_B),
        FORWARD,
        8,
        torch.device("cpu"),
    )
    assert result["choice"] == "B"
    assert result["selected_program"] == [list(item) for item in REVERSE]
    assert result["query_data_accessed"] is False


def test_predicted_program_reversal_preserves_fields_and_null_suffix():
    reversed_program, eligible = reverse_predicted_program(2, FORWARD)
    assert eligible
    assert reversed_program == REVERSE
