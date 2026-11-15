from __future__ import annotations

import pytest

from vrm2.inductor.order_causality import (
    generator_valid_orientations,
    instructions_distinct,
    interpreter_orientations_faithful,
    query_is_order_sensitive,
    reverse_two_instruction_program,
    subset_manifest_has_predictions,
    support_disambiguation_count,
)


NULL = (0, 0, 0, 0)
A = (1, 1, 1, 4)
B = (2, 2, 2, 7)
PROGRAM = (A, B, NULL)
GRID_A = ((1, 0), (0, 0))
GRID_B = ((0, 1), (0, 0))
GRID_C = ((0, 0), (1, 0))


def test_ground_truth_reversal_swaps_only_active_instruction_positions():
    assert reverse_two_instruction_program(PROGRAM) == (B, A, NULL)


def test_double_reversal_restores_original_program():
    assert reverse_two_instruction_program(reverse_two_instruction_program(PROGRAM)) == PROGRAM


def test_distinct_instruction_filtering():
    assert instructions_distinct(PROGRAM)
    assert not instructions_distinct((A, A, NULL))


def test_generator_valid_reverse_filtering():
    def executor(grid, program):
        if grid == GRID_C:
            raise ValueError("invalid scene state")
        return grid if program[0] == A else GRID_B

    valid, forward, reverse, error = generator_valid_orientations(
        (GRID_A, GRID_B), PROGRAM, (B, A, NULL), executor=executor
    )
    assert valid and len(forward) == len(reverse) == 2 and error is None
    valid, _, _, error = generator_valid_orientations(
        (GRID_A, GRID_C), PROGRAM, (B, A, NULL), executor=executor
    )
    assert not valid and error.startswith("execution_failure:")


def test_query_order_sensitivity_requires_forward_target_match_and_inequality():
    assert query_is_order_sensitive(GRID_A, GRID_B, GRID_A)
    assert not query_is_order_sensitive(GRID_A, GRID_A, GRID_A)
    assert not query_is_order_sensitive(GRID_A, GRID_B, GRID_C)


def test_support_identifiability_counts_differing_reversed_targets():
    assert support_disambiguation_count(
        (GRID_A, GRID_B, GRID_C), (GRID_A, GRID_C, GRID_A)
    ) == 2
    with pytest.raises(ValueError, match="exactly three supports"):
        support_disambiguation_count((GRID_A,), (GRID_B,))


def test_interpreter_faithfulness_requires_both_orientations_on_all_inputs():
    forward = (GRID_A, GRID_B)
    reverse = (GRID_B, GRID_A)
    assert interpreter_orientations_faithful(forward, reverse, forward, reverse)
    assert not interpreter_orientations_faithful(forward, reverse, forward, forward)


def test_subset_manifest_excludes_model_predictions():
    manifest = {
        "model_independent_selection": True,
        "final_records": [{"episode_id": "e1", "forward_program": [list(A)]}],
    }
    assert not subset_manifest_has_predictions(manifest)
    assert subset_manifest_has_predictions({**manifest, "predicted_program": PROGRAM})
    assert subset_manifest_has_predictions({**manifest, "checkpoint_id": "main"})
