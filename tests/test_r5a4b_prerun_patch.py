from __future__ import annotations

from pathlib import Path

from vrm.data import ArcEpisode, ArcPair, Grid, collate_episodes
from vrm.twin_tasks.r5a4b_prerun_patch_metrics import (
    CONDITIONS,
    DENSE_SCORING_NOTE,
    INTERPRETATION_SENTENCE,
    LIMITATION_SENTENCE,
    changed_cell_metrics,
    final_report_payload,
    rotate_output_pairing_over_visible_slots,
    shape_regime_metrics,
    target_changed_region_difficulty,
)


def grid(rows: list[list[int]]) -> Grid:
    return Grid.from_raw(rows)


def nonconsecutive_episode() -> ArcEpisode:
    supports = (
        ArcPair(input=grid([[1, 1]]), output=grid([[2, 2]]), pair_index=0),
        ArcPair(input=grid([[3], [3]]), output=grid([[4], [5]]), pair_index=2),
        ArcPair(input=grid([[6, 6], [6, 6]]), output=grid([[7, 8], [8, 7]]), pair_index=3),
    )
    query = ArcPair(input=grid([[1]]), output=grid([[2]]), pair_index=1)
    return ArcEpisode(
        task_id="nonconsecutive",
        source_split="test",
        supports=supports,
        query=query,
        query_pair_index=1,
        source_file="unit",
    )


def test_pairing_rotation_uses_actual_nonconsecutive_visible_pair_ids() -> None:
    batch = collate_episodes([nonconsecutive_episode()])
    rotated, metadata = rotate_output_pairing_over_visible_slots(batch)
    assert metadata["visible_pair_ids"] == [0, 2, 3]
    assert metadata["source_to_destination_rotation"] == {"0": 2, "2": 3, "3": 0}
    assert [row["destination_pair_index"] for row in metadata["rotation_rows"]] == [2, 3, 0]
    assert [row["source_output_shape"] for row in metadata["rotation_rows"]] == [[1, 2], [2, 1], [2, 2]]

    roles = batch["support_evidence_roles"][0]
    original_pair_indices = batch["support_evidence_pair_indices"][0]
    new_pair_indices = rotated["support_evidence_pair_indices"][0]
    colors = batch["support_evidence_colors"][0]
    output_mask = batch["support_evidence_mask"][0] & (roles == 1)
    for source, destination in [(0, 2), (2, 3), (3, 0)]:
        source_mask = output_mask & (original_pair_indices == source)
        assert source_mask.any()
        assert set(new_pair_indices[source_mask].tolist()) == {destination}
        assert colors[source_mask].tolist() == rotated["support_evidence_colors"][0, source_mask].tolist()
        assert batch["support_evidence_rows"][0, source_mask].tolist() == rotated["support_evidence_rows"][0, source_mask].tolist()
        assert batch["support_evidence_cols"][0, source_mask].tolist() == rotated["support_evidence_cols"][0, source_mask].tolist()


def sample_records(condition: str):
    return [
        {
            "condition": condition,
            "task_id": "a",
            "query_pair_index": 0,
            "same_shape_regime": True,
            "exact_output_shape_correct": True,
            "exact_full_grid_correct": False,
            "changed_cell_count": 1,
            "changed_cell_color_correct": 1,
            "changed_cell_color_accuracy": 1.0,
            "changed_cell_color_nll": 0.2,
            "predicted_change_count": 1,
            "true_positive_predicted_change_count": 1,
            "predicted_change_precision": 1.0,
            "predicted_change_recall": 1.0,
            "predicted_change_f1": 1.0,
            "correct_changed_color_rate": 1.0,
            "target_changed_colors": [2],
        },
        {
            "condition": condition,
            "task_id": "b",
            "query_pair_index": 1,
            "same_shape_regime": False,
            "exact_output_shape_correct": False,
            "exact_full_grid_correct": False,
            "changed_cell_count": 9,
            "changed_cell_color_correct": 0,
            "changed_cell_color_accuracy": 0.0,
            "changed_cell_color_nll": 3.0,
            "predicted_change_count": 3,
            "true_positive_predicted_change_count": 3,
            "predicted_change_precision": 1.0,
            "predicted_change_recall": 1 / 3,
            "predicted_change_f1": 0.5,
            "correct_changed_color_rate": 0.0,
            "target_changed_colors": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        },
    ]


def test_shape_regime_metrics_reports_slices_and_bootstrap() -> None:
    records = {
        "FRESH_INIT": sample_records("FRESH_INIT"),
        "TRAINED_FULL_SUPPORT_REPLAY": sample_records("TRAINED_FULL_SUPPORT_REPLAY"),
    }
    for condition in CONDITIONS:
        records.setdefault(condition, sample_records(condition))
    metrics = shape_regime_metrics(records)
    table = metrics["by_condition"]["TRAINED_FULL_SUPPORT_REPLAY"]
    assert table["same_shape_fold_count"] == 1
    assert table["shape_changing_fold_count"] == 1
    assert table["overall_correct_shape_count"] == 1
    assert "trained_minus_fresh_same_shape_shape_accuracy" in metrics["paired_bootstrap_intervals"]
    assert "trained_minus_fresh_shape_changing_shape_accuracy" in metrics["paired_bootstrap_intervals"]


def test_changed_cell_metrics_keeps_pooled_and_macro_separate() -> None:
    records = {condition: sample_records(condition) for condition in CONDITIONS}
    metrics = changed_cell_metrics(records)
    full = metrics["by_condition"]["TRAINED_FULL_SUPPORT_REPLAY"]
    assert full["eligible_changed_cell_fold_count"] == 2
    assert full["total_changed_cell_count"] == 10
    assert full["median_changed_cells_per_eligible_fold"] == 5.0
    assert full["pooled"]["changed_cell_color_accuracy"] == 0.1
    assert full["macro_per_fold"]["changed_cell_color_accuracy"]["mean"] == 0.5
    assert full["pooled"]["descriptive_only"] is True
    assert full["macro_per_fold"]["changed_cell_color_accuracy"]["primary_inferential_statistic"] is True


def test_target_changed_region_difficulty_is_diagnostic_not_baseline() -> None:
    records = {condition: sample_records(condition) for condition in CONDITIONS}
    diagnostics = target_changed_region_difficulty(records)
    assert diagnostics["analysis_name"] == "TARGET_CHANGED_REGION_DIFFICULTY_DIAGNOSTICS"
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["not_a_baseline"] is True
    assert diagnostics["not_used_in_verdict"] is True


def test_report_contains_required_language_and_artifact_paths() -> None:
    report = final_report_payload(
        shape_metrics={"by_condition": {}},
        changed_metrics={"by_condition": {}},
        difficulty={
            "analysis_name": "TARGET_CHANGED_REGION_DIFFICULTY_DIAGNOSTICS",
            "diagnostic_only": True,
            "not_a_baseline": True,
            "not_used_in_verdict": True,
            "pooled": {},
            "macro": {},
        },
        r5a4a_integrity={"unchanged": True},
    )
    assert LIMITATION_SENTENCE in report["limitations"]
    assert INTERPRETATION_SENTENCE in report["interpretation"]
    assert report["dense_prediction_scoring_ledger_path"].endswith("dense_prediction_scoring_ledger.json")
    assert report["conditions"] == list(CONDITIONS)
    assert report["no_training"] is True
    assert report["model_modified"] is False


def test_dense_scoring_note_is_exact_and_non_substitutive() -> None:
    assert "does not repair an incorrect predicted shape" in DENSE_SCORING_NOTE
    assert "does not convert color accuracy into exact-grid success" in DENSE_SCORING_NOTE
