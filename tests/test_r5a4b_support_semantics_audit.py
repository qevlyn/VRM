from __future__ import annotations

import json

import torch

from vrm.data import ARC_COLOR_COUNT, MAX_GRID_HEIGHT, MAX_GRID_TOKENS, MAX_GRID_WIDTH, ROLE_SUPPORT_INPUT, ROLE_SUPPORT_OUTPUT, ArcEpisode, ArcPair, Grid
from vrm.model import VRM, VRMConfig
from vrm.twin_tasks import r5a4b_support_semantics_audit as audit


def grid(rows: list[list[int]]) -> Grid:
    return Grid.from_raw(rows)


def nonconsecutive_episode(task_id: str = "nonconsecutive") -> ArcEpisode:
    supports = (
        ArcPair(input=grid([[1, 1]]), output=grid([[2, 2]]), pair_index=0),
        ArcPair(input=grid([[3], [3]]), output=grid([[4], [5]]), pair_index=2),
        ArcPair(input=grid([[6, 6], [6, 6]]), output=grid([[7, 8], [8, 7]]), pair_index=3),
    )
    query = ArcPair(input=grid([[1, 0], [0, 0]]), output=grid([[1, 2], [0, 3]]), pair_index=1)
    return ArcEpisode(
        task_id=task_id,
        source_split="unit",
        supports=supports,
        query=query,
        query_pair_index=1,
        source_file="unit",
    )


def tiny_model() -> VRM:
    torch.manual_seed(123)
    model = VRM(VRMConfig(hidden_size=16, num_heads=4, pair_encoder_layers=1, rewrite_layers=1, rewrite_steps=1, rule_slots=2))
    model.eval()
    return model


def sample_record(condition: str, *, same_shape: bool = True, changed_accuracy: float = 1.0) -> dict:
    changed_count = 2
    changed_correct = round(changed_accuracy * changed_count)
    return {
        "condition": condition,
        "episode_index": 0 if same_shape else 1,
        "task_id": "same" if same_shape else "changing",
        "query_pair_index": 0 if same_shape else 1,
        "query_shape": [2, 2],
        "target_shape": [2, 2] if same_shape else [3, 2],
        "predicted_shape": [2, 2],
        "same_shape_regime": same_shape,
        "exact_shape_correct": same_shape,
        "exact_full_grid_correct": False,
        "dense_cell_accuracy": changed_accuracy,
        "dense_color_nll": 1.0,
        "total_loss": 1.0,
        "color_nll": 1.0,
        "target_rectangle_cell_count": 4,
        "target_rectangle_color_correct": changed_correct + 1,
        "changed_cell_count": changed_count,
        "unchanged_cell_count": 2,
        "changed_cell_color_accuracy": changed_accuracy,
        "changed_cell_color_nll": 0.5 if changed_accuracy else 2.0,
        "predicted_change_precision": changed_accuracy,
        "predicted_change_recall": changed_accuracy,
        "predicted_change_f1": changed_accuracy,
        "correct_changed_color_rate": changed_accuracy,
        "unchanged_cell_color_accuracy": 0.5,
        "unchanged_cell_color_nll": 1.5,
        "predicted_change_count": 2,
        "true_positive_predicted_change_count": changed_correct,
        "false_positive_predicted_change_count": 0,
        "changed_cell_color_correct": changed_correct,
        "unchanged_cell_color_correct": 1,
        "target_changed_colors": [2, 3],
        "identity_copy_changed_cell_accuracy": 0.0,
        "identity_copy_unchanged_cell_accuracy": 1.0,
        "dense_argmax_canvas": [[0] * MAX_GRID_WIDTH for _ in range(MAX_GRID_HEIGHT)],
        "dense_argmax_canvas_sha256": "dense",
        "dense_prediction_saved_before_target_loading": True,
        "target_coordinate_scoring_entries": [
            {"row": 0, "col": 1, "target_color": 2, "changed_cell": True},
            {"row": 1, "col": 1, "target_color": 3, "changed_cell": True},
            {"row": 0, "col": 0, "target_color": 1, "changed_cell": False},
            {"row": 1, "col": 0, "target_color": 0, "changed_cell": False},
        ],
        "prediction_digest": "digest",
    }


def condition_records() -> dict[str, list[dict]]:
    return {
        condition: [
            sample_record(condition, same_shape=True, changed_accuracy=1.0),
            sample_record(condition, same_shape=False, changed_accuracy=0.0),
        ]
        for condition in audit.CONDITIONS
    }


def fake_outputs(predictions: dict[tuple[int, int], int] | None = None) -> dict[str, torch.Tensor]:
    predictions = predictions or {(0, 0): 1, (0, 1): 2, (1, 0): 0, (1, 1): 3}
    probs = torch.full((1, MAX_GRID_TOKENS, ARC_COLOR_COUNT), 1e-6)
    for (row, col), color in predictions.items():
        probs[0, row * MAX_GRID_WIDTH + col, color] = 1.0
    height_logits = torch.zeros((1, MAX_GRID_HEIGHT))
    width_logits = torch.zeros((1, MAX_GRID_WIDTH))
    height_logits[0, 1] = 1.0
    width_logits[0, 1] = 1.0
    return {
        "color_probabilities": probs,
        "height_logits": height_logits,
        "width_logits": width_logits,
    }


def test_frozen_input_hash_verification_passes_for_declared_inputs() -> None:
    payload = audit.verify_frozen_inputs()
    assert payload["status"] == "FROZEN_INPUT_INTEGRITY_PASS"
    assert {row["name"] for row in payload["rows"]} >= {
        "initial_checkpoint",
        "selected_checkpoint",
        "sentinel_manifest",
        "model_interface_audit",
        "patch_metrics_source",
        "patch_shape_regime_metrics",
        "patch_dense_prediction_scoring_ledger",
    }
    assert all(row["sha256"] == row["expected_sha256"] for row in payload["rows"])


def test_query_only_dummy_ab_bitwise_invariance_with_disable_support() -> None:
    episodes = [nonconsecutive_episode(f"dummy_{idx}") for idx in range(20)]
    payload = audit.query_only_dummy_invariance(tiny_model(), episodes, torch.device("cpu"))
    assert payload["status"] == "QUERY_ONLY_BYPASS_VALID"
    assert payload["fold_count"] == 20
    assert all(row["dense_color_logits_equal"] for row in payload["rows"])


def test_r5a4a_replay_equivalence_checker_accepts_matching_aggregate(monkeypatch) -> None:
    old = json.loads((audit.R5A4A_DIR / "trained_full_support_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    matching = {
        "fold_count": old["fold_count"],
        "task_count": old["task_count"],
        "exact_full_grid_accuracy": old["exact_full_grid_accuracy"],
        "exact_shape_accuracy": old["exact_output_shape_accuracy"],
        "dense_cell_accuracy_inside_true_target_rectangle": old["cell_accuracy_inside_true_target_rectangle"],
        "dense_color_nll_inside_true_target_rectangle": old["mean_color_negative_log_likelihood_inside_true_target_rectangle"],
        "mean_total_loss": old["mean_total_existing_model_loss"],
    }
    monkeypatch.setattr(audit, "aggregate_condition", lambda _records: matching)
    payload = audit.r5a4a_replay_check({audit.TRAINED_FULL: []})
    assert payload["status"] == "R5A4A_FULL_SUPPORT_REPLAY_PASS"


def test_pairing_rotation_preserves_outputs_and_uses_nonconsecutive_ids() -> None:
    batch = audit.forward_only_batch(nonconsecutive_episode())
    rotated, metadata = audit.rotate_output_pairing(batch)
    assert metadata["visible_pair_ids"] == [0, 2, 3]
    assert metadata["source_to_destination_rotation"] == {"0": 2, "2": 3, "3": 0}

    roles = batch["support_evidence_roles"]
    output_mask = batch["support_evidence_mask"] & (roles == ROLE_SUPPORT_OUTPUT)
    for source, destination in [(0, 2), (2, 3), (3, 0)]:
        source_mask = output_mask & (batch["support_evidence_pair_indices"] == source)
        assert set(rotated["support_evidence_pair_indices"][source_mask].tolist()) == {destination}
        for key in ("support_evidence_colors", "support_evidence_rows", "support_evidence_cols"):
            assert torch.equal(batch[key][source_mask], rotated[key][source_mask])


def test_geometry_scramble_preserves_each_output_coordinates_and_histogram() -> None:
    batch = audit.forward_only_batch(nonconsecutive_episode())
    scrambled, metadata = audit.scramble_output_geometry(batch, task_id="nonconsecutive", query_pair_index=1)
    assert metadata["scramble_rows"]
    for row in metadata["scramble_rows"]:
        assert row["coordinate_sha256_before"] == row["coordinate_sha256_after"]
        assert row["color_histogram_before"] == row["color_histogram_after"]

    non_color_keys = ("support_evidence_rows", "support_evidence_cols", "support_evidence_roles", "support_evidence_pair_indices", "support_evidence_mask")
    for key in non_color_keys:
        assert torch.equal(batch[key], scrambled[key])


def test_support_input_zeroing_preserves_outputs_and_metadata() -> None:
    batch = audit.forward_only_batch(nonconsecutive_episode())
    zeroed, metadata = audit.zero_support_inputs(batch)
    assert metadata["support_input_color_values_zeroed"] > 0
    for key in ("support_evidence_rows", "support_evidence_cols", "support_evidence_roles", "support_evidence_pair_indices", "support_evidence_mask", "query_inputs", "query_input_mask"):
        assert torch.equal(batch[key], zeroed[key])
    output_mask = batch["support_evidence_mask"] & (batch["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
    input_mask = batch["support_evidence_mask"] & (batch["support_evidence_roles"] == ROLE_SUPPORT_INPUT)
    assert torch.equal(batch["support_evidence_colors"][output_mask], zeroed["support_evidence_colors"][output_mask])
    assert torch.equal(zeroed["support_evidence_colors"][input_mask], torch.zeros_like(zeroed["support_evidence_colors"][input_mask]))


def test_target_unavailable_before_forward_and_dense_canvas_saved_before_target_loading() -> None:
    episode = nonconsecutive_episode()
    forward_batch = audit.forward_only_batch(episode)
    assert forward_batch["target_available_before_forward"] is False
    assert "query_targets" not in forward_batch

    target_batch = audit.target_batch_after_forward(episode)
    record = audit.score_outputs(
        condition=audit.TRAINED_FULL,
        episode_index=0,
        episode=episode,
        outputs=fake_outputs(),
        target_batch=target_batch,
        loss=torch.tensor(0.0),
        components={"color_loss": torch.tensor(0.0), "height_loss": torch.tensor(0.0), "width_loss": torch.tensor(0.0)},
        inference_seconds=0.0,
        construction_metadata={},
    )
    assert record["dense_prediction_saved_before_target_loading"] is True
    assert record["changed_cell_count"] == 2
    assert record["unchanged_cell_count"] == 2
    assert record["changed_cell_color_accuracy"] == 1.0


def test_same_shape_and_shape_changing_slices_are_disjoint_and_exhaustive() -> None:
    metrics = audit.shape_regime_summary(condition_records())
    full = metrics["by_condition"][audit.TRAINED_FULL]
    assert full["same_shape_fold_count"] == 1
    assert full["shape_changing_fold_count"] == 1
    assert full["same_shape_fold_count"] + full["shape_changing_fold_count"] == 2


def test_macro_and_pooled_changed_cell_metrics_are_separate() -> None:
    metrics = audit.changed_cell_summary(condition_records())
    full = metrics[audit.TRAINED_FULL]
    assert full["eligible_changed_cell_fold_count"] == 2
    assert full["total_changed_cell_count"] == 4
    assert full["pooled_descriptive_only"]["changed_cell_color_accuracy"] == 0.5
    assert full["macro_per_fold"]["changed_cell_color_accuracy"]["mean"] == 0.5
    assert full["macro_per_fold"]["changed_cell_color_accuracy"]["primary_inferential_statistic"] is True


def test_identity_copy_reference_uses_query_colors_only() -> None:
    reference = audit.identity_copy_reference(condition_records())
    assert reference["reference"] == "identity-copy query colors only"
    assert reference["identity_copy_changed_cell_accuracy_is_zero_by_construction"] is True
    assert reference["changed_cell_accuracy_pooled"] == 0.0


def test_copy_paint_diagnostics_use_unmodified_model_returned_tensors() -> None:
    records = condition_records()
    row = records[audit.TRAINED_FULL][0]
    copy_dist = torch.nn.functional.one_hot(torch.zeros((1, MAX_GRID_TOKENS), dtype=torch.long), num_classes=ARC_COLOR_COUNT).float()
    paint_dist = torch.nn.functional.one_hot(torch.full((1, MAX_GRID_TOKENS), 2, dtype=torch.long), num_classes=ARC_COLOR_COUNT).float()
    outputs = {
        (row["task_id"], row["query_pair_index"]): {
            "copy_distribution": copy_dist,
            "paint_distribution": paint_dist,
            "color_probabilities": fake_outputs()["color_probabilities"],
            "copy_gate": torch.full((1, MAX_GRID_TOKENS, 1), 0.25),
            "copy_entropy": torch.zeros((1, MAX_GRID_TOKENS)),
            "paint_entropy": torch.ones((1, MAX_GRID_TOKENS)),
        }
    }
    diagnostics = audit.copy_paint_diagnostics({audit.TRAINED_FULL: [row]}, outputs)
    assert diagnostics["uses_unmodified_model_returned_tensors"] == [
        "copy_distribution",
        "paint_distribution",
        "copy_gate",
        "color_probabilities",
        "copy_entropy",
        "paint_entropy",
    ]
    assert diagnostics["changed"]["mean_copy_gate"] == 0.25


def test_protected_source_manifest_is_stable_during_read_only_checks() -> None:
    before = audit.protected_source_manifest()
    _ = audit.visible_pair_ids(audit.forward_only_batch(nonconsecutive_episode()))
    after = audit.protected_source_manifest()
    assert before == after


def test_stable_payload_ignores_runtime_and_target_entry_payloads() -> None:
    records = condition_records()
    changed = condition_records()
    changed[audit.TRAINED_FULL][0]["inference_seconds"] = 123.0
    changed[audit.TRAINED_FULL][0]["target_coordinate_scoring_entries"] = [{"different": True}]
    assert audit.deterministic_payload(records) == audit.deterministic_payload(changed)
