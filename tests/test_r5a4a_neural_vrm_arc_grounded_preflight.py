from __future__ import annotations

import inspect
import json

import torch

from vrm.data import ROLE_SUPPORT_INPUT, ROLE_SUPPORT_OUTPUT, collate_episodes, model_inputs_from_batch
from vrm.model import VRM, VRMConfig
from vrm.twin_tasks.r5a4a_arc_manifest import (
    build_r5a4a_splits,
    episodes_for_task_ids,
    source_file_manifest,
)
from vrm.twin_tasks.r5a4a_neural_vrm_data import (
    changed_metadata_episode,
    model_input_digest,
    permute_support_pair_order,
    query_only_masked_batch,
    zero_support_output_colors,
)
from vrm.twin_tasks.r5a4a_neural_vrm_eval import (
    aggregate_records,
    condition_payload,
    evaluate_condition,
    stable_payload_without_runtime,
)
from vrm.twin_tasks.r5a4a_neural_vrm_train import (
    load_state,
    save_state,
    select_best_checkpoint_step,
)
from vrm.twin_tasks.run_r5a4a_neural_vrm_arc_preflight import protected_integrity_status


def tiny_model() -> VRM:
    torch.manual_seed(123)
    config = VRMConfig(hidden_size=16, num_heads=4, pair_encoder_layers=1, rewrite_layers=1, rewrite_steps=1, rule_slots=2)
    model = VRM(config)
    model.eval()
    return model


def sample_episodes():
    splits = build_r5a4a_splits()
    episodes = episodes_for_task_ids(splits.tasks_by_id, splits.sentinel_task_ids, source_split="sentinel")
    return splits, episodes


def test_task_ids_do_not_reach_model_forward() -> None:
    params = set(inspect.signature(VRM.forward).parameters)
    assert "task_id" not in params
    assert "task_ids" not in params
    _splits, episodes = sample_episodes()
    batch = collate_episodes([episodes[0]])
    inputs = model_inputs_from_batch(batch)
    assert "task_ids" in batch
    assert "query_targets" in batch
    assert "task_ids" not in inputs
    assert "query_targets" not in inputs
    assert "height_targets" not in inputs
    assert "width_targets" not in inputs


def test_changed_external_metadata_gives_identical_tensors_and_outputs() -> None:
    _splits, episodes = sample_episodes()
    episode = episodes[0]
    changed = changed_metadata_episode(episode, suffix="_different_label")
    batch_a = collate_episodes([episode])
    batch_b = collate_episodes([changed])
    assert batch_a["task_ids"] != batch_b["task_ids"]
    assert model_input_digest(model_inputs_from_batch(batch_a)) == model_input_digest(model_inputs_from_batch(batch_b))
    model = tiny_model()
    with torch.no_grad():
        out_a = model(**model_inputs_from_batch(batch_a))
        out_b = model(**model_inputs_from_batch(batch_b))
    for key, value in out_a.items():
        if torch.is_tensor(value):
            assert torch.equal(value, out_b[key]), key


def test_sentinel_manifest_deterministic_and_pools_disjoint() -> None:
    first = build_r5a4a_splits()
    second = build_r5a4a_splits()
    assert first.sentinel_task_ids == second.sentinel_task_ids
    assert first.validation_task_ids == second.validation_task_ids
    assert first.training_task_ids == second.training_task_ids
    assert len(first.sentinel_task_ids) == 48
    assert len(first.validation_task_ids) == 64
    assert set(first.sentinel_task_ids).isdisjoint(first.validation_task_ids)
    assert set(first.sentinel_task_ids).isdisjoint(first.training_task_ids)
    assert set(first.validation_task_ids).isdisjoint(first.training_task_ids)


def test_no_synthetic_task_generator_is_called() -> None:
    manifest = source_file_manifest(include_artifacts=False)
    r5a4a_paths = [row["path"] for row in manifest["files"] if "r5a4a" in row["path"]]
    assert r5a4a_paths
    for relpath in r5a4a_paths:
        if relpath.startswith("tests/"):
            continue
        text = (source_file_manifest.__globals__["PROJECT_ROOT"] / relpath).read_text(encoding="utf-8")
        assert "from vrm.twin_tasks.generator" not in text
        assert "import vrm.twin_tasks.generator" not in text
        assert "generate_twin" not in text
        assert "synthetic generated task" not in text.lower()


def test_support_output_zeroing_preserves_non_color_fields() -> None:
    _splits, episodes = sample_episodes()
    batch = collate_episodes([episodes[0]])
    zeroed = zero_support_output_colors(batch)
    for key in [
        "support_evidence_rows",
        "support_evidence_cols",
        "support_evidence_roles",
        "support_evidence_pair_indices",
        "support_evidence_mask",
        "query_inputs",
        "query_input_mask",
    ]:
        assert torch.equal(batch[key], zeroed[key]), key
    input_mask = batch["support_evidence_mask"] & (batch["support_evidence_roles"] == ROLE_SUPPORT_INPUT)
    output_mask = batch["support_evidence_mask"] & (batch["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
    assert torch.equal(batch["support_evidence_colors"][input_mask], zeroed["support_evidence_colors"][input_mask])
    assert torch.equal(zeroed["support_evidence_colors"][output_mask], torch.zeros_like(zeroed["support_evidence_colors"][output_mask]))


def test_query_only_mask_does_not_leak_support_values() -> None:
    _splits, episodes = sample_episodes()
    batch = collate_episodes([episodes[0]])
    masked = query_only_masked_batch(batch)
    assert not bool(masked["support_evidence_mask"].any().item())
    assert not bool(masked["support_mask"].any().item())
    assert torch.equal(batch["support_evidence_colors"], masked["support_evidence_colors"])
    assert torch.equal(batch["support_evidence_rows"], masked["support_evidence_rows"])
    assert torch.equal(batch["query_inputs"], masked["query_inputs"])


def test_support_pair_permutation_keeps_inputs_matched_to_outputs() -> None:
    _splits, episodes = sample_episodes()
    episode = next(ep for ep in episodes if len(ep.supports) >= 2)
    permuted, metadata = permute_support_pair_order(episode)
    permutation = metadata["support_pair_permutation_old_index_by_new_slot"]
    assert metadata["non_identity"]
    for new_slot, old_index in enumerate(permutation):
        assert permuted.supports[new_slot].pair_index == new_slot
        assert permuted.supports[new_slot].input == episode.supports[old_index].input
        assert permuted.supports[new_slot].output == episode.supports[old_index].output


def test_fresh_init_checkpoint_is_saved_before_optimizer_step(tmp_path) -> None:
    model = tiny_model()
    path = tmp_path / "initial_state_before_any_optimizer_step.pt"
    save_state(path, model=model, optimizer=None, config=model.config, step=0, extra={"saved_before_any_optimizer_step": True})
    loaded, payload = load_state(path, torch.device("cpu"))
    assert payload["step"] == 0
    assert payload["optimizer_state_dict"] is None
    assert payload["extra"]["saved_before_any_optimizer_step"] is True
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_trained_checkpoint_selection_uses_validation_only() -> None:
    curve = [
        {"step": 1000, "mean_total_loss": 2.0, "sentinel_loss": 0.1},
        {"step": 2000, "mean_total_loss": 1.5, "sentinel_loss": 9.0},
        {"step": 3000, "mean_total_loss": 1.7, "sentinel_loss": 0.0},
    ]
    assert select_best_checkpoint_step(curve) == 2000


def test_reported_metric_denominators_are_correct() -> None:
    records = [
        {
            "condition": "X",
            "task_id": "a",
            "query_pair_index": 0,
            "exact_full_grid_correct": True,
            "exact_output_shape_correct": False,
            "target_rectangle_color_correct": 1,
            "target_rectangle_cell_count": 2,
            "target_rectangle_cell_accuracy": 0.5,
            "color_nll": 2.0,
            "height_nll": 1.0,
            "width_nll": 3.0,
            "total_loss": 3.0,
            "inference_seconds": 0.1,
        },
        {
            "condition": "X",
            "task_id": "b",
            "query_pair_index": 1,
            "exact_full_grid_correct": False,
            "exact_output_shape_correct": True,
            "target_rectangle_color_correct": 3,
            "target_rectangle_cell_count": 6,
            "target_rectangle_cell_accuracy": 0.5,
            "color_nll": 4.0,
            "height_nll": 3.0,
            "width_nll": 5.0,
            "total_loss": 5.0,
            "inference_seconds": 0.3,
        },
    ]
    agg = aggregate_records("X", records)
    assert agg["fold_count"] == 2
    assert agg["task_count"] == 2
    assert agg["cell_accuracy_inside_true_target_rectangle"] == 4 / 8
    assert agg["exact_full_grid_accuracy"] == 0.5
    assert agg["exact_output_shape_accuracy"] == 0.5
    assert agg["mean_total_existing_model_loss"] == 4.0


def test_rerunning_evaluation_from_same_model_is_stable_excluding_runtime() -> None:
    _splits, episodes = sample_episodes()
    model = tiny_model()
    subset = [episodes[0]]
    first = evaluate_condition(model=model, episodes=subset, condition="TRAINED_FULL_SUPPORT", device=torch.device("cpu"), shape_loss_weight=0.5)
    second = evaluate_condition(model=model, episodes=subset, condition="TRAINED_FULL_SUPPORT", device=torch.device("cpu"), shape_loss_weight=0.5)
    assert stable_payload_without_runtime(condition_payload(*first)) == stable_payload_without_runtime(condition_payload(*second))


def test_r5a2_r5a3_protected_sources_and_artifacts_noop_integrity() -> None:
    before = source_file_manifest()
    after = source_file_manifest()
    status = protected_integrity_status(before, after)
    assert status["protected_r5a2_r5a3_unchanged"], json.dumps(status, indent=2)
