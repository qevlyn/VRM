from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import torch

from vrm.twin_tasks.composition_r4a0r_data import all_query_types, tensors_from_episodes
from vrm.twin_tasks.composition_r4a1q_reconcile import (
    APPROVED_CHECKPOINTS,
    R4A1Q_DATA_OUT,
    balanced_r4a0r_sealed_sample,
    entropy_from_attention,
    load_frozen_model,
    parameter_hashes,
    raw_attention_fixture_metrics,
    raw_pointer_metrics,
    reproduce_old_metrics,
)


def test_cuda_required_no_cpu_fallback():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    import vrm.twin_tasks.composition_r4a1q_reconcile as reconcile

    source = inspect.getsource(reconcile.cuda_environment_report)
    assert "CUDA_REQUIRED_NOT_AVAILABLE" in source
    assert '"cuda"' in source


def test_no_optimizer_is_constructed_and_no_backward_call_occurs():
    import vrm.twin_tasks.run_composition_r4a1q as runner
    import vrm.twin_tasks.composition_r4a1q_reconcile as reconcile

    source = inspect.getsource(runner) + inspect.getsource(reconcile)
    forbidden = ["torch.optim", ".backward(", "loss.backward", ".step(", "torch.save("]
    assert all(item not in source for item in forbidden)


def test_loaded_parameters_remain_byte_identical_after_raw_eval():
    device = torch.device("cuda")
    model, _ = load_frozen_model(20260628, device)
    sample = balanced_r4a0r_sealed_sample(20260628)[:8]
    before = parameter_hashes(model)
    raw_pointer_metrics(model, sample, device, batch_size=4)
    after = parameter_hashes(model)
    assert before == after
    assert all(not row["requires_grad"] for row in after.values())


def test_checkpoint_files_exist_and_are_limited_to_approved_three():
    assert set(APPROVED_CHECKPOINTS) == {20260628, 20260629, 20260630}
    assert all(path.exists() for path in APPROVED_CHECKPOINTS.values())
    assert all(path.name.endswith("_best_dev.pt") for path in APPROVED_CHECKPOINTS.values())


def test_raw_attention_rows_sum_to_one():
    device = torch.device("cuda")
    model, _ = load_frozen_model(20260628, device)
    sample = balanced_r4a0r_sealed_sample(20260628)[:4]
    batch = tensors_from_episodes(sample, device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    pointer = out["pointer_attention"].reshape(len(sample), 49, 49)
    assert torch.allclose(pointer.sum(dim=-1), torch.ones(len(sample), 49, device=device), atol=1e-6)


def test_entropy_is_computed_directly_from_raw_normalized_attention():
    attention = torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32)
    expected = -(attention * torch.log(attention)).sum(dim=-1)
    assert torch.allclose(entropy_from_attention(attention), expected)


def test_one_hot_raw_mass_fixture_has_max_one_and_zero_entropy():
    attention = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    oracle = torch.tensor([1], dtype=torch.long)
    metrics = raw_attention_fixture_metrics(attention, oracle)
    assert metrics["mean_raw_oracle_coordinate_mass"] == 1.0
    assert metrics["mean_raw_max_probability"] == 1.0
    assert metrics["mean_raw_entropy"] == 0.0


def test_exact_coordinate_scoring_does_not_accept_same_color_wrong_coordinate():
    import vrm.twin_tasks.composition_r4a1q_reconcile as reconcile

    source = inspect.getsource(reconcile.raw_pointer_metrics)
    assert "correct = argmax_index == oracle_index" in source
    assert "same_color_wrong_cell" in source


def test_old_metric_functions_are_called_without_source_edits():
    import vrm.twin_tasks.run_composition_r4a0r as old_runner

    assert reproduce_old_metrics.__globals__["r4a0r_evaluate_model"] is old_runner.evaluate_model
    assert inspect.getsourcefile(old_runner.evaluate_model).endswith("run_composition_r4a0r.py")


def test_comparison_records_use_identical_episode_ids_and_foreground_mask_when_artifact_exists():
    manifest_path = R4A1Q_DATA_OUT / "evaluation_sample_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for seed, row in manifest["r4a0r_v2_sealed_by_seed"].items():
        assert row["episode_count"] == 2007
        assert set(row["record_count_per_sequence_class"].values()) == {223}
        assert len(row["episode_ids"]) == len(set(row["episode_ids"]))
    assert manifest["foreground_mask_definition"].startswith("target grid cell value != 0")


def test_metadata_changes_do_not_alter_output_during_wrapper_evaluation():
    device = torch.device("cuda")
    model, _ = load_frozen_model(20260628, device)
    episode = balanced_r4a0r_sealed_sample(20260628)[0]
    mutated = replace(
        episode,
        episode_id="mutated_metadata_only",
        split="MUTATED",
        query_type=all_query_types()[-1],
        counterfactual_group_id="mutated_group",
    )
    left = tensors_from_episodes([episode], device)
    right = tensors_from_episodes([mutated], device)
    assert torch.equal(left["query_grids"], right["query_grids"])
    assert torch.equal(left["support_input_grids"], right["support_input_grids"])
    assert torch.equal(left["support_output_grids"], right["support_output_grids"])
    with torch.no_grad():
        out_a = model(left["query_grids"], left["support_input_grids"], left["support_output_grids"])
        out_b = model(right["query_grids"], right["support_input_grids"], right["support_output_grids"])
    assert torch.allclose(out_a["output_probs"], out_b["output_probs"], atol=1e-7)
    assert torch.allclose(out_a["pointer_attention"], out_b["pointer_attention"], atol=1e-7)
