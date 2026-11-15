from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.data import (
    ARC_COLOR_COUNT,
    DEFAULT_ARC_DATA_ROOT,
    LOCAL_ARC_DATA_ROOT,
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    PAD_ID,
    ROLE_SUPPORT_INPUT,
    ROLE_SUPPORT_OUTPUT,
    Grid,
    build_dataset,
    collate_episodes,
    discover_arc_training_files,
    grid_shape_mask,
    model_inputs_from_batch,
    pad_grid_to_tokens,
)
from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM, VRMConfig
from vrm.train import (
    CHECKPOINTS_DIR,
    DeterministicCyclingSampler,
    V1_1_BEST_CHECKPOINT,
    V1_1_ARTIFACTS_DIR,
    V1_1_SOURCE_ZIP,
    V1_1_SUPPORT_ABLATION_DIR,
    V1_1_TASK_MANIFEST,
    V1_ARTIFACTS_DIR,
    V1_TASK_MANIFEST,
    deterministic_color_permutation,
    episode_for_condition,
    evaluate_model_condition,
    hash_file,
    identity_copy_baseline_prediction,
    initialize_fresh_v1_1_model,
    paired_bootstrap_ci,
    prediction_metrics_for_episode,
    run_v1_1_support_ablation,
    select_v1_1_capacity50_episode_sets,
    support_shape_copy_baseline_prediction,
    transformation_cell_mask,
)

SEED = 20260627


def _dataset():
    return build_dataset(seed=SEED, manifest_path=None)


def _small_model() -> VRM:
    return VRM(
        VRMConfig(
            hidden_size=32,
            num_heads=4,
            pair_encoder_layers=1,
            rewrite_layers=1,
            rewrite_steps=1,
            rule_slots=8,
        )
    )


def _batch():
    data = _dataset()
    return data, collate_episodes([data["train_episodes"][0]])


def test_local_data_root_discovery_inside_vrm() -> None:
    assert DEFAULT_ARC_DATA_ROOT == LOCAL_ARC_DATA_ROOT
    sources = discover_arc_training_files()
    assert Path(sources["data_root"]) == LOCAL_ARC_DATA_ROOT
    assert Path(sources["training_challenges"]).is_file()
    assert str(Path(sources["training_challenges"])).startswith(str(LOCAL_ARC_DATA_ROOT))


def test_v1_1_reuses_prior_v1_50_train_and_50_unseen_ids_when_available() -> None:
    assert V1_TASK_MANIFEST.exists()
    prior = json.loads(V1_TASK_MANIFEST.read_text(encoding="utf-8"))
    data = _dataset()
    selection = select_v1_1_capacity50_episode_sets(data, SEED)
    assert selection["selection_status"] == "reused_prior_v1_manifest"
    assert selection["train_task_ids"] == sorted(prior["train_50_task_ids"])
    assert selection["unseen_task_ids"] == sorted(prior["unseen_50_task_ids"])


def test_train_and_unseen_task_ids_do_not_overlap() -> None:
    data = _dataset()
    selection = select_v1_1_capacity50_episode_sets(data, SEED)
    train_ids = set(selection["train_task_ids"])
    unseen_ids = set(selection["unseen_task_ids"])
    assert len(train_ids) == 50
    assert len(unseen_ids) == 50
    assert train_ids.isdisjoint(unseen_ids)


def test_leave_one_out_episodes_remain_correct() -> None:
    data = _dataset()
    task_by_id = {task.task_id: task for task in data["tasks"]}
    for episode in data["episodes"]:
        actual = {pair.pair_index for pair in episode.supports}
        expected = {
            pair.pair_index
            for pair in task_by_id[episode.task_id].train
            if pair.pair_index != episode.query_pair_index
        }
        assert actual == expected


def test_task_ids_never_enter_model_tensors_or_forward() -> None:
    _, batch = _batch()
    inputs = model_inputs_from_batch(batch)
    assert "task_ids" not in inputs
    assert all(torch.is_tensor(value) for value in inputs.values())
    forbidden = {"task_id", "task_ids", "puzzle_id", "puzzle_identifier", "puzzle_identifiers"}
    assert forbidden.isdisjoint(inspect.signature(VRM.forward).parameters)


def test_query_targets_never_enter_model_forward() -> None:
    _, batch = _batch()
    inputs = model_inputs_from_batch(batch)
    assert "query_targets" not in inputs
    assert "query_target" not in inputs
    assert "height_targets" not in inputs
    assert "width_targets" not in inputs
    forbidden = {"query_target", "query_targets", "labels", "height_targets", "width_targets"}
    assert forbidden.isdisjoint(inspect.signature(VRM.forward).parameters)


def test_query_input_mask_true_exactly_inside_original_query_rectangle() -> None:
    data = _dataset()
    episode = data["train_episodes"][0]
    batch = collate_episodes([episode])
    expected = grid_shape_mask(episode.query.input.height, episode.query.input.width)
    assert torch.equal(batch["query_input_mask"][0], expected)
    assert int(batch["query_input_mask"][0].sum()) == episode.query.input.height * episode.query.input.width
    assert batch["query_input_mask"].dtype == torch.bool


def test_query_cross_attention_receives_query_key_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    _, batch = _batch()
    model = _small_model()
    seen: dict[str, torch.Tensor] = {}
    original = model.rewrite_layers[0].query_attn.forward

    def wrapped(query, context=None, key_mask=None):
        seen["mask"] = key_mask
        return original(query, context, key_mask)

    monkeypatch.setattr(model.rewrite_layers[0].query_attn, "forward", wrapped)
    model.eval()
    with torch.no_grad():
        model(**model_inputs_from_batch(batch))
    assert "mask" in seen
    assert torch.equal(seen["mask"], batch["query_input_mask"])


def test_copy_attention_assigns_zero_probability_to_padded_query_positions() -> None:
    _, batch = _batch()
    model = _small_model()
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs_from_batch(batch))
    padded = ~batch["query_input_mask"][0]
    assert bool(padded.any())
    assert float(outputs["copy_attention"][0, :, padded].abs().max()) == pytest.approx(0.0, abs=1e-7)


def test_changing_only_padded_query_values_does_not_change_logits_or_decoded_output() -> None:
    _, batch = _batch()
    altered = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    padded = ~altered["query_input_mask"]
    altered["query_inputs"][padded] = 7
    torch.manual_seed(123)
    model = _small_model()
    model.eval()
    with torch.no_grad():
        original = model(**model_inputs_from_batch(batch))
        changed = model(**model_inputs_from_batch(altered))
    assert torch.allclose(original["color_logits"], changed["color_logits"], atol=0.0, rtol=0.0)
    assert torch.equal(original["height_logits"].argmax(dim=-1), changed["height_logits"].argmax(dim=-1))
    assert torch.equal(original["width_logits"].argmax(dim=-1), changed["width_logits"].argmax(dim=-1))


def test_output_head_shapes_are_factorized() -> None:
    data = _dataset()
    batch = collate_episodes(data["train_episodes"][:2])
    model = _small_model()
    outputs = model(**model_inputs_from_batch(batch))
    assert outputs["color_logits"].shape == (2, MAX_GRID_TOKENS, ARC_COLOR_COUNT)
    assert outputs["height_logits"].shape == (2, MAX_GRID_HEIGHT)
    assert outputs["width_logits"].shape == (2, MAX_GRID_HEIGHT)
    assert model.config.output_color_count == 10


def test_color_loss_ignores_positions_outside_true_target_rectangle() -> None:
    _, batch = _batch()
    torch.manual_seed(1)
    model = _small_model()
    outputs = model(**model_inputs_from_batch(batch))
    loss_a, components_a = factorized_shape_color_loss(outputs, batch)
    altered = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    altered["query_targets"][~altered["query_target_mask"]] = 3
    loss_b, components_b = factorized_shape_color_loss(outputs, altered)
    assert torch.allclose(components_a["color_loss"], components_b["color_loss"])
    assert torch.allclose(loss_a, loss_b)


def test_height_and_width_targets_use_minus_one_encoding() -> None:
    _, batch = _batch()
    expected_h = int(batch["query_output_heights"][0]) - 1
    expected_w = int(batch["query_output_widths"][0]) - 1
    assert int(batch["height_targets"][0]) == expected_h
    assert int(batch["width_targets"][0]) == expected_w


def test_final_output_probability_distribution_is_finite_and_sums_to_one() -> None:
    _, batch = _batch()
    model = _small_model()
    outputs = model(**model_inputs_from_batch(batch))
    probs = outputs["color_probabilities"]
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5, rtol=1e-5)


def test_one_real_arc_batch_gives_finite_factorized_losses() -> None:
    _, batch = _batch()
    torch.manual_seed(2)
    model = _small_model()
    outputs = model(**model_inputs_from_batch(batch))
    total, components = factorized_shape_color_loss(outputs, batch)
    assert torch.isfinite(total)
    assert torch.isfinite(components["color_loss"])
    assert torch.isfinite(components["height_loss"])
    assert torch.isfinite(components["width_loss"])


def test_support_ablation_mode_source_performs_no_training_update() -> None:
    source = inspect.getsource(run_v1_1_support_ablation)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "clip_grad" not in source
    assert "save_checkpoint(" not in source


def test_fresh_v1_1_initialization_loads_no_checkpoint_and_state_has_no_lookup_terms() -> None:
    model, report = initialize_fresh_v1_1_model(_small_model().config, torch.device("cpu"))
    assert report["fresh_random_initialization"] is True
    assert report["loaded_checkpoint"] is False
    banned = ["puzzle_emb", "puzzle_identifier", "sparse_embedding", "task_id"]
    assert all(not any(term in key.lower() for term in banned) for key in model.state_dict())


def test_unseen_episodes_are_never_sampled_for_optimizer_updates() -> None:
    data = _dataset()
    selection = select_v1_1_capacity50_episode_sets(data, SEED)
    unseen_ids = set(selection["unseen_task_ids"])
    sampler = DeterministicCyclingSampler(len(selection["train_episodes"]), SEED + 90_000)
    sampled = []
    for _ in range(len(selection["train_episodes"]) * 2 + 5):
        sampled.extend(sampler.next_indices(1))
    assert all(selection["train_episodes"][idx].task_id not in unseen_ids for idx in sampled)


def test_old_v1_artifacts_and_checkpoints_are_not_v1_1_targets() -> None:
    assert V1_ARTIFACTS_DIR.exists()
    assert (V1_ARTIFACTS_DIR / "vrm_v1_task_manifest.json").exists()
    assert (CHECKPOINTS_DIR / "vrm_v1_capacity50_step1500.pt").exists()
    assert V1_1_ARTIFACTS_DIR != V1_ARTIFACTS_DIR
    assert not str(V1_1_ARTIFACTS_DIR).startswith(str(V1_ARTIFACTS_DIR) + "/")


def test_support_evidence_bank_keeps_valid_positions_roles_and_pair_indices() -> None:
    data = _dataset()
    episode = data["train_episodes"][0]
    batch = collate_episodes([episode])
    expected_count = sum(
        pair.input.height * pair.input.width + pair.output.height * pair.output.width
        for pair in episode.supports
    )
    assert int(batch["support_evidence_counts"][0]) == expected_count
    assert int(batch["support_evidence_mask"][0].sum()) == expected_count
    first_support = episode.supports[0]
    assert int(batch["support_evidence_colors"][0, 0]) == first_support.input.cells[0][0]
    assert int(batch["support_evidence_rows"][0, 0]) == 0
    assert int(batch["support_evidence_cols"][0, 0]) == 0
    assert int(batch["support_evidence_roles"][0, 0]) == ROLE_SUPPORT_INPUT
    assert int(batch["support_evidence_pair_indices"][0, 0]) == first_support.pair_index
    assert ROLE_SUPPORT_OUTPUT in set(batch["support_evidence_roles"][0, :expected_count].tolist())


def test_pad_id_remains_distinct_from_color_zero() -> None:
    grid = Grid.from_raw([[0, 1], [2, 0]])
    tokens = pad_grid_to_tokens(grid).view(30, 30)
    assert int(tokens[0, 0]) == 0
    assert int(tokens[1, 1]) == 0
    assert int(tokens[0, 2]) == PAD_ID
    assert int(tokens[2, 0]) == PAD_ID


def _episode_with_multiple_supports():
    data = _dataset()
    return next(episode for episode in data["train_episodes"] if len(episode.supports) >= 2)


def test_normal_ablation_predictions_match_direct_forward() -> None:
    data = _dataset()
    episode = data["train_episodes"][0]
    model = _small_model()
    model.eval()
    aggregate, records, predictions = evaluate_model_condition(
        model=model,
        episodes=[episode],
        split="train_50",
        condition="NORMAL",
        device=torch.device("cpu"),
        seed=SEED,
    )
    batch = collate_episodes([episode])
    with torch.no_grad():
        direct = model(**model_inputs_from_batch(batch))
    assert aggregate["episode_count"] == 1
    assert records[0]["applicable"] is True
    assert torch.equal(predictions[0]["prediction_colors"], direct["color_log_probs"].argmax(dim=-1)[0].cpu())
    assert predictions[0]["predicted_height"] == int(direct["height_logits"].argmax(dim=-1)[0]) + 1
    assert predictions[0]["predicted_width"] == int(direct["width_logits"].argmax(dim=-1)[0]) + 1


def test_output_deranged_preserves_output_multiset_masks_shapes_roles_and_has_no_fixed_pair() -> None:
    episode = _episode_with_multiple_supports()
    deranged, metadata = episode_for_condition(episode, "OUTPUT_DERANGED_WITHIN_TASK", SEED)
    assert deranged is not None
    perm = metadata["output_derangement_permutation"]
    assert all(perm[idx] != idx for idx in range(len(perm)))
    original_outputs = sorted(str(pair.output.to_list()) for pair in episode.supports)
    deranged_outputs = sorted(str(pair.output.to_list()) for pair in deranged.supports)
    assert deranged_outputs == original_outputs
    assert sorted(pair.output.shape for pair in deranged.supports) == sorted(pair.output.shape for pair in episode.supports)
    original_batch = collate_episodes([episode])
    deranged_batch = collate_episodes([deranged])
    assert int(original_batch["support_evidence_mask"].sum()) == int(deranged_batch["support_evidence_mask"].sum())
    assert sorted(original_batch["support_evidence_roles"][0, original_batch["support_evidence_mask"][0]].tolist()) == sorted(
        deranged_batch["support_evidence_roles"][0, deranged_batch["support_evidence_mask"][0]].tolist()
    )


def test_pair_order_shuffled_preserves_complete_input_output_pair_identity() -> None:
    episode = _episode_with_multiple_supports()
    shuffled, metadata = episode_for_condition(episode, "PAIR_ORDER_SHUFFLED", SEED)
    assert shuffled is not None
    assert metadata["pair_order_permutation"] != list(range(len(episode.supports)))
    original_pairs = sorted((str(pair.input.to_list()), str(pair.output.to_list())) for pair in episode.supports)
    shuffled_pairs = sorted((str(pair.input.to_list()), str(pair.output.to_list())) for pair in shuffled.supports)
    assert shuffled_pairs == original_pairs
    assert [pair.pair_index for pair in shuffled.supports] == list(range(len(shuffled.supports)))


def test_output_color_permuted_preserves_geometry_shape_masks_and_token_count() -> None:
    episode = _episode_with_multiple_supports()
    permuted, metadata = episode_for_condition(episode, "OUTPUT_COLOR_PERMUTED", SEED)
    assert permuted is not None
    permutation = metadata["color_permutation"]
    assert permutation != list(range(ARC_COLOR_COUNT))
    assert sorted(permutation) == list(range(ARC_COLOR_COUNT))
    for original, changed in zip(episode.supports, permuted.supports):
        assert original.output.shape == changed.output.shape
        assert original.output.height * original.output.width == changed.output.height * changed.output.width
    original_batch = collate_episodes([episode])
    permuted_batch = collate_episodes([permuted])
    output_role = original_batch["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT
    assert int((original_batch["support_evidence_mask"] & output_role).sum()) == int(
        (permuted_batch["support_evidence_mask"] & (permuted_batch["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)).sum()
    )


def test_support_evidence_path_zeroed_does_not_change_parameters_and_is_finite() -> None:
    _, batch = _batch()
    model = _small_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**model_inputs_from_batch(batch), disable_support_evidence=True)
    assert all(torch.isfinite(value).all() for value in outputs.values() if torch.is_tensor(value))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_transformation_cell_mask_handles_overlap_growth_and_shrinkage_cases() -> None:
    query = Grid.from_raw([[1, 2, 3], [4, 5, 6]])
    target = Grid.from_raw([[1, 9], [7, 5], [8, 8]])
    episode = type("EpisodeLike", (), {})()
    episode.query = type("QueryLike", (), {"input": query, "output": target})()
    mask = transformation_cell_mask(episode).view(30, 30)
    assert mask[0, 0].item() is False
    assert mask[0, 1].item() is True
    assert mask[1, 0].item() is True
    assert mask[1, 1].item() is False
    assert mask[2, 0].item() is True
    assert mask[2, 1].item() is True
    assert int(mask.sum()) == 4


def test_zero_transformation_cell_episodes_are_handled_safely() -> None:
    query = Grid.from_raw([[1, 2], [3, 4]])
    target = Grid.from_raw([[1, 2], [3, 4]])
    episode = type("EpisodeLike", (), {})()
    episode.query = type("QueryLike", (), {"input": query, "output": target})()
    prediction = identity_copy_baseline_prediction(query)
    metrics = prediction_metrics_for_episode(
        episode=episode,
        prediction_colors=prediction["prediction_colors"],
        predicted_height=prediction["predicted_height"],
        predicted_width=prediction["predicted_width"],
    )
    assert metrics["transformation_cell_count"] == 0
    assert metrics["transformation_cell_accuracy"] is None
    assert metrics["transformation_cell_hamming_error"] is None


def test_baselines_do_not_accept_query_target_arguments() -> None:
    assert list(inspect.signature(identity_copy_baseline_prediction).parameters) == ["query_input"]
    assert list(inspect.signature(support_shape_copy_baseline_prediction).parameters) == ["query_input", "supports"]


def test_bootstrap_output_is_deterministic_from_seed() -> None:
    values = [0.0, 1.0, -1.0, 2.0]
    first = paired_bootstrap_ci(values, seed=SEED, resamples=200)
    second = paired_bootstrap_ci(values, seed=SEED, resamples=200)
    different = paired_bootstrap_ci(values, seed=SEED + 1, resamples=200)
    assert first == second
    assert first != different


def test_old_v1_1_checkpoints_reports_and_source_zip_are_not_ablation_targets() -> None:
    assert V1_1_TASK_MANIFEST.exists()
    assert V1_1_BEST_CHECKPOINT.exists()
    assert V1_1_SOURCE_ZIP.exists()
    before = {
        "checkpoint": hash_file(V1_1_BEST_CHECKPOINT),
        "manifest": hash_file(V1_1_TASK_MANIFEST),
        "source_zip": hash_file(V1_1_SOURCE_ZIP),
    }
    source = inspect.getsource(run_v1_1_support_ablation)
    assert "save_checkpoint(" not in source
    assert "create_source_snapshot" not in source
    assert "V1_1_ARTIFACTS_DIR / \"vrm_v1_1" not in source
    assert V1_1_SUPPORT_ABLATION_DIR != V1_1_ARTIFACTS_DIR
    after = {
        "checkpoint": hash_file(V1_1_BEST_CHECKPOINT),
        "manifest": hash_file(V1_1_TASK_MANIFEST),
        "source_zip": hash_file(V1_1_SOURCE_ZIP),
    }
    assert before == after
