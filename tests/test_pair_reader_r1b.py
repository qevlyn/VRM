from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.data import Grid
from vrm.twin_tasks.epoch_sampler import coverage_row, epoch_permutation_indices, epoch_training_view
from vrm.twin_tasks.pair_reader_r1 import PairReaderConfig
from vrm.twin_tasks.pair_reader_r1b import (
    PairKeyValueReaderR1B,
    assert_output_only_training_batch,
    output_only_loss,
    output_only_training_keys,
)
from vrm.twin_tasks.run_pair_reader_r1b import (
    candidate_metrics,
    cuda_environment_report,
    eval_forced_alpha,
    eval_forced_value,
    foreground_metric,
    load_split_episodes,
    protected_hashes,
    relevant_pair_index,
    training_batch_tensors,
)
from vrm.twin_tasks.vrm_probe_data import (
    TwinEpisode,
    build_opposite_lookup,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
)

DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
SEED = 20260628


def _episodes(split: str = "train", count: int = 8) -> list[TwinEpisode]:
    return records_to_episodes(read_jsonl(DATA / f"{split}.jsonl"))[:count]


def test_cuda_is_required_and_cpu_fallback_forbidden() -> None:
    report = cuda_environment_report()
    assert report["torch_cuda_is_available"] is True
    assert report["device"] == "cuda"


def test_existing_r1_source_hash_remains_unchanged() -> None:
    paths = [
        Path("/mnt/d/VRM/vrm/twin_tasks/pair_reader_r1.py"),
        Path("/mnt/d/VRM/vrm/twin_tasks/run_pair_reader_r1.py"),
    ]
    before = [path.read_bytes() for path in paths]
    _ = PairKeyValueReaderR1B(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    after = [path.read_bytes() for path in paths]
    assert before == after


def test_existing_model_py_hash_remains_unchanged() -> None:
    path = Path("/mnt/d/VRM/vrm/model.py")
    before = path.read_bytes()
    _ = output_only_training_keys()
    assert path.read_bytes() == before


def test_existing_artifacts_checkpoints_archives_unchanged_during_read_only_checks() -> None:
    before = protected_hashes()
    _ = _episodes("dev", 4)
    after = protected_hashes()
    assert before == after


def test_training_batch_cannot_contain_relevant_pair_index() -> None:
    episodes = _episodes("train", 2)
    views = [epoch_training_view(ep, base_seed=SEED, epoch_index=0)[0] for ep in episodes]
    batch = training_batch_tensors(views, episodes, torch.device("cuda"))
    assert "relevant_pair_indices" not in batch
    assert "candidate_labels" not in batch
    assert "label" not in batch
    assert set(batch) == set(output_only_training_keys())
    with pytest.raises(RuntimeError):
        assert_output_only_training_batch({**batch, "relevant_pair_indices": torch.zeros(2, dtype=torch.long, device="cuda")})


def test_training_loss_cannot_access_retrieval_labels() -> None:
    signature = inspect.signature(output_only_loss)
    assert "relevant_pair_indices" not in signature.parameters
    assert "retrieval_logits" not in signature.parameters
    probs = torch.zeros((1, 1, 1, 10), device="cuda")
    probs[..., 3] = 1.0
    loss = output_only_loss({"retrieved_color_probabilities": probs}, torch.tensor([[[3]]], device="cuda"))
    assert loss.item() == pytest.approx(0.0)


def test_candidate_labels_target_and_metadata_never_enter_forward() -> None:
    signature = inspect.signature(PairKeyValueReaderR1B.forward)
    assert not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    model = PairKeyValueReaderR1B(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1)).cuda()
    q = torch.zeros((1, 7, 7), dtype=torch.long, device="cuda")
    q[:, 1, 1] = 4
    supports = q[:, None].repeat(1, 3, 1, 1)
    with pytest.raises(TypeError):
        model(q, supports, supports, target_grid=q)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        model(q, supports, supports, label=torch.tensor([0], device="cuda"))  # type: ignore[call-arg]


def test_query_and_support_input_use_shared_matcher_weights() -> None:
    model = PairKeyValueReaderR1B(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    assert model.query_encoder is model.support_input_encoder


def test_no_pair_position_or_pair_id_embedding_exists() -> None:
    model = PairKeyValueReaderR1B(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    names = "\n".join(name for name, _ in model.named_parameters()).lower()
    assert "pair_position" not in names
    assert "pair_id" not in names
    assert "pair_index" not in names


def test_full_coverage_is_exact_for_seed_and_epoch() -> None:
    episodes = _episodes("train", 100)
    ids = [episode.episode_id for episode in episodes]
    for seed in (20260628, 20260629):
        for epoch in (0, 1):
            order = epoch_permutation_indices(len(episodes), base_seed=seed, epoch_index=epoch)
            row = coverage_row(epoch_index=epoch, order=order, observed=order, episode_ids=ids, base_seed=seed)
            assert row["passed"]
            assert row["duplicate_episode_count"] == 0
            assert row["missing_episode_count"] == 0


def test_all_six_coherent_pair_permutations_preserve_logical_output() -> None:
    import itertools

    episode = _episodes("train", 1)[0]
    original = sorted((pair.input.cells, pair.output.cells) for pair in episode.supports)
    for perm in itertools.permutations(range(3)):
        view = epoch_training_view(episode, base_seed=SEED, epoch_index=0)[0]
        reordered = [view.supports[idx] for idx in perm]
        assert sorted((pair.input.cells, pair.output.cells) for pair in reordered) == original


def test_twin_swap_preserves_query_and_unpaired_bags() -> None:
    episodes = _episodes("train", 16)
    episode = episodes[0]
    opposite = build_opposite_lookup(episodes)[f"{episode.base_id}:{episode.twin}"]
    assert opposite.base_id == episode.base_id
    assert opposite.query_input == episode.query_input
    assert support_input_bag_signature(opposite) == support_input_bag_signature(episode)
    assert support_output_bag_signature(opposite) == support_output_bag_signature(episode)


def test_forced_value_intervention_leaves_alpha_unchanged() -> None:
    model = PairKeyValueReaderR1B(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1)).cuda()
    report = eval_forced_value(model=model, episodes=_episodes("train", 2), split="unit", seed=SEED, device=torch.device("cuda"), batch_size=2)
    assert report["forced_value_alpha_unchanged_rate"] == 1.0


def test_forced_alpha_intervention_exactly_returns_selected_y() -> None:
    report = eval_forced_alpha(episodes=_episodes("train", 4), split="unit")
    assert report["forced_alpha_exact_route_rate"] == 1.0


def test_sealed_is_unread_before_checkpoint_selection() -> None:
    with pytest.raises(RuntimeError):
        load_split_episodes(DATA, "sealed", sealed_allowed=False)


def test_metric_calculations_agree_with_hand_fixture() -> None:
    pred = torch.tensor([[0, 2], [3, 0]])
    target = torch.tensor([[0, 2], [4, 0]])
    fg = foreground_metric(pred, target)
    assert fg["foreground_color_total"] == 2
    assert fg["foreground_color_correct"] == 1
    episode = TwinEpisode(
        record_index=0,
        split="unit",
        base_id="base",
        twin="A",
        family="unit",
        label=0,
        supports=tuple(),
        query_input=Grid.from_raw([[0, 1], [0, 0]]),
        target_output=Grid.from_raw([[0, 2], [2, 0]]),
        candidates=(Grid.from_raw([[0, 2], [2, 0]]), Grid.from_raw([[0, 3], [3, 0]])),
    )
    cand = candidate_metrics(torch.tensor([[0, 2], [2, 0]]), episode)
    assert cand["candidate_decision"] == 0
    assert cand["candidate_nearest_score"] == 1.0


def test_existing_r1_artifact_hash_values_remain_unchanged_after_read() -> None:
    path = Path("/mnt/d/VRM/artifacts/twin_pair_reader_r1/summary_report.json")
    before = path.read_bytes()
    _ = path.read_text(encoding="utf-8")[:128]
    assert path.read_bytes() == before


def test_retrieval_label_identification_is_unique_but_eval_only() -> None:
    for episode in _episodes("train", 8):
        idx = relevant_pair_index(episode)
        assert episode.supports[idx].input == episode.query_input
        assert sum(pair.input == episode.query_input for pair in episode.supports) == 1

