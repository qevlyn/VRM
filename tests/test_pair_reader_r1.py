from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.data import ArcPair, Grid
from vrm.twin_tasks.epoch_sampler import coverage_row, epoch_permutation_indices
from vrm.twin_tasks.pair_reader_r1 import PairKeyValueReader, PairReaderConfig, convex_value_mix
from vrm.twin_tasks.run_pair_reader_r1 import (
    candidate_metrics,
    cuda_environment_report,
    foreground_metric,
    load_split_episodes,
    oracle_adapter_report,
    protected_hashes,
    relevant_pair_index,
)
from vrm.twin_tasks.vrm_probe_data import (
    TwinEpisode,
    build_opposite_lookup,
    condition_view,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
)

DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
SEED = 20260628


def _episodes(split: str = "train", count: int = 8) -> list[TwinEpisode]:
    return records_to_episodes(read_jsonl(DATA / f"{split}.jsonl"))[:count]


def test_cuda_is_required_and_cpu_fallback_is_forbidden() -> None:
    report = cuda_environment_report()
    assert report["torch_cuda_is_available"] is True
    assert report["device"] == "cuda"


def test_existing_model_hash_is_unchanged_by_reader_import() -> None:
    path = Path("/mnt/d/VRM/vrm/model.py")
    before = path.read_bytes()
    _ = PairKeyValueReader(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    after = path.read_bytes()
    assert before == after


def test_existing_twin_probe_and_epoch_baseline_artifacts_are_unchanged_during_read_only_checks() -> None:
    before = protected_hashes()
    _ = _episodes("dev", 4)
    after = protected_hashes()
    assert before == after


def test_oracle_adapter_returns_exact_target_for_fixture_twins() -> None:
    episodes = {"train": _episodes("train", 8), "dev": _episodes("dev", 8), "sealed": _episodes("sealed", 8)}
    report = oracle_adapter_report(episodes)
    assert report["passed"]
    for split in episodes:
        assert report[split]["direct_grid_exact_accuracy"] == 1.0
        assert report[split]["foreground_color_accuracy"] == 1.0
        assert report[split]["target_rectangle_hamming_error"] == 0.0


def test_query_and_support_input_matcher_share_exact_weights() -> None:
    model = PairKeyValueReader(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    assert model.query_encoder is model.support_input_encoder


def test_forbidden_metadata_cannot_enter_reader_forward() -> None:
    signature = inspect.signature(PairKeyValueReader.forward)
    assert not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    model = PairKeyValueReader(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1)).cuda()
    query = torch.zeros((1, 7, 7), dtype=torch.long, device="cuda")
    query[:, 1, 1] = 2
    supports = query[:, None].repeat(1, 3, 1, 1)
    with pytest.raises(TypeError):
        model(query, supports, supports, label=torch.tensor([0], device="cuda"))  # type: ignore[call-arg]


def test_no_support_pair_or_pair_position_embedding_exists() -> None:
    model = PairKeyValueReader(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1))
    names = "\n".join(name for name, _ in model.named_parameters())
    assert "pair_position" not in names
    assert "pair_index" not in names
    assert "support_pair_embedding" not in names


def test_support_pair_permutation_permutes_alpha_and_preserves_retrieved_output() -> None:
    episode = _episodes("train", 1)[0]
    model = PairKeyValueReader(PairReaderConfig(hidden_size=32, num_heads=4, num_layers=1)).cuda().eval()
    query = torch.tensor([episode.query_input.to_list()], dtype=torch.long, device="cuda")
    support_inputs = torch.tensor([[pair.input.to_list() for pair in episode.supports]], dtype=torch.long, device="cuda")
    support_outputs = torch.tensor([[pair.output.to_list() for pair in episode.supports]], dtype=torch.long, device="cuda")
    order = [2, 0, 1]
    with torch.no_grad():
        base = model(query, support_inputs, support_outputs)
        permuted = model(query, support_inputs[:, order], support_outputs[:, order])
    restored_alpha = torch.empty_like(permuted["retrieval_weights"])
    for new_idx, old_idx in enumerate(order):
        restored_alpha[:, old_idx] = permuted["retrieval_weights"][:, new_idx]
    assert torch.allclose(base["retrieval_weights"], restored_alpha, atol=1e-6)
    assert torch.allclose(base["retrieved_color_probabilities"], permuted["retrieved_color_probabilities"], atol=1e-6)
    assert torch.equal(base["decoded_grid"], permuted["decoded_grid"])


def test_twin_swap_preserves_query_and_unpaired_support_bags() -> None:
    episodes = _episodes("train", 16)
    episode = episodes[0]
    opposite = build_opposite_lookup(episodes)[f"{episode.base_id}:{episode.twin}"]
    swapped = condition_view(episode, condition="TWIN_SWAP_SUPPORTS", opposite_by_episode=build_opposite_lookup(episodes), seed=SEED)
    assert swapped.query_input == episode.query_input
    assert support_input_bag_signature(swapped) == support_input_bag_signature(episode)
    assert support_output_bag_signature(swapped) == support_output_bag_signature(episode)
    assert support_signature(swapped) != support_signature(episode)
    assert opposite.base_id == episode.base_id


def test_retrieval_supervision_label_identifies_exactly_one_logical_pair() -> None:
    for episode in _episodes("train", 16):
        idx = relevant_pair_index(episode)
        assert episode.supports[idx].input == episode.query_input
        assert sum(pair.input == episode.query_input for pair in episode.supports) == 1


def test_raw_value_retrieval_routes_selected_pair_output_grid() -> None:
    support_outputs = torch.tensor(
        [[[[0, 1], [1, 0]], [[0, 2], [2, 0]], [[0, 3], [3, 0]]]],
        dtype=torch.long,
    )
    alpha = torch.tensor([[0.0, 1.0, 0.0]])
    probs = convex_value_mix(support_outputs, alpha)
    assert torch.equal(probs.argmax(dim=-1), support_outputs[:, 1])


def test_mixed_alpha_creates_expected_convex_combination() -> None:
    support_outputs = torch.tensor([[[[1]], [[2]]]], dtype=torch.long)
    alpha = torch.tensor([[0.25, 0.75]])
    probs = convex_value_mix(support_outputs, alpha)
    assert probs[0, 0, 0, 1].item() == pytest.approx(0.25)
    assert probs[0, 0, 0, 2].item() == pytest.approx(0.75)


def test_full_epoch_coverage_has_zero_duplicates_and_zero_missing() -> None:
    episodes = _episodes("train", 100)
    ids = [episode.episode_id for episode in episodes]
    order = epoch_permutation_indices(len(episodes), base_seed=SEED, epoch_index=0)
    row = coverage_row(epoch_index=0, order=order, observed=order, episode_ids=ids, base_seed=SEED)
    assert row["passed"]
    assert row["duplicate_episode_count"] == 0
    assert row["missing_episode_count"] == 0


def test_sealed_split_cannot_be_read_before_best_checkpoint_selection() -> None:
    with pytest.raises(RuntimeError):
        load_split_episodes(DATA, "sealed", sealed_allowed=False)


def test_checkpoint_hash_remains_unchanged_after_read_only_hashing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mini.pt"
    torch.save({"x": torch.tensor([1, 2, 3])}, checkpoint)
    before = checkpoint.read_bytes()
    _ = checkpoint.read_bytes()
    after = checkpoint.read_bytes()
    assert before == after


def test_reported_metrics_agree_with_small_hand_fixture() -> None:
    pred = torch.tensor([[0, 1], [2, 0]])
    target = torch.tensor([[0, 1], [3, 0]])
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
        supports=(
            ArcPair(Grid.from_raw([[0, 1], [0, 0]]), Grid.from_raw([[0, 1], [1, 0]]), 0),
            ArcPair(Grid.from_raw([[0, 2], [0, 0]]), Grid.from_raw([[0, 2], [2, 0]]), 1),
        ),
        query_input=Grid.from_raw([[0, 1], [0, 0]]),
        target_output=Grid.from_raw([[0, 1], [1, 0]]),
        candidates=(Grid.from_raw([[0, 1], [1, 0]]), Grid.from_raw([[0, 2], [2, 0]])),
    )
    cand = candidate_metrics(torch.tensor([[0, 1], [1, 0]]), episode)
    assert cand["candidate_decision"] == 0
    assert cand["candidate_nearest_score"] == 1.0

