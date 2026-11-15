from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.model import VRMConfig
from vrm.twin_tasks.run_pilot import protected_hashes
from vrm.twin_tasks.run_vrm_probe import hash_file
from vrm.twin_tasks.vrm_probe_data import (
    build_opposite_lookup,
    collate_twin_episodes,
    load_twin_episodes,
    model_inputs_for_twin_batch,
    pair_order_shuffled_view,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
    training_view,
    twin_swap_supports_view,
)
from vrm.twin_tasks.vrm_probe_metrics import candidate_decision, pair_level_metrics
from vrm.twin_tasks.vrm_probe_train import initialize_fresh_model, is_better_dev, save_checkpoint

DATA_DIR = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
OUT_DIR = Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0")
SEED = 20260628


def _episodes():
    return load_twin_episodes(DATA_DIR)


def _first_pair():
    episodes = _episodes()["train"]
    base = episodes[0].base_id
    twins = {episode.twin: episode for episode in episodes if episode.base_id == base}
    return twins["A"], twins["B"]


def _tiny_config() -> VRMConfig:
    return VRMConfig(hidden_size=16, num_heads=4, pair_encoder_layers=1, rewrite_layers=1, rewrite_steps=1, rule_slots=2)


def _state_hash(model) -> str:
    payload = b"".join(tensor.detach().cpu().numpy().tobytes() for tensor in model.state_dict().values())
    return hashlib.sha256(payload).hexdigest()


def test_fresh_initialization_does_not_load_v1_1_checkpoint(monkeypatch) -> None:
    def forbidden_load(*args, **kwargs):
        raise AssertionError("torch.load must not be called for fresh twin probe initialization")

    monkeypatch.setattr(torch, "load", forbidden_load)
    model, report = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED)
    assert report["fresh_random_initialization"]
    assert not report["loaded_checkpoint"]
    assert report["loaded_checkpoint_path"] is None
    assert model is not None


def test_fresh_initialization_is_deterministic_from_seed() -> None:
    first, _ = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED)
    second, _ = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED)
    third, _ = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED + 1)
    assert _state_hash(first) == _state_hash(second)
    assert _state_hash(first) != _state_hash(third)


def test_candidates_labels_targets_and_ids_do_not_enter_model_inputs() -> None:
    twin_a, _ = _first_pair()
    batch = collate_twin_episodes([twin_a])
    model_inputs = model_inputs_for_twin_batch(batch)
    forbidden = {
        "candidate_0",
        "candidate_1",
        "candidates",
        "label",
        "query_targets",
        "height_targets",
        "width_targets",
        "target",
        "base_id",
        "twin",
        "metadata",
        "pair_position",
    }
    assert forbidden.isdisjoint(model_inputs)


def test_normal_adapter_preserves_coherent_support_correspondence() -> None:
    twin_a, _ = _first_pair()
    assert len(twin_a.supports) == 3
    assert all(pair.input.height == 7 and pair.output.height == 7 for pair in twin_a.supports)
    assert len(set(support_signature(twin_a))) == 3


def test_twin_swap_preserves_query_candidates_and_bags_but_changes_correspondence() -> None:
    twin_a, twin_b = _first_pair()
    swapped = twin_swap_supports_view(twin_a, twin_b)
    assert swapped.query_input == twin_a.query_input
    assert swapped.candidates == twin_a.candidates
    assert support_input_bag_signature(swapped) == support_input_bag_signature(twin_a)
    assert support_output_bag_signature(swapped) == support_output_bag_signature(twin_a)
    assert support_signature(swapped) != support_signature(twin_a)


def test_pair_order_shuffled_preserves_complete_pairs() -> None:
    twin_a, _ = _first_pair()
    shuffled = pair_order_shuffled_view(twin_a, seed=SEED)
    assert sorted(support_signature(shuffled)) == sorted(support_signature(twin_a))
    assert support_signature(shuffled) != support_signature(twin_a)


def test_training_pair_randomization_preserves_pair_coherence() -> None:
    twin_a, _ = _first_pair()
    views = [training_view(twin_a, seed=SEED, step=step) for step in range(1, 8)]
    assert all(sorted(support_signature(view)) == sorted(support_signature(twin_a)) for view in views)
    assert len({tuple(support_signature(view)) for view in views}) >= 2


def test_candidate_decisions_handle_ties_without_candidate_order_bias() -> None:
    pred = torch.tensor([[1, 2], [3, 4]])
    cand0 = torch.tensor([[1, 2], [9, 9]])
    cand1 = torch.tensor([[9, 9], [3, 4]])
    decision = candidate_decision(pred, cand0, cand1, correct_label=0)
    assert decision["candidate_decision"] is None
    assert decision["candidate_nearest_score"] == 0.5


def test_candidate_distance_scoring_is_correct() -> None:
    pred = torch.tensor([[1, 2], [3, 4]])
    cand0 = torch.tensor([[1, 2], [3, 4]])
    cand1 = torch.tensor([[1, 2], [0, 0]])
    decision = candidate_decision(pred, cand0, cand1, correct_label=0)
    assert decision["distance_to_candidate_0"] == 0
    assert decision["distance_to_candidate_1"] == 2
    assert decision["candidate_decision"] == 0
    assert decision["candidate_nearest_score"] == 1.0


def test_twin_labels_remain_complementary() -> None:
    twin_a, twin_b = _first_pair()
    assert twin_a.label + twin_b.label == 1


def test_pair_level_joint_and_mapping_flip_metrics_on_fixture() -> None:
    twin_a, twin_b = _first_pair()
    normal = [
        {"episode_id": twin_a.episode_id, "candidate_decision": twin_a.label},
        {"episode_id": twin_b.episode_id, "candidate_decision": twin_b.label},
    ]
    swapped = [
        {"episode_id": twin_a.episode_id, "candidate_decision": twin_a.alternate_label},
        {"episode_id": twin_b.episode_id, "candidate_decision": twin_b.alternate_label},
    ]
    metrics = pair_level_metrics(
        [twin_a, twin_b],
        {
            "NORMAL": normal,
            "TWIN_SWAP_SUPPORTS": swapped,
        },
    )
    assert metrics["twin_pair_joint_normal_accuracy"] == 1.0
    assert metrics["counterfactual_switch_rate"] == 1.0
    assert metrics["correct_mapping_switch_rate"] == 1.0
    assert metrics["twin_pair_mapping_flip_consistency"] == 1.0


def test_dev_selection_never_requires_sealed_metrics() -> None:
    weaker = {
        "candidate_nearest_score": 0.8,
        "twin_pair_joint_normal_accuracy": 0.7,
        "direct_grid_exact_accuracy": 0.2,
        "mean_target_rectangle_hamming_error": 4.0,
    }
    stronger = {
        "candidate_nearest_score": 0.8,
        "twin_pair_joint_normal_accuracy": 0.8,
        "direct_grid_exact_accuracy": 0.1,
        "mean_target_rectangle_hamming_error": 9.0,
    }
    assert is_better_dev(stronger, weaker)
    assert "sealed" not in stronger


def test_sealed_evaluation_cannot_alter_selected_checkpoint(tmp_path) -> None:
    model, _ = initialize_fresh_model(_tiny_config(), torch.device("cpu"), SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "selected.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, config=_tiny_config(), step=0, extra={})
    before = hash_file(path)
    _ = build_opposite_lookup(_episodes()["sealed"][:2])
    after = hash_file(path)
    assert before == after


def test_existing_v1_1_files_artifacts_and_checkpoints_remain_unchanged() -> None:
    before = protected_hashes()
    _ = _episodes()
    after = protected_hashes()
    assert before == after


def test_p6_provenance_report_is_produced() -> None:
    path = OUT_DIR / "p6_provenance.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "P6_IS_SEPARATELY_TRAINED_ON_DERANGED_SUPPORTS" in text
