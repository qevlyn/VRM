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
from vrm.model import VRM
from vrm.twin_tasks.run_vrm_pair_reader_r2a import (
    build_model_for_seed,
    cuda_environment_report,
    endpoint_equivalence,
    eval_condition,
    load_split_episodes,
    protected_hashes,
    r2a_batch,
    relevant_pair_index,
    reorder_episode,
    train_step,
    twin_swap_supports_view,
)
from vrm.twin_tasks.vrm_pair_reader_r2a import VRMWithFrozenPairReaderR2A
from vrm.twin_tasks.vrm_probe_data import (
    TwinEpisode,
    build_opposite_lookup,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
)

DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
SEED = 20260628
READER_CKPTS = Path("/mnt/d/VRM/checkpoints/twin_pair_reader_r1b")


def _episodes(split: str = "train", count: int = 8) -> list[TwinEpisode]:
    return records_to_episodes(read_jsonl(DATA / f"{split}.jsonl"))[:count]


def _args(tmp_path: Path):
    class Args:
        reader_checkpoints = READER_CKPTS
        out = tmp_path
        checkpoint_dir = tmp_path
        learning_rate = 3e-4
        beta1 = 0.9
        beta2 = 0.95
        weight_decay = 0.01
        grad_clip_norm = 1.0
        batch_size = 4
        eval_batch_size = 4
        epochs = 1
        mode = "full"
        max_batches = None

    return Args()


def test_cuda_required_and_cpu_fallback_forbidden() -> None:
    report = cuda_environment_report()
    assert report["torch_cuda_is_available"] is True
    assert report["device"] == "cuda"


def test_model_py_r1_r1b_sources_unchanged_by_import() -> None:
    paths = [
        Path("/mnt/d/VRM/vrm/model.py"),
        Path("/mnt/d/VRM/vrm/twin_tasks/pair_reader_r1.py"),
        Path("/mnt/d/VRM/vrm/twin_tasks/run_pair_reader_r1.py"),
        Path("/mnt/d/VRM/vrm/twin_tasks/pair_reader_r1b.py"),
        Path("/mnt/d/VRM/vrm/twin_tasks/run_pair_reader_r1b.py"),
    ]
    before = [path.read_bytes() for path in paths]
    _ = VRMWithFrozenPairReaderR2A(legacy_vrm=VRM())
    assert [path.read_bytes() for path in paths] == before


def test_prior_artifacts_checkpoints_archives_unchanged_during_read_only_checks() -> None:
    before = protected_hashes()
    _ = _episodes("dev", 2)
    assert protected_hashes() == before


def test_gate_zero_equals_original_v1_1_within_tolerance(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    legacy, _ = endpoint_equivalence(model=model, episodes=_episodes("train", 4), device=torch.device("cuda"), batch_size=4)
    assert legacy["max_abs_difference"] <= 1e-6
    assert legacy["passed"]


def test_gate_one_equals_frozen_r1b_within_tolerance(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    _, reader = endpoint_equivalence(model=model, episodes=_episodes("train", 4), device=torch.device("cuda"), batch_size=4)
    assert reader["max_abs_difference"] <= 1e-6
    assert reader["decoded_grids_exactly_equal"]


def test_gate_output_is_neutral_at_initialization(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    batch = r2a_batch(_episodes("train", 2), torch.device("cuda"))
    with torch.no_grad():
        out = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    assert float(out["gate"].mean().item()) == pytest.approx(0.5, abs=1e-6)


def test_gradient_isolation_and_gate_gradient_nonzero(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    optimizer = torch.optim.AdamW(model.gate_head.parameters(), lr=3e-4)
    row = train_step(
        model=model,
        optimizer=optimizer,
        train_episodes=_episodes("train", 8),
        batch_indices=[0, 1, 2, 3],
        seed=SEED,
        epoch_index=0,
        global_step=1,
        grad_clip_norm=1.0,
        device=torch.device("cuda"),
    )
    assert row["legacy_grad_norm"] == 0.0
    assert row["reader_grad_norm"] == 0.0
    assert row["gate_grad_norm"] > 0.0


def test_target_relevant_pair_and_candidates_never_enter_forward(tmp_path: Path) -> None:
    signature = inspect.signature(VRMWithFrozenPairReaderR2A.forward)
    assert not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    batch = r2a_batch(_episodes("train", 1), torch.device("cuda"))
    with pytest.raises(TypeError):
        model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], target_grid=batch["target_grids"])  # type: ignore[call-arg]
    bad_inputs = dict(batch["vrm_inputs"])
    bad_inputs["relevant_pair_index"] = torch.tensor([0], device="cuda")
    with pytest.raises(ValueError):
        model(bad_inputs, batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])


def test_reader_branch_receives_only_q_x_y(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    signature = inspect.signature(model.reader.forward)
    assert list(signature.parameters) == ["query_grids", "support_input_grids", "support_output_grids"]


def test_complete_pair_permutation_preserves_correspondence() -> None:
    episode = _episodes("train", 1)[0]
    original = sorted(support_signature(episode))
    for order in [(0, 1, 2), (2, 0, 1), (1, 2, 0)]:
        view = reorder_episode(episode, order)
        assert sorted(support_signature(view)) == original


def test_all_six_pair_permutations_preserve_final_output_after_training_like_gate(tmp_path: Path) -> None:
    import itertools

    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    with torch.no_grad():
        model.gate_head.weight[:, -1].fill_(1.0)
    episode = _episodes("train", 1)[0]
    batch = r2a_batch([episode], torch.device("cuda"))
    with torch.no_grad():
        base = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    for order in itertools.permutations(range(3)):
        view = reorder_episode(episode, order)
        perm_batch = r2a_batch([view], torch.device("cuda"))
        with torch.no_grad():
            out = model(perm_batch["vrm_inputs"], perm_batch["query_grids"], perm_batch["support_input_grids"], perm_batch["support_output_grids"])
        assert torch.equal(base["decoded_grid"], out["decoded_grid"])


def test_twin_swap_preserves_query_and_unpaired_bags() -> None:
    episodes = _episodes("train", 16)
    episode = episodes[0]
    opposite = build_opposite_lookup(episodes)[f"{episode.base_id}:{episode.twin}"]
    swapped = twin_swap_supports_view(episode, opposite)
    assert swapped.query_input == episode.query_input
    assert support_input_bag_signature(swapped) == support_input_bag_signature(episode)
    assert support_output_bag_signature(swapped) == support_output_bag_signature(episode)


def test_reader_disabled_changes_only_reader_probs_not_gate_legacy_or_alpha(tmp_path: Path) -> None:
    model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    with torch.no_grad():
        model.gate_head.weight[:, -1].fill_(1.0)
    batch = r2a_batch(_episodes("train", 2), torch.device("cuda"))
    with torch.no_grad():
        normal = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        disabled = model(
            batch["vrm_inputs"],
            batch["query_grids"],
            batch["support_input_grids"],
            batch["support_output_grids"],
            reader_disabled=True,
        )
    assert torch.allclose(normal["gate"], disabled["gate"])
    assert torch.allclose(normal["legacy_probs"], disabled["legacy_probs"])
    assert torch.allclose(normal["reader_alpha"], disabled["reader_alpha"])
    assert not torch.allclose(normal["reader_probs_for_mix"], disabled["reader_probs_for_mix"])


def test_frozen_reader_checkpoint_sha256_stable_after_load(tmp_path: Path) -> None:
    path = READER_CKPTS / "pair_reader_r1b_seed_20260628_best_dev.pt"
    before = path.read_bytes()
    _model, _ = build_model_for_seed(SEED, _args(tmp_path), torch.device("cuda"))
    assert path.read_bytes() == before


def test_existing_model_behavior_untouched_outside_r2a_sources() -> None:
    model = VRM().cuda().eval()
    episodes = _episodes("train", 1)
    batch = r2a_batch(episodes, torch.device("cuda"))
    with torch.no_grad():
        out1 = model(**batch["vrm_inputs"])["color_probabilities"]
        out2 = model(**batch["vrm_inputs"])["color_probabilities"]
    assert torch.allclose(out1, out2)


def test_sealed_guard_before_selection() -> None:
    with pytest.raises(RuntimeError):
        load_split_episodes(DATA, "sealed", sealed_allowed=False)


def test_metric_fixture_relevant_pair_unique() -> None:
    episode = _episodes("train", 1)[0]
    idx = relevant_pair_index(episode)
    assert episode.supports[idx].input == episode.query_input
    assert sum(pair.input == episode.query_input for pair in episode.supports) == 1

