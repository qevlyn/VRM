from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from vrm2.inductor.episodes import load_support_episodes
from vrm2.inductor.train_v1 import (
    _length_audit,
    sequential_program_loss,
    shuffled_instruction_targets,
    train_one_seed_v1,
)


ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "artifacts/vrm2/inductor/runtime/v1/episodes"


def test_optimizer_corpus_contains_only_length1_and_length2_targets():
    episodes = load_support_episodes(EPISODES / "train_supports.jsonl")
    audit = _length_audit(episodes)
    assert audit == {
        "length_1_episodes": 1480,
        "length_2_episodes": 1396,
        "length_3_episodes_reaching_optimizer": 0,
        "maximum_active_target_length_observed_by_optimizer": 2,
    }


def test_length1_and_length2_targets_have_canonical_null_suffixes():
    episodes = load_support_episodes(EPISODES / "train_supports.jsonl")
    length1 = next(item for item in episodes if item.target_program_length == 1)
    length2 = next(item for item in episodes if item.target_program_length == 2)
    assert length1.target_typed_program_ids[1:] == ((0, 0, 0, 0), (0, 0, 0, 0))
    assert length2.target_typed_program_ids[2] == (0, 0, 0, 0)


def test_training_and_selection_source_has_no_coverage_corpus_dependency():
    source = inspect.getsource(train_one_seed_v1)
    assert "coverage" not in source.lower()
    assert "sealed" not in source.lower()


def test_shuffled_order_is_nonidentity_for_every_length2_target():
    programs = torch.tensor(
        [
            [[1, 1, 1, 1], [2, 2, 2, 4], [0, 0, 0, 0]],
            [[3, 3, 3, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    shuffled = shuffled_instruction_targets(programs, torch.tensor([2, 1]))
    assert torch.equal(shuffled[0, 0], programs[0, 1])
    assert torch.equal(shuffled[0, 1], programs[0, 0])
    assert torch.equal(shuffled[1], programs[1])


def test_supervised_loss_includes_null_suffix_positions():
    targets = torch.tensor([[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]])
    logits = {
        "operator": torch.zeros(1, 3, 4),
        "selector": torch.zeros(1, 3, 4),
        "argument_type": torch.zeros(1, 3, 4),
        "argument_value": torch.zeros(1, 3, 10),
    }
    baseline = sequential_program_loss(logits, targets)[0]
    logits["operator"][:, 1:, 0] = 10
    improved = sequential_program_loss(logits, targets)[0]
    assert improved < baseline


def test_training_audit_artifact_records_excluded_original_l3():
    audit = json.loads((EPISODES / "training_data_audit.json").read_text())
    assert audit["excluded_length_3_episodes"] == 7
    assert audit["maximum_active_target_length_observed_by_optimizer"] == 2
