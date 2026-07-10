from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from .pair_reader_r1 import PairKeyValueReader, PairReaderConfig, output_nll_from_retrieved_probs
from .vrm_probe_data import TwinEpisode


FORBIDDEN_TRAINING_BATCH_KEYS = {
    "relevant_pair_index",
    "relevant_pair_indices",
    "retrieval_label",
    "retrieval_labels",
    "candidate_label",
    "candidate_labels",
    "label",
    "labels",
    "candidate_0",
    "candidate_1",
    "candidates",
    "pair_order",
    "pair_metadata",
    "base_id",
    "twin",
    "episode_id",
    "rule_id",
    "template_id",
}


class PairKeyValueReaderR1B(PairKeyValueReader):
    """R1 architecture used with output-only supervision."""

    def initialization_report(self) -> dict[str, Any]:
        report = super().initialization_report()
        report.update(
            {
                "architecture": "Pair Key-Value Reader R1B",
                "training_objective": "output-only NLL from retrieved probabilities to target grid",
                "retrieval_supervision_used": False,
                "r1b_config": asdict(self.config),
            }
        )
        return report


def assert_output_only_training_batch(batch: dict[str, torch.Tensor]) -> None:
    forbidden = sorted(FORBIDDEN_TRAINING_BATCH_KEYS.intersection(batch))
    if forbidden:
        raise RuntimeError(f"forbidden output-only training keys present: {forbidden}")


def output_only_loss(outputs: dict[str, torch.Tensor], target_grids: torch.Tensor) -> torch.Tensor:
    return output_nll_from_retrieved_probs(outputs["retrieved_color_probabilities"], target_grids)


def output_only_training_keys() -> list[str]:
    return ["query_grids", "support_input_grids", "support_output_grids", "target_grids"]


__all__ = [
    "PairReaderConfig",
    "PairKeyValueReaderR1B",
    "assert_output_only_training_batch",
    "output_only_loss",
    "output_only_training_keys",
]

