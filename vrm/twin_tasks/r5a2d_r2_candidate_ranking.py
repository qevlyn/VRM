from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .r5a2d_r1_raw_grid_fidelity import Grid, canonical_grid_sha256, grid_shape

K_VALUES = [1, 4, 16, 64, 128]
SHAPE_MISMATCH_PENALTY_KIND = "target_height_times_target_width"


@dataclass(frozen=True)
class Candidate:
    source_expert: str
    frozen_candidate_id: str
    frozen_catalog_hash: str
    query_index: int
    output_grid: Grid | None
    output_shape: tuple[int, int | None] | None
    visible_support_count: int
    support_shape_mismatch_count: int
    support_cell_hamming_loss: int
    support_failed_pair_count: int
    raw_candidate_trace_ids: tuple[str, ...]
    canonical_output_hash: str | None
    raw_grid_fidelity_passed: bool
    execution_status: str

    def rank_tuple(self) -> tuple[int, int, int, str, str, str]:
        return (
            self.support_shape_mismatch_count,
            self.support_cell_hamming_loss,
            self.support_failed_pair_count,
            self.source_expert,
            self.frozen_candidate_id,
            self.canonical_output_hash or "",
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_expert": self.source_expert,
            "frozen_candidate_id": self.frozen_candidate_id,
            "frozen_catalog_hash": self.frozen_catalog_hash,
            "query_index": self.query_index,
            "output_grid": grid_to_list(self.output_grid),
            "output_shape": list(self.output_shape) if self.output_shape is not None else None,
            "visible_support_count": self.visible_support_count,
            "support_shape_mismatch_count": self.support_shape_mismatch_count,
            "support_cell_hamming_loss": self.support_cell_hamming_loss,
            "support_failed_pair_count": self.support_failed_pair_count,
            "raw_candidate_trace_ids": list(self.raw_candidate_trace_ids),
            "canonical_output_hash": self.canonical_output_hash,
            "raw_grid_fidelity_passed": self.raw_grid_fidelity_passed,
            "execution_status": self.execution_status,
            "rank_tuple": list(self.rank_tuple()),
        }


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grid_to_list(grid: Grid | None) -> list[list[int]] | None:
    if grid is None:
        return None
    return [list(row) for row in grid]


def render_grid(grid: Grid | list[list[int]] | None) -> str:
    if grid is None:
        return "<none>"
    rows = grid if isinstance(grid, list) else grid_to_list(grid)
    return "\n".join("".join(str(int(value)) for value in row) for row in rows)


def cell_hamming_loss(candidate: Grid | None, target: Grid) -> tuple[int, int, bool]:
    if candidate is None:
        height, width = grid_shape(target)
        penalty = height * int(width or 0)
        return 1, penalty, False
    candidate_shape = grid_shape(candidate)
    target_shape = grid_shape(target)
    if candidate_shape != target_shape:
        height, width = target_shape
        penalty = height * int(width or 0)
        return 1, penalty, False
    loss = sum(1 for y, row in enumerate(target) for x, value in enumerate(row) if int(candidate[y][x]) != int(value))
    return 0, loss, loss == 0


def support_score(
    *,
    candidate_id: str,
    source_expert: str,
    frozen_catalog_hash: str,
    query_index: int,
    output_grid: Grid | None,
    support_predictions: Iterable[Grid | None],
    support_targets: Iterable[Grid],
    raw_grid_fidelity_passed: bool,
    execution_status: str,
) -> Candidate:
    mismatch_count = 0
    hamming_loss = 0
    failed_count = 0
    for predicted, target in zip(support_predictions, support_targets):
        shape_mismatch, loss, exact = cell_hamming_loss(predicted, target)
        mismatch_count += shape_mismatch
        hamming_loss += loss
        failed_count += int(not exact)
    output_shape = grid_shape(output_grid) if output_grid is not None else None
    output_hash = canonical_grid_sha256(output_grid) if output_grid is not None else None
    return Candidate(
        source_expert=source_expert,
        frozen_candidate_id=candidate_id,
        frozen_catalog_hash=frozen_catalog_hash,
        query_index=query_index,
        output_grid=output_grid,
        output_shape=output_shape,
        visible_support_count=len(list(support_targets)) if not isinstance(support_targets, list) else len(support_targets),
        support_shape_mismatch_count=mismatch_count,
        support_cell_hamming_loss=hamming_loss,
        support_failed_pair_count=failed_count,
        raw_candidate_trace_ids=(candidate_id,),
        canonical_output_hash=output_hash,
        raw_grid_fidelity_passed=raw_grid_fidelity_passed,
        execution_status=execution_status,
    )


def deduplicate_and_rank(candidates: Iterable[Candidate]) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.execution_status != "OK" or not candidate.raw_grid_fidelity_passed or candidate.output_grid is None:
            continue
        grouped[candidate.canonical_output_hash or ""].append(candidate)
    representatives: list[Candidate] = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item.rank_tuple())
        best = rows[0]
        trace_ids = tuple(sorted({trace for row in rows for trace in row.raw_candidate_trace_ids}))
        canonical_ids = " + ".join(trace_ids)
        representatives.append(
            replace(
                best,
                frozen_candidate_id=best.frozen_candidate_id,
                raw_candidate_trace_ids=trace_ids,
            )
        )
    return sorted(representatives, key=lambda item: item.rank_tuple())


def exact_support_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.support_shape_mismatch_count == 0
        and candidate.support_cell_hamming_loss == 0
        and candidate.support_failed_pair_count == 0
    ]


def target_rank(candidates: list[Candidate], target: Grid) -> int | None:
    target_hash = canonical_grid_sha256(target)
    for index, candidate in enumerate(candidates, start=1):
        if candidate.canonical_output_hash == target_hash and candidate.output_grid == target:
            return index
    return None


def recall_at_k(candidates: list[Candidate], target: Grid, k: int) -> bool:
    if not candidates:
        return False
    effective_k = min(k, len(candidates))
    return target_rank(candidates[:effective_k], target) is not None


def monotonic_recalls(candidates: list[Candidate], target: Grid) -> dict[str, bool]:
    return {f"recall_at_{k}": recall_at_k(candidates, target, k) for k in K_VALUES}


def merge_candidate_pools(candidates_by_expert: Iterable[list[Candidate]]) -> list[Candidate]:
    return deduplicate_and_rank(candidate for rows in candidates_by_expert for candidate in rows)


def histogram(values: Iterable[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items(), key=lambda item: (int(item[0]), item[0])))


def numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {"count": len(values), "mean": sum(values) / len(values), "median": statistics.median(values)}


def candidate_ranking_spec() -> dict[str, Any]:
    return {
        "K_values": K_VALUES,
        "rank_tuple_order": [
            "support_shape_mismatch_count ascending",
            "support_cell_hamming_loss ascending",
            "support_failed_pair_count ascending",
            "source_expert canonical name ascending",
            "frozen_candidate_id canonical lexical order ascending",
            "canonical_output_hash ascending",
        ],
        "shape_mismatch_penalty": SHAPE_MISMATCH_PENALTY_KIND,
        "ranking_inputs": "visible support pairs and frozen candidate metadata only",
        "forbidden_ranking_inputs": [
            "hidden_LOO_train_output",
            "sealed_test_output",
            "task_id",
            "manual_task_label",
            "post_hoc_performance",
        ],
        "candidate_budget_counts": "unique output grids after canonical output hash deduplication",
    }
