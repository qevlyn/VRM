from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import compiled_topology_r5a2c_r2_data as topology_data
from . import compiled_topology_r5a2c_r2_executor as topology_executor
from . import transport_recolor_r5a0v2_oracles as v2_oracles
from .r5a2d_r1_raw_grid_fidelity import Grid, fidelity_json, grid_shape
from .r5a2d_r2_candidate_ranking import Candidate, deduplicate_and_rank, exact_support_candidates, support_score
from .r5a2d_r2_candidate_surface_audit import (
    candidate_surface_report,
    topology_candidate_id,
    topology_catalog_payload,
    v2_catalog_payload,
)
from .r5a2d_r2_candidate_ranking import stable_hash

OK = "OK"
OUT_GRID = "OUT_OF_GRID_DOMAIN"
OUT_PALETTE = "OUT_OF_PALETTE_DOMAIN"
OUT_SUPPORT = "OUT_OF_SUPPORT_COUNT_DOMAIN"
RAW_REJECTED = "RAW_INTERFACE_REJECTED"
RUNTIME_ERROR = "RUNTIME_ERROR"
ENUM_UNAVAILABLE = "CANDIDATE_ENUMERATION_UNAVAILABLE"
TOPOLOGY_LOO_INFEASIBLE = "TOPOLOGY_LOO_INFEASIBLE_SUPPORT_CONTRACT"


@dataclass(frozen=True)
class FoldVisibleInputs:
    task_id: str
    fold_index: int
    query_input: Grid
    support_inputs: tuple[Grid, ...]
    support_outputs: tuple[Grid, ...]


def _palette_ok(grid: Grid, values: set[int]) -> bool:
    return all(int(value) in values for row in grid for value in row)


def _grid_size_ok(grid: Grid, size: int) -> bool:
    return grid_shape(grid) == (size, size)


def v2_domain_status(fold: FoldVisibleInputs) -> str:
    grids = list(fold.support_inputs) + list(fold.support_outputs) + [fold.query_input]
    if not all(_grid_size_ok(grid, v2_oracles.GRID_SIZE) for grid in grids):
        return OUT_GRID
    if not all(_palette_ok(grid, set(range(v2_oracles.COLOR_COUNT))) for grid in grids):
        return OUT_PALETTE
    return OK


def topology_domain_status(fold: FoldVisibleInputs) -> str:
    grids = list(fold.support_inputs) + list(fold.support_outputs) + [fold.query_input]
    if not all(_grid_size_ok(grid, topology_data.GRID_SIZE) for grid in grids):
        return OUT_GRID
    if not all(_palette_ok(grid, set(topology_data.PALETTE)) for grid in grids):
        return OUT_PALETTE
    if len(fold.support_inputs) != topology_data.SUPPORT_PAIR_COUNT:
        return OUT_SUPPORT
    if len(fold.support_outputs) != topology_data.SUPPORT_PAIR_COUNT:
        return OUT_SUPPORT
    return OK


def _fidelity_records(fold: FoldVisibleInputs, *, expert: str) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    grids = list(fold.support_inputs) + list(fold.support_outputs) + [fold.query_input]
    for index, grid in enumerate(grids):
        rows.append(fidelity_json(grid, grid, context=f"{expert}:task={fold.task_id}:fold={fold.fold_index}:grid={index}"))
    return rows, all(row["passed"] for row in rows)


def _v2_catalog_hash() -> str:
    return stable_hash(v2_catalog_payload())


def _topology_catalog_hash() -> str:
    return stable_hash(topology_catalog_payload())


def enumerate_v2_candidates(fold: FoldVisibleInputs) -> dict[str, Any]:
    surface = candidate_surface_report()["surfaces"][0]
    if not surface["candidate_enumeration_available"]:
        return _empty_result("V2_transport_recolor", ENUM_UNAVAILABLE, [], [])
    domain = v2_domain_status(fold)
    fidelity_records: list[dict[str, Any]] = []
    if domain != OK:
        return _empty_result("V2_transport_recolor", domain, fidelity_records, [])
    fidelity_records, fidelity_passed = _fidelity_records(fold, expert="V2_transport_recolor")
    if not fidelity_passed:
        return _empty_result("V2_transport_recolor", RAW_REJECTED, fidelity_records, [])

    candidates: list[Candidate] = []
    catalog_hash = _v2_catalog_hash()
    for candidate_id in v2_oracles.SEALED_CLASSES:
        try:
            query_output = v2_oracles.apply_sequence_to_grid(fold.query_input, candidate_id)
            support_predictions = [v2_oracles.apply_sequence_to_grid(grid, candidate_id) for grid in fold.support_inputs]
            status = OK
        except Exception:
            query_output = None
            support_predictions = [None for _ in fold.support_inputs]
            status = RUNTIME_ERROR
        candidates.append(
            support_score(
                candidate_id=candidate_id,
                source_expert="V2_transport_recolor",
                frozen_catalog_hash=catalog_hash,
                query_index=fold.fold_index,
                output_grid=query_output,
                support_predictions=support_predictions,
                support_targets=list(fold.support_outputs),
                raw_grid_fidelity_passed=fidelity_passed,
                execution_status=status,
            )
        )
    return _summarize_candidates("V2_transport_recolor", OK, fidelity_records, candidates)


def enumerate_topology_candidates(fold: FoldVisibleInputs) -> dict[str, Any]:
    surface = candidate_surface_report()["surfaces"][1]
    if not surface["candidate_enumeration_available"]:
        return _empty_result("R5A-2C-R2_compiled_topology", ENUM_UNAVAILABLE, [], [])
    domain = topology_domain_status(fold)
    fidelity_records: list[dict[str, Any]] = []
    if domain == OUT_SUPPORT:
        return _empty_result("R5A-2C-R2_compiled_topology", TOPOLOGY_LOO_INFEASIBLE, fidelity_records, [])
    if domain != OK:
        return _empty_result("R5A-2C-R2_compiled_topology", domain, fidelity_records, [])
    fidelity_records, fidelity_passed = _fidelity_records(fold, expert="R5A-2C-R2_compiled_topology")
    if not fidelity_passed:
        return _empty_result("R5A-2C-R2_compiled_topology", RAW_REJECTED, fidelity_records, [])

    candidates: list[Candidate] = []
    catalog_hash = _topology_catalog_hash()
    for hypothesis in topology_data.candidate_hypotheses():
        payload = hypothesis.to_json()
        candidate_id = topology_candidate_id(payload)
        try:
            query_output = topology_executor.execute_hypothesis(fold.query_input, hypothesis)
            support_predictions = [topology_executor.execute_hypothesis(grid, hypothesis) for grid in fold.support_inputs]
            status = OK
        except Exception:
            query_output = None
            support_predictions = [None for _ in fold.support_inputs]
            status = RUNTIME_ERROR
        candidates.append(
            support_score(
                candidate_id=candidate_id,
                source_expert="R5A-2C-R2_compiled_topology",
                frozen_catalog_hash=catalog_hash,
                query_index=fold.fold_index,
                output_grid=query_output,
                support_predictions=support_predictions,
                support_targets=list(fold.support_outputs),
                raw_grid_fidelity_passed=fidelity_passed,
                execution_status=status,
            )
        )
    return _summarize_candidates("R5A-2C-R2_compiled_topology", OK, fidelity_records, candidates)


def _empty_result(expert: str, status: str, fidelity_records: list[dict[str, Any]], candidates: list[Candidate]) -> dict[str, Any]:
    return {
        "source_expert": expert,
        "status": status,
        "fidelity_records": fidelity_records,
        "raw_candidates": [candidate.to_json() for candidate in candidates],
        "ranked_all_catalog_candidates": [],
        "ranked_exact_support_candidates": [],
        "raw_executable_candidate_count": 0,
        "unique_output_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "deduplication_ratio": None,
        "candidate_source_breakdown": {},
        "original_frozen_executor_top1_candidate": None,
    }


def _summarize_candidates(expert: str, status: str, fidelity_records: list[dict[str, Any]], candidates: list[Candidate]) -> dict[str, Any]:
    executable = [candidate for candidate in candidates if candidate.execution_status == OK and candidate.output_grid is not None and candidate.raw_grid_fidelity_passed]
    ranked_all = deduplicate_and_rank(executable)
    ranked_exact = deduplicate_and_rank(exact_support_candidates(executable))
    duplicate_count = max(len(executable) - len(ranked_all), 0)
    original = _original_top1_candidate(expert, ranked_exact)
    return {
        "source_expert": expert,
        "status": status,
        "fidelity_records": fidelity_records,
        "raw_candidates": [candidate.to_json() for candidate in candidates],
        "ranked_all_catalog_candidates": [candidate.to_json() for candidate in ranked_all],
        "ranked_exact_support_candidates": [candidate.to_json() for candidate in ranked_exact],
        "raw_executable_candidate_count": len(executable),
        "unique_output_candidate_count": len(ranked_all),
        "duplicate_candidate_count": duplicate_count,
        "deduplication_ratio": duplicate_count / len(executable) if executable else None,
        "candidate_source_breakdown": {expert: len(executable)},
        "original_frozen_executor_top1_candidate": original.to_json() if original is not None else None,
    }


def _original_top1_candidate(expert: str, ranked_exact: list[Candidate]) -> Candidate | None:
    if len(ranked_exact) != 1:
        return None
    return ranked_exact[0]
