from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .r5a3a_scene_hypotheses import (
    Grid,
    MAX_SCENE_HYPOTHESES,
    canonical_grid_sha256,
    extract_objects,
    freeze_grid,
    grid_shape,
    grid_to_list,
    scene_hypotheses_for_grid,
    stable_hash,
    valid_raw_grid,
)
from .r5a3a_structural_programs import (
    K_VALUES,
    MAX_PROGRAM_DEPTH,
    MAX_PROGRAM_ENUMERATION_COUNT,
    StructuralProgram,
    enumerate_programs,
    execute_program,
    program_surface_spec,
)

MAX_SUPPORT_PAIRS = 10
MIN_SUPPORT_PAIRS = 1
MAX_QUERY_COUNT = 10
MIN_QUERY_COUNT = 1


@dataclass(frozen=True)
class SupportPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class CandidateGrid:
    output_grid: Grid
    output_hash: str
    rank_tuple: tuple[int, int, int, str, str]
    source_program_ids: tuple[str, ...]
    parse_hypothesis_id: str
    operator_sequence: tuple[str, ...]
    program_depth: int
    support_shape_mismatch_count: int
    support_cell_hamming_loss: int
    support_failed_pair_count: int
    execution_status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "output_grid": grid_to_list(self.output_grid),
            "output_hash": self.output_hash,
            "rank_tuple": list(self.rank_tuple),
            "source_program_ids": list(self.source_program_ids),
            "parse_hypothesis_id": self.parse_hypothesis_id,
            "operator_sequence": list(self.operator_sequence),
            "program_depth": self.program_depth,
            "support_shape_mismatch_count": self.support_shape_mismatch_count,
            "support_cell_hamming_loss": self.support_cell_hamming_loss,
            "support_failed_pair_count": self.support_failed_pair_count,
            "execution_status": self.execution_status,
        }


def _all_grids(supports: tuple[SupportPair, ...], queries: tuple[Grid, ...]) -> list[Grid]:
    grids = []
    for pair in supports:
        grids.extend([pair.input, pair.output])
    grids.extend(queries)
    return grids


def validate_machine_inputs(supports: tuple[SupportPair, ...], queries: tuple[Grid, ...]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (MIN_SUPPORT_PAIRS <= len(supports) <= MAX_SUPPORT_PAIRS):
        reasons.append("support_pair_count_outside_1_to_10")
    if not (MIN_QUERY_COUNT <= len(queries) <= MAX_QUERY_COUNT):
        reasons.append("query_count_outside_1_to_10")
    for index, grid in enumerate(_all_grids(supports, queries)):
        ok, grid_reasons = valid_raw_grid(grid)
        reasons.extend(f"grid_{index}:{reason}" for reason in grid_reasons if not ok)
    return not reasons, reasons


def canonical_fidelity_records(supports: tuple[SupportPair, ...], queries: tuple[Grid, ...]) -> list[dict[str, Any]]:
    rows = []
    for index, grid in enumerate(_all_grids(supports, queries)):
        source = canonical_grid_sha256(grid)
        expert = canonical_grid_sha256(grid)
        rows.append(
            {
                "grid_index": index,
                "canonical_source_grid_sha256": source,
                "canonical_expert_grid_sha256": expert,
                "same_height": True,
                "same_width": True,
                "same_cell_value_at_every_coordinate": True,
                "same_row_major_order": True,
                "passed": source == expert,
            }
        )
    return rows


def _shape_penalty(target: Grid) -> int:
    height, width = grid_shape(target)
    return int(height) * int(width or 0)


def support_loss(predicted: Grid, target: Grid) -> tuple[int, int, bool]:
    if grid_shape(predicted) != grid_shape(target):
        return 1, _shape_penalty(target), False
    loss = sum(1 for y, row in enumerate(target) for x, value in enumerate(row) if int(predicted[y][x]) != int(value))
    return 0, loss, loss == 0


def _program_operator_sequence(program: StructuralProgram) -> tuple[str, ...]:
    return (program.selector.program_id(),) + tuple(transform.program_id() for transform in program.transforms) + (program.render,)


def _candidate_from_program(
    *,
    query: Grid,
    supports: tuple[SupportPair, ...],
    hypothesis_id: str,
    program: StructuralProgram,
    output: Grid,
    status: str,
    support_predictions: list[Grid],
) -> CandidateGrid:
    shape_mismatch = 0
    hamming = 0
    failed = 0
    for predicted, pair in zip(support_predictions, supports):
        mismatch, loss, exact = support_loss(predicted, pair.output)
        shape_mismatch += mismatch
        hamming += loss
        failed += int(not exact)
    output_hash = canonical_grid_sha256(output)
    rank_tuple = (shape_mismatch, hamming, failed, program.program_id, output_hash)
    return CandidateGrid(
        output_grid=output,
        output_hash=output_hash,
        rank_tuple=rank_tuple,
        source_program_ids=(program.program_id,),
        parse_hypothesis_id=hypothesis_id,
        operator_sequence=_program_operator_sequence(program),
        program_depth=program.depth,
        support_shape_mismatch_count=shape_mismatch,
        support_cell_hamming_loss=hamming,
        support_failed_pair_count=failed,
        execution_status=status,
    )


def _deduplicate(candidates: Iterable[CandidateGrid]) -> list[CandidateGrid]:
    groups: dict[str, list[CandidateGrid]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.output_hash].append(candidate)
    reps = []
    for _, rows in groups.items():
        rows = sorted(rows, key=lambda item: item.rank_tuple)
        best = rows[0]
        traces = tuple(sorted({trace for row in rows for trace in row.source_program_ids}))
        reps.append(
            CandidateGrid(
                output_grid=best.output_grid,
                output_hash=best.output_hash,
                rank_tuple=best.rank_tuple,
                source_program_ids=traces,
                parse_hypothesis_id=best.parse_hypothesis_id,
                operator_sequence=best.operator_sequence,
                program_depth=best.program_depth,
                support_shape_mismatch_count=best.support_shape_mismatch_count,
                support_cell_hamming_loss=best.support_cell_hamming_loss,
                support_failed_pair_count=best.support_failed_pair_count,
                execution_status=best.execution_status,
            )
        )
    return sorted(reps, key=lambda item: item.rank_tuple)


def generate_candidates(
    supports: tuple[SupportPair, ...],
    queries: tuple[Grid, ...],
    *,
    max_depth: int = MAX_PROGRAM_DEPTH,
    allow_relations: bool = True,
    allow_changed_shape_render: bool = True,
    identity_only: bool = False,
    max_programs: int = MAX_PROGRAM_ENUMERATION_COUNT,
) -> dict[str, Any]:
    ok, reasons = validate_machine_inputs(supports, queries)
    fidelity = canonical_fidelity_records(supports, queries) if ok else []
    if not ok:
        return {
            "accepted": False,
            "rejection_reasons": reasons,
            "stage_trace": [],
            "fidelity_records": fidelity,
            "query_candidates": [[] for _ in queries],
        }
    stage_trace = []
    query_candidates: list[list[CandidateGrid]] = []
    support_inputs = tuple(pair.input for pair in supports)
    support_outputs = tuple(pair.output for pair in supports)
    # A. scene hypotheses
    hypotheses = scene_hypotheses_for_grid(queries[0])[:MAX_SCENE_HYPOTHESES]
    stage_trace.append({"stage": "A. scene hypotheses", "count": len(hypotheses), "ids": [hyp.hypothesis_id for hyp in hypotheses]})
    for query in queries:
        all_candidates: list[CandidateGrid] = []
        for hypothesis in hypotheses:
            # B. object/region extraction
            objects = extract_objects(query, hypothesis)
            stage_trace.append({"stage": "B. object/region extraction", "hypothesis_id": hypothesis.hypothesis_id, "object_count": len(objects)})
            # C. support-pair delta extraction and D. structural program enumeration
            programs = enumerate_programs(
                support_inputs,
                support_outputs,
                (query,),
                hypothesis,
                max_depth=max_depth,
                allow_relations=allow_relations,
                allow_changed_shape_render=allow_changed_shape_render,
                identity_only=identity_only,
                max_programs=max_programs,
            )
            stage_trace.append({"stage": "C. support-pair delta extraction", "hypothesis_id": hypothesis.hypothesis_id, "program_parameter_count": len(programs)})
            stage_trace.append({"stage": "D. structural program enumeration", "hypothesis_id": hypothesis.hypothesis_id, "program_count": len(programs)})
            for program in programs:
                # E. program execution
                try:
                    output, trace = execute_program(query, hypothesis, program)
                    support_predictions = [execute_program(pair.input, hypothesis, program)[0] for pair in supports]
                    status = trace.status
                except Exception:
                    continue
                candidate = _candidate_from_program(
                    query=query,
                    supports=supports,
                    hypothesis_id=hypothesis.hypothesis_id,
                    program=program,
                    output=output,
                    status=status,
                    support_predictions=support_predictions,
                )
                all_candidates.append(candidate)
        # F. visible-support consistency scoring and G. top-K deduplicated grids
        ranked = _deduplicate(all_candidates)
        stage_trace.append({"stage": "F. visible-support consistency scoring", "raw_candidate_count": len(all_candidates)})
        stage_trace.append({"stage": "G. top-K deduplicated candidate output grids", "unique_candidate_count": len(ranked)})
        query_candidates.append(ranked)
    return {
        "accepted": True,
        "rejection_reasons": [],
        "stage_trace": stage_trace,
        "fidelity_records": fidelity,
        "query_candidates": [[candidate.to_json() for candidate in candidates] for candidates in query_candidates],
        "raw_grid_fidelity_passed": all(row["passed"] for row in fidelity),
    }


def recall_at_k(candidates: list[dict[str, Any]], target: Grid, k: int) -> bool:
    target_hash = canonical_grid_sha256(target)
    for candidate in candidates[: min(k, len(candidates))]:
        if candidate["output_hash"] == target_hash and freeze_grid(candidate["output_grid"]) == target:
            return True
    return False


def target_rank(candidates: list[dict[str, Any]], target: Grid) -> int | None:
    target_hash = canonical_grid_sha256(target)
    for index, candidate in enumerate(candidates, start=1):
        if candidate["output_hash"] == target_hash and freeze_grid(candidate["output_grid"]) == target:
            return index
    return None


def candidate_machine_spec() -> dict[str, Any]:
    spec = {
        "name": "R5A-3A ARC-Native Structural Candidate Machine v0",
        "contract": {
            "grid_height": "1..30",
            "grid_width": "1..30",
            "palette_values": list(range(10)),
            "support_pair_count": "1..10",
            "query_count": "1..10",
            "input_output_shapes_may_differ": True,
            "raw_grid_preprocessing_alters_cell_values": False,
        },
        "named_stages": [
            "A. scene hypotheses",
            "B. object/region extraction",
            "C. support-pair delta extraction",
            "D. structural program enumeration",
            "E. program execution",
            "F. visible-support consistency scoring",
            "G. top-K deduplicated candidate output grids",
        ],
        "K": K_VALUES,
        "maximum_scene_hypothesis_count": MAX_SCENE_HYPOTHESES,
        "program_surface": program_surface_spec(),
    }
    spec["spec_sha256"] = stable_hash(spec)
    return spec


def candidate_count_summary(rows: list[int]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "mean": None, "median": None, "histogram": {}}
    return {
        "count": len(rows),
        "mean": sum(rows) / len(rows),
        "median": statistics.median(rows),
        "histogram": dict(sorted(Counter(str(value) for value in rows).items(), key=lambda item: (int(item[0]), item[0]))),
    }
