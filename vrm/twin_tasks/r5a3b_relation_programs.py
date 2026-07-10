from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .r5a3a_arc_native_candidate_machine import SupportPair, support_loss
from .r5a3a_scene_hypotheses import (
    Grid,
    SceneHypothesis,
    canonical_grid_sha256,
    freeze_grid,
    grid_shape,
    grid_to_list,
    stable_hash,
)
from .r5a3a_structural_programs import (
    StructuralProgram,
    enumerate_programs,
    execute_program,
)
from .r5a3b_constraint_acquisition import ConstraintModel, evaluate_candidate_constraints, parse_state_for_grid
from .r5a3b_object_graph import EDGE_PREDICATES, GraphEdge, ObjectGraph, build_object_graph
from .r5a3b_parse_hypotheses import MAX_PARSE_HYPOTHESES, PARSE_MENU, ParseState, enumerate_parse_states

MAX_RAW_PROGRAMS_PER_PARSE = 8
MAX_RAW_CANDIDATES_PER_FOLD = 32
TOP_K = [1, 4, 16, 64, 128]
MAX_PROGRAM_DEPTH = 2

CONDITIONS = (
    "OBJECT_ONLY",
    "RELATION_NO_PRUNE",
    "RELATION_SOFT_CONSTRAINT",
    "RELATION_HARD_CONSTRAINT",
    "MATCHED_RANDOM_PRUNE",
)

MATCHED_RANDOM_SALTS = (
    "R5A3B_P0_MATCHED_RANDOM_PRUNE_SALT_A",
    "R5A3B_P0_MATCHED_RANDOM_PRUNE_SALT_B",
    "R5A3B_P0_MATCHED_RANDOM_PRUNE_SALT_C",
)

RELATION_PROGRAM_PREDICATE_PRIORITY = (
    "nearest",
    "aligned_row",
    "aligned_column",
    "touching",
    "contains",
    "same_shape",
    "same_color",
    "same_size",
    "relative_direction",
    "border_relation",
)

RELATION_PROGRAM_ACTIONS = (
    "keep_relation_endpoints",
    "recolor_target_to_source",
    "recolor_source_to_target",
    "delete_target",
    "copy_source_by_relation_vector",
    "translate_target_by_relation_vector",
    "recolor_target_constant",
)

RENDER_MODES = ("same_canvas",)


@dataclass(frozen=True)
class RelationProgram:
    program_id: str
    predicate: str
    predicate_value: Any
    action: str
    constant_color: int | None
    depth: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "predicate": self.predicate,
            "predicate_value": self.predicate_value,
            "action": self.action,
            "constant_color": self.constant_color,
            "depth": self.depth,
            "render": "same_canvas",
        }


@dataclass(frozen=True)
class R5A3BCandidate:
    output_grid: Grid
    output_hash: str
    candidate_id: str
    raw_ordinal: int
    parse_hypothesis_id: str
    parse_background_color: int
    parse_connectivity: int
    source_program_ids: tuple[str, ...]
    operator_sequence: tuple[str, ...]
    program_family: str
    relation_predicates_used: tuple[str, ...]
    program_depth: int
    support_shape_mismatch_count: int
    support_cell_hamming_loss: int
    support_failed_pair_count: int
    execution_status: str
    provenance: dict[str, Any]
    execution_trace: dict[str, Any]

    @property
    def base_rank_tuple(self) -> tuple[int, int, int, str, str]:
        return (
            self.support_shape_mismatch_count,
            self.support_cell_hamming_loss,
            self.support_failed_pair_count,
            self.candidate_id,
            self.output_hash,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "output_grid": grid_to_list(self.output_grid),
            "output_hash": self.output_hash,
            "candidate_id": self.candidate_id,
            "raw_ordinal": self.raw_ordinal,
            "parse_hypothesis_id": self.parse_hypothesis_id,
            "parse_background_color": self.parse_background_color,
            "parse_connectivity": self.parse_connectivity,
            "source_program_ids": list(self.source_program_ids),
            "operator_sequence": list(self.operator_sequence),
            "program_family": self.program_family,
            "relation_predicates_used": list(self.relation_predicates_used),
            "program_depth": self.program_depth,
            "support_shape_mismatch_count": self.support_shape_mismatch_count,
            "support_cell_hamming_loss": self.support_cell_hamming_loss,
            "support_failed_pair_count": self.support_failed_pair_count,
            "execution_status": self.execution_status,
            "provenance": self.provenance,
            "execution_trace": self.execution_trace,
            "base_rank_tuple": list(self.base_rank_tuple),
        }


def relation_program_surface_spec() -> dict[str, Any]:
    payload = {
        "parse_menu": PARSE_MENU,
        "max_parse_hypotheses": MAX_PARSE_HYPOTHESES,
        "max_raw_programs_per_parse": MAX_RAW_PROGRAMS_PER_PARSE,
        "max_raw_candidates_per_fold": MAX_RAW_CANDIDATES_PER_FOLD,
        "top_k": TOP_K,
        "max_program_depth": MAX_PROGRAM_DEPTH,
        "relation_edge_predicates": EDGE_PREDICATES,
        "relation_program_predicate_priority": RELATION_PROGRAM_PREDICATE_PRIORITY,
        "relation_program_actions": RELATION_PROGRAM_ACTIONS,
        "render_modes": RENDER_MODES,
        "forbidden_interfaces": [
            "task_id_runtime_input",
            "arc_query_test_output_access",
            "learned_router",
            "learned_verifier",
            "neural_model_training",
            "explicit_task_specific_python_programs",
            "raw_grid_crop_pad_resize_remap_preprocessing",
        ],
    }
    payload["surface_sha256"] = stable_hash(payload)
    return payload


def _hypothesis(parse_state: ParseState) -> SceneHypothesis:
    return SceneHypothesis(f"bg={parse_state.background_color}|conn={parse_state.connectivity}", parse_state.background_color, parse_state.connectivity)


def _program_operator_sequence(program: StructuralProgram) -> tuple[str, ...]:
    return (program.selector.program_id(),) + tuple(transform.program_id() for transform in program.transforms) + (program.render,)


def _colors_from_grids(grids: Iterable[Grid]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for grid in grids for row in grid for value in row}))


def _score_support_predictions(predictions: list[Grid], supports: tuple[SupportPair, ...]) -> tuple[int, int, int]:
    shape_mismatch = 0
    hamming = 0
    failed = 0
    for predicted, pair in zip(predictions, supports):
        mismatch, loss, exact = support_loss(predicted, pair.output)
        shape_mismatch += mismatch
        hamming += loss
        failed += int(not exact)
    return shape_mismatch, hamming, failed


def _candidate_from_structural(
    *,
    query: Grid,
    supports: tuple[SupportPair, ...],
    parse_state: ParseState,
    program: StructuralProgram,
    raw_ordinal: int,
) -> R5A3BCandidate | None:
    hypothesis = _hypothesis(parse_state)
    try:
        output, trace = execute_program(query, hypothesis, program)
        support_predictions = [execute_program(pair.input, hypothesis, program)[0] for pair in supports]
    except Exception as exc:
        return None
    shape_mismatch, hamming, failed = _score_support_predictions(support_predictions, supports)
    output_hash = canonical_grid_sha256(output)
    candidate_id = stable_hash(
        {
            "family": "object",
            "parse": parse_state.parse_id,
            "program": program.program_id,
            "output": output_hash,
            "ordinal": raw_ordinal,
        }
    )[:24]
    return R5A3BCandidate(
        output_grid=output,
        output_hash=output_hash,
        candidate_id=candidate_id,
        raw_ordinal=raw_ordinal,
        parse_hypothesis_id=parse_state.parse_id,
        parse_background_color=parse_state.background_color,
        parse_connectivity=parse_state.connectivity,
        source_program_ids=(program.program_id,),
        operator_sequence=_program_operator_sequence(program),
        program_family="object",
        relation_predicates_used=tuple(),
        program_depth=program.depth,
        support_shape_mismatch_count=shape_mismatch,
        support_cell_hamming_loss=hamming,
        support_failed_pair_count=failed,
        execution_status=trace.status,
        provenance={
            "parse_id": parse_state.parse_id,
            "source": "R5A3A_structural_program_enumerator_with_relations_disabled",
            "program": program.to_json(),
        },
        execution_trace=trace.to_json(),
    )


def _edge_predicate_payload(edge: GraphEdge, predicate: str) -> Any:
    if predicate in {"relative_direction", "border_relation"}:
        return edge.predicates[predicate]
    return True


def _edge_matches(edge: GraphEdge, program: RelationProgram) -> bool:
    if program.predicate == "nearest" and edge.predicates.get("nearest_is_tied"):
        return False
    if program.predicate in {"relative_direction", "border_relation"}:
        return edge.predicates.get(program.predicate) == program.predicate_value
    return bool(edge.predicates.get(program.predicate)) is True


def enumerate_relation_programs(
    graph: ObjectGraph,
    visible_output_colors: tuple[int, ...],
    *,
    max_programs: int,
) -> tuple[RelationProgram, ...]:
    rows: list[RelationProgram] = []
    seen = set()
    constant_color = visible_output_colors[0] if visible_output_colors else None
    for edge in graph.edges:
        for predicate in RELATION_PROGRAM_PREDICATE_PRIORITY:
            if predicate == "nearest" and edge.predicates.get("nearest_is_tied"):
                continue
            if predicate not in edge.predicates:
                continue
            if predicate not in {"relative_direction", "border_relation"} and not bool(edge.predicates.get(predicate)):
                continue
            payload = _edge_predicate_payload(edge, predicate)
            for action in RELATION_PROGRAM_ACTIONS:
                if action == "recolor_target_constant" and constant_color is None:
                    continue
                program_id = f"relation={predicate}:{payload}|action={action}|render=same_canvas"
                if program_id in seen:
                    continue
                seen.add(program_id)
                rows.append(RelationProgram(program_id, predicate, payload, action, constant_color))
                if len(rows) >= max_programs:
                    return tuple(rows)
    return tuple(rows)


def _node_by_id(graph: ObjectGraph) -> dict[str, Any]:
    return {node.node_id: node.attrs for node in graph.nodes}


def _put_cell(canvas: list[list[int]], x: int, y: int, color: int, trace: dict[str, Any]) -> None:
    height = len(canvas)
    width = len(canvas[0]) if height else 0
    if not (0 <= x < width and 0 <= y < height):
        trace["dropped_out_of_bounds"].append([x, y, int(color)])
        return
    if int(canvas[y][x]) != int(color):
        trace["overwritten_destinations"].append([x, y, int(canvas[y][x]), int(color)])
    canvas[y][x] = int(color)


def _vector(source: dict[str, Any], target: dict[str, Any]) -> tuple[int, int]:
    sx, sy = source["top_left_position"]
    tx, ty = target["top_left_position"]
    return int(tx) - int(sx), int(ty) - int(sy)


def _execute_relation_program(
    grid: Grid,
    *,
    background_color: int,
    connectivity: int,
    program: RelationProgram,
) -> tuple[Grid, dict[str, Any]]:
    parse_state = parse_state_for_grid(grid, background_color, connectivity)
    graph = build_object_graph(parse_state)
    nodes = _node_by_id(graph)
    selected_edges = [edge for edge in graph.edges if _edge_matches(edge, program)]
    canvas = [list(row) for row in grid]
    trace: dict[str, Any] = {
        "status": "OK",
        "selected_edge_count": len(selected_edges),
        "selected_edges": [edge.to_json() for edge in selected_edges[:16]],
        "dropped_out_of_bounds": [],
        "overwritten_destinations": [],
        "colliding_destinations": [],
    }
    destination_counts: Counter[tuple[int, int]] = Counter()
    for edge in selected_edges:
        source = nodes[edge.source_id]
        target = nodes[edge.target_id]
        dx, dy = _vector(source, target)
        if program.action == "keep_relation_endpoints":
            continue
        if program.action == "delete_target":
            for x, y in target["cells"]:
                _put_cell(canvas, int(x), int(y), background_color, trace)
            continue
        if program.action == "recolor_target_to_source":
            color = int(source["color"])
            for x, y in target["cells"]:
                _put_cell(canvas, int(x), int(y), color, trace)
            continue
        if program.action == "recolor_source_to_target":
            color = int(target["color"])
            for x, y in source["cells"]:
                _put_cell(canvas, int(x), int(y), color, trace)
            continue
        if program.action == "recolor_target_constant":
            assert program.constant_color is not None
            for x, y in target["cells"]:
                _put_cell(canvas, int(x), int(y), int(program.constant_color), trace)
            continue
        if program.action == "copy_source_by_relation_vector":
            for x, y in source["cells"]:
                nx, ny = int(x) + dx, int(y) + dy
                destination_counts[(nx, ny)] += 1
                _put_cell(canvas, nx, ny, int(source["color"]), trace)
            continue
        if program.action == "translate_target_by_relation_vector":
            for x, y in target["cells"]:
                _put_cell(canvas, int(x), int(y), background_color, trace)
            for x, y in target["cells"]:
                nx, ny = int(x) + dx, int(y) + dy
                destination_counts[(nx, ny)] += 1
                _put_cell(canvas, nx, ny, int(target["color"]), trace)
            continue
    trace["colliding_destinations"] = [[x, y, count] for (x, y), count in sorted(destination_counts.items()) if count > 1]
    if selected_edges and trace["dropped_out_of_bounds"]:
        trace["status"] = "DROPPED_OUT_OF_BOUNDS"
    elif not selected_edges:
        trace["status"] = "EMPTY_RELATION_SELECTION"
    return freeze_grid(canvas), trace


def _candidate_from_relation(
    *,
    query: Grid,
    supports: tuple[SupportPair, ...],
    parse_state: ParseState,
    program: RelationProgram,
    raw_ordinal: int,
) -> R5A3BCandidate | None:
    try:
        output, trace = _execute_relation_program(
            query,
            background_color=parse_state.background_color,
            connectivity=parse_state.connectivity,
            program=program,
        )
        support_predictions = [
            _execute_relation_program(
                pair.input,
                background_color=parse_state.background_color,
                connectivity=parse_state.connectivity,
                program=program,
            )[0]
            for pair in supports
        ]
    except Exception:
        return None
    shape_mismatch, hamming, failed = _score_support_predictions(support_predictions, supports)
    output_hash = canonical_grid_sha256(output)
    candidate_id = stable_hash(
        {
            "family": "relation",
            "parse": parse_state.parse_id,
            "program": program.to_json(),
            "output": output_hash,
            "ordinal": raw_ordinal,
        }
    )[:24]
    return R5A3BCandidate(
        output_grid=output,
        output_hash=output_hash,
        candidate_id=candidate_id,
        raw_ordinal=raw_ordinal,
        parse_hypothesis_id=parse_state.parse_id,
        parse_background_color=parse_state.background_color,
        parse_connectivity=parse_state.connectivity,
        source_program_ids=(program.program_id,),
        operator_sequence=(f"edge:{program.predicate}:{program.predicate_value}", program.action, "same_canvas"),
        program_family="relation",
        relation_predicates_used=(program.predicate,),
        program_depth=program.depth,
        support_shape_mismatch_count=shape_mismatch,
        support_cell_hamming_loss=hamming,
        support_failed_pair_count=failed,
        execution_status=trace["status"],
        provenance={
            "parse_id": parse_state.parse_id,
            "source": "R5A3B_relation_conditioned_grammar",
            "program": program.to_json(),
        },
        execution_trace=trace,
    )


def _deduplicate(candidates: list[R5A3BCandidate], *, soft_scores: dict[str, int] | None = None) -> list[R5A3BCandidate]:
    groups: dict[str, list[R5A3BCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.output_hash].append(candidate)
    rows = []
    for _, group in groups.items():
        rows.append(
            sorted(
                group,
                key=lambda candidate: (
                    soft_scores.get(candidate.candidate_id, 0) if soft_scores else 0,
                    candidate.base_rank_tuple,
                ),
            )[0]
        )
    return sorted(
        rows,
        key=lambda candidate: (
            soft_scores.get(candidate.candidate_id, 0) if soft_scores else 0,
            candidate.base_rank_tuple,
        ),
    )


def build_raw_candidate_universe(supports: tuple[SupportPair, ...], query: Grid) -> dict[str, Any]:
    start = time.perf_counter()
    parse_start = time.perf_counter()
    parse_states = enumerate_parse_states(query)[:MAX_PARSE_HYPOTHESES]
    parse_seconds = time.perf_counter() - parse_start
    graph_seconds = 0.0
    enum_seconds = 0.0
    exec_seconds = 0.0
    raw_candidates: list[R5A3BCandidate] = []
    parse_rows = []
    raw_ordinal = 0
    visible_output_colors = _colors_from_grids(pair.output for pair in supports)
    for parse_state in parse_states:
        graph_start = time.perf_counter()
        graph = build_object_graph(parse_state)
        graph_seconds += time.perf_counter() - graph_start
        enum_start = time.perf_counter()
        hypothesis = _hypothesis(parse_state)
        object_programs = enumerate_programs(
            tuple(pair.input for pair in supports),
            tuple(pair.output for pair in supports),
            (query,),
            hypothesis,
            max_depth=MAX_PROGRAM_DEPTH,
            allow_relations=False,
            allow_changed_shape_render=False,
            identity_only=False,
            max_programs=max(1, MAX_RAW_PROGRAMS_PER_PARSE // 2),
        )
        relation_programs = enumerate_relation_programs(
            graph,
            visible_output_colors,
            max_programs=max(0, MAX_RAW_PROGRAMS_PER_PARSE - len(object_programs)),
        )
        enum_seconds += time.perf_counter() - enum_start
        parse_candidate_ids = []
        for program in object_programs:
            exec_start = time.perf_counter()
            candidate = _candidate_from_structural(
                query=query,
                supports=supports,
                parse_state=parse_state,
                program=program,
                raw_ordinal=raw_ordinal,
            )
            exec_seconds += time.perf_counter() - exec_start
            raw_ordinal += 1
            if candidate is not None:
                raw_candidates.append(candidate)
                parse_candidate_ids.append(candidate.candidate_id)
        for program in relation_programs:
            exec_start = time.perf_counter()
            candidate = _candidate_from_relation(
                query=query,
                supports=supports,
                parse_state=parse_state,
                program=program,
                raw_ordinal=raw_ordinal,
            )
            exec_seconds += time.perf_counter() - exec_start
            raw_ordinal += 1
            if candidate is not None:
                raw_candidates.append(candidate)
                parse_candidate_ids.append(candidate.candidate_id)
        parse_rows.append(
            {
                "parse_state": parse_state.to_json(),
                "object_graph": graph.to_json(),
                "object_program_count": len(object_programs),
                "relation_program_count": len(relation_programs),
                "candidate_ids": parse_candidate_ids,
            }
        )
        if len(raw_candidates) >= MAX_RAW_CANDIDATES_PER_FOLD:
            raw_candidates = raw_candidates[:MAX_RAW_CANDIDATES_PER_FOLD]
            break
    raw_candidates = sorted(raw_candidates, key=lambda candidate: candidate.base_rank_tuple)[:MAX_RAW_CANDIDATES_PER_FOLD]
    return {
        "accepted": True,
        "raw_candidates": raw_candidates,
        "raw_candidate_json": [candidate.to_json() for candidate in raw_candidates],
        "parse_rows": parse_rows,
        "runtime": {
            "scene_parsing_seconds": parse_seconds,
            "graph_construction_seconds": graph_seconds,
            "candidate_enumeration_seconds": enum_seconds,
            "execution_render_seconds": exec_seconds,
            "visible_support_scoring_seconds": 0.0,
            "total_candidate_universe_seconds": time.perf_counter() - start,
        },
        "unique_output_count": len({candidate.output_hash for candidate in raw_candidates}),
        "raw_candidate_count": len(raw_candidates),
    }


def visible_input_hash(supports: tuple[SupportPair, ...], query: Grid) -> str:
    return stable_hash(
        {
            "supports": [
                {
                    "input_hash": canonical_grid_sha256(pair.input),
                    "output_hash": canonical_grid_sha256(pair.output),
                }
                for pair in supports
            ],
            "query_input_hash": canonical_grid_sha256(query),
        }
    )


def _apply_condition_once(
    *,
    condition: str,
    raw_candidates: list[R5A3BCandidate],
    query: Grid,
    constraint_by_parse: dict[str, ConstraintModel],
    random_keep_count: int | None = None,
    random_salt: str | None = None,
    visible_hash: str | None = None,
) -> dict[str, Any]:
    if condition == "OBJECT_ONLY":
        base = [candidate for candidate in raw_candidates if candidate.program_family == "object"]
    else:
        base = list(raw_candidates)
    constraint_evaluations: dict[str, dict[str, Any]] = {}
    soft_scores: dict[str, int] = {}
    for candidate in base:
        model = constraint_by_parse.get(candidate.parse_hypothesis_id)
        if model is None:
            evaluation = {"soft_violation_count": 0, "hard_violation_count": 0, "soft_violations": [], "hard_violations": []}
        else:
            evaluation = evaluate_candidate_constraints(query, candidate.output_grid, model)
        constraint_evaluations[candidate.candidate_id] = evaluation
        soft_scores[candidate.candidate_id] = int(evaluation.get("soft_violation_count", 0))
    if condition == "RELATION_HARD_CONSTRAINT":
        base = [candidate for candidate in base if int(constraint_evaluations[candidate.candidate_id].get("hard_violation_count", 0)) == 0]
    elif condition == "MATCHED_RANDOM_PRUNE":
        assert random_keep_count is not None
        assert random_salt is not None
        assert visible_hash is not None
        ordered = sorted(
            base,
            key=lambda candidate: stable_hash(
                {
                    "salt": random_salt,
                    "visible_hash": visible_hash,
                    "candidate_provenance": candidate.provenance,
                    "candidate_output_hash": candidate.output_hash,
                }
            ),
        )
        base = ordered[: min(random_keep_count, len(ordered))]
    if condition == "RELATION_SOFT_CONSTRAINT" or condition == "RELATION_HARD_CONSTRAINT":
        ranked = _deduplicate(base, soft_scores=soft_scores)
    elif condition == "MATCHED_RANDOM_PRUNE":
        ranked = _deduplicate(base)
    else:
        ranked = _deduplicate(base)
    return {
        "condition": condition,
        "salt": random_salt,
        "raw_candidate_count_considered": len(base),
        "unique_candidate_count": len(ranked),
        "candidate_ids": [candidate.candidate_id for candidate in ranked],
        "candidate_hashes": [candidate.output_hash for candidate in ranked],
        "top_candidates": [candidate.to_json() for candidate in ranked[: max(TOP_K)]],
        "constraint_evaluations": constraint_evaluations,
        "hard_violation_candidate_count": sum(
            int(constraint_evaluations.get(candidate.candidate_id, {}).get("hard_violation_count", 0)) > 0
            for candidate in raw_candidates
        ),
        "soft_score_by_candidate_id": soft_scores,
    }


def apply_conditions(
    *,
    raw_candidates: list[R5A3BCandidate],
    query: Grid,
    constraint_by_parse: dict[str, ConstraintModel],
    visible_hash: str,
) -> dict[str, Any]:
    no_prune = _apply_condition_once(
        condition="RELATION_NO_PRUNE",
        raw_candidates=raw_candidates,
        query=query,
        constraint_by_parse=constraint_by_parse,
    )
    hard = _apply_condition_once(
        condition="RELATION_HARD_CONSTRAINT",
        raw_candidates=raw_candidates,
        query=query,
        constraint_by_parse=constraint_by_parse,
    )
    rows = {
        "OBJECT_ONLY": _apply_condition_once(
            condition="OBJECT_ONLY",
            raw_candidates=raw_candidates,
            query=query,
            constraint_by_parse=constraint_by_parse,
        ),
        "RELATION_NO_PRUNE": no_prune,
        "RELATION_SOFT_CONSTRAINT": _apply_condition_once(
            condition="RELATION_SOFT_CONSTRAINT",
            raw_candidates=raw_candidates,
            query=query,
            constraint_by_parse=constraint_by_parse,
        ),
        "RELATION_HARD_CONSTRAINT": hard,
    }
    keep_count = hard["unique_candidate_count"]
    random_rows = []
    for salt in MATCHED_RANDOM_SALTS:
        random_rows.append(
            _apply_condition_once(
                condition="MATCHED_RANDOM_PRUNE",
                raw_candidates=raw_candidates,
                query=query,
                constraint_by_parse=constraint_by_parse,
                random_keep_count=keep_count,
                random_salt=salt,
                visible_hash=visible_hash,
            )
        )
    rows["MATCHED_RANDOM_PRUNE"] = {
        "condition": "MATCHED_RANDOM_PRUNE",
        "salts": list(MATCHED_RANDOM_SALTS),
        "runs": random_rows,
        "matched_survivor_count": keep_count,
        "mean_unique_candidate_count": sum(row["unique_candidate_count"] for row in random_rows) / len(random_rows) if random_rows else 0.0,
        "min_unique_candidate_count": min((row["unique_candidate_count"] for row in random_rows), default=0),
        "max_unique_candidate_count": max((row["unique_candidate_count"] for row in random_rows), default=0),
    }
    return rows


def target_rank(candidates: list[dict[str, Any]], target: Grid) -> int | None:
    target_hash = canonical_grid_sha256(target)
    for index, candidate in enumerate(candidates, start=1):
        if candidate["output_hash"] == target_hash and freeze_grid(candidate["output_grid"]) == target:
            return index
    return None


def recall_flags(candidates: list[dict[str, Any]], target: Grid) -> dict[str, bool]:
    rank = target_rank(candidates, target)
    return {f"recall_at_{k}": rank is not None and rank <= min(k, len(candidates)) for k in TOP_K}
