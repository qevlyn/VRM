from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .r5a3a_arc_native_candidate_machine import SupportPair
from .r5a3a_scene_hypotheses import Grid, SceneHypothesis, canonical_grid_sha256, grid_to_list, stable_hash
from .r5a3a_structural_programs import enumerate_programs
from .r5a3b_object_graph import ObjectGraph, build_object_graph
from .r5a3b_parse_hypotheses import MAX_PARSE_HYPOTHESES, ParseState, enumerate_parse_states
from .r5a3b_relation_programs import (
    R5A3BCandidate,
    RELATION_PROGRAM_PREDICATE_PRIORITY,
    RelationProgram,
    _candidate_from_relation,
    _candidate_from_structural,
)

MAX_FINAL_PROGRAMS_PER_PARSE = 8
MAX_FINAL_CANDIDATES_PER_FOLD = 32
MAX_PROGRAM_DEPTH = 2
TOP_K = [1, 4, 16, 64, 128]
OBJECT_PROGRAM_SLOTS_PER_PARSE = 4
RELATION_PROGRAM_SLOTS_PER_PARSE = 4
MAX_RELATION_MICROPROPOSALS_PER_BUCKET = 10
MAX_RELATION_MICROPROPOSALS_PER_PARSE = 40

RELATION_ACTION_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("RELATION_TRANSPORT_COPY", ("copy_source_by_relation_vector",)),
    ("RELATION_TRANSPORT_TRANSLATE", ("translate_target_by_relation_vector",)),
    ("RELATION_RECOLOR", ("recolor_target_to_source", "recolor_source_to_target", "recolor_target_constant")),
    ("RELATION_STATE_EDIT", ("delete_target", "keep_relation_endpoints")),
)


@dataclass(frozen=True)
class BucketSelection:
    bucket: str
    attempted: bool
    status: str
    selected_candidate: R5A3BCandidate | None
    selected_program: RelationProgram | None
    selected_rank_tuple: tuple[int, int, int, str, str] | None
    considered_count: int
    legal_count: int
    executable_count: int
    discarded: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "attempted": self.attempted,
            "status": self.status,
            "selected_candidate_id": self.selected_candidate.candidate_id if self.selected_candidate else None,
            "selected_program": self.selected_program.to_json() if self.selected_program else None,
            "selected_program_support_rank_tuple": list(self.selected_rank_tuple) if self.selected_rank_tuple else None,
            "considered_count": self.considered_count,
            "legal_count": self.legal_count,
            "executable_count": self.executable_count,
            "discarded_candidate_count": len(self.discarded),
            "discarded": list(self.discarded),
        }


def _hypothesis(parse_state: ParseState) -> SceneHypothesis:
    return SceneHypothesis(f"bg={parse_state.background_color}|conn={parse_state.connectivity}", parse_state.background_color, parse_state.connectivity)


def _colors_from_grids(grids: Iterable[Grid]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for grid in grids for row in grid for value in row}))


def canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def bucket_for_action(action: str) -> str | None:
    for bucket, actions in RELATION_ACTION_BUCKETS:
        if action in actions:
            return bucket
    return None


def _edge_predicate_payload(edge: Any, predicate: str) -> Any:
    if predicate in {"relative_direction", "border_relation"}:
        return edge.predicates[predicate]
    return True


def _edge_supports_predicate(edge: Any, predicate: str) -> bool:
    if predicate == "nearest" and edge.predicates.get("nearest_is_tied"):
        return False
    if predicate in {"relative_direction", "border_relation"}:
        return predicate in edge.predicates
    return bool(edge.predicates.get(predicate))


def enumerate_bucket_microproposals(
    graph: ObjectGraph,
    visible_output_colors: tuple[int, ...],
    bucket: str,
) -> tuple[RelationProgram, ...]:
    actions = dict(RELATION_ACTION_BUCKETS)[bucket]
    constant_color = visible_output_colors[0] if visible_output_colors else None
    rows: list[RelationProgram] = []
    seen = set()
    for edge in graph.edges:
        for predicate in RELATION_PROGRAM_PREDICATE_PRIORITY:
            if not _edge_supports_predicate(edge, predicate):
                continue
            payload = _edge_predicate_payload(edge, predicate)
            for action in actions:
                if action == "recolor_target_constant" and constant_color is None:
                    continue
                program_id = f"relation={predicate}:{payload}|action={action}|render=same_canvas"
                if program_id in seen:
                    continue
                seen.add(program_id)
                rows.append(RelationProgram(program_id, predicate, payload, action, constant_color))
                if len(rows) >= MAX_RELATION_MICROPROPOSALS_PER_BUCKET:
                    return tuple(rows)
    return tuple(rows)


def relation_tie_break(program: RelationProgram, candidate: R5A3BCandidate) -> tuple[str, int, str, str, str]:
    predicate_index = RELATION_PROGRAM_PREDICATE_PRIORITY.index(program.predicate)
    return (
        program.action,
        predicate_index,
        canonical_value(program.predicate_value),
        program.program_id,
        candidate.output_hash,
    )


def _trace_counts(candidate: R5A3BCandidate) -> dict[str, int]:
    trace = candidate.execution_trace
    return {
        "selected_edge_count": int(trace.get("selected_edge_count", 0)),
        "dropped_out_of_bounds_count": len(trace.get("dropped_out_of_bounds", [])),
        "collision_count": len(trace.get("colliding_destinations", [])),
        "overwritten_destination_count": len(trace.get("overwritten_destinations", [])),
    }


def select_best_candidate_for_bucket(
    *,
    supports: tuple[SupportPair, ...],
    query: Grid,
    parse_state: ParseState,
    graph: ObjectGraph,
    visible_output_colors: tuple[int, ...],
    bucket: str,
    raw_ordinal: int,
) -> BucketSelection:
    proposals = enumerate_bucket_microproposals(graph, visible_output_colors, bucket)
    executable: list[tuple[RelationProgram, R5A3BCandidate]] = []
    discarded = []
    for index, program in enumerate(proposals):
        candidate = _candidate_from_relation(
            query=query,
            supports=supports,
            parse_state=parse_state,
            program=program,
            raw_ordinal=raw_ordinal,
        )
        if candidate is None:
            discarded.append(
                {
                    "program": program.to_json(),
                    "why_discarded": "EXECUTION_FAILED",
                    "microproposal_index": index,
                }
            )
            continue
        executable.append((program, candidate))
    if not proposals:
        return BucketSelection(bucket, True, "BUCKET_UNAVAILABLE", None, None, None, 0, 0, 0, tuple())
    if not executable:
        return BucketSelection(bucket, True, "NO_EXECUTABLE_OUTPUT", None, None, None, len(proposals), len(proposals), 0, tuple(discarded))
    ranked = sorted(executable, key=lambda item: (item[1].base_rank_tuple, relation_tie_break(item[0], item[1])))
    selected_program, selected_candidate = ranked[0]
    selected_key = (selected_program.program_id, selected_candidate.output_hash)
    for program, candidate in ranked[1:]:
        discarded.append(
            {
                "program": program.to_json(),
                "candidate_id": candidate.candidate_id,
                "output_hash": candidate.output_hash,
                "rank_tuple": list(candidate.base_rank_tuple),
                "tie_break": list(relation_tie_break(program, candidate)),
                "why_discarded": "LOST_SUPPORT_RANK_OR_TIE_BREAK",
                "lost_to": {"program_id": selected_program.program_id, "output_hash": selected_candidate.output_hash},
                "execution_trace_counts": _trace_counts(candidate),
            }
        )
    return BucketSelection(
        bucket=bucket,
        attempted=True,
        status="SELECTED",
        selected_candidate=selected_candidate,
        selected_program=selected_program,
        selected_rank_tuple=selected_candidate.base_rank_tuple,
        considered_count=len(proposals),
        legal_count=len(proposals),
        executable_count=len(executable),
        discarded=tuple(discarded),
    )


def generate_p0_equivalent_object_candidates(
    *,
    supports: tuple[SupportPair, ...],
    query: Grid,
    parse_state: ParseState,
    parse_index: int,
) -> tuple[R5A3BCandidate, ...]:
    programs = enumerate_programs(
        tuple(pair.input for pair in supports),
        tuple(pair.output for pair in supports),
        (query,),
        _hypothesis(parse_state),
        max_depth=MAX_PROGRAM_DEPTH,
        allow_relations=False,
        allow_changed_shape_render=False,
        identity_only=False,
        max_programs=OBJECT_PROGRAM_SLOTS_PER_PARSE,
    )
    rows = []
    for object_index, program in enumerate(programs):
        candidate = _candidate_from_structural(
            query=query,
            supports=supports,
            parse_state=parse_state,
            program=program,
            raw_ordinal=parse_index * MAX_FINAL_PROGRAMS_PER_PARSE + object_index,
        )
        if candidate is not None:
            rows.append(candidate)
    return tuple(rows)


def _deduplicate(candidates: list[R5A3BCandidate]) -> list[R5A3BCandidate]:
    groups: dict[str, list[R5A3BCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.output_hash].append(candidate)
    representatives = []
    for group in groups.values():
        representatives.append(sorted(group, key=lambda candidate: candidate.base_rank_tuple)[0])
    return sorted(representatives, key=lambda candidate: candidate.base_rank_tuple)


def build_p2a_candidate_universe(supports: tuple[SupportPair, ...], query: Grid) -> dict[str, Any]:
    start = time.perf_counter()
    parse_states = enumerate_parse_states(query)[:MAX_PARSE_HYPOTHESES]
    visible_output_colors = _colors_from_grids(pair.output for pair in supports)
    final_candidates: list[R5A3BCandidate] = []
    object_candidates: list[R5A3BCandidate] = []
    parse_reports = []
    microproposal_reports = []
    bucket_reports = []
    microproposal_count = 0
    for parse_index, parse_state in enumerate(parse_states):
        graph = build_object_graph(parse_state)
        objects = generate_p0_equivalent_object_candidates(
            supports=supports,
            query=query,
            parse_state=parse_state,
            parse_index=parse_index,
        )
        object_candidates.extend(objects)
        selected_relation_candidates = []
        parse_bucket_rows = []
        for bucket_index, (bucket, _) in enumerate(RELATION_ACTION_BUCKETS):
            raw_ordinal = parse_index * MAX_FINAL_PROGRAMS_PER_PARSE + OBJECT_PROGRAM_SLOTS_PER_PARSE + bucket_index
            selection = select_best_candidate_for_bucket(
                supports=supports,
                query=query,
                parse_state=parse_state,
                graph=graph,
                visible_output_colors=visible_output_colors,
                bucket=bucket,
                raw_ordinal=raw_ordinal,
            )
            microproposal_count += selection.considered_count
            if microproposal_count > MAX_RELATION_MICROPROPOSALS_PER_PARSE * len(parse_states):
                raise RuntimeError("microproposal cap exceeded")
            parse_bucket_rows.append(selection.to_json())
            bucket_reports.append({"parse_id": parse_state.parse_id, **selection.to_json()})
            if selection.selected_candidate is not None:
                selected_relation_candidates.append(selection.selected_candidate)
        parse_final = list(objects) + selected_relation_candidates
        if len(parse_final) > MAX_FINAL_PROGRAMS_PER_PARSE:
            raise RuntimeError("final program count exceeded per parse")
        final_candidates.extend(parse_final)
        microproposal_reports.append(
            {
                "parse_id": parse_state.parse_id,
                "microproposals_considered_total": sum(row["considered_count"] for row in parse_bucket_rows),
                "microproposals_considered_by_bucket": {row["bucket"]: row["considered_count"] for row in parse_bucket_rows},
                "legal_microproposals_by_bucket": {row["bucket"]: row["legal_count"] for row in parse_bucket_rows},
                "selected_program_by_bucket": {row["bucket"]: row["selected_program"] for row in parse_bucket_rows},
                "selected_program_support_rank_tuple_by_bucket": {row["bucket"]: row["selected_program_support_rank_tuple"] for row in parse_bucket_rows},
                "discarded_candidate_count_by_bucket": {row["bucket"]: row["discarded_candidate_count"] for row in parse_bucket_rows},
                "bucket_rows": parse_bucket_rows,
            }
        )
        parse_reports.append(
            {
                "parse_state": parse_state.to_json(),
                "object_candidate_count": len(objects),
                "relation_candidate_count": len(selected_relation_candidates),
                "final_candidate_count": len(parse_final),
                "candidate_ids": [candidate.candidate_id for candidate in parse_final],
            }
        )
    final_candidates = sorted(final_candidates, key=lambda candidate: candidate.base_rank_tuple)[:MAX_FINAL_CANDIDATES_PER_FOLD]
    top_candidates = _deduplicate(final_candidates)
    return {
        "raw_candidates": final_candidates,
        "object_candidates": object_candidates,
        "top_candidates": top_candidates,
        "raw_candidate_json": [candidate.to_json() for candidate in final_candidates],
        "top_candidate_json": [candidate.to_json() for candidate in top_candidates],
        "parse_reports": parse_reports,
        "microproposal_reports": microproposal_reports,
        "bucket_reports": bucket_reports,
        "microproposal_count": microproposal_count,
        "runtime_seconds": time.perf_counter() - start,
    }


def candidate_trace_counts_json(candidate: dict[str, Any]) -> dict[str, int]:
    trace = candidate.get("execution_trace", {})
    return {
        "selected_edge_count": int(trace.get("selected_edge_count", 0)),
        "dropped_out_of_bounds_count": len(trace.get("dropped_out_of_bounds", [])),
        "collision_count": len(trace.get("colliding_destinations", [])),
        "overwritten_destination_count": len(trace.get("overwritten_destinations", [])),
    }


def allocation_spec() -> dict[str, Any]:
    payload = {
        "name": "R5A-3B-P2A Balanced Relation-Transport Allocation",
        "max_parse_hypotheses": MAX_PARSE_HYPOTHESES,
        "max_final_programs_per_parse": MAX_FINAL_PROGRAMS_PER_PARSE,
        "max_final_candidates_per_fold": MAX_FINAL_CANDIDATES_PER_FOLD,
        "max_program_depth": MAX_PROGRAM_DEPTH,
        "top_k": TOP_K,
        "object_program_slots_per_parse": OBJECT_PROGRAM_SLOTS_PER_PARSE,
        "relation_program_slots_per_parse": RELATION_PROGRAM_SLOTS_PER_PARSE,
        "relation_action_buckets": {bucket: list(actions) for bucket, actions in RELATION_ACTION_BUCKETS},
        "relation_predicate_menu": list(RELATION_PROGRAM_PREDICATE_PRIORITY),
        "microproposal_caps": {
            "MAX_RELATION_MICROPROPOSALS_PER_BUCKET": MAX_RELATION_MICROPROPOSALS_PER_BUCKET,
            "MAX_RELATION_MICROPROPOSALS_PER_PARSE": MAX_RELATION_MICROPROPOSALS_PER_PARSE,
        },
        "tie_break_order": [
            "visible_support_rank_tuple",
            "action_name",
            "predicate_priority",
            "predicate_value_canonical_representation",
            "program_id",
            "output_hash",
        ],
        "semantic_freeze": "P0 global-edge semantics; no source-target edge binding.",
    }
    payload["spec_sha256"] = stable_hash(payload)
    return payload
