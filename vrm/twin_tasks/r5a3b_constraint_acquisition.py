from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .r5a3a_scene_hypotheses import Grid, SceneHypothesis, grid_shape, stable_hash
from .r5a3b_object_graph import EDGE_PREDICATES, build_object_graph
from .r5a3b_parse_hypotheses import ParseState, object_json_with_masks

CONSTRAINT_PREDICATES = (
    "position_unchanged",
    "color_unchanged",
    "size_unchanged",
    "shape_unchanged",
    "border_status_unchanged",
    "same_relation_preserved",
    "relation_created",
    "relation_removed",
)

BOOLEAN_RELATION_PREDICATES = (
    "same_color",
    "same_shape",
    "same_size",
    "nearest",
    "aligned_row",
    "aligned_column",
    "touching",
    "contains",
)


@dataclass(frozen=True)
class SupportPairLike:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class AcquiredConstraint:
    predicate: str
    true_count: int
    false_count: int
    ambiguous_count: int
    support_count: int
    contrastive: bool
    mechanically_detectable: bool
    hard: bool
    hard_rejection_reasons: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "true_count": self.true_count,
            "false_count": self.false_count,
            "ambiguous_count": self.ambiguous_count,
            "support_count": self.support_count,
            "universal": self.true_count > 0 and self.false_count == 0,
            "contrastive": self.contrastive,
            "mechanically_detectable": self.mechanically_detectable,
            "hard": self.hard,
            "hard_rejection_reasons": list(self.hard_rejection_reasons),
        }


@dataclass(frozen=True)
class ConstraintModel:
    parse_background_color: int
    parse_connectivity: int
    constraints: tuple[AcquiredConstraint, ...]
    support_correspondence_summary: tuple[dict[str, Any], ...]
    source: str = "visible_supports_only"

    def hard_constraints(self) -> tuple[AcquiredConstraint, ...]:
        return tuple(row for row in self.constraints if row.hard)

    def to_json(self) -> dict[str, Any]:
        return {
            "parse_background_color": self.parse_background_color,
            "parse_connectivity": self.parse_connectivity,
            "source": self.source,
            "constraints": [row.to_json() for row in self.constraints],
            "support_correspondence_summary": list(self.support_correspondence_summary),
            "hard_constraint_count": sum(row.hard for row in self.constraints),
        }


def parse_state_for_grid(grid: Grid, background_color: int, connectivity: int) -> ParseState:
    hypothesis = SceneHypothesis(f"bg={background_color}|conn={connectivity}", int(background_color), int(connectivity))
    objects = object_json_with_masks(grid, hypothesis)
    object_state_hash = stable_hash(objects)
    return ParseState(
        parse_id=f"bg={background_color}|conn={connectivity}|objects={len(objects)}|hash={object_state_hash[:12]}",
        background_color=int(background_color),
        connectivity=int(connectivity),
        object_count=len(objects),
        object_state_hash=object_state_hash,
        objects=objects,
    )


def _cells(obj: dict[str, Any]) -> set[tuple[int, int]]:
    return {tuple(cell) for cell in obj["cells"]}


def _correspond_objects(input_objects: tuple[dict[str, Any], ...], output_objects: tuple[dict[str, Any], ...]) -> tuple[dict[str, str], dict[str, Any]]:
    output_cells = {obj["object_id"]: _cells(obj) for obj in output_objects}
    mapping: dict[str, str] = {}
    ambiguous = []
    unmatched = []
    reused_outputs: Counter[str] = Counter()
    for obj in input_objects:
        overlaps = []
        cells = _cells(obj)
        for out in output_objects:
            count = len(cells & output_cells[out["object_id"]])
            if count > 0:
                overlaps.append((count, out["object_id"]))
        if not overlaps:
            unmatched.append(obj["object_id"])
            continue
        best = max(count for count, _ in overlaps)
        best_ids = sorted(out_id for count, out_id in overlaps if count == best)
        if len(best_ids) != 1:
            ambiguous.append({"input_object_id": obj["object_id"], "tied_output_object_ids": best_ids, "overlap": best})
            continue
        mapping[obj["object_id"]] = best_ids[0]
        reused_outputs[best_ids[0]] += 1
    duplicate_outputs = sorted(out_id for out_id, count in reused_outputs.items() if count > 1)
    return mapping, {
        "input_object_count": len(input_objects),
        "output_object_count": len(output_objects),
        "matched_count": len(mapping),
        "unmatched_input_object_ids": unmatched,
        "ambiguous_matches": ambiguous,
        "duplicate_output_object_ids": duplicate_outputs,
        "unambiguous": not ambiguous and not duplicate_outputs and len(mapping) == len(input_objects),
    }


def _edge_map(parse_state: ParseState) -> dict[tuple[str, str], dict[str, Any]]:
    graph = build_object_graph(parse_state)
    return {(edge.source_id, edge.target_id): edge.predicates for edge in graph.edges}


def _relation_flags_for_pair(
    before_edges: dict[tuple[str, str], dict[str, Any]],
    after_edges: dict[tuple[str, str], dict[str, Any]],
    before_source: str,
    before_target: str,
    after_source: str,
    after_target: str,
) -> dict[str, bool]:
    before = before_edges.get((before_source, before_target), {})
    after = after_edges.get((after_source, after_target), {})
    comparable = []
    created = []
    removed = []
    for predicate in BOOLEAN_RELATION_PREDICATES:
        b_val = bool(before.get(predicate, False)) and not bool(before.get("nearest_is_tied", False))
        a_val = bool(after.get(predicate, False)) and not bool(after.get("nearest_is_tied", False))
        comparable.append(b_val == a_val)
        created.append((not b_val) and a_val)
        removed.append(b_val and (not a_val))
    return {
        "same_relation_preserved": bool(comparable) and all(comparable),
        "relation_created": any(created),
        "relation_removed": any(removed),
    }


def _append(counter: dict[str, list[bool]], key: str, value: bool) -> None:
    counter.setdefault(key, []).append(bool(value))


def _support_observations(
    supports: tuple[SupportPairLike, ...],
    *,
    background_color: int,
    connectivity: int,
) -> tuple[dict[str, list[bool]], list[dict[str, Any]], dict[str, set[Any]], int]:
    observations: dict[str, list[bool]] = defaultdict(list)
    contrast_values: dict[str, set[Any]] = defaultdict(set)
    summaries = []
    ambiguous_count = 0
    for index, pair in enumerate(supports):
        before_parse = parse_state_for_grid(pair.input, background_color, connectivity)
        after_parse = parse_state_for_grid(pair.output, background_color, connectivity)
        mapping, summary = _correspond_objects(before_parse.objects, after_parse.objects)
        summary["support_index"] = index
        summaries.append(summary)
        if not summary["unambiguous"]:
            ambiguous_count += 1
            continue
        before_by_id = {obj["object_id"]: obj for obj in before_parse.objects}
        after_by_id = {obj["object_id"]: obj for obj in after_parse.objects}
        for before_id, after_id in mapping.items():
            before = before_by_id[before_id]
            after = after_by_id[after_id]
            _append(observations, "position_unchanged", before["top_left_position"] == after["top_left_position"])
            _append(observations, "color_unchanged", int(before["color"]) == int(after["color"]))
            _append(observations, "size_unchanged", int(before["cell_count"]) == int(after["cell_count"]))
            _append(observations, "shape_unchanged", before["shape_hash"] == after["shape_hash"])
            _append(observations, "border_status_unchanged", bool(before["border_touch_flag"]) == bool(after["border_touch_flag"]))
            contrast_values["position_unchanged"].add(tuple(before["top_left_position"]))
            contrast_values["color_unchanged"].add(int(before["color"]))
            contrast_values["size_unchanged"].add(int(before["cell_count"]))
            contrast_values["shape_unchanged"].add(before["shape_hash"])
            contrast_values["border_status_unchanged"].add(bool(before["border_touch_flag"]))
        before_edges = _edge_map(before_parse)
        after_edges = _edge_map(after_parse)
        for before_source, after_source in mapping.items():
            for before_target, after_target in mapping.items():
                if before_source == before_target:
                    continue
                flags = _relation_flags_for_pair(before_edges, after_edges, before_source, before_target, after_source, after_target)
                for key, value in flags.items():
                    _append(observations, key, value)
                before_preds = before_edges.get((before_source, before_target), {})
                relation_signature = tuple((name, before_preds.get(name)) for name in EDGE_PREDICATES if name in before_preds)
                contrast_values["same_relation_preserved"].add(relation_signature)
                contrast_values["relation_created"].add(flags["relation_created"])
                contrast_values["relation_removed"].add(flags["relation_removed"])
    return observations, summaries, contrast_values, ambiguous_count


def acquire_constraints(
    supports: tuple[SupportPairLike, ...],
    *,
    background_color: int,
    connectivity: int,
) -> ConstraintModel:
    observations, summaries, contrast_values, ambiguous_supports = _support_observations(
        supports,
        background_color=background_color,
        connectivity=connectivity,
    )
    constraints = []
    for predicate in CONSTRAINT_PREDICATES:
        values = observations.get(predicate, [])
        true_count = sum(values)
        false_count = len(values) - true_count
        reasons = []
        if not values:
            reasons.append("no_unambiguous_visible_observations")
        if false_count:
            reasons.append("not_universal_on_visible_supports")
        if ambiguous_supports:
            reasons.append("ambiguous_correspondence_present")
        contrastive = len(contrast_values.get(predicate, set())) >= 2
        if not contrastive:
            reasons.append("not_contrastive_on_visible_supports")
        mechanically_detectable = predicate in CONSTRAINT_PREDICATES
        if not mechanically_detectable:
            reasons.append("not_mechanically_detectable")
        hard = bool(values) and false_count == 0 and ambiguous_supports == 0 and contrastive and mechanically_detectable
        constraints.append(
            AcquiredConstraint(
                predicate=predicate,
                true_count=true_count,
                false_count=false_count,
                ambiguous_count=ambiguous_supports,
                support_count=len(supports),
                contrastive=contrastive,
                mechanically_detectable=mechanically_detectable,
                hard=hard,
                hard_rejection_reasons=tuple(reasons),
            )
        )
    return ConstraintModel(int(background_color), int(connectivity), tuple(constraints), tuple(summaries))


def _predicate_values_for_candidate(query: Grid, output: Grid, background_color: int, connectivity: int) -> tuple[dict[str, list[bool]], dict[str, Any]]:
    before_parse = parse_state_for_grid(query, background_color, connectivity)
    after_parse = parse_state_for_grid(output, background_color, connectivity)
    mapping, summary = _correspond_objects(before_parse.objects, after_parse.objects)
    observations: dict[str, list[bool]] = defaultdict(list)
    if not summary["unambiguous"]:
        return observations, summary
    before_by_id = {obj["object_id"]: obj for obj in before_parse.objects}
    after_by_id = {obj["object_id"]: obj for obj in after_parse.objects}
    for before_id, after_id in mapping.items():
        before = before_by_id[before_id]
        after = after_by_id[after_id]
        _append(observations, "position_unchanged", before["top_left_position"] == after["top_left_position"])
        _append(observations, "color_unchanged", int(before["color"]) == int(after["color"]))
        _append(observations, "size_unchanged", int(before["cell_count"]) == int(after["cell_count"]))
        _append(observations, "shape_unchanged", before["shape_hash"] == after["shape_hash"])
        _append(observations, "border_status_unchanged", bool(before["border_touch_flag"]) == bool(after["border_touch_flag"]))
    before_edges = _edge_map(before_parse)
    after_edges = _edge_map(after_parse)
    for before_source, after_source in mapping.items():
        for before_target, after_target in mapping.items():
            if before_source == before_target:
                continue
            flags = _relation_flags_for_pair(before_edges, after_edges, before_source, before_target, after_source, after_target)
            for key, value in flags.items():
                _append(observations, key, value)
    return observations, summary


def evaluate_candidate_constraints(query: Grid, output: Grid, model: ConstraintModel) -> dict[str, Any]:
    observations, correspondence = _predicate_values_for_candidate(
        query,
        output,
        model.parse_background_color,
        model.parse_connectivity,
    )
    soft_violations = []
    hard_violations = []
    if not correspondence["unambiguous"]:
        for constraint in model.hard_constraints():
            hard_violations.append(
                {
                    "predicate": constraint.predicate,
                    "reason": "candidate_correspondence_ambiguous_or_incomplete",
                }
            )
        return {
            "soft_violation_count": len(model.constraints),
            "hard_violation_count": len(hard_violations),
            "soft_violations": [{"predicate": row.predicate, "reason": "candidate_correspondence_ambiguous_or_incomplete"} for row in model.constraints],
            "hard_violations": hard_violations,
            "candidate_correspondence": correspondence,
        }
    for constraint in model.constraints:
        values = observations.get(constraint.predicate, [])
        violated = not values or not all(values)
        if violated:
            soft_violations.append({"predicate": constraint.predicate, "reason": "candidate_does_not_satisfy_visible_support_pattern"})
            if constraint.hard:
                hard_violations.append({"predicate": constraint.predicate, "reason": "hard_constraint_violation"})
    return {
        "soft_violation_count": len(soft_violations),
        "hard_violation_count": len(hard_violations),
        "soft_violations": soft_violations,
        "hard_violations": hard_violations,
        "candidate_correspondence": correspondence,
    }


def constraint_model_signature(model: ConstraintModel) -> str:
    return stable_hash(model.to_json())
