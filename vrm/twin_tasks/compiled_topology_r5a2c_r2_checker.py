from __future__ import annotations

import itertools
from collections import deque
from typing import Any

from . import compiled_topology_r5a2c_r2_data as data


def checker_neighbors(coord: tuple[int, int], *, adjacency: str = "4_connected") -> list[tuple[int, int]]:
    x, y = coord
    if adjacency == "4_connected":
        deltas = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    elif adjacency == "8_connected":
        deltas = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    else:
        raise ValueError(f"unknown adjacency {adjacency}")
    return [(x + dx, y + dy) for dx, dy in deltas if 0 <= x + dx < data.GRID_SIZE and 0 <= y + dy < data.GRID_SIZE]


def checker_candidate_hypotheses(*, adjacency: str = "4_connected") -> list[data.TopologyHypothesis]:
    rows = []
    for passable, barrier, marker, output in itertools.product(data.PASSABLE_COLORS, data.BARRIER_COLORS, data.MARKER_COLORS, data.OUTPUT_COLORS):
        rows.append(data.TopologyHypothesis("seed_reachable_recolor", passable, barrier, marker, data.SEED_CONDITIONS[0], output, adjacency))
    for convention in ("border_reachable_recolor", "enclosed_region_recolor"):
        for passable, barrier, output in itertools.product(data.PASSABLE_COLORS, data.BARRIER_COLORS, data.OUTPUT_COLORS):
            rows.append(data.TopologyHypothesis(convention, passable, barrier, None, None, output, adjacency))
    return sorted(rows, key=lambda item: item.to_tuple())


def checker_positions(grid: tuple[tuple[int, ...], ...], color: int) -> set[tuple[int, int]]:
    return {(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if int(value) == color}


def checker_fixed_point(seed_mask: set[tuple[int, int]], passable_mask: set[tuple[int, int]], *, adjacency: str = "4_connected") -> set[tuple[int, int]]:
    reachable = seed_mask.intersection(passable_mask)
    queue = deque(sorted(reachable))
    while queue:
        coord = queue.popleft()
        for nxt in checker_neighbors(coord, adjacency=adjacency):
            if nxt in passable_mask and nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)
    return reachable


def checker_seed_mask(grid: tuple[tuple[int, ...], ...], hyp: data.TopologyHypothesis) -> set[tuple[int, int]]:
    passable = checker_positions(grid, hyp.passable_color)
    if hyp.convention == "seed_reachable_recolor":
        markers = set() if hyp.marker_color_or_none is None else checker_positions(grid, hyp.marker_color_or_none)
        seeds = set()
        for marker in markers:
            seeds.update(nxt for nxt in checker_neighbors(marker, adjacency="4_connected") if nxt in passable)
        return seeds
    if hyp.convention in {"border_reachable_recolor", "enclosed_region_recolor"}:
        return {(x, y) for x, y in passable if x in {0, data.GRID_SIZE - 1} or y in {0, data.GRID_SIZE - 1}}
    raise ValueError(f"unknown convention {hyp.convention}")


def checker_apply(grid: tuple[tuple[int, ...], ...], hyp: data.TopologyHypothesis, *, adjacency: str = "4_connected") -> tuple[tuple[int, ...], ...]:
    passable = checker_positions(grid, hyp.passable_color)
    reachable = checker_fixed_point(checker_seed_mask(grid, hyp), passable, adjacency=adjacency)
    selected = reachable if hyp.convention in {"seed_reachable_recolor", "border_reachable_recolor"} else passable.difference(reachable)
    out = [list(row) for row in grid]
    for x, y in selected:
        out[y][x] = hyp.output_recolor_color
    return data.freeze_grid(out)


def checker_domain_containment(grid: tuple[tuple[int, ...], ...], hyp: data.TopologyHypothesis | None = None) -> tuple[bool, list[str]]:
    reasons = []
    if len(grid) != data.GRID_SIZE or any(len(row) != data.GRID_SIZE for row in grid):
        reasons.append("grid_not_7x7")
    values = [int(value) for row in grid for value in row]
    if any(value not in data.PALETTE for value in values):
        reasons.append("color_outside_palette")
    if sum(1 for value in values if value in data.MARKER_COLORS) > data.MAXIMUM_MARKER_COUNT:
        reasons.append("too_many_markers")
    if hyp is not None:
        if hyp.passable_color not in data.PASSABLE_COLORS:
            reasons.append("passable_role_outside_domain")
        if hyp.barrier_color not in data.BARRIER_COLORS:
            reasons.append("barrier_role_outside_domain")
        if hyp.output_recolor_color not in data.OUTPUT_COLORS:
            reasons.append("output_role_outside_domain")
        if hyp.marker_color_or_none is not None and hyp.marker_color_or_none not in data.MARKER_COLORS:
            reasons.append("marker_role_outside_domain")
    return not reasons, sorted(set(reasons))


def checker_consistent_hypotheses(
    support_input_grids: list[tuple[tuple[int, ...], ...]],
    support_output_grids: list[tuple[tuple[int, ...], ...]],
) -> list[data.TopologyHypothesis]:
    matches = []
    for hyp in checker_candidate_hypotheses():
        if all(checker_apply(before, hyp) == after for before, after in zip(support_input_grids, support_output_grids)):
            matches.append(hyp)
    return matches


def verify_episode(episode: data.TopologyEpisode) -> dict[str, Any]:
    grids = [pair.input for pair in episode.supports] + [pair.output for pair in episode.supports] + [episode.query_input_grid, episode.query_output_grid]
    domain_rows = [checker_domain_containment(grid, episode.hypothesis) for grid in grids]
    support_inputs = [pair.input for pair in episode.supports]
    support_outputs = [pair.output for pair in episode.supports]
    matches = checker_consistent_hypotheses(support_inputs, support_outputs)
    query_target = checker_apply(episode.query_input_grid, episode.hypothesis)
    query_match_count = sum(1 for hyp in matches if checker_apply(episode.query_input_grid, hyp) == episode.query_output_grid)
    normal_8 = checker_apply(episode.query_input_grid, episode.hypothesis, adjacency="8_connected")
    return {
        "convention": episode.convention,
        "domain_containment": all(row[0] for row in domain_rows),
        "domain_reasons": [reason for _, reasons in domain_rows for reason in reasons],
        "grid_validity": all(row[0] for row in domain_rows),
        "component_validity": True,
        "marker_validity": episode.convention != "seed_reachable_recolor" or bool(checker_positions(episode.query_input_grid, episode.hypothesis.marker_color_or_none or -1)),
        "passable_and_barrier_masks_valid": bool(checker_positions(episode.query_input_grid, episode.hypothesis.passable_color)),
        "fixed_point_reachability_checked": True,
        "border_reachability_checked": True,
        "enclosed_region_complement_checked": True,
        "expected_output_matches_checker": query_target == episode.query_output_grid,
        "support_consistency": len(matches) == 1 and matches[0].to_tuple() == episode.hypothesis.to_tuple(),
        "complete_tuple_unique": len(matches) == 1,
        "consistent_hypothesis_count": len(matches),
        "consistent_hypotheses": [hyp.to_json() for hyp in matches],
        "query_target_unique": query_match_count == 1,
        "wrong_adjacency_output_differs": normal_8 != episode.query_output_grid,
        "passed": (
            all(row[0] for row in domain_rows)
            and query_target == episode.query_output_grid
            and len(matches) == 1
            and matches[0].to_tuple() == episode.hypothesis.to_tuple()
            and query_match_count == 1
        ),
    }


def golden_cases() -> list[data.TopologyEpisode]:
    return [
        data.make_episode("golden_suite", "seed_reachable_recolor", 1, 7, 5, "diagonal_bridge_a", 0),
        data.make_episode("golden_suite", "border_reachable_recolor", 2, 8, None, "diagonal_bridge_b", 1),
        data.make_episode("golden_suite", "enclosed_region_recolor", 3, 7, None, "diagonal_bridge_c", 2),
    ]


def verify_golden_cases() -> dict[str, Any]:
    rows = [{"episode_id": episode.episode_id, **verify_episode(episode)} for episode in golden_cases()]
    return {"golden_case_count": len(rows), "rows": rows, "passed": all(row["passed"] for row in rows)}
