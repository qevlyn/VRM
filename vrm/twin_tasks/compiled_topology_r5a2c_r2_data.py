from __future__ import annotations

import hashlib
import itertools
import json
import random
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GRID_SIZE = 7
PALETTE = tuple(range(9))
BACKGROUND_COLORS = (0,)
PASSABLE_COLORS = (1, 2, 3)
BARRIER_COLORS = (4,)
MARKER_COLORS = (5, 6)
OUTPUT_COLORS = (7, 8)
CONVENTIONS = ("seed_reachable_recolor", "border_reachable_recolor", "enclosed_region_recolor")
SEED_CONDITIONS = ("marker_adjacent_4_connected",)
SUPPORT_PAIR_COUNT = 3
QUERY_COUNT = 1
ADJACENCY_4 = ("north", "south", "east", "west")
ADJACENCY_8 = ("north", "north_east", "east", "south_east", "south", "south_west", "west", "north_west")
CONJUNCTION_DEPTH_BOUND = 2
MAXIMUM_COMPONENT_COUNT = 8
MAXIMUM_MARKER_COUNT = 1
SHORTCUT_THRESHOLD = 0.60

PREDICATES = (
    "is_largest_component",
    "is_smallest_component",
    "touches_border",
    "does_not_touch_border",
    "is_marker_adjacent",
    "is_not_marker_adjacent",
    "has_region_size_above_median",
    "has_region_size_below_median",
    "has_barrier_density_above_median",
    "has_seed_distance_above_median",
)

SINGLE_SHORTCUT_RULES = (
    "identity_output",
    "paint_all_passable",
    "always_seed_reachable",
    "always_border_reachable",
    "always_enclosed",
    "always_largest_passable_component",
    "always_non_border_component",
    "always_marker_adjacent_component",
    "one_step_only",
    "barrier_ignored",
    "wrong_adjacency_8_connected",
)


@dataclass(frozen=True)
class TopologyHypothesis:
    convention: str
    passable_color: int
    barrier_color: int
    marker_color_or_none: int | None
    seed_condition_or_none: str | None
    output_recolor_color: int
    adjacency: str = "4_connected"

    def to_tuple(self) -> tuple[Any, ...]:
        return (
            self.convention,
            self.passable_color,
            self.barrier_color,
            self.marker_color_or_none,
            self.seed_condition_or_none,
            self.output_recolor_color,
            self.adjacency,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "convention": self.convention,
            "passable_color": self.passable_color,
            "barrier_color": self.barrier_color,
            "marker_color_or_none": self.marker_color_or_none,
            "seed_condition_or_none": self.seed_condition_or_none,
            "output_recolor_color": self.output_recolor_color,
            "adjacency": self.adjacency,
        }


@dataclass(frozen=True)
class SupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TopologyEpisode:
    episode_id: str
    suite_name: str
    convention: str
    split_group: str
    barrier_geometry: str
    passable_color: int
    barrier_color: int
    marker_color_or_none: int | None
    output_recolor_color: int
    hypothesis: TopologyHypothesis
    supports: tuple[SupportPair, ...]
    query_input_grid: tuple[tuple[int, ...], ...]
    query_output_grid: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "suite_name": self.suite_name,
            "convention": self.convention,
            "split_group": self.split_group,
            "barrier_geometry": self.barrier_geometry,
            "passable_color": self.passable_color,
            "barrier_color": self.barrier_color,
            "marker_color_or_none": self.marker_color_or_none,
            "output_recolor_color": self.output_recolor_color,
            "hypothesis": self.hypothesis.to_json(),
            "supports": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in self.supports],
            "query_input_grid": grid_to_list(self.query_input_grid),
            "query_output_grid": grid_to_list(self.query_output_grid),
            "metadata": self.metadata,
        }


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_grid(grid: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def grid_to_list(grid: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(row) for row in grid]


def empty_grid() -> list[list[int]]:
    return [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def neighbors(coord: tuple[int, int], *, adjacency: str = "4_connected") -> list[tuple[int, int]]:
    x, y = coord
    deltas = [(0, -1), (0, 1), (1, 0), (-1, 0)] if adjacency == "4_connected" else [
        (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]
    out = []
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            out.append((nx, ny))
    return out


def candidate_hypotheses(*, adjacency: str = "4_connected") -> list[TopologyHypothesis]:
    rows: list[TopologyHypothesis] = []
    for passable, barrier, marker, output in itertools.product(PASSABLE_COLORS, BARRIER_COLORS, MARKER_COLORS, OUTPUT_COLORS):
        rows.append(TopologyHypothesis("seed_reachable_recolor", passable, barrier, marker, SEED_CONDITIONS[0], output, adjacency))
    for convention in ("border_reachable_recolor", "enclosed_region_recolor"):
        for passable, barrier, output in itertools.product(PASSABLE_COLORS, BARRIER_COLORS, OUTPUT_COLORS):
            rows.append(TopologyHypothesis(convention, passable, barrier, None, None, output, adjacency))
    return sorted(rows, key=lambda item: item.to_tuple())


def closure_domain_payload() -> dict[str, Any]:
    payload = {
        "grid_size": "7x7 only",
        "palette": list(PALETTE),
        "background_color_domain": list(BACKGROUND_COLORS),
        "passable_color_domain": list(PASSABLE_COLORS),
        "barrier_color_domain": list(BARRIER_COLORS),
        "marker_color_domain": list(MARKER_COLORS),
        "output_recolor_domain": list(OUTPUT_COLORS),
        "maximum_component_count": MAXIMUM_COMPONENT_COUNT,
        "maximum_marker_count": MAXIMUM_MARKER_COUNT,
        "barrier_geometry_catalog": ["diagonal_bridge_a", "diagonal_bridge_b", "diagonal_bridge_c", "diagonal_bridge_d"],
        "support_pair_count": SUPPORT_PAIR_COUNT,
        "query_count": QUERY_COUNT,
        "candidate_hypothesis_catalog": [hyp.to_json() for hyp in candidate_hypotheses()],
        "candidate_hypothesis_count": len(candidate_hypotheses()),
        "adjacency": "4_connected",
        "adjacency_neighbors": list(ADJACENCY_4),
        "wrong_adjacency_ablation": "8_connected",
        "wrong_adjacency_neighbors": list(ADJACENCY_8),
        "conjunction_depth_bound": CONJUNCTION_DEPTH_BOUND,
    }
    payload["closure_domain_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "closure_domain_sha256"})
    return payload


def color_positions(grid: tuple[tuple[int, ...], ...], color: int) -> set[tuple[int, int]]:
    return {(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if int(value) == color}


def passable_positions(grid: tuple[tuple[int, ...], ...], passable_color: int) -> set[tuple[int, int]]:
    return color_positions(grid, passable_color)


def barrier_positions(grid: tuple[tuple[int, ...], ...], barrier_color: int) -> set[tuple[int, int]]:
    return color_positions(grid, barrier_color)


def marker_positions(grid: tuple[tuple[int, ...], ...], marker_color: int | None) -> set[tuple[int, int]]:
    return set() if marker_color is None else color_positions(grid, marker_color)


def fixed_point_reachable(
    seed_mask: set[tuple[int, int]],
    passable_mask: set[tuple[int, int]],
    *,
    adjacency: str = "4_connected",
) -> set[tuple[int, int]]:
    reachable = set(seed_mask).intersection(passable_mask)
    queue = deque(sorted(reachable))
    while queue:
        coord = queue.popleft()
        for nxt in neighbors(coord, adjacency=adjacency):
            if nxt in passable_mask and nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)
    return reachable


def seed_mask_for_hypothesis(grid: tuple[tuple[int, ...], ...], hyp: TopologyHypothesis) -> set[tuple[int, int]]:
    passable = passable_positions(grid, hyp.passable_color)
    if hyp.convention == "seed_reachable_recolor":
        markers = marker_positions(grid, hyp.marker_color_or_none)
        seeds = set()
        for marker in markers:
            seeds.update(neighbor for neighbor in neighbors(marker, adjacency="4_connected") if neighbor in passable)
        return seeds
    if hyp.convention in {"border_reachable_recolor", "enclosed_region_recolor"}:
        return {(x, y) for x, y in passable if x in {0, GRID_SIZE - 1} or y in {0, GRID_SIZE - 1}}
    raise ValueError(f"unknown convention {hyp.convention}")


def selected_region(grid: tuple[tuple[int, ...], ...], hyp: TopologyHypothesis, *, adjacency: str | None = None) -> set[tuple[int, int]]:
    adj = adjacency or hyp.adjacency
    passable = passable_positions(grid, hyp.passable_color)
    reachable = fixed_point_reachable(seed_mask_for_hypothesis(grid, hyp), passable, adjacency=adj)
    if hyp.convention in {"seed_reachable_recolor", "border_reachable_recolor"}:
        return reachable
    if hyp.convention == "enclosed_region_recolor":
        return passable.difference(reachable)
    raise ValueError(f"unknown convention {hyp.convention}")


def apply_hypothesis_reference(
    grid: tuple[tuple[int, ...], ...],
    hyp: TopologyHypothesis,
    *,
    adjacency: str | None = None,
    no_propagation: bool = False,
    one_step_only: bool = False,
    barrier_ignored: bool = False,
    border_seed_disabled: bool = False,
) -> tuple[tuple[int, ...], ...]:
    passable = passable_positions(grid, hyp.passable_color)
    if barrier_ignored:
        passable = passable.union(barrier_positions(grid, hyp.barrier_color))
    seeds = seed_mask_for_hypothesis(grid, hyp)
    if border_seed_disabled and hyp.convention in {"border_reachable_recolor", "enclosed_region_recolor"}:
        seeds = set()
    adj = adjacency or hyp.adjacency
    if no_propagation:
        reachable = seeds.intersection(passable)
    elif one_step_only:
        reachable = seeds.intersection(passable)
        for seed in list(reachable):
            reachable.update(nxt for nxt in neighbors(seed, adjacency=adj) if nxt in passable)
    else:
        reachable = fixed_point_reachable(seeds, passable, adjacency=adj)
    region = reachable if hyp.convention in {"seed_reachable_recolor", "border_reachable_recolor"} else passable.difference(reachable)
    out = [list(row) for row in grid]
    for x, y in region:
        if grid[y][x] == hyp.passable_color:
            out[y][x] = hyp.output_recolor_color
    return freeze_grid(out)


def connected_components(coords: set[tuple[int, int]], *, adjacency: str = "4_connected") -> list[set[tuple[int, int]]]:
    unseen = set(coords)
    comps = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        comp = {start}
        queue = deque([start])
        while queue:
            coord = queue.popleft()
            for nxt in neighbors(coord, adjacency=adjacency):
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        comps.append(comp)
    return sorted(comps, key=lambda comp: (len(comp), sorted(comp)))


def shortest_distance_to_border(coords: set[tuple[int, int]]) -> int | None:
    if not coords:
        return None
    return min(min(x, y, GRID_SIZE - 1 - x, GRID_SIZE - 1 - y) for x, y in coords)


def shortest_marker_to_region_distance(grid: tuple[tuple[int, ...], ...], hyp: TopologyHypothesis) -> int | None:
    markers = marker_positions(grid, hyp.marker_color_or_none)
    region = selected_region(grid, hyp)
    if not markers or not region:
        return None
    return min(abs(mx - x) + abs(my - y) for mx, my in markers for x, y in region)


def grid_metrics(grid: tuple[tuple[int, ...], ...], hyp: TopologyHypothesis) -> dict[str, Any]:
    passable = passable_positions(grid, hyp.passable_color)
    reachable = fixed_point_reachable(seed_mask_for_hypothesis(grid, hyp), passable)
    enclosed = passable.difference(fixed_point_reachable({coord for coord in passable if coord[0] in {0, GRID_SIZE - 1} or coord[1] in {0, GRID_SIZE - 1}}, passable))
    barriers = barrier_positions(grid, hyp.barrier_color)
    passable_components = connected_components(passable)
    enclosed_components = connected_components(enclosed)
    return {
        "reachable_region_size": len(reachable),
        "enclosed_region_size": len(enclosed),
        "barrier_density": len(barriers) / (GRID_SIZE * GRID_SIZE),
        "number_of_passable_components": len(passable_components),
        "number_of_enclosed_components": len(enclosed_components),
        "seed_to_border_shortest_distance": shortest_distance_to_border(seed_mask_for_hypothesis(grid, hyp)),
        "marker_to_target_distance": shortest_marker_to_region_distance(grid, hyp),
        "number_of_candidate_hypotheses_before_support_filtering": len(candidate_hypotheses()),
    }


def domain_check(grid: tuple[tuple[int, ...], ...], hyp: TopologyHypothesis | None = None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(grid) != GRID_SIZE or any(len(row) != GRID_SIZE for row in grid):
        reasons.append("grid_not_7x7")
    values = [int(value) for row in grid for value in row]
    if any(value not in PALETTE for value in values):
        reasons.append("color_outside_palette")
    markers = sum(1 for value in values if value in MARKER_COLORS)
    if markers > MAXIMUM_MARKER_COUNT:
        reasons.append("too_many_markers")
    if hyp is not None:
        if hyp.passable_color not in PASSABLE_COLORS or hyp.barrier_color not in BARRIER_COLORS or hyp.output_recolor_color not in OUTPUT_COLORS:
            reasons.append("hypothesis_roles_outside_domain")
        if hyp.marker_color_or_none is not None and hyp.marker_color_or_none not in MARKER_COLORS:
            reasons.append("marker_role_outside_domain")
        comps = connected_components(passable_positions(grid, hyp.passable_color))
        if len(comps) > MAXIMUM_COMPONENT_COUNT:
            reasons.append("too_many_passable_components")
    return not reasons, sorted(set(reasons))


def build_grid(
    convention: str,
    *,
    passable_color: int,
    barrier_color: int,
    marker_color: int | None,
    geometry: str,
    variant: int,
) -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    mirror = variant % 2 == 1

    def put(coords: Iterable[tuple[int, int]], color: int) -> None:
        for x, y in coords:
            px = GRID_SIZE - 1 - x if mirror else x
            grid[y][px] = color

    if convention == "seed_reachable_recolor":
        y = 2 + (variant % 3)
        put([(1, y)], marker_color if marker_color is not None else MARKER_COLORS[0])
        put([(2, y), (3, y), (1, y - 1), (1, y - 2), (4, y - 1), (5, y - 1), (5, y)], passable_color)
        put([(4, y), (3, y - 1), (4, y + 1)], barrier_color)
    else:
        y = 2 + (variant % 3)
        extra_border = (6, 1) if y != 1 else (6, 5)
        extra_enclosed = (1, 1) if y != 1 else (1, 5)
        put([(0, y), (1, y), (2, y), extra_border, (3, y - 1), (4, y - 1), (4, y), (5, y), extra_enclosed], passable_color)
        put([(3, y), (2, y - 1), (3, y + 1)], barrier_color)
    if geometry.endswith("_b"):
        put([(1, 1), (5, 5)], barrier_color)
    if geometry.endswith("_c"):
        put([(1, 5), (5, 1)], barrier_color)
    if geometry.endswith("_d"):
        put([(2, 5), (4, 1)], barrier_color)
    return freeze_grid(grid)


def make_episode(
    suite_name: str,
    convention: str,
    passable_color: int,
    output_color: int,
    marker_color: int | None,
    geometry: str,
    variant: int,
) -> TopologyEpisode:
    hyp = TopologyHypothesis(
        convention=convention,
        passable_color=passable_color,
        barrier_color=BARRIER_COLORS[0],
        marker_color_or_none=marker_color if convention == "seed_reachable_recolor" else None,
        seed_condition_or_none=SEED_CONDITIONS[0] if convention == "seed_reachable_recolor" else None,
        output_recolor_color=output_color,
        adjacency="4_connected",
    )
    supports = []
    for support_index in range(SUPPORT_PAIR_COUNT):
        grid = build_grid(convention, passable_color=passable_color, barrier_color=hyp.barrier_color, marker_color=hyp.marker_color_or_none, geometry=geometry, variant=variant + support_index)
        supports.append(SupportPair(input=grid, output=apply_hypothesis_reference(grid, hyp)))
    query = build_grid(convention, passable_color=passable_color, barrier_color=hyp.barrier_color, marker_color=hyp.marker_color_or_none, geometry=geometry, variant=variant + 10)
    target = apply_hypothesis_reference(query, hyp)
    split = "stress" if suite_name == "generator_stress_suite" else "randomized" if suite_name == "randomized_suite" else "golden"
    metadata = {
        "instance_in_declared_topology_domain": True,
        "adjacency_discrimination": apply_hypothesis_reference(query, hyp, adjacency="8_connected") != target,
        "query_metrics": grid_metrics(query, hyp),
    }
    return TopologyEpisode(
        episode_id=f"{suite_name}:{convention}:p{passable_color}:o{output_color}:m{marker_color}:g{geometry}:v{variant}",
        suite_name=suite_name,
        convention=convention,
        split_group=split,
        barrier_geometry=geometry,
        passable_color=passable_color,
        barrier_color=hyp.barrier_color,
        marker_color_or_none=hyp.marker_color_or_none,
        output_recolor_color=output_color,
        hypothesis=hyp,
        supports=tuple(supports),
        query_input_grid=query,
        query_output_grid=target,
        metadata=metadata,
    )


def make_ambiguous_rejected_episode(index: int) -> dict[str, Any]:
    grid = freeze_grid(empty_grid())
    supports = [{"input": grid_to_list(grid), "output": grid_to_list(grid)} for _ in range(SUPPORT_PAIR_COUNT)]
    return {
        "episode_id": f"rejected:ambiguous:{index}",
        "supports": supports,
        "query_input_grid": grid_to_list(grid),
        "query_output_grid": grid_to_list(grid),
        "rejection_reason": "multiple_consistent_hypotheses",
        "metrics": {
            "reachable_region_size": 0,
            "enclosed_region_size": 0,
            "barrier_density": 0.0,
            "number_of_passable_components": 0,
            "number_of_enclosed_components": 0,
            "seed_to_border_shortest_distance": None,
            "marker_to_target_distance": None,
            "number_of_candidate_hypotheses_before_support_filtering": len(candidate_hypotheses()),
        },
    }


def generate_suites() -> tuple[dict[str, list[TopologyEpisode]], list[dict[str, Any]]]:
    geometries = ("diagonal_bridge_a", "diagonal_bridge_b", "diagonal_bridge_c", "diagonal_bridge_d")
    suites = {"golden_suite": [], "randomized_suite": [], "generator_stress_suite": []}
    for idx, convention in enumerate(CONVENTIONS):
        marker = MARKER_COLORS[idx % len(MARKER_COLORS)] if convention == "seed_reachable_recolor" else None
        suites["golden_suite"].append(make_episode("golden_suite", convention, PASSABLE_COLORS[idx], OUTPUT_COLORS[idx % 2], marker, geometries[idx], idx))
    variant = 0
    for suite_name, variants_per_cell in (("randomized_suite", 4), ("generator_stress_suite", 4)):
        for convention, passable, output in itertools.product(CONVENTIONS, PASSABLE_COLORS, OUTPUT_COLORS):
            for repeat in range(variants_per_cell):
                geometry = geometries[(variant + repeat) % len(geometries)]
                marker = MARKER_COLORS[(variant + repeat) % len(MARKER_COLORS)] if convention == "seed_reachable_recolor" else None
                suites[suite_name].append(make_episode(suite_name, convention, passable, output, marker, geometry, variant + repeat))
            variant += 7
    rejected = [make_ambiguous_rejected_episode(index) for index in range(8)]
    return suites, rejected


def enumerate_shortcut_catalog() -> dict[str, Any]:
    rules = [{"name": name, "kind": "single", "predicates": []} for name in SINGLE_SHORTCUT_RULES]
    for left, right in itertools.combinations(PREDICATES, 2):
        rules.append({"name": f"conjunction::{left}__AND__{right}", "kind": "pairwise_conjunction", "predicates": [left, right]})
    for first, second in itertools.permutations(PREDICATES, 2):
        rules.append({"name": f"lexicographic::{first}__THEN__{second}", "kind": "ordered_lexicographic", "predicates": [first, second]})
    rules = sorted(rules, key=lambda row: row["name"])
    payload = {
        "conjunction_depth_bound": CONJUNCTION_DEPTH_BOUND,
        "single_rule_count": len(SINGLE_SHORTCUT_RULES),
        "pairwise_conjunction_count": 45,
        "lexicographic_rule_count": 90,
        "total_rule_count": len(rules),
        "rules": rules,
    }
    payload["shortcut_catalog_sha256"] = stable_hash({key: value for key, value in payload.items() if key != "shortcut_catalog_sha256"})
    return payload


def summarize_values(values: list[Any]) -> dict[str, Any]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return {"count": len(values), "min": None, "max": None, "mean": None, "median": None, "histogram_or_bucketed_distribution": {}}
    hist = Counter(str(value) for value in numeric)
    return {
        "count": len(values),
        "min": min(numeric),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
        "median": statistics.median(numeric),
        "histogram_or_bucketed_distribution": dict(sorted(hist.items())),
    }


def distribution_report(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "reachable_region_size",
        "enclosed_region_size",
        "barrier_density",
        "number_of_passable_components",
        "number_of_enclosed_components",
        "seed_to_border_shortest_distance",
        "marker_to_target_distance",
        "number_of_candidate_hypotheses_before_support_filtering",
    )
    return {key: summarize_values([row.get(key) for row in metric_rows]) for key in keys}


def counterbalance_report(episodes: list[TopologyEpisode]) -> dict[str, Any]:
    fields = (
        "convention",
        "passable_color",
        "barrier_color",
        "marker_color_or_none",
        "output_recolor_color",
        "barrier_geometry",
    )
    records = [episode.to_record() for episode in episodes]
    marginals = {field: dict(Counter(str(record[field]) for record in records)) for field in fields}
    pairwise = {}
    under = []
    for left, right in itertools.combinations(fields, 2):
        counts = Counter((str(record[left]), str(record[right])) for record in records)
        pairwise[f"{left}__x__{right}"] = {f"{a}|{b}": count for (a, b), count in sorted(counts.items())}
        if min(counts.values(), default=0) < 2:
            under.append({"pair": [left, right], "minimum_observed_count": min(counts.values(), default=0)})
    return {
        "marginal_contingency_summaries": marginals,
        "pairwise_contingency_summaries": pairwise,
        "empty_or_underrepresented_cells": under,
    }
