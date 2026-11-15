from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from . import transport_recolor_r5a2a_declared_closure as closure

Grid = tuple[tuple[int, ...], ...]
Cell = tuple[int, int, int]

FAMILY_ORDER = (
    "largest_component_select_then_v2_transform",
    "nearest_marker_select_then_v2_transform",
    "unique_shape_select_then_v2_transform",
    "border_relation_select_then_v2_transform",
)

SELECTOR_RULE_BY_FAMILY = {
    "largest_component_select_then_v2_transform": "largest_component",
    "nearest_marker_select_then_v2_transform": "nearest_marker",
    "unique_shape_select_then_v2_transform": "unique_shape",
    "border_relation_select_then_v2_transform": "border_relation",
}

TRANSFORM_PROGRAMS = closure.FRONTIER_TRANSFORM_PROGRAMS
SPLIT_BY_TRANSFORM = {"C": "train_visible", "H": "train_visible", "V": "held_out", "C->L": "held_out"}
MARKER_COLOR = 8
SHORTCUT_THRESHOLD = 0.60

SHAPES = {
    "single": ((0, 0),),
    "domino_h": ((0, 0), (1, 0)),
    "domino_v": ((0, 0), (0, 1)),
    "tri_l": ((0, 0), (0, 1), (1, 1)),
    "tri_bar_h": ((0, 0), (1, 0), (2, 0)),
    "tri_bar_v": ((0, 0), (0, 1), (0, 2)),
}


@dataclass(frozen=True)
class SupportPair:
    input: Grid
    output: Grid
    selected_cells_before: tuple[Cell, ...]
    selected_cells_after: tuple[Cell, ...]


@dataclass(frozen=True)
class ProbeEpisode:
    episode_id: str
    probe_family: str
    selector_rule: str
    transform_program: str
    split_group: str
    supports: tuple[SupportPair, ...]
    query: Grid
    target: Grid
    selected_cells_before: tuple[Cell, ...]
    selected_cells_after: tuple[Cell, ...]
    isolated_supports: tuple[SupportPair, ...]
    isolated_query: Grid
    isolated_target: Grid
    validation: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "probe_family": self.probe_family,
            "selector_rule": self.selector_rule,
            "transform_program": self.transform_program,
            "split_group": self.split_group,
            "supports": [
                {
                    "input": closure.grid_to_list(pair.input),
                    "output": closure.grid_to_list(pair.output),
                    "selected_cells_before": [list(cell) for cell in pair.selected_cells_before],
                    "selected_cells_after": [list(cell) for cell in pair.selected_cells_after],
                }
                for pair in self.supports
            ],
            "query": closure.grid_to_list(self.query),
            "target": closure.grid_to_list(self.target),
            "selected_cells_before": [list(cell) for cell in self.selected_cells_before],
            "selected_cells_after": [list(cell) for cell in self.selected_cells_after],
            "isolated_supports": [
                {
                    "input": closure.grid_to_list(pair.input),
                    "output": closure.grid_to_list(pair.output),
                    "selected_cells_before": [list(cell) for cell in pair.selected_cells_before],
                    "selected_cells_after": [list(cell) for cell in pair.selected_cells_after],
                }
                for pair in self.isolated_supports
            ],
            "isolated_query": closure.grid_to_list(self.isolated_query),
            "isolated_target": closure.grid_to_list(self.isolated_target),
            "validation": self.validation,
        }


def stable_int(text: str) -> int:
    return int(closure.stable_hash(text)[:16], 16)


def grid_from_cells(cells: Iterable[Cell]) -> Grid:
    grid = closure.empty_grid()
    for x, y, color in cells:
        if grid[y][x] != closure.BACKGROUND:
            raise ValueError("overlapping cell placement")
        grid[y][x] = int(color)
    return closure.freeze_grid(grid)


def cells_from_grid(grid: Grid) -> tuple[Cell, ...]:
    return tuple(closure.foreground_cells(grid))


def component_signature(component: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    coords = component["coords"]
    min_x = min(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    return tuple(sorted((x - min_x, y - min_y) for x, y in coords))


def component_cells(component: dict[str, Any], grid: Grid) -> tuple[Cell, ...]:
    return tuple(sorted((x, y, int(grid[y][x])) for x, y in component["coords"]))


def connected_components(grid: Grid) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    components = []
    for y in range(closure.GRID_SIZE):
        for x in range(closure.GRID_SIZE):
            if grid[y][x] == closure.BACKGROUND or (x, y) in seen:
                continue
            queue = deque([(x, y)])
            seen.add((x, y))
            coords = []
            colors = []
            while queue:
                cx, cy = queue.popleft()
                coords.append((cx, cy))
                colors.append(int(grid[cy][cx]))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < closure.GRID_SIZE and 0 <= ny < closure.GRID_SIZE):
                        continue
                    if grid[ny][nx] == closure.BACKGROUND or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    queue.append((nx, ny))
            colors_counter = Counter(colors)
            component = {
                "coords": tuple(sorted(coords)),
                "size": len(coords),
                "colors": tuple(sorted(colors_counter)),
                "dominant_color": colors_counter.most_common(1)[0][0],
                "is_marker": len(colors_counter) == 1 and colors[0] == MARKER_COLOR,
            }
            component["shape_signature"] = component_signature(component)
            component["centroid"] = (
                sum(x for x, _ in component["coords"]) / component["size"],
                sum(y for _, y in component["coords"]) / component["size"],
            )
            components.append(component)
    return sorted(components, key=lambda row: (min(y for _, y in row["coords"]), min(x for x, _ in row["coords"]), row["size"]))


def non_marker_components(grid: Grid) -> list[dict[str, Any]]:
    return [component for component in connected_components(grid) if not component["is_marker"]]


def select_component(grid: Grid, selector_rule: str) -> dict[str, Any] | None:
    components = non_marker_components(grid)
    if not components:
        return None
    if selector_rule == "largest_component":
        max_size = max(component["size"] for component in components)
        candidates = [component for component in components if component["size"] == max_size]
        return candidates[0] if len(candidates) == 1 else None
    if selector_rule == "nearest_marker":
        markers = [component for component in connected_components(grid) if component["is_marker"]]
        if len(markers) != 1:
            return None
        marker_coords = markers[0]["coords"]
        scored = []
        for component in components:
            distance = min(abs(x - mx) + abs(y - my) for x, y in component["coords"] for mx, my in marker_coords)
            scored.append((distance, component))
        best = min(distance for distance, _ in scored)
        candidates = [component for distance, component in scored if distance == best]
        return candidates[0] if len(candidates) == 1 else None
    if selector_rule == "unique_shape":
        counts = Counter(component["shape_signature"] for component in components)
        candidates = [component for component in components if counts[component["shape_signature"]] == 1]
        return candidates[0] if len(candidates) == 1 else None
    if selector_rule == "border_relation":
        candidates = []
        for component in components:
            touched = set()
            for x, y in component["coords"]:
                if x == 0:
                    touched.add("left")
                if x == closure.GRID_SIZE - 1:
                    touched.add("right")
                if y == 0:
                    touched.add("top")
                if y == closure.GRID_SIZE - 1:
                    touched.add("bottom")
            if len(touched) == 1:
                candidates.append(component)
        return candidates[0] if len(candidates) == 1 else None
    raise ValueError(f"unknown selector rule {selector_rule}")


def transform_cells(cells: Iterable[Cell], program: str) -> tuple[Cell, ...]:
    return tuple(sorted(closure.apply_program_to_cell(program, x, y, color) for x, y, color in cells))


def selected_only_output(grid: Grid, selected_cells: Iterable[Cell], program: str) -> tuple[Grid, tuple[Cell, ...]]:
    selected = tuple(selected_cells)
    selected_coords = {(x, y) for x, y, _ in selected}
    result = closure.empty_grid()
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color != closure.BACKGROUND and (x, y) not in selected_coords:
                result[y][x] = int(color)
    transformed = transform_cells(selected, program)
    for x, y, color in transformed:
        if result[y][x] != closure.BACKGROUND:
            raise ValueError("selected transform collides with non-selected object")
        result[y][x] = int(color)
    output = closure.freeze_grid(result)
    if accidental_merge(output):
        raise ValueError("selected transform creates accidental component merge")
    return output, transformed


def isolate_cells(cells: Iterable[Cell]) -> Grid:
    return grid_from_cells(cells)


def accidental_merge(grid: Grid) -> bool:
    comps = connected_components(grid)
    cell_count = len(cells_from_grid(grid))
    return sum(component["size"] for component in comps) != cell_count


def shape_cells(shape_name: str, anchor: tuple[int, int], color: int) -> tuple[Cell, ...]:
    ax, ay = anchor
    return tuple((ax + dx, ay + dy, color) for dx, dy in SHAPES[shape_name])


def occupied_and_neighbor_cells(cells: Iterable[Cell]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for x, y, _ in cells:
        for nx, ny in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < closure.GRID_SIZE and 0 <= ny < closure.GRID_SIZE:
                blocked.add((nx, ny))
    return blocked


def place_shape(
    rng: random.Random,
    existing: list[Cell],
    shape_name: str,
    color: int,
    *,
    allow_border: bool = False,
    require_border: str | None = None,
) -> tuple[Cell, ...]:
    max_dx = max(dx for dx, _ in SHAPES[shape_name])
    max_dy = max(dy for _, dy in SHAPES[shape_name])
    anchors = []
    for ay in range(closure.GRID_SIZE - max_dy):
        for ax in range(closure.GRID_SIZE - max_dx):
            cells = shape_cells(shape_name, (ax, ay), color)
            if not allow_border and any(x in {0, closure.GRID_SIZE - 1} or y in {0, closure.GRID_SIZE - 1} for x, y, _ in cells):
                continue
            if require_border and not any(
                (require_border == "left" and x == 0)
                or (require_border == "right" and x == closure.GRID_SIZE - 1)
                or (require_border == "top" and y == 0)
                or (require_border == "bottom" and y == closure.GRID_SIZE - 1)
                for x, y, _ in cells
            ):
                continue
            blocked = occupied_and_neighbor_cells(existing)
            if any((x, y) in blocked for x, y, _ in cells):
                continue
            anchors.append(cells)
    if not anchors:
        raise ValueError("no legal shape anchor")
    return rng.choice(anchors)


def scene_cells_for_family(family: str, transform_program: str, seed_text: str) -> tuple[Grid, tuple[Cell, ...], tuple[Cell, ...]]:
    rng = random.Random(stable_int(seed_text))
    cells: list[Cell] = []
    marker_cells: tuple[Cell, ...] = ()
    colors = [1 + (stable_int(f"{seed_text}|color|{index}") % 4) for index in range(6)]
    if family == "largest_component_select_then_v2_transform":
        for shape_name, color in zip(("single", "domino_h", rng.choice(("tri_l", "tri_bar_h", "tri_bar_v"))), colors):
            cells.extend(place_shape(rng, cells, shape_name, color, allow_border=False))
    elif family == "unique_shape_select_then_v2_transform":
        if stable_int(seed_text) % 2 == 0:
            shape_names = ("domino_h", "domino_h", "domino_v")
        else:
            shape_names = ("domino_v", "domino_v", "domino_h")
        for shape_name, color in zip(shape_names, colors):
            cells.extend(place_shape(rng, cells, shape_name, color, allow_border=False))
    elif family == "nearest_marker_select_then_v2_transform":
        object_shapes = ("domino_h", "domino_v", "domino_h")
        object_colors = [1 + (stable_int(f"{seed_text}|nearest_color|{index}") % 3) for index in range(3)]
        for shape_name, color in zip(object_shapes, object_colors):
            cells.extend(place_shape(rng, cells, shape_name, color, allow_border=False))
        marker_cells = place_shape(rng, cells, "single", MARKER_COLOR, allow_border=True)
        cells.extend(marker_cells)
    elif family == "border_relation_select_then_v2_transform":
        boundary = ("left", "right", "top", "bottom")[stable_int(seed_text) % 4]
        cells.extend(place_shape(rng, cells, rng.choice(("single", "domino_h", "domino_v")), colors[0], allow_border=True, require_border=boundary))
        for shape_name, color in zip(("domino_h", "domino_v", "single"), colors[1:]):
            cells.extend(place_shape(rng, cells, shape_name, color, allow_border=False))
    else:
        raise ValueError(f"unknown family {family}")
    grid = grid_from_cells(cells)
    selector_rule = SELECTOR_RULE_BY_FAMILY[family]
    selected = select_component(grid, selector_rule)
    if selected is None:
        raise ValueError("intended selector is not unique")
    selected_cells = component_cells(selected, grid)
    output, transformed = selected_only_output(grid, selected_cells, transform_program)
    _ = output
    return grid, selected_cells, transformed


def make_support_pair(family: str, transform_program: str, seed_text: str) -> SupportPair:
    grid, selected, transformed = scene_cells_for_family(family, transform_program, seed_text)
    output, transformed = selected_only_output(grid, selected, transform_program)
    return SupportPair(input=grid, output=output, selected_cells_before=selected, selected_cells_after=transformed)


def make_isolated_support_pair(pair: SupportPair, transform_program: str) -> SupportPair:
    isolated_input = isolate_cells(pair.selected_cells_before)
    isolated_output = closure.apply_program_to_grid(isolated_input, transform_program)
    return SupportPair(
        input=isolated_input,
        output=isolated_output,
        selected_cells_before=pair.selected_cells_before,
        selected_cells_after=tuple(cells_from_grid(isolated_output)),
    )


def selector_agreement_key(component: dict[str, Any] | None) -> tuple[tuple[int, int], ...] | None:
    return None if component is None else tuple(component["coords"])


def validate_episode_fields(
    family: str,
    transform_program: str,
    supports: tuple[SupportPair, ...],
    query: Grid,
    target: Grid,
    selected: tuple[Cell, ...],
    transformed: tuple[Cell, ...],
    isolated_supports: tuple[SupportPair, ...],
    isolated_query: Grid,
    isolated_target: Grid,
) -> dict[str, Any]:
    selector_rule = SELECTOR_RULE_BY_FAMILY[family]
    scenes = [(pair.input, pair.selected_cells_before) for pair in supports] + [(query, selected)]
    selector_failures = []
    alternative_consistency: dict[str, list[bool]] = {rule: [] for rule in SELECTOR_RULE_BY_FAMILY.values() if rule != selector_rule}
    for grid, selected_cells in scenes:
        intended = select_component(grid, selector_rule)
        intended_key = selector_agreement_key(intended)
        selected_key = tuple(sorted((x, y) for x, y, _ in selected_cells))
        if intended_key != selected_key:
            selector_failures.append({"selector_rule": selector_rule, "selected": selected_key, "computed": intended_key})
        for alt_rule in alternative_consistency:
            alt = select_component(grid, alt_rule)
            alternative_consistency[alt_rule].append(selector_agreement_key(alt) == selected_key)
    transform_matches = [
        closure.selected_object_transform_matches(pair.selected_cells_before, pair.selected_cells_after)
        for pair in supports
    ]
    transform_intersection = sorted(set(transform_matches[0]).intersection(*map(set, transform_matches[1:]))) if transform_matches else []
    full_closure = closure.representability_check([pair.input for pair in supports], [pair.output for pair in supports], query, target)
    isolated_closure = closure.representability_check(
        [pair.input for pair in isolated_supports],
        [pair.output for pair in isolated_supports],
        isolated_query,
        isolated_target,
    )
    inconsistent_alternatives = [rule for rule, values in alternative_consistency.items() if not all(values)]
    return {
        "intended_rule_is_consistent": selector_failures == [],
        "selector_failures": selector_failures,
        "alternative_rule_consistency": {rule: all(values) for rule, values in alternative_consistency.items()},
        "at_least_one_alternative_rule_is_inconsistent": bool(inconsistent_alternatives),
        "selection_is_unique_in_every_support_and_query_grid": selector_failures == [],
        "transform_rule_is_identifiable_from_support_pairs": transform_intersection == [transform_program],
        "transform_matches_by_support_pair": transform_matches,
        "transform_intersection": transform_intersection,
        "full_scene_representability": full_closure,
        "isolated_target_representability": isolated_closure,
        "full_scene_outside_declared_closure": not full_closure["representable_by_declared_v2_closure"],
        "isolated_target_inside_declared_closure": isolated_closure["matching_programs"] == [transform_program],
        "passes": (
            selector_failures == []
            and bool(inconsistent_alternatives)
            and transform_intersection == [transform_program]
            and not full_closure["representable_by_declared_v2_closure"]
            and isolated_closure["matching_programs"] == [transform_program]
        ),
    }


def make_episode(family: str, transform_program: str, variant: int, *, max_attempts: int = 512) -> tuple[ProbeEpisode, list[str]]:
    rejections = []
    for attempt in range(max_attempts):
        seed_prefix = f"r5a2a|{family}|{transform_program}|variant={variant}|attempt={attempt}"
        try:
            supports = tuple(make_support_pair(family, transform_program, f"{seed_prefix}|support={index}") for index in range(3))
            query_pair = make_support_pair(family, transform_program, f"{seed_prefix}|query")
            isolated_supports = tuple(make_isolated_support_pair(pair, transform_program) for pair in supports)
            isolated_query = isolate_cells(query_pair.selected_cells_before)
            isolated_target = closure.apply_program_to_grid(isolated_query, transform_program)
            validation = validate_episode_fields(
                family,
                transform_program,
                supports,
                query_pair.input,
                query_pair.output,
                query_pair.selected_cells_before,
                query_pair.selected_cells_after,
                isolated_supports,
                isolated_query,
                isolated_target,
            )
            if not validation["passes"]:
                rejections.append("validation_failed")
                continue
            episode = ProbeEpisode(
                episode_id=f"{family}:{transform_program}:variant={variant:03d}",
                probe_family=family,
                selector_rule=SELECTOR_RULE_BY_FAMILY[family],
                transform_program=transform_program,
                split_group=SPLIT_BY_TRANSFORM[transform_program],
                supports=supports,
                query=query_pair.input,
                target=query_pair.output,
                selected_cells_before=query_pair.selected_cells_before,
                selected_cells_after=query_pair.selected_cells_after,
                isolated_supports=isolated_supports,
                isolated_query=isolated_query,
                isolated_target=isolated_target,
                validation=validation,
            )
            return episode, rejections
        except ValueError as exc:
            rejections.append(str(exc).split(":")[0])
    raise RuntimeError(f"failed to build {family} {transform_program} variant {variant}: {Counter(rejections)}")


def generate_probe_panel(variants_per_combo: int = 6) -> tuple[list[ProbeEpisode], dict[str, Any]]:
    episodes = []
    rejected: dict[str, Counter[str]] = defaultdict(Counter)
    for family in FAMILY_ORDER:
        for transform in TRANSFORM_PROGRAMS:
            for variant in range(variants_per_combo):
                episode, reasons = make_episode(family, transform, variant)
                episodes.append(episode)
                rejected[family].update(reasons)
    manifest = {
        "variants_per_selector_transform_combo": variants_per_combo,
        "family_order": list(FAMILY_ORDER),
        "transform_programs": list(TRANSFORM_PROGRAMS),
        "episode_count": len(episodes),
        "panel_hash": closure.stable_hash([episode.to_record() for episode in episodes]),
        "rejection_reasons_by_family": {family: dict(counter) for family, counter in sorted(rejected.items())},
    }
    return episodes, manifest


def choose_shortcut_component(grid: Grid, shortcut: str) -> dict[str, Any] | None:
    components = non_marker_components(grid)
    if not components:
        return None
    if shortcut == "always_choose_largest":
        return sorted(components, key=lambda row: (-row["size"], row["coords"]))[0]
    if shortcut == "always_choose_smallest":
        return sorted(components, key=lambda row: (row["size"], row["coords"]))[0]
    if shortcut == "always_choose_nearest_corner":
        corners = ((0, 0), (0, closure.GRID_SIZE - 1), (closure.GRID_SIZE - 1, 0), (closure.GRID_SIZE - 1, closure.GRID_SIZE - 1))
        return sorted(
            components,
            key=lambda row: (
                min(abs(row["centroid"][0] - cx) + abs(row["centroid"][1] - cy) for cx, cy in corners),
                row["coords"],
            ),
        )[0]
    if shortcut == "always_choose_rarest_color":
        color_counts = Counter(component["dominant_color"] for component in components)
        return sorted(components, key=lambda row: (color_counts[row["dominant_color"]], row["coords"]))[0]
    if shortcut == "always_choose_fixed_position":
        anchor = (0, 0)
        return sorted(
            components,
            key=lambda row: (
                min(abs(x - anchor[0]) + abs(y - anchor[1]) for x, y in row["coords"]),
                row["coords"],
            ),
        )[0]
    raise ValueError(f"unknown shortcut {shortcut}")


def shortcut_prediction(episode: ProbeEpisode, shortcut: str) -> Grid:
    if shortcut == "always_transform_all_objects":
        return closure.apply_program_to_grid(episode.query, episode.transform_program)
    component = choose_shortcut_component(episode.query, shortcut)
    if component is None:
        return episode.query
    selected = component_cells(component, episode.query)
    try:
        output, _ = selected_only_output(episode.query, selected, episode.transform_program)
        return output
    except ValueError:
        return episode.query


def shortcut_audit(episodes: list[ProbeEpisode]) -> dict[str, Any]:
    shortcuts = (
        "always_choose_largest",
        "always_choose_smallest",
        "always_choose_nearest_corner",
        "always_choose_rarest_color",
        "always_choose_fixed_position",
        "always_transform_all_objects",
    )
    by_family: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        family_rows = [episode for episode in episodes if episode.probe_family == family]
        rows = {}
        for shortcut in shortcuts:
            exact = sum(1 for episode in family_rows if shortcut_prediction(episode, shortcut) == episode.target)
            rows[shortcut] = {"exact_count": exact, "episode_count": len(family_rows), "exact": exact / max(len(family_rows), 1)}
        max_shortcut = max(row["exact"] for row in rows.values()) if rows else 0.0
        by_family[family] = {
            "shortcut_scores": rows,
            "max_shortcut_exact": max_shortcut,
            "passes_threshold": max_shortcut <= SHORTCUT_THRESHOLD,
            "threshold": SHORTCUT_THRESHOLD,
        }
    return {"by_family": by_family, "threshold": SHORTCUT_THRESHOLD}


def split_audit(episodes: list[ProbeEpisode]) -> dict[str, Any]:
    by_cell = Counter((episode.probe_family, episode.transform_program, episode.split_group) for episode in episodes)
    semantic_train = {(episode.probe_family, episode.transform_program) for episode in episodes if episode.split_group == "train_visible"}
    semantic_held = {(episode.probe_family, episode.transform_program) for episode in episodes if episode.split_group == "held_out"}
    counts_by_family = Counter(episode.probe_family for episode in episodes)
    counts_by_transform = Counter(episode.transform_program for episode in episodes)
    counts_by_split = Counter(episode.split_group for episode in episodes)
    expected_per_cell = len(episodes) // (len(FAMILY_ORDER) * len(TRANSFORM_PROGRAMS))
    return {
        "composition_cells": [
            {"probe_family": family, "transform_program": transform, "split_group": split, "count": count}
            for (family, transform, split), count in sorted(by_cell.items())
        ],
        "train_visible_cells": [list(item) for item in sorted(semantic_train)],
        "held_out_cells": [list(item) for item in sorted(semantic_held)],
        "semantic_collision_between_train_and_heldout": bool(semantic_train.intersection(semantic_held)),
        "counts_by_family": dict(counts_by_family),
        "counts_by_transform": dict(counts_by_transform),
        "counts_by_split_group": dict(counts_by_split),
        "exact_balance_across_selector_rules": len(set(counts_by_family.values())) == 1,
        "exact_balance_across_transform_programs": len(set(counts_by_transform.values())) == 1,
        "exact_balance_across_composition_cells": all(count == expected_per_cell for count in by_cell.values()),
        "passed": (
            not semantic_train.intersection(semantic_held)
            and len(set(counts_by_family.values())) == 1
            and len(set(counts_by_transform.values())) == 1
            and all(count == expected_per_cell for count in by_cell.values())
        ),
    }


def counterbalance_report(episodes: list[ProbeEpisode]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        rows = [episode for episode in episodes if episode.probe_family == family]
        target_colors = Counter(color for episode in rows for _, _, color in episode.selected_cells_before)
        target_sizes = Counter(len(episode.selected_cells_before) for episode in rows)
        target_shapes = Counter(tuple(sorted((x - min(cx for cx, _, _ in episode.selected_cells_before), y - min(cy for _, cy, _ in episode.selected_cells_before)) for x, y, _ in episode.selected_cells_before)) for episode in rows)
        target_positions = Counter((round(sum(x for x, _, _ in episode.selected_cells_before) / len(episode.selected_cells_before), 1), round(sum(y for _, y, _ in episode.selected_cells_before) / len(episode.selected_cells_before), 1)) for episode in rows)
        distractor_counts = Counter(len(non_marker_components(episode.query)) - 1 for episode in rows)
        by_family[family] = {
            "target_color_histogram": dict(target_colors),
            "target_size_histogram": dict(target_sizes),
            "target_shape_count": len(target_shapes),
            "target_centroid_count": len(target_positions),
            "distractor_count_histogram": dict(distractor_counts),
            "transform_program_histogram": dict(Counter(episode.transform_program for episode in rows)),
            "split_group_histogram": dict(Counter(episode.split_group for episode in rows)),
        }
    return by_family


def generator_audit(episodes: list[ProbeEpisode], manifest: dict[str, Any]) -> dict[str, Any]:
    shortcut = shortcut_audit(episodes)
    split = split_audit(episodes)
    by_family: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        rows = [episode for episode in episodes if episode.probe_family == family]
        selector_ambiguity_count = sum(1 for episode in rows if not episode.validation["selection_is_unique_in_every_support_and_query_grid"])
        transform_ambiguity_count = sum(1 for episode in rows if not episode.validation["transform_rule_is_identifiable_from_support_pairs"])
        closure_in_count = sum(1 for episode in rows if episode.validation["full_scene_representability"]["representable_by_declared_v2_closure"])
        isolated_in_count = sum(1 for episode in rows if episode.validation["isolated_target_representability"]["representable_by_declared_v2_closure"])
        support_failures = sum(1 for episode in rows if not episode.validation["at_least_one_alternative_rule_is_inconsistent"])
        shortcut_pass = shortcut["by_family"][family]["passes_threshold"]
        by_family[family] = {
            "episode_count": len(rows),
            "accepted_episode_count": len(rows),
            "rejected_episode_reasons": manifest["rejection_reasons_by_family"].get(family, {}),
            "selector_ambiguity_count": selector_ambiguity_count,
            "transform_ambiguity_count": transform_ambiguity_count,
            "support_identifiability_failure_count": support_failures,
            "full_scene_closure_representable_count": closure_in_count,
            "isolated_target_closure_representable_count": isolated_in_count,
            "shortcut_validity": shortcut["by_family"][family],
            "generator_valid_for_toolbox_expansion": (
                selector_ambiguity_count == 0
                and transform_ambiguity_count == 0
                and support_failures == 0
                and closure_in_count == 0
                and isolated_in_count == len(rows)
                and shortcut_pass
            ),
            "human_readable_samples": [episode.to_record() for episode in rows[:2]],
        }
    return {
        "manifest": manifest,
        "object_definition_rules": {
            "connectedness": "4-connected non-background components",
            "marker_color": MARKER_COLOR,
            "marker_excluded_from_candidate_object_selectors": True,
            "spacing": "placement rejects overlap and 4-neighbor touching between distinct objects",
        },
        "counterbalance_report": counterbalance_report(episodes),
        "split_audit": split,
        "shortcut_audit": shortcut,
        "by_family": by_family,
        "all_non_largest_families_generator_valid": all(
            row["generator_valid_for_toolbox_expansion"]
            for family, row in by_family.items()
            if family != "largest_component_select_then_v2_transform"
        ),
    }


def panel_to_records(episodes: list[ProbeEpisode]) -> list[dict[str, Any]]:
    return [episode.to_record() for episode in episodes]


def load_episode_record(record: dict[str, Any]) -> ProbeEpisode:
    supports = tuple(
        SupportPair(
            input=closure.freeze_grid(pair["input"]),
            output=closure.freeze_grid(pair["output"]),
            selected_cells_before=tuple(tuple(int(v) for v in cell) for cell in pair["selected_cells_before"]),
            selected_cells_after=tuple(tuple(int(v) for v in cell) for cell in pair["selected_cells_after"]),
        )
        for pair in record["supports"]
    )
    isolated_supports = tuple(
        SupportPair(
            input=closure.freeze_grid(pair["input"]),
            output=closure.freeze_grid(pair["output"]),
            selected_cells_before=tuple(tuple(int(v) for v in cell) for cell in pair["selected_cells_before"]),
            selected_cells_after=tuple(tuple(int(v) for v in cell) for cell in pair["selected_cells_after"]),
        )
        for pair in record["isolated_supports"]
    )
    return ProbeEpisode(
        episode_id=str(record["episode_id"]),
        probe_family=str(record["probe_family"]),
        selector_rule=str(record["selector_rule"]),
        transform_program=str(record["transform_program"]),
        split_group=str(record["split_group"]),
        supports=supports,
        query=closure.freeze_grid(record["query"]),
        target=closure.freeze_grid(record["target"]),
        selected_cells_before=tuple(tuple(int(v) for v in cell) for cell in record["selected_cells_before"]),
        selected_cells_after=tuple(tuple(int(v) for v in cell) for cell in record["selected_cells_after"]),
        isolated_supports=isolated_supports,
        isolated_query=closure.freeze_grid(record["isolated_query"]),
        isolated_target=closure.freeze_grid(record["isolated_target"]),
        validation=copy.deepcopy(record["validation"]),
    )
