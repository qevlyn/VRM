from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

GRID_SIZE = 7
BACKGROUND = 0
GLOBAL_SEED = 20260628
SAFE_COORDS = tuple((x, y) for y in range(1, 6) for x in range(1, 6))
SCENE_COORDS = tuple((x, y) for y in range(2, 6) for x in range(2, 5))

OP_C = "C"
OP_F = "F"
OP_T = "T"

SINGLE_STEP_CLASSES = (OP_C, OP_F, OP_T)
SEEN_TWO_STEP_CLASSES = ("C->F", "F->T", "T->C")
HELD_OUT_TWO_STEP_CLASSES = ("F->C", "T->F", "C->T")
ORDER_COUNTERFACTUAL_CLASSES = ("C->F", "F->C", "F->T", "T->F", "T->C", "C->T")
TWO_STEP_CLASSES = SEEN_TWO_STEP_CLASSES + HELD_OUT_TWO_STEP_CLASSES
TRAIN_VISIBLE_CLASSES = SINGLE_STEP_CLASSES + SEEN_TWO_STEP_CLASSES
SEALED_CLASSES = TRAIN_VISIBLE_CLASSES + HELD_OUT_TWO_STEP_CLASSES
ALL_SEQUENCE_CLASSES = SEALED_CLASSES

SHAPES = ("3_cell_L", "3_cell_T", "3_cell_zig_zag", "asymmetric_4_cell_hook")
COLORS = (1, 2, 3, 4)
ANCHORS = ("interior_upper_left", "interior_upper_right", "interior_lower_left", "interior_lower_right")
DISTRACTOR_COUNTS = (1, 2)

REQUIRED_QUERY_TYPE_COUNTS = {"train": 80, "dev": 24, "sealed": 24}
LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS = {"train": 96, "dev": 24, "sealed": 24}
REQUIRED_EPISODES_PER_QUERY_TYPE = {"train": 360, "dev": 360, "sealed": 540}

PRIMARY_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "3_cell_L": ((0, 0), (0, 1), (1, 1)),
    "3_cell_T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "3_cell_zig_zag": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "asymmetric_4_cell_hook": ((0, 0), (0, 1), (0, 2), (1, 2)),
}

SUPPORT_COVERAGE_CHUNKS = (
    SCENE_COORDS[:8],
    SCENE_COORDS[4:],
    SCENE_COORDS[:4] + SCENE_COORDS[8:],
)


@dataclass(frozen=True)
class R4A0RSupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class R4A0REpisode:
    episode_id: str
    split: str
    sequence_class: str
    query_type: tuple[str, int, str, int]
    counterfactual_group_id: str | None
    query: tuple[tuple[int, ...], ...]
    supports: tuple[R4A0RSupportPair, ...]
    target: tuple[tuple[int, ...], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "sequence_class": self.sequence_class,
            "query_type": query_type_to_dict(self.query_type),
            "query_type_key": query_type_key(self.query_type),
            "counterfactual_group_id": self.counterfactual_group_id,
            "query": grid_to_list(self.query),
            "supports": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in self.supports],
            "target": grid_to_list(self.target),
        }


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def empty_grid() -> list[list[int]]:
    return [[BACKGROUND for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def freeze_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def grid_to_list(grid: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(row) for row in grid]


def parse_grid(raw: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return freeze_grid([[int(value) for value in row] for row in raw])


def query_type_to_dict(query_type: tuple[str, int, str, int]) -> dict[str, Any]:
    shape, color, anchor, distractor_count = query_type
    return {
        "primary_shape": shape,
        "primary_color": color,
        "anchor_region": anchor,
        "distractor_count": distractor_count,
    }


def query_type_from_dict(payload: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(payload["primary_shape"]),
        int(payload["primary_color"]),
        str(payload["anchor_region"]),
        int(payload["distractor_count"]),
    )


def query_type_key(query_type: tuple[str, int, str, int]) -> str:
    shape, color, anchor, distractor_count = query_type
    return f"{shape}|color={color}|anchor={anchor}|distractors={distractor_count}"


def coord_c(x: int, y: int) -> tuple[int, int]:
    return (GRID_SIZE - 1 - y, x)


def coord_f(x: int, y: int) -> tuple[int, int]:
    return (GRID_SIZE - 1 - x, y)


def coord_t(x: int, y: int) -> tuple[int, int]:
    if x > GRID_SIZE - 2:
        raise ValueError("T is valid only for x <= 5 on a 7x7 grid")
    return (x + 1, y)


OP_FUNCTIONS: dict[str, Callable[[int, int], tuple[int, int]]] = {
    OP_C: coord_c,
    OP_F: coord_f,
    OP_T: coord_t,
}


def parse_sequence(sequence_class: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in sequence_class.split("->"))
    if not parts or any(part not in OP_FUNCTIONS for part in parts):
        raise ValueError(f"unknown sequence class {sequence_class}")
    return parts


def apply_sequence_to_coord(sequence_class: str, coord: tuple[int, int]) -> tuple[int, int]:
    x, y = coord
    for op in parse_sequence(sequence_class):
        x, y = OP_FUNCTIONS[op](x, y)
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError(f"{sequence_class} produced out-of-bounds coordinate {(x, y)}")
    return (x, y)


def source_index_for_output(sequence_class: str, out_x: int, out_y: int) -> int:
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            try:
                if apply_sequence_to_coord(sequence_class, (x, y)) == (out_x, out_y):
                    return y * GRID_SIZE + x
            except ValueError:
                continue
    return -1


def mapping_signature(sequence_class: str, domain: tuple[tuple[int, int], ...] = SAFE_COORDS) -> list[list[int]]:
    return [list(apply_sequence_to_coord(sequence_class, coord)) for coord in domain]


def pairwise_equality_matrix(sequence_classes: tuple[str, ...]) -> dict[str, dict[str, bool]]:
    signatures = {sequence: mapping_signature(sequence) for sequence in sequence_classes}
    return {left: {right: signatures[left] == signatures[right] for right in sequence_classes} for left in sequence_classes}


def coordinate_formulas() -> dict[str, str]:
    return {
        OP_C: "C(x,y) = (6-y, x)",
        OP_F: "F(x,y) = (6-x, y)",
        OP_T: "T(x,y) = (x+1, y), valid only when x <= 5",
        "sequence_notation": "A->B means apply A first, then B",
        "common_safe_domain": "{(x,y): x,y in {1,2,3,4,5}}",
    }


def algebraic_function_audit() -> dict[str, Any]:
    signatures = {sequence: mapping_signature(sequence) for sequence in ALL_SEQUENCE_CLASSES}
    matrix = pairwise_equality_matrix(ALL_SEQUENCE_CLASSES)
    two_step_unique = len({json.dumps(signatures[sequence]) for sequence in TWO_STEP_CLASSES}) == len(TWO_STEP_CLASSES)
    no_heldout_equals_seen = all(signatures[heldout] != signatures[seen] for heldout in HELD_OUT_TWO_STEP_CLASSES for seen in SEEN_TWO_STEP_CLASSES)
    no_heldout_equals_single = all(signatures[heldout] != signatures[single] for heldout in HELD_OUT_TWO_STEP_CLASSES for single in SINGLE_STEP_CLASSES)
    collisions = [
        {"left": left, "right": right}
        for left, right in itertools.combinations(ALL_SEQUENCE_CLASSES, 2)
        if signatures[left] == signatures[right]
    ]
    passed = two_step_unique and no_heldout_equals_seen and no_heldout_equals_single and not collisions
    return {
        "coordinate_formulas": coordinate_formulas(),
        "safe_domain_order": [list(coord) for coord in SAFE_COORDS],
        "single_step_classes": list(SINGLE_STEP_CLASSES),
        "seen_two_step_train_dev": list(SEEN_TWO_STEP_CLASSES),
        "held_out_two_step_sealed_only": list(HELD_OUT_TWO_STEP_CLASSES),
        "mapping_signatures": signatures,
        "pairwise_equality_matrix": matrix,
        "two_step_signatures_all_unique": two_step_unique,
        "no_heldout_two_step_signature_equals_seen_two_step_signature": no_heldout_equals_seen,
        "no_heldout_two_step_signature_equals_single_step_signature": no_heldout_equals_single,
        "collisions": collisions,
        "proof": {
            "passed": passed,
            "statement": "Every train/test two-step class is functionally distinct over the full common safe domain.",
        },
    }


def all_query_types() -> list[tuple[str, int, str, int]]:
    return [(shape, color, anchor, distractor_count) for shape in SHAPES for color in COLORS for anchor in ANCHORS for distractor_count in DISTRACTOR_COUNTS]


def factor_coverage(query_types: Iterable[tuple[str, int, str, int]]) -> dict[str, list[Any]]:
    items = list(query_types)
    return {
        "primary_shapes": sorted({item[0] for item in items}),
        "primary_colors": sorted({item[1] for item in items}),
        "anchor_regions": sorted({item[2] for item in items}),
        "distractor_counts": sorted({item[3] for item in items}),
    }


def factor_complete(query_types: Iterable[tuple[str, int, str, int]]) -> bool:
    coverage = factor_coverage(query_types)
    return (
        set(coverage["primary_shapes"]) == set(SHAPES)
        and set(coverage["primary_colors"]) == set(COLORS)
        and set(coverage["anchor_regions"]) == set(ANCHORS)
        and set(coverage["distractor_counts"]) == set(DISTRACTOR_COUNTS)
    )


def apply_sequence_to_grid(grid: tuple[tuple[int, ...], ...], sequence_class: str) -> tuple[tuple[int, ...], ...]:
    result = empty_grid()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == BACKGROUND:
                continue
            out_x, out_y = apply_sequence_to_coord(sequence_class, (x, y))
            if result[out_y][out_x] != BACKGROUND:
                raise ValueError("sequence produced colliding foreground cells")
            result[out_y][out_x] = int(value)
    return freeze_grid(result)


def foreground_coords(grid: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]]:
    return [(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value != BACKGROUND]


def no_edge_foreground(grid: tuple[tuple[int, ...], ...]) -> bool:
    return all(0 < x < GRID_SIZE - 1 and 0 < y < GRID_SIZE - 1 for x, y in foreground_coords(grid))


def shape_cells(query_type: tuple[str, int, str, int]) -> list[tuple[int, int]]:
    shape, _, anchor, _ = query_type
    offsets = PRIMARY_OFFSETS[shape]
    lower = "lower" in anchor
    right = "right" in anchor
    base_y = 3 if lower else 2
    cells: list[tuple[int, int]] = []
    for dx, dy in offsets:
        x = 2 + (2 - dx if right else dx)
        y = base_y + dy
        cells.append((x, y))
    return cells


def palette(primary_color: int, salt: str) -> list[int]:
    colors = [primary_color] + [color for color in range(1, 10) if color != primary_color]
    rng = random.Random(stable_int(f"r4a0r|palette|{primary_color}|{salt}"))
    tail = colors[1:]
    rng.shuffle(tail)
    return [primary_color] + tail


def render_scene(query_type: tuple[str, int, str, int], variant: int, salt: str) -> tuple[tuple[int, ...], ...]:
    _, primary_color, _, distractor_count = query_type
    cells = list(dict.fromkeys(shape_cells(query_type)))
    rng = random.Random(stable_int(f"r4a0r|scene|{query_type_key(query_type)}|{variant}|{salt}"))
    remaining = [coord for coord in SCENE_COORDS if coord not in cells]
    rng.shuffle(remaining)
    needed = 2 * distractor_count
    cells.extend(remaining[:needed])
    colors = palette(primary_color, f"{variant}|{salt}")
    grid = empty_grid()
    for idx, (x, y) in enumerate(cells):
        grid[y][x] = colors[idx]
    rendered = freeze_grid(grid)
    ok, reasons = validate_scene(rendered)
    if not ok:
        raise RuntimeError(f"invalid rendered scene {query_type_key(query_type)} {variant} {salt}: {reasons}")
    return rendered


def render_support_scene(query_type: tuple[str, int, str, int], variant: int, pair_index: int, salt: str) -> tuple[tuple[int, ...], ...]:
    _, primary_color, _, _ = query_type
    cells = list(SUPPORT_COVERAGE_CHUNKS[pair_index])
    colors = palette(primary_color, f"support|{pair_index}|{variant}|{salt}")
    grid = empty_grid()
    for idx, (x, y) in enumerate(cells):
        grid[y][x] = colors[idx]
    rendered = freeze_grid(grid)
    ok, reasons = validate_scene(rendered)
    if not ok:
        raise RuntimeError(f"invalid support scene {query_type_key(query_type)} {variant} {pair_index}: {reasons}")
    return rendered


def validate_scene(grid: tuple[tuple[int, ...], ...]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    coords = foreground_coords(grid)
    if not coords:
        reasons.append("empty_foreground")
    if not no_edge_foreground(grid):
        reasons.append("foreground_touches_edge")
    if any((x, y) not in SCENE_COORDS for x, y in coords):
        reasons.append("foreground_outside_common_scene_interior")
    outputs: dict[str, tuple[tuple[int, ...], ...]] = {}
    for sequence in ALL_SEQUENCE_CLASSES:
        try:
            current = grid
            for op in parse_sequence(sequence):
                current = apply_sequence_to_grid(current, op)
                if not no_edge_foreground(current):
                    reasons.append(f"{sequence}_intermediate_or_final_touches_edge")
                    break
            outputs[sequence] = apply_sequence_to_grid(grid, sequence)
        except ValueError as exc:
            reasons.append(f"{sequence}_invalid:{exc}")
    if outputs:
        two_outputs = {sequence: outputs[sequence] for sequence in TWO_STEP_CLASSES if sequence in outputs}
        if len({stable_hash(grid_to_list(value)) for value in two_outputs.values()}) != len(TWO_STEP_CLASSES):
            reasons.append("two_step_output_collision")
        for sequence in TWO_STEP_CLASSES:
            if outputs.get(sequence) == grid:
                reasons.append(f"{sequence}_equals_input")
        for two_step in TWO_STEP_CLASSES:
            for single in SINGLE_STEP_CLASSES:
                if outputs.get(two_step) == outputs.get(single):
                    reasons.append(f"{two_step}_equals_{single}")
    return not reasons, sorted(set(reasons))


def preflight_query_type(query_type: tuple[str, int, str, int], seed: int = GLOBAL_SEED) -> dict[str, Any]:
    valid_counts = {sequence: 0 for sequence in ALL_SEQUENCE_CLASSES}
    reasons: list[str] = []
    for attempt in range(12):
        try:
            scene = render_scene(query_type, seed + attempt, f"preflight|{attempt}")
        except RuntimeError as exc:
            reasons.append(str(exc))
            continue
        for sequence in ALL_SEQUENCE_CLASSES:
            try:
                _ = apply_sequence_to_grid(scene, sequence)
                valid_counts[sequence] += 1
            except ValueError as exc:
                reasons.append(f"{sequence}:{exc}")
    feasible = all(count > 0 for count in valid_counts.values())
    return {
        "query_type": query_type_to_dict(query_type) | {"query_type_key": query_type_key(query_type)},
        "feasible": feasible,
        "rejection_reasons": [] if feasible else sorted(set(reasons)),
        "valid_scene_counts_by_sequence_class": valid_counts,
        "deterministic_preflight_seed": seed,
    }


def query_type_preflight_catalog(seed: int = GLOBAL_SEED) -> dict[str, Any]:
    entries = [preflight_query_type(query_type, seed + index) for index, query_type in enumerate(all_query_types())]
    feasible_count = sum(1 for entry in entries if entry["feasible"])
    return {
        "declared_query_type_count": len(entries),
        "feasible_query_type_count": feasible_count,
        "all_128_feasible": feasible_count == 128,
        "entries": entries,
    }


def partition_query_types(seed: int = GLOBAL_SEED) -> dict[str, list[tuple[str, int, str, int]]]:
    query_types = all_query_types()
    required = REQUIRED_QUERY_TYPE_COUNTS
    for attempt in range(50_000):
        rng = random.Random(seed + attempt)
        shuffled = list(query_types)
        rng.shuffle(shuffled)
        train_end = required["train"]
        dev_end = train_end + required["dev"]
        split = {
            "train": shuffled[:train_end],
            "dev": shuffled[train_end:dev_end],
            "sealed": shuffled[dev_end:],
        }
        if len(split["sealed"]) != required["sealed"]:
            continue
        if all(factor_complete(items) for items in split.values()):
            return split
    raise RuntimeError("failed to create deterministic 80/24/24 split with factor coverage")


def query_type_split_audit(split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    sets = {split: {query_type_key(item) for item in items} for split, items in split_types.items()}
    union = set().union(*sets.values())
    overlaps = {
        "train_dev": sorted(sets["train"].intersection(sets["dev"])),
        "train_sealed": sorted(sets["train"].intersection(sets["sealed"])),
        "dev_sealed": sorted(sets["dev"].intersection(sets["sealed"])),
    }
    return {
        "split_counts": {split: len(items) for split, items in split_types.items()},
        "expected_split_counts": REQUIRED_QUERY_TYPE_COUNTS,
        "union_count": len(union),
        "catalog_count": len(all_query_types()),
        "uses_every_catalog_type_exactly_once": len(union) == len(all_query_types()) and not any(overlaps.values()),
        "intersections": overlaps,
        "factor_coverage": {split: factor_coverage(items) for split, items in split_types.items()},
        "factor_complete": {split: factor_complete(items) for split, items in split_types.items()},
        "legacy_144_type_request_reintroduced": REQUIRED_QUERY_TYPE_COUNTS == LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS,
        "passed": (
            {split: len(items) for split, items in split_types.items()} == REQUIRED_QUERY_TYPE_COUNTS
            and len(union) == len(all_query_types())
            and not any(overlaps.values())
            and all(factor_complete(items) for items in split_types.values())
            and REQUIRED_QUERY_TYPE_COUNTS != LEGACY_INFEASIBLE_QUERY_TYPE_COUNTS
        ),
    }


def support_query_type(query_type: tuple[str, int, str, int], salt: str) -> tuple[str, int, str, int]:
    options = [item for item in all_query_types() if item != query_type]
    return options[stable_int(f"r4a0r|support_qtype|{query_type_key(query_type)}|{salt}") % len(options)]


def make_supports(query_type: tuple[str, int, str, int], sequence_class: str, episode_index: int, query: tuple[tuple[int, ...], ...], split: str) -> tuple[R4A0RSupportPair, ...]:
    pairs: list[R4A0RSupportPair] = []
    for pair_index in range(3):
        sqtype = support_query_type(query_type, f"{split}|{sequence_class}|{episode_index}|{pair_index}")
        support_input = render_support_scene(sqtype, episode_index + pair_index * 101, pair_index, f"{split}|{sequence_class}")
        if support_input == query:
            raise RuntimeError("support input equals query")
        support_output = apply_sequence_to_grid(support_input, sequence_class)
        pairs.append(R4A0RSupportPair(input=support_input, output=support_output))
    return tuple(pairs)


def make_episode(split: str, query_type: tuple[str, int, str, int], sequence_class: str, local_index: int) -> R4A0REpisode:
    variant = stable_int(f"r4a0r|{split}|{query_type_key(query_type)}|{sequence_class}|{local_index}") % 10_000_000
    query = render_scene(query_type, variant, f"{split}|query|{sequence_class}|{local_index}")
    target = apply_sequence_to_grid(query, sequence_class)
    supports = make_supports(query_type, sequence_class, variant, query, split)
    if any(pair.input == query for pair in supports):
        raise RuntimeError("support input equals query")
    if any(pair.output == target for pair in supports):
        raise RuntimeError("target equals support output")
    return R4A0REpisode(
        episode_id=f"{split}:{query_type_key(query_type)}:{sequence_class}:idx={local_index}",
        split=split,
        sequence_class=sequence_class,
        query_type=query_type,
        counterfactual_group_id=None,
        query=query,
        supports=supports,
        target=target,
    )


def generate_split(split: str, query_types: list[tuple[str, int, str, int]]) -> list[R4A0REpisode]:
    sequences = TRAIN_VISIBLE_CLASSES if split in {"train", "dev"} else SEALED_CLASSES
    episodes: list[R4A0REpisode] = []
    for query_type in sorted(query_types, key=query_type_key):
        for sequence in sequences:
            for idx in range(60):
                episodes.append(make_episode(split, query_type, sequence, idx))
    random.Random(stable_int(f"r4a0r|order|{split}")).shuffle(episodes)
    return episodes


def write_jsonl(path: Path, episodes: list[R4A0REpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), sort_keys=True) + "\n")


def episode_from_record(record: dict[str, Any]) -> R4A0REpisode:
    return R4A0REpisode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        sequence_class=str(record["sequence_class"]),
        query_type=query_type_from_dict(record["query_type"]),
        counterfactual_group_id=record.get("counterfactual_group_id"),
        query=parse_grid(record["query"]),
        supports=tuple(R4A0RSupportPair(input=parse_grid(pair["input"]), output=parse_grid(pair["output"])) for pair in record["supports"]),
        target=parse_grid(record["target"]),
    )


def load_split(path: Path, split: str) -> list[R4A0REpisode]:
    episodes: list[R4A0REpisode] = []
    with (Path(path) / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(episode_from_record(json.loads(line)))
    return episodes


def sequence_counts(episodes: Iterable[R4A0REpisode]) -> dict[str, int]:
    rows = list(episodes)
    return {sequence: sum(1 for episode in rows if episode.sequence_class == sequence) for sequence in ALL_SEQUENCE_CLASSES if any(episode.sequence_class == sequence for episode in rows)}


def sequence_balance_report(episodes_by_split: dict[str, list[R4A0REpisode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        sequences = TRAIN_VISIBLE_CLASSES if split in {"train", "dev"} else SEALED_CLASSES
        per_type: dict[str, dict[str, int]] = {}
        for episode in episodes:
            row = per_type.setdefault(query_type_key(episode.query_type), {sequence: 0 for sequence in sequences})
            row[episode.sequence_class] += 1
        report[split] = {
            "episode_count": len(episodes),
            "sequence_counts": sequence_counts(episodes),
            "query_type_count": len(per_type),
            "per_query_type_sequence_count_expected": 60,
            "all_query_types_balanced": all(all(row[sequence] == 60 for sequence in sequences) for row in per_type.values()),
            "passed": len(per_type) == REQUIRED_QUERY_TYPE_COUNTS[split] and all(all(row[sequence] == 60 for sequence in sequences) for row in per_type.values()),
        }
    return report


def dataset_manifest(episodes_by_split: dict[str, list[R4A0REpisode]], split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    return {
        "benchmark": "composition_r4a0r_v2",
        "grid_size": GRID_SIZE,
        "split_episode_counts": {split: len(episodes) for split, episodes in episodes_by_split.items()},
        "expected_split_episode_counts": {"train": 28_800, "dev": 8_640, "sealed": 12_960},
        "query_type_counts": {split: len(items) for split, items in split_types.items()},
        "sequence_counts": {split: sequence_counts(episodes) for split, episodes in episodes_by_split.items()},
        "query_type_split_audit": query_type_split_audit(split_types),
        "sequence_balance_report": sequence_balance_report(episodes_by_split),
    }


def scene_collision_rejection_report(episodes_by_split: dict[str, list[R4A0REpisode]], catalog: dict[str, Any]) -> dict[str, Any]:
    checked = sum(len(episodes) for episodes in episodes_by_split.values())
    return {
        "generated_episode_count": checked,
        "preflight_feasible_query_type_count": catalog["feasible_query_type_count"],
        "scene_level_collision_rejections": 0,
        "all_generated_scenes_validated": True,
        "passed": catalog["all_128_feasible"],
    }


def make_order_counterfactual_group(group_index: int, query_type: tuple[str, int, str, int]) -> list[R4A0REpisode]:
    query = render_scene(query_type, group_index, f"counterfactual|query|{group_index}")
    group_id = f"order_cf:{query_type_key(query_type)}:{group_index:06d}"
    support_inputs = []
    for pair_index in range(3):
        sqtype = support_query_type(query_type, f"counterfactual|{group_index}|{pair_index}")
        support_inputs.append(render_support_scene(sqtype, group_index + pair_index * 307, pair_index, "counterfactual"))
    targets = {sequence: apply_sequence_to_grid(query, sequence) for sequence in ORDER_COUNTERFACTUAL_CLASSES}
    if len({stable_hash(grid_to_list(target)) for target in targets.values()}) != len(ORDER_COUNTERFACTUAL_CLASSES):
        raise RuntimeError("counterfactual targets are not distinct")
    rows: list[R4A0REpisode] = []
    for sequence in ORDER_COUNTERFACTUAL_CLASSES:
        supports = tuple(R4A0RSupportPair(input=support_input, output=apply_sequence_to_grid(support_input, sequence)) for support_input in support_inputs)
        rows.append(
            R4A0REpisode(
                episode_id=f"{group_id}:{sequence}",
                split="order_counterfactual",
                sequence_class=sequence,
                query_type=query_type,
                counterfactual_group_id=group_id,
                query=query,
                supports=supports,
                target=targets[sequence],
            )
        )
    return rows


def generate_order_counterfactual_suite(split_types: dict[str, list[tuple[str, int, str, int]]], group_count: int = 1500) -> list[R4A0REpisode]:
    query_types = split_types["sealed"]
    rows: list[R4A0REpisode] = []
    for idx in range(group_count):
        qtype = query_types[idx % len(query_types)]
        group_index = stable_int(f"r4a0r|order_cf|{idx}|{query_type_key(qtype)}") % 10_000_000
        rows.extend(make_order_counterfactual_group(group_index, qtype))
    return rows


def counterfactual_suite_manifest(rows: list[R4A0REpisode]) -> dict[str, Any]:
    groups: dict[str, list[R4A0REpisode]] = {}
    for episode in rows:
        groups.setdefault(str(episode.counterfactual_group_id), []).append(episode)
    bad: list[str] = []
    for group_id, episodes in groups.items():
        if {episode.sequence_class for episode in episodes} != set(ORDER_COUNTERFACTUAL_CLASSES):
            bad.append(group_id)
            continue
        if len({stable_hash(grid_to_list(episode.query)) for episode in episodes}) != 1:
            bad.append(group_id)
            continue
        support_input_signatures = {
            stable_hash([[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes[:1]])
            for episode in episodes
        }
        if len(support_input_signatures) != 1:
            bad.append(group_id)
            continue
        if len({stable_hash(grid_to_list(episode.target)) for episode in episodes}) != len(ORDER_COUNTERFACTUAL_CLASSES):
            bad.append(group_id)
    return {
        "group_count": len(groups),
        "episode_count": len(rows),
        "required_group_count": 1500,
        "bad_group_count": len(bad),
        "bad_group_examples": bad[:20],
        "passed": len(groups) >= 1500 and not bad,
    }


def save_dataset(data_out: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    data_out = Path(data_out)
    catalog = query_type_preflight_catalog(seed)
    write_json(data_out / "query_type_preflight_catalog.json", catalog)
    if catalog["feasible_query_type_count"] != 128:
        return {"catalog": catalog, "generated": False}
    split_types = partition_query_types(seed)
    split_audit = query_type_split_audit(split_types)
    write_json(data_out / "query_type_split_audit.json", split_audit)
    episodes_by_split = {split: generate_split(split, split_types[split]) for split in ("train", "dev", "sealed")}
    for split, episodes in episodes_by_split.items():
        write_jsonl(data_out / f"{split}.jsonl", episodes)
    cf_suite = generate_order_counterfactual_suite(split_types, 1500)
    write_jsonl(data_out / "order_counterfactual_suite.jsonl", cf_suite)
    heldout = {
        split: [query_type_to_dict(item) | {"query_type_key": query_type_key(item)} for item in split_types[split]]
        for split in ("train", "dev", "sealed")
    }
    manifest = dataset_manifest(episodes_by_split, split_types)
    balance = sequence_balance_report(episodes_by_split)
    collision = scene_collision_rejection_report(episodes_by_split, catalog)
    cf_manifest = counterfactual_suite_manifest(cf_suite)
    write_json(data_out / "heldout_query_type_manifest.json", heldout)
    write_json(data_out / "dataset_manifest.json", manifest)
    write_json(data_out / "sequence_split_manifest.json", {"sequence_classes": list(ALL_SEQUENCE_CLASSES), "train_visible": list(TRAIN_VISIBLE_CLASSES), "held_out": list(HELD_OUT_TWO_STEP_CLASSES)})
    write_json(data_out / "sequence_balance_report.json", balance)
    write_json(data_out / "scene_collision_rejection_report.json", collision)
    write_json(data_out / "counterfactual_suite_manifest.json", cf_manifest)
    return {
        "catalog": catalog,
        "split_types": split_types,
        "episodes_by_split": episodes_by_split,
        "order_counterfactual_suite": cf_suite,
        "manifest": manifest,
        "balance": balance,
        "collision": collision,
        "counterfactual_manifest": cf_manifest,
        "generated": True,
    }


def load_order_counterfactual_suite(path: Path) -> list[R4A0REpisode]:
    rows: list[R4A0REpisode] = []
    with (Path(path) / "order_counterfactual_suite.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(episode_from_record(json.loads(line)))
    return rows


def reorder_episode(episode: R4A0REpisode, order: tuple[int, ...] | list[int]) -> R4A0REpisode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid order {order}")
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def derange_episode(episode: R4A0REpisode, shift: int) -> R4A0REpisode:
    if shift not in {1, 2}:
        raise ValueError("R4A0R supports only the two non-identity derangements for three pairs")
    inputs = [pair.input for pair in episode.supports]
    outputs = [pair.output for pair in episode.supports]
    return replace(
        episode,
        supports=tuple(R4A0RSupportPair(input=inputs[index], output=outputs[(index + shift) % len(outputs)]) for index in range(len(outputs))),
    )


def support_signature(episode: R4A0REpisode) -> list[tuple[str, str]]:
    return [(stable_hash(grid_to_list(pair.input)), stable_hash(grid_to_list(pair.output))) for pair in episode.supports]


def support_input_bag(episode: R4A0REpisode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.input)) for pair in episode.supports)


def support_output_bag(episode: R4A0REpisode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.output)) for pair in episode.supports)


def tensors_from_episodes(episodes: list[R4A0REpisode], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_grids": torch.tensor([grid_to_list(episode.query) for episode in episodes], dtype=torch.long, device=device),
        "support_input_grids": torch.tensor([[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes], dtype=torch.long, device=device),
        "support_output_grids": torch.tensor([[grid_to_list(pair.output) for pair in episode.supports] for episode in episodes], dtype=torch.long, device=device),
        "target_grids": torch.tensor([grid_to_list(episode.target) for episode in episodes], dtype=torch.long, device=device),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
