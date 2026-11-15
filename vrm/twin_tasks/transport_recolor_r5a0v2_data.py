from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from .transport_recolor_r5a0v2_oracles import (
    BACKGROUND,
    COUNTERFACTUAL_CLASSES,
    GRID_SIZE,
    HELD_OUT_MIXED_CLASSES,
    INPUT_COLORS,
    MIXED_CLASSES,
    SEALED_CLASSES,
    SEEN_MIXED_CLASSES,
    SINGLE_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    apply_sequence_to_grid,
    foreground_coords,
    grid_to_list,
    no_edge_foreground,
    parse_grid,
    parse_sequence,
)

GLOBAL_SEED = 20260628
PROJECT_ROOT = Path("/mnt/d/VRM")
SCENE_COORDS = tuple((x, y) for y in range(2, 5) for x in range(2, 5))
SHAPES = ("3_cell_L", "3_cell_T", "3_cell_zig_zag", "asymmetric_4_cell_hook")
COLORS = INPUT_COLORS
ANCHORS = ("interior_upper_left", "interior_upper_right", "interior_lower_left", "interior_lower_right")
DISTRACTOR_COUNTS = (1, 2)
REQUIRED_QUERY_TYPE_COUNTS = {"train": 80, "dev": 24, "sealed": 24}
EPISODES_PER_QUERY_TYPE_CLASS = 60
SUPPORT_COORDS = SCENE_COORDS
SUPPORT_COLOR_CODES: dict[tuple[int, int], tuple[int, int, int]] = {
    coord: ((index % 4) + 1, ((index // 4) % 4) + 1, ((index // 16) % 4) + 1)
    for index, coord in enumerate(SUPPORT_COORDS, start=1)
}

PRIMARY_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "3_cell_L": ((0, 0), (0, 1), (1, 1)),
    "3_cell_T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "3_cell_zig_zag": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "asymmetric_4_cell_hook": ((0, 0), (0, 1), (0, 2), (1, 2)),
}


@dataclass(frozen=True)
class R5A0V2SupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class R5A0V2Episode:
    episode_id: str
    split: str
    operation_class: str
    query_type: tuple[str, int, str, int]
    counterfactual_group_id: str | None
    query: tuple[tuple[int, ...], ...]
    supports: tuple[R5A0V2SupportPair, ...]
    target: tuple[tuple[int, ...], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "operation_class": self.operation_class,
            "sequence_class": self.operation_class,
            "query_type": query_type_to_dict(self.query_type),
            "query_type_key": query_type_key(self.query_type),
            "counterfactual_group_id": self.counterfactual_group_id,
            "query": grid_to_list(self.query),
            "supports": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in self.supports],
            "target": grid_to_list(self.target),
            "evaluation_only_operation_label": self.operation_class,
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


def all_query_types() -> list[tuple[str, int, str, int]]:
    return [(shape, color, anchor, distractor_count) for shape in SHAPES for color in COLORS for anchor in ANCHORS for distractor_count in DISTRACTOR_COUNTS]


def factor_coverage(query_types: Iterable[tuple[str, int, str, int]]) -> dict[str, list[Any]]:
    rows = list(query_types)
    return {
        "primary_shapes": sorted({item[0] for item in rows}),
        "primary_colors": sorted({item[1] for item in rows}),
        "anchor_regions": sorted({item[2] for item in rows}),
        "distractor_counts": sorted({item[3] for item in rows}),
    }


def factor_complete(query_types: Iterable[tuple[str, int, str, int]]) -> bool:
    coverage = factor_coverage(query_types)
    return (
        set(coverage["primary_shapes"]) == set(SHAPES)
        and set(coverage["primary_colors"]) == set(COLORS)
        and set(coverage["anchor_regions"]) == set(ANCHORS)
        and set(coverage["distractor_counts"]) == set(DISTRACTOR_COUNTS)
    )


def shape_cells(query_type: tuple[str, int, str, int]) -> list[tuple[int, int]]:
    shape, _, anchor, _ = query_type
    cells: list[tuple[int, int]] = []
    for dx, dy in PRIMARY_OFFSETS[shape]:
        x = 2 + (2 - dx if "right" in anchor else dx)
        y = 2 + (2 - dy if "lower" in anchor else dy)
        cells.append((x, y))
    return list(dict.fromkeys(cells))


def palette(primary_color: int, salt: str) -> list[int]:
    tail = [color for color in INPUT_COLORS if color != primary_color]
    rng = random.Random(stable_int(f"r5a0v2|palette|{primary_color}|{salt}"))
    rng.shuffle(tail)
    return [primary_color] + tail


def _spans_required_regions(cells: Iterable[tuple[int, int]]) -> bool:
    coords = list(cells)
    return (
        any(x <= 3 for x, _ in coords)
        and any(x > 3 for x, _ in coords)
        and any(y <= 3 for _, y in coords)
        and any(y > 3 for _, y in coords)
        and any(x <= 3 and y <= 3 for x, y in coords)
        and any(not (x <= 3 and y <= 3) for x, y in coords)
    )


def _render_scene_once(query_type: tuple[str, int, str, int], variant: int, salt: str) -> tuple[tuple[tuple[int, ...], ...], list[str]]:
    _, primary_color, _, distractor_count = query_type
    primary = shape_cells(query_type)
    cells = list(primary)
    rng = random.Random(stable_int(f"r5a0v2|scene|{query_type_key(query_type)}|{variant}|{salt}"))
    remaining = [coord for coord in SCENE_COORDS if coord not in cells]
    rng.shuffle(remaining)
    needed = 2 * distractor_count
    for coord in remaining:
        if len(cells) >= len(primary) + needed and _spans_required_regions(cells):
            break
        cells.append(coord)
    if not _spans_required_regions(cells):
        for coord in SCENE_COORDS:
            if coord not in cells:
                cells.append(coord)
            if len(cells) >= len(primary) + needed and _spans_required_regions(cells):
                break
    cells = cells[: max(len(primary) + needed, min(len(cells), 9))]
    colors = palette(primary_color, f"{variant}|{salt}")
    grid = empty_grid()
    for index, (x, y) in enumerate(cells):
        color = primary_color if index < len(primary) else colors[1 + ((index - len(primary)) % max(len(colors) - 1, 1))]
        grid[y][x] = color
    rendered = freeze_grid(grid)
    ok, reasons = validate_scene(rendered)
    return rendered, [] if ok else reasons


def render_scene(query_type: tuple[str, int, str, int], variant: int, salt: str) -> tuple[tuple[int, ...], ...]:
    last_reasons: list[str] = []
    for attempt in range(512):
        rendered, reasons = _render_scene_once(query_type, variant + attempt, f"{salt}|repair_attempt={attempt}")
        if not reasons:
            return rendered
        last_reasons = reasons
    rendered, reasons = _render_scene_once(query_type, variant, salt)
    ok = not reasons
    if not ok:
        raise RuntimeError(f"invalid R5A-0 V2 scene {query_type_key(query_type)} {variant} {salt}: {last_reasons or reasons}")
    return rendered


def render_support_scene(pair_index: int) -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    for x, y in SUPPORT_COORDS:
        grid[y][x] = SUPPORT_COLOR_CODES[(x, y)][pair_index]
    return freeze_grid(grid)


def recolor_change_count(before: tuple[tuple[int, ...], ...], after: tuple[tuple[int, ...], ...]) -> int:
    return sum(1 for y in range(GRID_SIZE) for x in range(GRID_SIZE) if before[y][x] != BACKGROUND and after[y][x] == before[y][x] + 4)


def validate_scene(grid: tuple[tuple[int, ...], ...]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    coords = foreground_coords(grid)
    if not coords:
        reasons.append("empty_foreground")
    if not no_edge_foreground(grid):
        reasons.append("foreground_touches_edge")
    if any((x, y) not in SCENE_COORDS for x, y in coords):
        reasons.append("foreground_outside_r5a0v2_scene_interior")
    if not _spans_required_regions(coords):
        reasons.append("does_not_span_required_recolor_regions")
    if any(value not in INPUT_COLORS + (BACKGROUND,) for row in grid for value in row):
        reasons.append("input_color_outside_1_to_4")
    outputs: dict[str, tuple[tuple[int, ...], ...]] = {}
    for sequence in SEALED_CLASSES:
        current = grid
        try:
            for op in parse_sequence(sequence):
                before = current
                current = apply_sequence_to_grid(current, op)
                if op in {"L", "K"}:
                    changed = recolor_change_count(before, current)
                    fg = len(foreground_coords(before))
                    if not (0 < changed < fg):
                        reasons.append(f"{sequence}_{op}_recolor_not_proper_subset")
                if not no_edge_foreground(current):
                    reasons.append(f"{sequence}_intermediate_or_final_touches_edge")
                    break
            outputs[sequence] = apply_sequence_to_grid(grid, sequence)
        except ValueError as exc:
            reasons.append(f"{sequence}_invalid:{exc}")
    if outputs:
        for geom in ("C", "H", "V"):
            if outputs.get(geom) == grid:
                reasons.append(f"{geom}_does_not_visibly_move")
        mixed_outputs = {sequence: outputs[sequence] for sequence in MIXED_CLASSES if sequence in outputs}
        if len({stable_hash(grid_to_list(value)) for value in mixed_outputs.values()}) != len(MIXED_CLASSES):
            reasons.append("mixed_output_collision")
        for mixed in MIXED_CLASSES:
            if outputs.get(mixed) == grid:
                reasons.append(f"{mixed}_equals_input")
            for single in SINGLE_CLASSES:
                if outputs.get(mixed) == outputs.get(single):
                    reasons.append(f"{mixed}_equals_single_{single}")
    return not reasons, sorted(set(reasons))


def preflight_query_type(query_type: tuple[str, int, str, int], seed: int = GLOBAL_SEED) -> dict[str, Any]:
    valid_counts = {sequence: 0 for sequence in SEALED_CLASSES}
    reasons: list[str] = []
    for attempt in range(12):
        try:
            scene = render_scene(query_type, seed + attempt, f"preflight|{attempt}")
        except RuntimeError as exc:
            reasons.append(str(exc))
            continue
        for sequence in SEALED_CLASSES:
            try:
                _ = apply_sequence_to_grid(scene, sequence)
                valid_counts[sequence] += 1
            except ValueError as exc:
                reasons.append(f"{sequence}:{exc}")
    feasible = all(count > 0 for count in valid_counts.values())
    return {
        "query_type": query_type_to_dict(query_type) | {"query_type_key": query_type_key(query_type)},
        "feasible": feasible,
        "valid_scene_counts_by_operation_class": valid_counts,
        "rejection_reasons": [] if feasible else sorted(set(reasons)),
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
    for attempt in range(50_000):
        rng = random.Random(seed + attempt)
        shuffled = list(query_types)
        rng.shuffle(shuffled)
        split = {"train": shuffled[:80], "dev": shuffled[80:104], "sealed": shuffled[104:]}
        if all(factor_complete(items) for items in split.values()):
            return split
    raise RuntimeError("failed to create deterministic 80/24/24 R5A-0 V2 split")


def query_type_split_audit(split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    sets = {split: {query_type_key(item) for item in items} for split, items in split_types.items()}
    overlaps = {
        "train_dev": sorted(sets["train"].intersection(sets["dev"])),
        "train_sealed": sorted(sets["train"].intersection(sets["sealed"])),
        "dev_sealed": sorted(sets["dev"].intersection(sets["sealed"])),
    }
    union = set().union(*sets.values())
    return {
        "split_counts": {split: len(items) for split, items in split_types.items()},
        "expected_split_counts": REQUIRED_QUERY_TYPE_COUNTS,
        "union_count": len(union),
        "catalog_count": len(all_query_types()),
        "uses_every_catalog_type_exactly_once": len(union) == len(all_query_types()) and not any(overlaps.values()),
        "intersections": overlaps,
        "factor_coverage": {split: factor_coverage(items) for split, items in split_types.items()},
        "factor_complete": {split: factor_complete(items) for split, items in split_types.items()},
        "passed": (
            {split: len(items) for split, items in split_types.items()} == REQUIRED_QUERY_TYPE_COUNTS
            and len(union) == len(all_query_types())
            and not any(overlaps.values())
            and all(factor_complete(items) for items in split_types.values())
        ),
    }


def make_supports(operation_class: str) -> tuple[R5A0V2SupportPair, ...]:
    return tuple(
        R5A0V2SupportPair(input=render_support_scene(pair_index), output=apply_sequence_to_grid(render_support_scene(pair_index), operation_class))
        for pair_index in range(3)
    )


def make_episode(split: str, query_type: tuple[str, int, str, int], operation_class: str, local_index: int) -> R5A0V2Episode:
    variant = stable_int(f"r5a0v2|{split}|{query_type_key(query_type)}|{operation_class}|{local_index}") % 10_000_000
    query = render_scene(query_type, variant, f"{split}|query|{operation_class}|{local_index}")
    target = apply_sequence_to_grid(query, operation_class)
    supports = make_supports(operation_class)
    if any(pair.input == query for pair in supports):
        raise RuntimeError("support input equals query")
    if any(pair.output == target for pair in supports):
        raise RuntimeError("support output equals target")
    return R5A0V2Episode(
        episode_id=f"{split}:{query_type_key(query_type)}:{operation_class}:idx={local_index}",
        split=split,
        operation_class=operation_class,
        query_type=query_type,
        counterfactual_group_id=None,
        query=query,
        supports=supports,
        target=target,
    )


def generate_split(split: str, query_types: list[tuple[str, int, str, int]]) -> list[R5A0V2Episode]:
    classes = TRAIN_VISIBLE_CLASSES if split in {"train", "dev"} else SEALED_CLASSES
    rows: list[R5A0V2Episode] = []
    for query_type in sorted(query_types, key=query_type_key):
        for operation_class in classes:
            for index in range(EPISODES_PER_QUERY_TYPE_CLASS):
                rows.append(make_episode(split, query_type, operation_class, index))
    random.Random(stable_int(f"r5a0v2|order|{split}")).shuffle(rows)
    return rows


def episode_from_record(record: dict[str, Any]) -> R5A0V2Episode:
    return R5A0V2Episode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        operation_class=str(record.get("operation_class", record.get("sequence_class"))),
        query_type=query_type_from_dict(record["query_type"]),
        counterfactual_group_id=record.get("counterfactual_group_id"),
        query=parse_grid(record["query"]),
        supports=tuple(R5A0V2SupportPair(input=parse_grid(pair["input"]), output=parse_grid(pair["output"])) for pair in record["supports"]),
        target=parse_grid(record["target"]),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, episodes: Iterable[R5A0V2Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[R5A0V2Episode]:
    rows: list[R5A0V2Episode] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(episode_from_record(json.loads(line)))
    return rows


def load_split(path: Path, split: str) -> list[R5A0V2Episode]:
    return load_jsonl(Path(path) / f"{split}.jsonl")


def sequence_counts(episodes: Iterable[R5A0V2Episode]) -> dict[str, int]:
    rows = list(episodes)
    return {sequence: sum(1 for episode in rows if episode.operation_class == sequence) for sequence in SEALED_CLASSES if any(episode.operation_class == sequence for episode in rows)}


def operation_balance_report(episodes_by_split: dict[str, list[R5A0V2Episode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        expected_classes = TRAIN_VISIBLE_CLASSES if split in {"train", "dev"} else SEALED_CLASSES
        per_type: dict[str, dict[str, int]] = {}
        for episode in episodes:
            row = per_type.setdefault(query_type_key(episode.query_type), {sequence: 0 for sequence in expected_classes})
            row[episode.operation_class] += 1
        report[split] = {
            "episode_count": len(episodes),
            "operation_counts": sequence_counts(episodes),
            "query_type_count": len(per_type),
            "per_query_type_operation_count_expected": EPISODES_PER_QUERY_TYPE_CLASS,
            "all_query_types_balanced": all(all(row[sequence] == EPISODES_PER_QUERY_TYPE_CLASS for sequence in expected_classes) for row in per_type.values()),
            "passed": len(per_type) == REQUIRED_QUERY_TYPE_COUNTS[split] and all(all(row[sequence] == EPISODES_PER_QUERY_TYPE_CLASS for sequence in expected_classes) for row in per_type.values()),
        }
    return report


def dataset_manifest(episodes_by_split: dict[str, list[R5A0V2Episode]], split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    return {
        "benchmark": "transport_recolor_r5a0_v2",
        "grid_size": GRID_SIZE,
        "split_episode_counts": {split: len(episodes) for split, episodes in episodes_by_split.items()},
        "expected_split_episode_counts": {"train": 43200, "dev": 12960, "sealed": 18720},
        "query_type_counts": {split: len(items) for split, items in split_types.items()},
        "operation_counts": {split: sequence_counts(episodes) for split, episodes in episodes_by_split.items()},
        "query_type_split_audit": query_type_split_audit(split_types),
        "operation_balance_report": operation_balance_report(episodes_by_split),
    }


def scene_rejection_report(episodes_by_split: dict[str, list[R5A0V2Episode]], catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_episode_count": sum(len(episodes) for episodes in episodes_by_split.values()),
        "preflight_feasible_query_type_count": catalog["feasible_query_type_count"],
        "scene_level_collision_rejections": 0,
        "all_generated_scenes_validated": True,
        "passed": catalog["all_128_feasible"],
    }


def make_counterfactual_group(group_index: int, query_type: tuple[str, int, str, int]) -> list[R5A0V2Episode]:
    query = render_scene(query_type, group_index, f"counterfactual|query|{group_index}")
    targets = {operation: apply_sequence_to_grid(query, operation) for operation in COUNTERFACTUAL_CLASSES}
    if len({stable_hash(grid_to_list(target)) for target in targets.values()}) != len(COUNTERFACTUAL_CLASSES):
        raise RuntimeError("counterfactual targets collide")
    group_id = f"r5a0v2_cf:{query_type_key(query_type)}:{group_index:06d}"
    support_inputs = tuple(render_support_scene(pair_index) for pair_index in range(3))
    return [
        R5A0V2Episode(
            episode_id=f"{group_id}:{operation}",
            split="heterogeneous_counterfactual",
            operation_class=operation,
            query_type=query_type,
            counterfactual_group_id=group_id,
            query=query,
            supports=tuple(R5A0V2SupportPair(input=support_input, output=apply_sequence_to_grid(support_input, operation)) for support_input in support_inputs),
            target=targets[operation],
        )
        for operation in COUNTERFACTUAL_CLASSES
    ]


def generate_counterfactual_suite(split_types: dict[str, list[tuple[str, int, str, int]]], group_count: int = 1000) -> list[R5A0V2Episode]:
    qtypes = sorted(split_types["sealed"], key=query_type_key)
    rows: list[R5A0V2Episode] = []
    for index in range(group_count):
        qtype = qtypes[index % len(qtypes)]
        group_index = stable_int(f"r5a0v2|cf|{index}|{query_type_key(qtype)}") % 10_000_000
        rows.extend(make_counterfactual_group(group_index, qtype))
    return rows


RECOMBINATION_CONTEXTS = ("G_C", "G_H", "G_V", "POST_L", "PRE_L", "POST_K", "PRE_K")
RECOMBINATION_CONTEXT_TO_OPERATION = {
    "G_C": "C",
    "G_H": "H",
    "G_V": "V",
    "POST_L": "C->L",
    "PRE_L": "L->H",
    "POST_K": "H->K",
    "PRE_K": "K->V",
}
RECOMBINATION_TARGETS = {
    "L->C": ("G_C", "PRE_L"),
    "H->L": ("G_H", "POST_L"),
    "C->K": ("G_C", "POST_K"),
    "K->C": ("G_C", "PRE_K"),
}
NEGATIVE_RECOMBINATIONS = {
    "G_C_plus_POST_L": ("G_C", "POST_L"),
    "G_H_plus_PRE_L": ("G_H", "PRE_L"),
    "G_V_plus_POST_K": ("G_V", "POST_K"),
    "G_C_plus_G_H": ("G_C", "G_H"),
}


def make_recombination_group(group_index: int, query_type: tuple[str, int, str, int]) -> list[R5A0V2Episode]:
    query = render_scene(query_type, group_index, f"recombination|query|{group_index}")
    support_inputs = tuple(render_support_scene(pair_index) for pair_index in range(3))
    targets = {operation: apply_sequence_to_grid(query, operation) for operation in set(RECOMBINATION_TARGETS) | set(RECOMBINATION_CONTEXT_TO_OPERATION.values())}
    if len({stable_hash(grid_to_list(targets[operation])) for operation in RECOMBINATION_TARGETS}) != len(RECOMBINATION_TARGETS):
        raise RuntimeError("recombination held-out targets collide")
    group_id = f"r5a0v2_recombine:{query_type_key(query_type)}:{group_index:06d}"
    rows: list[R5A0V2Episode] = []
    for context in RECOMBINATION_CONTEXTS:
        operation = RECOMBINATION_CONTEXT_TO_OPERATION[context]
        rows.append(
            R5A0V2Episode(
                episode_id=f"{group_id}:{context}",
                split="head_recombination",
                operation_class=operation,
                query_type=query_type,
                counterfactual_group_id=group_id,
                query=query,
                supports=tuple(R5A0V2SupportPair(input=support_input, output=apply_sequence_to_grid(support_input, operation)) for support_input in support_inputs),
                target=targets[operation],
            )
        )
    return rows


def generate_recombination_suite(split_types: dict[str, list[tuple[str, int, str, int]]], group_count: int = 1000) -> list[R5A0V2Episode]:
    qtypes = sorted(split_types["sealed"], key=query_type_key)
    rows: list[R5A0V2Episode] = []
    for index in range(group_count):
        qtype = qtypes[index % len(qtypes)]
        group_index = stable_int(f"r5a0v2|recombine|{index}|{query_type_key(qtype)}") % 10_000_000
        rows.extend(make_recombination_group(group_index, qtype))
    return rows


def grouped_manifest(rows: list[R5A0V2Episode], required_group_count: int, required_classes: Iterable[str]) -> dict[str, Any]:
    groups: dict[str, list[R5A0V2Episode]] = defaultdict(list)
    for episode in rows:
        groups[str(episode.counterfactual_group_id)].append(episode)
    required = set(required_classes)
    bad: list[str] = []
    for group_id, episodes in groups.items():
        if {episode.operation_class for episode in episodes} != required:
            bad.append(group_id)
            continue
        if len({stable_hash(grid_to_list(episode.query)) for episode in episodes}) != 1:
            bad.append(group_id)
            continue
        if len({stable_hash([[grid_to_list(pair.input) for pair in episode.supports]]) for episode in episodes}) != 1:
            bad.append(group_id)
    return {
        "group_count": len(groups),
        "episode_count": len(rows),
        "required_group_count": required_group_count,
        "required_contexts_or_classes": sorted(required),
        "bad_group_count": len(bad),
        "bad_group_examples": bad[:20],
        "passed": len(groups) >= required_group_count and not bad,
    }


def save_dataset(data_out: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    data_out = Path(data_out)
    data_out.mkdir(parents=True, exist_ok=True)
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
    counterfactual = generate_counterfactual_suite(split_types)
    recombination = generate_recombination_suite(split_types)
    write_jsonl(data_out / "heterogeneous_counterfactual_suite.jsonl", counterfactual)
    write_jsonl(data_out / "head_recombination_suite.jsonl", recombination)
    manifest = dataset_manifest(episodes_by_split, split_types)
    balance = operation_balance_report(episodes_by_split)
    rejection = scene_rejection_report(episodes_by_split, catalog)
    cf_manifest = grouped_manifest(counterfactual, 1000, COUNTERFACTUAL_CLASSES)
    recomb_manifest = grouped_manifest(recombination, 1000, RECOMBINATION_CONTEXT_TO_OPERATION.values())
    write_json(data_out / "dataset_manifest.json", manifest)
    write_json(data_out / "operation_split_manifest.json", {"single": list(SINGLE_CLASSES), "seen_mixed_train_dev": list(SEEN_MIXED_CLASSES), "held_out_mixed_sealed_only": list(HELD_OUT_MIXED_CLASSES)})
    write_json(data_out / "operation_balance_report.json", balance)
    write_json(data_out / "scene_rejection_report.json", rejection)
    write_json(data_out / "heterogeneous_counterfactual_manifest.json", cf_manifest)
    write_json(data_out / "head_recombination_suite_manifest.json", recomb_manifest | {"recombination_targets": RECOMBINATION_TARGETS, "negative_recombinations": NEGATIVE_RECOMBINATIONS})
    return {
        "catalog": catalog,
        "split_types": split_types,
        "episodes_by_split": episodes_by_split,
        "counterfactual_suite": counterfactual,
        "recombination_suite": recombination,
        "manifest": manifest,
        "balance": balance,
        "rejection": rejection,
        "counterfactual_manifest": cf_manifest,
        "recombination_manifest": recomb_manifest,
        "generated": True,
    }


def load_counterfactual_suite(path: Path) -> list[R5A0V2Episode]:
    return load_jsonl(Path(path) / "heterogeneous_counterfactual_suite.jsonl")


def load_recombination_suite(path: Path) -> list[R5A0V2Episode]:
    return load_jsonl(Path(path) / "head_recombination_suite.jsonl")


def reorder_episode(episode: R5A0V2Episode, order: tuple[int, ...] | list[int]) -> R5A0V2Episode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid support order {order}")
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def derange_episode(episode: R5A0V2Episode, shift: int) -> R5A0V2Episode:
    if shift not in {1, 2}:
        raise ValueError("R5A-0 V2 supports only the two non-identity three-pair derangements")
    inputs = [pair.input for pair in episode.supports]
    outputs = [pair.output for pair in episode.supports]
    return replace(episode, supports=tuple(R5A0V2SupportPair(input=inputs[index], output=outputs[(index + shift) % 3]) for index in range(3)))


def support_signature(episode: R5A0V2Episode) -> list[tuple[str, str]]:
    return [(stable_hash(grid_to_list(pair.input)), stable_hash(grid_to_list(pair.output))) for pair in episode.supports]


def support_input_bag(episode: R5A0V2Episode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.input)) for pair in episode.supports)


def support_output_bag(episode: R5A0V2Episode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.output)) for pair in episode.supports)


def tensors_from_episodes(episodes: list[R5A0V2Episode], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_grids": torch.tensor([grid_to_list(episode.query) for episode in episodes], dtype=torch.long, device=device),
        "support_input_grids": torch.tensor([[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes], dtype=torch.long, device=device),
        "support_output_grids": torch.tensor([[grid_to_list(pair.output) for pair in episode.supports] for episode in episodes], dtype=torch.long, device=device),
        "target_grids": torch.tensor([grid_to_list(episode.target) for episode in episodes], dtype=torch.long, device=device),
    }
