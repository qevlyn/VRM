from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from vrm.data import ARC_COLOR_COUNT

from .transform_r3b0_data import (
    ANCHORS,
    COLORS,
    DISTRACTOR_COUNTS,
    GLOBAL_SEED,
    GRID_SIZE,
    SHAPES,
    TRANSFORMS,
    apply_transform,
    factor_coverage,
    factor_complete,
    grid_to_list,
    is_nondegenerate,
    partition_query_types,
    query_type_key,
    query_type_to_dict,
    render_query,
    stable_hash,
    stable_int,
)

FAMILY_RETRIEVAL = "RETRIEVAL_CLEAR"
FAMILY_TRANSFORM = "TRANSFORM_CLEAR"
FAMILY_HARD_NEGATIVE = "TRANSFORM_HARD_NEGATIVE"
CORE_FAMILIES = (FAMILY_RETRIEVAL, FAMILY_TRANSFORM, FAMILY_HARD_NEGATIVE)

TRUE_ROUTE_READER = "reader"
TRUE_ROUTE_TRANSFORM = "transform"

BOUNDARY_EXACT_RETRIEVAL = "BOUNDARY_EXACT_RETRIEVAL"
BOUNDARY_TRANSFORM = "BOUNDARY_TRANSFORM"
BOUNDARY_HARD_NEGATIVE = "BOUNDARY_HARD_NEGATIVE"
BOUNDARY_BOTH_GOOD = "BOUNDARY_BOTH_GOOD"
BOUNDARY_BOTH_BAD = "BOUNDARY_BOTH_BAD"


@dataclass(frozen=True)
class R3B2SupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class R3B2Episode:
    episode_id: str
    split: str
    route_family: str
    query_type: tuple[str, int, str, int]
    counterfactual_group_id: str
    query: tuple[tuple[int, ...], ...]
    supports: tuple[R3B2SupportPair, ...]
    target: tuple[tuple[int, ...], ...]
    transform_spec: str
    true_route: str | None
    relevant_pair_index: int | None
    boundary_kind: str | None = None
    boundary_hamming_level: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "route_family": self.route_family,
            "query_type": query_type_to_dict(self.query_type),
            "query_type_key": query_type_key(self.query_type),
            "counterfactual_group_id": self.counterfactual_group_id,
            "query": grid_to_list(self.query),
            "supports": [
                {"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)}
                for pair in self.supports
            ],
            "target": grid_to_list(self.target),
            "transform_spec": self.transform_spec,
            "true_route": self.true_route,
            "relevant_pair_index": self.relevant_pair_index,
            "boundary_kind": self.boundary_kind,
            "boundary_hamming_level": self.boundary_hamming_level,
        }


def freeze_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def parse_grid(raw: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return freeze_grid([[int(value) for value in row] for row in raw])


def query_type_from_record(payload: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(payload["marker_shape"]),
        int(payload["marker_color"]),
        str(payload["anchor_position"]),
        int(payload["distractor_count"]),
    )


def episode_from_record(record: dict[str, Any]) -> R3B2Episode:
    return R3B2Episode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        route_family=str(record["route_family"]),
        query_type=query_type_from_record(record["query_type"]),
        counterfactual_group_id=str(record["counterfactual_group_id"]),
        query=parse_grid(record["query"]),
        supports=tuple(
            R3B2SupportPair(input=parse_grid(pair["input"]), output=parse_grid(pair["output"]))
            for pair in record["supports"]
        ),
        target=parse_grid(record["target"]),
        transform_spec=str(record["transform_spec"]),
        true_route=None if record["true_route"] is None else str(record["true_route"]),
        relevant_pair_index=None if record["relevant_pair_index"] is None else int(record["relevant_pair_index"]),
        boundary_kind=record.get("boundary_kind"),
        boundary_hamming_level=record.get("boundary_hamming_level"),
    )


def empty_grid() -> list[list[int]]:
    return [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def centered_plus_glyph(color: int) -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    for row, col in ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3)):
        grid[row][col] = color
    return freeze_grid(grid)


def checker_glyph(color_a: int, color_b: int) -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    for row, col, color in ((1, 1, color_a), (1, 2, color_b), (2, 1, color_b), (2, 2, color_a)):
        grid[row][col] = color
    return freeze_grid(grid)


def transform_for_group(query_type: tuple[str, int, str, int], group_index: int) -> str:
    return TRANSFORMS[stable_int(f"r3b2|{query_type_key(query_type)}|{group_index}|transform") % len(TRANSFORMS)]


def output_glyphs(query_type: tuple[str, int, str, int], group_index: int) -> list[tuple[tuple[int, ...], ...]]:
    colors = list(range(1, ARC_COLOR_COUNT))
    random.Random(stable_int(f"r3b2|{query_type_key(query_type)}|{group_index}|glyphs")).shuffle(colors)
    return [centered_plus_glyph(color) for color in colors[:3]]


def different_query_type(query_type: tuple[str, int, str, int], salt: str) -> tuple[str, int, str, int]:
    shape, color, anchor, distractors = query_type
    options = [
        (candidate, color, anchor, distractors)
        for candidate in SHAPES
        if candidate != shape
    ] + [
        (shape, candidate, anchor, distractors)
        for candidate in COLORS
        if candidate != color
    ] + [
        (shape, color, candidate, distractors)
        for candidate in ANCHORS
        if candidate != anchor
    ] + [
        (shape, color, anchor, candidate)
        for candidate in DISTRACTOR_COUNTS
        if candidate != distractors
    ]
    return options[stable_int(f"r3b2|{query_type_key(query_type)}|{salt}") % len(options)]


def support_query_types_for_transform(
    query_type: tuple[str, int, str, int],
    group_index: int,
    transform_spec: str,
) -> list[tuple[str, int, str, int]]:
    shape, color, anchor, distractors = query_type
    items: list[tuple[str, int, str, int]] = []
    for pair_index in range(3):
        support_color = COLORS[pair_index]
        support_shape = SHAPES[(stable_int(f"r3b2|shape|{group_index}|{transform_spec}|{pair_index}") + pair_index) % len(SHAPES)]
        support_anchor = ANCHORS[(stable_int(f"r3b2|anchor|{group_index}|{transform_spec}|{pair_index}") + pair_index) % len(ANCHORS)]
        candidate = (support_shape, support_color, support_anchor, 0)
        if candidate == query_type:
            candidate = (shape, support_color, ANCHORS[(ANCHORS.index(anchor) + 1) % len(ANCHORS)], 0)
        if candidate == query_type:
            candidate = (SHAPES[(SHAPES.index(shape) + 1) % len(SHAPES)], support_color, support_anchor, distractors)
        items.append(candidate)
    return items


def hamming_grid(a: tuple[tuple[int, ...], ...], b: tuple[tuple[int, ...], ...]) -> int:
    return sum(1 for row_a, row_b in zip(a, b) for value_a, value_b in zip(row_a, row_b) if value_a != value_b)


def mutate_query_hamming(
    query: tuple[tuple[int, ...], ...],
    level: int,
    salt: str,
) -> tuple[tuple[int, ...], ...]:
    if level <= 0:
        raise ValueError("hamming level must be positive")
    grid = [list(row) for row in query]
    foreground = [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] != 0]
    background = [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == 0]
    random.Random(stable_int(f"r3b2|mutate|{salt}|fg")).shuffle(foreground)
    random.Random(stable_int(f"r3b2|mutate|{salt}|bg")).shuffle(background)
    cells = foreground[: min(level, len(foreground))]
    if len(cells) < level:
        cells.extend(background[: level - len(cells)])
    for idx, (row, col) in enumerate(cells[:level]):
        current = grid[row][col]
        grid[row][col] = ((current + idx + 1) % (ARC_COLOR_COUNT - 1)) + 1
        if grid[row][col] == current:
            grid[row][col] = (current % (ARC_COLOR_COUNT - 1)) + 1
    mutated = freeze_grid(grid)
    if mutated == query:
        raise RuntimeError("mutation failed to alter query")
    return mutated


def make_retrieval_episode(
    split: str,
    query_type: tuple[str, int, str, int],
    group_index: int,
    query: tuple[tuple[int, ...], ...],
    group_id: str,
) -> R3B2Episode:
    relevant = stable_int(f"r3b2|{group_id}|relevant") % 3
    glyphs = output_glyphs(query_type, group_index)
    supports: list[R3B2SupportPair] = []
    for pair_index in range(3):
        if pair_index == relevant:
            support_input = query
        else:
            support_type = different_query_type(query_type, f"{group_id}|retrieval|{pair_index}")
            support_input = render_query(support_type, group_index + 17 + pair_index, f"r3b2_retrieval_{pair_index}")
            if support_input == query:
                support_input = mutate_query_hamming(query, pair_index + 1, f"{group_id}|retrieval_fallback|{pair_index}")
        supports.append(R3B2SupportPair(input=support_input, output=glyphs[pair_index]))
    target = glyphs[relevant]
    transform_spec = transform_for_group(query_type, group_index)
    if any(apply_transform(query, spec) == target for spec in TRANSFORMS):
        raise RuntimeError("retrieval target collided with a transform output")
    return R3B2Episode(
        episode_id=f"{split}:{group_id}:{FAMILY_RETRIEVAL}",
        split=split,
        route_family=FAMILY_RETRIEVAL,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_READER,
        relevant_pair_index=relevant,
    )


def make_transform_episode(
    split: str,
    query_type: tuple[str, int, str, int],
    group_index: int,
    query: tuple[tuple[int, ...], ...],
    group_id: str,
) -> R3B2Episode:
    transform_spec = transform_for_group(query_type, group_index)
    target = apply_transform(query, transform_spec)
    supports: list[R3B2SupportPair] = []
    for pair_index, support_type in enumerate(support_query_types_for_transform(query_type, group_index, transform_spec)):
        support_input = render_query(support_type, group_index + 101 + pair_index, f"r3b2_transform_{pair_index}")
        if support_input == query:
            support_input = mutate_query_hamming(query, pair_index + 1, f"{group_id}|transform_fallback|{pair_index}")
        support_output = apply_transform(support_input, transform_spec)
        if support_output == target:
            raise RuntimeError("transform support output collided with target")
        supports.append(R3B2SupportPair(input=support_input, output=support_output))
    return R3B2Episode(
        episode_id=f"{split}:{group_id}:{FAMILY_TRANSFORM}",
        split=split,
        route_family=FAMILY_TRANSFORM,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_TRANSFORM,
        relevant_pair_index=None,
    )


def make_hard_negative_episode(
    split: str,
    query_type: tuple[str, int, str, int],
    group_index: int,
    query: tuple[tuple[int, ...], ...],
    group_id: str,
) -> R3B2Episode:
    shape, color, anchor, distractors = query_type
    transform_spec = transform_for_group(query_type, group_index)
    target = apply_transform(query, transform_spec)
    hard_types = [
        (shape, color, ANCHORS[(ANCHORS.index(anchor) + 1) % len(ANCHORS)], distractors),
        (shape, COLORS[(COLORS.index(color) + 1) % len(COLORS)], anchor, distractors),
        (shape, color, anchor, 1 if distractors == 0 else distractors),
    ]
    supports: list[R3B2SupportPair] = []
    for pair_index, support_type in enumerate(hard_types):
        support_input = render_query(support_type, group_index + 211 + pair_index, f"r3b2_hard_{pair_index}")
        if support_input == query:
            support_input = mutate_query_hamming(query, 1, f"{group_id}|hard_fallback|{pair_index}")
        support_output = apply_transform(support_input, transform_spec)
        if support_output == target:
            raise RuntimeError("hard-negative support output collided with target")
        supports.append(R3B2SupportPair(input=support_input, output=support_output))
    return R3B2Episode(
        episode_id=f"{split}:{group_id}:{FAMILY_HARD_NEGATIVE}",
        split=split,
        route_family=FAMILY_HARD_NEGATIVE,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_TRANSFORM,
        relevant_pair_index=None,
    )


def make_counterfactual_group(
    split: str,
    query_type: tuple[str, int, str, int],
    group_index: int,
) -> list[R3B2Episode]:
    for attempt in range(100):
        salted_index = group_index + attempt * 10_000_000
        query = render_query(query_type, salted_index, f"r3b2_query|{split}|{attempt}")
        if not is_nondegenerate(query):
            continue
        group_id = f"{split}:{query_type_key(query_type)}:group={group_index:05d}:attempt={attempt}"
        try:
            return [
                make_retrieval_episode(split, query_type, salted_index, query, group_id),
                make_transform_episode(split, query_type, salted_index, query, group_id),
                make_hard_negative_episode(split, query_type, salted_index, query, group_id),
            ]
        except RuntimeError:
            continue
    raise RuntimeError(f"failed to create R3B2 group for {split} {query_type_key(query_type)} {group_index}")


def generate_split(
    split: str,
    query_types: list[tuple[str, int, str, int]],
    *,
    seed: int = GLOBAL_SEED,
    groups_per_query_type: int = 100,
) -> list[R3B2Episode]:
    episodes: list[R3B2Episode] = []
    for query_type in sorted(query_types, key=query_type_key):
        for local_index in range(groups_per_query_type):
            group_index = stable_int(f"r3b2|{seed}|{split}|{query_type_key(query_type)}|{local_index}") % 10_000_000
            episodes.extend(make_counterfactual_group(split, query_type, group_index))
    random.Random(seed + stable_int(f"r3b2|{split}|order")).shuffle(episodes)
    return episodes


def generate_compatibility_set(count_per_family: int = 1500, seed: int = GLOBAL_SEED + 17) -> list[R3B2Episode]:
    query_types = partition_query_types(seed)["dev"]
    groups_needed = count_per_family
    episodes: list[R3B2Episode] = []
    for group_index in range(groups_needed):
        query_type = query_types[group_index % len(query_types)]
        salted = stable_int(f"r3b2|compat|{seed}|{group_index}|{query_type_key(query_type)}") % 10_000_000
        episodes.extend(make_counterfactual_group("compatibility", query_type, salted))
    return episodes


def make_boundary_episode(
    query_type: tuple[str, int, str, int],
    index: int,
    kind: str,
    *,
    hamming_level: int | None = None,
) -> R3B2Episode:
    split = "boundary"
    query = render_query(query_type, index, f"r3b2_boundary_query|{kind}")
    group_id = f"boundary:{query_type_key(query_type)}:{kind}:{index:06d}"
    if kind == BOUNDARY_EXACT_RETRIEVAL:
        episode = make_retrieval_episode(split, query_type, index, query, group_id)
        return replace(episode, episode_id=f"{group_id}:{kind}", boundary_kind=kind)
    if kind == BOUNDARY_TRANSFORM:
        episode = make_transform_episode(split, query_type, index, query, group_id)
        return replace(episode, episode_id=f"{group_id}:{kind}", boundary_kind=kind)
    if kind == BOUNDARY_HARD_NEGATIVE:
        level = int(hamming_level or 1)
        transform_spec = transform_for_group(query_type, index)
        target = apply_transform(query, transform_spec)
        near = mutate_query_hamming(query, level, f"{group_id}|level={level}")
        supports = [R3B2SupportPair(input=near, output=apply_transform(near, transform_spec))]
        for pair_index, support_type in enumerate(support_query_types_for_transform(query_type, index, transform_spec)[1:], start=1):
            support_input = render_query(support_type, index + 400 + pair_index, f"r3b2_boundary_hard_{pair_index}")
            if support_input == query:
                support_input = mutate_query_hamming(query, pair_index + 1, f"{group_id}|fallback|{pair_index}")
            supports.append(R3B2SupportPair(input=support_input, output=apply_transform(support_input, transform_spec)))
        if any(pair.output == target for pair in supports):
            raise RuntimeError("boundary hard-negative support output collided with target")
        return R3B2Episode(
            episode_id=f"{group_id}:{kind}:level={level}",
            split=split,
            route_family=FAMILY_HARD_NEGATIVE,
            query_type=query_type,
            counterfactual_group_id=group_id,
            query=query,
            supports=tuple(supports),
            target=target,
            transform_spec=transform_spec,
            true_route=TRUE_ROUTE_TRANSFORM,
            relevant_pair_index=None,
            boundary_kind=kind,
            boundary_hamming_level=level,
        )
    if kind == BOUNDARY_BOTH_GOOD:
        transform_spec = transform_for_group(query_type, index)
        target = apply_transform(query, transform_spec)
        supports = [R3B2SupportPair(input=query, output=target)]
        for pair_index, support_type in enumerate(support_query_types_for_transform(query_type, index, transform_spec)[1:], start=1):
            support_input = render_query(support_type, index + 500 + pair_index, f"r3b2_boundary_both_good_{pair_index}")
            if support_input == query:
                support_input = mutate_query_hamming(query, pair_index + 1, f"{group_id}|both_good_fallback|{pair_index}")
            supports.append(R3B2SupportPair(input=support_input, output=apply_transform(support_input, transform_spec)))
        return R3B2Episode(
            episode_id=f"{group_id}:{kind}",
            split=split,
            route_family=kind,
            query_type=query_type,
            counterfactual_group_id=group_id,
            query=query,
            supports=tuple(supports),
            target=target,
            transform_spec=transform_spec,
            true_route=None,
            relevant_pair_index=0,
            boundary_kind=kind,
        )
    if kind == BOUNDARY_BOTH_BAD:
        transform_spec = transform_for_group(query_type, index)
        supports = make_transform_episode(split, query_type, index, query, group_id).supports
        target = checker_glyph(8, 9)
        if target == apply_transform(query, transform_spec) or any(pair.output == target for pair in supports):
            target = checker_glyph(7, 9)
        return R3B2Episode(
            episode_id=f"{group_id}:{kind}",
            split=split,
            route_family=kind,
            query_type=query_type,
            counterfactual_group_id=group_id,
            query=query,
            supports=supports,
            target=target,
            transform_spec=transform_spec,
            true_route=None,
            relevant_pair_index=None,
            boundary_kind=kind,
        )
    raise ValueError(f"unknown boundary kind {kind}")


def generate_boundary_pool(seed: int = GLOBAL_SEED + 99, count: int = 12_000) -> list[R3B2Episode]:
    query_types = partition_query_types(seed)["sealed"]
    plan: list[tuple[str, int | None, int]] = [
        (BOUNDARY_EXACT_RETRIEVAL, None, 2000),
        (BOUNDARY_TRANSFORM, None, 2000),
        (BOUNDARY_HARD_NEGATIVE, 1, 1000),
        (BOUNDARY_HARD_NEGATIVE, 2, 1000),
        (BOUNDARY_HARD_NEGATIVE, 3, 1000),
        (BOUNDARY_HARD_NEGATIVE, 4, 1000),
        (BOUNDARY_BOTH_GOOD, None, 2000),
        (BOUNDARY_BOTH_BAD, None, 2000),
    ]
    episodes: list[R3B2Episode] = []
    cursor = 0
    for kind, level, item_count in plan:
        for local_index in range(item_count):
            query_type = query_types[(cursor + local_index) % len(query_types)]
            source_index = stable_int(f"r3b2|boundary|{seed}|{kind}|{level}|{local_index}") % 10_000_000
            episodes.append(make_boundary_episode(query_type, source_index, kind, hamming_level=level))
        cursor += item_count
    if len(episodes) < count:
        raise RuntimeError("boundary pool generation produced too few episodes")
    random.Random(seed).shuffle(episodes)
    return episodes[:count]


def write_jsonl(path: Path, episodes: list[R3B2Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), sort_keys=True) + "\n")


def load_split(path: Path, split: str) -> list[R3B2Episode]:
    episodes: list[R3B2Episode] = []
    with (Path(path) / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(episode_from_record(json.loads(line)))
    return episodes


def load_boundary_pool(path: Path) -> list[R3B2Episode]:
    episodes: list[R3B2Episode] = []
    with (Path(path) / "boundary_candidate_pool.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(episode_from_record(json.loads(line)))
    return episodes


def family_counts(episodes: Iterable[R3B2Episode]) -> dict[str, int]:
    rows = list(episodes)
    return {family: sum(1 for episode in rows if episode.route_family == family) for family in CORE_FAMILIES}


def query_type_counts(episodes: Iterable[R3B2Episode]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for episode in episodes:
        if episode.route_family not in CORE_FAMILIES:
            continue
        row = counts.setdefault(query_type_key(episode.query_type), {family: 0 for family in CORE_FAMILIES})
        row[episode.route_family] += 1
    return counts


def family_balance_report(episodes_by_split: dict[str, list[R3B2Episode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        qtype_counts = query_type_counts(episodes)
        family_count = family_counts(episodes)
        report[split] = {
            "episode_count": len(episodes),
            "family_counts": family_count,
            "family_balanced": len(set(family_count.values())) == 1,
            "query_type_count": len(qtype_counts),
            "query_types_all_have_core_families": all(
                all(row[family] > 0 for family in CORE_FAMILIES) for row in qtype_counts.values()
            ),
            "query_types_balanced_by_family": all(len({row[family] for family in CORE_FAMILIES}) == 1 for row in qtype_counts.values()),
        }
        report[split]["passed"] = (
            report[split]["family_balanced"]
            and report[split]["query_types_all_have_core_families"]
            and report[split]["query_types_balanced_by_family"]
        )
    return report


def counterfactual_group_manifest(episodes_by_split: dict[str, list[R3B2Episode]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        groups: dict[str, list[R3B2Episode]] = {}
        for episode in episodes:
            groups.setdefault(episode.counterfactual_group_id, []).append(episode)
        bad: list[str] = []
        for group_id, rows in groups.items():
            if {row.route_family for row in rows} != set(CORE_FAMILIES):
                bad.append(group_id)
                continue
            if len({stable_hash(grid_to_list(row.query)) for row in rows}) != 1:
                bad.append(group_id)
        manifest[split] = {
            "group_count": len(groups),
            "bad_group_count": len(bad),
            "bad_group_examples": bad[:20],
            "passed": not bad,
        }
    return manifest


def dataset_manifest(episodes_by_split: dict[str, list[R3B2Episode]]) -> dict[str, Any]:
    qtype_sets = {
        split: {query_type_key(episode.query_type) for episode in episodes}
        for split, episodes in episodes_by_split.items()
    }
    return {
        "benchmark": "mixed_router_r3b2_v1",
        "grid_size": GRID_SIZE,
        "seed": GLOBAL_SEED,
        "split_episode_counts": {split: len(episodes) for split, episodes in episodes_by_split.items()},
        "family_counts": {split: family_counts(episodes) for split, episodes in episodes_by_split.items()},
        "query_type_counts": {split: len(qtypes) for split, qtypes in qtype_sets.items()},
        "query_type_sets_disjoint": (
            qtype_sets["train"].isdisjoint(qtype_sets["dev"])
            and qtype_sets["train"].isdisjoint(qtype_sets["sealed"])
            and qtype_sets["dev"].isdisjoint(qtype_sets["sealed"])
        ),
        "factor_coverage": {
            split: factor_coverage({episode.query_type for episode in episodes})
            for split, episodes in episodes_by_split.items()
        },
        "factor_complete": {
            split: factor_complete({episode.query_type for episode in episodes})
            for split, episodes in episodes_by_split.items()
        },
        "family_balance_report": family_balance_report(episodes_by_split),
        "counterfactual_group_manifest": counterfactual_group_manifest(episodes_by_split),
    }


def heldout_query_type_manifest(split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    return {
        split: [query_type_to_dict(item) | {"query_type_key": query_type_key(item)} for item in split_types[split]]
        for split in ("train", "dev", "sealed")
    }


def boundary_candidate_pool_manifest(episodes: list[R3B2Episode]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    levels: dict[str, int] = {}
    for episode in episodes:
        key = str(episode.boundary_kind)
        counts[key] = counts.get(key, 0) + 1
        if episode.boundary_hamming_level is not None:
            level_key = str(episode.boundary_hamming_level)
            levels[level_key] = levels.get(level_key, 0) + 1
    return {
        "episode_count": len(episodes),
        "boundary_kind_counts": counts,
        "hard_negative_hamming_level_counts": levels,
        "audit_only": True,
        "contributes_to_training": False,
    }


def save_dataset(data_out: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    data_out = Path(data_out)
    split_types = partition_query_types(seed)
    episodes_by_split = {
        split: generate_split(split, split_types[split], seed=seed)
        for split in ("train", "dev", "sealed")
    }
    for split, episodes in episodes_by_split.items():
        write_jsonl(data_out / f"{split}.jsonl", episodes)
    boundary_pool = generate_boundary_pool(seed + 99)
    write_jsonl(data_out / "boundary_candidate_pool.jsonl", boundary_pool)
    manifest = dataset_manifest(episodes_by_split)
    heldout = heldout_query_type_manifest(split_types)
    balance = family_balance_report(episodes_by_split)
    counterfactual = counterfactual_group_manifest(episodes_by_split)
    boundary_manifest = boundary_candidate_pool_manifest(boundary_pool)
    (data_out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (data_out / "heldout_query_type_manifest.json").write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (data_out / "family_balance_report.json").write_text(json.dumps(balance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (data_out / "counterfactual_group_manifest.json").write_text(json.dumps(counterfactual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (data_out / "boundary_candidate_pool_manifest.json").write_text(json.dumps(boundary_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "episodes_by_split": episodes_by_split,
        "boundary_pool": boundary_pool,
        "manifest": manifest,
        "heldout_query_types": heldout,
        "family_balance_report": balance,
        "counterfactual_group_manifest": counterfactual,
        "boundary_candidate_pool_manifest": boundary_manifest,
    }


def reorder_episode(episode: R3B2Episode, order: tuple[int, ...] | list[int]) -> R3B2Episode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid support order {order}")
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def support_signature(episode: R3B2Episode) -> list[tuple[str, str]]:
    return [
        (stable_hash(grid_to_list(pair.input)), stable_hash(grid_to_list(pair.output)))
        for pair in episode.supports
    ]


def tensors_from_episodes(episodes: list[R3B2Episode], device: torch.device) -> dict[str, torch.Tensor]:
    query = torch.tensor([grid_to_list(episode.query) for episode in episodes], dtype=torch.long, device=device)
    support_inputs = torch.tensor(
        [[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes],
        dtype=torch.long,
        device=device,
    )
    support_outputs = torch.tensor(
        [[grid_to_list(pair.output) for pair in episode.supports] for episode in episodes],
        dtype=torch.long,
        device=device,
    )
    target = torch.tensor([grid_to_list(episode.target) for episode in episodes], dtype=torch.long, device=device)
    return {
        "query_grids": query,
        "support_input_grids": support_inputs,
        "support_output_grids": support_outputs,
        "target_grids": target,
    }
