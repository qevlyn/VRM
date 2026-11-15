from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch

from vrm.data import ARC_COLOR_COUNT

GRID_SIZE = 7
BACKGROUND = 0
GLOBAL_SEED = 20260628

ROUTE_RETRIEVAL = "RETRIEVAL_ELIGIBLE"
ROUTE_TRANSFORM = "TRANSFORM_ONLY"
ROUTE_HARD_NEGATIVE = "HARD_NEGATIVE_NO_MATCH"
ROUTE_FAMILIES = (ROUTE_RETRIEVAL, ROUTE_TRANSFORM, ROUTE_HARD_NEGATIVE)

TRUE_ROUTE_READER = "reader"
TRUE_ROUTE_TRANSFORM = "transform"

TRANSFORMS = ("HFLIP", "VFLIP", "ROT180")
SHAPES = ("marker_0", "marker_1", "marker_2", "marker_3")
COLORS = (1, 2, 3, 4)
ANCHORS = ("upper_left", "upper_right", "lower_left", "lower_right")
DISTRACTOR_COUNTS = (0, 1, 2)

PRIMARY_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "marker_0": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "marker_1": ((0, 0), (1, 0), (1, 1)),
    "marker_2": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "marker_3": ((0, 0), (1, 1), (2, 1)),
}

ANCHOR_TOP_LEFT = {
    "upper_left": (0, 0),
    "upper_right": (0, 4),
    "lower_left": (4, 0),
    "lower_right": (4, 4),
}


@dataclass(frozen=True)
class R3ASupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class R3AEpisode:
    episode_id: str
    split: str
    route_family: str
    query_type: tuple[str, int, str, int]
    counterfactual_group_id: str
    query: tuple[tuple[int, ...], ...]
    supports: tuple[R3ASupportPair, ...]
    target: tuple[tuple[int, ...], ...]
    transform_spec: str
    true_route: str
    relevant_pair_index: int | None

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


def query_type_to_dict(query_type: tuple[str, int, str, int]) -> dict[str, Any]:
    shape, color, anchor, distractors = query_type
    return {
        "marker_shape": shape,
        "marker_color": color,
        "anchor_position": anchor,
        "distractor_count": distractors,
    }


def query_type_key(query_type: tuple[str, int, str, int]) -> str:
    shape, color, anchor, distractors = query_type
    return f"{shape}|color={color}|anchor={anchor}|distractors={distractors}"


def all_query_types() -> list[tuple[str, int, str, int]]:
    return [
        (shape, color, anchor, distractors)
        for shape in SHAPES
        for color in COLORS
        for anchor in ANCHORS
        for distractors in DISTRACTOR_COUNTS
    ]


def _check_factor_coverage(query_types: Iterable[tuple[str, int, str, int]]) -> bool:
    query_types = list(query_types)
    return (
        {item[0] for item in query_types} == set(SHAPES)
        and {item[1] for item in query_types} == set(COLORS)
        and {item[2] for item in query_types} == set(ANCHORS)
        and {item[3] for item in query_types} == set(DISTRACTOR_COUNTS)
    )


def partition_query_types(seed: int = GLOBAL_SEED) -> dict[str, list[tuple[str, int, str, int]]]:
    query_types = all_query_types()
    for attempt in range(10_000):
        rng = random.Random(seed + attempt)
        shuffled = list(query_types)
        rng.shuffle(shuffled)
        split_types = {
            "train": shuffled[:120],
            "dev": shuffled[120:156],
            "sealed": shuffled[156:192],
        }
        if all(_check_factor_coverage(items) for items in split_types.values()):
            return split_types
    raise RuntimeError("failed to partition query types with full factor coverage")


def group_counts_for_split(split: str, query_types: list[tuple[str, int, str, int]]) -> dict[tuple[str, int, str, int], int]:
    total = 12_000 if split == "train" else 3_000
    base = total // len(query_types)
    remainder = total % len(query_types)
    ranked = sorted(query_types, key=lambda item: stable_int(f"{split}|{query_type_key(item)}"))
    extra = set(ranked[:remainder])
    return {query_type: base + (1 if query_type in extra else 0) for query_type in query_types}


def place_primary(grid: list[list[int]], query_type: tuple[str, int, str, int]) -> set[tuple[int, int]]:
    shape, color, anchor, _ = query_type
    row0, col0 = ANCHOR_TOP_LEFT[anchor]
    occupied: set[tuple[int, int]] = set()
    for dr, dc in PRIMARY_SHAPES[shape]:
        row = row0 + dr
        col = col0 + dc
        grid[row][col] = color
        occupied.add((row, col))
    return occupied


def distractor_cells(
    query_type: tuple[str, int, str, int],
    group_index: int,
    salt: str,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    _, primary_color, _, count = query_type
    if count == 0:
        return []
    rng = random.Random(stable_int(f"{query_type_key(query_type)}|{group_index}|{salt}|distractors"))
    available = [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if (row, col) not in occupied]
    rng.shuffle(available)
    colors = [color for color in range(1, ARC_COLOR_COUNT) if color != primary_color]
    cells: list[tuple[int, int, int]] = []
    for idx in range(count):
        row, col = available[idx]
        color = colors[(stable_int(f"{salt}|{group_index}|{idx}") + idx) % len(colors)]
        cells.append((row, col, color))
    return cells


def render_query(query_type: tuple[str, int, str, int], group_index: int, salt: str = "query") -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    occupied = place_primary(grid, query_type)
    for row, col, color in distractor_cells(query_type, group_index, salt, occupied):
        grid[row][col] = color
    return freeze_grid(grid)


def apply_transform(grid: tuple[tuple[int, ...], ...], transform_spec: str) -> tuple[tuple[int, ...], ...]:
    rows = grid_to_list(grid)
    if transform_spec == "HFLIP":
        return freeze_grid([list(reversed(row)) for row in rows])
    if transform_spec == "VFLIP":
        return freeze_grid(list(reversed(rows)))
    if transform_spec == "ROT180":
        return freeze_grid([list(reversed(row)) for row in reversed(rows)])
    raise ValueError(f"unknown transform {transform_spec}")


def centered_plus_glyph(color: int) -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    for row, col in ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3)):
        grid[row][col] = color
    return freeze_grid(grid)


def different_query_type(query_type: tuple[str, int, str, int], salt: str) -> tuple[str, int, str, int]:
    shape, color, anchor, distractors = query_type
    options = [
        (other_shape, color, anchor, distractors)
        for other_shape in SHAPES
        if other_shape != shape
    ] + [
        (shape, other_color, anchor, distractors)
        for other_color in COLORS
        if other_color != color
    ] + [
        (shape, color, other_anchor, distractors)
        for other_anchor in ANCHORS
        if other_anchor != anchor
    ] + [
        (shape, color, anchor, other_count)
        for other_count in DISTRACTOR_COUNTS
        if other_count != distractors
    ]
    return options[stable_int(f"{query_type_key(query_type)}|{salt}") % len(options)]


def transform_for_group(query_type: tuple[str, int, str, int], group_index: int) -> str:
    return TRANSFORMS[stable_int(f"{query_type_key(query_type)}|{group_index}|transform") % len(TRANSFORMS)]


def glyph_colors(query_type: tuple[str, int, str, int], group_index: int) -> list[int]:
    colors = list(range(1, ARC_COLOR_COUNT))
    rng = random.Random(stable_int(f"{query_type_key(query_type)}|{group_index}|glyph_colors"))
    rng.shuffle(colors)
    return colors[:3]


def make_retrieval_episode(split: str, query_type: tuple[str, int, str, int], group_index: int, query: tuple[tuple[int, ...], ...], group_id: str) -> R3AEpisode:
    relevant = stable_int(f"{group_id}|relevant") % 3
    colors = glyph_colors(query_type, group_index)
    outputs = [centered_plus_glyph(color) for color in colors]
    supports: list[R3ASupportPair] = []
    for pair_idx in range(3):
        if pair_idx == relevant:
            support_input = query
        else:
            support_type = different_query_type(query_type, f"{group_id}|retrieval|{pair_idx}")
            support_input = render_query(support_type, group_index + pair_idx + 17, f"retrieval_support_{pair_idx}")
            if support_input == query:
                raise AssertionError("retrieval nonmatch accidentally equals query")
        supports.append(R3ASupportPair(input=support_input, output=outputs[pair_idx]))
    target = outputs[relevant]
    transform_spec = transform_for_group(query_type, group_index)
    if apply_transform(query, transform_spec) == target:
        transform_spec = TRANSFORMS[(TRANSFORMS.index(transform_spec) + 1) % len(TRANSFORMS)]
    assert apply_transform(query, transform_spec) != target
    assert sum(1 for pair in supports if pair.input == query) == 1
    assert sum(1 for pair in supports if pair.output == target) == 1
    return R3AEpisode(
        episode_id=f"{split}:{group_id}:{ROUTE_RETRIEVAL}",
        split=split,
        route_family=ROUTE_RETRIEVAL,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_READER,
        relevant_pair_index=relevant,
    )


def make_transform_episode(split: str, query_type: tuple[str, int, str, int], group_index: int, query: tuple[tuple[int, ...], ...], group_id: str) -> R3AEpisode:
    transform_spec = transform_for_group(query_type, group_index)
    target = apply_transform(query, transform_spec)
    supports: list[R3ASupportPair] = []
    for pair_idx in range(3):
        support_type = different_query_type(query_type, f"{group_id}|transform|{pair_idx}")
        support_input = render_query(support_type, group_index + 101 + pair_idx, f"transform_support_{pair_idx}")
        if support_input == query:
            raise AssertionError("transform support equals query")
        support_output = apply_transform(support_input, transform_spec)
        if support_output == target:
            raise AssertionError("transform support output equals target")
        supports.append(R3ASupportPair(input=support_input, output=support_output))
    return R3AEpisode(
        episode_id=f"{split}:{group_id}:{ROUTE_TRANSFORM}",
        split=split,
        route_family=ROUTE_TRANSFORM,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_TRANSFORM,
        relevant_pair_index=None,
    )


def make_hard_negative_episode(split: str, query_type: tuple[str, int, str, int], group_index: int, query: tuple[tuple[int, ...], ...], group_id: str) -> R3AEpisode:
    shape, color, anchor, distractors = query_type
    transform_spec = transform_for_group(query_type, group_index)
    target = apply_transform(query, transform_spec)
    different_anchor = next(item for item in ANCHORS if item != anchor)
    different_color = next(item for item in COLORS if item != color)
    different_distractors = (distractors + 1) % len(DISTRACTOR_COUNTS)
    hard_types = [
        (shape, color, different_anchor, distractors),
        (shape, different_color, anchor, distractors),
        (shape, color, anchor, different_distractors),
    ]
    supports: list[R3ASupportPair] = []
    for pair_idx, support_type in enumerate(hard_types):
        support_input = render_query(support_type, group_index + 211 + pair_idx, f"hard_support_{pair_idx}")
        if support_input == query:
            raise AssertionError("hard-negative support equals query")
        support_output = apply_transform(support_input, transform_spec)
        if support_output == target:
            raise AssertionError("hard-negative support output equals target")
        supports.append(R3ASupportPair(input=support_input, output=support_output))
    return R3AEpisode(
        episode_id=f"{split}:{group_id}:{ROUTE_HARD_NEGATIVE}",
        split=split,
        route_family=ROUTE_HARD_NEGATIVE,
        query_type=query_type,
        counterfactual_group_id=group_id,
        query=query,
        supports=tuple(supports),
        target=target,
        transform_spec=transform_spec,
        true_route=TRUE_ROUTE_TRANSFORM,
        relevant_pair_index=None,
    )


def make_counterfactual_group(split: str, query_type: tuple[str, int, str, int], group_index: int) -> list[R3AEpisode]:
    group_id = f"{split}:{query_type_key(query_type)}:group={group_index:05d}"
    query = render_query(query_type, group_index, "counterfactual_query")
    return [
        make_retrieval_episode(split, query_type, group_index, query, group_id),
        make_transform_episode(split, query_type, group_index, query, group_id),
        make_hard_negative_episode(split, query_type, group_index, query, group_id),
    ]


def generate_split(split: str, query_types: list[tuple[str, int, str, int]], seed: int = GLOBAL_SEED) -> list[R3AEpisode]:
    episodes: list[R3AEpisode] = []
    counts = group_counts_for_split(split, query_types)
    for query_type in sorted(query_types, key=query_type_key):
        for local_index in range(counts[query_type]):
            group_index = stable_int(f"{seed}|{split}|{query_type_key(query_type)}|{local_index}") % 10_000_000
            episodes.extend(make_counterfactual_group(split, query_type, group_index))
    rng = random.Random(seed + stable_int(split))
    rng.shuffle(episodes)
    return episodes


def write_jsonl(path: Path, episodes: list[R3AEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), sort_keys=True) + "\n")


def parse_grid(raw: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return freeze_grid([[int(value) for value in row] for row in raw])


def episode_from_record(record: dict[str, Any]) -> R3AEpisode:
    qtype = record["query_type"]
    query_type = (
        str(qtype["marker_shape"]),
        int(qtype["marker_color"]),
        str(qtype["anchor_position"]),
        int(qtype["distractor_count"]),
    )
    return R3AEpisode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        route_family=str(record["route_family"]),
        query_type=query_type,
        counterfactual_group_id=str(record["counterfactual_group_id"]),
        query=parse_grid(record["query"]),
        supports=tuple(
            R3ASupportPair(input=parse_grid(pair["input"]), output=parse_grid(pair["output"]))
            for pair in record["supports"]
        ),
        target=parse_grid(record["target"]),
        transform_spec=str(record["transform_spec"]),
        true_route=str(record["true_route"]),
        relevant_pair_index=None if record["relevant_pair_index"] is None else int(record["relevant_pair_index"]),
    )


def load_split(path: Path, split: str) -> list[R3AEpisode]:
    episodes: list[R3AEpisode] = []
    with (Path(path) / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                episodes.append(episode_from_record(json.loads(line)))
    return episodes


def save_dataset(data_out: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    data_out = Path(data_out)
    split_types = partition_query_types(seed)
    episodes_by_split = {split: generate_split(split, types, seed) for split, types in split_types.items()}
    for split, episodes in episodes_by_split.items():
        write_jsonl(data_out / f"{split}.jsonl", episodes)
    manifest = dataset_manifest(episodes_by_split)
    (data_out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    heldout = {
        split: [query_type_to_dict(item) | {"query_type_key": query_type_key(item)} for item in split_types[split]]
        for split in ("train", "dev", "sealed")
    }
    (data_out / "heldout_query_type_manifest.json").write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes_by_split": episodes_by_split, "manifest": manifest, "heldout_query_types": heldout}


def family_counts(episodes: list[R3AEpisode]) -> dict[str, int]:
    return {family: sum(1 for episode in episodes if episode.route_family == family) for family in ROUTE_FAMILIES}


def query_type_counts(episodes: list[R3AEpisode]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for episode in episodes:
        row = counts.setdefault(query_type_key(episode.query_type), {family: 0 for family in ROUTE_FAMILIES})
        row[episode.route_family] += 1
    return counts


def route_balance_report(episodes_by_split: dict[str, list[R3AEpisode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        qtype_counts = query_type_counts(episodes)
        per_type_all_families = all(all(row[family] > 0 for family in ROUTE_FAMILIES) for row in qtype_counts.values())
        per_type_balanced = all(len({row[family] for family in ROUTE_FAMILIES}) == 1 for row in qtype_counts.values())
        report[split] = {
            "episode_count": len(episodes),
            "route_family_counts": family_counts(episodes),
            "query_type_count": len(qtype_counts),
            "query_types_all_have_three_families": per_type_all_families,
            "query_types_balanced_by_family": per_type_balanced,
            "passed": per_type_all_families and per_type_balanced and len(set(family_counts(episodes).values())) == 1,
        }
    return report


def counterfactual_group_report(episodes_by_split: dict[str, list[R3AEpisode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        groups: dict[str, list[R3AEpisode]] = {}
        for episode in episodes:
            groups.setdefault(episode.counterfactual_group_id, []).append(episode)
        bad: list[str] = []
        for group_id, rows in groups.items():
            if {row.route_family for row in rows} != set(ROUTE_FAMILIES):
                bad.append(group_id)
                continue
            queries = {stable_hash(grid_to_list(row.query)) for row in rows}
            if len(queries) != 1:
                bad.append(group_id)
        report[split] = {
            "counterfactual_group_count": len(groups),
            "bad_group_count": len(bad),
            "bad_group_examples": bad[:20],
            "passed": not bad,
        }
    return report


def dataset_manifest(episodes_by_split: dict[str, list[R3AEpisode]]) -> dict[str, Any]:
    split_counts = {split: len(episodes) for split, episodes in episodes_by_split.items()}
    route_counts = {split: family_counts(episodes) for split, episodes in episodes_by_split.items()}
    qtype_sets = {
        split: {query_type_key(episode.query_type) for episode in episodes}
        for split, episodes in episodes_by_split.items()
    }
    return {
        "benchmark": "mixed_router_r3a_v1",
        "grid_size": GRID_SIZE,
        "seed": GLOBAL_SEED,
        "split_episode_counts": split_counts,
        "route_family_counts": route_counts,
        "query_type_counts": {split: len(items) for split, items in qtype_sets.items()},
        "query_type_sets_disjoint": (
            qtype_sets["train"].isdisjoint(qtype_sets["dev"])
            and qtype_sets["train"].isdisjoint(qtype_sets["sealed"])
            and qtype_sets["dev"].isdisjoint(qtype_sets["sealed"])
        ),
        "factor_coverage": {
            split: {
                "marker_shapes": sorted({episode.query_type[0] for episode in episodes}),
                "marker_colors": sorted({episode.query_type[1] for episode in episodes}),
                "anchor_positions": sorted({episode.query_type[2] for episode in episodes}),
                "distractor_counts": sorted({episode.query_type[3] for episode in episodes}),
            }
            for split, episodes in episodes_by_split.items()
        },
        "route_balance_report": route_balance_report(episodes_by_split),
        "counterfactual_group_report": counterfactual_group_report(episodes_by_split),
    }


def reorder_episode(episode: R3AEpisode, order: tuple[int, ...] | list[int]) -> R3AEpisode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid support order {order}")
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def tensors_from_episodes(episodes: list[R3AEpisode], device: torch.device) -> dict[str, torch.Tensor]:
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
    targets = torch.tensor([grid_to_list(episode.target) for episode in episodes], dtype=torch.long, device=device)
    return {
        "query_grids": query,
        "support_input_grids": support_inputs,
        "support_output_grids": support_outputs,
        "target_grids": targets,
    }


def support_signature(episode: R3AEpisode) -> list[tuple[str, str]]:
    return [
        (stable_hash(grid_to_list(pair.input)), stable_hash(grid_to_list(pair.output)))
        for pair in episode.supports
    ]

