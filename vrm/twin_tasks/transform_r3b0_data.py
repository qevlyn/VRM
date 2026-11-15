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

TRANSFORM_H = "HFLIP"
TRANSFORM_V = "VFLIP"
TRANSFORM_R = "ROT180"
TRANSFORMS = (TRANSFORM_H, TRANSFORM_V, TRANSFORM_R)

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
class R3B0SupportPair:
    input: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class R3B0Episode:
    episode_id: str
    split: str
    query_type: tuple[str, int, str, int]
    query: tuple[tuple[int, ...], ...]
    supports: tuple[R3B0SupportPair, ...]
    target: tuple[tuple[int, ...], ...]
    transform_spec: str
    counterfactual_group_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "query_type": query_type_to_dict(self.query_type),
            "query_type_key": query_type_key(self.query_type),
            "query": grid_to_list(self.query),
            "supports": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in self.supports],
            "target": grid_to_list(self.target),
            "transform_spec": self.transform_spec,
            "counterfactual_group_id": self.counterfactual_group_id,
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


def list_to_grid(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return freeze_grid([[int(value) for value in row] for row in grid])


def query_type_to_dict(query_type: tuple[str, int, str, int]) -> dict[str, Any]:
    shape, color, anchor, distractors = query_type
    return {
        "marker_shape": shape,
        "marker_color": color,
        "anchor_position": anchor,
        "distractor_count": distractors,
    }


def query_type_from_dict(payload: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(payload["marker_shape"]),
        int(payload["marker_color"]),
        str(payload["anchor_position"]),
        int(payload["distractor_count"]),
    )


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


def factor_coverage(query_types: Iterable[tuple[str, int, str, int]]) -> dict[str, list[Any]]:
    items = list(query_types)
    return {
        "marker_shapes": sorted({item[0] for item in items}),
        "marker_colors": sorted({item[1] for item in items}),
        "anchor_positions": sorted({item[2] for item in items}),
        "distractor_counts": sorted({item[3] for item in items}),
    }


def factor_complete(query_types: Iterable[tuple[str, int, str, int]]) -> bool:
    coverage = factor_coverage(query_types)
    return (
        set(coverage["marker_shapes"]) == set(SHAPES)
        and set(coverage["marker_colors"]) == set(COLORS)
        and set(coverage["anchor_positions"]) == set(ANCHORS)
        and set(coverage["distractor_counts"]) == set(DISTRACTOR_COUNTS)
    )


def partition_query_types(seed: int = GLOBAL_SEED) -> dict[str, list[tuple[str, int, str, int]]]:
    query_types = all_query_types()
    for attempt in range(20_000):
        rng = random.Random(seed + attempt)
        shuffled = list(query_types)
        rng.shuffle(shuffled)
        split_types = {
            "train": shuffled[:120],
            "dev": shuffled[120:150],
            "sealed": shuffled[150:180],
        }
        if all(factor_complete(values) for values in split_types.values()):
            return split_types
    raise RuntimeError("failed to partition query types with full factor coverage")


def apply_transform(grid: tuple[tuple[int, ...], ...], transform_spec: str) -> tuple[tuple[int, ...], ...]:
    rows = grid_to_list(grid)
    if transform_spec == TRANSFORM_H:
        return freeze_grid([list(reversed(row)) for row in rows])
    if transform_spec == TRANSFORM_V:
        return freeze_grid(list(reversed(rows)))
    if transform_spec == TRANSFORM_R:
        return freeze_grid([list(reversed(row)) for row in reversed(rows)])
    raise ValueError(f"unknown transform {transform_spec}")


def true_source_index(transform_spec: str, out_row: int, out_col: int) -> int:
    if transform_spec == TRANSFORM_H:
        row, col = out_row, GRID_SIZE - 1 - out_col
    elif transform_spec == TRANSFORM_V:
        row, col = GRID_SIZE - 1 - out_row, out_col
    elif transform_spec == TRANSFORM_R:
        row, col = GRID_SIZE - 1 - out_row, GRID_SIZE - 1 - out_col
    else:
        raise ValueError(f"unknown transform {transform_spec}")
    return row * GRID_SIZE + col


def is_nondegenerate(grid: tuple[tuple[int, ...], ...]) -> bool:
    transformed = [apply_transform(grid, spec) for spec in TRANSFORMS]
    return len(set(transformed)) == len(TRANSFORMS)


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
    episode_index: int,
    salt: str,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    _, primary_color, _, count = query_type
    if count == 0:
        return []
    rng = random.Random(stable_int(f"{query_type_key(query_type)}|{episode_index}|{salt}|distractors"))
    available = [(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if (row, col) not in occupied]
    rng.shuffle(available)
    colors = [color for color in range(1, ARC_COLOR_COUNT) if color != primary_color]
    cells: list[tuple[int, int, int]] = []
    for idx in range(count):
        row, col = available[idx]
        color = colors[(stable_int(f"{salt}|{episode_index}|{idx}") + idx) % len(colors)]
        cells.append((row, col, color))
    return cells


def render_query(query_type: tuple[str, int, str, int], episode_index: int, salt: str = "query") -> tuple[tuple[int, ...], ...]:
    grid = empty_grid()
    occupied = place_primary(grid, query_type)
    for row, col, color in distractor_cells(query_type, episode_index, salt, occupied):
        grid[row][col] = color
    rendered = freeze_grid(grid)
    if not is_nondegenerate(rendered):
        raise RuntimeError(f"degenerate rendered query: {query_type_key(query_type)} {episode_index} {salt}")
    return rendered


def different_query_type(query_type: tuple[str, int, str, int], salt: str) -> tuple[str, int, str, int]:
    options = [candidate for candidate in all_query_types() if candidate != query_type]
    return options[stable_int(f"{query_type_key(query_type)}|{salt}|different") % len(options)]


def support_query_types(query_type: tuple[str, int, str, int], episode_index: int, transform_spec: str) -> list[tuple[str, int, str, int]]:
    types: list[tuple[str, int, str, int]] = []
    query_shape, query_color, query_anchor, query_distractors = query_type
    for pair_index in range(3):
        color = COLORS[pair_index]
        shape = SHAPES[(stable_int(f"{episode_index}|{transform_spec}|shape|{pair_index}") + pair_index) % len(SHAPES)]
        anchor = ANCHORS[(stable_int(f"{episode_index}|{transform_spec}|anchor|{pair_index}") + pair_index) % len(ANCHORS)]
        candidate = (shape, color, anchor, 0)
        if candidate == query_type:
            candidate = (query_shape, color, ANCHORS[(ANCHORS.index(query_anchor) + 1) % len(ANCHORS)], 0)
        if candidate == query_type:
            candidate = (SHAPES[(SHAPES.index(query_shape) + 1) % len(SHAPES)], color, anchor, query_distractors)
        types.append(candidate)
    return types


def make_episode(split: str, query_type: tuple[str, int, str, int], transform_spec: str, index: int) -> R3B0Episode:
    for attempt in range(100):
        base_salt = f"{split}|{transform_spec}|{index}|attempt={attempt}"
        query = render_query(query_type, index + attempt * 10_000, base_salt)
        target = apply_transform(query, transform_spec)
        support_pairs: list[R3B0SupportPair] = []
        for pair_index, support_qtype in enumerate(support_query_types(query_type, index + attempt * 31, transform_spec)):
            support_input = render_query(support_qtype, index + pair_index * 1_003 + attempt * 10_000, f"{base_salt}|support={pair_index}")
            if support_input == query:
                break
            support_pairs.append(R3B0SupportPair(input=support_input, output=apply_transform(support_input, transform_spec)))
        if len(support_pairs) != 3:
            continue
        if any(pair.output == target for pair in support_pairs):
            continue
        episode_id = f"{split}:{query_type_key(query_type)}:{transform_spec}:idx={index}:attempt={attempt}"
        return R3B0Episode(
            episode_id=episode_id,
            split=split,
            query_type=query_type,
            query=query,
            supports=tuple(support_pairs),
            target=target,
            transform_spec=transform_spec,
        )
    raise RuntimeError(f"failed to create episode {split} {query_type_key(query_type)} {transform_spec} {index}")


def generate_split(split: str, query_types: list[tuple[str, int, str, int]]) -> list[R3B0Episode]:
    episodes: list[R3B0Episode] = []
    for query_type in query_types:
        for transform_spec in TRANSFORMS:
            for idx in range(100):
                episodes.append(make_episode(split, query_type, transform_spec, idx))
    rng = random.Random(stable_int(f"{GLOBAL_SEED}|{split}|order"))
    rng.shuffle(episodes)
    return episodes


def make_counterfactual_group(split: str, query_type: tuple[str, int, str, int], group_index: int) -> list[R3B0Episode]:
    for attempt in range(100):
        salt = f"{split}|cf|{group_index}|attempt={attempt}"
        query = render_query(query_type, group_index + attempt * 10_000, salt)
        targets = {spec: apply_transform(query, spec) for spec in TRANSFORMS}
        if len(set(targets.values())) != len(TRANSFORMS):
            continue
        support_inputs: list[tuple[tuple[int, ...], ...]] = []
        for pair_index, support_qtype in enumerate(support_query_types(query_type, group_index + attempt * 17, "counterfactual")):
            support_input = render_query(support_qtype, group_index + pair_index * 2_003 + attempt * 10_000, f"{salt}|support={pair_index}")
            if support_input == query:
                break
            support_inputs.append(support_input)
        if len(support_inputs) != 3:
            continue
        group_id = f"{split}:{query_type_key(query_type)}:cf={group_index}:attempt={attempt}"
        episodes: list[R3B0Episode] = []
        valid = True
        for spec in TRANSFORMS:
            supports = tuple(R3B0SupportPair(input=support_input, output=apply_transform(support_input, spec)) for support_input in support_inputs)
            if any(pair.output == targets[spec] for pair in supports):
                valid = False
                break
            episodes.append(
                R3B0Episode(
                    episode_id=f"{group_id}:{spec}",
                    split=split,
                    query_type=query_type,
                    query=query,
                    supports=supports,
                    target=targets[spec],
                    transform_spec=spec,
                    counterfactual_group_id=group_id,
                )
            )
        if valid:
            return episodes
    raise RuntimeError(f"failed to create counterfactual group {split} {query_type_key(query_type)} {group_index}")


def generate_counterfactual_suite(split: str, query_types: list[tuple[str, int, str, int]], groups_per_type: int) -> list[R3B0Episode]:
    suite: list[R3B0Episode] = []
    for query_type in query_types:
        for group_index in range(groups_per_type):
            suite.extend(make_counterfactual_group(split, query_type, group_index))
    return suite


def save_jsonl(path: Path, episodes: list[R3B0Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_record(), sort_keys=True) + "\n")


def episode_from_record(record: dict[str, Any]) -> R3B0Episode:
    return R3B0Episode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        query_type=query_type_from_dict(record["query_type"]),
        query=list_to_grid(record["query"]),
        supports=tuple(R3B0SupportPair(input=list_to_grid(pair["input"]), output=list_to_grid(pair["output"])) for pair in record["supports"]),
        target=list_to_grid(record["target"]),
        transform_spec=str(record["transform_spec"]),
        counterfactual_group_id=record.get("counterfactual_group_id"),
    )


def load_jsonl(path: Path) -> list[R3B0Episode]:
    with path.open("r", encoding="utf-8") as handle:
        return [episode_from_record(json.loads(line)) for line in handle if line.strip()]


def load_split(root: Path, split: str) -> list[R3B0Episode]:
    return load_jsonl(root / f"{split}.jsonl")


def load_counterfactual_suite(root: Path, name: str = "support_rule_counterfactual_suite.jsonl") -> list[R3B0Episode]:
    return load_jsonl(root / name)


def reorder_episode(episode: R3B0Episode, order: tuple[int, int, int]) -> R3B0Episode:
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def derange_episode(episode: R3B0Episode, shift: int) -> R3B0Episode:
    outputs = [pair.output for pair in episode.supports]
    deranged = tuple(
        R3B0SupportPair(input=pair.input, output=outputs[(idx + shift) % len(outputs)])
        for idx, pair in enumerate(episode.supports)
    )
    return replace(episode, supports=deranged)


def support_signature(episode: R3B0Episode) -> list[tuple[str, str]]:
    return [(stable_hash(pair.input), stable_hash(pair.output)) for pair in episode.supports]


def support_input_bag(episode: R3B0Episode) -> list[str]:
    return sorted(stable_hash(pair.input) for pair in episode.supports)


def support_output_bag(episode: R3B0Episode) -> list[str]:
    return sorted(stable_hash(pair.output) for pair in episode.supports)


def tensors_from_episodes(episodes: list[R3B0Episode], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "query_grids": torch.tensor([grid_to_list(episode.query) for episode in episodes], dtype=torch.long, device=device),
        "support_input_grids": torch.tensor(
            [[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes],
            dtype=torch.long,
            device=device,
        ),
        "support_output_grids": torch.tensor(
            [[grid_to_list(pair.output) for pair in episode.supports] for episode in episodes],
            dtype=torch.long,
            device=device,
        ),
        "target_grids": torch.tensor([grid_to_list(episode.target) for episode in episodes], dtype=torch.long, device=device),
    }


def transform_specs(episodes: list[R3B0Episode]) -> list[str]:
    return [episode.transform_spec for episode in episodes]


def query_type_sets_report(split_types: dict[str, list[tuple[str, int, str, int]]]) -> dict[str, Any]:
    sets = {split: {query_type_key(item) for item in values} for split, values in split_types.items()}
    return {
        "seed": GLOBAL_SEED,
        "query_type_counts": {split: len(values) for split, values in split_types.items()},
        "factor_coverage": {split: factor_coverage(values) for split, values in split_types.items()},
        "sets_disjoint": sets["train"].isdisjoint(sets["dev"]) and sets["train"].isdisjoint(sets["sealed"]) and sets["dev"].isdisjoint(sets["sealed"]),
        "splits": {split: sorted(values, key=query_type_key) for split, values in split_types.items()},
    }


def transform_balance_report(episodes_by_split: dict[str, list[R3B0Episode]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        by_query_type: dict[str, dict[str, int]] = {}
        counts = {spec: 0 for spec in TRANSFORMS}
        for episode in episodes:
            key = query_type_key(episode.query_type)
            by_query_type.setdefault(key, {spec: 0 for spec in TRANSFORMS})
            by_query_type[key][episode.transform_spec] += 1
            counts[episode.transform_spec] += 1
        report[split] = {
            "episode_count": len(episodes),
            "query_type_count": len(by_query_type),
            "transform_counts": counts,
            "all_query_types_balanced": all(set(row.values()) == {100} for row in by_query_type.values()),
            "transform_independent_of_query_type": all(row == {spec: 100 for spec in TRANSFORMS} for row in by_query_type.values()),
        }
    return report


def nondegeneracy_report(episodes_by_split: dict[str, list[R3B0Episode]], suites: dict[str, list[R3B0Episode]] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, episodes in episodes_by_split.items():
        total = 0
        bad = 0
        for episode in episodes:
            for grid in [episode.query, *(pair.input for pair in episode.supports)]:
                total += 1
                bad += int(not is_nondegenerate(grid))
        report[split] = {"grid_count": total, "bad_grid_count": bad, "passed": bad == 0}
    for name, episodes in (suites or {}).items():
        total = 0
        bad = 0
        for episode in episodes:
            for grid in [episode.query, *(pair.input for pair in episode.supports)]:
                total += 1
                bad += int(not is_nondegenerate(grid))
        report[name] = {"grid_count": total, "bad_grid_count": bad, "passed": bad == 0}
    return report


def counterfactual_suite_report(episodes: list[R3B0Episode]) -> dict[str, Any]:
    groups: dict[str, list[R3B0Episode]] = {}
    for episode in episodes:
        if episode.counterfactual_group_id is None:
            raise ValueError("counterfactual suite episode missing group id")
        groups.setdefault(episode.counterfactual_group_id, []).append(episode)
    bad = []
    for group_id, items in groups.items():
        queries = {stable_hash(item.query) for item in items}
        input_bags = {tuple(support_input_bag(item)) for item in items}
        target_hashes = {stable_hash(item.target) for item in items}
        transforms = {item.transform_spec for item in items}
        if len(items) != 3 or transforms != set(TRANSFORMS) or len(queries) != 1 or len(input_bags) != 1 or len(target_hashes) != 3:
            bad.append(group_id)
    return {
        "episode_count": len(episodes),
        "counterfactual_group_count": len(groups),
        "bad_group_count": len(bad),
        "bad_group_examples": bad[:10],
        "passed": not bad,
    }


def dataset_manifest(
    split_types: dict[str, list[tuple[str, int, str, int]]],
    episodes_by_split: dict[str, list[R3B0Episode]],
    counterfactual_suite: list[R3B0Episode],
) -> dict[str, Any]:
    return {
        "benchmark": "transform_r3b0_v1",
        "seed": GLOBAL_SEED,
        "grid_size": GRID_SIZE,
        "split_episode_counts": {split: len(episodes) for split, episodes in episodes_by_split.items()},
        "query_type_counts": {split: len(values) for split, values in split_types.items()},
        "query_type_sets_disjoint": query_type_sets_report(split_types)["sets_disjoint"],
        "transform_balance_report": transform_balance_report(episodes_by_split),
        "nondegeneracy_report": nondegeneracy_report(episodes_by_split, {"support_rule_counterfactual_suite": counterfactual_suite}),
        "counterfactual_suite_manifest": counterfactual_suite_report(counterfactual_suite),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_dataset(root: Path, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    if seed != GLOBAL_SEED:
        raise ValueError("R3B0 dataset seed is fixed for reproducibility")
    root.mkdir(parents=True, exist_ok=True)
    split_types = partition_query_types(seed)
    episodes_by_split = {split: generate_split(split, query_types) for split, query_types in split_types.items()}
    counterfactual_suite = generate_counterfactual_suite("sealed_cf", split_types["sealed"], groups_per_type=50)
    dev_counterfactual_suite = generate_counterfactual_suite("dev_cf", split_types["dev"], groups_per_type=10)
    for split, episodes in episodes_by_split.items():
        save_jsonl(root / f"{split}.jsonl", episodes)
    save_jsonl(root / "support_rule_counterfactual_suite.jsonl", counterfactual_suite)
    save_jsonl(root / "dev_support_rule_counterfactual_suite.jsonl", dev_counterfactual_suite)
    manifest = dataset_manifest(split_types, episodes_by_split, counterfactual_suite)
    heldout = query_type_sets_report(split_types)
    balance = transform_balance_report(episodes_by_split)
    nondegenerate = nondegeneracy_report(episodes_by_split, {"support_rule_counterfactual_suite": counterfactual_suite})
    cf_manifest = counterfactual_suite_report(counterfactual_suite)
    save_json(root / "dataset_manifest.json", manifest)
    save_json(root / "heldout_query_type_manifest.json", heldout)
    save_json(root / "transform_balance_report.json", balance)
    save_json(root / "nondegeneracy_report.json", nondegenerate)
    save_json(root / "counterfactual_suite_manifest.json", cf_manifest)
    return {
        "split_types": split_types,
        "episodes_by_split": episodes_by_split,
        "counterfactual_suite": counterfactual_suite,
        "dev_counterfactual_suite": dev_counterfactual_suite,
        "manifest": manifest,
    }
