from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Grid = list[list[int]]

FAMILY = "marker_to_output_bijection_v1"
BACKGROUND = 0
DEFAULT_GRID_SIZE = 7
MARKER_SHAPES = (
    ((0, 0),),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
)
OUTPUT_GLYPH = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


@dataclass(frozen=True)
class GenerationResult:
    records: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    split_base_ids: dict[str, list[str]]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def empty_grid(size: int) -> Grid:
    return [[BACKGROUND for _ in range(size)] for _ in range(size)]


def draw_shape(size: int, shape: tuple[tuple[int, int], ...], anchor: tuple[int, int], color: int) -> Grid:
    grid = empty_grid(size)
    row0, col0 = anchor
    for dr, dc in shape:
        row = row0 + dr
        col = col0 + dc
        if row < 0 or row >= size or col < 0 or col >= size:
            raise ValueError("shape anchor out of bounds")
        grid[row][col] = color
    return grid


def draw_center_output(size: int, color: int) -> Grid:
    start = (size - 3) // 2
    return draw_shape(size, OUTPUT_GLYPH, (start, start), color)


def valid_anchors(size: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    max_r = max(row for row, _ in shape)
    max_c = max(col for _, col in shape)
    return [(row, col) for row in range(size - max_r) for col in range(size - max_c)]


def choose_distinct_markers(rng: random.Random, size: int) -> list[dict[str, Any]]:
    marker_colors = rng.sample(range(1, 10), 3)
    markers = []
    for idx, shape in enumerate(MARKER_SHAPES):
        anchor = rng.choice(valid_anchors(size, shape))
        grid = draw_shape(size, shape, anchor, marker_colors[idx])
        markers.append({"marker_index": idx, "shape_index": idx, "color": marker_colors[idx], "anchor": anchor, "grid": grid})
    hashes = [stable_hash(marker["grid"]) for marker in markers]
    if len(set(hashes)) != 3:
        raise ValueError("marker grids are not distinct")
    return markers


def choose_output_symbols(rng: random.Random, size: int) -> list[dict[str, Any]]:
    colors = rng.sample(range(1, 10), 3)
    symbols = []
    for idx, color in enumerate(colors):
        symbols.append({"symbol_index": idx, "color": color, "grid": draw_center_output(size, color)})
    hashes = [stable_hash(symbol["grid"]) for symbol in symbols]
    if len(set(hashes)) != 3:
        raise ValueError("output grids are not distinct")
    return symbols


def random_derangement_of_three(rng: random.Random) -> list[int]:
    return [1, 2, 0] if rng.randrange(2) == 0 else [2, 0, 1]


def candidate_payload(symbols: list[dict[str, Any]], target_a: int, target_b: int, rng: random.Random) -> tuple[list[Grid], int, int, list[int]]:
    order = [target_a, target_b]
    if rng.randrange(2):
        order.reverse()
    label_a = order.index(target_a)
    label_b = order.index(target_b)
    return [symbols[idx]["grid"] for idx in order], label_a, label_b, order


def support_records(
    *,
    markers: list[dict[str, Any]],
    symbols: list[dict[str, Any]],
    assignment: list[int],
    pair_order: list[int],
) -> list[dict[str, Any]]:
    pairs = []
    for new_position, marker_idx in enumerate(pair_order):
        output_idx = assignment[marker_idx]
        pairs.append(
            {
                "pair_position": new_position,
                "marker_index": marker_idx,
                "output_symbol_index": output_idx,
                "input": markers[marker_idx]["grid"],
                "output": symbols[output_idx]["grid"],
            }
        )
    return pairs


def make_base_pair(base_id: str, rng: random.Random, size: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    markers = choose_distinct_markers(rng, size)
    symbols = choose_output_symbols(rng, size)
    p_a = list(range(3))
    rng.shuffle(p_a)
    derangement = random_derangement_of_three(rng)
    p_b = [p_a[derangement[idx]] for idx in range(3)]
    query_marker = rng.randrange(3)
    target_a = p_a[query_marker]
    target_b = p_b[query_marker]
    if target_a == target_b:
        raise ValueError("deranged target did not differ")
    candidates, label_a, label_b, candidate_order = candidate_payload(symbols, target_a, target_b, rng)
    pair_order_a = list(range(3))
    pair_order_b = list(range(3))
    rng.shuffle(pair_order_a)
    rng.shuffle(pair_order_b)
    metadata = {
        "base_id": base_id,
        "split": split,
        "grid_size": size,
        "pA": p_a,
        "pB": p_b,
        "derangement": derangement,
        "query_marker_index": query_marker,
        "candidate_order_symbol_indices": candidate_order,
        "pair_order_A": pair_order_a,
        "pair_order_B": pair_order_b,
        "marker_hashes": [stable_hash(marker["grid"]) for marker in markers],
        "output_hashes": [stable_hash(symbol["grid"]) for symbol in symbols],
    }
    common = {
        "family": FAMILY,
        "base_id": base_id,
        "split": split,
        "grid_size": size,
        "query": markers[query_marker]["grid"],
        "candidates": candidates,
    }
    twin_a = {
        **common,
        "twin": "A",
        "supports": support_records(markers=markers, symbols=symbols, assignment=p_a, pair_order=pair_order_a),
        "target": symbols[target_a]["grid"],
        "label": label_a,
        "metadata": metadata,
    }
    twin_b = {
        **common,
        "twin": "B",
        "supports": support_records(markers=markers, symbols=symbols, assignment=p_b, pair_order=pair_order_b),
        "target": symbols[target_b]["grid"],
        "label": label_b,
        "metadata": metadata,
    }
    return [twin_a, twin_b], metadata


def generate_records(
    *,
    seed: int,
    split_pair_counts: dict[str, int],
    grid_size: int = DEFAULT_GRID_SIZE,
) -> GenerationResult:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    split_base_ids: dict[str, list[str]] = {split: [] for split in split_pair_counts}
    seen_unpaired: set[str] = set()
    sequence = 0
    for split, desired_count in split_pair_counts.items():
        while len(split_base_ids[split]) < desired_count:
            base_id = f"{FAMILY}_{sequence:08d}"
            sequence += 1
            try:
                twins, _metadata = make_base_pair(base_id, rng, grid_size, split)
                from .dataset import unpaired_payload_bytes
                from .invariants import validate_twin_pair

                validation = validate_twin_pair(twins[0], twins[1])
                if not validation["all_passed"]:
                    raise ValueError(";".join(validation["failed"]))
                unpaired_hash = hashlib.sha256(unpaired_payload_bytes(twins[0])).hexdigest()
                if unpaired_hash in seen_unpaired:
                    raise ValueError("duplicate_unpaired_payload")
                seen_unpaired.add(unpaired_hash)
            except Exception as exc:
                rejections.append({"base_id": base_id, "split": split, "reason": str(exc)})
                continue
            records.extend(twins)
            split_base_ids[split].append(base_id)
    return GenerationResult(records=records, rejections=rejections, split_base_ids=split_base_ids)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rejection_summary(rejections: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["reason"] for row in rejections)
    return {
        "rejected_count": len(rejections),
        "rejection_reasons": dict(sorted(counts.items())),
        "rejections": rejections,
    }

