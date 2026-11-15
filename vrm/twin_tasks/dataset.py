from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .generator import canonical_json, read_jsonl, stable_hash


def grid_hash(grid: list[list[int]]) -> str:
    return stable_hash(grid)


def canonical_grid_bag(grids: list[list[list[int]]]) -> list[str]:
    return sorted(grid_hash(grid) for grid in grids)


def support_input_bag(record: dict[str, Any]) -> list[str]:
    return canonical_grid_bag([support["input"] for support in record["supports"]])


def support_output_bag(record: dict[str, Any]) -> list[str]:
    return canonical_grid_bag([support["output"] for support in record["supports"]])


def unpaired_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": record["query"],
        "candidate_0": record["candidates"][0],
        "candidate_1": record["candidates"][1],
        "support_input_bag": sorted([support["input"] for support in record["supports"]], key=grid_hash),
        "support_output_bag": sorted([support["output"] for support in record["supports"]], key=grid_hash),
    }


def unpaired_payload_bytes(record: dict[str, Any]) -> bytes:
    return canonical_json(unpaired_payload(record)).encode("utf-8")


def unpaired_payload_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(unpaired_payload_bytes(record)).hexdigest()


def paired_payload(record: dict[str, Any]) -> dict[str, Any]:
    pairs = [
        {"input": support["input"], "output": support["output"]}
        for support in record["supports"]
    ]
    pairs.sort(key=lambda pair: (grid_hash(pair["input"]), grid_hash(pair["output"])))
    return {
        "query": record["query"],
        "candidate_0": record["candidates"][0],
        "candidate_1": record["candidates"][1],
        "support_pairs": pairs,
    }


def paired_payload_bytes(record: dict[str, Any]) -> bytes:
    return canonical_json(paired_payload(record)).encode("utf-8")


def paired_payload_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(paired_payload_bytes(record)).hexdigest()


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "sealed"):
        rows.extend(read_jsonl(data_dir / f"{split}.jsonl"))
    return rows


def load_split_records(data_dir: Path, split: str) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / f"{split}.jsonl")


def group_twins(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["base_id"], []).append(record)
    for twins in grouped.values():
        twins.sort(key=lambda row: row["twin"])
    return grouped


def visible_non_correspondence_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": record["family"],
        "grid_size": record["grid_size"],
        "support_count": len(record["supports"]),
        "candidate_hashes": [grid_hash(grid) for grid in record["candidates"]],
        "query_hash": grid_hash(record["query"]),
        "support_input_bag": support_input_bag(record),
        "support_output_bag": support_output_bag(record),
    }


def label_balance(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["label"] for record in records)
    total = sum(counts.values())
    return {
        "counts": {str(label): counts[label] for label in sorted(counts)},
        "total": total,
        "balanced": counts[0] == counts[1],
    }


def candidate_order_label_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter[int]] = {}
    for record in records:
        key = "|".join(grid_hash(candidate) for candidate in record["candidates"])
        counts.setdefault(key, Counter())[record["label"]] += 1
    leaking = {
        key: dict(value)
        for key, value in counts.items()
        if value[0] != value[1]
    }
    return {
        "candidate_order_groups": len(counts),
        "non_balanced_candidate_order_groups": len(leaking),
        "examples": dict(list(leaking.items())[:10]),
        "passed": not leaking,
    }


def duplicate_unpaired_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(unpaired_payload_hash(record), []).append(record)
    bad = []
    for payload_hash, rows in grouped.items():
        labels = sorted({row["label"] for row in rows})
        base_ids = sorted({row["base_id"] for row in rows})
        if len(base_ids) > 1 or labels != [0, 1] or len(rows) != 2:
            bad.append({"hash": payload_hash, "labels": labels, "base_ids": base_ids, "count": len(rows)})
    return {
        "duplicate_payload_groups": sum(1 for rows in grouped.values() if len(rows) > 1),
        "non_complementary_or_cross_base_duplicates": bad,
        "passed": not bad,
    }


def partition_integrity(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base_to_split: dict[str, str] = {}
    overlaps = []
    rendered_hash_to_split: dict[str, str] = {}
    rendered_overlaps = []
    for split, records in records_by_split.items():
        for record in records:
            previous = base_to_split.setdefault(record["base_id"], split)
            if previous != split:
                overlaps.append({"base_id": record["base_id"], "splits": sorted({previous, split})})
            rendered_support_query_grids = [record["query"], *[support["input"] for support in record["supports"]]]
            for grid in rendered_support_query_grids:
                ghash = grid_hash(grid)
                prior_split = rendered_hash_to_split.setdefault(ghash, split)
                if prior_split != split:
                    rendered_overlaps.append({"grid_hash": ghash, "splits": sorted({prior_split, split})})
    return {
        "base_instance_overlap_count": len(overlaps),
        "base_instance_overlaps": overlaps[:20],
        "rendered_grid_cross_split_overlap_note": (
            "informational only; this pilot partitions by base twin ID, and marker/output "
            "vocabularies are intentionally reused across splits"
        ),
        "rendered_grid_cross_split_overlap_count": len(rendered_overlaps),
        "rendered_grid_cross_split_overlaps": rendered_overlaps[:20],
        "passed": not overlaps,
    }
