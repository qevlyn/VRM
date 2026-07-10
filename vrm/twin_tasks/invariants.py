from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .dataset import (
    grid_hash,
    paired_payload_hash,
    support_input_bag,
    support_output_bag,
    unpaired_payload_bytes,
    visible_non_correspondence_metadata,
)


def assignment_from_record(record: dict[str, Any]) -> dict[int, int]:
    return {support["marker_index"]: support["output_symbol_index"] for support in record["supports"]}


def is_bijection(mapping: dict[int, int]) -> bool:
    return sorted(mapping) == [0, 1, 2] and sorted(mapping.values()) == [0, 1, 2]


def is_deranged_relative(mapping_a: dict[int, int], mapping_b: dict[int, int]) -> bool:
    return all(mapping_a[idx] != mapping_b[idx] for idx in sorted(mapping_a))


def validate_twin_pair(twin_a: dict[str, Any], twin_b: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["same_base_id"] = twin_a["base_id"] == twin_b["base_id"]
    checks["same_split"] = twin_a["split"] == twin_b["split"]
    checks["query_A_equals_query_B"] = twin_a["query"] == twin_b["query"]
    checks["candidate_grids_and_order_equal"] = twin_a["candidates"] == twin_b["candidates"]
    checks["support_input_multiset_equal"] = support_input_bag(twin_a) == support_input_bag(twin_b)
    checks["support_output_multiset_equal"] = support_output_bag(twin_a) == support_output_bag(twin_b)
    checks["support_count_equal"] = len(twin_a["supports"]) == len(twin_b["supports"])
    checks["visible_non_correspondence_metadata_equal"] = (
        visible_non_correspondence_metadata(twin_a) == visible_non_correspondence_metadata(twin_b)
    )
    checks["labels_complementary"] = twin_a["label"] != twin_b["label"] and twin_a["label"] + twin_b["label"] == 1
    checks["targets_differ"] = twin_a["target"] != twin_b["target"]
    mapping_a = assignment_from_record(twin_a)
    mapping_b = assignment_from_record(twin_b)
    checks["pair_mapping_A_differs_from_B"] = mapping_a != mapping_b
    checks["assignment_A_bijective"] = is_bijection(mapping_a)
    checks["assignment_B_bijective"] = is_bijection(mapping_b)
    checks["rule_B_deranged_relative_to_A"] = is_deranged_relative(mapping_a, mapping_b)
    checks["unpaired_payloads_byte_identical"] = unpaired_payload_bytes(twin_a) == unpaired_payload_bytes(twin_b)
    checks["paired_payloads_differ"] = paired_payload_hash(twin_a) != paired_payload_hash(twin_b)
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "base_id": twin_a.get("base_id"),
        "all_passed": not failed,
        "checks": checks,
        "failed": failed,
        "mapping_A": mapping_a,
        "mapping_B": mapping_b,
    }


def validate_all_twins(grouped_twins: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    failed = []
    for base_id, twins in sorted(grouped_twins.items()):
        if len(twins) != 2:
            row = {"base_id": base_id, "all_passed": False, "failed": ["base_id_does_not_have_two_twins"]}
        else:
            by_side = {row["twin"]: row for row in twins}
            row = validate_twin_pair(by_side["A"], by_side["B"]) if set(by_side) == {"A", "B"} else {
                "base_id": base_id,
                "all_passed": False,
                "failed": ["missing_A_or_B_twin"],
            }
        rows.append(row)
        if not row["all_passed"]:
            failed.append(row)
    check_counts: Counter[str] = Counter()
    for row in rows:
        for name, passed in row.get("checks", {}).items():
            if passed:
                check_counts[f"{name}:passed"] += 1
            else:
                check_counts[f"{name}:failed"] += 1
    return {
        "base_pair_count": len(grouped_twins),
        "failed_base_pair_count": len(failed),
        "all_passed": not failed,
        "check_counts": dict(sorted(check_counts.items())),
        "failed_examples": failed[:20],
        "rows": rows,
    }


def unpaired_payload_hash_report(grouped_twins: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    failures = []
    for base_id, twins in sorted(grouped_twins.items()):
        if len(twins) != 2:
            continue
        left, right = sorted(twins, key=lambda row: row["twin"])
        same = unpaired_payload_bytes(left) == unpaired_payload_bytes(right)
        row = {
            "base_id": base_id,
            "hash_A": hashlib.sha256(unpaired_payload_bytes(left)).hexdigest(),
            "hash_B": hashlib.sha256(unpaired_payload_bytes(right)).hexdigest(),
            "byte_identical": same,
            "labels": [left["label"], right["label"]],
        }
        rows.append(row)
        if not same:
            failures.append(row)
    return {
        "pair_count": len(rows),
        "failed_count": len(failures),
        "all_passed": not failures,
        "failed_examples": failures[:20],
        "rows": rows,
    }
