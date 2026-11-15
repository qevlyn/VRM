from __future__ import annotations

import hashlib
import itertools
import random
from typing import Iterable, Mapping, Sequence

from vrm2.data.operations import OPERATION_NAMES, OPERATION_REGISTRY_VERSION

SPLIT_VERSION = "vrm2-composition-split-v1"
VALID_SPLITS = ("train", "dev", "sealed")


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_signature(operation_names: Iterable[str]) -> str:
    names = tuple(operation_names)
    if not names:
        raise ValueError("a composition must contain at least one operation")
    if any(name not in OPERATION_NAMES for name in names):
        raise ValueError("composition contains an unknown operation")
    if len(set(names)) != len(names):
        raise ValueError("Stage-0 compositions may not repeat an operation")
    return "->".join(names)


def parse_signature(signature: str) -> tuple[str, ...]:
    names = tuple(signature.split("->"))
    if make_signature(names) != signature:
        raise ValueError(f"non-canonical composition signature: {signature!r}")
    return names


def enumerate_compositions(lengths: Sequence[int] = (1, 2, 3)) -> Mapping[int, tuple[str, ...]]:
    result: dict[int, tuple[str, ...]] = {}
    for length in lengths:
        if length < 1 or length > len(OPERATION_NAMES):
            raise ValueError(f"invalid composition length: {length}")
        result[length] = tuple(make_signature(items) for items in itertools.permutations(OPERATION_NAMES, length))
    return result


def build_split_manifest(
    split_seed: int = 20260713,
    split_version: str = SPLIT_VERSION,
) -> dict[str, object]:
    universe = enumerate_compositions()
    length_two = list(universe[2])
    random.Random(stable_seed(split_version, split_seed, "length-2")).shuffle(length_two)
    length_three = list(universe[3])
    random.Random(stable_seed(split_version, split_seed, "length-3")).shuffle(length_three)

    split_signatures = {
        "train": list(universe[1]) + length_two[:18],
        "dev": length_two[18:24],
        "sealed": length_two[24:] + length_three,
    }
    validate_split_signatures(split_signatures)
    assigned = set().union(*(set(values) for values in split_signatures.values()))
    enumerated = set().union(*(set(values) for values in universe.values()))
    if assigned != enumerated:
        raise AssertionError("composition split does not cover the enumerated universe")

    return {
        "split_version": split_version,
        "operation_registry_version": OPERATION_REGISTRY_VERSION,
        "split_seed": split_seed,
        "signature_format": "ordered operation names joined by '->'; parameters excluded",
        "validity_rule": "length 1-3 without repeated operations",
        "policy": {
            "train": "all 6 atoms and first 18 deterministically shuffled length-2 signatures",
            "dev": "next 6 deterministically shuffled length-2 signatures",
            "sealed": "final 6 deterministically shuffled length-2 signatures and all 120 length-3 signatures",
        },
        "enumerated_counts_by_length": {str(length): len(values) for length, values in universe.items()},
        "splits": {
            split: {
                "count": len(signatures),
                "length_counts": {
                    str(length): sum(len(parse_signature(signature)) == length for signature in signatures)
                    for length in (1, 2, 3)
                },
                "signatures": signatures,
            }
            for split, signatures in split_signatures.items()
        },
        "disjoint": True,
        "all_atomic_operations_seen_in_train": set(universe[1]).issubset(split_signatures["train"]),
    }


def split_signature_sets(manifest: Mapping[str, object]) -> dict[str, set[str]]:
    splits = manifest["splits"]
    if not isinstance(splits, Mapping):
        raise ValueError("manifest splits must be a mapping")
    result: dict[str, set[str]] = {}
    for split in VALID_SPLITS:
        record = splits[split]
        if not isinstance(record, Mapping) or not isinstance(record.get("signatures"), list):
            raise ValueError(f"invalid manifest record for {split}")
        result[split] = set(record["signatures"])
    return result


def validate_split_signatures(split_signatures: Mapping[str, Sequence[str]]) -> None:
    if set(split_signatures) != set(VALID_SPLITS):
        raise ValueError("composition split must define train, dev, and sealed")
    sets = {split: set(signatures) for split, signatures in split_signatures.items()}
    if any(len(sets[split]) != len(split_signatures[split]) for split in VALID_SPLITS):
        raise ValueError("a split contains duplicate composition signatures")
    if sets["train"] & sets["dev"] or sets["train"] & sets["sealed"] or sets["dev"] & sets["sealed"]:
        raise ValueError("composition signatures overlap across splits")
    for signatures in split_signatures.values():
        for signature in signatures:
            parse_signature(signature)
