from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.inductor.episodes import (
    exact_program_signature,
    input_grid_fingerprint,
    write_episode_corpus,
)
from vrm2.inductor.hashing import file_sha256
from vrm2.inductor.types import QueryEpisode, SupportEpisode
from vrm2.oracle.dataset import example_fingerprint
from vrm2.oracle.types import OracleExample


NULL_INSTRUCTION = (0, 0, 0, 0)
EPISODE_CORPUS_VERSION = "vrm2-inductor-v1-episodes-v1"


def _signature_bytes(signature: tuple[int, tuple[tuple[int, ...], ...]]) -> bytes:
    return canonical_json_bytes([signature[0], [list(item) for item in signature[1]]])


def assemble_length1_episode_records(
    examples: Sequence[OracleExample], split_name: str, assembly_seed: int
) -> tuple[tuple[SupportEpisode, ...], tuple[QueryEpisode, ...], dict[str, Any]]:
    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[OracleExample]] = (
        collections.defaultdict(list)
    )
    for example in examples:
        if example.program_length == 1:
            groups[exact_program_signature(example)].append(example)

    supports: list[SupportEpisode] = []
    queries: list[QueryEpisode] = []
    episode_fingerprints = []
    per_signature_counts: dict[str, int] = {}
    consumed_examples: list[str] = []
    duplicate_inputs_discarded = 0
    remainder_discarded = 0
    for signature in sorted(groups, key=_signature_bytes):
        ordered = sorted(groups[signature], key=example_fingerprint)
        local_seed = assembly_seed ^ int.from_bytes(
            hashlib.sha256(_signature_bytes(signature)).digest()[:8], "big"
        )
        random.Random(local_seed).shuffle(ordered)
        unique_inputs = []
        seen_inputs: set[str] = set()
        for example in ordered:
            fingerprint = input_grid_fingerprint(example)
            if fingerprint in seen_inputs:
                duplicate_inputs_discarded += 1
                continue
            seen_inputs.add(fingerprint)
            unique_inputs.append(example)
        usable = len(unique_inputs) - len(unique_inputs) % 4
        remainder_discarded += len(unique_inputs) - usable
        signature_key = _signature_bytes(signature).decode("utf-8")
        per_signature_counts[signature_key] = usable // 4
        for start in range(0, usable, 4):
            chunk = unique_inputs[start : start + 4]
            example_ids = [example_fingerprint(item) for item in chunk]
            input_ids = [input_grid_fingerprint(item) for item in chunk]
            if len(set(example_ids)) != 4 or len(set(input_ids)) != 4:
                raise AssertionError("V1 episode scenes must be distinct")
            if len({item.split for item in chunk}) != 1:
                raise AssertionError("V1 episode cannot cross splits")
            if len({exact_program_signature(item) for item in chunk}) != 1:
                raise AssertionError("V1 episode requires one exact program")
            episode_id = sha256_hex(
                canonical_json_bytes(["v1", split_name, signature_key, example_ids])
            )[:24]
            program = (tuple(chunk[0].typed_program_ids[0]), NULL_INSTRUCTION, NULL_INSTRUCTION)
            supports.append(
                SupportEpisode(
                    episode_id=episode_id,
                    support_input_grids=tuple(item.input_grid for item in chunk[:3]),  # type: ignore[arg-type]
                    support_output_grids=tuple(item.target_output_grid for item in chunk[:3]),  # type: ignore[arg-type]
                    target_program_length=1,
                    target_typed_program_ids=program,  # type: ignore[arg-type]
                )
            )
            queries.append(
                QueryEpisode(
                    episode_id=episode_id,
                    query_input_grid=chunk[3].input_grid,
                    query_target_grid=chunk[3].target_output_grid,
                    evaluator_program_length=1,
                    evaluator_typed_program_ids=program,  # type: ignore[arg-type]
                )
            )
            consumed_examples.extend(example_ids)
            episode_fingerprints.append(
                {
                    "episode_id": episode_id,
                    "support_example_fingerprints": example_ids[:3],
                    "query_example_fingerprint": example_ids[3],
                    "support_input_fingerprints": input_ids[:3],
                    "query_input_fingerprint": input_ids[3],
                }
            )
    if len(consumed_examples) != len(set(consumed_examples)):
        raise AssertionError("V1 length-1 example reuse detected")
    eligible = sum(item.program_length == 1 for item in examples)
    return tuple(supports), tuple(queries), {
        "episode_corpus_version": EPISODE_CORPUS_VERSION,
        "split_name": split_name,
        "assembly_seed": assembly_seed,
        "eligible_program_lengths": [1],
        "source_examples": len(examples),
        "eligible_examples": eligible,
        "ineligible_examples": len(examples) - eligible,
        "episode_count": len(supports),
        "examples_consumed": len(consumed_examples),
        "examples_discarded": len(examples) - len(consumed_examples),
        "duplicate_inputs_discarded": duplicate_inputs_discarded,
        "remainder_discarded": remainder_discarded,
        "signature_count": sum(value > 0 for value in per_signature_counts.values()),
        "per_signature_episode_counts": per_signature_counts,
        "consumed_example_fingerprints": consumed_examples,
        "support_query_fingerprints": episode_fingerprints,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    payload = b"\n".join(canonical_json_bytes(item) for item in records) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_hex(payload)


def combine_length1_and_frozen_length2(
    output_dir: Path,
    split_name: str,
    length1_support_path: Path,
    length1_query_path: Path,
    length2_support_path: Path,
    length2_query_path: Path,
) -> dict[str, Any]:
    support_records = _read_jsonl(length1_support_path) + _read_jsonl(length2_support_path)
    query_records = _read_jsonl(length1_query_path) + _read_jsonl(length2_query_path)
    support_records.sort(key=lambda item: item["episode_id"])
    query_records.sort(key=lambda item: item["episode_id"])
    support_ids = [item["episode_id"] for item in support_records]
    if support_ids != [item["episode_id"] for item in query_records]:
        raise ValueError("combined V1 support/query records are not aligned")
    if len(support_ids) != len(set(support_ids)):
        raise ValueError("combined V1 episode IDs are not unique")
    support_path = output_dir / f"{split_name}_supports.jsonl"
    query_path = output_dir / f"{split_name}_queries.jsonl"
    support_hash = _write_jsonl(support_path, support_records)
    query_hash = _write_jsonl(query_path, query_records)
    length_counts = collections.Counter(
        int(item["training_target"]["program_length"]) for item in support_records
    )
    manifest = {
        "episode_corpus_version": EPISODE_CORPUS_VERSION,
        "split_name": split_name,
        "episode_count": len(support_records),
        "program_length_counts": {str(key): length_counts[key] for key in sorted(length_counts)},
        "support_file": str(support_path),
        "query_file": str(query_path),
        "support_sha256": support_hash,
        "query_sha256": query_hash,
        "episode_corpus_sha256": sha256_hex(support_path.read_bytes() + query_path.read_bytes()),
        "sources": {
            str(length1_support_path): file_sha256(length1_support_path),
            str(length1_query_path): file_sha256(length1_query_path),
            str(length2_support_path): file_sha256(length2_support_path),
            str(length2_query_path): file_sha256(length2_query_path),
        },
    }
    manifest_path = output_dir / f"{split_name}_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def copy_frozen_episode_split(
    output_dir: Path, split_name: str, source_dir: Path, source_name: str
) -> dict[str, Any]:
    support_source = source_dir / f"{source_name}_supports.jsonl"
    query_source = source_dir / f"{source_name}_queries.jsonl"
    support_path = output_dir / f"{split_name}_supports.jsonl"
    query_path = output_dir / f"{split_name}_queries.jsonl"
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_bytes(support_source.read_bytes())
    query_path.write_bytes(query_source.read_bytes())
    support_records = _read_jsonl(support_path)
    manifest = {
        "episode_corpus_version": EPISODE_CORPUS_VERSION,
        "split_name": split_name,
        "episode_count": len(support_records),
        "program_length_counts": dict(
            sorted(
                collections.Counter(
                    str(item["training_target"]["program_length"])
                    for item in support_records
                ).items()
            )
        ),
        "support_file": str(support_path),
        "query_file": str(query_path),
        "support_sha256": file_sha256(support_path),
        "query_sha256": file_sha256(query_path),
        "source_support": str(support_source),
        "source_query": str(query_source),
        "byte_identical_to_frozen_v0": (
            file_sha256(support_path) == file_sha256(support_source)
            and file_sha256(query_path) == file_sha256(query_source)
        ),
    }
    (output_dir / f"{split_name}_manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    return manifest


def prepare_v1_episode_corpora(
    datasets: Mapping[str, Sequence[OracleExample]],
    output_dir: Path,
    frozen_v0_dir: Path,
    assembly_seeds: Mapping[str, int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    length1_manifests = {}
    for split_name, dataset_name in (("train", "train"), ("train_eval", "train_eval")):
        supports, queries, manifest = assemble_length1_episode_records(
            datasets[dataset_name], f"{split_name}_length1", int(assembly_seeds[split_name])
        )
        length1_manifests[split_name] = write_episode_corpus(
            output_dir, f"{split_name}_length1", supports, queries, manifest
        )

    manifests = {
        "train": combine_length1_and_frozen_length2(
            output_dir,
            "train",
            output_dir / "train_length1_supports.jsonl",
            output_dir / "train_length1_queries.jsonl",
            frozen_v0_dir / "train_supports.jsonl",
            frozen_v0_dir / "train_queries.jsonl",
        ),
        "train_eval": combine_length1_and_frozen_length2(
            output_dir,
            "train_eval",
            output_dir / "train_eval_length1_supports.jsonl",
            output_dir / "train_eval_length1_queries.jsonl",
            frozen_v0_dir / "train_eval_supports.jsonl",
            frozen_v0_dir / "train_eval_queries.jsonl",
        ),
        "dev": copy_frozen_episode_split(output_dir, "dev", frozen_v0_dir, "dev"),
        "sealed_l2": copy_frozen_episode_split(
            output_dir, "sealed_l2", frozen_v0_dir, "sealed_l2"
        ),
        "original_l3": copy_frozen_episode_split(
            output_dir, "original_l3", frozen_v0_dir, "sealed_l3"
        ),
    }
    train_lengths = manifests["train"]["program_length_counts"]
    audit = {
        "length_1_episodes": int(train_lengths.get("1", 0)),
        "length_2_episodes": int(train_lengths.get("2", 0)),
        "excluded_length_3_episodes": int(
            manifests["original_l3"]["program_length_counts"].get("3", 0)
        ),
        "maximum_active_target_length_observed_by_optimizer": max(
            int(value) for value in train_lengths
        ),
        "optimizer_allowed_target_lengths": [1, 2],
    }
    if audit["maximum_active_target_length_observed_by_optimizer"] != 2:
        raise AssertionError("V1 optimizer target-length invariant failed")
    (output_dir / "training_data_audit.json").write_bytes(
        canonical_json_bytes(audit) + b"\n"
    )
    summary = {
        "length1_manifests": length1_manifests,
        "episode_corpora": manifests,
        "training_data_audit": audit,
    }
    (output_dir / "assembly_summary.json").write_bytes(
        canonical_json_bytes(summary) + b"\n"
    )
    return summary
