from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.data.types import grid_as_lists, make_grid
from vrm2.oracle.dataset import example_fingerprint
from vrm2.oracle.types import OracleExample
from vrm2.inductor.types import PaddedTypedProgramIds, QueryEpisode, SupportEpisode

EPISODE_CORPUS_VERSION = "vrm2-inductor-episodes-v0"
NULL_INSTRUCTION = (0, 0, 0, 0)


def padded_program(example: OracleExample) -> PaddedTypedProgramIds:
    instructions = list(example.typed_program_ids)
    if len(instructions) not in (2, 3):
        raise ValueError("inductor V0 episodes require program length 2 or 3")
    instructions.extend([NULL_INSTRUCTION] * (3 - len(instructions)))
    return tuple(instructions)  # type: ignore[return-value]


def exact_program_signature(example: OracleExample) -> tuple[int, tuple[tuple[int, ...], ...]]:
    return example.program_length, tuple(tuple(item) for item in example.typed_program_ids)


def _signature_bytes(signature: tuple[int, tuple[tuple[int, ...], ...]]) -> bytes:
    return canonical_json_bytes([signature[0], [list(item) for item in signature[1]]])


def input_grid_fingerprint(example: OracleExample) -> str:
    return sha256_hex(canonical_json_bytes(grid_as_lists(example.input_grid)))


def assemble_episode_records(
    examples: Sequence[OracleExample],
    split_name: str,
    assembly_seed: int,
    eligible_lengths: Sequence[int],
) -> tuple[tuple[SupportEpisode, ...], tuple[QueryEpisode, ...], dict[str, Any]]:
    eligible_length_set = set(int(value) for value in eligible_lengths)
    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[OracleExample]] = collections.defaultdict(list)
    ineligible = 0
    for example in examples:
        if example.program_length not in eligible_length_set:
            ineligible += 1
            continue
        groups[exact_program_signature(example)].append(example)

    supports: list[SupportEpisode] = []
    queries: list[QueryEpisode] = []
    signature_episode_counts: dict[str, int] = {}
    consumed_fingerprints: list[str] = []
    consumed_input_fingerprints: list[str] = []
    episode_fingerprints: list[dict[str, Any]] = []
    duplicate_inputs_discarded = 0
    remainder_discarded = 0

    for signature in sorted(groups, key=_signature_bytes):
        ordered = sorted(groups[signature], key=example_fingerprint)
        local_seed = assembly_seed ^ int.from_bytes(
            hashlib.sha256(_signature_bytes(signature)).digest()[:8], "big"
        )
        random.Random(local_seed).shuffle(ordered)
        unique_inputs: list[OracleExample] = []
        seen_inputs: set[str] = set()
        for example in ordered:
            input_fingerprint = input_grid_fingerprint(example)
            if input_fingerprint in seen_inputs:
                duplicate_inputs_discarded += 1
                continue
            seen_inputs.add(input_fingerprint)
            unique_inputs.append(example)
        usable_count = len(unique_inputs) - len(unique_inputs) % 4
        remainder_discarded += len(unique_inputs) - usable_count
        signature_key = _signature_bytes(signature).decode("utf-8")
        signature_episode_counts[signature_key] = usable_count // 4
        for start in range(0, usable_count, 4):
            chunk = unique_inputs[start : start + 4]
            example_hashes = [example_fingerprint(item) for item in chunk]
            input_hashes = [input_grid_fingerprint(item) for item in chunk]
            if len(set(example_hashes)) != 4 or len(set(input_hashes)) != 4:
                raise AssertionError("episode examples and inputs must be distinct")
            if len({item.split for item in chunk}) != 1:
                raise AssertionError("episode cannot cross original splits")
            if len({exact_program_signature(item) for item in chunk}) != 1:
                raise AssertionError("episode program signatures must be identical")
            episode_id = sha256_hex(
                canonical_json_bytes([split_name, signature_key, example_hashes])
            )[:24]
            program = padded_program(chunk[0])
            supports.append(
                SupportEpisode(
                    episode_id=episode_id,
                    support_input_grids=tuple(item.input_grid for item in chunk[:3]),  # type: ignore[arg-type]
                    support_output_grids=tuple(item.target_output_grid for item in chunk[:3]),  # type: ignore[arg-type]
                    target_program_length=chunk[0].program_length,
                    target_typed_program_ids=program,
                )
            )
            queries.append(
                QueryEpisode(
                    episode_id=episode_id,
                    query_input_grid=chunk[3].input_grid,
                    query_target_grid=chunk[3].target_output_grid,
                    evaluator_program_length=chunk[0].program_length,
                    evaluator_typed_program_ids=program,
                )
            )
            consumed_fingerprints.extend(example_hashes)
            consumed_input_fingerprints.extend(input_hashes)
            episode_fingerprints.append(
                {
                    "episode_id": episode_id,
                    "support_example_fingerprints": example_hashes[:3],
                    "query_example_fingerprint": example_hashes[3],
                    "support_input_fingerprints": input_hashes[:3],
                    "query_input_fingerprint": input_hashes[3],
                }
            )

    if len(consumed_fingerprints) != len(set(consumed_fingerprints)):
        raise AssertionError("an example was reused within an assembled corpus")
    consumed = len(consumed_fingerprints)
    manifest: dict[str, Any] = {
        "episode_corpus_version": EPISODE_CORPUS_VERSION,
        "split_name": split_name,
        "assembly_seed": assembly_seed,
        "eligible_program_lengths": sorted(eligible_length_set),
        "source_examples": len(examples),
        "eligible_examples": len(examples) - ineligible,
        "ineligible_examples": ineligible,
        "episode_count": len(supports),
        "examples_consumed": consumed,
        "examples_discarded": len(examples) - consumed,
        "duplicate_inputs_discarded": duplicate_inputs_discarded,
        "remainder_discarded": remainder_discarded,
        "signature_count": sum(count > 0 for count in signature_episode_counts.values()),
        "per_signature_episode_counts": signature_episode_counts,
        "episode_ids": [item.episode_id for item in supports],
        "consumed_example_fingerprints": consumed_fingerprints,
        "consumed_input_fingerprints": consumed_input_fingerprints,
        "support_query_fingerprints": episode_fingerprints,
    }
    return tuple(supports), tuple(queries), manifest


def _support_record(episode: SupportEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "supports": [
            {
                "input_grid": grid_as_lists(input_grid),
                "output_grid": grid_as_lists(output_grid),
            }
            for input_grid, output_grid in zip(
                episode.support_input_grids, episode.support_output_grids, strict=True
            )
        ],
        "training_target": {
            "program_length": episode.target_program_length,
            "typed_program_ids": [list(item) for item in episode.target_typed_program_ids],
        },
    }


def _query_record(episode: QueryEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "query_input_grid": grid_as_lists(episode.query_input_grid),
        "query_target_grid": grid_as_lists(episode.query_target_grid),
        "evaluator_private": {
            "program_length": episode.evaluator_program_length,
            "typed_program_ids": [list(item) for item in episode.evaluator_typed_program_ids],
        },
    }


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [canonical_json_bytes(record) for record in records]
    return b"\n".join(lines) + (b"\n" if lines else b"")


def write_episode_corpus(
    output_dir: Path,
    split_name: str,
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    support_bytes = _jsonl_bytes(_support_record(item) for item in supports)
    query_bytes = _jsonl_bytes(_query_record(item) for item in queries)
    support_path = output_dir / f"{split_name}_supports.jsonl"
    query_path = output_dir / f"{split_name}_queries.jsonl"
    support_path.write_bytes(support_bytes)
    query_path.write_bytes(query_bytes)
    final_manifest = {
        **dict(manifest),
        "support_file": str(support_path),
        "query_file": str(query_path),
        "support_sha256": sha256_hex(support_bytes),
        "query_sha256": sha256_hex(query_bytes),
        "episode_corpus_sha256": sha256_hex(support_bytes + query_bytes),
    }
    manifest_path = output_dir / f"{split_name}_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(final_manifest) + b"\n")
    return final_manifest


def load_support_episodes(path: Path) -> tuple[SupportEpisode, ...]:
    episodes = []
    for line in path.read_bytes().splitlines():
        record = json.loads(line)
        supports = record["supports"]
        target = record["training_target"]
        episodes.append(
            SupportEpisode(
                episode_id=record["episode_id"],
                support_input_grids=tuple(
                    make_grid(item["input_grid"], expected_size=8) for item in supports
                ),  # type: ignore[arg-type]
                support_output_grids=tuple(
                    make_grid(item["output_grid"], expected_size=8) for item in supports
                ),  # type: ignore[arg-type]
                target_program_length=int(target["program_length"]),
                target_typed_program_ids=tuple(
                    tuple(int(value) for value in item) for item in target["typed_program_ids"]
                ),  # type: ignore[arg-type]
            )
        )
    if not episodes:
        raise ValueError("support episode corpus is empty")
    return tuple(episodes)


def load_query_episodes(path: Path) -> tuple[QueryEpisode, ...]:
    episodes = []
    for line in path.read_bytes().splitlines():
        record = json.loads(line)
        private = record["evaluator_private"]
        episodes.append(
            QueryEpisode(
                episode_id=record["episode_id"],
                query_input_grid=make_grid(record["query_input_grid"], expected_size=8),
                query_target_grid=make_grid(record["query_target_grid"], expected_size=8),
                evaluator_program_length=int(private["program_length"]),
                evaluator_typed_program_ids=tuple(
                    tuple(int(value) for value in item) for item in private["typed_program_ids"]
                ),  # type: ignore[arg-type]
            )
        )
    if not episodes:
        raise ValueError("query episode corpus is empty")
    return tuple(episodes)
