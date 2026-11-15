from __future__ import annotations

import collections
import json
import platform
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from vrm2.data.composition_split import build_split_manifest
from vrm2.data.microworld import generate_dataset
from vrm2.data.operations import execute_operation
from vrm2.data.program_schema import TYPED_PROGRAM_SCHEMA_VERSION
from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.data.types import Split, TaskEpisode, grid_as_lists, make_grid
from vrm2.oracle.types import OracleBatch, OracleExample, TypedProgramIds

ORACLE_DATASET_VERSION = "vrm2-oracle-dataset-v1"
ORACLE_MODEL_INPUT_FIELDS = ("input_grid", "typed_program_ids")


def example_from_episode(episode: TaskEpisode) -> OracleExample:
    state = episode.query_input
    intermediate = []
    for step in episode.hidden_program.steps:
        state = execute_operation(state, step)
        intermediate.append(state)
    if state != episode.query_output:
        raise ValueError("oracle intermediate replay does not reach the episode target")
    typed_program: TypedProgramIds = tuple(
        instruction.as_ids() for instruction in episode.hidden_program.typed_instructions
    )
    return OracleExample(
        input_grid=episode.query_input,
        typed_program_ids=typed_program,
        target_output_grid=episode.query_output,
        intermediate_target_grids=tuple(intermediate),
        episode_seed=episode.episode_seed,
        split=episode.split,
        composition_signature=episode.hidden_program.composition_signature,
        diagnostic_bucket=f"{episode.split}_len{len(typed_program)}",
    )


def build_oracle_examples(
    stage0_config: Mapping[str, Any],
    split: Split,
    num_examples: int,
    dataset_seed: int,
) -> tuple[OracleExample, ...]:
    split_manifest = build_split_manifest(
        int(stage0_config["seeds"]["composition_split"]),
        str(stage0_config["split_version"]),
    )
    episodes, _, _ = generate_dataset(
        stage0_config,
        split,
        num_examples,
        dataset_seed,
        split_manifest,
    )
    return tuple(example_from_episode(episode) for episode in episodes)


def oracle_record(example: OracleExample) -> dict[str, object]:
    return {
        "input_grid": grid_as_lists(example.input_grid),
        "typed_program_ids": [list(instruction) for instruction in example.typed_program_ids],
        "target_output_grid": grid_as_lists(example.target_output_grid),
        "intermediate_target_grids": [grid_as_lists(grid) for grid in example.intermediate_target_grids],
        "episode_seed": example.episode_seed,
        "diagnostics": {
            "split": example.split,
            "composition_signature": example.composition_signature,
            "bucket": example.diagnostic_bucket,
        },
    }


def parse_oracle_record(record: Mapping[str, Any]) -> OracleExample:
    diagnostics = record["diagnostics"]
    typed = tuple(tuple(int(value) for value in instruction) for instruction in record["typed_program_ids"])
    if any(len(instruction) != 4 for instruction in typed):
        raise ValueError("every typed instruction must contain four integer factors")
    return OracleExample(
        input_grid=make_grid(record["input_grid"], expected_size=8),
        typed_program_ids=typed,
        target_output_grid=make_grid(record["target_output_grid"], expected_size=8),
        intermediate_target_grids=tuple(
            make_grid(grid, expected_size=8) for grid in record["intermediate_target_grids"]
        ),
        episode_seed=int(record["episode_seed"]),
        split=diagnostics["split"],
        composition_signature=str(diagnostics["composition_signature"]),
        diagnostic_bucket=str(diagnostics["bucket"]),
    )


def corpus_bytes(examples: Iterable[OracleExample]) -> bytes:
    lines = [canonical_json_bytes(oracle_record(example)) for example in examples]
    return b"\n".join(lines) + (b"\n" if lines else b"")


def write_oracle_corpus(
    path: Path,
    examples: Sequence[OracleExample],
    *,
    dataset_seed: int,
    versions: Mapping[str, str],
) -> dict[str, object]:
    payload = corpus_bytes(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    signature_counts = collections.Counter(example.composition_signature for example in examples)
    length_counts = collections.Counter(example.program_length for example in examples)
    manifest: dict[str, object] = {
        "oracle_dataset_version": ORACLE_DATASET_VERSION,
        "typed_program_schema_version": TYPED_PROGRAM_SCHEMA_VERSION,
        **dict(versions),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_seed": dataset_seed,
        "num_examples": len(examples),
        "dataset_sha256": sha256_hex(payload),
        "dataset_bytes": len(payload),
        "program_length_counts": {str(key): length_counts[key] for key in sorted(length_counts)},
        "signature_counts": dict(sorted(signature_counts.items())),
        "model_input_fields": list(ORACLE_MODEL_INPUT_FIELDS),
        "target_only_fields": ["target_output_grid", "intermediate_target_grids"],
        "diagnostic_only_fields": ["episode_seed", "split", "composition_signature", "diagnostic_bucket"],
    }
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_oracle_corpus(path: Path, expected_sha256: str | None = None) -> tuple[OracleExample, ...]:
    payload = path.read_bytes()
    if expected_sha256 is not None and sha256_hex(payload) != expected_sha256:
        raise ValueError("oracle corpus SHA-256 does not match the pinned manifest")
    examples = tuple(parse_oracle_record(json.loads(line)) for line in payload.splitlines() if line)
    if not examples:
        raise ValueError("oracle corpus is empty")
    return examples


def make_oracle_batch(examples: Sequence[OracleExample]) -> OracleBatch:
    if not examples:
        raise ValueError("cannot batch zero oracle examples")
    lengths = {example.program_length for example in examples}
    if len(lengths) != 1:
        raise ValueError("oracle batches must be homogeneous in program length")
    return OracleBatch(
        input_grid=torch.tensor([example.input_grid for example in examples], dtype=torch.long),
        typed_program_ids=torch.tensor([example.typed_program_ids for example in examples], dtype=torch.long),
        target_output_grid=torch.tensor([example.target_output_grid for example in examples], dtype=torch.long),
        intermediate_target_grids=torch.tensor(
            [example.intermediate_target_grids for example in examples], dtype=torch.long
        ),
    )


def iter_oracle_batches(
    examples: Sequence[OracleExample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[OracleBatch, tuple[int, ...]]]:
    by_length: dict[int, list[int]] = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_length[example.program_length].append(index)
    rng = random.Random(seed)
    chunks: list[tuple[int, ...]] = []
    for length in sorted(by_length):
        indices = by_length[length]
        if shuffle:
            rng.shuffle(indices)
        chunks.extend(tuple(indices[start : start + batch_size]) for start in range(0, len(indices), batch_size))
    if shuffle:
        rng.shuffle(chunks)
    for indices in chunks:
        yield make_oracle_batch([examples[index] for index in indices]), indices


def example_fingerprint(example: OracleExample) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "input_grid": example.input_grid,
                "typed_program_ids": example.typed_program_ids,
                "target_output_grid": example.target_output_grid,
            }
        )
    )


def assert_disjoint_examples(left: Sequence[OracleExample], right: Sequence[OracleExample]) -> None:
    left_hashes = {example_fingerprint(example) for example in left}
    right_hashes = {example_fingerprint(example) for example in right}
    overlap = left_hashes & right_hashes
    if overlap:
        raise ValueError(f"oracle corpora overlap in {len(overlap)} exact examples")
