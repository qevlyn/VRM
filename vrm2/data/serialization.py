from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from vrm2.data.operations import OPERATION_NAMES
from vrm2.data.types import Grid, TaskEpisode, grid_as_lists

MODEL_RECORD_KEYS = frozenset({"support", "query_input", "query_output"})
SUPPORT_RECORD_KEYS = frozenset({"input_grid", "output_grid"})
FORBIDDEN_MODEL_TERMS = frozenset(
    {
        "hidden_program",
        "operation",
        "composition",
        "signature",
        "task_id",
        "episode_seed",
        "split",
        "generator",
        "metadata",
        "parameters",
        *OPERATION_NAMES,
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def model_record(episode: TaskEpisode) -> dict[str, object]:
    record: dict[str, object] = {
        "support": [
            {"input_grid": grid_as_lists(pair.input_grid), "output_grid": grid_as_lists(pair.output_grid)}
            for pair in episode.support
        ],
        "query_input": grid_as_lists(episode.query_input),
        "query_output": grid_as_lists(episode.query_output),
    }
    assert_model_record_safe(record)
    return record


def generator_episode_record(episode: TaskEpisode) -> dict[str, object]:
    return {
        "episode_seed": episode.episode_seed,
        "split": episode.split,
        "composition_signature": episode.hidden_program.composition_signature,
        "hidden_program": [
            {"operation": step.name, "parameters": dict(step.parameters)} for step in episode.hidden_program.steps
        ],
        "model_record_sha256": sha256_hex(canonical_json_bytes(model_record(episode))),
    }


def episode_bytes(episode: TaskEpisode) -> bytes:
    return canonical_json_bytes(
        {
            "model_record": model_record(episode),
            "generator_record": generator_episode_record(episode),
        }
    )


def _walk_terms(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).lower()
            yield from _walk_terms(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_terms(child)
    elif isinstance(value, str):
        yield value.lower()


def assert_model_record_safe(record: Mapping[str, object]) -> None:
    if set(record) != MODEL_RECORD_KEYS:
        raise ValueError(f"unexpected model record fields: {sorted(set(record) - MODEL_RECORD_KEYS)}")
    support = record.get("support")
    if not isinstance(support, list) or not support:
        raise ValueError("model record support must be a non-empty list")
    if any(not isinstance(pair, Mapping) or set(pair) != SUPPORT_RECORD_KEYS for pair in support):
        raise ValueError("support records must contain only input_grid and output_grid")
    terms = set(_walk_terms(record))
    leaked = sorted(term for term in terms if term in FORBIDDEN_MODEL_TERMS)
    if leaked:
        raise ValueError(f"generator metadata leaked into model record: {leaked}")


def jsonl_bytes(episodes: Iterable[TaskEpisode]) -> bytes:
    lines = [canonical_json_bytes(model_record(episode)) for episode in episodes]
    return b"\n".join(lines) + (b"\n" if lines else b"")


def write_model_dataset(path: Path, episodes: Iterable[TaskEpisode]) -> tuple[int, str]:
    payload = jsonl_bytes(episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload), sha256_hex(payload)


def write_canonical_json(path: Path, value: Any) -> tuple[int, str]:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload), sha256_hex(payload)


def grid_color_histogram(grids: Iterable[Grid]) -> dict[str, int]:
    counts = {str(color): 0 for color in range(10)}
    for grid in grids:
        for row in grid:
            for value in row:
                counts[str(value)] += 1
    return counts
