from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

COLOR_IDS = tuple(range(10))
ARC_COLOR_COUNT = 10
PAD_ID = 10
VOCAB_SIZE = 11
MAX_GRID_HEIGHT = 30
MAX_GRID_WIDTH = 30
MAX_GRID_TOKENS = MAX_GRID_HEIGHT * MAX_GRID_WIDTH
ROLE_SUPPORT_INPUT = 0
ROLE_SUPPORT_OUTPUT = 1
MAX_SUPPORT_PAIRS = 16
LOCAL_ARC_DATA_ROOT = Path("/mnt/d/VRM/arc-prize-2026-arc-agi-2")
LEGACY_ARC_DATA_ROOT = Path("/mnt/d/ARC2/arc-prize-2026-arc-agi-2")
DEFAULT_ARC_DATA_ROOT = LOCAL_ARC_DATA_ROOT
DEFAULT_SPLIT_MANIFEST = Path("/mnt/d/VRM/artifacts/task_split_manifest.json")


@dataclass(frozen=True)
class Grid:
    cells: tuple[tuple[int, ...], ...]

    @classmethod
    def from_raw(cls, raw: Any) -> "Grid":
        if not isinstance(raw, list) or not raw:
            raise ValueError("grid must be a non-empty list of rows")
        rows: list[tuple[int, ...]] = []
        width: int | None = None
        for row in raw:
            if not isinstance(row, list) or not row:
                raise ValueError("grid rows must be non-empty lists")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError("grid rows must be rectangular")
            parsed: list[int] = []
            for value in row:
                if not isinstance(value, int) or value not in COLOR_IDS:
                    raise ValueError(f"ARC cell must be an integer color 0..9, got {value!r}")
                parsed.append(value)
            rows.append(tuple(parsed))
        grid = cls(tuple(rows))
        if grid.height > MAX_GRID_HEIGHT or grid.width > MAX_GRID_WIDTH:
            raise ValueError(f"grid shape {grid.shape} exceeds {MAX_GRID_HEIGHT}x{MAX_GRID_WIDTH}")
        return grid

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def to_list(self) -> list[list[int]]:
        return [list(row) for row in self.cells]


@dataclass(frozen=True)
class ArcPair:
    input: Grid
    output: Grid
    pair_index: int


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[ArcPair, ...]
    test_inputs: tuple[Grid, ...]
    source_file: str


@dataclass(frozen=True)
class ArcEpisode:
    task_id: str
    source_split: str
    supports: tuple[ArcPair, ...]
    query: ArcPair
    query_pair_index: int
    source_file: str


def discover_arc_training_files(data_root: Path = DEFAULT_ARC_DATA_ROOT) -> dict[str, Any]:
    data_root = Path(data_root)
    json_files = sorted(path for path in data_root.rglob("*.json") if path.is_file())
    training_challenges = [
        path for path in json_files if "training" in path.name.lower() and "challenges" in path.name.lower()
    ]
    training_solutions = [
        path for path in json_files if "training" in path.name.lower() and "solutions" in path.name.lower()
    ]
    if len(training_challenges) != 1:
        raise FileNotFoundError(f"expected exactly one training challenge JSON, found {training_challenges}")
    if len(training_solutions) != 1:
        raise FileNotFoundError(f"expected exactly one training solution JSON, found {training_solutions}")
    return {
        "data_root": str(data_root),
        "training_challenges": str(training_challenges[0]),
        "training_solutions": str(training_solutions[0]),
        "all_json_files": [str(path) for path in json_files],
    }


def load_arc_training_tasks(data_root: Path = DEFAULT_ARC_DATA_ROOT) -> tuple[list[ArcTask], dict[str, Any]]:
    sources = discover_arc_training_files(data_root)
    challenge_path = Path(sources["training_challenges"])
    with challenge_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("training challenges JSON must be a task-id dictionary")

    tasks: list[ArcTask] = []
    for task_id in sorted(raw):
        payload = raw[task_id]
        if not isinstance(payload, dict):
            raise ValueError(f"task {task_id} payload must be a dictionary")
        train_pairs = []
        for idx, pair in enumerate(payload.get("train", [])):
            if "input" not in pair or "output" not in pair:
                raise ValueError(f"train pair {idx} in task {task_id} lacks input/output")
            train_pairs.append(
                ArcPair(
                    input=Grid.from_raw(pair["input"]),
                    output=Grid.from_raw(pair["output"]),
                    pair_index=idx,
                )
            )
        test_inputs = []
        for pair in payload.get("test", []):
            if "input" not in pair:
                raise ValueError(f"test pair in task {task_id} lacks input")
            test_inputs.append(Grid.from_raw(pair["input"]))
        tasks.append(
            ArcTask(
                task_id=task_id,
                train=tuple(train_pairs),
                test_inputs=tuple(test_inputs),
                source_file=str(challenge_path),
            )
        )
    return tasks, sources


def split_tasks(
    tasks: list[ArcTask],
    seed: int = 12345,
    validation_fraction: float = 0.2,
) -> tuple[dict[str, str], dict[str, str]]:
    skipped = {task.task_id: "too_few_train_pairs" for task in tasks if len(task.train) < 2}
    usable_ids = [task.task_id for task in tasks if task.task_id not in skipped]
    rng = random.Random(seed)
    shuffled = usable_ids[:]
    rng.shuffle(shuffled)
    if not shuffled:
        return {}, skipped
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_ids = set(shuffled[:validation_count])
    split = {task_id: ("validation" if task_id in validation_ids else "train") for task_id in usable_ids}
    return split, skipped


def build_leave_one_out_episodes(
    tasks: list[ArcTask],
    split_assignments: dict[str, str] | None = None,
) -> tuple[list[ArcEpisode], dict[str, str]]:
    skipped: dict[str, str] = {}
    episodes: list[ArcEpisode] = []
    for task in tasks:
        if len(task.train) < 2:
            skipped[task.task_id] = "too_few_train_pairs"
            continue
        source_split = split_assignments.get(task.task_id, "all") if split_assignments else "all"
        for query_idx, query_pair in enumerate(task.train):
            supports = tuple(pair for idx, pair in enumerate(task.train) if idx != query_idx)
            if not supports:
                raise AssertionError("leave-one-out episode must have at least one support")
            episodes.append(
                ArcEpisode(
                    task_id=task.task_id,
                    source_split=source_split,
                    supports=supports,
                    query=query_pair,
                    query_pair_index=query_pair.pair_index,
                    source_file=task.source_file,
                )
            )
    return episodes, skipped


def write_split_manifest(
    path: Path,
    sources: dict[str, Any],
    seed: int,
    split_assignments: dict[str, str],
    skipped: dict[str, str],
    episodes: list[ArcEpisode],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    train_task_ids = sorted(task_id for task_id, split in split_assignments.items() if split == "train")
    validation_task_ids = sorted(
        task_id for task_id, split in split_assignments.items() if split == "validation"
    )
    episode_counts = Counter(ep.source_split for ep in episodes)
    manifest = {
        "source_json_filenames": {
            "training_challenges": sources["training_challenges"],
            "training_solutions_discovered_not_used_for_episodes": sources["training_solutions"],
        },
        "split_seed": seed,
        "train_task_ids": train_task_ids,
        "validation_task_ids": validation_task_ids,
        "episode_counts": dict(sorted(episode_counts.items())),
        "skipped_task_ids_and_reasons": dict(sorted(skipped.items())),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_dataset(
    data_root: Path = DEFAULT_ARC_DATA_ROOT,
    seed: int = 12345,
    validation_fraction: float = 0.2,
    manifest_path: Path | None = DEFAULT_SPLIT_MANIFEST,
) -> dict[str, Any]:
    tasks, sources = load_arc_training_tasks(data_root)
    split_assignments, split_skipped = split_tasks(tasks, seed=seed, validation_fraction=validation_fraction)
    episodes, episode_skipped = build_leave_one_out_episodes(tasks, split_assignments)
    skipped = {**episode_skipped, **split_skipped}
    train_episodes = [ep for ep in episodes if ep.source_split == "train"]
    validation_episodes = [ep for ep in episodes if ep.source_split == "validation"]
    manifest = None
    if manifest_path is not None:
        manifest = write_split_manifest(
            manifest_path,
            sources,
            seed,
            split_assignments,
            skipped,
            episodes,
        )
    return {
        "tasks": tasks,
        "sources": sources,
        "split_assignments": split_assignments,
        "skipped": skipped,
        "episodes": episodes,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "manifest": manifest,
    }


def pad_grid_to_tokens(grid: Grid) -> torch.Tensor:
    tokens = torch.full((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), PAD_ID, dtype=torch.long)
    raw = torch.tensor(grid.to_list(), dtype=torch.long)
    tokens[: grid.height, : grid.width] = raw
    return tokens.reshape(MAX_GRID_TOKENS)


def grid_shape_mask(height: int, width: int) -> torch.Tensor:
    if height < 1 or height > MAX_GRID_HEIGHT or width < 1 or width > MAX_GRID_WIDTH:
        raise ValueError(f"grid shape {height}x{width} is outside 1..{MAX_GRID_HEIGHT} by 1..{MAX_GRID_WIDTH}")
    mask = torch.zeros((MAX_GRID_HEIGHT, MAX_GRID_WIDTH), dtype=torch.bool)
    mask[:height, :width] = True
    return mask.reshape(MAX_GRID_TOKENS)


def collate_episodes(episodes: list[ArcEpisode] | tuple[ArcEpisode, ...]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    batch_size = len(episodes)
    max_supports = max(len(ep.supports) for ep in episodes)
    max_evidence_tokens = max(
        sum(pair.input.height * pair.input.width + pair.output.height * pair.output.width for pair in ep.supports)
        for ep in episodes
    )
    support_inputs = torch.full(
        (batch_size, max_supports, MAX_GRID_TOKENS), PAD_ID, dtype=torch.long
    )
    support_outputs = torch.full(
        (batch_size, max_supports, MAX_GRID_TOKENS), PAD_ID, dtype=torch.long
    )
    support_mask = torch.zeros((batch_size, max_supports), dtype=torch.bool)
    query_inputs = torch.full((batch_size, MAX_GRID_TOKENS), PAD_ID, dtype=torch.long)
    query_input_mask = torch.zeros((batch_size, MAX_GRID_TOKENS), dtype=torch.bool)
    query_input_heights = torch.empty((batch_size,), dtype=torch.long)
    query_input_widths = torch.empty((batch_size,), dtype=torch.long)
    query_output_heights = torch.empty((batch_size,), dtype=torch.long)
    query_output_widths = torch.empty((batch_size,), dtype=torch.long)
    height_targets = torch.empty((batch_size,), dtype=torch.long)
    width_targets = torch.empty((batch_size,), dtype=torch.long)
    query_targets = torch.full((batch_size, MAX_GRID_TOKENS), PAD_ID, dtype=torch.long)
    query_target_mask = torch.zeros((batch_size, MAX_GRID_TOKENS), dtype=torch.bool)
    support_evidence_colors = torch.full((batch_size, max_evidence_tokens), PAD_ID, dtype=torch.long)
    support_evidence_rows = torch.zeros((batch_size, max_evidence_tokens), dtype=torch.long)
    support_evidence_cols = torch.zeros((batch_size, max_evidence_tokens), dtype=torch.long)
    support_evidence_roles = torch.zeros((batch_size, max_evidence_tokens), dtype=torch.long)
    support_evidence_pair_indices = torch.zeros((batch_size, max_evidence_tokens), dtype=torch.long)
    support_evidence_mask = torch.zeros((batch_size, max_evidence_tokens), dtype=torch.bool)
    support_evidence_counts = torch.zeros((batch_size,), dtype=torch.long)

    for batch_idx, ep in enumerate(episodes):
        evidence_idx = 0
        for support_idx, pair in enumerate(ep.supports):
            if pair.pair_index >= MAX_SUPPORT_PAIRS:
                raise ValueError(
                    f"support pair index {pair.pair_index} exceeds MAX_SUPPORT_PAIRS={MAX_SUPPORT_PAIRS}"
                )
            support_inputs[batch_idx, support_idx] = pad_grid_to_tokens(pair.input)
            support_outputs[batch_idx, support_idx] = pad_grid_to_tokens(pair.output)
            support_mask[batch_idx, support_idx] = True
            for role, grid in ((ROLE_SUPPORT_INPUT, pair.input), (ROLE_SUPPORT_OUTPUT, pair.output)):
                for row_idx, row in enumerate(grid.cells):
                    for col_idx, color in enumerate(row):
                        support_evidence_colors[batch_idx, evidence_idx] = color
                        support_evidence_rows[batch_idx, evidence_idx] = row_idx
                        support_evidence_cols[batch_idx, evidence_idx] = col_idx
                        support_evidence_roles[batch_idx, evidence_idx] = role
                        support_evidence_pair_indices[batch_idx, evidence_idx] = pair.pair_index
                        support_evidence_mask[batch_idx, evidence_idx] = True
                        evidence_idx += 1
        support_evidence_counts[batch_idx] = evidence_idx
        query_inputs[batch_idx] = pad_grid_to_tokens(ep.query.input)
        query_input_mask[batch_idx] = grid_shape_mask(ep.query.input.height, ep.query.input.width)
        query_targets[batch_idx] = pad_grid_to_tokens(ep.query.output)
        query_target_mask[batch_idx] = grid_shape_mask(ep.query.output.height, ep.query.output.width)
        query_input_heights[batch_idx] = ep.query.input.height
        query_input_widths[batch_idx] = ep.query.input.width
        query_output_heights[batch_idx] = ep.query.output.height
        query_output_widths[batch_idx] = ep.query.output.width
        height_targets[batch_idx] = ep.query.output.height - 1
        width_targets[batch_idx] = ep.query.output.width - 1

    return {
        "support_inputs": support_inputs,
        "support_outputs": support_outputs,
        "support_mask": support_mask,
        "query_inputs": query_inputs,
        "query_input_mask": query_input_mask,
        "query_input_heights": query_input_heights,
        "query_input_widths": query_input_widths,
        "query_output_heights": query_output_heights,
        "query_output_widths": query_output_widths,
        "height_targets": height_targets,
        "width_targets": width_targets,
        "query_targets": query_targets,
        "query_target_mask": query_target_mask,
        "support_evidence_colors": support_evidence_colors,
        "support_evidence_rows": support_evidence_rows,
        "support_evidence_cols": support_evidence_cols,
        "support_evidence_roles": support_evidence_roles,
        "support_evidence_pair_indices": support_evidence_pair_indices,
        "support_evidence_mask": support_evidence_mask,
        "support_evidence_counts": support_evidence_counts,
        "task_ids": [ep.task_id for ep in episodes],
        "query_pair_indices": [ep.query_pair_index for ep in episodes],
        "source_splits": [ep.source_split for ep in episodes],
    }


def model_inputs_from_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "support_evidence_colors": batch["support_evidence_colors"],
        "support_evidence_rows": batch["support_evidence_rows"],
        "support_evidence_cols": batch["support_evidence_cols"],
        "support_evidence_roles": batch["support_evidence_roles"],
        "support_evidence_pair_indices": batch["support_evidence_pair_indices"],
        "support_evidence_mask": batch["support_evidence_mask"],
        "query_inputs": batch["query_inputs"],
        "query_input_mask": batch["query_input_mask"],
    }


def move_tensor_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def grid_to_text(grid: Grid) -> str:
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid.cells)


def shape_distribution(grids: list[Grid]) -> dict[str, int]:
    counts = Counter(f"{grid.height}x{grid.width}" for grid in grids)
    return dict(sorted(counts.items()))


def support_count_distribution(episodes: list[ArcEpisode]) -> dict[str, int]:
    counts = Counter(len(ep.supports) for ep in episodes)
    return {str(k): counts[k] for k in sorted(counts)}
