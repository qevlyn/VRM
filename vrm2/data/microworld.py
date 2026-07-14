from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from vrm2.data.composition_split import (
    SPLIT_VERSION,
    VALID_SPLITS,
    build_split_manifest,
    parse_signature,
    stable_seed,
)
from vrm2.data.operations import (
    DIRECTIONS,
    OPERATION_REGISTRY_VERSION,
    SceneValidationError,
    execute_program,
    find_objects,
)
from vrm2.data.program_schema import factorize_instruction
from vrm2.data.serialization import (
    canonical_json_bytes,
    generator_episode_record,
    grid_color_histogram,
    sha256_hex,
    write_canonical_json,
    write_model_dataset,
)
from vrm2.data.types import EpisodeDiagnostics, Grid, GridPair, HiddenProgram, OperationStep, Split, TaskEpisode, make_grid

GENERATOR_VERSION = "vrm2-microworld-v1"

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
)


class GenerationLimitError(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Stage-0 config must use the dependency-free JSON subset of YAML 1.2"
        ) from exc
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required_versions = {
        "generator_version": GENERATOR_VERSION,
        "operation_registry_version": OPERATION_REGISTRY_VERSION,
        "split_version": SPLIT_VERSION,
    }
    for key, expected in required_versions.items():
        if config.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if config.get("grid_size") != 8 or config.get("output_shape") != "fixed":
        raise ValueError("Stage 0 is frozen to fixed 8x8 grids")
    if config.get("support_pairs_per_task") != 3 or config.get("query_pairs_per_task") != 1:
        raise ValueError("Stage 0 is frozen to three supports and one query")
    if config.get("objects_per_scene") != [1, 3]:
        raise ValueError("Stage 0 object range must be [1, 3]")
    if config.get("program_lengths") != {"train": [1, 2], "dev": [2], "sealed": [2, 3]}:
        raise ValueError("Stage-0 composition lengths differ from the frozen contract")
    for key in ("model_sees_rule_metadata", "variable_shape", "natural_language_verbalizer", "arc_native_data"):
        if config.get(key) is not False:
            raise ValueError(f"{key} must remain false in Stage 0")
    if config.get("deterministic_mode") is not True:
        raise ValueError("deterministic_mode must be true")
    for section in ("task_counts", "scientific_task_counts", "seeds", "output_paths", "validation_limits"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"missing mapping: {section}")
    for key in ("max_scene_attempts", "max_episode_attempts", "max_object_placement_attempts"):
        value = config["validation_limits"].get(key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"validation limit {key} must be a positive integer")


def _cardinal_neighbors(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, col = cell
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def sample_scene(
    rng: random.Random,
    grid_size: int,
    object_range: Sequence[int],
    max_object_placement_attempts: int,
) -> Grid:
    minimum, maximum = int(object_range[0]), int(object_range[1])
    object_count = rng.randint(minimum, maximum)
    colors = rng.sample(range(1, 10), object_count)
    occupied: set[tuple[int, int]] = set()
    placed: list[tuple[frozenset[tuple[int, int]], int]] = []

    for color in colors:
        for _ in range(max_object_placement_attempts):
            shape = rng.choice(SHAPES)
            height = max(row for row, _ in shape) + 1
            width = max(col for _, col in shape) + 1
            origin_row = rng.randrange(grid_size - height + 1)
            origin_col = rng.randrange(grid_size - width + 1)
            cells = {(origin_row + row, origin_col + col) for row, col in shape}
            if cells & occupied:
                continue
            if any(neighbor in occupied for cell in cells for neighbor in _cardinal_neighbors(cell)):
                continue
            frozen_cells = frozenset(cells)
            placed.append((frozen_cells, color))
            occupied.update(cells)
            break
        else:
            raise SceneValidationError("object_placement_exhausted")

    rows = [[0] * grid_size for _ in range(grid_size)]
    for cells, color in placed:
        for row, col in cells:
            rows[row][col] = color
    grid = make_grid(rows, expected_size=grid_size)
    if len(find_objects(grid)) != object_count:
        raise SceneValidationError("object_count_mismatch")
    return grid


def sample_hidden_program(
    composition_signature: str,
    episode_seed: int,
    *,
    generator_version: str = GENERATOR_VERSION,
    split_version: str = SPLIT_VERSION,
) -> HiddenProgram:
    names = parse_signature(composition_signature)
    rng = random.Random(
        stable_seed(
            generator_version,
            split_version,
            OPERATION_REGISTRY_VERSION,
            composition_signature,
            episode_seed,
            "operation-parameters",
        )
    )
    recolor_targets = list(range(1, 10))
    rng.shuffle(recolor_targets)
    steps: list[OperationStep] = []
    for name in names:
        if name.startswith("translate_"):
            parameters: dict[str, object] = {"direction": rng.choice(tuple(DIRECTIONS))}
        elif name.startswith("recolor_"):
            parameters = {"color": recolor_targets.pop()}
        else:
            parameters = {}
        steps.append(
            OperationStep(
                name=name,
                parameters=parameters,
                typed_instruction=factorize_instruction(name, parameters),
            )
        )
    return HiddenProgram(composition_signature=composition_signature, steps=tuple(steps))


def _generate_pair(
    program: HiddenProgram,
    scene_seed: int,
    config: Mapping[str, Any],
    rejection_counts: collections.Counter[str],
) -> tuple[GridPair, int]:
    rng = random.Random(scene_seed)
    limits = config["validation_limits"]
    for scene_attempt in range(1, int(limits["max_scene_attempts"]) + 1):
        try:
            input_grid = sample_scene(
                rng,
                int(config["grid_size"]),
                config["objects_per_scene"],
                int(limits["max_object_placement_attempts"]),
            )
            output_grid = execute_program(input_grid, program.steps)
            return GridPair(input_grid=input_grid, output_grid=output_grid), scene_attempt
        except SceneValidationError as exc:
            rejection_counts[exc.reason] += 1
    raise GenerationLimitError(
        f"scene generation exceeded {limits['max_scene_attempts']} attempts for {program.composition_signature}"
    )


def _reject_obvious_episode_ambiguity(support: Sequence[GridPair], query: GridPair, program: HiddenProgram) -> None:
    support_inputs = {pair.input_grid for pair in support}
    support_outputs = {pair.output_grid for pair in support}
    if len(support_inputs) != len(support):
        raise SceneValidationError("duplicate_support_input")
    if len(support_outputs) != len(support):
        raise SceneValidationError("duplicate_support_output")
    if query.input_grid in support_inputs:
        raise SceneValidationError("query_duplicates_support_input")
    if query.output_grid in support_outputs:
        raise SceneValidationError("query_duplicates_support_output")

    if len(program.steps) > 1:
        for step in program.steps:
            try:
                trivial_outputs = tuple(execute_program(pair.input_grid, (step,)) for pair in support)
            except SceneValidationError:
                continue
            if trivial_outputs == tuple(pair.output_grid for pair in support):
                raise SceneValidationError("single_step_explains_support")


def validate_episode(episode: TaskEpisode, config: Mapping[str, Any]) -> None:
    size = int(config["grid_size"])
    if len(episode.support) != int(config["support_pairs_per_task"]):
        raise ValueError("episode support count does not match config")
    grids = [
        *(pair.input_grid for pair in episode.support),
        *(pair.output_grid for pair in episode.support),
        episode.query_input,
        episode.query_output,
    ]
    for grid in grids:
        make_grid(grid, expected_size=size)
    for pair in episode.support:
        if execute_program(pair.input_grid, episode.hidden_program.steps) != pair.output_grid:
            raise ValueError("support output does not replay exactly")
    if execute_program(episode.query_input, episode.hidden_program.steps) != episode.query_output:
        raise ValueError("query output does not replay exactly")


def generate_episode_with_diagnostics(
    composition_signature: str,
    episode_seed: int,
    split: Split,
    config: Mapping[str, Any],
) -> tuple[TaskEpisode, EpisodeDiagnostics]:
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split: {split}")
    program = sample_hidden_program(
        composition_signature,
        episode_seed,
        generator_version=str(config["generator_version"]),
        split_version=str(config["split_version"]),
    )
    rejection_counts: collections.Counter[str] = collections.Counter()
    max_episode_attempts = int(config["validation_limits"]["max_episode_attempts"])

    for episode_attempt in range(1, max_episode_attempts + 1):
        pairs: list[GridPair] = []
        scene_attempts: list[int] = []
        try:
            for pair_index in range(int(config["support_pairs_per_task"]) + 1):
                scene_seed = stable_seed(
                    config["generator_version"],
                    config["split_version"],
                    composition_signature,
                    episode_seed,
                    "episode-attempt",
                    episode_attempt,
                    "scene",
                    pair_index,
                )
                pair, attempts = _generate_pair(program, scene_seed, config, rejection_counts)
                pairs.append(pair)
                scene_attempts.append(attempts)
            support = tuple(pairs[:-1])
            query = pairs[-1]
            _reject_obvious_episode_ambiguity(support, query, program)
            episode = TaskEpisode(
                support=support,
                query_input=query.input_grid,
                query_output=query.output_grid,
                hidden_program=program,
                episode_seed=episode_seed,
                split=split,
            )
            validate_episode(episode, config)
            diagnostics = EpisodeDiagnostics(
                episode_attempt=episode_attempt,
                scene_attempts=tuple(scene_attempts),
                rejection_counts=dict(sorted(rejection_counts.items())),
            )
            return episode, diagnostics
        except SceneValidationError as exc:
            rejection_counts[exc.reason] += 1
        except GenerationLimitError:
            rejection_counts["scene_retry_limit"] += 1
    raise GenerationLimitError(
        f"episode generation exceeded {max_episode_attempts} attempts for seed={episode_seed}, "
        f"signature={composition_signature}; rejections={dict(rejection_counts)}"
    )


def generate_episode(
    composition_signature: str,
    episode_seed: int,
    split: Split,
    config: Mapping[str, Any],
) -> TaskEpisode:
    return generate_episode_with_diagnostics(composition_signature, episode_seed, split, config)[0]


def schedule_compositions(
    split_manifest: Mapping[str, Any],
    split: Split,
    allowed_lengths: Sequence[int],
    num_tasks: int,
    dataset_seed: int,
) -> tuple[str, ...]:
    signatures = split_manifest["splits"][split]["signatures"]
    by_length = {
        length: [signature for signature in signatures if len(parse_signature(signature)) == length]
        for length in allowed_lengths
    }
    if any(not values for values in by_length.values()):
        raise ValueError(f"split {split} has no signatures for one of lengths {allowed_lengths}")
    for length, values in by_length.items():
        random.Random(stable_seed(split_manifest["split_version"], dataset_seed, split, length)).shuffle(values)

    scheduled: list[str] = []
    offsets = {length: 0 for length in allowed_lengths}
    while len(scheduled) < num_tasks:
        for length in allowed_lengths:
            values = by_length[length]
            scheduled.append(values[offsets[length] % len(values)])
            offsets[length] += 1
            if len(scheduled) == num_tasks:
                break
    return tuple(scheduled)


def generate_dataset(
    config: Mapping[str, Any],
    split: Split,
    num_tasks: int,
    dataset_seed: int,
    split_manifest: Mapping[str, Any] | None = None,
) -> tuple[tuple[TaskEpisode, ...], tuple[EpisodeDiagnostics, ...], tuple[str, ...]]:
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    manifest = split_manifest or build_split_manifest(
        int(config["seeds"]["composition_split"]), str(config["split_version"])
    )
    signatures = schedule_compositions(
        manifest,
        split,
        config["program_lengths"][split],
        num_tasks,
        dataset_seed,
    )
    episodes: list[TaskEpisode] = []
    diagnostics: list[EpisodeDiagnostics] = []
    for index, signature in enumerate(signatures):
        episode_seed = dataset_seed + index
        episode, episode_diagnostics = generate_episode_with_diagnostics(
            signature, episode_seed, split, config
        )
        episodes.append(episode)
        diagnostics.append(episode_diagnostics)
    return tuple(episodes), tuple(diagnostics), signatures


def dataset_summary(
    episodes: Sequence[TaskEpisode],
    diagnostics: Sequence[EpisodeDiagnostics],
    dataset_bytes: int,
    dataset_sha256: str,
) -> dict[str, object]:
    length_counts = collections.Counter(len(episode.hidden_program.steps) for episode in episodes)
    rejection_counts: collections.Counter[str] = collections.Counter()
    for item in diagnostics:
        rejection_counts.update(item.rejection_counts)
    grids = []
    for episode in episodes:
        for pair in episode.support:
            grids.extend((pair.input_grid, pair.output_grid))
        grids.extend((episode.query_input, episode.query_output))
    return {
        "split": episodes[0].split if episodes else None,
        "num_tasks": len(episodes),
        "unique_composition_count": len({episode.hidden_program.composition_signature for episode in episodes}),
        "program_length_counts": {str(key): length_counts[key] for key in sorted(length_counts)},
        "episode_attempts_total": sum(item.episode_attempt for item in diagnostics),
        "scene_attempts_total": sum(sum(item.scene_attempts) for item in diagnostics),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "grid_color_histogram": grid_color_histogram(grids),
        "dataset_bytes": dataset_bytes,
        "dataset_sha256": dataset_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic VRM v2 Stage-0 microworld tasks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=VALID_SPLITS, required=True)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--split-manifest-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    split: Split = args.split
    num_tasks = args.num_tasks if args.num_tasks is not None else int(config["task_counts"][split])
    dataset_seed = args.seed if args.seed is not None else int(config["seeds"][split])
    output = args.output or Path(config["output_paths"]["dataset_dir"]) / f"{split}.jsonl"
    manifest_output = args.manifest_output or output.with_name(f"{output.stem}.generator-manifest.json")
    split_manifest_output = args.split_manifest_output or output.with_name(f"{output.stem}.split-manifest.json")

    split_manifest = build_split_manifest(
        int(config["seeds"]["composition_split"]), str(config["split_version"])
    )
    episodes, diagnostics, signatures = generate_dataset(
        config, split, num_tasks, dataset_seed, split_manifest
    )
    dataset_size, dataset_digest = write_model_dataset(output, episodes)
    summary = dataset_summary(episodes, diagnostics, dataset_size, dataset_digest)
    generator_manifest = {
        "generator_version": config["generator_version"],
        "operation_registry_version": config["operation_registry_version"],
        "split_version": config["split_version"],
        "split_manifest_sha256": sha256_hex(canonical_json_bytes(split_manifest)),
        "split": split,
        "dataset_seed": dataset_seed,
        "num_tasks": num_tasks,
        "composition_schedule": list(signatures),
        "episodes": [
            {
                "index": index,
                **generator_episode_record(episode),
                "diagnostics": {
                    "episode_attempt": item.episode_attempt,
                    "scene_attempts": list(item.scene_attempts),
                    "rejection_counts": dict(item.rejection_counts),
                },
            }
            for index, (episode, item) in enumerate(zip(episodes, diagnostics, strict=True))
        ],
        "summary": summary,
    }
    write_canonical_json(manifest_output, generator_manifest)
    write_canonical_json(split_manifest_output, split_manifest)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
