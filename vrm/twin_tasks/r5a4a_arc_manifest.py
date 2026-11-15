from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrm.data import (
    ARC_COLOR_COUNT,
    DEFAULT_ARC_DATA_ROOT,
    MAX_GRID_HEIGHT,
    MAX_GRID_WIDTH,
    MAX_SUPPORT_PAIRS,
    ArcEpisode,
    ArcTask,
    Grid,
    load_arc_training_tasks,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "r5a4a_neural_vrm_arc_grounded_preflight"
SENTINEL_SALT = "R5A4A_ARC_SENTINEL_V1"
VALIDATION_SALT = "R5A4A_ARC_VALIDATION_V1"
INITIALIZATION_SEED = 20260701
VALIDATION_TASK_COUNT = 64
SENTINEL_TASK_COUNT = 48


@dataclass(frozen=True)
class R5A4ASplits:
    tasks_by_id: dict[str, ArcTask]
    sentinel_task_ids: list[str]
    validation_task_ids: list[str]
    training_task_ids: list[str]
    source_manifest: dict[str, Any]
    sentinel_manifest: dict[str, Any]
    validation_manifest: dict[str, Any]
    training_manifest: dict[str, Any]


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def deterministic_key(salt: str, task_id: str) -> str:
    return hashlib.sha256(f"{salt}:{task_id}".encode("utf-8")).hexdigest()


def task_summary(task: ArcTask, *, salt: str | None = None) -> dict[str, Any]:
    max_height = max(max(pair.input.height, pair.output.height) for pair in task.train)
    max_width = max(max(pair.input.width, pair.output.width) for pair in task.train)
    colors = sorted(
        {
            cell
            for pair in task.train
            for grid in (pair.input, pair.output)
            for row in grid.cells
            for cell in row
        }
    )
    row = {
        "task_id": task.task_id,
        "train_pair_count": len(task.train),
        "loo_fold_count": len(task.train),
        "max_grid_height": max_height,
        "max_grid_width": max_width,
        "colors": colors,
        "source_file": task.source_file,
    }
    if salt is not None:
        row["sha256_order_key"] = deterministic_key(salt, task.task_id)
    return row


def eligible_task_ids(tasks: list[ArcTask]) -> tuple[list[str], dict[str, str]]:
    eligible: list[str] = []
    skipped: dict[str, str] = {}
    for task in tasks:
        if len(task.train) < 2:
            skipped[task.task_id] = "fewer_than_two_train_pairs"
            continue
        if len(task.train) - 1 > MAX_SUPPORT_PAIRS:
            skipped[task.task_id] = "too_many_support_pairs_for_model"
            continue
        bad_reason = None
        for pair in task.train:
            for grid in (pair.input, pair.output):
                if grid.height > MAX_GRID_HEIGHT or grid.width > MAX_GRID_WIDTH:
                    bad_reason = "grid_exceeds_model_limits"
                if any(cell < 0 or cell >= ARC_COLOR_COUNT for row in grid.cells for cell in row):
                    bad_reason = "color_outside_model_palette"
        if bad_reason:
            skipped[task.task_id] = bad_reason
            continue
        eligible.append(task.task_id)
    return sorted(eligible), dict(sorted(skipped.items()))


def make_pool_manifest(
    *,
    pool_name: str,
    task_ids: list[str],
    tasks_by_id: dict[str, ArcTask],
    selection_salt: str | None,
    data_root: Path,
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pool_name": pool_name,
        "data_root": str(data_root),
        "training_challenges_path": sources["training_challenges"],
        "training_solutions_path_discovered_but_not_used": sources["training_solutions"],
        "selection_salt": selection_salt,
        "task_count": len(task_ids),
        "loo_fold_count": sum(len(tasks_by_id[task_id].train) for task_id in task_ids),
        "task_ids": task_ids,
        "tasks": [task_summary(tasks_by_id[task_id], salt=selection_salt) for task_id in task_ids],
    }


def build_r5a4a_splits(data_root: Path = DEFAULT_ARC_DATA_ROOT) -> R5A4ASplits:
    tasks, sources = load_arc_training_tasks(data_root)
    tasks_by_id = {task.task_id: task for task in tasks}
    eligible_ids, skipped = eligible_task_ids(tasks)
    if len(eligible_ids) < SENTINEL_TASK_COUNT + VALIDATION_TASK_COUNT:
        raise RuntimeError(f"not enough eligible ARC training tasks for R5A-4A ({len(eligible_ids)} eligible)")
    sentinel_task_ids = sorted(eligible_ids, key=lambda task_id: deterministic_key(SENTINEL_SALT, task_id))[
        :SENTINEL_TASK_COUNT
    ]
    remaining = [task_id for task_id in eligible_ids if task_id not in set(sentinel_task_ids)]
    validation_task_ids = sorted(remaining, key=lambda task_id: deterministic_key(VALIDATION_SALT, task_id))[
        :VALIDATION_TASK_COUNT
    ]
    reserved = set(sentinel_task_ids) | set(validation_task_ids)
    training_task_ids = sorted(task_id for task_id in eligible_ids if task_id not in reserved)
    source_manifest = {
        "manifest_name": "r5a4a_arc_training_source_manifest",
        "data_root": str(data_root),
        "training_challenges_path": sources["training_challenges"],
        "training_challenges_sha256": file_sha256(Path(sources["training_challenges"])),
        "training_solutions_path_discovered_but_not_used": sources["training_solutions"],
        "task_count_total": len(tasks),
        "eligible_task_count": len(eligible_ids),
        "skipped_task_count": len(skipped),
        "skipped_tasks": skipped,
        "eligible_task_ids": eligible_ids,
        "task_summaries": [task_summary(tasks_by_id[task_id]) for task_id in eligible_ids],
        "raw_arc_policy": "Only arc-agi_training_challenges.json train outputs are used as supervised targets.",
    }
    return R5A4ASplits(
        tasks_by_id=tasks_by_id,
        sentinel_task_ids=sentinel_task_ids,
        validation_task_ids=validation_task_ids,
        training_task_ids=training_task_ids,
        source_manifest=source_manifest,
        sentinel_manifest=make_pool_manifest(
            pool_name="sentinel",
            task_ids=sentinel_task_ids,
            tasks_by_id=tasks_by_id,
            selection_salt=SENTINEL_SALT,
            data_root=Path(data_root),
            sources=sources,
        ),
        validation_manifest=make_pool_manifest(
            pool_name="validation",
            task_ids=validation_task_ids,
            tasks_by_id=tasks_by_id,
            selection_salt=VALIDATION_SALT,
            data_root=Path(data_root),
            sources=sources,
        ),
        training_manifest=make_pool_manifest(
            pool_name="training",
            task_ids=training_task_ids,
            tasks_by_id=tasks_by_id,
            selection_salt=None,
            data_root=Path(data_root),
            sources=sources,
        ),
    )


def episodes_for_task_ids(
    tasks_by_id: dict[str, ArcTask],
    task_ids: list[str],
    *,
    source_split: str,
) -> list[ArcEpisode]:
    episodes: list[ArcEpisode] = []
    for task_id in task_ids:
        task = tasks_by_id[task_id]
        if len(task.train) < 2:
            continue
        for query_idx, query_pair in enumerate(task.train):
            supports = tuple(pair for idx, pair in enumerate(task.train) if idx != query_idx)
            if supports:
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
    return episodes


def write_split_artifacts(splits: R5A4ASplits, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "arc_source_task_manifest": artifact_dir / "arc_source_task_manifest.json",
        "arc_sentinel_manifest": artifact_dir / "arc_sentinel_manifest.json",
        "arc_validation_pool_manifest": artifact_dir / "arc_validation_pool_manifest.json",
        "arc_training_pool_manifest": artifact_dir / "arc_training_pool_manifest.json",
        "arc_sentinel_manifest_sha256": artifact_dir / "arc_sentinel_manifest_sha256.txt",
    }
    write_json(paths["arc_source_task_manifest"], splits.source_manifest)
    write_json(paths["arc_sentinel_manifest"], splits.sentinel_manifest)
    write_json(paths["arc_validation_pool_manifest"], splits.validation_manifest)
    write_json(paths["arc_training_pool_manifest"], splits.training_manifest)
    write_text(paths["arc_sentinel_manifest_sha256"], file_sha256(paths["arc_sentinel_manifest"]) + "\n")
    return {name: str(path) for name, path in paths.items()}


def source_file_manifest(
    *,
    project_root: Path = PROJECT_ROOT,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    globs = [
        "vrm/model.py",
        "vrm/data.py",
        "vrm/metrics.py",
        "vrm/twin_tasks/r5a4a_*.py",
        "vrm/twin_tasks/run_r5a4a_*.py",
        "tests/test_r5a4a_*.py",
        "docs/R5A4A_*.md",
        "vrm/twin_tasks/r5a2*.py",
        "vrm/twin_tasks/r5a3*.py",
        "tests/test_r5a2*.py",
        "tests/test_r5a3*.py",
    ]
    if include_artifacts:
        globs.extend(["artifacts/r5a2*/*", "artifacts/r5a3*/*"])
    files: list[Path] = []
    for pattern in globs:
        files.extend(path for path in project_root.glob(pattern) if path.is_file())
    rows = []
    for path in sorted(set(files), key=lambda item: item.relative_to(project_root).as_posix()):
        if "__pycache__" in path.parts:
            continue
        rows.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {"project_root": str(project_root), "file_count": len(rows), "files": rows}


def render_source_manifest_text(manifest: dict[str, Any]) -> str:
    lines = [f"project_root: {manifest['project_root']}", f"file_count: {manifest['file_count']}", ""]
    for row in manifest["files"]:
        lines.append(f"{row['sha256']}  {row['size_bytes']:>10}  {row['path']}")
    return "\n".join(lines).rstrip() + "\n"


def manifest_hashes_for_commitment(splits: R5A4ASplits) -> dict[str, str]:
    return {
        "full_arc_source_task_manifest_sha256": sha256_json(splits.source_manifest),
        "sentinel_manifest_sha256": sha256_json(splits.sentinel_manifest),
        "training_pool_manifest_sha256": sha256_json(splits.training_manifest),
        "validation_pool_manifest_sha256": sha256_json(splits.validation_manifest),
        "model_py_sha256": file_sha256(PROJECT_ROOT / "vrm" / "model.py"),
        "data_py_sha256": file_sha256(PROJECT_ROOT / "vrm" / "data.py"),
        "metrics_py_sha256": file_sha256(PROJECT_ROOT / "vrm" / "metrics.py"),
    }


def assert_disjoint_splits(splits: R5A4ASplits) -> None:
    sentinel = set(splits.sentinel_task_ids)
    validation = set(splits.validation_task_ids)
    training = set(splits.training_task_ids)
    if sentinel & validation or sentinel & training or validation & training:
        raise RuntimeError("R5A-4A task pools are not disjoint")


def split_counts(splits: R5A4ASplits) -> dict[str, int]:
    return {
        "sentinel_task_count": len(splits.sentinel_task_ids),
        "sentinel_loo_fold_count": splits.sentinel_manifest["loo_fold_count"],
        "validation_task_count": len(splits.validation_task_ids),
        "validation_loo_fold_count": splits.validation_manifest["loo_fold_count"],
        "training_task_count": len(splits.training_task_ids),
        "training_loo_fold_count": splits.training_manifest["loo_fold_count"],
    }
