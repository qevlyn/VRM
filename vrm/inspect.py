from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .data import (
    DEFAULT_ARC_DATA_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    build_dataset,
    grid_to_text,
    shape_distribution,
    support_count_distribution,
)

DEFAULT_OUTPUT = Path("/mnt/d/VRM/artifacts/episode_inspection.txt")


def _append_grid(lines: list[str], title: str, grid) -> None:
    lines.append(f"{title} shape={grid.height}x{grid.width}")
    lines.append(grid_to_text(grid))


def render_inspection(data: dict, limit: int = 20) -> str:
    tasks = data["tasks"]
    episodes = data["episodes"]
    train_episodes = data["train_episodes"]
    validation_episodes = data["validation_episodes"]
    skipped = data["skipped"]
    split_assignments = data["split_assignments"]
    usable_tasks = [task for task in tasks if task.task_id in split_assignments]
    all_inputs = [pair.input for task in usable_tasks for pair in task.train]
    all_outputs = [pair.output for task in usable_tasks for pair in task.train]

    lines: list[str] = []
    lines.append("VRM ARC episode inspection")
    lines.append(f"training challenge JSON: {data['sources']['training_challenges']}")
    lines.append(
        "training solution JSON discovered but not used for episodes: "
        f"{data['sources']['training_solutions']}"
    )
    lines.append("")
    lines.append("Summary")
    lines.append(f"number of raw tasks: {len(tasks)}")
    lines.append(f"number of usable tasks: {len(usable_tasks)}")
    lines.append(f"tasks skipped for too few train pairs: {len(skipped)}")
    lines.append(f"train task count: {sum(1 for split in split_assignments.values() if split == 'train')}")
    lines.append(
        f"validation task count: {sum(1 for split in split_assignments.values() if split == 'validation')}"
    )
    lines.append(f"train episode count: {len(train_episodes)}")
    lines.append(f"validation episode count: {len(validation_episodes)}")
    lines.append(f"support-pair count distribution: {support_count_distribution(episodes)}")
    lines.append(f"input-shape distribution: {shape_distribution(all_inputs)}")
    lines.append(f"output-shape distribution: {shape_distribution(all_outputs)}")
    skipped_reasons = Counter(skipped.values())
    lines.append(f"skipped reason distribution: {dict(sorted(skipped_reasons.items()))}")
    lines.append("")

    for index, episode in enumerate(episodes[:limit], start=1):
        lines.append("=" * 72)
        lines.append(f"episode {index}")
        lines.append(f"task ID: {episode.task_id}")
        lines.append(f"source split: {episode.source_split}")
        lines.append(f"query pair number: {episode.query_pair_index}")
        lines.append(
            "original query input/output shapes: "
            f"{episode.query.input.height}x{episode.query.input.width} -> "
            f"{episode.query.output.height}x{episode.query.output.width}"
        )
        for support_number, support in enumerate(episode.supports, start=1):
            lines.append("")
            lines.append(f"support pair number: {support.pair_index} (shown as support {support_number})")
            lines.append(
                "original support input/output shapes: "
                f"{support.input.height}x{support.input.width} -> "
                f"{support.output.height}x{support.output.width}"
            )
            _append_grid(lines, "support input grid", support.input)
            _append_grid(lines, "support output grid", support.output)
        lines.append("")
        _append_grid(lines, "query input grid", episode.query.input)
        _append_grid(lines, "hidden query target grid", episode.query.output)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real ARC leave-one-out episodes for VRM.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ARC_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()

    data = build_dataset(args.data_root, seed=args.seed, manifest_path=args.manifest)
    text = render_inspection(data, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"saved inspection artifact: {args.output}")
    print(f"saved split manifest: {args.manifest}")


if __name__ == "__main__":
    main()
