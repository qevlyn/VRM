from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrm.model import VRMConfig

from .run_pilot import protected_hashes
from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_PAIR_ORDER_SHUFFLED,
    CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    collate_twin_episodes,
    data_manifest,
    load_twin_episodes,
    model_inputs_for_twin_batch,
    pair_order_shuffled_view,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
    training_view,
    twin_swap_supports_view,
)
from .vrm_probe_metrics import (
    condition_deltas,
    evaluate_all_conditions,
    evaluate_condition,
    pair_level_metrics,
    save_per_episode,
    twin_pair_joint_normal_accuracy,
)
from .vrm_probe_train import (
    compute_lr,
    configure_determinism,
    dev_selection_key,
    finite_parameters,
    initialize_fresh_model,
    is_better_dev,
    load_state_dict_copy,
    resolve_device,
    save_checkpoint,
    state_dict_copy,
    train_step,
)

DEFAULT_DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
DEFAULT_OUT = Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0")
DEFAULT_CHECKPOINT_DIR = Path("/mnt/d/VRM/checkpoints/twin_vrm_probe_v0")
DEFAULT_SEED = 20260628
ARCHITECTURE_NAME = "VRM V1.1 - Masked Query Spatial Evidence Rewrite"
PROBE_NAME = "VRM Twin Probe v0 - Correspondence Capacity Test"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_own_probe_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if "twin_vrm_probe" not in path.name:
        raise ValueError(f"refusing to resume from non-probe checkpoint: {path}")
    return torch.load(path, map_location=device)


def checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"twin_vrm_probe_step{step:04d}.pt"


def best_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "twin_vrm_probe_best_dev_candidate.pt"


def runtime_report(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
    }


def make_config(args: argparse.Namespace) -> VRMConfig:
    return VRMConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        pair_encoder_layers=args.pair_encoder_layers,
        rewrite_layers=args.rewrite_layers,
        rewrite_steps=args.rewrite_steps,
        rule_slots=args.rule_slots,
        dropout=args.dropout,
    )


def write_p6_provenance(out: Path) -> None:
    text = """# P6 Provenance

Classification:

```text
P6_IS_SEPARATELY_TRAINED_ON_DERANGED_SUPPORTS
```

Evidence:

- `vrm/twin_tasks/run_pilot.py` builds `all_runs` by looping over every `probe` in `PROBE_NAMES` and every seed, then calls `train_probe(probe_name=probe, ...)`. This means P6 has its own training runs rather than reusing the trained P5 probe under an altered evaluation view.
- `vrm/twin_tasks/probes.py` routes `P6_RELATION_AWARE_DERANGED` through `feature_scores(..., "P6_RELATION_AWARE_DERANGED", seed)`, which calls `relation_scores(record, deranged=True, seed=seed)`.
- `deranged_supports(record, seed)` deterministically permutes support outputs within each episode while preserving support inputs and output bags. The resulting P6 features are materialized inside `tensors_for_probe(...)` for train, dev, and sealed splits.
- `artifacts/twin_task_pilot_v1/probe_config.json` lists `P6_RELATION_AWARE_DERANGED` as one of the trained `probe_names`.
- `artifacts/twin_task_pilot_v1/probe_train_logs.jsonl` contains rows with `"probe": "P6_RELATION_AWARE_DERANGED"` and per-epoch train/dev accuracies, confirming separate optimization records for P6.
- `artifacts/twin_task_pilot_v1/probe_sealed_metrics.json` reports P6 sealed accuracies near 25% across seeds, generated from those separate P6 runs.

The below-chance P6 result is not a blocker for the VRM Twin Probe. Because the P6 view deterministically deranges the coherent support correspondence while preserving the same query and support bags, a simple relation scorer can become anti-correlated: matching the query to its deranged support output may systematically prefer the alternate candidate instead of producing a random tie. Thus below-chance accuracy can arise from deterministic anti-mapping in the deranged control, not from P6 having access to forbidden labels or candidates during the VRM probe.
"""
    save_text(out / "p6_provenance.md", text)


def run_pytest(out: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd="/mnt/d/VRM",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError("pytest failed:\n" + result.stdout)
    return result.stdout.strip()


def write_common_reports(
    *,
    args: argparse.Namespace,
    out: Path,
    episodes_by_split: dict[str, list[TwinEpisode]],
    initialization_report: dict[str, Any],
    config: VRMConfig,
    device: torch.device,
    determinism: dict[str, Any],
) -> None:
    save_json(
        out / "probe_config.json",
        {
            "probe": PROBE_NAME,
            "architecture": ARCHITECTURE_NAME,
            "seed": args.seed,
            "mode": args.mode,
            "max_steps": args.max_steps,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "betas": [args.beta1, args.beta2],
            "grad_clip_norm": args.grad_clip_norm,
            "shape_loss_weight": args.shape_loss_weight,
            "data": str(args.data),
            "out": str(out),
            "checkpoint_dir": str(args.checkpoint_dir),
            "resume_from": str(args.resume_from) if args.resume_from is not None else None,
            "model_config": asdict(config),
            "determinism": determinism,
            "runtime": runtime_report(device),
            "sealed_protocol": "sealed is evaluated once after the best dev checkpoint is selected",
        },
    )
    save_json(out / "data_manifest.json", data_manifest(args.data, episodes_by_split))
    save_json(out / "initialization_report.json", initialization_report)
    write_p6_provenance(out)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    determinism = configure_determinism(args.seed)
    device = resolve_device(args.device)
    episodes_by_split = load_twin_episodes(args.data)
    config = make_config(args)
    model, initialization_report = initialize_fresh_model(config, device, args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    write_common_reports(
        args=args,
        out=out,
        episodes_by_split=episodes_by_split,
        initialization_report=initialization_report,
        config=config,
        device=device,
        determinism=determinism,
    )
    sample_episode = episodes_by_split["train"][0]
    batch = collate_twin_episodes([sample_episode])
    model_input_keys = sorted(model_inputs_for_twin_batch(batch).keys())
    view_a = training_view(sample_episode, seed=args.seed, step=1)
    view_b = training_view(sample_episode, seed=args.seed, step=2)
    shuffled = pair_order_shuffled_view(sample_episode, seed=args.seed)
    opposite = build_opposite_lookup(episodes_by_split["train"])[f"{sample_episode.base_id}:{sample_episode.twin}"]
    swapped = twin_swap_supports_view(sample_episode, opposite)
    losses = []
    train_rows = []
    for step in range(1, args.steps + 1):
        lr = compute_lr(step, args.steps, args.learning_rate, args.warmup_steps, args.min_learning_rate)
        row = train_step(
            model=model,
            optimizer=optimizer,
            train_episodes=episodes_by_split["train"],
            step=step,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            shape_loss_weight=args.shape_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            learning_rate=lr,
        )
        losses.append(row["train_loss"])
        train_rows.append(row)
    after = protected_hashes()
    report = {
        "mode": "dry_run",
        "fresh_initialization": initialization_report["fresh_random_initialization"],
        "loaded_checkpoint": initialization_report["loaded_checkpoint"],
        "steps": args.steps,
        "finite_losses": all(math_is_finite(value) for value in losses),
        "finite_gradients": all(row["finite_gradients"] for row in train_rows),
        "finite_parameters": finite_parameters(model),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "candidate_metadata_excluded_from_model_input": "candidates" not in model_input_keys and "label" not in model_input_keys,
        "target_excluded_from_model_input": "query_targets" not in model_input_keys,
        "model_input_keys": model_input_keys,
        "support_pair_shuffling_works": support_signature(view_a) != support_signature(view_b),
        "training_pair_shuffling_preserves_input_bag": support_input_bag_signature(view_a) == support_input_bag_signature(sample_episode),
        "training_pair_shuffling_preserves_output_bag": support_output_bag_signature(view_a) == support_output_bag_signature(sample_episode),
        "pair_order_shuffled_preserves_coherent_pairs": sorted(support_signature(shuffled)) == sorted(support_signature(sample_episode)),
        "twin_swap_preserves_query": swapped.query_input == sample_episode.query_input,
        "twin_swap_preserves_candidates": swapped.candidates == sample_episode.candidates,
        "twin_swap_preserves_input_bag": support_input_bag_signature(swapped) == support_input_bag_signature(sample_episode),
        "twin_swap_preserves_output_bag": support_output_bag_signature(swapped) == support_output_bag_signature(sample_episode),
        "twin_swap_changes_paired_correspondence": support_signature(swapped) != support_signature(sample_episode),
        "protected_hashes_unchanged": before == after,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "runtime": runtime_report(device),
        "train_rows": train_rows,
    }
    report["all_passed"] = all(
        bool(report[key])
        for key in (
            "fresh_initialization",
            "finite_losses",
            "finite_gradients",
            "finite_parameters",
            "candidate_metadata_excluded_from_model_input",
            "target_excluded_from_model_input",
            "training_pair_shuffling_preserves_input_bag",
            "training_pair_shuffling_preserves_output_bag",
            "pair_order_shuffled_preserves_coherent_pairs",
            "twin_swap_preserves_query",
            "twin_swap_preserves_candidates",
            "twin_swap_preserves_input_bag",
            "twin_swap_preserves_output_bag",
            "twin_swap_changes_paired_correspondence",
            "protected_hashes_unchanged",
        )
    )
    save_json(out / "dry_run_report.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def math_is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def save_step_checkpoint_if_due(
    *,
    args: argparse.Namespace,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: VRMConfig,
    step: int,
    dev_row: dict[str, Any] | None,
) -> None:
    if step == 0 or step % args.checkpoint_every == 0 or step == args.max_steps:
        save_checkpoint(
            checkpoint_path(args.checkpoint_dir, step),
            model=model,
            optimizer=optimizer,
            config=config,
            step=step,
            extra={"dev_normal_metrics": dev_row},
        )


def load_log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def truncate_log_to_step(path: Path, step: int) -> list[dict[str, Any]]:
    rows = [row for row in load_log_rows(path) if int(row.get("step", -1)) <= step]
    path.write_text(
        "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    determinism = configure_determinism(args.seed)
    device = resolve_device(args.device)
    episodes_by_split = load_twin_episodes(args.data)
    config = make_config(args)
    model, initialization_report = initialize_fresh_model(config, device, args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    write_common_reports(
        args=args,
        out=out,
        episodes_by_split=episodes_by_split,
        initialization_report=initialization_report,
        config=config,
        device=device,
        determinism=determinism,
    )
    pytest_result = run_pytest(out)
    log_path = out / "training_log.jsonl"
    resume_step = 0
    resumed_from: str | None = None
    loaded_own_checkpoint = False
    if args.resume_from is None and log_path.exists():
        log_path.unlink()
    elif args.resume_from is not None:
        payload = load_own_probe_checkpoint(args.resume_from, device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        resume_step = int(payload["step"])
        resumed_from = str(args.resume_from)
        loaded_own_checkpoint = True
        truncate_log_to_step(log_path, resume_step)

    dev_rows: list[dict[str, Any]] = [
        row for row in load_log_rows(log_path) if row.get("event") == "dev_eval"
    ]
    best_dev: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for row in dev_rows:
        if is_better_dev(row, best_dev):
            best_dev = row
    if best_dev is not None and best_checkpoint_path(args.checkpoint_dir).exists():
        best_payload = load_own_probe_checkpoint(best_checkpoint_path(args.checkpoint_dir), device)
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in best_payload["model_state_dict"].items()}

    def dev_eval_and_select(step: int, last_train_loss: float | None = None) -> dict[str, Any]:
        nonlocal best_dev, best_state
        row, normal_rows = evaluate_condition(
            model=model,
            episodes=episodes_by_split["dev"],
            split="dev",
            condition=CONDITION_NORMAL,
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
            shape_loss_weight=args.shape_loss_weight,
        )
        row = dict(row)
        row["event"] = "dev_eval"
        row["step"] = step
        row["twin_pair_joint_normal_accuracy"] = twin_pair_joint_normal_accuracy(episodes_by_split["dev"], normal_rows)
        row["selection_key"] = list(dev_selection_key(row))
        row["sealed_metrics_accessed_for_selection"] = False
        row["last_train_loss"] = last_train_loss
        dev_rows.append(row)
        append_jsonl(log_path, row)
        if is_better_dev(row, best_dev):
            best_dev = row
            best_state = state_dict_copy(model)
            save_checkpoint(
                best_checkpoint_path(args.checkpoint_dir),
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                extra={"best_dev_normal_metrics": row, "sealed_metrics_accessed_for_selection": False},
            )
        return row

    start_time = time.perf_counter()
    if resume_step == 0:
        dev0 = dev_eval_and_select(0, last_train_loss=None)
        save_step_checkpoint_if_due(args=args, model=model, optimizer=optimizer, config=config, step=0, dev_row=dev0)
    last_train_loss = None
    for step in range(resume_step + 1, args.max_steps + 1):
        lr = compute_lr(step, args.max_steps, args.learning_rate, args.warmup_steps, args.min_learning_rate)
        train_row = train_step(
            model=model,
            optimizer=optimizer,
            train_episodes=episodes_by_split["train"],
            step=step,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            shape_loss_weight=args.shape_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
            learning_rate=lr,
        )
        last_train_loss = train_row["train_loss"]
        append_jsonl(log_path, train_row)
        dev_row = None
        if step % args.eval_every == 0 or step == args.max_steps:
            dev_row = dev_eval_and_select(step, last_train_loss=last_train_loss)
            print(json.dumps(to_jsonable(dev_row), sort_keys=True), flush=True)
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            save_step_checkpoint_if_due(
                args=args,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                dev_row=dev_row,
            )

    if best_dev is None or best_state is None:
        raise RuntimeError("no dev-selected checkpoint was produced")
    load_state_dict_copy(model, best_state, device)
    selected_hash_before_sealed = hash_file(best_checkpoint_path(args.checkpoint_dir))
    dev_aggregates, dev_per_condition, dev_pair_metrics = evaluate_all_conditions(
        model=model,
        episodes=episodes_by_split["dev"],
        split="dev",
        device=device,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        shape_loss_weight=args.shape_loss_weight,
    )
    sealed_aggregates, sealed_per_condition, sealed_pair_metrics = evaluate_all_conditions(
        model=model,
        episodes=episodes_by_split["sealed"],
        split="sealed",
        device=device,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        shape_loss_weight=args.shape_loss_weight,
    )
    selected_hash_after_sealed = hash_file(best_checkpoint_path(args.checkpoint_dir))
    save_json(out / "dev_metrics.json", {"conditions": dev_aggregates, "pair_metrics": dev_pair_metrics})
    save_json(out / "sealed_metrics.json", {"conditions": sealed_aggregates, "pair_metrics": sealed_pair_metrics})
    save_per_episode(out / "per_episode_dev.json", dev_per_condition)
    save_per_episode(out / "per_episode_sealed.json", sealed_per_condition)
    paired = {"dev": dev_pair_metrics, "sealed": sealed_pair_metrics}
    save_json(out / "paired_twin_metrics.json", paired)
    deltas = condition_deltas({"dev": dev_aggregates, "sealed": sealed_aggregates})
    save_json(out / "condition_deltas.json", deltas)
    examples = render_prediction_examples(
        episodes_by_split["sealed"],
        sealed_per_condition,
    )
    save_text(out / "prediction_examples.txt", examples)
    report = {
        "probe": PROBE_NAME,
        "architecture": ARCHITECTURE_NAME,
        "mode": "full",
        "resumed_from": resumed_from,
        "loaded_own_probe_checkpoint_for_resume": loaded_own_checkpoint,
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - start_time,
        "runtime": runtime_report(device),
        "pytest_result": pytest_result,
        "train_episode_count": len(episodes_by_split["train"]),
        "dev_episode_count": len(episodes_by_split["dev"]),
        "sealed_episode_count": len(episodes_by_split["sealed"]),
        "best_dev_checkpoint": str(best_checkpoint_path(args.checkpoint_dir)),
        "best_dev_step": best_dev["step"],
        "best_dev_metrics": best_dev,
        "sealed_metrics_accessed_for_selection": False,
        "selected_checkpoint_hash_before_sealed": selected_hash_before_sealed,
        "selected_checkpoint_hash_after_sealed": selected_hash_after_sealed,
        "sealed_evaluation_left_selected_checkpoint_unchanged": selected_hash_before_sealed == selected_hash_after_sealed,
        "training_curve": dev_rows,
        "dev_metrics": {"conditions": dev_aggregates, "pair_metrics": dev_pair_metrics},
        "sealed_metrics": {"conditions": sealed_aggregates, "pair_metrics": sealed_pair_metrics},
        "condition_deltas": deltas,
        "support_path_zeroed_available": True,
        "support_path_zeroed_note": "supplementary/OOD; uses existing VRM.forward disable_support_evidence flag",
        "protected_hashes_before": before,
        "protected_hashes_after": protected_hashes(),
        "protected_hashes_unchanged": before == protected_hashes(),
        "single_seed_limit": "single seed 20260628 only; no ARC-2 transfer claim",
    }
    save_json(out / "summary_report.json", report)
    save_text(out / "summary.md", render_summary(report))
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True), flush=True)
    return report


def render_grid(grid: list[list[int]]) -> list[str]:
    return [" ".join(str(cell) for cell in row) for row in grid]


def render_episode_block(title: str, episode: TwinEpisode) -> list[str]:
    lines = [title]
    for idx, pair in enumerate(episode.supports):
        lines.append(f"support pair {idx} input:")
        lines.extend(render_grid(pair.input.to_list()))
        lines.append(f"support pair {idx} output:")
        lines.extend(render_grid(pair.output.to_list()))
    return lines


def render_prediction_row(label: str, row: dict[str, Any]) -> list[str]:
    lines = [
        f"{label}: decision={row['candidate_decision_name']} distances=({row['distance_to_candidate_0']},{row['distance_to_candidate_1']}) "
        f"predicted_shape={row['predicted_shape']} exact={row['direct_grid_exact']}",
        "decoded grid:",
    ]
    lines.extend(render_grid(row["decoded_prediction"]))
    return lines


def render_prediction_examples(episodes: list[TwinEpisode], per_condition: dict[str, list[dict[str, Any]]]) -> str:
    rows_by_condition = {
        condition: {row["episode_id"]: row for row in rows}
        for condition, rows in per_condition.items()
    }
    grouped = {base_id: twins for base_id, twins in sorted_grouped_pairs(episodes).items()}
    correct_flip: list[str] = []
    normal_correct_no_switch: list[str] = []
    failures: list[str] = []
    for base_id, twins in grouped.items():
        ep_a = twins["A"]
        ep_b = twins["B"]
        normal_a = rows_by_condition[CONDITION_NORMAL][ep_a.episode_id]
        normal_b = rows_by_condition[CONDITION_NORMAL][ep_b.episode_id]
        swap_a = rows_by_condition[CONDITION_TWIN_SWAP_SUPPORTS][ep_a.episode_id]
        swap_b = rows_by_condition[CONDITION_TWIN_SWAP_SUPPORTS][ep_b.episode_id]
        normal_pair = normal_a["candidate_decision"] == ep_a.label and normal_b["candidate_decision"] == ep_b.label
        flip_pair = normal_pair and swap_a["candidate_decision"] == ep_a.alternate_label and swap_b["candidate_decision"] == ep_b.alternate_label
        any_switch = (
            normal_a["candidate_decision"] is not None
            and swap_a["candidate_decision"] is not None
            and normal_a["candidate_decision"] != swap_a["candidate_decision"]
        ) or (
            normal_b["candidate_decision"] is not None
            and swap_b["candidate_decision"] is not None
            and normal_b["candidate_decision"] != swap_b["candidate_decision"]
        )
        if flip_pair:
            correct_flip.append(base_id)
        elif normal_pair and not any_switch:
            normal_correct_no_switch.append(base_id)
        elif not normal_pair:
            failures.append(base_id)
    selected: list[tuple[str, str]] = []
    selected.extend(("strongest_correct_mapping_flip", base_id) for base_id in correct_flip[:4])
    selected.extend(("normal_correct_no_switch", base_id) for base_id in normal_correct_no_switch[:4])
    selected.extend(("failure", base_id) for base_id in failures[:4])
    used = {base_id for _, base_id in selected}
    if len(selected) < 12:
        for base_id in grouped:
            if base_id not in used:
                selected.append(("supplemental", base_id))
                used.add(base_id)
            if len(selected) >= 12:
                break
    lines = [
        "# VRM Twin Probe v0 prediction examples",
        "",
        f"selected_base_pair_count: {len(selected)}",
        "",
    ]
    for category, base_id in selected[:12]:
        twins = grouped[base_id]
        ep_a = twins["A"]
        ep_b = twins["B"]
        lines.extend([f"## {category}: {base_id}", ""])
        lines.extend(render_episode_block("support pairs for Twin A:", ep_a))
        lines.extend(render_episode_block("support pairs for Twin B:", ep_b))
        lines.append("shared query:")
        lines.extend(render_grid(ep_a.query_input.to_list()))
        lines.append("candidate_0:")
        lines.extend(render_grid(ep_a.candidates[0].to_list()))
        lines.append("candidate_1:")
        lines.extend(render_grid(ep_a.candidates[1].to_list()))
        lines.append(f"correct label A: {ep_a.label}")
        lines.append(f"correct label B: {ep_b.label}")
        for episode in (ep_a, ep_b):
            lines.append(f"### Twin {episode.twin}")
            for condition, display in (
                (CONDITION_NORMAL, "normal prediction"),
                (CONDITION_TWIN_SWAP_SUPPORTS, "swapped-support prediction"),
                (CONDITION_PAIR_ORDER_SHUFFLED, "pair-order-shuffled prediction"),
            ):
                row = rows_by_condition[condition][episode.episode_id]
                lines.extend(render_prediction_row(display, row))
        normal_a = rows_by_condition[CONDITION_NORMAL][ep_a.episode_id]
        normal_b = rows_by_condition[CONDITION_NORMAL][ep_b.episode_id]
        swap_a = rows_by_condition[CONDITION_TWIN_SWAP_SUPPORTS][ep_a.episode_id]
        swap_b = rows_by_condition[CONDITION_TWIN_SWAP_SUPPORTS][ep_b.episode_id]
        flip_status = (
            normal_a["candidate_decision"] == ep_a.label
            and normal_b["candidate_decision"] == ep_b.label
            and swap_a["candidate_decision"] == ep_a.alternate_label
            and swap_b["candidate_decision"] == ep_b.alternate_label
        )
        lines.extend([f"mapping-flip status: {flip_status}", ""])
    return "\n".join(lines).rstrip() + "\n"


def sorted_grouped_pairs(episodes: list[TwinEpisode]) -> dict[str, dict[str, TwinEpisode]]:
    grouped: dict[str, dict[str, TwinEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.base_id, {})[episode.twin] = episode
    return {base_id: grouped[base_id] for base_id in sorted(grouped)}


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# VRM Twin Probe v0 - Correspondence Capacity Test",
        "",
        "Single-seed synthetic marker twin capacity probe only. This is not ARC-2 training and not an ARC-solving claim.",
        "",
        f"- seed: {report['seed']}",
        f"- device: {report['runtime']['device']}",
        f"- runtime_seconds: {report['runtime_seconds']:.3f}",
        f"- best_dev_checkpoint: `{report['best_dev_checkpoint']}`",
        f"- best_dev_step: {report['best_dev_step']}",
        f"- sealed_metrics_accessed_for_selection: {report['sealed_metrics_accessed_for_selection']}",
        "",
        "## Training curve",
        "",
        "|step|train loss|dev normal candidate score|dev joint twin accuracy|dev exact|dev mean Hamming|",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["training_curve"]:
        train_loss = row.get("last_train_loss")
        train_loss_text = "NA" if train_loss is None else f"{train_loss:.6f}"
        lines.append(
            f"|{row['step']}|{train_loss_text}|{row['candidate_nearest_score']:.6f}|"
            f"{row['twin_pair_joint_normal_accuracy']:.6f}|{row['direct_grid_exact_accuracy']:.6f}|"
            f"{row['mean_target_rectangle_hamming_error']:.6f}|"
        )
    lines.extend(["", "## Final condition metrics", ""])
    for split in ("dev", "sealed"):
        lines.extend([f"### {split}", "", "|condition|candidate score|tie rate|candidate exact|direct exact|shape match|color acc|mean Hamming|", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        rows = report[f"{split}_metrics"]["conditions"]
        for condition, row in rows.items():
            lines.append(
                f"|{condition}|{row['candidate_nearest_score']:.6f}|{row['candidate_tie_rate']:.6f}|"
                f"{row['candidate_exact_match_rate']:.6f}|{row['direct_grid_exact_accuracy']:.6f}|"
                f"{row['shape_match_rate']:.6f}|{row['target_rectangle_color_accuracy']:.6f}|"
                f"{row['mean_target_rectangle_hamming_error']:.6f}|"
            )
        pair = report[f"{split}_metrics"]["pair_metrics"]
        lines.extend(
            [
                "",
                f"- twin_pair_joint_normal_accuracy: {pair['twin_pair_joint_normal_accuracy']:.6f}",
                f"- counterfactual_switch_rate: {pair['counterfactual_switch_rate']:.6f}",
                f"- correct_mapping_switch_rate: {pair['correct_mapping_switch_rate']:.6f}",
                f"- twin_pair_mapping_flip_consistency: {pair['twin_pair_mapping_flip_consistency']:.6f}",
                "",
            ]
        )
    lines.append("Support-path-zeroed is supplementary/OOD and uses the existing `disable_support_evidence` flag.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=PROBE_NAME)
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--shape-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--pair-encoder-layers", type=int, default=1)
    parser.add_argument("--rewrite-layers", type=int, default=2)
    parser.add_argument("--rewrite-steps", type=int, default=4)
    parser.add_argument("--rule-slots", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "dry_run":
        dry_run(args)
    elif args.mode == "full":
        run_full(args)


if __name__ == "__main__":
    main()
