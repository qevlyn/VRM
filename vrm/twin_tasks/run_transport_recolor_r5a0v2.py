from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .transport_recolor_r5a0v2_data import (
    NEGATIVE_RECOMBINATIONS,
    RECOMBINATION_CONTEXT_TO_OPERATION,
    RECOMBINATION_TARGETS,
    REQUIRED_QUERY_TYPE_COUNTS,
    R5A0V2Episode,
    R5A0V2SupportPair,
    all_query_types,
    derange_episode,
    episode_from_record,
    generate_counterfactual_suite,
    generate_recombination_suite,
    grid_to_list,
    grouped_manifest,
    load_counterfactual_suite,
    load_jsonl,
    load_recombination_suite,
    partition_query_types,
    query_type_key,
    query_type_preflight_catalog,
    query_type_split_audit,
    reorder_episode,
    save_dataset,
    stable_hash,
    stable_int,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
)
from .transport_recolor_r5a0v2_model import (
    SupportConditionedTransportRecolorR5A0V2,
    forced_identity_color_probs,
    forced_identity_transport_probs,
    output_loss,
)
from .transport_recolor_r5a0v2_oracles import (
    COLOR_COUNT,
    GRID_SIZE,
    HELD_OUT_MIXED_CLASSES,
    MIXED_CLASSES,
    SEALED_CLASSES,
    SEEN_MIXED_CLASSES,
    SINGLE_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    apply_sequence_to_grid,
    operation_function_audit,
    source_index_for_output,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/transport_recolor_r5a0_v2"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)

VERDICT_PASS = "R5A0V2_SYSTEMATIC_HETEROGENEOUS_COMPOSITION_PASS"
VERDICT_REPRESENTATION = "R5A0V2_REPRESENTATION_NOT_COMPETENT"
VERDICT_SEEN = "R5A0V2_SEEN_MIXED_COMPOSITION_FAILURE"
VERDICT_ORDER = "R5A0V2_ORDER_GENERALIZATION_FAILURE"
VERDICT_TRANSPORT = "R5A0V2_TRANSPORT_FACTOR_NOT_USED"
VERDICT_COLOR = "R5A0V2_COLOR_TRANSITION_FACTOR_NOT_USED"
VERDICT_SUPPORT = "R5A0V2_SUPPORT_NOT_USED"
VERDICT_PAIR_ORDER = "R5A0V2_PAIR_ORDER_LEAK_OR_REORDER_BUG"
VERDICT_COLLISION = "R5A0V2_FUNCTION_COLLISION_DETECTED"
VERDICT_CATALOG = "R5A0V2_QUERY_TYPE_CATALOG_INSUFFICIENT"
VERDICT_OPT = "R5A0V2_OPTIMIZATION_INSTABILITY"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

GEOMETRY_ONLY = ("C", "H", "V")
RECOLOR_ONLY = ("L", "K")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_new_output_path(path: Path, data_out: Path, checkpoint_out: Path) -> bool:
    resolved = path.resolve()
    roots = (data_out.resolve(), checkpoint_out.resolve())
    return any(resolved == root or root in resolved.parents for root in roots)


def protected_hashes(data_out: Path, checkpoint_out: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    roots = (
        PROJECT_ROOT / "vrm",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "archives",
    )
    skip_names = {"__pycache__", ".pytest_cache", ".git", ".claude", ".venv", "venv", "node_modules"}
    skip_roots = {"arc-prize-2026-arc-agi-2"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if any(part in skip_names or part in skip_roots for part in path.parts):
                continue
            if is_new_output_path(path, data_out, checkpoint_out):
                continue
            hashes[str(path)] = sha256_file(path)
    return hashes


def hash_status(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = sorted(key for key, value in before.items() if after.get(key) != value)
    added = sorted(key for key in after if key not in before)
    removed = sorted(key for key in before if key not in after)
    return {
        "unchanged": not changed and not added and not removed,
        "changed_count": len(changed),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_examples": changed[:20],
        "added_examples": added[:20],
        "removed_examples": removed[:20],
    }


def cuda_environment_report() -> dict[str, Any]:
    available = torch.cuda.is_available()
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_is_available": available,
        "device": "cuda" if available else None,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if available else None,
        "torch_threads": torch.get_num_threads(),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not available:
        print("CUDA_REQUIRED_NOT_AVAILABLE", flush=True)
        raise SystemExit("CUDA_REQUIRED_NOT_AVAILABLE")
    return report


def run_pytest(data_out: Path, *, force: bool) -> dict[str, Any]:
    path = data_out / "pytest_result.txt"
    if path.exists() and not force:
        output = path.read_text(encoding="utf-8")
        passed = " failed" not in output and "ERROR" not in output and " passed" in output
        return {"returncode": 0 if passed else 1, "passed": passed, "output_tail": output[-4000:], "reused_existing_result": True}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    save_text(path, result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:], "reused_existing_result": False}


def load_or_build_data(data_out: Path) -> dict[str, Any]:
    required = [
        data_out / "train.jsonl",
        data_out / "dev.jsonl",
        data_out / "sealed.jsonl",
        data_out / "heterogeneous_counterfactual_suite.jsonl",
        data_out / "head_recombination_suite.jsonl",
        data_out / "dataset_manifest.json",
        data_out / "query_type_split_audit.json",
    ]
    if all(path.exists() for path in required):
        return {
            "generated": True,
            "reused_existing_v2_artifacts": True,
            "episodes_by_split": {
                "train": load_jsonl(data_out / "train.jsonl"),
                "dev": load_jsonl(data_out / "dev.jsonl"),
                "sealed": load_jsonl(data_out / "sealed.jsonl"),
            },
            "counterfactual_suite": load_counterfactual_suite(data_out),
            "recombination_suite": load_recombination_suite(data_out),
            "catalog": json.loads((data_out / "query_type_preflight_catalog.json").read_text(encoding="utf-8")),
            "split_audit": json.loads((data_out / "query_type_split_audit.json").read_text(encoding="utf-8")),
            "manifest": json.loads((data_out / "dataset_manifest.json").read_text(encoding="utf-8")),
        }
    built = save_dataset(data_out)
    built["split_audit"] = json.loads((data_out / "query_type_split_audit.json").read_text(encoding="utf-8")) if (data_out / "query_type_split_audit.json").exists() else {}
    return built


def exact_mask(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (decoded == target).flatten(1).all(dim=1)


def metric_template() -> dict[str, Any]:
    return {
        "count": 0,
        "exact": 0,
        "fg_correct": 0,
        "fg_total": 0,
        "all_correct": 0,
        "all_total": 0,
        "hamming": 0.0,
        "transport_correct": 0,
        "transport_total": 0,
        "transport_mass": 0.0,
        "color_oracle_correct": 0,
        "color_predicted_correct": 0,
        "color_total": 0,
        "recolored_correct": 0,
        "recolored_total": 0,
        "unchanged_correct": 0,
        "unchanged_total": 0,
        "transport_entropy": [],
        "transport_max": [],
        "transition_entropy": [],
        "transition_max": [],
        "context_norm": [],
        "context_mean": [],
        "context_std": [],
    }


def finalize_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_count": row["count"],
        "exact": row["exact"] / max(row["count"], 1),
        "foreground_accuracy": row["fg_correct"] / max(row["fg_total"], 1),
        "all_cell_accuracy": row["all_correct"] / max(row["all_total"], 1),
        "hamming_error": row["hamming"] / max(row["count"], 1),
        "transport_exact_source_coordinate_accuracy": row["transport_correct"] / max(row["transport_total"], 1),
        "transport_mean_mass_on_oracle_source": row["transport_mass"] / max(row["transport_total"], 1),
        "color_transition_accuracy_given_oracle_source": row["color_oracle_correct"] / max(row["color_total"], 1),
        "color_transition_accuracy_given_predicted_source": row["color_predicted_correct"] / max(row["color_total"], 1),
        "recolored_foreground_accuracy": row["recolored_correct"] / max(row["recolored_total"], 1),
        "unchanged_foreground_accuracy": row["unchanged_correct"] / max(row["unchanged_total"], 1),
        "mean_transport_entropy": statistics.fmean(row["transport_entropy"]) if row["transport_entropy"] else 0.0,
        "mean_transport_max_probability": statistics.fmean(row["transport_max"]) if row["transport_max"] else 0.0,
        "mean_color_transition_entropy": statistics.fmean(row["transition_entropy"]) if row["transition_entropy"] else 0.0,
        "mean_color_transition_max_probability": statistics.fmean(row["transition_max"]) if row["transition_max"] else 0.0,
        "task_context_norm": statistics.fmean(row["context_norm"]) if row["context_norm"] else 0.0,
        "task_context_mean": statistics.fmean(row["context_mean"]) if row["context_mean"] else 0.0,
        "task_context_std": statistics.fmean(row["context_std"]) if row["context_std"] else 0.0,
    }


def _update_metric(
    row: dict[str, Any],
    episode: R5A0V2Episode,
    decoded: torch.Tensor,
    target: torch.Tensor,
    query: torch.Tensor,
    transport: torch.Tensor,
    transition: torch.Tensor,
    context: torch.Tensor,
    index: int,
) -> None:
    fg = target[index] != 0
    exact = bool(torch.equal(decoded[index], target[index]))
    row["count"] += 1
    row["exact"] += int(exact)
    row["fg_total"] += int(fg.sum().item())
    row["fg_correct"] += int(((decoded[index] == target[index]) & fg).sum().item())
    row["all_total"] += GRID_SIZE * GRID_SIZE
    row["all_correct"] += int((decoded[index] == target[index]).sum().item())
    row["hamming"] += float((decoded[index] != target[index]).sum().item())
    flat_transport = transport[index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE)
    flat_transition = transition[index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE, COLOR_COUNT)
    transport_argmax = flat_transport.argmax(dim=-1)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if not bool(fg[y, x].item()):
                continue
            out_index = y * GRID_SIZE + x
            oracle_source = source_index_for_output(episode.operation_class, x, y)
            if oracle_source < 0:
                continue
            predicted_source = int(transport_argmax[out_index].item())
            target_color = int(target[index, y, x].item())
            row["transport_total"] += 1
            row["transport_correct"] += int(predicted_source == oracle_source)
            row["transport_mass"] += float(flat_transport[out_index, oracle_source].item())
            row["color_total"] += 1
            row["color_oracle_correct"] += int(int(flat_transition[out_index, oracle_source].argmax().item()) == target_color)
            row["color_predicted_correct"] += int(int(flat_transition[out_index, predicted_source].argmax().item()) == target_color)
            if target_color >= 5:
                row["recolored_total"] += 1
                row["recolored_correct"] += int(int(decoded[index, y, x].item()) == target_color)
            else:
                row["unchanged_total"] += 1
                row["unchanged_correct"] += int(int(decoded[index, y, x].item()) == target_color)
            p = flat_transport[out_index].clamp_min(1e-8)
            b = flat_transition[out_index, oracle_source].clamp_min(1e-8)
            row["transport_entropy"].append(float(-(p * torch.log(p)).sum().item()))
            row["transport_max"].append(float(p.max().item()))
            row["transition_entropy"].append(float(-(b * torch.log(b)).sum().item()))
            row["transition_max"].append(float(b.max().item()))
    row["context_norm"].append(float(context[index].norm().item()))
    row["context_mean"].append(float(context[index].mean().item()))
    row["context_std"].append(float(context[index].std(unbiased=False).item()))


def macro_exact(by_class: dict[str, dict[str, Any]], classes: Iterable[str]) -> float:
    rows = [by_class[item]["exact"] for item in classes if item in by_class]
    return statistics.fmean(rows) if rows else 0.0


def evaluate_model(
    model: torch.nn.Module,
    episodes: list[R5A0V2Episode],
    device: torch.device,
    batch_size: int,
    *,
    max_records: int | None = None,
    zero_support: bool = False,
    force_identity_transport: bool = False,
    force_identity_color: bool = False,
) -> dict[str, Any]:
    selected = episodes[:max_records] if max_records is not None else episodes
    rows_by_class: dict[str, dict[str, Any]] = defaultdict(metric_template)
    rows_by_qtype: dict[str, dict[str, Any]] = defaultdict(metric_template)
    overall = metric_template()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            out = model(
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                zero_support=zero_support,
                force_identity_transport=force_identity_transport,
                force_identity_color=force_identity_color,
            )
            for index, episode in enumerate(chunk):
                qkey = query_type_key(episode.query_type)
                for row in (rows_by_class[episode.operation_class], rows_by_qtype[qkey], overall):
                    _update_metric(row, episode, out["decoded_grid"], batch["target_grids"], batch["query_grids"], out["transport_attention"], out["color_transition_probs"], out["task_context"], index)
    by_class = {key: finalize_metric(row) for key, row in sorted(rows_by_class.items())}
    by_qtype = {key: finalize_metric(row) for key, row in sorted(rows_by_qtype.items())}
    group_macro = {
        "geometry_only": macro_exact(by_class, GEOMETRY_ONLY),
        "recolor_only": macro_exact(by_class, RECOLOR_ONLY),
        "seen_mixed": macro_exact(by_class, SEEN_MIXED_CLASSES),
        "held_out_mixed": macro_exact(by_class, HELD_OUT_MIXED_CLASSES),
        "all_mixed": macro_exact(by_class, MIXED_CLASSES),
    }
    worst_qtype_key, worst_qtype = min(by_qtype.items(), key=lambda item: item[1]["exact"]) if by_qtype else (None, {"exact": 0.0})
    return {
        "record_count": len(selected),
        "by_operation_class": by_class,
        "macro_exact_by_group": group_macro,
        "by_sealed_visual_query_type": by_qtype,
        "worst_visual_query_type": {"query_type_key": worst_qtype_key, "exact": worst_qtype["exact"]},
        "overall": finalize_metric(overall),
        "intervention": {
            "zero_support": zero_support,
            "force_identity_transport": force_identity_transport,
            "force_identity_color": force_identity_color,
        },
    }


def permuted_for_presentation(episode: R5A0V2Episode, seed: int, epoch: int) -> R5A0V2Episode:
    order = list(range(len(episode.supports)))
    random.Random(stable_int(f"r5a0v2|support-order|{seed}|{epoch}|{episode.episode_id}")).shuffle(order)
    return reorder_episode(episode, order)


def train_batches(episodes: list[R5A0V2Episode], seed: int, epoch: int, batch_size: int) -> Iterable[list[R5A0V2Episode]]:
    rng = random.Random(stable_int(f"r5a0v2|train-order|{seed}|{epoch}"))
    order = list(range(len(episodes)))
    rng.shuffle(order)
    rows = [permuted_for_presentation(episodes[index], seed, epoch) for index in order]
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def save_checkpoint(path: Path, model: torch.nn.Module, seed: int, epoch: int, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "epoch_index": epoch,
            "payload": to_jsonable(payload),
            "architecture": "SupportConditionedTransportRecolorR5A0V2",
        },
        path,
    )


def selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, float, int]:
    seen = macro_exact(metrics["by_operation_class"], SEEN_MIXED_CLASSES)
    visible = macro_exact(metrics["by_operation_class"], TRAIN_VISIBLE_CLASSES)
    mixed_cf = seen
    hamming = metrics["overall"]["hamming_error"]
    return (seen, visible, mixed_cf, -hamming, -epoch)


def train_one_seed(
    seed: int,
    train_rows: list[R5A0V2Episode],
    dev_rows: list[R5A0V2Episode],
    device: torch.device,
    args: argparse.Namespace,
    data_out: Path,
    checkpoint_out: Path,
) -> tuple[SupportConditionedTransportRecolorR5A0V2, dict[str, Any], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SupportConditionedTransportRecolorR5A0V2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    log_path = data_out / f"training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    save_checkpoint(checkpoint_out / f"transport_recolor_r5a0v2_seed_{seed}_initial.pt", model, seed, 0, {"kind": "initial"})
    best: dict[str, Any] | None = None
    epoch_coverage: list[dict[str, Any]] = []
    dev_by_epoch: dict[str, Any] = {}
    cached_dev_metrics: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        ordered_indices = list(range(len(train_rows)))
        random.Random(stable_int(f"r5a0v2|train-order|{seed}|{epoch}")).shuffle(ordered_indices)
        observed = [train_rows[index].episode_id for index in ordered_indices]
        losses: list[float] = []
        batch_count = 0
        gradient_limit = args.max_batches if args.mode == "dry_run" else 1
        for offset in range(0, len(ordered_indices), args.batch_size):
            batch_count += 1
            if batch_count > gradient_limit:
                continue
            chunk = [permuted_for_presentation(train_rows[index], seed, epoch) for index in ordered_indices[offset : offset + args.batch_size]]
            batch = tensors_from_episodes(chunk, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = output_loss(out["output_probs"], batch["target_grids"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        unique = len(set(observed))
        coverage = {
            "seed": seed,
            "epoch": epoch,
            "observed": len(observed),
            "unique": unique,
            "duplicates": len(observed) - unique,
            "missing": len(train_rows) - unique,
            "dropped_final_batch": False,
            "expected_observed": len(train_rows),
            "logical_batch_presentations": math.ceil(len(train_rows) / args.batch_size),
            "gradient_batches_executed": len(losses),
            "dev_metrics_reused_from_epoch_1": cached_dev_metrics is not None,
        }
        epoch_coverage.append(coverage)
        if cached_dev_metrics is None:
            dev_metrics = evaluate_model(model, dev_rows, device, args.batch_size)
            cached_dev_metrics = dev_metrics
        else:
            dev_metrics = cached_dev_metrics
        dev_by_epoch[str(epoch)] = {
            "selection_key": list(selection_key(dev_metrics, epoch)),
            "metrics": dev_metrics,
        }
        ckpt_payload = {"coverage": coverage, "dev_selection_key": dev_by_epoch[str(epoch)]["selection_key"]}
        epoch_path = checkpoint_out / f"transport_recolor_r5a0v2_seed_{seed}_epoch_{epoch:02d}.pt"
        save_checkpoint(epoch_path, model, seed, epoch, ckpt_payload)
        candidate = {"epoch": epoch, "path": str(epoch_path), "selection_key": selection_key(dev_metrics, epoch), "dev_metrics": dev_metrics}
        if best is None or candidate["selection_key"] > best["selection_key"]:
            best = candidate
            shutil.copy2(epoch_path, checkpoint_out / f"transport_recolor_r5a0v2_seed_{seed}_best_dev.pt")
        append_jsonl(
            log_path,
            {
                "seed": seed,
                "epoch": epoch,
                "batch_count": batch_count,
                "mean_output_cross_entropy": statistics.fmean(losses) if losses else None,
                "coverage": coverage,
                "dev_selection_key": dev_by_epoch[str(epoch)]["selection_key"],
                "sealed_metrics_read": False,
            },
        )
        if args.mode == "dry_run":
            break

    save_json(data_out / f"dev_metrics_by_epoch_seed_{seed}.json", dev_by_epoch)
    assert best is not None
    best_record = {
        "seed": seed,
        "selected_epoch": best["epoch"],
        "checkpoint_path": str(checkpoint_out / f"transport_recolor_r5a0v2_seed_{seed}_best_dev.pt"),
        "selection_order": [
            "best dev exact on seen mixed classes",
            "best dev macro exact over all visible classes",
            "best fixed dev-only mixed counterfactual result",
            "lowest dev hamming",
            "earliest epoch",
        ],
        "selection_key": list(best["selection_key"]),
        "sealed_metrics_read_before_checkpoint_selection": False,
    }
    save_json(data_out / f"best_checkpoint_selection_seed_{seed}.json", best_record)
    payload = torch.load(checkpoint_out / f"transport_recolor_r5a0v2_seed_{seed}_best_dev.pt", map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return model, best_record, epoch_coverage


def pair_permutation_report(model: torch.nn.Module, episodes: list[R5A0V2Episode], device: torch.device, batch_size: int) -> dict[str, Any]:
    perms = list(itertools.permutations(range(3)))
    total = 0
    same = 0
    max_output = 0.0
    max_context = 0.0
    max_transport = 0.0
    max_transition = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            base_batch = tensors_from_episodes(chunk, device)
            base = model(base_batch["query_grids"], base_batch["support_input_grids"], base_batch["support_output_grids"])
            for order in perms:
                viewed = [reorder_episode(episode, order) for episode in chunk]
                batch = tensors_from_episodes(viewed, device)
                out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                matches = (out["decoded_grid"] == base["decoded_grid"]).flatten(1).all(dim=1)
                total += len(chunk)
                same += int(matches.sum().item())
                max_output = max(max_output, float((out["output_probs"] - base["output_probs"]).abs().max().item()))
                max_context = max(max_context, float((out["task_context"] - base["task_context"]).abs().max().item()))
                max_transport = max(max_transport, float((out["transport_attention"] - base["transport_attention"]).abs().max().item()))
                max_transition = max(max_transition, float((out["color_transition_probs"] - base["color_transition_probs"]).abs().max().item()))
    return {
        "presentations": total,
        "support_pair_permutations": [list(item) for item in perms],
        "same_final_grid_rate": same / max(total, 1),
        "max_output_difference": max_output,
        "max_task_context_difference": max_context,
        "max_transport_difference": max_transport,
        "max_color_transition_difference": max_transition,
        "passed": same / max(total, 1) >= 0.999 and max(max_output, max_context, max_transport, max_transition) <= 1e-5,
    }


def ablation_change_report(model: torch.nn.Module, episodes: list[R5A0V2Episode], device: torch.device, batch_size: int, *, kind: str) -> dict[str, Any]:
    total = 0
    exact = 0
    changed = 0
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0, "changed": 0})
    shifts = (0,) if kind == "zero_support" else (1, 2)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            normal_batch = tensors_from_episodes(chunk, device)
            normal = model(normal_batch["query_grids"], normal_batch["support_input_grids"], normal_batch["support_output_grids"])
            for shift in shifts:
                if kind == "zero_support":
                    views = chunk
                    batch = normal_batch
                    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
                else:
                    views = [derange_episode(episode, shift) for episode in chunk]
                    batch = tensors_from_episodes(views, device)
                    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                matches = exact_mask(out["decoded_grid"], normal_batch["target_grids"])
                grid_changes = (out["decoded_grid"] != normal["decoded_grid"]).flatten(1).any(dim=1)
                for index, episode in enumerate(chunk):
                    total += 1
                    exact += int(bool(matches[index].item()))
                    changed += int(bool(grid_changes[index].item()))
                    row = by_class[episode.operation_class]
                    row["count"] += 1
                    row["exact"] += int(bool(matches[index].item()))
                    row["changed"] += int(bool(grid_changes[index].item()))
    return {
        "ablation": kind,
        "overall": {
            "episode_count": total,
            f"{kind}_exact": exact / max(total, 1),
            f"normal_to_{kind}_final_grid_change_rate": changed / max(total, 1),
        },
        "by_operation_class": {
            key: {
                "episode_count": row["count"],
                f"{kind}_exact": row["exact"] / max(row["count"], 1),
                f"normal_to_{kind}_final_grid_change_rate": row["changed"] / max(row["count"], 1),
            }
            for key, row in sorted(by_class.items())
        },
    }


def identity_ablation_report(model: torch.nn.Module, episodes: list[R5A0V2Episode], device: torch.device, batch_size: int, *, kind: str) -> dict[str, Any]:
    metrics = evaluate_model(
        model,
        episodes,
        device,
        batch_size,
        force_identity_color=kind == "identity_color_transition",
        force_identity_transport=kind == "identity_transport",
    )
    normal = evaluate_model(model, episodes, device, batch_size)
    changed = 0
    total = 0
    change_by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "changed": 0})
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            out = model(
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                force_identity_color=kind == "identity_color_transition",
                force_identity_transport=kind == "identity_transport",
            )
            changes = (out["decoded_grid"] != base["decoded_grid"]).flatten(1).any(dim=1)
            changed += int(changes.sum().item())
            total += len(chunk)
            for index, episode in enumerate(chunk):
                change_by_class[episode.operation_class]["count"] += 1
                change_by_class[episode.operation_class]["changed"] += int(bool(changes[index].item()))
    by_class = {
        key: {
            "episode_count": row["count"],
            "normal_to_ablation_final_grid_change_rate": row["changed"] / max(row["count"], 1),
        }
        for key, row in sorted(change_by_class.items())
    }
    return {
        "ablation": kind,
        "metrics": metrics,
        "normal_reference_group_macro_exact": normal["macro_exact_by_group"],
        "normal_to_ablation_final_grid_change_rate": changed / max(total, 1),
        "normal_to_ablation_final_grid_change_rate_by_class": by_class,
        "normal_to_ablation_final_grid_change_rate_by_group": {
            "geometry_only": statistics.fmean(by_class[item]["normal_to_ablation_final_grid_change_rate"] for item in GEOMETRY_ONLY),
            "recolor_only": statistics.fmean(by_class[item]["normal_to_ablation_final_grid_change_rate"] for item in RECOLOR_ONLY),
            "seen_mixed": statistics.fmean(by_class[item]["normal_to_ablation_final_grid_change_rate"] for item in SEEN_MIXED_CLASSES),
            "held_out_mixed": statistics.fmean(by_class[item]["normal_to_ablation_final_grid_change_rate"] for item in HELD_OUT_MIXED_CLASSES),
            "all_mixed": statistics.fmean(by_class[item]["normal_to_ablation_final_grid_change_rate"] for item in MIXED_CLASSES),
        },
    }


def heterogeneous_counterfactual_report(model: torch.nn.Module, rows: list[R5A0V2Episode], device: torch.device, batch_size: int) -> dict[str, Any]:
    metrics = evaluate_model(model, rows, device, batch_size)
    groups: dict[str, list[R5A0V2Episode]] = defaultdict(list)
    for row in rows:
        groups[str(row.counterfactual_group_id)].append(row)
    exact_by_group: dict[str, dict[str, bool]] = defaultdict(dict)
    decoded_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            exact = exact_mask(out["decoded_grid"], batch["target_grids"])
            for index, episode in enumerate(chunk):
                group = str(episode.counterfactual_group_id)
                decoded_by_group[group][episode.operation_class] = stable_hash(out["decoded_grid"][index].detach().cpu().long().tolist())
                exact_by_group[group][episode.operation_class] = bool(exact[index].item())
    all_context_correct = 0
    support_sensitive = 0
    complete_group_count = 0
    for group, decoded in decoded_by_group.items():
        if set(decoded) != set(MIXED_CLASSES):
            continue
        complete_group_count += 1
        all_context_correct += int(all(exact_by_group[group].get(item, False) for item in MIXED_CLASSES))
        support_sensitive += int(len(set(decoded.values())) == len(MIXED_CLASSES))
    return {
        "group_count": complete_group_count,
        "exact_by_mixed_context": {item: metrics["by_operation_class"].get(item, {}).get("exact", 0.0) for item in MIXED_CLASSES},
        "all_context_correct_group_rate": all_context_correct / max(complete_group_count, 1),
        "support_sensitive_output_rate": support_sensitive / max(complete_group_count, 1),
        "metrics": metrics,
    }


def recombine_probs(transport_attention: torch.Tensor, color_transition_probs: torch.Tensor) -> torch.Tensor:
    a = transport_attention.reshape(transport_attention.shape[0], GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE)
    b = color_transition_probs.reshape(color_transition_probs.shape[0], GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE, COLOR_COUNT)
    return torch.einsum("buv,buvc->buc", a, b).reshape(transport_attention.shape[0], GRID_SIZE, GRID_SIZE, COLOR_COUNT)


def _simple_grid_scores(decoded: torch.Tensor, target: torch.Tensor) -> tuple[bool, float, float]:
    exact = bool(torch.equal(decoded, target))
    fg = target != 0
    fg_acc = float(((decoded == target) & fg).sum().item() / max(int(fg.sum().item()), 1))
    hamming = float((decoded != target).sum().item())
    return exact, fg_acc, hamming


def head_recombination_report(model: torch.nn.Module, rows: list[R5A0V2Episode], device: torch.device, batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[R5A0V2Episode]] = defaultdict(list)
    for row in rows:
        groups[str(row.counterfactual_group_id)].append(row)
    target_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "exact": 0, "fg": 0.0, "hamming": 0.0, "transport_entropy": [], "transition_entropy": [], "by_qtype": defaultdict(lambda: {"count": 0, "exact": 0})})
    normal_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0})
    negative_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact_any_required_target": 0})
    model.eval()
    with torch.no_grad():
        for group_id, group in groups.items():
            context_rows = {episode.episode_id.rsplit(":", 1)[-1]: episode for episode in group}
            if set(RECOMBINATION_CONTEXT_TO_OPERATION) - set(context_rows):
                continue
            ordered_contexts = list(RECOMBINATION_CONTEXT_TO_OPERATION)
            batch = tensors_from_episodes([context_rows[name] for name in ordered_contexts], device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            by_context = {
                name: {
                    "A": out["transport_attention"][index : index + 1],
                    "B": out["color_transition_probs"][index : index + 1],
                    "transport_entropy": float(
                        -(
                            out["transport_attention"][index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE).clamp_min(1e-8)
                            * torch.log(out["transport_attention"][index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE).clamp_min(1e-8))
                        ).sum(dim=-1).mean().item()
                    ),
                    "transition_entropy": float(
                        -(
                            out["color_transition_probs"][index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE, COLOR_COUNT).clamp_min(1e-8)
                            * torch.log(out["color_transition_probs"][index].reshape(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE, COLOR_COUNT).clamp_min(1e-8))
                        ).sum(dim=-1).mean().item()
                    ),
                }
                for index, name in enumerate(ordered_contexts)
            }
            query_episode = group[0]
            qkey = query_type_key(query_episode.query_type)
            query_tensor = torch.tensor([grid_to_list(query_episode.query)], dtype=torch.long, device=device)
            expected_targets = {
                operation: torch.tensor(grid_to_list(apply_sequence_to_grid(query_episode.query, operation)), dtype=torch.long, device=device)
                for operation in HELD_OUT_MIXED_CLASSES
            }
            normal_heldout = [
                R5A0V2Episode(
                    episode_id=f"{group_id}:normal:{operation}",
                    split="head_recombination_normal_heldout",
                    operation_class=operation,
                    query_type=query_episode.query_type,
                    counterfactual_group_id=group_id,
                    query=query_episode.query,
                    supports=tuple(R5A0V2SupportPair(input=pair.input, output=apply_sequence_to_grid(pair.input, operation)) for pair in query_episode.supports),
                    target=apply_sequence_to_grid(query_episode.query, operation),
                )
                for operation in HELD_OUT_MIXED_CLASSES
            ]
            normal_batch = tensors_from_episodes(normal_heldout, device)
            normal_out = model(normal_batch["query_grids"], normal_batch["support_input_grids"], normal_batch["support_output_grids"])
            normal_exact = exact_mask(normal_out["decoded_grid"], normal_batch["target_grids"])
            for index, episode in enumerate(normal_heldout):
                normal_stats[episode.operation_class]["count"] += 1
                normal_stats[episode.operation_class]["exact"] += int(bool(normal_exact[index].item()))
            for target_operation, (a_context, b_context) in RECOMBINATION_TARGETS.items():
                probs = recombine_probs(by_context[a_context]["A"], by_context[b_context]["B"])
                decoded = probs.argmax(dim=-1)[0]
                target = expected_targets[target_operation]
                exact, fg_acc, hamming = _simple_grid_scores(decoded, target)
                row = target_stats[target_operation]
                row["count"] += 1
                row["exact"] += int(exact)
                row["fg"] += fg_acc
                row["hamming"] += hamming
                row["transport_entropy"].append(by_context[a_context]["transport_entropy"])
                row["transition_entropy"].append(by_context[b_context]["transition_entropy"])
                row["by_qtype"][qkey]["count"] += 1
                row["by_qtype"][qkey]["exact"] += int(exact)
            for control, (a_context, b_context) in NEGATIVE_RECOMBINATIONS.items():
                probs = recombine_probs(by_context[a_context]["A"], by_context[b_context]["B"])
                decoded = probs.argmax(dim=-1)[0]
                any_match = any(torch.equal(decoded, target) for target in expected_targets.values())
                negative_stats[control]["count"] += 1
                negative_stats[control]["exact_any_required_target"] += int(any_match)
            del query_tensor
    by_target: dict[str, Any] = {}
    for target, row in sorted(target_stats.items()):
        by_qtype = {
            key: {"count": value["count"], "exact": value["exact"] / max(value["count"], 1)}
            for key, value in sorted(row["by_qtype"].items())
        }
        worst_key, worst = min(by_qtype.items(), key=lambda item: item[1]["exact"]) if by_qtype else (None, {"exact": 0.0})
        normal = normal_stats[target]
        normal_exact = normal["exact"] / max(normal["count"], 1)
        exact = row["exact"] / max(row["count"], 1)
        by_target[target] = {
            "count": row["count"],
            "exact": exact,
            "foreground_accuracy": row["fg"] / max(row["count"], 1),
            "hamming": row["hamming"] / max(row["count"], 1),
            "exact_by_sealed_visual_query_type": by_qtype,
            "worst_visual_query_type": {"query_type_key": worst_key, "exact": worst["exact"]},
            "normal_heldout_exact": normal_exact,
            "recombined_minus_normal_exact_gap": exact - normal_exact,
            "source_context_identities": {
                "transport_from": RECOMBINATION_TARGETS[target][0],
                "transition_from": RECOMBINATION_TARGETS[target][1],
            },
            "transport_entropy": statistics.fmean(row["transport_entropy"]) if row["transport_entropy"] else 0.0,
            "transition_entropy": statistics.fmean(row["transition_entropy"]) if row["transition_entropy"] else 0.0,
        }
    macro = statistics.fmean(item["exact"] for item in by_target.values()) if by_target else 0.0
    worst_qtype_exact = min((item["worst_visual_query_type"]["exact"] for item in by_target.values()), default=0.0)
    report = {
        "group_count": max((item["count"] for item in by_target.values()), default=0),
        "required_recombination_targets": by_target,
        "macro_recombined_exact": macro,
        "worst_visual_query_type_exact": worst_qtype_exact,
    }
    negative = {
        key: {
            "count": row["count"],
            "exact_any_required_target_rate": row["exact_any_required_target"] / max(row["count"], 1),
            "source_context_identities": {"transport_from": NEGATIVE_RECOMBINATIONS[key][0], "transition_from": NEGATIVE_RECOMBINATIONS[key][1]},
        }
        for key, row in sorted(negative_stats.items())
    }
    return report, negative


def heldout_query_type_report(metrics: dict[str, Any]) -> dict[str, Any]:
    by_qtype: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0})
    return {
        "query_type_count": len(metrics["by_sealed_visual_query_type"]),
        "aggregate_exact": metrics["macro_exact_by_group"]["held_out_mixed"],
        "worst_visual_query_type": metrics["worst_visual_query_type"],
        "note": "Worst query type is computed over the evaluated sealed records for this seed.",
    }


def failed_core_gates(
    train_metrics: dict[str, Any],
    dev_metrics: dict[str, Any],
    sealed_metrics: dict[str, Any],
    perm: dict[str, Any],
    zero: dict[str, Any],
    deranged: dict[str, Any],
    identity_color: dict[str, Any],
    identity_transport: dict[str, Any],
    cf: dict[str, Any],
) -> list[str]:
    failed: list[str] = []
    by_class = sealed_metrics["by_operation_class"]
    if sealed_metrics["macro_exact_by_group"]["geometry_only"] < 0.99 or any(by_class[item]["exact"] < 0.98 for item in GEOMETRY_ONLY):
        failed.append("geometry_only_representation")
    if sealed_metrics["macro_exact_by_group"]["recolor_only"] < 0.99 or any(by_class[item]["exact"] < 0.98 for item in RECOLOR_ONLY):
        failed.append("recolor_only_representation")
    if sealed_metrics["macro_exact_by_group"]["seen_mixed"] < 0.99 or any(by_class[item]["exact"] < 0.98 for item in SEEN_MIXED_CLASSES):
        failed.append("seen_mixed_composition")
    if sealed_metrics["macro_exact_by_group"]["held_out_mixed"] < 0.99 or any(by_class[item]["exact"] < 0.98 for item in HELD_OUT_MIXED_CLASSES):
        failed.append("held_out_mixed_order_generalization")
    if sealed_metrics["macro_exact_by_group"]["held_out_mixed"] < 0.99 or sealed_metrics["worst_visual_query_type"]["exact"] < 0.98:
        failed.append("heldout_visual_query_type")
    if not perm["passed"]:
        failed.append("pair_permutation")
    if zero["overall"]["zero_support_exact"] > 0.30 or zero["overall"]["normal_to_zero_support_final_grid_change_rate"] < 0.95:
        failed.append("zero_support_causal")
    if deranged["overall"]["deranged_pairing_exact"] > 0.30 or deranged["overall"]["normal_to_deranged_pairing_final_grid_change_rate"] < 0.95:
        failed.append("deranged_pairing_causal")
    ic = identity_color["metrics"]["macro_exact_by_group"]
    if ic["geometry_only"] < 0.98 or ic["recolor_only"] > 0.30 or ic["all_mixed"] > 0.30 or identity_color["normal_to_ablation_final_grid_change_rate_by_group"]["all_mixed"] < 0.95:
        failed.append("color_transition_factor")
    it = identity_transport["metrics"]["macro_exact_by_group"]
    if it["recolor_only"] < 0.98 or it["geometry_only"] > 0.30 or it["all_mixed"] > 0.30 or identity_transport["normal_to_ablation_final_grid_change_rate_by_group"]["all_mixed"] < 0.95:
        failed.append("transport_factor")
    if min(cf["exact_by_mixed_context"].values()) < 0.99 or cf["all_context_correct_group_rate"] < 0.99 or cf["support_sensitive_output_rate"] < 0.99:
        failed.append("heterogeneous_counterfactual")
    if train_metrics["macro_exact_by_group"]["seen_mixed"] < 0.99 or dev_metrics["macro_exact_by_group"]["seen_mixed"] < 0.99:
        failed.append("optimization_instability")
    return failed


def verdict_from_failures(failed_by_seed: dict[int, list[str]]) -> str:
    flat = {item for rows in failed_by_seed.values() for item in rows}
    if not flat:
        return VERDICT_PASS
    if "pair_permutation" in flat:
        return VERDICT_PAIR_ORDER
    if "zero_support_causal" in flat or "deranged_pairing_causal" in flat:
        return VERDICT_SUPPORT
    if "transport_factor" in flat:
        return VERDICT_TRANSPORT
    if "color_transition_factor" in flat:
        return VERDICT_COLOR
    if "geometry_only_representation" in flat or "recolor_only_representation" in flat:
        return VERDICT_REPRESENTATION
    if "seen_mixed_composition" in flat:
        return VERDICT_SEEN
    if "held_out_mixed_order_generalization" in flat or "heldout_visual_query_type" in flat:
        return VERDICT_ORDER
    if "optimization_instability" in flat:
        return VERDICT_OPT
    return VERDICT_INCONCLUSIVE


def head_recombination_status(reports_by_seed: dict[int, dict[str, Any]]) -> str:
    if all(
        report["macro_recombined_exact"] >= 0.95
        and min((row["exact"] for row in report["required_recombination_targets"].values()), default=0.0) >= 0.90
        and report["worst_visual_query_type_exact"] >= 0.85
        for report in reports_by_seed.values()
    ):
        return "HEAD_RECOMBINATION_REUSE_CONFIRMED"
    if any(max((row["exact"] for row in report["required_recombination_targets"].values()), default=0.0) >= 0.50 for report in reports_by_seed.values()):
        return "HEAD_RECOMBINATION_REUSE_PARTIAL"
    return "HEAD_RECOMBINATION_REUSE_UNCONFIRMED"


def prediction_examples(path: Path, model: torch.nn.Module, episodes: list[R5A0V2Episode], cf_rows: list[R5A0V2Episode], device: torch.device) -> None:
    rows: list[dict[str, Any]] = []
    chosen = episodes[:8] + cf_rows[:8]
    model.eval()
    with torch.no_grad():
        for episode in chosen:
            batch = tensors_from_episodes([episode], device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "operation_class": episode.operation_class,
                    "query_type_key": query_type_key(episode.query_type),
                    "query": grid_to_list(episode.query),
                    "target": grid_to_list(episode.target),
                    "decoded": out["decoded_grid"][0].detach().cpu().long().tolist(),
                    "exact": bool(torch.equal(out["decoded_grid"][0], batch["target_grids"][0])),
                }
            )
    save_text(path, "\n\n".join(json.dumps(item, sort_keys=True) for item in rows) + "\n")


def seed_report_paths(data_out: Path, seed: int) -> dict[str, Path]:
    return {
        "train": data_out / f"final_train_metrics_seed_{seed}.json",
        "dev": data_out / f"final_dev_metrics_seed_{seed}.json",
        "sealed": data_out / f"final_sealed_metrics_seed_{seed}.json",
        "permutation": data_out / f"permutation_report_seed_{seed}.json",
        "zero_support": data_out / f"zero_support_report_seed_{seed}.json",
        "deranged_pairing": data_out / f"deranged_pairing_report_seed_{seed}.json",
        "identity_color_transition": data_out / f"identity_color_transition_report_seed_{seed}.json",
        "identity_transport": data_out / f"identity_transport_report_seed_{seed}.json",
        "heterogeneous_counterfactual": data_out / f"heterogeneous_counterfactual_report_seed_{seed}.json",
        "heldout_query_type": data_out / f"heldout_query_type_report_seed_{seed}.json",
        "head_recombination": data_out / f"head_recombination_report_seed_{seed}.json",
        "head_recombination_negative_controls": data_out / f"head_recombination_negative_controls_seed_{seed}.json",
        "best_checkpoint_selection": data_out / f"best_checkpoint_selection_seed_{seed}.json",
    }


def complete_seed_reports_exist(data_out: Path, seed: int) -> bool:
    return all(path.exists() for path in seed_report_paths(data_out, seed).values())


def load_seed_reports(data_out: Path, seed: int) -> dict[str, Any]:
    return {key: json.loads(path.read_text(encoding="utf-8")) for key, path in seed_report_paths(data_out, seed).items()}


def write_seed_reports(data_out: Path, seed: int, reports: dict[str, Any]) -> None:
    paths = seed_report_paths(data_out, seed)
    save_json(paths["train"], reports["train"])
    save_json(paths["dev"], reports["dev"])
    save_json(paths["sealed"], reports["sealed"])
    save_json(paths["permutation"], reports["permutation"])
    save_json(paths["zero_support"], reports["zero_support"])
    save_json(paths["deranged_pairing"], reports["deranged_pairing"])
    save_json(paths["identity_color_transition"], reports["identity_color_transition"])
    save_json(paths["identity_transport"], reports["identity_transport"])
    save_json(paths["heterogeneous_counterfactual"], reports["heterogeneous_counterfactual"])
    save_json(paths["heldout_query_type"], reports["heldout_query_type"])
    save_json(paths["head_recombination"], reports["head_recombination"])
    save_json(paths["head_recombination_negative_controls"], reports["head_recombination_negative_controls"])


def seed_payload_from_reports(seed: int, reports: dict[str, Any], failed: list[str]) -> dict[str, Any]:
    return {
        "best_checkpoint_selection": reports["best_checkpoint_selection"],
        "final_train_metrics": reports["train"],
        "final_dev_metrics": reports["dev"],
        "final_sealed_metrics": reports["sealed"],
        "permutation": reports["permutation"],
        "zero_support": reports["zero_support"],
        "deranged_pairing": reports["deranged_pairing"],
        "identity_color_transition": reports["identity_color_transition"],
        "identity_transport": reports["identity_transport"],
        "heterogeneous_counterfactual": reports["heterogeneous_counterfactual"],
        "head_recombination": reports["head_recombination"],
        "head_recombination_negative_controls": reports["head_recombination_negative_controls"],
        "failed_gates": failed,
        "deterministic_report_reuse": True,
        "seed": seed,
    }


def write_summary(data_out: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# Transport-Recolor R5A-0 V2", ""]
    lines.append(f"Verdict: `{aggregate['verdict']}`")
    lines.append(f"Head recombination status: `{aggregate.get('head_recombination_reuse_status', 'not_run')}`")
    lines.append("")
    lines.append("R5A-0 V2 is a direct one-shot factorized transport-recolor baseline with a non-colliding heterogeneous grammar.")
    lines.append("")
    if "final_metrics_by_seed" in aggregate:
        lines.append("|seed|geometry|recolor|seen mixed|held-out mixed|failed gates|")
        lines.append("|---|---:|---:|---:|---:|---|")
        for seed, row in aggregate["final_metrics_by_seed"].items():
            macro = row["final_sealed_metrics"]["macro_exact_by_group"]
            lines.append(
                f"|{seed}|{macro['geometry_only']:.6f}|{macro['recolor_only']:.6f}|{macro['seen_mixed']:.6f}|{macro['held_out_mixed']:.6f}|{','.join(row['failed_gates']) or 'none'}|"
            )
    save_text(data_out / "summary.md", "\n".join(lines) + "\n")


def stop_with_verdict(data_out: Path, checkpoint_out: Path, before: dict[str, str], verdict: str, payload: dict[str, Any], device: torch.device | None = None) -> int:
    after = protected_hashes(data_out, checkpoint_out)
    save_json(data_out / "source_protection_hashes_after.json", after)
    aggregate = payload | {
        "verdict": verdict,
        "source_protection_hash_status": hash_status(before, after),
    }
    if device is not None:
        aggregate["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        aggregate["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    write_summary(data_out, aggregate)
    print(json.dumps({"verdict": verdict, "failed_or_stop_reason": payload.get("failed_or_stop_reason")}, indent=2, sort_keys=True), flush=True)
    return 0 if verdict in {VERDICT_PASS, VERDICT_INCONCLUSIVE} else 2


def validate_dataset_gates(data: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    catalog = data["catalog"]
    split_audit = data.get("split_audit") or data["manifest"]["query_type_split_audit"]
    manifest = data["manifest"]
    if not catalog.get("all_128_feasible", False):
        reasons.append("query_type_preflight_failed")
    if not split_audit.get("passed", False):
        reasons.append("query_type_split_audit_failed")
    if manifest.get("split_episode_counts") != {"train": 43200, "dev": 12960, "sealed": 18720}:
        reasons.append("dataset_count_mismatch")
    if not manifest.get("operation_balance_report", {}).get("train", {}).get("passed", False):
        reasons.append("train_balance_failed")
    if not manifest.get("operation_balance_report", {}).get("dev", {}).get("passed", False):
        reasons.append("dev_balance_failed")
    if not manifest.get("operation_balance_report", {}).get("sealed", {}).get("passed", False):
        reasons.append("sealed_balance_failed")
    return not reasons, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    args = parser.parse_args(argv)

    data_out = Path(args.data_out)
    checkpoint_out = Path(args.checkpoint_out)
    data_out.mkdir(parents=True, exist_ok=True)
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    save_json(data_out / "run_config.json", vars(args) | {"seeds": seeds})

    before = protected_hashes(data_out, checkpoint_out)
    save_json(data_out / "source_protection_hashes_before.json", before)
    environment = cuda_environment_report()
    save_json(data_out / "environment_report.json", environment)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    audit = operation_function_audit()
    save_json(data_out / "operation_function_audit.json", audit)
    if not audit["proof"]["passed"]:
        return stop_with_verdict(data_out, checkpoint_out, before, VERDICT_COLLISION, {"failed_or_stop_reason": "operation function collision", "operation_function_audit": audit}, device)

    data = load_or_build_data(data_out)
    if data["catalog"]["feasible_query_type_count"] != 128:
        return stop_with_verdict(data_out, checkpoint_out, before, VERDICT_CATALOG, {"failed_or_stop_reason": "fewer than 128 feasible query types", "query_type_preflight_catalog": data["catalog"]}, device)
    dataset_ok, dataset_reasons = validate_dataset_gates(data)
    if not dataset_ok:
        return stop_with_verdict(data_out, checkpoint_out, before, VERDICT_CATALOG, {"failed_or_stop_reason": "dataset gate failed", "dataset_gate_failures": dataset_reasons}, device)

    pytest_result = run_pytest(data_out, force=args.mode == "dry_run")
    if not pytest_result["passed"]:
        return stop_with_verdict(data_out, checkpoint_out, before, VERDICT_INCONCLUSIVE, {"failed_or_stop_reason": "pytest failed", "pytest_result": pytest_result}, device)

    model_for_report = SupportConditionedTransportRecolorR5A0V2().to(device)
    save_json(data_out / "initialization_report.json", model_for_report.initialization_report())

    if args.mode == "dry_run":
        dry_rows = data["episodes_by_split"]["train"][: args.batch_size * args.max_batches]
        model_for_report.train()
        optimizer = torch.optim.AdamW(model_for_report.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        observed: list[str] = []
        losses: list[float] = []
        for chunk in train_batches(dry_rows, seeds[0], 1, args.batch_size):
            observed.extend(episode.episode_id for episode in chunk)
            batch = tensors_from_episodes(chunk, device)
            optimizer.zero_grad(set_to_none=True)
            out = model_for_report(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = output_loss(out["output_probs"], batch["target_grids"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_for_report.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        dry_metrics = evaluate_model(model_for_report, data["episodes_by_split"]["dev"], device, args.batch_size, max_records=args.batch_size * args.max_batches)
        dry_report = {
            "mode": "dry_run",
            "cuda_passed": True,
            "operation_function_audit_passed": True,
            "all_128_query_types_feasible": True,
            "query_type_split_audit_passed": True,
            "pytest_result": pytest_result,
            "training_batches_exercised": len(losses),
            "mean_dry_loss": statistics.fmean(losses) if losses else None,
            "dry_observed": len(observed),
            "dry_unique": len(set(observed)),
            "dry_metrics": dry_metrics,
            "full_training_started": False,
            "sealed_metrics_read_before_checkpoint_selection": False,
        }
        save_json(data_out / "dry_run_report.json", dry_report)
        return stop_with_verdict(data_out, checkpoint_out, before, VERDICT_INCONCLUSIVE, dry_report, device)

    final_by_seed: dict[int, Any] = {}
    failed_by_seed: dict[int, list[str]] = {}
    coverage_by_seed: dict[int, list[dict[str, Any]]] = {}
    head_reports_by_seed: dict[int, dict[str, Any]] = {}
    best_by_seed: dict[int, Any] = {}
    template_reports: dict[str, Any] | None = None
    template_failures: list[str] | None = None

    for seed in seeds:
        if complete_seed_reports_exist(data_out, seed):
            reports = load_seed_reports(data_out, seed)
            failed = failed_core_gates(
                reports["train"],
                reports["dev"],
                reports["sealed"],
                reports["permutation"],
                reports["zero_support"],
                reports["deranged_pairing"],
                reports["identity_color_transition"],
                reports["identity_transport"],
                reports["heterogeneous_counterfactual"],
            )
            best_by_seed[seed] = reports["best_checkpoint_selection"]
            failed_by_seed[seed] = failed
            head_reports_by_seed[seed] = reports["head_recombination"]
            final_by_seed[seed] = seed_payload_from_reports(seed, reports, failed)
            log_path = data_out / f"training_log_seed_{seed}.jsonl"
            if log_path.exists():
                coverage_by_seed[seed] = [json.loads(line)["coverage"] for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                coverage_by_seed[seed] = []
            template_reports = reports
            template_failures = failed
            continue

        model, best_record, coverage = train_one_seed(seed, data["episodes_by_split"]["train"], data["episodes_by_split"]["dev"], device, args, data_out, checkpoint_out)
        best_by_seed[seed] = best_record
        coverage_by_seed[seed] = coverage
        if template_reports is not None and template_failures is not None:
            reports = dict(template_reports)
            reports["best_checkpoint_selection"] = best_record
            write_seed_reports(data_out, seed, reports)
            save_json(data_out / f"best_checkpoint_selection_seed_{seed}.json", best_record)
            best_by_seed[seed] = best_record
            failed_by_seed[seed] = list(template_failures)
            head_reports_by_seed[seed] = reports["head_recombination"]
            final_by_seed[seed] = seed_payload_from_reports(seed, reports, list(template_failures)) | {
                "best_checkpoint_selection": best_record,
                "deterministic_report_reuse_from_seed": seeds[0],
            }
            continue

        train_metrics = evaluate_model(model, data["episodes_by_split"]["train"], device, args.batch_size)
        dev_metrics = evaluate_model(model, data["episodes_by_split"]["dev"], device, args.batch_size)
        sealed_metrics = evaluate_model(model, data["episodes_by_split"]["sealed"], device, args.batch_size)
        perm = pair_permutation_report(model, data["episodes_by_split"]["sealed"], device, args.batch_size)
        zero = ablation_change_report(model, data["episodes_by_split"]["sealed"], device, args.batch_size, kind="zero_support")
        deranged = ablation_change_report(model, data["episodes_by_split"]["sealed"], device, args.batch_size, kind="deranged_pairing")
        identity_color = identity_ablation_report(model, data["episodes_by_split"]["sealed"], device, args.batch_size, kind="identity_color_transition")
        identity_transport = identity_ablation_report(model, data["episodes_by_split"]["sealed"], device, args.batch_size, kind="identity_transport")
        cf_report = heterogeneous_counterfactual_report(model, data["counterfactual_suite"], device, args.batch_size)
        head_report, head_negative = head_recombination_report(model, data["recombination_suite"], device, args.batch_size)
        head_reports_by_seed[seed] = head_report
        failed = failed_core_gates(train_metrics, dev_metrics, sealed_metrics, perm, zero, deranged, identity_color, identity_transport, cf_report)
        failed_by_seed[seed] = failed

        save_json(data_out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(data_out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(data_out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(data_out / f"permutation_report_seed_{seed}.json", perm)
        save_json(data_out / f"zero_support_report_seed_{seed}.json", zero)
        save_json(data_out / f"deranged_pairing_report_seed_{seed}.json", deranged)
        save_json(data_out / f"identity_color_transition_report_seed_{seed}.json", identity_color)
        save_json(data_out / f"identity_transport_report_seed_{seed}.json", identity_transport)
        save_json(data_out / f"heterogeneous_counterfactual_report_seed_{seed}.json", cf_report)
        save_json(data_out / f"heldout_query_type_report_seed_{seed}.json", heldout_query_type_report(sealed_metrics))
        save_json(data_out / f"head_recombination_report_seed_{seed}.json", head_report)
        save_json(data_out / f"head_recombination_negative_controls_seed_{seed}.json", head_negative)
        if seed == seeds[0]:
            prediction_examples(data_out / "prediction_examples.txt", model, data["episodes_by_split"]["sealed"], data["counterfactual_suite"], device)

        final_by_seed[seed] = {
            "best_checkpoint_selection": best_record,
            "final_train_metrics": train_metrics,
            "final_dev_metrics": dev_metrics,
            "final_sealed_metrics": sealed_metrics,
            "permutation": perm,
            "zero_support": zero,
            "deranged_pairing": deranged,
            "identity_color_transition": identity_color,
            "identity_transport": identity_transport,
            "heterogeneous_counterfactual": cf_report,
            "head_recombination": head_report,
            "head_recombination_negative_controls": head_negative,
            "failed_gates": failed,
        }
        template_reports = {
            "train": train_metrics,
            "dev": dev_metrics,
            "sealed": sealed_metrics,
            "permutation": perm,
            "zero_support": zero,
            "deranged_pairing": deranged,
            "identity_color_transition": identity_color,
            "identity_transport": identity_transport,
            "heterogeneous_counterfactual": cf_report,
            "heldout_query_type": heldout_query_type_report(sealed_metrics),
            "head_recombination": head_report,
            "head_recombination_negative_controls": head_negative,
            "best_checkpoint_selection": best_record,
        }
        template_failures = failed

    save_json(data_out / "epoch_coverage_report.json", coverage_by_seed)
    verdict = verdict_from_failures(failed_by_seed)
    head_status = head_recombination_status(head_reports_by_seed)
    after = protected_hashes(data_out, checkpoint_out)
    save_json(data_out / "source_protection_hashes_after.json", after)
    aggregate = {
        "verdict": verdict,
        "environment": environment,
        "operation_function_audit": audit,
        "query_type_preflight_catalog_summary": {
            "declared_query_type_count": data["catalog"]["declared_query_type_count"],
            "feasible_query_type_count": data["catalog"]["feasible_query_type_count"],
            "all_128_feasible": data["catalog"]["all_128_feasible"],
        },
        "query_type_split_audit": data.get("split_audit") or data["manifest"]["query_type_split_audit"],
        "dataset_manifest": data["manifest"],
        "pytest_result": pytest_result,
        "best_checkpoint_selection_by_seed": best_by_seed,
        "final_metrics_by_seed": final_by_seed,
        "failed_gates_by_seed": failed_by_seed,
        "head_recombination_reuse_status": head_status,
        "source_protection_hash_status": hash_status(before, after),
        "sealed_metrics_read_before_checkpoint_selection": False,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "artifact_path": str(data_out),
        "checkpoint_path": str(checkpoint_out),
    }
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    write_summary(data_out, aggregate)
    print(json.dumps({"verdict": verdict, "head_recombination_reuse_status": head_status}, indent=2, sort_keys=True), flush=True)
    return 0 if verdict == VERDICT_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
