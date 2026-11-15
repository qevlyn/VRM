from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT

from .composition_r4a0r_data import (
    ALL_SEQUENCE_CLASSES,
    HELD_OUT_TWO_STEP_CLASSES,
    ORDER_COUNTERFACTUAL_CLASSES,
    REQUIRED_QUERY_TYPE_COUNTS,
    SEALED_CLASSES,
    SEEN_TWO_STEP_CLASSES,
    SINGLE_STEP_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    R4A0REpisode,
    algebraic_function_audit,
    apply_sequence_to_grid,
    counterfactual_suite_manifest,
    derange_episode,
    factor_coverage,
    foreground_coords,
    generate_order_counterfactual_suite,
    grid_to_list,
    load_order_counterfactual_suite,
    load_split,
    partition_query_types,
    query_type_key,
    query_type_preflight_catalog,
    query_type_split_audit,
    reorder_episode,
    save_dataset,
    sequence_balance_report,
    source_index_for_output,
    stable_hash,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/composition_r4a0r_v2"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/composition_r4a0r_v2"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)
FAILED_V1_OUT = PROJECT_ROOT / "artifacts/composition_r4a0r_v1"

PROTECTED_FILES = (
    PROJECT_ROOT / "vrm/model.py",
    PROJECT_ROOT / "vrm/data.py",
    PROJECT_ROOT / "vrm/metrics.py",
    PROJECT_ROOT / "vrm/train.py",
    PROJECT_ROOT / "vrm/twin_tasks/pair_reader_r1.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_pair_reader_r1.py",
    PROJECT_ROOT / "vrm/twin_tasks/pair_reader_r1b.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_pair_reader_r1b.py",
    PROJECT_ROOT / "vrm/twin_tasks/vrm_pair_reader_r2a.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_vrm_pair_reader_r2a.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3a_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3a_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3a_reader.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3a_router.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_mixed_router_r3a.py",
    PROJECT_ROOT / "vrm/twin_tasks/transform_r3b0_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transform_r3b0_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/transform_r3b0_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transform_r3b0.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3b2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/mixed_router_r3b2_gate.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_mixed_router_r3b2.py",
)

PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "archives",
    PROJECT_ROOT / "arc-prize-2026-arc-agi-2",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
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
    return (
        resolved == data_out.resolve()
        or data_out.resolve() in resolved.parents
        or resolved == checkpoint_out.resolve()
        or checkpoint_out.resolve() in resolved.parents
    )


def protected_hashes(data_out: Path, checkpoint_out: Path) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    seen: set[str] = set()
    for path in PROTECTED_FILES:
        key = str(path)
        seen.add(key)
        hashes[key] = sha256_file(path) if path.exists() else None
    for directory in PROTECTED_DIRS:
        if not directory.exists():
            hashes[str(directory)] = None
            continue
        for path in sorted(child for child in directory.rglob("*") if child.is_file()):
            if is_new_output_path(path, data_out, checkpoint_out):
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            key = str(path)
            if key not in seen:
                hashes[key] = sha256_file(path)
                seen.add(key)
    return hashes


def hash_status(before: dict[str, str | None], after: dict[str, str | None]) -> dict[str, Any]:
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


def configure_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class SupportConditionedCompositePointerR4A0R(nn.Module):
    """Support-conditioned query-copy pointer from observed pair correspondence."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(40.0), dtype=torch.float32))

    @property
    def cell_count(self) -> int:
        return 49

    def encode_pairs(self, support_input_grids: torch.Tensor, support_output_grids: torch.Tensor) -> torch.Tensor:
        batch, pairs, height, width = support_input_grids.shape
        flat_inputs = support_input_grids.reshape(batch * pairs, height * width)
        flat_outputs = support_output_grids.reshape(batch * pairs, height * width)
        input_one_hot = F.one_hot(flat_inputs.long(), num_classes=ARC_COLOR_COUNT).to(torch.float32)[..., 1:]
        output_one_hot = F.one_hot(flat_outputs.long(), num_classes=ARC_COLOR_COUNT).to(torch.float32)[..., 1:]
        relation = torch.einsum("btc,bsc->bts", output_one_hot, input_one_hot)
        relation = relation.reshape(batch, pairs, self.cell_count * self.cell_count)
        return relation

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        *,
        zero_support: bool = False,
    ) -> dict[str, torch.Tensor]:
        pair_representations = self.encode_pairs(support_input_grids, support_output_grids)
        if zero_support:
            context_flat = torch.zeros_like(pair_representations).mean(dim=1)
        else:
            context_flat = pair_representations.mean(dim=1)
        task_context = context_flat
        pointer_logits = context_flat.reshape(-1, self.cell_count, self.cell_count) * self.logit_scale.exp().clamp(1.0, 120.0)
        pointer_attention = torch.softmax(pointer_logits, dim=-1)
        query_one_hot = F.one_hot(query_grids.reshape(query_grids.shape[0], self.cell_count).long(), num_classes=ARC_COLOR_COUNT).to(torch.float32)
        output_probs = torch.einsum("buv,bvc->buc", pointer_attention, query_one_hot)
        output_probs = output_probs.reshape(query_grids.shape[0], 7, 7, ARC_COLOR_COUNT)
        return {
            "output_probs": output_probs,
            "decoded_grid": output_probs.argmax(dim=-1),
            "task_context": task_context,
            "pair_representations": pair_representations,
            "pointer_attention": pointer_attention.reshape(query_grids.shape[0], 7, 7, self.cell_count),
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "SupportConditionedCompositePointerR4A0R",
            "operation_labels_in_forward": False,
            "sequence_tokens": False,
            "intermediate_outputs": False,
            "paint_head_bypass": False,
            "pair_id_embeddings": False,
            "pair_position_embeddings": False,
            "query_type_or_target_metadata": False,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "initial_pointer_logit_scale": float(self.logit_scale.exp().detach().cpu().item()),
        }


def output_loss(output_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = output_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def save_checkpoint(path: Path, model: SupportConditionedCompositePointerR4A0R, optimizer: torch.optim.Optimizer, seed: int, epoch: int, step: int, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "epoch_index": epoch,
            "global_step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> SupportConditionedCompositePointerR4A0R:
    model = SupportConditionedCompositePointerR4A0R().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def batches(order: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def support_order(episode_id: str, seed: int, epoch: int) -> tuple[int, int, int]:
    perms = list(itertools.permutations(range(3)))
    return tuple(perms[int(hashlib.sha256(f"{seed}|{epoch}|{episode_id}".encode()).hexdigest()[:8], 16) % len(perms)])


def training_view(episode: R4A0REpisode, seed: int, epoch: int) -> R4A0REpisode:
    return reorder_episode(episode, support_order(episode.episode_id, seed, epoch))


def exact_mask(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (decoded == target).flatten(1).all(dim=1)


def sequence_group(sequence: str) -> str:
    if sequence in SINGLE_STEP_CLASSES:
        return "singles"
    if sequence in SEEN_TWO_STEP_CLASSES:
        return "seen_pairs"
    return "heldout_pairs"


def metric_template() -> dict[str, Any]:
    return {
        "count": 0,
        "exact": 0,
        "fg_correct": 0,
        "fg_total": 0,
        "all_correct": 0,
        "all_total": 0,
        "hamming": 0.0,
        "pointer_correct": 0,
        "pointer_total": 0,
        "pointer_entropy": [],
        "pointer_max": [],
        "context_norm": [],
        "context_mean": [],
        "context_std": [],
    }


def finalize_metric(row: dict[str, Any]) -> dict[str, Any]:
    count = max(row["count"], 1)
    return {
        "episode_count": row["count"],
        "direct_grid_exact_accuracy": row["exact"] / count,
        "foreground_color_accuracy": row["fg_correct"] / max(row["fg_total"], 1),
        "all_cell_color_accuracy": row["all_correct"] / max(row["all_total"], 1),
        "hamming_error": row["hamming"] / count,
        "pointer_source_position_accuracy": row["pointer_correct"] / max(row["pointer_total"], 1),
        "mean_pointer_entropy": statistics.fmean(row["pointer_entropy"]) if row["pointer_entropy"] else 0.0,
        "mean_max_pointer_probability": statistics.fmean(row["pointer_max"]) if row["pointer_max"] else 0.0,
        "task_context_norm": statistics.fmean(row["context_norm"]) if row["context_norm"] else 0.0,
        "task_context_mean": statistics.fmean(row["context_mean"]) if row["context_mean"] else 0.0,
        "task_context_std": statistics.fmean(row["context_std"]) if row["context_std"] else 0.0,
    }


def evaluate_model(
    model: SupportConditionedCompositePointerR4A0R,
    episodes: list[R4A0REpisode],
    device: torch.device,
    batch_size: int,
    *,
    zero_support: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = {sequence: metric_template() for sequence in sorted({episode.sequence_class for episode in episodes})}
    group_rows = {group: metric_template() for group in ("singles", "seen_pairs", "heldout_pairs")}
    details: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=zero_support)
            decoded = outputs["decoded_grid"]
            target = batch["target_grids"]
            exact = exact_mask(decoded, target)
            pointer = outputs["pointer_attention"].reshape(len(chunk), 49, 49)
            pointer_argmax = pointer.argmax(dim=-1)
            pointer_entropy = -(pointer.clamp_min(1e-8) * torch.log(pointer.clamp_min(1e-8))).sum(dim=-1).reshape(len(chunk), 7, 7)
            pointer_max = pointer.max(dim=-1).values.reshape(len(chunk), 7, 7)
            context = outputs["task_context"]
            for index, episode in enumerate(chunk):
                fg = target[index] != 0
                true_sources = []
                pred_sources = []
                for y in range(7):
                    for x in range(7):
                        if bool(fg[y, x].item()):
                            src = source_index_for_output(episode.sequence_class, x, y)
                            if src >= 0:
                                true_sources.append(src)
                                pred_sources.append(int(pointer_argmax[index, y * 7 + x].item()))
                for row in (rows[episode.sequence_class], group_rows[sequence_group(episode.sequence_class)]):
                    row["count"] += 1
                    row["exact"] += int(bool(exact[index].item()))
                    row["fg_total"] += int(fg.sum().item())
                    row["fg_correct"] += int(((decoded[index] == target[index]) & fg).sum().item())
                    row["all_total"] += 49
                    row["all_correct"] += int((decoded[index] == target[index]).sum().item())
                    row["hamming"] += float((decoded[index] != target[index]).sum().item())
                    row["pointer_total"] += len(true_sources)
                    row["pointer_correct"] += sum(int(a == b) for a, b in zip(pred_sources, true_sources))
                    row["pointer_entropy"].append(float(pointer_entropy[index].mean().item()))
                    row["pointer_max"].append(float(pointer_max[index].mean().item()))
                    row["context_norm"].append(float(context[index].norm().item()))
                    row["context_mean"].append(float(context[index].mean().item()))
                    row["context_std"].append(float(context[index].std(unbiased=False).item()))
                details.append(
                    {
                        "episode_id": episode.episode_id,
                        "sequence_class": episode.sequence_class,
                        "query_type_key": query_type_key(episode.query_type),
                        "exact": bool(exact[index].item()),
                        "decoded_hash": stable_hash(decoded[index].detach().cpu().long().tolist()),
                        "target_hash": stable_hash(target[index].detach().cpu().long().tolist()),
                    }
                )
    metrics = {
        "by_sequence": {sequence: finalize_metric(row) for sequence, row in rows.items()},
        "by_group": {group: finalize_metric(row) for group, row in group_rows.items() if row["count"] > 0},
    }
    heldout = [metrics["by_sequence"][sequence]["direct_grid_exact_accuracy"] for sequence in HELD_OUT_TWO_STEP_CLASSES if sequence in metrics["by_sequence"]]
    metrics["heldout_composition_macro_exact"] = statistics.fmean(heldout) if heldout else 0.0
    visible = [metrics["by_sequence"][sequence]["direct_grid_exact_accuracy"] for sequence in TRAIN_VISIBLE_CLASSES if sequence in metrics["by_sequence"]]
    metrics["train_visible_macro_exact"] = statistics.fmean(visible) if visible else 0.0
    return metrics, details


def train_seed(seed: int, train: list[R4A0REpisode], dev: list[R4A0REpisode], device: torch.device, args: argparse.Namespace, data_out: Path, checkpoint_out: Path) -> dict[str, Any]:
    configure_seed(seed)
    model = SupportConditionedCompositePointerR4A0R().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    save_checkpoint(checkpoint_out / f"composition_r4a0r_seed_{seed}_initial.pt", model, optimizer, seed, -1, 0)
    log_path = data_out / f"training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(args.epochs):
        order = epoch_order(len(train), seed, epoch)
        observed: list[str] = []
        losses: list[float] = []
        model.train()
        for batch_indices in batches(order, args.batch_size):
            source = [train[index] for index in batch_indices]
            view = [training_view(episode, seed, epoch) for episode in source]
            observed.extend(episode.episode_id for episode in source)
            batch = tensors_from_episodes(view, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = output_loss(outputs["output_probs"], batch["target_grids"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
        unique = len(set(observed))
        cov = {
            "seed": seed,
            "epoch_index": epoch,
            "observed": len(observed),
            "unique": unique,
            "duplicates": len(observed) - unique,
            "missing": len(train) - unique,
            "full_coverage": len(observed) == len(train) and unique == len(train),
        }
        coverage.append(cov)
        dev_metrics, _ = evaluate_model(model, dev, device, args.batch_size)
        cf_score = dev_metrics["by_group"]["seen_pairs"]["direct_grid_exact_accuracy"]
        row = {
            "seed": seed,
            "epoch_index": epoch,
            "global_step": global_step,
            "mean_loss": statistics.fmean(losses),
            "coverage": cov,
            "dev_metrics": dev_metrics,
            "dev_fixed_order_counterfactual_score": cf_score,
        }
        rows.append(row)
        append_jsonl(log_path, row)
        save_checkpoint(checkpoint_out / f"composition_r4a0r_seed_{seed}_epoch_{epoch:02d}.pt", model, optimizer, seed, epoch, global_step, {"dev_metrics": dev_metrics})
    save_json(data_out / f"dev_metrics_by_epoch_seed_{seed}.json", rows)

    def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
        metrics = row["dev_metrics"]
        seen = metrics["by_group"]["seen_pairs"]["direct_grid_exact_accuracy"]
        visible = metrics["train_visible_macro_exact"]
        cf = row["dev_fixed_order_counterfactual_score"]
        hamming = statistics.fmean(metrics["by_sequence"][seq]["hamming_error"] for seq in TRAIN_VISIBLE_CLASSES)
        return (seen, visible, cf, -hamming, -row["epoch_index"])

    best = max(rows, key=selection_key)
    source_path = checkpoint_out / f"composition_r4a0r_seed_{seed}_epoch_{best['epoch_index']:02d}.pt"
    best_path = checkpoint_out / f"composition_r4a0r_seed_{seed}_best_dev.pt"
    payload = torch.load(source_path, map_location=device, weights_only=False)
    torch.save(payload, best_path)
    selection = {
        "seed": seed,
        "selected_epoch": best["epoch_index"],
        "selected_global_step": best["global_step"],
        "best_checkpoint": best_path,
        "selection_order": [
            "dev exact on seen two-step sequences",
            "dev macro exact on all train-visible classes",
            "dev fixed order-counterfactual score",
            "lowest dev hamming error",
            "earliest epoch",
        ],
        "selected_dev_metrics": best["dev_metrics"],
    }
    save_json(data_out / f"best_checkpoint_selection_seed_{seed}.json", selection)
    return {"selection": selection, "coverage": coverage}


def permutation_report(model: SupportConditionedCompositePointerR4A0R, episodes: list[R4A0REpisode], device: torch.device, batch_size: int) -> dict[str, Any]:
    perms = list(itertools.permutations(range(3)))
    same_grid = 0
    total = 0
    max_prob = 0.0
    max_context = 0.0
    max_pointer = 0.0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            base_batch = tensors_from_episodes(chunk, device)
            base = model(base_batch["query_grids"], base_batch["support_input_grids"], base_batch["support_output_grids"])
            for order in perms:
                views = [reorder_episode(episode, order) for episode in chunk]
                batch = tensors_from_episodes(views, device)
                out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                total += len(chunk)
                same_grid += int((out["decoded_grid"] == base["decoded_grid"]).flatten(1).all(dim=1).sum().item())
                max_prob = max(max_prob, float((out["output_probs"] - base["output_probs"]).abs().max().item()))
                max_context = max(max_context, float((out["task_context"] - base["task_context"]).abs().max().item()))
                max_pointer = max(max_pointer, float((out["pointer_attention"] - base["pointer_attention"]).abs().max().item()))
    return {
        "presentations": total,
        "same_final_grid_rate": same_grid / max(total, 1),
        "max_output_probability_difference": max_prob,
        "max_task_context_difference": max_context,
        "max_pointer_attention_difference": max_pointer,
        "passed": same_grid / max(total, 1) >= 0.999 and max(max_prob, max_context, max_pointer) <= 1e-5,
    }


def ablation_report(model: SupportConditionedCompositePointerR4A0R, episodes: list[R4A0REpisode], device: torch.device, batch_size: int, *, kind: str) -> dict[str, Any]:
    rows = {group: {"count": 0, "exact": 0, "grid_change": 0} for group in ("singles", "seen_pairs", "heldout_pairs")}
    shifts = (0,) if kind == "zero_support" else (1, 2)
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
                target = normal_batch["target_grids"]
                exact = exact_mask(out["decoded_grid"], target)
                for index, episode in enumerate(chunk):
                    row = rows[sequence_group(episode.sequence_class)]
                    row["count"] += 1
                    row["exact"] += int(bool(exact[index].item()))
                    row["grid_change"] += int(not torch.equal(out["decoded_grid"][index], normal["decoded_grid"][index]))
    return {
        "ablation": kind,
        "groups": {
            group: {
                "episode_count": row["count"],
                "exact_accuracy": row["exact"] / max(row["count"], 1),
                "normal_to_ablation_grid_change_rate": row["grid_change"] / max(row["count"], 1),
            }
            for group, row in rows.items()
        },
    }


def order_counterfactual_report(model: SupportConditionedCompositePointerR4A0R, episodes: list[R4A0REpisode], device: torch.device, batch_size: int) -> dict[str, Any]:
    metrics, details = evaluate_model(model, episodes, device, batch_size)
    by_group: dict[str, list[dict[str, Any]]] = {}
    by_id = {episode.episode_id: episode for episode in episodes}
    for detail in details:
        by_group.setdefault(str(by_id[detail["episode_id"]].counterfactual_group_id), []).append(detail)
    switch_correct = 0
    sensitive = 0
    pair_rates = {"C->F_vs_F->C": [], "F->T_vs_T->F": [], "T->C_vs_C->T": []}
    total = 0
    for rows in by_group.values():
        mapping = {row["sequence_class"]: row for row in rows}
        if not all(sequence in mapping for sequence in ORDER_COUNTERFACTUAL_CLASSES):
            continue
        total += 1
        switch_correct += int(all(mapping[sequence]["exact"] for sequence in ORDER_COUNTERFACTUAL_CLASSES))
        sensitive += int(len({mapping[sequence]["decoded_hash"] for sequence in ORDER_COUNTERFACTUAL_CLASSES}) == len(ORDER_COUNTERFACTUAL_CLASSES))
        pair_rates["C->F_vs_F->C"].append(float(mapping["C->F"]["decoded_hash"] != mapping["F->C"]["decoded_hash"]))
        pair_rates["F->T_vs_T->F"].append(float(mapping["F->T"]["decoded_hash"] != mapping["T->F"]["decoded_hash"]))
        pair_rates["T->C_vs_C->T"].append(float(mapping["T->C"]["decoded_hash"] != mapping["C->T"]["decoded_hash"]))
    return {
        "exact_by_sequence_context": {sequence: metrics["by_sequence"][sequence]["direct_grid_exact_accuracy"] for sequence in ORDER_COUNTERFACTUAL_CLASSES},
        "ordered_rule_switch_correct_rate": switch_correct / max(total, 1),
        "same_query_support_sensitive_output_rate": sensitive / max(total, 1),
        "pairwise_output_switch_rates": {key: statistics.fmean(values) if values else 0.0 for key, values in pair_rates.items()},
        "group_count": total,
    }


def heldout_query_type_report(details: list[dict[str, Any]], episodes: list[R4A0REpisode]) -> dict[str, Any]:
    by_episode = {episode.episode_id: episode for episode in episodes}
    rows: dict[str, dict[str, Any]] = {}
    for detail in details:
        episode = by_episode[detail["episode_id"]]
        key = query_type_key(episode.query_type)
        row = rows.setdefault(key, {"count": 0, "exact": 0, "sequences": set()})
        row["count"] += 1
        row["exact"] += int(detail["exact"])
        row["sequences"].add(episode.sequence_class)
    finalized = {
        key: {
            "episode_count": row["count"],
            "direct_grid_exact_accuracy": row["exact"] / max(row["count"], 1),
            "sequences": sorted(row["sequences"]),
        }
        for key, row in rows.items()
    }
    return {
        "query_type_count": len(finalized),
        "aggregate_exact": statistics.fmean(row["direct_grid_exact_accuracy"] for row in finalized.values()) if finalized else 0.0,
        "worst_query_type_exact": min(row["direct_grid_exact_accuracy"] for row in finalized.values()) if finalized else 0.0,
        "query_types": finalized,
    }


def run_pytest(data_out: Path) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    save_text(data_out / "pytest_result.txt", result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:]}


def failed_gates(sealed_metrics: dict[str, Any], heldout: dict[str, Any], perm: dict[str, Any], zero: dict[str, Any], deranged: dict[str, Any], cf: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if sealed_metrics["by_group"]["singles"]["direct_grid_exact_accuracy"] < 0.99:
        failed.append("single_step_macro")
    if sealed_metrics["by_group"]["seen_pairs"]["direct_grid_exact_accuracy"] < 0.99:
        failed.append("seen_two_step_macro")
    for sequence in TRAIN_VISIBLE_CLASSES:
        if sealed_metrics["by_sequence"][sequence]["direct_grid_exact_accuracy"] < 0.98:
            failed.append(f"visible_sequence_{sequence}")
    if sealed_metrics["heldout_composition_macro_exact"] < 0.99:
        failed.append("heldout_pair_macro")
    for sequence in HELD_OUT_TWO_STEP_CLASSES:
        if sealed_metrics["by_sequence"][sequence]["direct_grid_exact_accuracy"] < 0.98:
            failed.append(f"heldout_sequence_{sequence}")
    if heldout["aggregate_exact"] < 0.99 or heldout["worst_query_type_exact"] < 0.98:
        failed.append("heldout_visual_query_type")
    if not perm["passed"]:
        failed.append("pair_permutation")
    for name, report in (("zero_support", zero), ("deranged_pairing", deranged)):
        for group in ("seen_pairs", "heldout_pairs"):
            row = report["groups"][group]
            if row["normal_to_ablation_grid_change_rate"] < 0.95 or row["exact_accuracy"] > 0.30:
                failed.append(f"{name}_{group}")
    if min(cf["exact_by_sequence_context"].values()) < 0.99 or cf["ordered_rule_switch_correct_rate"] < 0.99 or cf["same_query_support_sensitive_output_rate"] < 0.99:
        failed.append("order_counterfactual")
    return failed


def verdict_from_failures(failed_by_seed: dict[int, list[str]]) -> str:
    flat = {item for rows in failed_by_seed.values() for item in rows}
    if not flat:
        return "R4A0R_SYSTEMATIC_COMPOSITION_PASS"
    if any(item.startswith("visible_sequence") or item in {"single_step_macro", "seen_two_step_macro"} for item in flat):
        return "COMPOSITION_EXPERT_NOT_COMPETENT"
    if any(item.startswith("heldout") for item in flat):
        return "COMPOSITION_GENERALIZATION_FAILURE"
    if "pair_permutation" in flat:
        return "PAIR_ORDER_LEAK_OR_REORDER_BUG"
    if any(item.startswith("zero_support") or item.startswith("deranged_pairing") for item in flat):
        return "SUPPORT_NOT_USED"
    if "order_counterfactual" in flat:
        return "ORDER_COUNTERFACTUAL_FAILURE"
    return "INCONCLUSIVE"


def write_prediction_examples(path: Path, episodes: list[R4A0REpisode], model: SupportConditionedCompositePointerR4A0R, device: torch.device) -> None:
    lines: list[str] = []
    examples = []
    for sequence in ALL_SEQUENCE_CLASSES:
        examples.extend([episode for episode in episodes if episode.sequence_class == sequence][:2])
    with torch.no_grad():
        for episode in examples:
            batch = tensors_from_episodes([episode], device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            lines.append(
                json.dumps(
                    {
                        "sequence_class": episode.sequence_class,
                        "query": grid_to_list(episode.query),
                        "support_pairs": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in episode.supports],
                        "prediction": out["decoded_grid"][0].detach().cpu().tolist(),
                        "target": grid_to_list(episode.target),
                        "exact": bool(torch.equal(out["decoded_grid"][0], batch["target_grids"][0])),
                    },
                    sort_keys=True,
                )
            )
    save_text(path, "\n\n".join(lines) + "\n")


def write_summary(data_out: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# Composition R4A-0R v2", ""]
    lines.append(f"Verdict: `{aggregate['verdict']}`")
    lines.append("")
    lines.append("|seed|single exact|seen exact|heldout exact|worst qtype|failed gates|")
    lines.append("|---|---:|---:|---:|---:|---|")
    for seed, row in aggregate["final_metrics_by_seed"].items():
        sealed = row["sealed_metrics"]
        held = row["heldout_query_type_report"]
        lines.append(f"|{seed}|{sealed['by_group']['singles']['direct_grid_exact_accuracy']:.6f}|{sealed['by_group']['seen_pairs']['direct_grid_exact_accuracy']:.6f}|{sealed['heldout_composition_macro_exact']:.6f}|{held['worst_query_type_exact']:.6f}|{','.join(row['failed_gates']) or 'none'}|")
    save_text(data_out / "summary.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    args = parser.parse_args(argv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    data_out = Path(args.data_out)
    checkpoint_out = Path(args.checkpoint_out)
    data_out.mkdir(parents=True, exist_ok=True)
    checkpoint_out.mkdir(parents=True, exist_ok=True)

    before = protected_hashes(data_out, checkpoint_out)
    v1_before = protected_hashes(FAILED_V1_OUT / "__not_v2__", checkpoint_out).get(str(FAILED_V1_OUT), None)
    save_json(data_out / "source_protection_hashes_before.json", before)
    environment = cuda_environment_report()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    save_json(data_out / "environment_report.json", environment)
    save_json(data_out / "run_config.json", vars(args) | {"seeds": seeds, "cuda_required": True, "cpu_fallback": False})

    algebra = algebraic_function_audit()
    save_json(data_out / "algebraic_function_audit.json", algebra)
    if not algebra["proof"]["passed"]:
        print("COMPOSITION_FUNCTION_COLLISION_DETECTED", flush=True)
        return 2

    dataset = save_dataset(data_out)
    if dataset["catalog"]["feasible_query_type_count"] != 128:
        after = protected_hashes(data_out, checkpoint_out)
        save_json(data_out / "source_protection_hashes_after.json", after)
        aggregate = {"verdict": "QUERY_TYPE_CATALOG_INSUFFICIENT", "query_type_catalog": dataset["catalog"], "source_protection_hashes": hash_status(before, after)}
        save_json(data_out / "aggregate_seed_summary.json", aggregate)
        print("QUERY_TYPE_CATALOG_INSUFFICIENT", flush=True)
        return 2
    if not dataset["manifest"]["query_type_split_audit"]["passed"]:
        raise RuntimeError("query type split audit failed")

    pytest_result = run_pytest(data_out)
    if not pytest_result["passed"]:
        after = protected_hashes(data_out, checkpoint_out)
        save_json(data_out / "source_protection_hashes_after.json", after)
        aggregate = {"verdict": "INCONCLUSIVE", "pytest_result": pytest_result, "source_protection_hashes": hash_status(before, after)}
        save_json(data_out / "aggregate_seed_summary.json", aggregate)
        print(json.dumps({"verdict": "INCONCLUSIVE", "pytest_passed": False}, indent=2), flush=True)
        return 4

    if args.mode == "dry_run":
        after = protected_hashes(data_out, checkpoint_out)
        report = {
            "mode": "dry_run",
            "cuda_passed": True,
            "algebraic_function_audit_passed": True,
            "query_type_preflight_128_of_128": True,
            "split_audit_passed": dataset["manifest"]["query_type_split_audit"]["passed"],
            "pytest_result": pytest_result,
            "training_started": False,
            "source_protection_hashes": hash_status(before, after),
        }
        save_json(data_out / "dry_run_report.json", report)
        save_json(data_out / "aggregate_seed_summary.json", report | {"verdict": "INCONCLUSIVE"})
        save_json(data_out / "source_protection_hashes_after.json", after)
        print(json.dumps({"dry_run_passed": True, "query_type_preflight": "128/128", "pytest_passed": True}, indent=2, sort_keys=True), flush=True)
        return 0

    train = dataset["episodes_by_split"]["train"]
    dev = dataset["episodes_by_split"]["dev"]
    sealed = dataset["episodes_by_split"]["sealed"]
    cf_suite = dataset["order_counterfactual_suite"]
    coverage_report: dict[int, Any] = {}
    final_by_seed: dict[int, Any] = {}
    failed_by_seed: dict[int, list[str]] = {}

    for seed in seeds:
        train_result = train_seed(seed, train, dev, device, args, data_out, checkpoint_out)
        coverage_report[seed] = train_result["coverage"]
    save_json(data_out / "epoch_coverage_report.json", coverage_report)

    for seed in seeds:
        selection = json.loads((data_out / f"best_checkpoint_selection_seed_{seed}.json").read_text())
        model = load_checkpoint(Path(selection["best_checkpoint"]), device)
        train_metrics, _ = evaluate_model(model, train, device, args.batch_size)
        dev_metrics, _ = evaluate_model(model, dev, device, args.batch_size)
        sealed_metrics, sealed_details = evaluate_model(model, sealed, device, args.batch_size)
        heldout = heldout_query_type_report(sealed_details, sealed)
        perm = permutation_report(model, sealed, device, args.batch_size)
        zero = ablation_report(model, sealed, device, args.batch_size, kind="zero_support")
        deranged = ablation_report(model, sealed, device, args.batch_size, kind="deranged_pairing")
        cf = order_counterfactual_report(model, cf_suite, device, args.batch_size)
        failed = failed_gates(sealed_metrics, heldout, perm, zero, deranged, cf)
        failed_by_seed[seed] = failed
        final_by_seed[seed] = {
            "selection": selection,
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "sealed_metrics": sealed_metrics,
            "heldout_query_type_report": heldout,
            "permutation_report": perm,
            "zero_support_ablation": zero,
            "deranged_pairing_ablation": deranged,
            "order_counterfactual_report": cf,
            "failed_gates": failed,
        }
        save_json(data_out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(data_out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(data_out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(data_out / f"heldout_query_type_report_seed_{seed}.json", heldout)
        save_json(data_out / f"exhaustive_permutation_report_seed_{seed}.json", perm)
        save_json(data_out / f"zero_support_ablation_seed_{seed}.json", zero)
        save_json(data_out / f"deranged_pairing_ablation_seed_{seed}.json", deranged)
        save_json(data_out / f"order_counterfactual_report_seed_{seed}.json", cf)
        if seed == seeds[0]:
            write_prediction_examples(data_out / "prediction_examples.txt", sealed, model, device)

    after = protected_hashes(data_out, checkpoint_out)
    verdict = verdict_from_failures(failed_by_seed)
    aggregate = {
        "verdict": verdict,
        "cuda": environment,
        "algebraic_function_audit_passed": True,
        "query_type_catalog": dataset["catalog"],
        "split_audit": dataset["manifest"]["query_type_split_audit"],
        "pytest_result": pytest_result,
        "coverage": coverage_report,
        "failed_gates_by_seed": failed_by_seed,
        "final_metrics_by_seed": final_by_seed,
        "source_protection_hashes": hash_status(before, after),
        "failed_v1_artifact_directory_preserved": (FAILED_V1_OUT.exists() and hash_status(before, after)["unchanged"]),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    save_json(data_out / "summary_report.json", aggregate)
    save_json(data_out / "source_protection_hashes_after.json", after)
    write_summary(data_out, aggregate)
    print(json.dumps({"verdict": verdict, "failed_gates_by_seed": failed_by_seed}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
