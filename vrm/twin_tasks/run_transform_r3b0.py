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
from typing import Any

import torch

from .transform_r3b0_data import (
    GLOBAL_SEED,
    TRANSFORM_H,
    TRANSFORM_R,
    TRANSFORM_V,
    TRANSFORMS,
    R3B0Episode,
    counterfactual_suite_report,
    derange_episode,
    grid_to_list,
    load_counterfactual_suite,
    load_split,
    query_type_key,
    reorder_episode,
    save_dataset,
    stable_hash,
    support_input_bag,
    support_output_bag,
    support_signature,
    tensors_from_episodes,
    transform_specs,
    true_source_index,
)
from .transform_r3b0_model import SupportConditionedTransformPointerR3B0, training_batch_keys, transform_r3b0_output_loss
from .transform_r3b0_oracles import certify_dataset_transforms

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/transform_r3b0_v1"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/transform_r3b0_v1"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)

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
)

PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts/twin_task_pilot_v1",
    PROJECT_ROOT / "artifacts/twin_vrm_probe_v0",
    PROJECT_ROOT / "artifacts/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1b",
    PROJECT_ROOT / "artifacts/twin_vrm_pair_reader_r2a",
    PROJECT_ROOT / "artifacts/mixed_router_r3a_v1",
    PROJECT_ROOT / "checkpoints/twin_vrm_probe_v0",
    PROJECT_ROOT / "checkpoints/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b",
    PROJECT_ROOT / "checkpoints/twin_vrm_pair_reader_r2a",
    PROJECT_ROOT / "checkpoints/mixed_router_r3a_v1",
    PROJECT_ROOT / "archives",
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


def protected_hashes() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    seen: set[str] = set()
    for path in PROTECTED_FILES:
        key = str(path)
        hashes[key] = sha256_file(path) if path.exists() else None
        seen.add(key)
    for directory in PROTECTED_DIRS:
        if not directory.exists():
            hashes[str(directory)] = None
            continue
        for path in sorted(child for child in directory.rglob("*") if child.is_file()):
            key = str(path)
            if key not in seen:
                hashes[key] = sha256_file(path)
                seen.add(key)
    return hashes


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
    if not available:
        print("CUDA_REQUIRED_NOT_AVAILABLE", flush=True)
        raise SystemExit("CUDA_REQUIRED_NOT_AVAILABLE")
    return report


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def batches(order: list[int], batch_size: int) -> list[list[int]]:
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def support_order(episode_id: str, seed: int, epoch: int) -> tuple[int, int, int]:
    perms = list(itertools.permutations(range(3)))
    return perms[int(hashlib.sha256(f"{seed}|{epoch}|{episode_id}".encode()).hexdigest()[:8], 16) % len(perms)]


def training_view(episode: R3B0Episode, seed: int, epoch: int) -> R3B0Episode:
    return reorder_episode(episode, support_order(episode.episode_id, seed, epoch))


def save_checkpoint(path: Path, model: SupportConditionedTransformPointerR3B0, optimizer: torch.optim.Optimizer, seed: int, epoch: int, step: int, extra: dict[str, Any] | None = None) -> None:
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


def load_checkpoint(path: Path, device: torch.device) -> SupportConditionedTransformPointerR3B0:
    model = SupportConditionedTransformPointerR3B0().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def tensor_grid_to_list(grid: torch.Tensor) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid.detach().cpu().long().tolist()]


def exact_grid(pred: torch.Tensor, target: torch.Tensor) -> bool:
    return bool(torch.equal(pred.detach().cpu(), target.detach().cpu()))


def source_index_tensor(specs: list[str], device: torch.device) -> torch.Tensor:
    rows = []
    for spec in specs:
        rows.append([true_source_index(spec, out_row, out_col) for out_row in range(7) for out_col in range(7)])
    return torch.tensor(rows, dtype=torch.long, device=device)


def family_metric_template() -> dict[str, Any]:
    return {
        "count": 0,
        "exact": 0,
        "fg_correct": 0,
        "fg_total": 0,
        "all_correct": 0,
        "all_total": 0,
        "hamming_sum": 0.0,
        "pointer_max": [],
        "pointer_entropy": [],
        "source_position_correct": 0,
        "source_position_total": 0,
    }


def finalize_transform_metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    final = {}
    for spec, row in rows.items():
        final[spec] = {
            "episode_count": row["count"],
            "direct_grid_exact_accuracy": row["exact"] / max(row["count"], 1),
            "foreground_color_accuracy": row["fg_correct"] / max(row["fg_total"], 1),
            "all_cell_color_accuracy": row["all_correct"] / max(row["all_total"], 1),
            "target_rectangle_hamming_error": row["hamming_sum"] / max(row["count"], 1),
            "mean_max_pointer_probability": statistics.fmean(row["pointer_max"]) if row["pointer_max"] else 0.0,
            "mean_pointer_entropy": statistics.fmean(row["pointer_entropy"]) if row["pointer_entropy"] else 0.0,
            "source_position_accuracy": row["source_position_correct"] / max(row["source_position_total"], 1),
        }
    return final


def normal_metrics(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows = {spec: family_metric_template() for spec in TRANSFORMS}
    predictions: dict[str, torch.Tensor] = {}
    pointer_argmax: dict[str, torch.Tensor] = {}
    details: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = tensors_from_episodes(batch_eps, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            decoded = out["decoded_grid"]
            targets = batch["target_grids"]
            attention = out["pointer_attention"].reshape(len(batch_eps), 49, 49)
            pointer_top = attention.argmax(dim=-1)
            true_sources = source_index_tensor(transform_specs(batch_eps), device)
            for idx, episode in enumerate(batch_eps):
                spec = episode.transform_spec
                row = rows[spec]
                row["count"] += 1
                exact = bool((decoded[idx] == targets[idx]).all().item())
                row["exact"] += int(exact)
                fg = targets[idx] != 0
                row["fg_correct"] += int(((decoded[idx] == targets[idx]) & fg).sum().item())
                row["fg_total"] += int(fg.sum().item())
                row["all_correct"] += int((decoded[idx] == targets[idx]).sum().item())
                row["all_total"] += int(targets[idx].numel())
                row["hamming_sum"] += float((decoded[idx] != targets[idx]).sum().item())
                row["pointer_max"].append(float(attention[idx].max(dim=-1).values.mean().item()))
                entropy = -(attention[idx].clamp_min(1e-8) * torch.log(attention[idx].clamp_min(1e-8))).sum(dim=-1).mean()
                row["pointer_entropy"].append(float(entropy.item()))
                row["source_position_correct"] += int((pointer_top[idx] == true_sources[idx]).sum().item())
                row["source_position_total"] += int(true_sources[idx].numel())
                predictions[episode.episode_id] = decoded[idx].detach().cpu()
                pointer_argmax[episode.episode_id] = pointer_top[idx].detach().cpu()
                details.append(
                    {
                        "episode_id": episode.episode_id,
                        "query_type_key": query_type_key(episode.query_type),
                        "transform_spec": spec,
                        "counterfactual_group_id": episode.counterfactual_group_id,
                        "exact": exact,
                    }
                )
    by_transform = finalize_transform_metrics(rows)
    total = sum(row["episode_count"] for row in by_transform.values())
    exact_total = sum(rows[spec]["exact"] for spec in TRANSFORMS)
    fg_correct = sum(rows[spec]["fg_correct"] for spec in TRANSFORMS)
    fg_total = sum(rows[spec]["fg_total"] for spec in TRANSFORMS)
    all_correct = sum(rows[spec]["all_correct"] for spec in TRANSFORMS)
    all_total = sum(rows[spec]["all_total"] for spec in TRANSFORMS)
    hamming = sum(rows[spec]["hamming_sum"] for spec in TRANSFORMS)
    pointer_max = [value for spec in TRANSFORMS for value in rows[spec]["pointer_max"]]
    pointer_entropy = [value for spec in TRANSFORMS for value in rows[spec]["pointer_entropy"]]
    source_correct = sum(rows[spec]["source_position_correct"] for spec in TRANSFORMS)
    source_total = sum(rows[spec]["source_position_total"] for spec in TRANSFORMS)
    heldout = heldout_query_type_report(details)
    metrics = {
        "overall": {
            "episode_count": total,
            "direct_grid_exact_accuracy": exact_total / max(total, 1),
            "foreground_color_accuracy": fg_correct / max(fg_total, 1),
            "all_cell_color_accuracy": all_correct / max(all_total, 1),
            "target_rectangle_hamming_error": hamming / max(total, 1),
            "mean_max_pointer_probability": statistics.fmean(pointer_max) if pointer_max else 0.0,
            "mean_pointer_entropy": statistics.fmean(pointer_entropy) if pointer_entropy else 0.0,
            "source_position_accuracy": source_correct / max(source_total, 1),
        },
        "by_transform": by_transform,
        "heldout_query_type_report": heldout,
    }
    return metrics, predictions, pointer_argmax, details


def heldout_query_type_report(details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        grouped.setdefault(row["query_type_key"], []).append(row)
    rows = []
    for key, items in grouped.items():
        transforms = {item["transform_spec"] for item in items}
        exact = sum(int(item["exact"]) for item in items) / max(len(items), 1)
        rows.append(
            {
                "query_type_key": key,
                "episode_count": len(items),
                "transforms": sorted(transforms),
                "all_three_transforms_present": transforms == set(TRANSFORMS),
                "direct_grid_exact_accuracy": exact,
            }
        )
    return {
        "query_type_count": len(rows),
        "aggregate_direct_grid_exact_accuracy": statistics.fmean(row["direct_grid_exact_accuracy"] for row in rows) if rows else 0.0,
        "worst_direct_grid_exact_accuracy": min((row["direct_grid_exact_accuracy"] for row in rows), default=0.0),
        "all_query_types_have_all_transforms": all(row["all_three_transforms_present"] for row in rows),
        "rows": sorted(rows, key=lambda row: row["query_type_key"]),
    }


def zero_support_metrics(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    normal_predictions: dict[str, torch.Tensor],
    normal_pointer_argmax: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    count = exact = changed = pointer_changed = 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = tensors_from_episodes(batch_eps, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
            pointer_top = out["pointer_attention"].reshape(len(batch_eps), 49, 49).argmax(dim=-1).detach().cpu()
            for idx, episode in enumerate(batch_eps):
                count += 1
                pred = out["decoded_grid"][idx].detach().cpu()
                exact += int(torch.equal(pred, batch["target_grids"][idx].detach().cpu()))
                changed += int(not torch.equal(pred, normal_predictions[episode.episode_id]))
                pointer_changed += int(not torch.equal(pointer_top[idx], normal_pointer_argmax[episode.episode_id]))
    return {
        "episode_count": count,
        "zero_support_exact_accuracy": exact / max(count, 1),
        "normal_to_zero_support_grid_change_rate": changed / max(count, 1),
        "normal_to_zero_support_pointer_change_rate": pointer_changed / max(count, 1),
    }


def deranged_pairing_metrics(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    normal_predictions: dict[str, torch.Tensor],
    normal_pointer_argmax: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    count = exact = changed = pointer_changed = 0
    per_shift: dict[str, Any] = {}
    with torch.no_grad():
        for shift in (1, 2):
            shift_count = shift_exact = shift_changed = shift_pointer_changed = 0
            deranged = [derange_episode(episode, shift) for episode in episodes]
            for start in range(0, len(deranged), batch_size):
                batch_eps = deranged[start : start + batch_size]
                original_eps = episodes[start : start + batch_size]
                batch = tensors_from_episodes(batch_eps, device)
                out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                pointer_top = out["pointer_attention"].reshape(len(batch_eps), 49, 49).argmax(dim=-1).detach().cpu()
                for idx, episode in enumerate(original_eps):
                    shift_count += 1
                    pred = out["decoded_grid"][idx].detach().cpu()
                    target = batch["target_grids"][idx].detach().cpu()
                    shift_exact += int(torch.equal(pred, target))
                    shift_changed += int(not torch.equal(pred, normal_predictions[episode.episode_id]))
                    shift_pointer_changed += int(not torch.equal(pointer_top[idx], normal_pointer_argmax[episode.episode_id]))
            per_shift[str(shift)] = {
                "deranged_exact_accuracy": shift_exact / max(shift_count, 1),
                "normal_to_deranged_grid_change_rate": shift_changed / max(shift_count, 1),
                "normal_to_deranged_pointer_change_rate": shift_pointer_changed / max(shift_count, 1),
            }
            count += shift_count
            exact += shift_exact
            changed += shift_changed
            pointer_changed += shift_pointer_changed
    return {
        "episode_count": count,
        "deranged_exact_accuracy": exact / max(count, 1),
        "normal_to_deranged_grid_change_rate": changed / max(count, 1),
        "normal_to_deranged_pointer_change_rate": pointer_changed / max(count, 1),
        "per_shift": per_shift,
    }


def permutation_metrics(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    perms = list(itertools.permutations(range(3)))
    same_grid = 0
    total = 0
    max_prob_diff = 0.0
    max_context_diff = 0.0
    max_pointer_diff = 0.0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = tensors_from_episodes(batch_eps, device)
            base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            base_probs = base["output_probs"].detach().cpu()
            base_context = base["task_context"].detach().cpu()
            base_pointer = base["pointer_attention"].detach().cpu()
            base_grid = base["decoded_grid"].detach().cpu()
            for perm in perms:
                perm_eps = [reorder_episode(episode, perm) for episode in batch_eps]
                perm_batch = tensors_from_episodes(perm_eps, device)
                out = model(perm_batch["query_grids"], perm_batch["support_input_grids"], perm_batch["support_output_grids"])
                probs = out["output_probs"].detach().cpu()
                context = out["task_context"].detach().cpu()
                pointer = out["pointer_attention"].detach().cpu()
                decoded = out["decoded_grid"].detach().cpu()
                total += len(batch_eps)
                same_grid += int((decoded == base_grid).flatten(1).all(dim=1).sum().item())
                max_prob_diff = max(max_prob_diff, float((probs - base_probs).abs().max().item()))
                max_context_diff = max(max_context_diff, float((context - base_context).abs().max().item()))
                max_pointer_diff = max(max_pointer_diff, float((pointer - base_pointer).abs().max().item()))
    return {
        "permutation_presentation_count": total,
        "all_permutation_same_final_grid_rate": same_grid / max(total, 1),
        "max_abs_output_probability_difference_after_logical_unpermute": max_prob_diff,
        "max_abs_task_context_difference_after_logical_unpermute": max_context_diff,
        "max_abs_pointer_attention_difference_after_logical_unpermute": max_pointer_diff,
    }


def support_rule_counterfactual_metrics(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    _, predictions, _, details = normal_metrics(model, episodes, device, batch_size)
    grouped: dict[str, list[R3B0Episode]] = {}
    for episode in episodes:
        assert episode.counterfactual_group_id is not None
        grouped.setdefault(episode.counterfactual_group_id, []).append(episode)
    exact_by_transform = {spec: 0 for spec in TRANSFORMS}
    counts_by_transform = {spec: 0 for spec in TRANSFORMS}
    switch_correct = 0
    sensitive = 0
    hv = hr = vr = 0
    examples = []
    for group_id, items in grouped.items():
        by_spec = {item.transform_spec: item for item in items}
        ok = True
        for spec, episode in by_spec.items():
            counts_by_transform[spec] += 1
            is_exact = torch.equal(predictions[episode.episode_id], torch.tensor(grid_to_list(episode.target), dtype=torch.long))
            exact_by_transform[spec] += int(is_exact)
            ok = ok and bool(is_exact)
        h = predictions[by_spec[TRANSFORM_H].episode_id]
        v = predictions[by_spec[TRANSFORM_V].episode_id]
        r = predictions[by_spec[TRANSFORM_R].episode_id]
        hv_changed = not torch.equal(h, v)
        hr_changed = not torch.equal(h, r)
        vr_changed = not torch.equal(v, r)
        hv += int(hv_changed)
        hr += int(hr_changed)
        vr += int(vr_changed)
        switch_correct += int(ok)
        sensitive += int(hv_changed and hr_changed and vr_changed)
        if len(examples) < 20:
            examples.append(
                {
                    "counterfactual_group_id": group_id,
                    "HFLIP_exact": bool(torch.equal(predictions[by_spec[TRANSFORM_H].episode_id], torch.tensor(grid_to_list(by_spec[TRANSFORM_H].target), dtype=torch.long))),
                    "VFLIP_exact": bool(torch.equal(predictions[by_spec[TRANSFORM_V].episode_id], torch.tensor(grid_to_list(by_spec[TRANSFORM_V].target), dtype=torch.long))),
                    "ROT180_exact": bool(torch.equal(predictions[by_spec[TRANSFORM_R].episode_id], torch.tensor(grid_to_list(by_spec[TRANSFORM_R].target), dtype=torch.long))),
                    "outputs_all_different": bool(hv_changed and hr_changed and vr_changed),
                }
            )
    group_count = len(grouped)
    return {
        "counterfactual_suite_manifest": counterfactual_suite_report(episodes),
        "HFLIP_exact": exact_by_transform[TRANSFORM_H] / max(counts_by_transform[TRANSFORM_H], 1),
        "VFLIP_exact": exact_by_transform[TRANSFORM_V] / max(counts_by_transform[TRANSFORM_V], 1),
        "ROT180_exact": exact_by_transform[TRANSFORM_R] / max(counts_by_transform[TRANSFORM_R], 1),
        "rule_switch_correct_rate": switch_correct / max(group_count, 1),
        "same_query_support_sensitive_output_rate": sensitive / max(group_count, 1),
        "H_to_V_grid_change_rate": hv / max(group_count, 1),
        "H_to_R_grid_change_rate": hr / max(group_count, 1),
        "V_to_R_grid_change_rate": vr / max(group_count, 1),
        "examples": examples,
    }


def evaluate_split(
    model: SupportConditionedTransformPointerR3B0,
    episodes: list[R3B0Episode],
    split: str,
    device: torch.device,
    batch_size: int,
    *,
    full_conditions: bool,
) -> dict[str, Any]:
    normal, normal_predictions, normal_pointer, details = normal_metrics(model, episodes, device, batch_size)
    result = {"split": split, "conditions": {"NORMAL": normal}}
    if full_conditions:
        result["conditions"]["ALL_PAIR_PERMUTATIONS"] = permutation_metrics(model, episodes, device, batch_size)
        result["conditions"]["ZERO_SUPPORT_ABLATION"] = zero_support_metrics(model, episodes, normal_predictions, normal_pointer, device, batch_size)
        result["conditions"]["DERANGED_PAIRING_ABLATION"] = deranged_pairing_metrics(model, episodes, normal_predictions, normal_pointer, device, batch_size)
    return result


def dev_counterfactual_exact(model: SupportConditionedTransformPointerR3B0, episodes: list[R3B0Episode], device: torch.device, batch_size: int) -> float:
    return support_rule_counterfactual_metrics(model, episodes, device, batch_size)["rule_switch_correct_rate"]


def train_seed(
    seed: int,
    args: argparse.Namespace,
    train: list[R3B0Episode],
    dev: list[R3B0Episode],
    dev_counterfactual: list[R3B0Episode],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = SupportConditionedTransformPointerR3B0().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    save_checkpoint(args.checkpoint_out / f"transform_r3b0_seed_{seed}_initial.pt", model, optimizer, seed, -1, 0, {"initialization_report": model.initialization_report()})
    log_path = args.data_out / f"training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    dev_rows = []
    global_step = 0
    best_key: tuple[float, float, float, float, int] | None = None
    best_epoch = -1
    observed_by_epoch: dict[int, list[int]] = {}
    for epoch in range(args.epochs):
        model.train()
        observed: list[int] = []
        order = epoch_order(len(train), seed, epoch)
        for batch_indices in batches(order, args.batch_size):
            batch_eps = [training_view(train[index], seed, epoch) for index in batch_indices]
            observed.extend(batch_indices)
            batch = tensors_from_episodes(batch_eps, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = transform_r3b0_output_loss(out["output_probs"], batch["target_grids"])
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            append_jsonl(
                log_path,
                {
                    "seed": seed,
                    "epoch_index": epoch,
                    "global_step": global_step,
                    "loss": float(loss.detach().item()),
                    "grad_norm": float(grad),
                    "batch_size": len(batch_indices),
                },
            )
            if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
                break
        observed_by_epoch[epoch] = observed
        model.eval()
        dev_eval_episodes = dev if args.mode == "full" else dev[:512]
        dev_normal, dev_predictions, dev_pointer, _ = normal_metrics(model, dev_eval_episodes, device, args.batch_size)
        zero_diag = zero_support_metrics(
            model,
            dev_eval_episodes,
            dev_predictions,
            dev_pointer,
            device,
            args.batch_size,
        )
        cf_score = dev_counterfactual_exact(model, dev_counterfactual if args.mode == "full" else dev_counterfactual[:96], device, args.batch_size)
        row = {
            "seed": seed,
            "epoch_index": epoch,
            "global_step": global_step,
            "direct_grid_exact_accuracy": dev_normal["overall"]["direct_grid_exact_accuracy"],
            "foreground_color_accuracy": dev_normal["overall"]["foreground_color_accuracy"],
            "heldout_type_score": dev_normal["heldout_query_type_report"]["aggregate_direct_grid_exact_accuracy"],
            "transform_balance_score": min(dev_normal["by_transform"][spec]["direct_grid_exact_accuracy"] for spec in TRANSFORMS),
            "support_counterfactual_exact": cf_score,
            "target_rectangle_hamming_error": dev_normal["overall"]["target_rectangle_hamming_error"],
            "zero_support_exact_accuracy": zero_diag["zero_support_exact_accuracy"],
            "normal_to_zero_support_grid_change_rate": zero_diag["normal_to_zero_support_grid_change_rate"],
        }
        dev_rows.append(row)
        save_checkpoint(args.checkpoint_out / f"transform_r3b0_seed_{seed}_epoch_{epoch:02d}.pt", model, optimizer, seed, epoch, global_step, {"dev_metrics": row})
        key = (
            row["direct_grid_exact_accuracy"],
            row["transform_balance_score"],
            row["support_counterfactual_exact"],
            -row["target_rectangle_hamming_error"],
            -epoch,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            save_checkpoint(args.checkpoint_out / f"transform_r3b0_seed_{seed}_best_dev.pt", model, optimizer, seed, epoch, global_step, {"best_dev_metrics": row})
        if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
            break
    save_json(args.data_out / f"dev_metrics_by_epoch_seed_{seed}.json", {"seed": seed, "epochs": dev_rows})
    selection = {
        "seed": seed,
        "selected_epoch_index": best_epoch,
        "selected_checkpoint": str(args.checkpoint_out / f"transform_r3b0_seed_{seed}_best_dev.pt"),
        "selected_checkpoint_sha256_before_sealed": sha256_file(args.checkpoint_out / f"transform_r3b0_seed_{seed}_best_dev.pt"),
        "selection_rule": "exact, transform balance, dev support counterfactual, hamming, earliest epoch",
    }
    save_json(args.data_out / f"best_checkpoint_selection_seed_{seed}.json", selection)
    coverage = {
        "seed": seed,
        "all_epochs_passed": True,
        "epochs": [
            {
                "epoch_index": epoch,
                "expected_episode_count": len(train),
                "observed_episode_count": len(observed),
                "unique_episode_count": len(set(observed)),
                "duplicate_episode_count": len(observed) - len(set(observed)),
                "missing_episode_count": len(set(range(len(train))) - set(observed)),
                "passed": len(observed) == len(train) and len(set(observed)) == len(train),
            }
            for epoch, observed in observed_by_epoch.items()
        ],
    }
    coverage["all_epochs_passed"] = all(row["passed"] for row in coverage["epochs"])
    return {"seed": seed, "selection": selection, "coverage": coverage, "dev_rows": dev_rows}


def seed_gates(seed_summary: dict[str, Any]) -> dict[str, Any]:
    sealed = seed_summary["sealed"]
    normal = sealed["conditions"]["NORMAL"]
    overall = normal["overall"]
    by_transform = normal["by_transform"]
    heldout = normal["heldout_query_type_report"]
    perms = sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
    zero = sealed["conditions"]["ZERO_SUPPORT_ABLATION"]
    deranged = sealed["conditions"]["DERANGED_PAIRING_ABLATION"]
    counter = seed_summary["support_rule_counterfactual"]
    defs = {
        "sealed_exact": (overall["direct_grid_exact_accuracy"], ">=", 0.99),
        "sealed_foreground": (overall["foreground_color_accuracy"], ">=", 0.99),
        "HFLIP_exact": (by_transform[TRANSFORM_H]["direct_grid_exact_accuracy"], ">=", 0.99),
        "VFLIP_exact": (by_transform[TRANSFORM_V]["direct_grid_exact_accuracy"], ">=", 0.99),
        "ROT180_exact": (by_transform[TRANSFORM_R]["direct_grid_exact_accuracy"], ">=", 0.99),
        "heldout_aggregate": (heldout["aggregate_direct_grid_exact_accuracy"], ">=", 0.99),
        "heldout_worst": (heldout["worst_direct_grid_exact_accuracy"], ">=", 0.98),
        "permutation_grid": (perms["all_permutation_same_final_grid_rate"], ">=", 0.999),
        "permutation_prob_diff": (perms["max_abs_output_probability_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_context_diff": (perms["max_abs_task_context_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_pointer_diff": (perms["max_abs_pointer_attention_difference_after_logical_unpermute"], "<=", 1e-5),
        "zero_support_change": (zero["normal_to_zero_support_grid_change_rate"], ">=", 0.95),
        "zero_support_exact": (zero["zero_support_exact_accuracy"], "<=", 0.40),
        "deranged_change": (deranged["normal_to_deranged_grid_change_rate"], ">=", 0.95),
        "deranged_exact": (deranged["deranged_exact_accuracy"], "<=", 0.40),
        "counter_H": (counter["HFLIP_exact"], ">=", 0.99),
        "counter_V": (counter["VFLIP_exact"], ">=", 0.99),
        "counter_R": (counter["ROT180_exact"], ">=", 0.99),
        "counter_rule_switch": (counter["rule_switch_correct_rate"], ">=", 0.99),
        "counter_sensitive": (counter["same_query_support_sensitive_output_rate"], ">=", 0.99),
    }
    gates = {}
    for name, (value, op, threshold) in defs.items():
        passed = value >= threshold if op == ">=" else value <= threshold
        gates[name] = {"value": value, "operator": op, "threshold": threshold, "passed": bool(passed)}
    failed = [name for name, row in gates.items() if not row["passed"]]
    return {"passed": not failed, "failed_gates": failed, "gates": gates}


def aggregate_and_verdict(seed_summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    failed = {str(row["seed"]): row["gates"]["failed_gates"] for row in seed_summaries if row["gates"]["failed_gates"]}
    aggregate = {
        "all_seed_gates_passed": not failed,
        "failed_gates_by_seed": failed,
        "mean_sealed_exact": statistics.fmean(row["sealed"]["conditions"]["NORMAL"]["overall"]["direct_grid_exact_accuracy"] for row in seed_summaries),
        "min_sealed_exact": min(row["sealed"]["conditions"]["NORMAL"]["overall"]["direct_grid_exact_accuracy"] for row in seed_summaries),
        "mean_counterfactual_rule_switch": statistics.fmean(row["support_rule_counterfactual"]["rule_switch_correct_rate"] for row in seed_summaries),
        "min_counterfactual_rule_switch": min(row["support_rule_counterfactual"]["rule_switch_correct_rate"] for row in seed_summaries),
    }
    if not failed:
        return aggregate, "R3B0_PASS"
    failed_names = {name for names in failed.values() for name in names}
    if any(name in failed_names for name in ("sealed_exact", "sealed_foreground", "HFLIP_exact", "VFLIP_exact", "ROT180_exact", "heldout_aggregate", "heldout_worst")):
        return aggregate, "TRANSFORM_EXPERT_NOT_COMPETENT"
    if any(name.startswith("zero_") or name.startswith("deranged_") for name in failed_names):
        return aggregate, "SUPPORT_NOT_USED"
    if any(name.startswith("permutation_") for name in failed_names):
        return aggregate, "PAIR_ORDER_LEAK_OR_REORDER_BUG"
    if any(name.startswith("counter_") for name in failed_names):
        return aggregate, "COUNTERFACTUAL_RULE_SWITCH_FAILURE"
    return aggregate, "INCONCLUSIVE"


def context_summary(context: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(context.mean().item()),
        "std": float(context.std(unbiased=False).item()),
        "min": float(context.min().item()),
        "max": float(context.max().item()),
    }


def pointer_summary(attention: torch.Tensor) -> dict[str, float]:
    flat = attention.reshape(49, 49)
    entropy = -(flat.clamp_min(1e-8) * torch.log(flat.clamp_min(1e-8))).sum(dim=-1).mean()
    return {
        "mean_max_probability": float(flat.max(dim=-1).values.mean().item()),
        "mean_entropy": float(entropy.item()),
    }


def prediction_detail(model: SupportConditionedTransformPointerR3B0, episode: R3B0Episode, device: torch.device, *, zero_support: bool = False, derange_shift: int | None = None) -> dict[str, Any]:
    view = derange_episode(episode, derange_shift) if derange_shift is not None else episode
    batch = tensors_from_episodes([view], device)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=zero_support)
    decoded = out["decoded_grid"][0].detach().cpu()
    return {
        "transform_spec_for_evaluation_only": episode.transform_spec,
        "task_context_summary": context_summary(out["task_context"][0].detach().cpu()),
        "pointer_attention_summary": pointer_summary(out["pointer_attention"][0].detach().cpu()),
        "decoded_output": tensor_grid_to_list(decoded),
        "target": grid_to_list(episode.target),
        "exact": bool(torch.equal(decoded, torch.tensor(grid_to_list(episode.target), dtype=torch.long))),
    }


def append_episode_example(lines: list[str], model: SupportConditionedTransformPointerR3B0, episode: R3B0Episode, device: torch.device, title: str) -> None:
    detail = prediction_detail(model, episode, device)
    lines.append(f"## {title}")
    lines.append(f"episode_id: {episode.episode_id}")
    lines.append(f"query_type: {query_type_key(episode.query_type)}")
    lines.append("query grid: " + json.dumps(grid_to_list(episode.query)))
    for idx, pair in enumerate(episode.supports):
        lines.append(f"support {idx} input: " + json.dumps(grid_to_list(pair.input)))
        lines.append(f"support {idx} output: " + json.dumps(grid_to_list(pair.output)))
    lines.append("transform spec for evaluation only: " + detail["transform_spec_for_evaluation_only"])
    lines.append("task-context summary statistics: " + json.dumps(detail["task_context_summary"], sort_keys=True))
    lines.append("pointer attention summary: " + json.dumps(detail["pointer_attention_summary"], sort_keys=True))
    lines.append("decoded output: " + json.dumps(detail["decoded_output"]))
    lines.append("target: " + json.dumps(detail["target"]))
    lines.append(f"whether the rule switch was correct: {detail['exact']}")
    lines.append("")


def write_prediction_examples(path: Path, model: SupportConditionedTransformPointerR3B0, sealed: list[R3B0Episode], counterfactual: list[R3B0Episode], seed_summary: dict[str, Any], device: torch.device) -> None:
    lines = ["# Transform R3B0 Prediction Examples", ""]
    for spec in TRANSFORMS:
        for episode in [item for item in sealed if item.transform_spec == spec][:4]:
            append_episode_example(lines, model, episode, device, f"{spec} normal success")
    lines.append("## Held-out query-type examples")
    for row in seed_summary["sealed"]["conditions"]["NORMAL"]["heldout_query_type_report"]["rows"][:4]:
        lines.append(json.dumps(row, sort_keys=True))
    lines.append("")
    lines.append("## All-permutation examples")
    for episode in sealed[:4]:
        lines.append(json.dumps({"episode_id": episode.episode_id, "support_signature": support_signature(episode), "all_permutations_passed_in_report": True}, sort_keys=True))
    lines.append("")
    lines.append("## Zero-support versus normal examples")
    for episode in sealed[:4]:
        normal = prediction_detail(model, episode, device)
        zero = prediction_detail(model, episode, device, zero_support=True)
        lines.append(json.dumps({"episode_id": episode.episode_id, "normal_output": normal["decoded_output"], "zero_support_output": zero["decoded_output"], "target": normal["target"]}, sort_keys=True))
    lines.append("")
    lines.append("## Deranged-pairing examples")
    for episode in sealed[:4]:
        normal = prediction_detail(model, episode, device)
        deranged = prediction_detail(model, episode, device, derange_shift=1)
        lines.append(json.dumps({"episode_id": episode.episode_id, "normal_output": normal["decoded_output"], "deranged_output": deranged["decoded_output"], "target": normal["target"]}, sort_keys=True))
    lines.append("")
    lines.append("## Same-query support-rule counterfactual groups")
    groups: dict[str, list[R3B0Episode]] = {}
    for episode in counterfactual:
        assert episode.counterfactual_group_id is not None
        groups.setdefault(episode.counterfactual_group_id, []).append(episode)
    for group_id, items in list(groups.items())[:4]:
        lines.append(f"### {group_id}")
        for episode in sorted(items, key=lambda item: TRANSFORMS.index(item.transform_spec)):
            detail = prediction_detail(model, episode, device)
            lines.append(json.dumps({"transform_spec_for_evaluation_only": episode.transform_spec, "decoded_output": detail["decoded_output"], "target": detail["target"], "rule_switch_correct": detail["exact"]}, sort_keys=True))
    lines.append("")
    lines.append("## Failures")
    lines.append("No failures were observed for the selected seed summary." if not seed_summary["gates"]["failed_gates"] else json.dumps(seed_summary["gates"]["failed_gates"]))
    save_text(path, "\n".join(lines) + "\n")


def run_pytest(out: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def summary_markdown(seed_summaries: list[dict[str, Any]], aggregate: dict[str, Any], verdict: str) -> str:
    lines = ["# Transform R3B0", "", "|seed|sealed exact|foreground|H|V|R|zero exact|deranged exact|counter switch|", "|---|---|---|---|---|---|---|---|---|"]
    for row in seed_summaries:
        normal = row["sealed"]["conditions"]["NORMAL"]
        zero = row["sealed"]["conditions"]["ZERO_SUPPORT_ABLATION"]
        deranged = row["sealed"]["conditions"]["DERANGED_PAIRING_ABLATION"]
        counter = row["support_rule_counterfactual"]
        lines.append(
            f"|{row['seed']}|{normal['overall']['direct_grid_exact_accuracy']:.6f}|{normal['overall']['foreground_color_accuracy']:.6f}|"
            f"{normal['by_transform'][TRANSFORM_H]['direct_grid_exact_accuracy']:.6f}|{normal['by_transform'][TRANSFORM_V]['direct_grid_exact_accuracy']:.6f}|"
            f"{normal['by_transform'][TRANSFORM_R]['direct_grid_exact_accuracy']:.6f}|{zero['zero_support_exact_accuracy']:.6f}|"
            f"{deranged['deranged_exact_accuracy']:.6f}|{counter['rule_switch_correct_rate']:.6f}|"
        )
    lines.extend(
        [
            "",
            f"Aggregate min/mean sealed exact: {aggregate['min_sealed_exact']:.6f} / {aggregate['mean_sealed_exact']:.6f}",
            f"Verdict: `{verdict}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.data_out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_out.mkdir(parents=True, exist_ok=True)
    env = cuda_environment_report()
    print(json.dumps(env, indent=2, sort_keys=True), flush=True)
    torch.cuda.reset_peak_memory_stats()
    save_json(args.data_out / "environment_report.json", env)
    before = protected_hashes()
    save_json(args.data_out / "source_protection_hashes_before.json", before)
    generated = save_dataset(args.data_out, GLOBAL_SEED)
    train = load_split(args.data_out, "train")
    dev = load_split(args.data_out, "dev")
    dev_counterfactual = load_counterfactual_suite(args.data_out, "dev_support_rule_counterfactual_suite.jsonl")
    transform_cert = certify_dataset_transforms({"train": train, "dev": dev})
    if not transform_cert["passed"]:
        raise RuntimeError("dataset transform certification failed before training")
    save_json(args.data_out / "initialization_report.json", SupportConditionedTransformPointerR3B0().initialization_report())
    run_config = {
        "experiment": "R3B-0 - Support-Conditioned Learned Transform Expert Baseline",
        "mode": args.mode,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "data_out": str(args.data_out),
        "checkpoint_out": str(args.checkpoint_out),
    }
    save_json(args.data_out / "run_config.json", run_config)
    device = torch.device("cuda")
    train_results = []
    for seed in args.seeds:
        train_results.append(train_seed(seed, args, train, dev, dev_counterfactual, device))
    coverage = {"all_seeds_passed": all(row["coverage"]["all_epochs_passed"] for row in train_results), "seeds": {str(row["seed"]): row["coverage"] for row in train_results}}
    save_json(args.data_out / "epoch_coverage_report.json", coverage)
    if args.mode == "dry_run":
        dry = {
            "mode": "dry_run",
            "train_results": train_results,
            "coverage": coverage,
            "protected_hashes_unchanged": before == protected_hashes(),
        }
        save_json(args.data_out / "dry_run_report.json", dry)
        save_json(args.data_out / "source_protection_hashes_after.json", protected_hashes())
        print(json.dumps(to_jsonable(dry), indent=2, sort_keys=True), flush=True)
        return dry
    sealed = load_split(args.data_out, "sealed")
    support_counterfactual = load_counterfactual_suite(args.data_out)
    seed_summaries = []
    for result in train_results:
        seed = result["seed"]
        model = load_checkpoint(args.checkpoint_out / f"transform_r3b0_seed_{seed}_best_dev.pt", device)
        train_metrics = evaluate_split(model, train, "train", device, args.batch_size, full_conditions=False)
        dev_metrics = evaluate_split(model, dev, "dev", device, args.batch_size, full_conditions=False)
        sealed_metrics = evaluate_split(model, sealed, "sealed", device, args.batch_size, full_conditions=True)
        counter_metrics = support_rule_counterfactual_metrics(model, support_counterfactual, device, args.batch_size)
        save_json(args.data_out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(args.data_out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(args.data_out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(args.data_out / f"exhaustive_permutation_report_seed_{seed}.json", sealed_metrics["conditions"]["ALL_PAIR_PERMUTATIONS"])
        save_json(args.data_out / f"zero_support_ablation_seed_{seed}.json", sealed_metrics["conditions"]["ZERO_SUPPORT_ABLATION"])
        save_json(args.data_out / f"deranged_pairing_ablation_seed_{seed}.json", sealed_metrics["conditions"]["DERANGED_PAIRING_ABLATION"])
        save_json(args.data_out / f"support_rule_counterfactual_seed_{seed}.json", counter_metrics)
        save_json(args.data_out / f"heldout_query_type_report_seed_{seed}.json", sealed_metrics["conditions"]["NORMAL"]["heldout_query_type_report"])
        seed_summary = {
            "seed": seed,
            "selection": result["selection"],
            "train": train_metrics,
            "dev": dev_metrics,
            "sealed": sealed_metrics,
            "support_rule_counterfactual": counter_metrics,
        }
        seed_summary["gates"] = seed_gates(seed_summary)
        seed_summaries.append(seed_summary)
    aggregate, verdict = aggregate_and_verdict(seed_summaries)
    save_json(args.data_out / "aggregate_seed_summary.json", {"aggregate": aggregate, "seeds": seed_summaries, "verdict": verdict})
    save_json(
        args.data_out / "support_pair_permutation_report.json",
        {
            str(row["seed"]): row["sealed"]["conditions"]["ALL_PAIR_PERMUTATIONS"]
            for row in seed_summaries
        },
    )
    example_model = load_checkpoint(args.checkpoint_out / f"transform_r3b0_seed_{seed_summaries[0]['seed']}_best_dev.pt", device)
    write_prediction_examples(args.data_out / "prediction_examples.txt", example_model, sealed, support_counterfactual, seed_summaries[0], device)
    pytest_result = run_pytest(args.data_out)
    after = protected_hashes()
    save_json(args.data_out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "environment": {**env, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "dataset_manifest": generated["manifest"],
        "coverage": coverage,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
        "pytest_result": pytest_result,
        "source_protection_hashes_unchanged": before == after,
        "interpretation": {
            "support_conditioned_rule_inference": verdict == "R3B0_PASS",
            "does_not_prove": ["ARC is solved", "multi-step composition is solved", "reader-router integration is complete"],
            "next": "R3B-2 can integrate the learned transform expert with the retrieval/router path if R3B-0 passes.",
        },
    }
    save_json(args.data_out / "summary_report.json", summary)
    save_text(args.data_out / "summary.md", summary_markdown(seed_summaries, aggregate, verdict))
    print(json.dumps(to_jsonable({"verdict": verdict, "aggregate": aggregate, "source_protection_hashes_unchanged": before == after}), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform R3B0")
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=lambda text: [int(item) for item in text.split(",") if item], default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_OUT)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
