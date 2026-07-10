from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrm.data import ARC_COLOR_COUNT, Grid

from .epoch_sampler import (
    coverage_row,
    epoch_batches,
    epoch_permutation_indices,
    epoch_training_view,
    sha256_json,
    support_pair_order_report,
)
from .pair_reader_r1 import PairKeyValueReader, PairReaderConfig, entropy_from_weights, pair_reader_loss
from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_PAIR_ORDER_SHUFFLED,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    condition_view,
    group_by_base,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
)

DEFAULT_DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
DEFAULT_OUT = Path("/mnt/d/VRM/artifacts/twin_pair_reader_r1")
DEFAULT_CHECKPOINT_DIR = Path("/mnt/d/VRM/checkpoints/twin_pair_reader_r1")
SEED = 20260628
CONDITIONS = (CONDITION_NORMAL, CONDITION_TWIN_SWAP_SUPPORTS, CONDITION_PAIR_ORDER_SHUFFLED)
PROJECT_ROOT = Path("/mnt/d/VRM")

PROTECTED_FILES = (
    PROJECT_ROOT / "vrm/model.py",
    PROJECT_ROOT / "vrm/data.py",
    PROJECT_ROOT / "vrm/metrics.py",
    PROJECT_ROOT / "vrm/train.py",
)
PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts/twin_task_pilot_v1",
    PROJECT_ROOT / "artifacts/twin_vrm_probe_v0",
    PROJECT_ROOT / "artifacts/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "checkpoints/twin_vrm_probe_v0",
    PROJECT_ROOT / "checkpoints/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "archives",
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
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
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes() -> dict[str, str | None]:
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
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            hashes[key] = sha256_file(path)
    return hashes


def configure_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {"seed": seed, "torch_deterministic_algorithms": "enabled_warn_only"}


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


def load_split_episodes(data: Path, split: str, *, sealed_allowed: bool) -> list[TwinEpisode]:
    if split == "sealed" and not sealed_allowed:
        raise RuntimeError("sealed split read attempted before best checkpoint selection was written")
    return records_to_episodes(read_jsonl(Path(data) / f"{split}.jsonl"))


def load_all_for_oracle(data: Path) -> dict[str, list[TwinEpisode]]:
    return {split: records_to_episodes(read_jsonl(Path(data) / f"{split}.jsonl")) for split in ("train", "dev", "sealed")}


def grid_tensor(grid: Grid, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.tensor(grid.to_list(), dtype=torch.long)
    return tensor.to(device) if device is not None else tensor


def grid_hash(grid: Grid | torch.Tensor | list[list[int]]) -> str:
    if isinstance(grid, Grid):
        payload = grid.to_list()
    elif isinstance(grid, torch.Tensor):
        payload = grid.detach().cpu().to(torch.long).tolist()
    else:
        payload = grid
    return sha256_json(payload)


def relevant_pair_index(episode: TwinEpisode) -> int:
    matches = [idx for idx, pair in enumerate(episode.supports) if pair.input == episode.query_input]
    if len(matches) != 1:
        raise ValueError(f"{episode.episode_id} has {len(matches)} query-matching support inputs")
    return matches[0]


def foreground_metric(pred: torch.Tensor, target: torch.Tensor, background: int = 0) -> dict[str, Any]:
    mask = target != background
    total = int(mask.sum().item())
    correct = int((pred[mask] == target[mask]).sum().item()) if total else 0
    return {
        "foreground_color_correct": correct,
        "foreground_color_total": total,
        "foreground_color_accuracy": correct / max(total, 1),
    }


def candidate_metrics(pred: torch.Tensor, episode: TwinEpisode) -> dict[str, Any]:
    c0 = grid_tensor(episode.candidates[0])
    c1 = grid_tensor(episode.candidates[1])
    d0 = int((pred != c0).sum().item())
    d1 = int((pred != c1).sum().item())
    if d0 < d1:
        decision: int | None = 0
    elif d1 < d0:
        decision = 1
    else:
        decision = None
    return {
        "distance_to_candidate_0": d0,
        "distance_to_candidate_1": d1,
        "candidate_decision": decision,
        "candidate_decision_name": "TIE" if decision is None else str(decision),
        "candidate_nearest_score": 0.5 if decision is None else (1.0 if decision == episode.label else 0.0),
        "candidate_tie": decision is None,
        "candidate_exact_match": bool(torch.equal(pred, c0) or torch.equal(pred, c1)),
    }


def oracle_adapter_report(episodes_by_split: dict[str, list[TwinEpisode]]) -> dict[str, Any]:
    report: dict[str, Any] = {"adapter": "exact query grid equals exactly one support input grid"}
    all_passed = True
    for split, episodes in episodes_by_split.items():
        rows = []
        all_correct = 0
        all_total = 0
        fg_correct = 0
        fg_total = 0
        for episode in episodes:
            idx = relevant_pair_index(episode)
            pred = grid_tensor(episode.supports[idx].output)
            target = grid_tensor(episode.target_output)
            hamming = int((pred != target).sum().item())
            fg = foreground_metric(pred, target)
            row = {
                "episode_id": episode.episode_id,
                "relevant_pair_index": idx,
                "direct_grid_exact": hamming == 0,
                "target_rectangle_hamming_error": hamming,
                **fg,
            }
            rows.append(row)
            all_correct += int((pred == target).sum().item())
            all_total += int(target.numel())
            fg_correct += fg["foreground_color_correct"]
            fg_total += fg["foreground_color_total"]
        count = len(rows)
        split_report = {
            "episode_count": count,
            "direct_grid_exact_accuracy": sum(row["direct_grid_exact"] for row in rows) / max(count, 1),
            "foreground_color_accuracy": fg_correct / max(fg_total, 1),
            "all_cell_color_accuracy": all_correct / max(all_total, 1),
            "target_rectangle_hamming_error": statistics.fmean(row["target_rectangle_hamming_error"] for row in rows)
            if rows
            else 0.0,
            "foreground_cell_count_per_episode": {
                "unique": sorted({row["foreground_color_total"] for row in rows}),
                "expected_informative_foreground_count_verified": sorted({row["foreground_color_total"] for row in rows}) == [5],
            },
        }
        split_report["passed"] = (
            split_report["direct_grid_exact_accuracy"] == 1.0
            and split_report["foreground_color_accuracy"] == 1.0
            and split_report["target_rectangle_hamming_error"] == 0.0
        )
        report[split] = split_report
        all_passed = all_passed and split_report["passed"]
    report["passed"] = all_passed
    return report


def data_manifest(data: Path, episodes_by_split: dict[str, list[TwinEpisode]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "data_dir": str(data),
        "source_files": {},
        "episode_counts": {},
        "base_pair_counts": {},
        "foreground_target_counts": {},
        "candidate_and_labels_eval_only": True,
        "raw_arc_data_used": False,
    }
    for split, episodes in episodes_by_split.items():
        path = Path(data) / f"{split}.jsonl"
        manifest["source_files"][split] = {"path": str(path), "sha256": sha256_file(path)}
        manifest["episode_counts"][split] = len(episodes)
        manifest["base_pair_counts"][split] = len(group_by_base(episodes))
        manifest["foreground_target_counts"][split] = sorted(
            {sum(1 for row in episode.target_output.cells for value in row if value != 0) for episode in episodes}
        )
    return manifest


def batch_tensors(
    view_episodes: list[TwinEpisode],
    source_episodes: list[TwinEpisode],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    queries = torch.stack([grid_tensor(episode.query_input) for episode in view_episodes]).to(device)
    support_inputs = torch.stack(
        [torch.stack([grid_tensor(pair.input) for pair in episode.supports]) for episode in view_episodes]
    ).to(device)
    support_outputs = torch.stack(
        [torch.stack([grid_tensor(pair.output) for pair in episode.supports]) for episode in view_episodes]
    ).to(device)
    targets = torch.stack([grid_tensor(episode.target_output) for episode in source_episodes]).to(device)
    relevant = torch.tensor([relevant_pair_index(episode) for episode in view_episodes], dtype=torch.long, device=device)
    return {
        "query_grids": queries,
        "support_input_grids": support_inputs,
        "support_output_grids": support_outputs,
        "target_grids": targets,
        "relevant_pair_indices": relevant,
    }


def evaluate_condition(
    *,
    model: PairKeyValueReader,
    episodes: list[TwinEpisode],
    split: str,
    condition: str,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    opposite = build_opposite_lookup(episodes)
    rows: list[dict[str, Any]] = []
    all_correct = 0
    all_total = 0
    fg_correct = 0
    fg_total = 0
    retrieval_loss_values: list[float] = []
    output_loss_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            view_batch = [
                condition_view(episode, condition=condition, opposite_by_episode=opposite, seed=seed)
                for episode in source_batch
            ]
            batch = batch_tensors(view_batch, source_batch, device)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            _, losses = pair_reader_loss(
                outputs,
                batch["target_grids"],
                batch["relevant_pair_indices"],
                lambda_retrieval=1.0,
            )
            retrieval_loss_values.append(float(losses["retrieval_loss"].cpu().item()))
            output_loss_values.append(float(losses["output_loss"].cpu().item()))
            decoded = outputs["decoded_grid"].detach().cpu().to(torch.long)
            alpha = outputs["retrieval_weights"].detach().cpu()
            entropy = entropy_from_weights(alpha).detach().cpu()
            for local_idx, source_episode in enumerate(source_batch):
                view_episode = view_batch[local_idx]
                pred = decoded[local_idx]
                target = grid_tensor(source_episode.target_output)
                relevant_idx = relevant_pair_index(view_episode)
                top_idx = int(alpha[local_idx].argmax().item())
                fg = foreground_metric(pred, target)
                correct = int((pred == target).sum().item())
                total = int(target.numel())
                hamming = total - correct
                cand = candidate_metrics(pred, source_episode)
                input_hashes = [grid_hash(pair.input) for pair in view_episode.supports]
                logical_alpha = {input_hashes[idx]: float(alpha[local_idx, idx].item()) for idx in range(len(input_hashes))}
                row = {
                    "split": split,
                    "condition": condition,
                    "episode_index": start + local_idx,
                    "episode_id": source_episode.episode_id,
                    "base_id": source_episode.base_id,
                    "twin": source_episode.twin,
                    "label": source_episode.label,
                    "alternate_label": source_episode.alternate_label,
                    "relevant_pair_index": relevant_idx,
                    "selected_pair_index": top_idx,
                    "retrieval_top1_correct": top_idx == relevant_idx,
                    "relevant_pair_mass": float(alpha[local_idx, relevant_idx].item()),
                    "retrieval_entropy": float(entropy[local_idx].item()),
                    "retrieval_alpha": [float(value) for value in alpha[local_idx].tolist()],
                    "logical_alpha_by_support_input_hash": logical_alpha,
                    "selected_logical_support_input_hash": input_hashes[top_idx],
                    "relevant_logical_support_input_hash": input_hashes[relevant_idx],
                    "prediction_sha256": grid_hash(pred),
                    "target_sha256": grid_hash(target),
                    "decoded_grid": pred.tolist(),
                    "target_grid": target.tolist(),
                    "direct_grid_exact": hamming == 0,
                    "all_cell_color_correct": correct,
                    "all_cell_color_total": total,
                    "all_cell_color_accuracy": correct / max(total, 1),
                    "target_rectangle_hamming_error": hamming,
                    **fg,
                    **cand,
                }
                rows.append(row)
                all_correct += correct
                all_total += total
                fg_correct += fg["foreground_color_correct"]
                fg_total += fg["foreground_color_total"]
    count = len(rows)
    hamming_values = [row["target_rectangle_hamming_error"] for row in rows]
    relevant_masses = [row["relevant_pair_mass"] for row in rows]
    foreground_counts = [row["foreground_color_total"] for row in rows]
    aggregate = {
        "split": split,
        "condition": condition,
        "episode_count": count,
        "retrieval_top1_accuracy": sum(row["retrieval_top1_correct"] for row in rows) / max(count, 1),
        "mean_relevant_pair_mass": statistics.fmean(relevant_masses) if relevant_masses else 0.0,
        "median_relevant_pair_mass": statistics.median(relevant_masses) if relevant_masses else 0.0,
        "retrieval_entropy": statistics.fmean(row["retrieval_entropy"] for row in rows) if rows else 0.0,
        "logical_relevant_pair_accuracy_after_permutation": sum(
            row["selected_logical_support_input_hash"] == row["relevant_logical_support_input_hash"] for row in rows
        )
        / max(count, 1),
        "direct_grid_exact_accuracy": sum(row["direct_grid_exact"] for row in rows) / max(count, 1),
        "foreground_color_accuracy": fg_correct / max(fg_total, 1),
        "all_cell_color_accuracy": all_correct / max(all_total, 1),
        "foreground_cell_count_per_episode": {
            "mean": statistics.fmean(foreground_counts) if foreground_counts else 0.0,
            "unique": sorted(set(foreground_counts)),
            "expected_informative_foreground_count_verified": sorted(set(foreground_counts)) == [5],
        },
        "mean_target_rectangle_hamming_error": statistics.fmean(hamming_values) if hamming_values else 0.0,
        "median_target_rectangle_hamming_error": statistics.median(hamming_values) if hamming_values else 0.0,
        "candidate_nearest_score": statistics.fmean(row["candidate_nearest_score"] for row in rows) if rows else 0.0,
        "candidate_tie_rate": sum(row["candidate_tie"] for row in rows) / max(count, 1),
        "candidate_exact_match_rate": sum(row["candidate_exact_match"] for row in rows) / max(count, 1),
        "retrieval_loss": statistics.fmean(retrieval_loss_values) if retrieval_loss_values else 0.0,
        "output_loss": statistics.fmean(output_loss_values) if output_loss_values else 0.0,
    }
    return aggregate, rows


def by_episode_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in rows}


def derived_metrics(episodes: list[TwinEpisode], per_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    normal = by_episode_id(per_condition[CONDITION_NORMAL])
    swapped = by_episode_id(per_condition[CONDITION_TWIN_SWAP_SUPPORTS])
    shuffled = by_episode_id(per_condition[CONDITION_PAIR_ORDER_SHUFFLED])
    by_base = group_by_base(episodes)
    pair_count = len(by_base)
    twin_same_grid = 0
    twin_same_candidate = 0
    joint_normal = 0
    mapping_flip = 0
    normal_swapped_grid_change = 0
    normal_swapped_candidate_change = 0
    correct_switch = 0
    normal_shuffled_same_grid = 0
    normal_shuffled_same_candidate = 0
    normal_shuffled_same_logical_pair = 0
    max_alpha_diff = 0.0
    for episode in episodes:
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        p = shuffled[episode.episode_id]
        if n["prediction_sha256"] != s["prediction_sha256"]:
            normal_swapped_grid_change += 1
        if n["candidate_decision"] is not None and s["candidate_decision"] is not None and n["candidate_decision"] != s["candidate_decision"]:
            normal_swapped_candidate_change += 1
        if n["candidate_decision"] == episode.label and s["candidate_decision"] == episode.alternate_label:
            correct_switch += 1
        if n["prediction_sha256"] == p["prediction_sha256"]:
            normal_shuffled_same_grid += 1
        if n["candidate_decision"] == p["candidate_decision"]:
            normal_shuffled_same_candidate += 1
        if n["selected_logical_support_input_hash"] == p["selected_logical_support_input_hash"]:
            normal_shuffled_same_logical_pair += 1
        keys = set(n["logical_alpha_by_support_input_hash"]) | set(p["logical_alpha_by_support_input_hash"])
        for key in keys:
            diff = abs(
                float(n["logical_alpha_by_support_input_hash"].get(key, 0.0))
                - float(p["logical_alpha_by_support_input_hash"].get(key, 0.0))
            )
            max_alpha_diff = max(max_alpha_diff, diff)
    for twins in by_base.values():
        if set(twins) != {"A", "B"}:
            continue
        ep_a = twins["A"]
        ep_b = twins["B"]
        na = normal[ep_a.episode_id]
        nb = normal[ep_b.episode_id]
        sa = swapped[ep_a.episode_id]
        sb = swapped[ep_b.episode_id]
        if na["prediction_sha256"] == nb["prediction_sha256"]:
            twin_same_grid += 1
        if na["candidate_decision"] == nb["candidate_decision"]:
            twin_same_candidate += 1
        normal_correct = na["candidate_decision"] == ep_a.label and nb["candidate_decision"] == ep_b.label
        if normal_correct:
            joint_normal += 1
        if normal_correct and sa["candidate_decision"] == ep_a.alternate_label and sb["candidate_decision"] == ep_b.alternate_label:
            mapping_flip += 1
    episode_count = len(episodes)
    return {
        "base_pair_count": pair_count,
        "twin_same_predicted_grid_rate": twin_same_grid / max(pair_count, 1),
        "twin_same_candidate_decision_rate": twin_same_candidate / max(pair_count, 1),
        "normal_to_swapped_grid_change_rate": normal_swapped_grid_change / max(episode_count, 1),
        "normal_to_swapped_candidate_change_rate": normal_swapped_candidate_change / max(episode_count, 1),
        "correct_counterfactual_switch_rate": correct_switch / max(episode_count, 1),
        "twin_pair_joint_normal_accuracy": joint_normal / max(pair_count, 1),
        "twin_pair_mapping_flip_consistency": mapping_flip / max(pair_count, 1),
        "normal_vs_pair_order_same_grid_rate": normal_shuffled_same_grid / max(episode_count, 1),
        "normal_vs_pair_order_same_candidate_rate": normal_shuffled_same_candidate / max(episode_count, 1),
        "normal_vs_pair_order_same_logical_pair_rate": normal_shuffled_same_logical_pair / max(episode_count, 1),
        "max_abs_probability_difference_after_logical_unpermute": max_alpha_diff,
    }


def evaluate_all_conditions(
    *,
    model: PairKeyValueReader,
    episodes: list[TwinEpisode],
    split: str,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    conditions: dict[str, Any] = {}
    per_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        aggregate, rows = evaluate_condition(
            model=model,
            episodes=episodes,
            split=split,
            condition=condition,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )
        conditions[condition] = aggregate
        per_condition[condition] = rows
    return {"conditions": conditions, "derived_metrics": derived_metrics(episodes, per_condition)}, per_condition


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["retrieval_top1_accuracy"]),
        float(row["direct_grid_exact_accuracy"]),
        float(row["correct_counterfactual_switch_rate"]),
        -float(row["mean_target_rectangle_hamming_error"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model: PairKeyValueReader,
    optimizer: torch.optim.Optimizer,
    config: PairReaderConfig,
    epoch_index: int,
    global_step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch_index": epoch_index,
            "global_step": global_step,
            "extra": extra or {},
        },
        path,
    )


def train_batch(
    *,
    model: PairKeyValueReader,
    optimizer: torch.optim.Optimizer,
    train_episodes: list[TwinEpisode],
    batch_indices: list[int],
    epoch_index: int,
    global_step: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    view_batch = [epoch_training_view(train_episodes[idx], base_seed=args.seed, epoch_index=epoch_index)[0] for idx in batch_indices]
    source_batch = [train_episodes[idx] for idx in batch_indices]
    batch = batch_tensors(view_batch, source_batch, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss, components = pair_reader_loss(
        outputs,
        batch["target_grids"],
        batch["relevant_pair_indices"],
        lambda_retrieval=args.lambda_retrieval,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite pair reader loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm))
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise RuntimeError("non-finite pair reader parameter")
    with torch.no_grad():
        alpha = outputs["retrieval_weights"]
        top1 = (alpha.argmax(dim=-1) == batch["relevant_pair_indices"]).float().mean().item()
        mass = alpha.gather(1, batch["relevant_pair_indices"].view(-1, 1)).mean().item()
    return {
        "event": "train_step",
        "epoch_index": epoch_index,
        "global_step": global_step,
        "episode_indices": batch_indices,
        "train_loss": float(loss.detach().cpu().item()),
        "output_loss": float(components["output_loss"].cpu().item()),
        "retrieval_loss": float(components["retrieval_loss"].cpu().item()),
        "batch_retrieval_top1_accuracy": float(top1),
        "batch_mean_relevant_pair_mass": float(mass),
        "grad_norm_before_clip": grad_norm,
    }


def run_pytest(out: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def r1_gates(oracle: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    sealed = final["sealed"]
    normal = sealed["conditions"][CONDITION_NORMAL]
    derived = sealed["derived_metrics"]
    gate_defs = {
        "oracle_direct_exact": (oracle["sealed"]["direct_grid_exact_accuracy"], "==", 1.0),
        "retrieval_top1_accuracy": (normal["retrieval_top1_accuracy"], ">=", 0.95),
        "mean_relevant_pair_mass": (normal["mean_relevant_pair_mass"], ">=", 0.95),
        "direct_grid_exact_accuracy": (normal["direct_grid_exact_accuracy"], ">=", 0.95),
        "foreground_color_accuracy": (normal["foreground_color_accuracy"], ">=", 0.95),
        "twin_same_predicted_grid_rate": (derived["twin_same_predicted_grid_rate"], "<=", 0.05),
        "correct_counterfactual_switch_rate": (derived["correct_counterfactual_switch_rate"], ">=", 0.95),
        "twin_pair_mapping_flip_consistency": (derived["twin_pair_mapping_flip_consistency"], ">=", 0.95),
        "normal_vs_pair_order_same_grid_rate": (derived["normal_vs_pair_order_same_grid_rate"], ">=", 0.99),
        "normal_vs_pair_order_same_logical_pair_rate": (
            derived["normal_vs_pair_order_same_logical_pair_rate"],
            ">=",
            0.99,
        ),
    }
    gates: dict[str, Any] = {}
    for name, (value, op, threshold) in gate_defs.items():
        passed = value == threshold if op == "==" else value >= threshold if op == ">=" else value <= threshold
        gates[name] = {"value": value, "operator": op, "threshold": threshold, "passed": bool(passed)}
    failed = [name for name, row in gates.items() if not row["passed"]]
    return {"gates": gates, "failed_gates": failed, "passed": not failed}


def verdict_from_gates(gates: dict[str, Any], final: dict[str, Any]) -> str:
    if gates["passed"]:
        return "PASS"
    failed = set(gates["failed_gates"])
    if "oracle_direct_exact" in failed:
        return "ADAPTER_LABEL_BUG"
    if failed & {"normal_vs_pair_order_same_grid_rate", "normal_vs_pair_order_same_logical_pair_rate"}:
        return "PERMUTATION_BUG"
    normal = final["sealed"]["conditions"][CONDITION_NORMAL]
    train_normal = final["train"]["conditions"][CONDITION_NORMAL]
    if failed & {"retrieval_top1_accuracy", "mean_relevant_pair_mass"}:
        if train_normal["retrieval_top1_accuracy"] < 0.95:
            return "OPTIMIZATION_INSTABILITY"
        return "MATCHER_FAILURE"
    if failed & {"direct_grid_exact_accuracy", "foreground_color_accuracy"}:
        return "VALUE_ROUTE_FAILURE"
    if failed & {"twin_same_predicted_grid_rate", "correct_counterfactual_switch_rate", "twin_pair_mapping_flip_consistency"}:
        if normal["retrieval_top1_accuracy"] >= 0.95:
            return "VALUE_ROUTE_FAILURE"
        return "MATCHER_FAILURE"
    return "INCONCLUSIVE"


def comparison_markdown(final: dict[str, Any], gates: dict[str, Any], verdict: str) -> str:
    headers = [
        "split",
        "condition",
        "retrieval top1",
        "mean relevant mass",
        "direct exact",
        "foreground acc",
        "all-cell acc",
        "candidate score",
        "tie rate",
        "twin same-grid",
        "correct swap",
        "pair-order same logical",
        "mean Hamming",
    ]
    lines = [
        "# Pair Key-Value Reader R1",
        "",
        "Standalone learned associative-memory unit test on the frozen Twin dataset.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for split in ("train", "dev", "sealed"):
        derived = final[split]["derived_metrics"]
        for condition, row in final[split]["conditions"].items():
            values = [
                split,
                condition,
                f"{row['retrieval_top1_accuracy']:.6f}",
                f"{row['mean_relevant_pair_mass']:.6f}",
                f"{row['direct_grid_exact_accuracy']:.6f}",
                f"{row['foreground_color_accuracy']:.6f}",
                f"{row['all_cell_color_accuracy']:.6f}",
                f"{row['candidate_nearest_score']:.6f}",
                f"{row['candidate_tie_rate']:.6f}",
                f"{derived['twin_same_predicted_grid_rate']:.6f}",
                f"{derived['correct_counterfactual_switch_rate']:.6f}",
                f"{derived['normal_vs_pair_order_same_logical_pair_rate']:.6f}",
                f"{row['mean_target_rectangle_hamming_error']:.6f}",
            ]
            lines.append("|" + "|".join(values) + "|")
    lines.extend(["", "## Sealed Gates", ""])
    for name, row in gates["gates"].items():
        lines.append(
            f"- {name}: {row['value']:.6f} {row['operator']} {row['threshold']:.6f} -> {'PASS' if row['passed'] else 'FAIL'}"
        )
    lines.extend(
        [
            "",
            f"Verdict: `{verdict}`",
            "",
            "If R1 passes, it proves only that this standalone learned reader solves the marker Twin correspondence unit test.",
            "It does not prove VRM is repaired, ARC is solved, or that raw support-output copying is sufficient for ARC.",
        ]
    )
    return "\n".join(lines) + "\n"


def grid_block(grid: list[list[int]] | tuple[tuple[int, ...], ...]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def write_prediction_examples(
    path: Path,
    episodes: list[TwinEpisode],
    per_condition: dict[str, list[dict[str, Any]]],
) -> None:
    normal = by_episode_id(per_condition[CONDITION_NORMAL])
    swapped = by_episode_id(per_condition[CONDITION_TWIN_SWAP_SUPPORTS])
    shuffled = by_episode_id(per_condition[CONDITION_PAIR_ORDER_SHUFFLED])
    chosen: list[tuple[str, TwinEpisode]] = []
    for episode in episodes:
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        if n["direct_grid_exact"] and s["candidate_decision"] == episode.alternate_label:
            chosen.append(("correct normal-and-swap", episode))
        if len([item for item in chosen if item[0] == "correct normal-and-swap"]) >= 6:
            break
    for episode in episodes:
        n = normal[episode.episode_id]
        p = shuffled[episode.episode_id]
        if n["prediction_sha256"] == p["prediction_sha256"] and n["selected_logical_support_input_hash"] == p["selected_logical_support_input_hash"]:
            chosen.append(("pair-order-shuffled invariant", episode))
        if len([item for item in chosen if item[0] == "pair-order-shuffled invariant"]) >= 6:
            break
    for episode in episodes:
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        p = shuffled[episode.episode_id]
        if not n["direct_grid_exact"] or s["candidate_decision"] != episode.alternate_label or n["prediction_sha256"] != p["prediction_sha256"]:
            chosen.append(("failure", episode))
        if len([item for item in chosen if item[0] == "failure"]) >= 6:
            break
    lines: list[str] = []
    for idx, (kind, episode) in enumerate(chosen, start=1):
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        p = shuffled[episode.episode_id]
        lines.append(f"## Example {idx}: {kind}")
        lines.append(f"episode_id: {episode.episode_id}")
        lines.append(f"relevant logical pair: {relevant_pair_index(episode)}")
        lines.append("query:")
        lines.append(grid_block(episode.query_input.cells))
        for pair_idx, pair in enumerate(episode.supports):
            lines.append(f"support {pair_idx} input:")
            lines.append(grid_block(pair.input.cells))
            lines.append(f"support {pair_idx} output:")
            lines.append(grid_block(pair.output.cells))
        lines.append(f"alpha NORMAL: {n['retrieval_alpha']}")
        lines.append(f"alpha TWIN_SWAP_SUPPORTS: {s['retrieval_alpha']}")
        lines.append(f"alpha PAIR_ORDER_SHUFFLED: {p['retrieval_alpha']}")
        lines.append("retrieved/decoded grid NORMAL:")
        lines.append(grid_block(n["decoded_grid"]))
        lines.append("target grid:")
        lines.append(grid_block(n["target_grid"]))
        lines.append(f"candidate decision NORMAL: {n['candidate_decision_name']}")
        lines.append(f"candidate decision TWIN_SWAP_SUPPORTS: {s['candidate_decision_name']}")
        lines.append(f"output switched correctly: {s['candidate_decision'] == episode.alternate_label}")
        lines.append(
            "pair-order invariance held: "
            f"{n['prediction_sha256'] == p['prediction_sha256'] and n['selected_logical_support_input_hash'] == p['selected_logical_support_input_hash']}"
        )
        lines.append("")
    save_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    ckpt_dir = args.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    environment = cuda_environment_report()
    print(json.dumps(environment, indent=2, sort_keys=True), flush=True)
    save_json(out / "environment_report.json", environment)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    determinism = configure_seed(args.seed)

    before = protected_hashes()
    save_json(out / "source_protection_hashes_before.json", before)

    oracle_episodes = load_all_for_oracle(args.data)
    oracle = oracle_adapter_report(oracle_episodes)
    save_json(out / "oracle_adapter_report.json", oracle)
    if not oracle["passed"]:
        print("ORACLE_ADAPTER_VALIDATION_FAILED", flush=True)
        raise SystemExit("ORACLE_ADAPTER_VALIDATION_FAILED")
    save_json(out / "data_manifest.json", data_manifest(args.data, oracle_episodes))
    del oracle_episodes

    train_episodes = load_split_episodes(args.data, "train", sealed_allowed=False)
    dev_episodes = load_split_episodes(args.data, "dev", sealed_allowed=False)
    steps_per_epoch = math.ceil(len(train_episodes) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    config = PairReaderConfig(hidden_size=args.hidden_size, temperature=args.temperature)
    model = PairKeyValueReader(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    run_config = {
        "experiment": "Pair Key-Value Reader R1 - Learned Correspondence Unit Test",
        "mode": args.mode,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_episode_count": len(train_episodes),
        "dev_episode_count": len(dev_episodes),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "grad_clip_norm": args.grad_clip_norm,
        "lambda_retrieval": args.lambda_retrieval,
        "device": str(device),
        "determinism": determinism,
        "model_config": asdict(config),
        "data": str(args.data),
        "out": str(out),
        "checkpoint_dir": str(ckpt_dir),
    }
    save_json(out / "run_config.json", run_config)
    save_json(out / "initialization_report.json", model.initialization_report())
    save_json(out / "support_pair_permutation_report.json", support_pair_order_report(train_episodes, base_seed=args.seed, epochs=max(2, args.epochs)))

    save_checkpoint(
        ckpt_dir / "pair_reader_r1_initial.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        epoch_index=-1,
        global_step=0,
        extra={"fresh_initialization": True, "loaded_checkpoint": False},
    )

    log_path = out / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    observed_by_epoch: dict[int, list[int]] = {epoch: [] for epoch in range(args.epochs)}
    global_step = 0
    start_time = time.perf_counter()
    max_batches_remaining = args.max_batches
    dev_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None

    for epoch_index in range(args.epochs):
        order = epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=epoch_index)
        for batch_indices in epoch_batches(order, args.batch_size):
            global_step += 1
            row = train_batch(
                model=model,
                optimizer=optimizer,
                train_episodes=train_episodes,
                batch_indices=batch_indices,
                epoch_index=epoch_index,
                global_step=global_step,
                args=args,
                device=device,
            )
            observed_by_epoch[epoch_index].extend(batch_indices)
            append_jsonl(log_path, row)
            if max_batches_remaining is not None:
                max_batches_remaining -= 1
                if max_batches_remaining <= 0:
                    dry_report = {
                        "mode": "dry_run",
                        "batches_run": args.max_batches,
                        "global_step": global_step,
                        "finite_parameters": all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()),
                        "fresh_initialization": True,
                        "loaded_checkpoint": False,
                        "candidate_metadata_excluded": True,
                        "target_excluded_from_forward": True,
                        "last_train_loss": row["train_loss"],
                        "last_batch_retrieval_top1_accuracy": row["batch_retrieval_top1_accuracy"],
                        "last_batch_mean_relevant_pair_mass": row["batch_mean_relevant_pair_mass"],
                        "cuda_device": environment["gpu_name"],
                        "protected_hashes_unchanged": before == protected_hashes(),
                    }
                    save_json(out / "dry_run_report.json", dry_report)
                    save_json(out / "source_protection_hashes_after.json", protected_hashes())
                    print(json.dumps(to_jsonable(dry_report), indent=2, sort_keys=True), flush=True)
                    return dry_report

        save_checkpoint(
            ckpt_dir / f"pair_reader_r1_epoch_{epoch_index:02d}.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch_index=epoch_index,
            global_step=global_step,
            extra={"checkpoint_reason": "epoch_end"},
        )
        dev_metrics, _ = evaluate_all_conditions(
            model=model,
            episodes=dev_episodes,
            split="dev",
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
        )
        normal = dict(dev_metrics["conditions"][CONDITION_NORMAL])
        normal["epoch_index"] = epoch_index
        normal["global_step"] = global_step
        normal["correct_counterfactual_switch_rate"] = dev_metrics["derived_metrics"]["correct_counterfactual_switch_rate"]
        normal["twin_same_predicted_grid_rate"] = dev_metrics["derived_metrics"]["twin_same_predicted_grid_rate"]
        normal["selection_key"] = list(selection_key(normal))
        dev_rows.append(normal)
        append_jsonl(log_path, {"event": "dev_epoch_eval", **normal})
        key = selection_key(normal)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch_index
            best_metrics = normal
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            save_checkpoint(
                ckpt_dir / "pair_reader_r1_best_dev.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch_index=epoch_index,
                global_step=global_step,
                extra={"best_dev_metrics": normal},
            )
        print(json.dumps(to_jsonable({"event": "epoch_complete", **normal}), sort_keys=True), flush=True)

    episode_ids = [episode.episode_id for episode in train_episodes]
    coverage = {"all_epochs_passed": True, "epochs": []}
    for epoch_index in range(args.epochs):
        order = epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=epoch_index)
        row = coverage_row(epoch_index=epoch_index, order=order, observed=observed_by_epoch[epoch_index], episode_ids=episode_ids, base_seed=args.seed)
        coverage["epochs"].append(row)
        coverage["all_epochs_passed"] = coverage["all_epochs_passed"] and row["passed"]
    save_json(out / "epoch_coverage_report.json", coverage)
    save_json(out / "dev_metrics_by_epoch.json", {"epochs": dev_rows})
    if best_state is None or best_metrics is None:
        raise RuntimeError("no best dev checkpoint was selected")

    model.load_state_dict({name: tensor.to(device) for name, tensor in best_state.items()})
    save_checkpoint(
        ckpt_dir / "pair_reader_r1_best_dev.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        epoch_index=best_epoch,
        global_step=int(best_metrics["global_step"]),
        extra={"best_checkpoint_selection": best_metrics},
    )
    best_hash_before = sha256_file(ckpt_dir / "pair_reader_r1_best_dev.pt")
    best_selection = {
        "selected_epoch_index": best_epoch,
        "selected_global_step": int(best_metrics["global_step"]),
        "selection_rule": "highest dev retrieval_top1_accuracy; tie-break direct_grid_exact_accuracy, correct_counterfactual_switch_rate, lowest hamming",
        "selection_metrics": best_metrics,
        "selected_checkpoint": str(ckpt_dir / "pair_reader_r1_best_dev.pt"),
        "selected_checkpoint_sha256_before_sealed": best_hash_before,
        "sealed_split_read_before_selection": False,
    }
    save_json(out / "best_checkpoint_selection.json", best_selection)

    sealed_episodes = load_split_episodes(args.data, "sealed", sealed_allowed=(out / "best_checkpoint_selection.json").exists())
    final: dict[str, Any] = {}
    per_episode: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, episodes in (("train", train_episodes), ("dev", dev_episodes), ("sealed", sealed_episodes)):
        metrics, rows = evaluate_all_conditions(
            model=model,
            episodes=episodes,
            split=split,
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
        )
        final[split] = metrics
        per_episode[split] = rows
        save_json(out / f"final_{split}_metrics.json", metrics)
        save_json(out / f"per_episode_{split}.json", rows)

    best_hash_after = sha256_file(ckpt_dir / "pair_reader_r1_best_dev.pt")
    best_selection["selected_checkpoint_sha256_after_sealed"] = best_hash_after
    best_selection["checkpoint_hash_unchanged_after_sealed"] = best_hash_before == best_hash_after
    save_json(out / "best_checkpoint_selection.json", best_selection)

    gates = r1_gates(oracle, final)
    verdict = verdict_from_gates(gates, final)
    save_json(out / "pair_order_invariance_report.json", {split: final[split]["derived_metrics"] for split in final})
    save_json(out / "twin_counterfactual_report.json", {split: final[split]["derived_metrics"] for split in final})
    write_prediction_examples(out / "prediction_examples.txt", sealed_episodes, per_episode["sealed"])
    pytest_result = run_pytest(out)
    after = protected_hashes()
    save_json(out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "gates": gates,
        "runtime_seconds": time.perf_counter() - start_time,
        "environment": {**environment, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "oracle": oracle,
        "coverage_passed": coverage["all_epochs_passed"],
        "support_pair_permutation_passed": json.loads((out / "support_pair_permutation_report.json").read_text())["passed"],
        "best_checkpoint_selection": best_selection,
        "source_protection_hashes_unchanged": before == after,
        "pytest_result": pytest_result,
        "final_metrics": final,
        "interpretation": {
            "if_passes": "The standalone learned reader can solve the marker Twin correspondence unit test.",
            "does_not_prove": [
                "VRM is repaired",
                "ARC is solved",
                "raw support-output copying is sufficient for ARC",
                "R1 should be inserted into V1.1 without a separate integration test",
            ],
            "r2_integration_justified": verdict == "PASS",
        },
    }
    save_json(out / "summary_report.json", summary)
    save_text(out / "summary.md", comparison_markdown(final, gates, verdict))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair Key-Value Reader R1")
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-retrieval", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

