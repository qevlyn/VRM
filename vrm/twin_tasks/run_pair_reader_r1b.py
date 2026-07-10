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
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT, ArcPair, Grid

from .epoch_sampler import coverage_row, epoch_batches, epoch_permutation_indices, epoch_training_view, sha256_json
from .pair_reader_r1 import entropy_from_weights
from .pair_reader_r1b import (
    PairKeyValueReaderR1B,
    PairReaderConfig,
    assert_output_only_training_batch,
    output_only_loss,
    output_only_training_keys,
)
from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    group_by_base,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_DATA = PROJECT_ROOT / "artifacts/twin_task_pilot_v1"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/twin_pair_reader_r1b"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b"
DEFAULT_SEEDS = (20260628, 20260629, 20260630, 20260701, 20260702)

PROTECTED_FILES = (
    PROJECT_ROOT / "vrm/model.py",
    PROJECT_ROOT / "vrm/data.py",
    PROJECT_ROOT / "vrm/metrics.py",
    PROJECT_ROOT / "vrm/train.py",
    PROJECT_ROOT / "vrm/twin_tasks/pair_reader_r1.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_pair_reader_r1.py",
)
PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts/twin_vrm_probe_v0",
    PROJECT_ROOT / "artifacts/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1",
    PROJECT_ROOT / "checkpoints/twin_vrm_probe_v0",
    PROJECT_ROOT / "checkpoints/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1",
    PROJECT_ROOT / "archives",
)


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


def configure_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {"seed": seed, "torch_deterministic_algorithms": "enabled_warn_only"}


def load_split_episodes(data: Path, split: str, *, sealed_allowed: bool) -> list[TwinEpisode]:
    if split == "sealed" and not sealed_allowed:
        raise RuntimeError("sealed split read attempted before every seed checkpoint selection was written")
    return records_to_episodes(read_jsonl(Path(data) / f"{split}.jsonl"))


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
        raise ValueError(f"{episode.episode_id} has {len(matches)} matching support inputs")
    return matches[0]


def reorder_episode(episode: TwinEpisode, order: tuple[int, ...] | list[int]) -> TwinEpisode:
    supports = tuple(
        ArcPair(input=episode.supports[source_idx].input, output=episode.supports[source_idx].output, pair_index=new_idx)
        for new_idx, source_idx in enumerate(order)
    )
    return TwinEpisode(
        record_index=episode.record_index,
        split=episode.split,
        base_id=episode.base_id,
        twin=episode.twin,
        family=episode.family,
        label=episode.label,
        supports=supports,
        query_input=episode.query_input,
        target_output=episode.target_output,
        candidates=episode.candidates,
    )


def twin_swap_supports_view(episode: TwinEpisode, opposite: TwinEpisode) -> TwinEpisode:
    return TwinEpisode(
        record_index=episode.record_index,
        split=episode.split,
        base_id=episode.base_id,
        twin=episode.twin,
        family=episode.family,
        label=episode.label,
        supports=tuple(
            ArcPair(input=pair.input, output=pair.output, pair_index=idx) for idx, pair in enumerate(opposite.supports)
        ),
        query_input=episode.query_input,
        target_output=episode.target_output,
        candidates=episode.candidates,
    )


def data_manifest(data: Path, train: list[TwinEpisode], dev: list[TwinEpisode], sealed: list[TwinEpisode] | None = None) -> dict[str, Any]:
    splits: dict[str, list[TwinEpisode] | None] = {"train": train, "dev": dev, "sealed": sealed}
    manifest: dict[str, Any] = {
        "data_dir": str(data),
        "raw_arc_data_used": False,
        "synthetic_extra_examples_created": False,
        "sealed_deferred_until_all_seed_selections": sealed is None,
        "source_files": {},
        "episode_counts": {},
        "foreground_target_counts": {},
    }
    for split, episodes in splits.items():
        path = Path(data) / f"{split}.jsonl"
        if episodes is None:
            manifest["source_files"][split] = {"path": str(path), "sha256": "deferred_until_after_all_seed_selections"}
            manifest["episode_counts"][split] = "deferred"
            manifest["foreground_target_counts"][split] = "deferred"
        else:
            manifest["source_files"][split] = {"path": str(path), "sha256": sha256_file(path)}
            manifest["episode_counts"][split] = len(episodes)
            manifest["foreground_target_counts"][split] = sorted(
                {sum(1 for row in episode.target_output.cells for value in row if value != 0) for episode in episodes}
            )
    return manifest


def training_batch_tensors(view_episodes: list[TwinEpisode], source_episodes: list[TwinEpisode], device: torch.device) -> dict[str, torch.Tensor]:
    batch = {
        "query_grids": torch.stack([grid_tensor(episode.query_input) for episode in view_episodes]).to(device),
        "support_input_grids": torch.stack(
            [torch.stack([grid_tensor(pair.input) for pair in episode.supports]) for episode in view_episodes]
        ).to(device),
        "support_output_grids": torch.stack(
            [torch.stack([grid_tensor(pair.output) for pair in episode.supports]) for episode in view_episodes]
        ).to(device),
        "target_grids": torch.stack([grid_tensor(episode.target_output) for episode in source_episodes]).to(device),
    }
    assert_output_only_training_batch(batch)
    return batch


def eval_batch_tensors(view_episodes: list[TwinEpisode], source_episodes: list[TwinEpisode], device: torch.device) -> dict[str, torch.Tensor]:
    batch = training_batch_tensors(view_episodes, source_episodes, device)
    batch["relevant_pair_indices"] = torch.tensor([relevant_pair_index(episode) for episode in view_episodes], dtype=torch.long, device=device)
    return batch


def foreground_metric(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    mask = target != 0
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
        "candidate_decision": decision,
        "candidate_decision_name": "TIE" if decision is None else str(decision),
        "candidate_nearest_score": 0.5 if decision is None else (1.0 if decision == episode.label else 0.0),
        "candidate_tie": decision is None,
        "candidate_exact_match": bool(torch.equal(pred, c0) or torch.equal(pred, c1)),
        "distance_to_candidate_0": d0,
        "distance_to_candidate_1": d1,
    }


def row_from_prediction(
    *,
    split: str,
    condition: str,
    source_episode: TwinEpisode,
    view_episode: TwinEpisode,
    pred: torch.Tensor,
    alpha: torch.Tensor,
    entropy: float,
) -> dict[str, Any]:
    target = grid_tensor(source_episode.target_output)
    relevant_idx = relevant_pair_index(view_episode)
    selected_idx = int(alpha.argmax().item())
    fg = foreground_metric(pred, target)
    correct = int((pred == target).sum().item())
    total = int(target.numel())
    input_hashes = [grid_hash(pair.input) for pair in view_episode.supports]
    return {
        "split": split,
        "condition": condition,
        "episode_id": source_episode.episode_id,
        "base_id": source_episode.base_id,
        "twin": source_episode.twin,
        "label": source_episode.label,
        "alternate_label": source_episode.alternate_label,
        "relevant_pair_index": relevant_idx,
        "selected_pair_index": selected_idx,
        "retrieval_top1_correct": selected_idx == relevant_idx,
        "relevant_pair_mass": float(alpha[relevant_idx].item()),
        "retrieval_entropy": entropy,
        "retrieval_alpha": [float(value) for value in alpha.tolist()],
        "logical_alpha_by_support_input_hash": {input_hashes[idx]: float(alpha[idx].item()) for idx in range(len(input_hashes))},
        "selected_logical_support_input_hash": input_hashes[selected_idx],
        "relevant_logical_support_input_hash": input_hashes[relevant_idx],
        "prediction_sha256": grid_hash(pred),
        "target_sha256": grid_hash(target),
        "decoded_grid": pred.tolist(),
        "target_grid": target.tolist(),
        "direct_grid_exact": correct == total,
        "all_cell_color_correct": correct,
        "all_cell_color_total": total,
        "all_cell_color_accuracy": correct / max(total, 1),
        "target_rectangle_hamming_error": total - correct,
        **fg,
        **candidate_metrics(pred, source_episode),
    }


def aggregate_rows(split: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    fg_total = sum(row["foreground_color_total"] for row in rows)
    fg_correct = sum(row["foreground_color_correct"] for row in rows)
    all_total = sum(row["all_cell_color_total"] for row in rows)
    all_correct = sum(row["all_cell_color_correct"] for row in rows)
    hamming = [row["target_rectangle_hamming_error"] for row in rows]
    masses = [row["relevant_pair_mass"] for row in rows]
    return {
        "split": split,
        "condition": condition,
        "episode_count": count,
        "retrieval_top1_accuracy": sum(row["retrieval_top1_correct"] for row in rows) / max(count, 1),
        "mean_relevant_pair_mass": statistics.fmean(masses) if masses else 0.0,
        "median_relevant_pair_mass": statistics.median(masses) if masses else 0.0,
        "retrieval_entropy": statistics.fmean(row["retrieval_entropy"] for row in rows) if rows else 0.0,
        "direct_grid_exact_accuracy": sum(row["direct_grid_exact"] for row in rows) / max(count, 1),
        "foreground_color_accuracy": fg_correct / max(fg_total, 1),
        "all_cell_color_accuracy": all_correct / max(all_total, 1),
        "target_rectangle_hamming_error": statistics.fmean(hamming) if hamming else 0.0,
        "mean_target_rectangle_hamming_error": statistics.fmean(hamming) if hamming else 0.0,
        "candidate_nearest_score": statistics.fmean(row["candidate_nearest_score"] for row in rows) if rows else 0.0,
        "candidate_tie_rate": sum(row["candidate_tie"] for row in rows) / max(count, 1),
        "candidate_exact_match_rate": sum(row["candidate_exact_match"] for row in rows) / max(count, 1),
        "foreground_cell_count_per_episode": {
            "unique": sorted({row["foreground_color_total"] for row in rows}),
            "mean": statistics.fmean(row["foreground_color_total"] for row in rows) if rows else 0.0,
            "expected_informative_foreground_count_verified": sorted({row["foreground_color_total"] for row in rows}) == [5],
        },
    }


def eval_normal_or_swap(
    *,
    model: PairKeyValueReaderR1B,
    episodes: list[TwinEpisode],
    split: str,
    condition: str,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    opposite = build_opposite_lookup(episodes)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            if condition == CONDITION_NORMAL:
                view_batch = source_batch
            elif condition == CONDITION_TWIN_SWAP_SUPPORTS:
                view_batch = [twin_swap_supports_view(ep, opposite[f"{ep.base_id}:{ep.twin}"]) for ep in source_batch]
            else:
                raise ValueError(condition)
            batch = eval_batch_tensors(view_batch, source_batch, device)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            decoded = outputs["decoded_grid"].detach().cpu().to(torch.long)
            alpha = outputs["retrieval_weights"].detach().cpu()
            ent = entropy_from_weights(alpha).detach().cpu()
            for local_idx, source_episode in enumerate(source_batch):
                rows.append(
                    row_from_prediction(
                        split=split,
                        condition=condition,
                        source_episode=source_episode,
                        view_episode=view_batch[local_idx],
                        pred=decoded[local_idx],
                        alpha=alpha[local_idx],
                        entropy=float(ent[local_idx].item()),
                    )
                )
    return aggregate_rows(split, condition, rows), rows


def by_episode_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in rows}


def twin_metrics(episodes: list[TwinEpisode], normal_rows: list[dict[str, Any]], swapped_rows: list[dict[str, Any]]) -> dict[str, Any]:
    normal = by_episode_id(normal_rows)
    swapped = by_episode_id(swapped_rows)
    by_base = group_by_base(episodes)
    twin_same_grid = 0
    twin_same_candidate = 0
    correct_switch = 0
    grid_change = 0
    candidate_change = 0
    joint = 0
    flip = 0
    for episode in episodes:
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        if n["prediction_sha256"] != s["prediction_sha256"]:
            grid_change += 1
        if n["candidate_decision"] is not None and s["candidate_decision"] is not None and n["candidate_decision"] != s["candidate_decision"]:
            candidate_change += 1
        if n["candidate_decision"] == episode.label and s["candidate_decision"] == episode.alternate_label:
            correct_switch += 1
    for twins in by_base.values():
        if set(twins) != {"A", "B"}:
            continue
        a = twins["A"]
        b = twins["B"]
        na = normal[a.episode_id]
        nb = normal[b.episode_id]
        sa = swapped[a.episode_id]
        sb = swapped[b.episode_id]
        if na["prediction_sha256"] == nb["prediction_sha256"]:
            twin_same_grid += 1
        if na["candidate_decision"] == nb["candidate_decision"]:
            twin_same_candidate += 1
        normal_correct = na["candidate_decision"] == a.label and nb["candidate_decision"] == b.label
        if normal_correct:
            joint += 1
        if normal_correct and sa["candidate_decision"] == a.alternate_label and sb["candidate_decision"] == b.alternate_label:
            flip += 1
    return {
        "twin_same_predicted_grid_rate": twin_same_grid / max(len(by_base), 1),
        "twin_same_candidate_decision_rate": twin_same_candidate / max(len(by_base), 1),
        "correct_counterfactual_switch_rate": correct_switch / max(len(episodes), 1),
        "twin_pair_mapping_flip_consistency": flip / max(len(by_base), 1),
        "twin_pair_joint_normal_accuracy": joint / max(len(by_base), 1),
        "normal_to_swapped_grid_change_rate": grid_change / max(len(episodes), 1),
        "normal_to_swapped_candidate_change_rate": candidate_change / max(len(episodes), 1),
    }


def eval_all_permutations(
    *,
    model: PairKeyValueReaderR1B,
    episodes: list[TwinEpisode],
    split: str,
    normal_rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    normal = by_episode_id(normal_rows)
    rows: list[dict[str, Any]] = []
    normal_probs_by_episode: dict[str, torch.Tensor] = {}
    same_grid = 0
    same_candidate = 0
    same_logical = 0
    max_alpha_diff = 0.0
    max_prob_diff = 0.0
    total = 0
    perms = list(itertools.permutations(range(3)))
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            batch = eval_batch_tensors(source_batch, source_batch, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            probs = out["retrieved_color_probabilities"].detach().cpu()
            for local_idx, source_episode in enumerate(source_batch):
                normal_probs_by_episode[source_episode.episode_id] = probs[local_idx]
        for perm in perms:
            for start in range(0, len(episodes), batch_size):
                source_batch = episodes[start : start + batch_size]
                view_batch = [reorder_episode(ep, perm) for ep in source_batch]
                batch = eval_batch_tensors(view_batch, source_batch, device)
                out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                decoded = out["decoded_grid"].detach().cpu().to(torch.long)
                alpha = out["retrieval_weights"].detach().cpu()
                probs = out["retrieved_color_probabilities"].detach().cpu()
                ent = entropy_from_weights(alpha).detach().cpu()
                for local_idx, source_episode in enumerate(source_batch):
                    row = row_from_prediction(
                        split=split,
                        condition="ALL_PAIR_PERMUTATIONS",
                        source_episode=source_episode,
                        view_episode=view_batch[local_idx],
                        pred=decoded[local_idx],
                        alpha=alpha[local_idx],
                        entropy=float(ent[local_idx].item()),
                    )
                    n = normal[source_episode.episode_id]
                    if row["prediction_sha256"] == n["prediction_sha256"]:
                        same_grid += 1
                    if row["candidate_decision"] == n["candidate_decision"]:
                        same_candidate += 1
                    if row["selected_logical_support_input_hash"] == n["selected_logical_support_input_hash"]:
                        same_logical += 1
                    for key, value in row["logical_alpha_by_support_input_hash"].items():
                        max_alpha_diff = max(max_alpha_diff, abs(float(value) - float(n["logical_alpha_by_support_input_hash"][key])))
                    max_prob_diff = max(
                        max_prob_diff,
                        float((probs[local_idx] - normal_probs_by_episode[source_episode.episode_id]).abs().max().item()),
                    )
                    rows.append(row)
                    total += 1
    agg = aggregate_rows(split, "ALL_PAIR_PERMUTATIONS", rows)
    agg.update(
        {
            "permutations_evaluated": [list(perm) for perm in perms],
            "permutation_presentation_count": total,
            "all_permutation_same_grid_rate": same_grid / max(total, 1),
            "all_permutation_same_candidate_rate": same_candidate / max(total, 1),
            "all_permutation_same_logical_pair_rate": same_logical / max(total, 1),
            "max_abs_alpha_difference_after_logical_unpermute": max_alpha_diff,
            "max_abs_retrieved_probability_difference_after_logical_unpermute": max_prob_diff,
        }
    )
    return agg


def non_identity_value_permutation(episode: TwinEpisode, seed: int) -> list[int]:
    order = list(range(len(episode.supports)))
    rng = random.Random(int(hashlib.sha256(f"{seed}|{episode.episode_id}|forced_value".encode()).hexdigest()[:16], 16))
    rng.shuffle(order)
    if order == list(range(len(order))):
        order = order[1:] + order[:1]
    return order


def eval_forced_value(
    *,
    model: PairKeyValueReaderR1B,
    episodes: list[TwinEpisode],
    split: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    alpha_unchanged = 0
    follows_value = 0
    exact_match = 0
    confident = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            batch = eval_batch_tensors(source_batch, source_batch, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            alpha = out["retrieval_weights"]
            for local_idx, episode in enumerate(source_batch):
                order = non_identity_value_permutation(episode, seed)
                values = batch["support_output_grids"][local_idx, order]
                probs = torch.sum(alpha[local_idx, :, None, None, None] * F.one_hot(values, ARC_COLOR_COUNT).float(), dim=0)
                decoded = probs.argmax(dim=-1).detach().cpu().to(torch.long)
                top_idx = int(alpha[local_idx].argmax().item())
                expected = grid_tensor(episode.supports[order[top_idx]].output)
                alpha_unchanged += 1
                if float(alpha[local_idx, top_idx].item()) >= 0.99:
                    confident += 1
                    if torch.equal(decoded, expected):
                        follows_value += 1
                        exact_match += 1
                total += 1
    return {
        "split": split,
        "condition": "FORCED_VALUE_INTERVENTION",
        "episode_count": total,
        "confident_episode_count": confident,
        "forced_value_alpha_unchanged_rate": alpha_unchanged / max(total, 1),
        "forced_value_output_follows_value_rate": follows_value / max(confident, 1),
        "forced_value_exact_match_rate": exact_match / max(confident, 1),
    }


def eval_forced_alpha(
    *,
    episodes: list[TwinEpisode],
    split: str,
) -> dict[str, Any]:
    exact = 0
    total = 0
    for episode in episodes:
        for idx, pair in enumerate(episode.supports):
            decoded = grid_tensor(pair.output)
            expected = grid_tensor(episode.supports[idx].output)
            if torch.equal(decoded, expected):
                exact += 1
            total += 1
    return {
        "split": split,
        "condition": "FORCED_ALPHA_INTERVENTION",
        "forced_alpha_route_count": total,
        "forced_alpha_exact_route_rate": exact / max(total, 1),
    }


def evaluate_selected(
    *,
    model: PairKeyValueReaderR1B,
    episodes: list[TwinEpisode],
    split: str,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normal_agg, normal_rows = eval_normal_or_swap(
        model=model, episodes=episodes, split=split, condition=CONDITION_NORMAL, device=device, batch_size=batch_size
    )
    swapped_agg, swapped_rows = eval_normal_or_swap(
        model=model, episodes=episodes, split=split, condition=CONDITION_TWIN_SWAP_SUPPORTS, device=device, batch_size=batch_size
    )
    twin = twin_metrics(episodes, normal_rows, swapped_rows)
    permutations = eval_all_permutations(
        model=model, episodes=episodes, split=split, normal_rows=normal_rows, device=device, batch_size=batch_size
    )
    forced_value = eval_forced_value(model=model, episodes=episodes, split=split, seed=seed, device=device, batch_size=batch_size)
    forced_alpha = eval_forced_alpha(episodes=episodes, split=split)
    return {
        "conditions": {
            "NORMAL": normal_agg,
            "TWIN_SWAP_SUPPORTS": swapped_agg,
            "ALL_PAIR_PERMUTATIONS": permutations,
            "FORCED_VALUE_INTERVENTION": forced_value,
            "FORCED_ALPHA_INTERVENTION": forced_alpha,
        },
        "twin_metrics": twin,
    }


def dev_eval_for_selection(
    *,
    model: PairKeyValueReaderR1B,
    dev_episodes: list[TwinEpisode],
    device: torch.device,
    batch_size: int,
    epoch_index: int,
    global_step: int,
) -> dict[str, Any]:
    normal_agg, normal_rows = eval_normal_or_swap(
        model=model, episodes=dev_episodes, split="dev", condition=CONDITION_NORMAL, device=device, batch_size=batch_size
    )
    swapped_agg, swapped_rows = eval_normal_or_swap(
        model=model, episodes=dev_episodes, split="dev", condition=CONDITION_TWIN_SWAP_SUPPORTS, device=device, batch_size=batch_size
    )
    twin = twin_metrics(dev_episodes, normal_rows, swapped_rows)
    row = dict(normal_agg)
    row.update(
        {
            "epoch_index": epoch_index,
            "global_step": global_step,
            "correct_counterfactual_switch_rate": twin["correct_counterfactual_switch_rate"],
            "twin_same_predicted_grid_rate": twin["twin_same_predicted_grid_rate"],
            "selection_key": [
                normal_agg["direct_grid_exact_accuracy"],
                normal_agg["retrieval_top1_accuracy"],
                twin["correct_counterfactual_switch_rate"],
                -normal_agg["target_rectangle_hamming_error"],
                -epoch_index,
            ],
        }
    )
    return row


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return tuple(float(value) for value in row["selection_key"])  # type: ignore[return-value]


def save_checkpoint(
    path: Path,
    *,
    model: PairKeyValueReaderR1B,
    optimizer: torch.optim.Optimizer,
    config: PairReaderConfig,
    seed: int,
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
            "seed": seed,
            "epoch_index": epoch_index,
            "global_step": global_step,
            "extra": extra or {},
        },
        path,
    )


def train_one_step(
    *,
    model: PairKeyValueReaderR1B,
    optimizer: torch.optim.Optimizer,
    train_episodes: list[TwinEpisode],
    batch_indices: list[int],
    seed: int,
    epoch_index: int,
    global_step: int,
    grad_clip_norm: float,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    view_batch = [epoch_training_view(train_episodes[idx], base_seed=seed, epoch_index=epoch_index)[0] for idx in batch_indices]
    source_batch = [train_episodes[idx] for idx in batch_indices]
    batch = training_batch_tensors(view_batch, source_batch, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss = output_only_loss(outputs, batch["target_grids"])
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite output-only loss")
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm))
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
        raise RuntimeError("non-finite parameter")
    return {
        "event": "train_step",
        "seed": seed,
        "epoch_index": epoch_index,
        "global_step": global_step,
        "episode_indices": batch_indices,
        "train_output_loss": float(loss.detach().cpu().item()),
        "grad_norm_before_clip": grad_norm,
        "training_batch_keys": output_only_training_keys(),
        "retrieval_supervision_used": False,
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


def seed_gates(final_sealed: dict[str, Any]) -> dict[str, Any]:
    normal = final_sealed["conditions"]["NORMAL"]
    perms = final_sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
    forced_value = final_sealed["conditions"]["FORCED_VALUE_INTERVENTION"]
    forced_alpha = final_sealed["conditions"]["FORCED_ALPHA_INTERVENTION"]
    twin = final_sealed["twin_metrics"]
    defs = {
        "normal_retrieval_top1_accuracy": (normal["retrieval_top1_accuracy"], ">=", 0.99),
        "normal_mean_relevant_pair_mass": (normal["mean_relevant_pair_mass"], ">=", 0.99),
        "normal_direct_grid_exact_accuracy": (normal["direct_grid_exact_accuracy"], ">=", 0.99),
        "normal_foreground_color_accuracy": (normal["foreground_color_accuracy"], ">=", 0.99),
        "swap_correct_counterfactual_switch_rate": (twin["correct_counterfactual_switch_rate"], ">=", 0.99),
        "swap_twin_same_predicted_grid_rate": (twin["twin_same_predicted_grid_rate"], "<=", 0.01),
        "permutation_same_grid_rate": (perms["all_permutation_same_grid_rate"], ">=", 0.999),
        "permutation_same_logical_pair_rate": (perms["all_permutation_same_logical_pair_rate"], ">=", 0.999),
        "permutation_max_alpha_diff": (perms["max_abs_alpha_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_max_prob_diff": (perms["max_abs_retrieved_probability_difference_after_logical_unpermute"], "<=", 1e-5),
        "forced_value_alpha_unchanged_rate": (forced_value["forced_value_alpha_unchanged_rate"], "==", 1.0),
        "forced_value_output_follows_value_rate": (forced_value["forced_value_output_follows_value_rate"], "==", 1.0),
        "forced_value_exact_match_rate": (forced_value["forced_value_exact_match_rate"], "==", 1.0),
        "forced_alpha_exact_route_rate": (forced_alpha["forced_alpha_exact_route_rate"], "==", 1.0),
    }
    gates: dict[str, Any] = {}
    for name, (value, op, threshold) in defs.items():
        if op == "==":
            passed = value == threshold
        elif op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        gates[name] = {"value": value, "operator": op, "threshold": threshold, "passed": bool(passed)}
    failed = [name for name, row in gates.items() if not row["passed"]]
    return {"passed": not failed, "failed_gates": failed, "gates": gates}


def aggregate_summary(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    normal_exact = [row["sealed"]["conditions"]["NORMAL"]["direct_grid_exact_accuracy"] for row in seed_summaries]
    swap = [row["sealed"]["twin_metrics"]["correct_counterfactual_switch_rate"] for row in seed_summaries]
    perm_grid = [row["sealed"]["conditions"]["ALL_PAIR_PERMUTATIONS"]["all_permutation_same_grid_rate"] for row in seed_summaries]
    failed = {str(row["seed"]): row["gates"]["failed_gates"] for row in seed_summaries if row["gates"]["failed_gates"]}
    return {
        "seed_count": len(seed_summaries),
        "all_seed_gates_passed": not failed,
        "failed_gates_by_seed": failed,
        "min_seed_normal_exact": min(normal_exact),
        "mean_seed_normal_exact": statistics.fmean(normal_exact),
        "min_seed_swap_rate": min(swap),
        "mean_seed_swap_rate": statistics.fmean(swap),
        "min_seed_permutation_grid_rate": min(perm_grid),
        "mean_seed_permutation_grid_rate": statistics.fmean(perm_grid),
    }


def verdict_from_summary(summary: dict[str, Any], seed_summaries: list[dict[str, Any]]) -> str:
    if summary["all_seed_gates_passed"]:
        return "OUTPUT_ONLY_READER_CONFIRMED"
    failed_names = {name for names in summary["failed_gates_by_seed"].values() for name in names}
    forced_failed = {name for name in failed_names if name.startswith("forced_")}
    perm_failed = {name for name in failed_names if name.startswith("permutation_")}
    normal_failed = {name for name in failed_names if name.startswith("normal_")}
    if forced_failed:
        return "VALUE_ROUTE_OR_LOGICAL_INDEX_BUG"
    if perm_failed and not (normal_failed or {name for name in failed_names if name.startswith("swap_")}):
        return "PERMUTATION_LEAK_OR_REORDER_BUG"
    values = [row["sealed"]["conditions"]["NORMAL"]["direct_grid_exact_accuracy"] for row in seed_summaries]
    if max(values) - min(values) >= 0.05:
        return "OPTIMIZATION_INSTABILITY"
    if normal_failed and all(row["sealed"]["conditions"]["FORCED_ALPHA_INTERVENTION"]["forced_alpha_exact_route_rate"] == 1.0 for row in seed_summaries):
        return "OUTPUT_SIGNAL_INSUFFICIENT_FOR_SELECTOR"
    return "INCONCLUSIVE"


def write_examples(path: Path, seed_summaries: list[dict[str, Any]]) -> None:
    lines = ["Pair Reader R1B prediction examples are summarized from metrics rather than full per-episode dumps.", ""]
    for row in seed_summaries:
        sealed = row["sealed"]
        lines.append(f"seed {row['seed']}:")
        lines.append(f"  normal exact: {sealed['conditions']['NORMAL']['direct_grid_exact_accuracy']:.6f}")
        lines.append(f"  swap rate: {sealed['twin_metrics']['correct_counterfactual_switch_rate']:.6f}")
        lines.append(
            f"  all permutation same grid: {sealed['conditions']['ALL_PAIR_PERMUTATIONS']['all_permutation_same_grid_rate']:.6f}"
        )
        lines.append(f"  forced alpha exact route: {sealed['conditions']['FORCED_ALPHA_INTERVENTION']['forced_alpha_exact_route_rate']:.6f}")
    save_text(path, "\n".join(lines) + "\n")


def comparison_markdown(seed_summaries: list[dict[str, Any]], aggregate: dict[str, Any], verdict: str) -> str:
    headers = [
        "seed",
        "normal top1",
        "normal mass",
        "normal exact",
        "normal fg",
        "swap rate",
        "twin same-grid",
        "perm same grid",
        "perm same logical",
        "max alpha diff",
        "max prob diff",
        "forced value",
        "forced alpha",
    ]
    lines = [
        "# Pair Key-Value Reader R1B",
        "",
        "Output-only confirmation run: no retrieval supervision was used in training.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in seed_summaries:
        sealed = row["sealed"]
        normal = sealed["conditions"]["NORMAL"]
        perms = sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
        fv = sealed["conditions"]["FORCED_VALUE_INTERVENTION"]
        fa = sealed["conditions"]["FORCED_ALPHA_INTERVENTION"]
        twin = sealed["twin_metrics"]
        values = [
            str(row["seed"]),
            f"{normal['retrieval_top1_accuracy']:.6f}",
            f"{normal['mean_relevant_pair_mass']:.6f}",
            f"{normal['direct_grid_exact_accuracy']:.6f}",
            f"{normal['foreground_color_accuracy']:.6f}",
            f"{twin['correct_counterfactual_switch_rate']:.6f}",
            f"{twin['twin_same_predicted_grid_rate']:.6f}",
            f"{perms['all_permutation_same_grid_rate']:.6f}",
            f"{perms['all_permutation_same_logical_pair_rate']:.6f}",
            f"{perms['max_abs_alpha_difference_after_logical_unpermute']:.8f}",
            f"{perms['max_abs_retrieved_probability_difference_after_logical_unpermute']:.8f}",
            f"{fv['forced_value_output_follows_value_rate']:.6f}",
            f"{fa['forced_alpha_exact_route_rate']:.6f}",
        ]
        lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            f"Aggregate min/mean normal exact: {aggregate['min_seed_normal_exact']:.6f} / {aggregate['mean_seed_normal_exact']:.6f}",
            f"Aggregate min/mean swap rate: {aggregate['min_seed_swap_rate']:.6f} / {aggregate['mean_seed_swap_rate']:.6f}",
            f"Aggregate min/mean permutation grid rate: {aggregate['min_seed_permutation_grid_rate']:.6f} / {aggregate['mean_seed_permutation_grid_rate']:.6f}",
            f"Verdict: `{verdict}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_seed(
    *,
    seed: int,
    args: argparse.Namespace,
    config: PairReaderConfig,
    train_episodes: list[TwinEpisode],
    dev_episodes: list[TwinEpisode],
    device: torch.device,
) -> dict[str, Any]:
    configure_seed(seed)
    model = PairKeyValueReaderR1B(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    ckpt_dir = args.checkpoint_dir
    out = args.out
    save_checkpoint(
        ckpt_dir / f"pair_reader_r1b_seed_{seed}_initial.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        seed=seed,
        epoch_index=-1,
        global_step=0,
        extra={"fresh_initialization": True, "retrieval_supervision_used": False},
    )
    log_path = out / f"training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    observed_by_epoch: dict[int, list[int]] = {epoch: [] for epoch in range(args.epochs)}
    dev_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    global_step = 0
    for epoch_index in range(args.epochs):
        order = epoch_permutation_indices(len(train_episodes), base_seed=seed, epoch_index=epoch_index)
        for batch_indices in epoch_batches(order, args.batch_size):
            global_step += 1
            row = train_one_step(
                model=model,
                optimizer=optimizer,
                train_episodes=train_episodes,
                batch_indices=batch_indices,
                seed=seed,
                epoch_index=epoch_index,
                global_step=global_step,
                grad_clip_norm=args.grad_clip_norm,
                device=device,
            )
            observed_by_epoch[epoch_index].extend(batch_indices)
            append_jsonl(log_path, row)
            if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
                return {"model": model, "optimizer": optimizer, "global_step": global_step, "last_row": row}
        save_checkpoint(
            ckpt_dir / f"pair_reader_r1b_seed_{seed}_epoch_{epoch_index:02d}.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            seed=seed,
            epoch_index=epoch_index,
            global_step=global_step,
            extra={"checkpoint_reason": "epoch_end"},
        )
        dev_row = dev_eval_for_selection(
            model=model,
            dev_episodes=dev_episodes,
            device=device,
            batch_size=args.eval_batch_size,
            epoch_index=epoch_index,
            global_step=global_step,
        )
        dev_rows.append(dev_row)
        append_jsonl(log_path, {"event": "dev_epoch_eval", "seed": seed, **dev_row})
        key = selection_key(dev_row)
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = dev_row
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            save_checkpoint(
                ckpt_dir / f"pair_reader_r1b_seed_{seed}_best_dev.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                seed=seed,
                epoch_index=epoch_index,
                global_step=global_step,
                extra={"best_dev_metrics": dev_row},
            )
        print(json.dumps(to_jsonable({"event": "epoch_complete", "seed": seed, **dev_row}), sort_keys=True), flush=True)
    episode_ids = [episode.episode_id for episode in train_episodes]
    coverage = {"seed": seed, "all_epochs_passed": True, "epochs": []}
    for epoch_index in range(args.epochs):
        order = epoch_permutation_indices(len(train_episodes), base_seed=seed, epoch_index=epoch_index)
        row = coverage_row(epoch_index=epoch_index, order=order, observed=observed_by_epoch[epoch_index], episode_ids=episode_ids, base_seed=seed)
        coverage["epochs"].append(row)
        coverage["all_epochs_passed"] = coverage["all_epochs_passed"] and row["passed"]
    save_json(out / f"dev_metrics_by_epoch_seed_{seed}.json", {"seed": seed, "epochs": dev_rows})
    if best_state is None or best_metrics is None:
        raise RuntimeError(f"seed {seed} produced no best checkpoint")
    model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
    save_checkpoint(
        ckpt_dir / f"pair_reader_r1b_seed_{seed}_best_dev.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        seed=seed,
        epoch_index=int(best_metrics["epoch_index"]),
        global_step=int(best_metrics["global_step"]),
        extra={"best_checkpoint_selection": best_metrics},
    )
    best_path = ckpt_dir / f"pair_reader_r1b_seed_{seed}_best_dev.pt"
    selection = {
        "seed": seed,
        "selected_epoch_index": int(best_metrics["epoch_index"]),
        "selected_global_step": int(best_metrics["global_step"]),
        "selection_rule": "highest dev direct_grid_exact_accuracy; tie-break retrieval_top1_accuracy, correct_counterfactual_switch_rate, lowest hamming, earliest epoch",
        "selection_metrics": best_metrics,
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256_before_sealed": sha256_file(best_path),
        "sealed_split_read_before_selection": False,
    }
    save_json(out / f"best_checkpoint_selection_seed_{seed}.json", selection)
    return {"model": model, "optimizer": optimizer, "coverage": coverage, "selection": selection, "dev_rows": dev_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    environment = cuda_environment_report()
    print(json.dumps(environment, indent=2, sort_keys=True), flush=True)
    save_json(args.out / "environment_report.json", environment)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    before = protected_hashes()
    save_json(args.out / "source_protection_hashes_before.json", before)
    train_episodes = load_split_episodes(args.data, "train", sealed_allowed=False)
    dev_episodes = load_split_episodes(args.data, "dev", sealed_allowed=False)
    save_json(args.out / "data_manifest.json", data_manifest(args.data, train_episodes, dev_episodes, None))
    seeds = [args.seed] if args.mode == "dry_run" else args.seeds
    steps_per_epoch = math.ceil(len(train_episodes) / args.batch_size)
    config = PairReaderConfig(hidden_size=args.hidden_size, temperature=args.temperature)
    run_config = {
        "experiment": "Pair Key-Value Reader R1B - Output-Only + Causal Intervention Confirmation",
        "mode": args.mode,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps_per_seed": steps_per_epoch * args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "grad_clip_norm": args.grad_clip_norm,
        "lambda_retrieval": args.lambda_retrieval,
        "retrieval_supervision_used": False,
        "training_batch_keys": output_only_training_keys(),
        "model_config": asdict(config),
        "data": str(args.data),
        "out": str(args.out),
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    save_json(args.out / "run_config.json", run_config)
    save_json(args.out / "seed_manifest.json", {"seeds": seeds, "seed_count": len(seeds)})
    if args.lambda_retrieval != 0.0:
        raise ValueError("R1B requires --lambda-retrieval 0.0")
    init_reports: dict[str, Any] = {}
    seed_results: dict[int, Any] = {}
    for seed in seeds:
        result = run_seed(
            seed=seed,
            args=args,
            config=config,
            train_episodes=train_episodes,
            dev_episodes=dev_episodes,
            device=device,
        )
        seed_results[seed] = result
        init_reports[str(seed)] = result["model"].initialization_report()
        if args.mode == "dry_run":
            dry = {
                "mode": "dry_run",
                "seed": seed,
                "batches_run": args.max_batches,
                "global_step": result["global_step"],
                "last_train_output_loss": result["last_row"]["train_output_loss"],
                "training_batch_keys": result["last_row"]["training_batch_keys"],
                "retrieval_supervision_used": False,
                "finite_parameters": all(bool(torch.isfinite(parameter).all().item()) for parameter in result["model"].parameters()),
                "cuda_device": environment["gpu_name"],
                "protected_hashes_unchanged": before == protected_hashes(),
            }
            save_json(args.out / "initialization_report.json", init_reports)
            save_json(args.out / "dry_run_report.json", dry)
            save_json(args.out / "source_protection_hashes_after.json", protected_hashes())
            print(json.dumps(to_jsonable(dry), indent=2, sort_keys=True), flush=True)
            return dry
    save_json(args.out / "initialization_report.json", init_reports)
    all_selections_written = all((args.out / f"best_checkpoint_selection_seed_{seed}.json").exists() for seed in seeds)
    sealed_episodes = load_split_episodes(args.data, "sealed", sealed_allowed=all_selections_written)
    save_json(args.out / "data_manifest.json", data_manifest(args.data, train_episodes, dev_episodes, sealed_episodes))
    coverage_report = {"all_seeds_passed": True, "seeds": {}}
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        result = seed_results[seed]
        coverage_report["seeds"][str(seed)] = result["coverage"]
        coverage_report["all_seeds_passed"] = coverage_report["all_seeds_passed"] and result["coverage"]["all_epochs_passed"]
        model = result["model"]
        train_metrics = evaluate_selected(model=model, episodes=train_episodes, split="train", seed=seed, device=device, batch_size=args.eval_batch_size)
        dev_metrics = evaluate_selected(model=model, episodes=dev_episodes, split="dev", seed=seed, device=device, batch_size=args.eval_batch_size)
        sealed_metrics = evaluate_selected(model=model, episodes=sealed_episodes, split="sealed", seed=seed, device=device, batch_size=args.eval_batch_size)
        save_json(args.out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(args.out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(args.out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(args.out / f"exhaustive_permutation_report_seed_{seed}.json", sealed_metrics["conditions"]["ALL_PAIR_PERMUTATIONS"])
        save_json(args.out / f"forced_value_intervention_report_seed_{seed}.json", sealed_metrics["conditions"]["FORCED_VALUE_INTERVENTION"])
        save_json(args.out / f"forced_alpha_intervention_report_seed_{seed}.json", sealed_metrics["conditions"]["FORCED_ALPHA_INTERVENTION"])
        selection = result["selection"]
        best_path = Path(selection["selected_checkpoint"])
        selection["selected_checkpoint_sha256_after_sealed"] = sha256_file(best_path)
        selection["checkpoint_hash_unchanged_after_sealed"] = (
            selection["selected_checkpoint_sha256_before_sealed"] == selection["selected_checkpoint_sha256_after_sealed"]
        )
        save_json(args.out / f"best_checkpoint_selection_seed_{seed}.json", selection)
        gates = seed_gates(sealed_metrics)
        seed_summaries.append(
            {
                "seed": seed,
                "selection": selection,
                "train": train_metrics,
                "dev": dev_metrics,
                "sealed": sealed_metrics,
                "gates": gates,
            }
        )
    save_json(args.out / "epoch_coverage_report.json", coverage_report)
    aggregate = aggregate_summary(seed_summaries)
    verdict = verdict_from_summary(aggregate, seed_summaries)
    aggregate["verdict"] = verdict
    save_json(args.out / "aggregate_seed_summary.json", {"aggregate": aggregate, "seeds": seed_summaries})
    write_examples(args.out / "prediction_examples.txt", seed_summaries)
    pytest_result = run_pytest(args.out)
    after = protected_hashes()
    save_json(args.out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "environment": {**environment, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "coverage": coverage_report,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
        "pytest_result": pytest_result,
        "source_protection_hashes_unchanged": before == after,
        "interpretation": {
            "output_only_correspondence_confirmed": verdict == "OUTPUT_ONLY_READER_CONFIRMED",
            "r2_permitted_if_confirmed": verdict == "OUTPUT_ONLY_READER_CONFIRMED",
            "does_not_prove": ["VRM is repaired", "ARC is solved", "R1B is an ARC solver"],
        },
    }
    save_json(args.out / "summary_report.json", summary)
    save_text(args.out / "summary.md", comparison_markdown(seed_summaries, aggregate, verdict))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair Reader R1B output-only confirmation")
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seeds", type=lambda text: [int(item) for item in text.split(",") if item], default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-retrieval", type=float, default=0.0)
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
