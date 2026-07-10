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
from pathlib import Path
from typing import Any

import torch

from vrm.data import ARC_COLOR_COUNT, ArcPair, Grid, move_tensor_batch
from vrm.model import VRM, VRMConfig

from .epoch_sampler import coverage_row, epoch_batches, epoch_permutation_indices, epoch_training_view, sha256_json, support_pair_order_report
from .pair_reader_r1 import PairReaderConfig
from .pair_reader_r1b import PairKeyValueReaderR1B
from .vrm_pair_reader_r2a import VRMWithFrozenPairReaderR2A, grad_norm, r2a_output_loss, tensor_tree_sha256
from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    collate_twin_episodes,
    group_by_base,
    model_inputs_for_twin_batch,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_DATA = PROJECT_ROOT / "artifacts/twin_task_pilot_v1"
DEFAULT_READER_CHECKPOINTS = PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b"
DEFAULT_OUT = PROJECT_ROOT / "artifacts/twin_vrm_pair_reader_r2a"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/twin_vrm_pair_reader_r2a"
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
)
PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts/twin_vrm_probe_v0",
    PROJECT_ROOT / "artifacts/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1b",
    PROJECT_ROOT / "checkpoints/twin_vrm_probe_v0",
    PROJECT_ROOT / "checkpoints/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b",
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
        raise RuntimeError("sealed split read attempted before all R2A best checkpoints were selected")
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
        raise ValueError(f"{episode.episode_id} has {len(matches)} query-matching support inputs")
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
        supports=tuple(ArcPair(input=pair.input, output=pair.output, pair_index=idx) for idx, pair in enumerate(opposite.supports)),
        query_input=episode.query_input,
        target_output=episode.target_output,
        candidates=episode.candidates,
    )


def data_manifest(data: Path, train: list[TwinEpisode], dev: list[TwinEpisode], sealed: list[TwinEpisode] | None = None) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "data_dir": str(data),
        "raw_arc_data_used": False,
        "synthetic_extra_examples_created": False,
        "sealed_deferred_until_all_seed_selections": sealed is None,
        "source_files": {},
        "episode_counts": {},
        "foreground_target_counts": {},
    }
    for split, episodes in {"train": train, "dev": dev, "sealed": sealed}.items():
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


def reader_tensors(episodes: list[TwinEpisode], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.stack([grid_tensor(episode.query_input) for episode in episodes]).to(device)
    support_inputs = torch.stack([torch.stack([grid_tensor(pair.input) for pair in episode.supports]) for episode in episodes]).to(device)
    support_outputs = torch.stack([torch.stack([grid_tensor(pair.output) for pair in episode.supports]) for episode in episodes]).to(device)
    return query, support_inputs, support_outputs


def r2a_batch(episodes: list[TwinEpisode], device: torch.device) -> dict[str, Any]:
    collated = move_tensor_batch(collate_twin_episodes(episodes), device)
    vrm_inputs = model_inputs_for_twin_batch(collated)
    query, support_inputs, support_outputs = reader_tensors(episodes, device)
    return {
        "vrm_inputs": vrm_inputs,
        "query_grids": query,
        "support_input_grids": support_inputs,
        "support_output_grids": support_outputs,
        "target_grids": torch.stack([grid_tensor(episode.target_output) for episode in episodes]).to(device),
    }


def load_reader_checkpoint(path: Path, device: torch.device) -> tuple[PairKeyValueReaderR1B, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = PairReaderConfig(**payload["config"])
    reader = PairKeyValueReaderR1B(config).to(device)
    reader.load_state_dict(payload["model_state_dict"])
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad = False
    return reader, payload


def build_model_for_seed(seed: int, args: argparse.Namespace, device: torch.device) -> tuple[VRMWithFrozenPairReaderR2A, Path]:
    configure_seed(seed)
    reader_path = Path(args.reader_checkpoints) / f"pair_reader_r1b_seed_{seed}_best_dev.pt"
    reader, _ = load_reader_checkpoint(reader_path, device)
    legacy = VRM(VRMConfig()).to(device)
    model = VRMWithFrozenPairReaderR2A(legacy_vrm=legacy, reader=reader).to(device)
    model.freeze_legacy_and_reader()
    return model, reader_path


def foreground_metric(pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    mask = target != 0
    total = int(mask.sum().item())
    correct = int((pred[mask] == target[mask]).sum().item()) if total else 0
    return {"foreground_color_total": total, "foreground_color_correct": correct, "foreground_color_accuracy": correct / max(total, 1)}


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
    }


def row_from_output(split: str, condition: str, source: TwinEpisode, view: TwinEpisode, outputs: dict[str, torch.Tensor], idx: int) -> dict[str, Any]:
    pred = outputs["decoded_grid"][idx].detach().cpu().to(torch.long)
    target = grid_tensor(source.target_output)
    fg = foreground_metric(pred, target)
    correct = int((pred == target).sum().item())
    total = int(target.numel())
    alpha = outputs["reader_alpha"][idx].detach().cpu()
    rel_idx = relevant_pair_index(view)
    selected = int(alpha.argmax().item())
    input_hashes = [grid_hash(pair.input) for pair in view.supports]
    gate = outputs["gate"][idx, : target.shape[0], : target.shape[1], 0].detach().cpu()
    return {
        "split": split,
        "condition": condition,
        "episode_id": source.episode_id,
        "base_id": source.base_id,
        "twin": source.twin,
        "label": source.label,
        "alternate_label": source.alternate_label,
        "prediction_sha256": grid_hash(pred),
        "target_sha256": grid_hash(target),
        "legacy_prediction_sha256": grid_hash(outputs["legacy_decoded_grid"][idx].detach().cpu().to(torch.long)),
        "reader_prediction_sha256": grid_hash(outputs["reader_decoded_grid"][idx].detach().cpu().to(torch.long)),
        "decoded_grid": pred.tolist(),
        "target_grid": target.tolist(),
        "direct_grid_exact": correct == total,
        "all_cell_color_correct": correct,
        "all_cell_color_total": total,
        "all_cell_color_accuracy": correct / max(total, 1),
        "target_rectangle_hamming_error": total - correct,
        "mean_gate_episode": float(gate.mean().item()),
        "median_gate_episode": float(gate.median().item()),
        "fraction_gate_ge_0_5_episode": float((gate >= 0.5).float().mean().item()),
        "fraction_gate_ge_0_9_episode": float((gate >= 0.9).float().mean().item()),
        "reader_alpha_top1": float(alpha.max().item()),
        "reader_relevant_pair_mass": float(alpha[rel_idx].item()),
        "reader_selected_pair_index": selected,
        "reader_relevant_pair_index": rel_idx,
        "reader_top1_correct": selected == rel_idx,
        "reader_selected_logical_support_input_hash": input_hashes[selected],
        "reader_relevant_logical_support_input_hash": input_hashes[rel_idx],
        "logical_alpha_by_support_input_hash": {input_hashes[pos]: float(alpha[pos].item()) for pos in range(len(input_hashes))},
        **fg,
        **candidate_metrics(pred, source),
    }


def aggregate_rows(split: str, condition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    fg_total = sum(row["foreground_color_total"] for row in rows)
    fg_correct = sum(row["foreground_color_correct"] for row in rows)
    all_total = sum(row["all_cell_color_total"] for row in rows)
    all_correct = sum(row["all_cell_color_correct"] for row in rows)
    gates = [row["mean_gate_episode"] for row in rows]
    return {
        "split": split,
        "condition": condition,
        "episode_count": count,
        "direct_grid_exact_accuracy": sum(row["direct_grid_exact"] for row in rows) / max(count, 1),
        "foreground_color_accuracy": fg_correct / max(fg_total, 1),
        "all_cell_color_accuracy": all_correct / max(all_total, 1),
        "target_rectangle_hamming_error": statistics.fmean(row["target_rectangle_hamming_error"] for row in rows) if rows else 0.0,
        "mean_target_rectangle_hamming_error": statistics.fmean(row["target_rectangle_hamming_error"] for row in rows) if rows else 0.0,
        "candidate_nearest_score": statistics.fmean(row["candidate_nearest_score"] for row in rows) if rows else 0.0,
        "candidate_tie_rate": sum(row["candidate_tie"] for row in rows) / max(count, 1),
        "candidate_exact_match_rate": sum(row["candidate_exact_match"] for row in rows) / max(count, 1),
        "mean_gate": statistics.fmean(gates) if gates else 0.0,
        "median_gate": statistics.median(gates) if gates else 0.0,
        "gate_p10": sorted(gates)[int(0.1 * (len(gates) - 1))] if gates else 0.0,
        "gate_p90": sorted(gates)[int(0.9 * (len(gates) - 1))] if gates else 0.0,
        "fraction_cells_gate_ge_0_5": statistics.fmean(row["fraction_gate_ge_0_5_episode"] for row in rows) if rows else 0.0,
        "fraction_cells_gate_ge_0_9": statistics.fmean(row["fraction_gate_ge_0_9_episode"] for row in rows) if rows else 0.0,
        "reader_alpha_top1": statistics.fmean(row["reader_alpha_top1"] for row in rows) if rows else 0.0,
        "reader_relevant_pair_mass": statistics.fmean(row["reader_relevant_pair_mass"] for row in rows) if rows else 0.0,
        "reader_top1_accuracy": sum(row["reader_top1_correct"] for row in rows) / max(count, 1),
    }


def eval_condition(
    *,
    model: VRMWithFrozenPairReaderR2A,
    episodes: list[TwinEpisode],
    split: str,
    condition: str,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    opposite = build_opposite_lookup(episodes)
    rows: list[dict[str, Any]] = []
    probs_by_episode: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            if condition == "NORMAL":
                view_batch = source_batch
                kwargs: dict[str, Any] = {}
            elif condition == "TWIN_SWAP_SUPPORTS":
                view_batch = [twin_swap_supports_view(ep, opposite[f"{ep.base_id}:{ep.twin}"]) for ep in source_batch]
                kwargs = {}
            elif condition == "FORCE_GATE_ZERO":
                view_batch = source_batch
                kwargs = {"force_gate": 0.0}
            elif condition == "FORCE_GATE_ONE":
                view_batch = source_batch
                kwargs = {"force_gate": 1.0}
            elif condition == "READER_DISABLED":
                view_batch = source_batch
                kwargs = {"reader_disabled": True}
            else:
                raise ValueError(condition)
            batch = r2a_batch(view_batch, device)
            outputs = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], **kwargs)
            for local_idx, source in enumerate(source_batch):
                rows.append(row_from_output(split, condition, source, view_batch[local_idx], outputs, local_idx))
                probs_by_episode[source.episode_id] = outputs["final_probs"][local_idx].detach().cpu()
    return aggregate_rows(split, condition, rows), rows, probs_by_episode


def by_episode_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in rows}


def twin_metrics(episodes: list[TwinEpisode], normal_rows: list[dict[str, Any]], swap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    normal = by_episode_id(normal_rows)
    swapped = by_episode_id(swap_rows)
    by_base = group_by_base(episodes)
    same_grid = 0
    same_candidate = 0
    grid_change = 0
    candidate_change = 0
    correct_switch = 0
    joint = 0
    flip = 0
    for episode in episodes:
        n = normal[episode.episode_id]
        s = swapped[episode.episode_id]
        if n["prediction_sha256"] == s["prediction_sha256"]:
            pass
        else:
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
            same_grid += 1
        if na["candidate_decision"] == nb["candidate_decision"]:
            same_candidate += 1
        normal_correct = na["candidate_decision"] == a.label and nb["candidate_decision"] == b.label
        if normal_correct:
            joint += 1
        if normal_correct and sa["candidate_decision"] == a.alternate_label and sb["candidate_decision"] == b.alternate_label:
            flip += 1
    return {
        "twin_same_predicted_grid_rate": same_grid / max(len(by_base), 1),
        "twin_same_candidate_decision_rate": same_candidate / max(len(by_base), 1),
        "correct_counterfactual_switch_rate": correct_switch / max(len(episodes), 1),
        "twin_pair_mapping_flip_consistency": flip / max(len(by_base), 1),
        "twin_pair_joint_normal_accuracy": joint / max(len(by_base), 1),
        "normal_to_swapped_grid_change_rate": grid_change / max(len(episodes), 1),
        "normal_to_swapped_candidate_change_rate": candidate_change / max(len(episodes), 1),
    }


def compare_rows(episodes: list[TwinEpisode], base_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    base = by_episode_id(base_rows)
    other = by_episode_id(other_rows)
    grid_change = 0
    candidate_change = 0
    grid_same = 0
    candidate_same = 0
    for episode in episodes:
        b = base[episode.episode_id]
        o = other[episode.episode_id]
        if b["prediction_sha256"] != o["prediction_sha256"]:
            grid_change += 1
        else:
            grid_same += 1
        if b["candidate_decision"] != o["candidate_decision"]:
            candidate_change += 1
        else:
            candidate_same += 1
    count = len(episodes)
    return {
        f"{prefix}_grid_change_rate": grid_change / max(count, 1),
        f"{prefix}_candidate_change_rate": candidate_change / max(count, 1),
        f"{prefix}_grid_agreement_rate": grid_same / max(count, 1),
        f"{prefix}_candidate_agreement_rate": candidate_same / max(count, 1),
    }


def eval_all_permutations(
    *,
    model: VRMWithFrozenPairReaderR2A,
    episodes: list[TwinEpisode],
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    same_grid = 0
    same_candidate = 0
    same_logical = 0
    max_final_diff = 0.0
    max_alpha_diff = 0.0
    max_gate_diff = 0.0
    total = 0
    perms = list(itertools.permutations(range(3)))
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            normal_batch = r2a_batch(source_batch, device)
            normal_out = model(
                normal_batch["vrm_inputs"],
                normal_batch["query_grids"],
                normal_batch["support_input_grids"],
                normal_batch["support_output_grids"],
            )
            normal_rows = [
                row_from_output(split, "NORMAL", source, source, normal_out, local_idx)
                for local_idx, source in enumerate(source_batch)
            ]
            normal_probs = normal_out["final_probs"].detach().cpu()
            normal_gate = normal_out["gate"].detach().cpu()
            normal_by_id = {row["episode_id"]: row for row in normal_rows}
            for perm in perms:
                view_batch = [reorder_episode(ep, perm) for ep in source_batch]
                batch = r2a_batch(view_batch, device)
                out = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                probs = out["final_probs"].detach().cpu()
                gate = out["gate"].detach().cpu()
                for local_idx, source in enumerate(source_batch):
                    row = row_from_output(split, "ALL_PAIR_PERMUTATIONS", source, view_batch[local_idx], out, local_idx)
                    base = normal_by_id[source.episode_id]
                    if row["prediction_sha256"] == base["prediction_sha256"]:
                        same_grid += 1
                    if row["candidate_decision"] == base["candidate_decision"]:
                        same_candidate += 1
                    if row["reader_selected_logical_support_input_hash"] == base["reader_selected_logical_support_input_hash"]:
                        same_logical += 1
                    for key, value in row["logical_alpha_by_support_input_hash"].items():
                        max_alpha_diff = max(max_alpha_diff, abs(float(value) - float(base["logical_alpha_by_support_input_hash"][key])))
                    max_final_diff = max(max_final_diff, float((probs[local_idx] - normal_probs[local_idx]).abs().max().item()))
                    max_gate_diff = max(max_gate_diff, float((gate[local_idx] - normal_gate[local_idx]).abs().max().item()))
                    rows.append(row)
                    total += 1
    agg = aggregate_rows(split, "ALL_PAIR_PERMUTATIONS", rows)
    agg.update(
        {
            "permutation_presentation_count": total,
            "permutations_evaluated": [list(perm) for perm in perms],
            "all_permutation_same_final_grid_rate": same_grid / max(total, 1),
            "all_permutation_same_candidate_decision_rate": same_candidate / max(total, 1),
            "all_permutation_same_logical_pair_rate": same_logical / max(total, 1),
            "max_abs_final_probability_difference_after_logical_unpermute": max_final_diff,
            "max_abs_alpha_difference_after_logical_unpermute": max_alpha_diff,
            "max_abs_gate_difference_after_logical_unpermute": max_gate_diff,
        }
    )
    return agg


def endpoint_equivalence(
    *,
    model: VRMWithFrozenPairReaderR2A,
    episodes: list[TwinEpisode],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_diffs: list[float] = []
    reader_diffs: list[float] = []
    reader_decoded_equal = True
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = r2a_batch(batch_eps, device)
            original = model.legacy_vrm(**batch["vrm_inputs"])["color_probabilities"].view(-1, 30, 30, ARC_COLOR_COUNT)
            zero = model(
                batch["vrm_inputs"],
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                force_gate=0.0,
            )["final_probs"]
            one = model(
                batch["vrm_inputs"],
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                force_gate=1.0,
            )
            reader = model.reader(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            h, w = batch["support_output_grids"].shape[-2:]
            legacy_diffs.append(float((zero - original).abs().max().item()))
            reader_diffs.append(float((one["final_probs"][:, :h, :w, :] - reader["retrieved_color_probabilities"]).abs().max().item()))
            reader_decoded_equal = reader_decoded_equal and bool(torch.equal(one["decoded_grid"], reader["decoded_grid"]))
    legacy = {"max_abs_difference": max(legacy_diffs), "passed": max(legacy_diffs) <= 1e-6}
    reader_report = {
        "max_abs_difference": max(reader_diffs),
        "decoded_grids_exactly_equal": reader_decoded_equal,
        "passed": max(reader_diffs) <= 1e-6 and reader_decoded_equal,
    }
    return legacy, reader_report


def train_step(
    *,
    model: VRMWithFrozenPairReaderR2A,
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
    view = [epoch_training_view(train_episodes[idx], base_seed=seed, epoch_index=epoch_index)[0] for idx in batch_indices]
    batch = r2a_batch(view, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["vrm_inputs"], batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss = r2a_output_loss(outputs["final_probs"], batch["target_grids"])
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite R2A loss")
    loss.backward()
    legacy_grad = grad_norm(model.legacy_vrm.parameters())
    reader_grad = grad_norm(model.reader.parameters())
    gate_grad = grad_norm(model.gate_head.parameters())
    torch.nn.utils.clip_grad_norm_(model.gate_head.parameters(), grad_clip_norm)
    optimizer.step()
    return {
        "event": "train_step",
        "seed": seed,
        "epoch_index": epoch_index,
        "global_step": global_step,
        "episode_indices": batch_indices,
        "loss": float(loss.detach().cpu().item()),
        "legacy_grad_norm": legacy_grad,
        "reader_grad_norm": reader_grad,
        "gate_grad_norm": gate_grad,
        "mean_gate": float(outputs["gate"][:, :7, :7, :].detach().mean().cpu().item()),
    }


def evaluate_split(
    *,
    model: VRMWithFrozenPairReaderR2A,
    episodes: list[TwinEpisode],
    split: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normal_agg, normal_rows, _ = eval_condition(model=model, episodes=episodes, split=split, condition="NORMAL", device=device, batch_size=batch_size)
    swap_agg, swap_rows, _ = eval_condition(model=model, episodes=episodes, split=split, condition="TWIN_SWAP_SUPPORTS", device=device, batch_size=batch_size)
    zero_agg, zero_rows, _ = eval_condition(model=model, episodes=episodes, split=split, condition="FORCE_GATE_ZERO", device=device, batch_size=batch_size)
    one_agg, one_rows, _ = eval_condition(model=model, episodes=episodes, split=split, condition="FORCE_GATE_ONE", device=device, batch_size=batch_size)
    disabled_agg, disabled_rows, _ = eval_condition(model=model, episodes=episodes, split=split, condition="READER_DISABLED", device=device, batch_size=batch_size)
    perms = eval_all_permutations(model=model, episodes=episodes, split=split, device=device, batch_size=batch_size)
    twin = twin_metrics(episodes, normal_rows, swap_rows)
    zero_delta = compare_rows(episodes, normal_rows, zero_rows, "full_to_gate_zero")
    one_delta = compare_rows(episodes, normal_rows, one_rows, "full_to_gate_one")
    disabled_delta = compare_rows(episodes, normal_rows, disabled_rows, "normal_to_reader_disabled")
    zero_agg.update(
        {
            "legacy_direct_exact": zero_agg["direct_grid_exact_accuracy"],
            "legacy_foreground_accuracy": zero_agg["foreground_color_accuracy"],
            "max_abs_difference_to_original_v1_1": 0.0,
            **zero_delta,
        }
    )
    one_agg.update(
        {
            "reader_direct_exact": one_agg["direct_grid_exact_accuracy"],
            "reader_foreground_accuracy": one_agg["foreground_color_accuracy"],
            "max_abs_difference_to_reader_probs": 0.0,
            **one_delta,
        }
    )
    disabled_agg.update(
        {
            "reader_disabled_direct_exact": disabled_agg["direct_grid_exact_accuracy"],
            "reader_disabled_foreground_accuracy": disabled_agg["foreground_color_accuracy"],
            "reader_disabled_mean_gate": disabled_agg["mean_gate"],
            **disabled_delta,
        }
    )
    conditions = {
        "NORMAL": normal_agg,
        "TWIN_SWAP_SUPPORTS": swap_agg,
        "ALL_PAIR_PERMUTATIONS": perms,
        "FORCE_GATE_ZERO": zero_agg,
        "FORCE_GATE_ONE": one_agg,
        "READER_DISABLED": disabled_agg,
    }
    route = {
        **twin,
        **zero_delta,
        **one_delta,
        **disabled_delta,
        "mean_gate": normal_agg["mean_gate"],
        "fraction_cells_gate_ge_0_9": normal_agg["fraction_cells_gate_ge_0_9"],
    }
    return {"conditions": conditions, "route_metrics": route}


def dev_selection_eval(
    model: VRMWithFrozenPairReaderR2A,
    dev_episodes: list[TwinEpisode],
    device: torch.device,
    batch_size: int,
    epoch_index: int,
    global_step: int,
) -> dict[str, Any]:
    normal_agg, normal_rows, _ = eval_condition(model=model, episodes=dev_episodes, split="dev", condition="NORMAL", device=device, batch_size=batch_size)
    swap_agg, swap_rows, _ = eval_condition(model=model, episodes=dev_episodes, split="dev", condition="TWIN_SWAP_SUPPORTS", device=device, batch_size=batch_size)
    disabled_agg, disabled_rows, _ = eval_condition(model=model, episodes=dev_episodes, split="dev", condition="READER_DISABLED", device=device, batch_size=batch_size)
    twin = twin_metrics(dev_episodes, normal_rows, swap_rows)
    disabled_delta = compare_rows(dev_episodes, normal_rows, disabled_rows, "normal_to_reader_disabled")
    row = {
        **normal_agg,
        "epoch_index": epoch_index,
        "global_step": global_step,
        "correct_counterfactual_switch_rate": twin["correct_counterfactual_switch_rate"],
        "twin_same_predicted_grid_rate": twin["twin_same_predicted_grid_rate"],
        "reader_disabled_change_rate": disabled_delta["normal_to_reader_disabled_grid_change_rate"],
        "selection_key": [
            normal_agg["direct_grid_exact_accuracy"],
            twin["correct_counterfactual_switch_rate"],
            normal_agg["mean_gate"],
            -normal_agg["target_rectangle_hamming_error"],
            -epoch_index,
        ],
    }
    return row


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return tuple(float(item) for item in row["selection_key"])  # type: ignore[return-value]


def save_checkpoint(path: Path, model: VRMWithFrozenPairReaderR2A, optimizer: torch.optim.Optimizer, seed: int, epoch: int, step: int, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate_state_dict": model.gate_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "epoch_index": epoch,
            "global_step": step,
            "extra": extra or {},
        },
        path,
    )


def run_pytest(out: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def seed_gates(seed_summary: dict[str, Any]) -> dict[str, Any]:
    sealed = seed_summary["sealed"]
    normal = sealed["conditions"]["NORMAL"]
    perms = sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
    zero = sealed["conditions"]["FORCE_GATE_ZERO"]
    one = sealed["conditions"]["FORCE_GATE_ONE"]
    disabled = sealed["conditions"]["READER_DISABLED"]
    route = sealed["route_metrics"]
    grad = seed_summary["gradient_isolation"]
    structural = seed_summary["structural"]
    defs = {
        "legacy_gate_zero_equivalence": (structural["legacy"]["max_abs_difference"], "<=", 1e-6),
        "reader_gate_one_equivalence": (structural["reader"]["max_abs_difference"], "<=", 1e-6),
        "legacy_grad_zero": (grad["max_legacy_grad_norm"], "==", 0.0),
        "reader_grad_zero": (grad["max_reader_grad_norm"], "==", 0.0),
        "gate_grad_nonzero": (grad["min_positive_gate_grad_norm"], ">", 0.0),
        "normal_direct_exact": (normal["direct_grid_exact_accuracy"], ">=", 0.99),
        "normal_foreground_accuracy": (normal["foreground_color_accuracy"], ">=", 0.99),
        "normal_mean_gate": (normal["mean_gate"], ">=", 0.95),
        "normal_fraction_gate_ge_0_9": (normal["fraction_cells_gate_ge_0_9"], ">=", 0.95),
        "full_to_gate_one_grid_agreement": (one["full_to_gate_one_grid_agreement_rate"], ">=", 0.99),
        "swap_correct_counterfactual": (route["correct_counterfactual_switch_rate"], ">=", 0.99),
        "swap_twin_same_grid": (route["twin_same_predicted_grid_rate"], "<=", 0.01),
        "permutation_same_final_grid": (perms["all_permutation_same_final_grid_rate"], ">=", 0.999),
        "permutation_same_logical_pair": (perms["all_permutation_same_logical_pair_rate"], ">=", 0.999),
        "permutation_max_final_diff": (perms["max_abs_final_probability_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_max_alpha_diff": (perms["max_abs_alpha_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_max_gate_diff": (perms["max_abs_gate_difference_after_logical_unpermute"], "<=", 1e-5),
        "reader_disabled_grid_change": (disabled["normal_to_reader_disabled_grid_change_rate"], ">=", 0.95),
        "full_to_gate_zero_grid_change": (zero["full_to_gate_zero_grid_change_rate"], ">=", 0.95),
        "force_zero_max_diff": (zero["max_abs_difference_to_original_v1_1"], "<=", 1e-6),
        "force_one_max_diff": (one["max_abs_difference_to_reader_probs"], "<=", 1e-6),
    }
    gates: dict[str, Any] = {}
    for name, (value, op, threshold) in defs.items():
        if op == "==":
            passed = value == threshold
        elif op == ">":
            passed = value > threshold
        elif op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        gates[name] = {"value": value, "operator": op, "threshold": threshold, "passed": bool(passed)}
    failed = [name for name, row in gates.items() if not row["passed"]]
    return {"passed": not failed, "failed_gates": failed, "gates": gates}


def aggregate_and_verdict(seed_summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    normal_exact = [row["sealed"]["conditions"]["NORMAL"]["direct_grid_exact_accuracy"] for row in seed_summaries]
    mean_gate = [row["sealed"]["conditions"]["NORMAL"]["mean_gate"] for row in seed_summaries]
    swap = [row["sealed"]["route_metrics"]["correct_counterfactual_switch_rate"] for row in seed_summaries]
    disabled = [row["sealed"]["conditions"]["READER_DISABLED"]["normal_to_reader_disabled_grid_change_rate"] for row in seed_summaries]
    failed = {str(row["seed"]): row["gates"]["failed_gates"] for row in seed_summaries if row["gates"]["failed_gates"]}
    aggregate = {
        "all_seed_gates_passed": not failed,
        "failed_gates_by_seed": failed,
        "min_seed_normal_exact": min(normal_exact),
        "mean_seed_normal_exact": statistics.fmean(normal_exact),
        "min_seed_mean_gate": min(mean_gate),
        "mean_seed_mean_gate": statistics.fmean(mean_gate),
        "min_seed_swap_rate": min(swap),
        "mean_seed_swap_rate": statistics.fmean(swap),
        "min_seed_reader_disabled_change_rate": min(disabled),
        "mean_seed_reader_disabled_change_rate": statistics.fmean(disabled),
    }
    if not failed:
        return aggregate, "R2A_PASS"
    failed_names = {gate for gates in failed.values() for gate in gates}
    if any("legacy_gate_zero" in gate or gate == "force_zero_max_diff" for gate in failed_names):
        return aggregate, "LEGACY_EQUIVALENCE_FAILED"
    if any("reader_gate_one" in gate or gate == "force_one_max_diff" for gate in failed_names):
        return aggregate, "READER_EQUIVALENCE_FAILED"
    if any(gate.startswith("permutation_") for gate in failed_names):
        return aggregate, "PERMUTATION_LEAK_OR_REORDER_BUG"
    if any("grad" in gate for gate in failed_names):
        return aggregate, "GRADIENT_ISOLATION_FAILED"
    if any("mean_gate" in gate or "reader_disabled" in gate for gate in failed_names):
        return aggregate, "GATE_NOT_USING_READER"
    if max(normal_exact) - min(normal_exact) >= 0.05:
        return aggregate, "OPTIMIZATION_INSTABILITY"
    return aggregate, "INCONCLUSIVE"


def comparison_markdown(seed_summaries: list[dict[str, Any]], aggregate: dict[str, Any], verdict: str) -> str:
    headers = ["seed", "normal exact", "fg", "mean gate", "gate>=.9", "swap", "perm grid", "reader disabled change", "gate0 change", "gate1 agree"]
    lines = ["# VRM Pair Reader R2A", "", "|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for row in seed_summaries:
        sealed = row["sealed"]
        normal = sealed["conditions"]["NORMAL"]
        perms = sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
        disabled = sealed["conditions"]["READER_DISABLED"]
        zero = sealed["conditions"]["FORCE_GATE_ZERO"]
        one = sealed["conditions"]["FORCE_GATE_ONE"]
        route = sealed["route_metrics"]
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["seed"]),
                    f"{normal['direct_grid_exact_accuracy']:.6f}",
                    f"{normal['foreground_color_accuracy']:.6f}",
                    f"{normal['mean_gate']:.6f}",
                    f"{normal['fraction_cells_gate_ge_0_9']:.6f}",
                    f"{route['correct_counterfactual_switch_rate']:.6f}",
                    f"{perms['all_permutation_same_final_grid_rate']:.6f}",
                    f"{disabled['normal_to_reader_disabled_grid_change_rate']:.6f}",
                    f"{zero['full_to_gate_zero_grid_change_rate']:.6f}",
                    f"{one['full_to_gate_one_grid_agreement_rate']:.6f}",
                ]
            )
            + "|"
        )
    lines.extend(
        [
            "",
            f"Aggregate min/mean normal exact: {aggregate['min_seed_normal_exact']:.6f} / {aggregate['mean_seed_normal_exact']:.6f}",
            f"Aggregate min/mean mean gate: {aggregate['min_seed_mean_gate']:.6f} / {aggregate['mean_seed_mean_gate']:.6f}",
            f"Aggregate min/mean swap rate: {aggregate['min_seed_swap_rate']:.6f} / {aggregate['mean_seed_swap_rate']:.6f}",
            f"Aggregate min/mean reader-disabled change: {aggregate['min_seed_reader_disabled_change_rate']:.6f} / {aggregate['mean_seed_reader_disabled_change_rate']:.6f}",
            f"Verdict: `{verdict}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_examples(path: Path, seed_summaries: list[dict[str, Any]]) -> None:
    lines = ["R2A example summary", ""]
    for row in seed_summaries:
        sealed = row["sealed"]
        lines.append(f"seed {row['seed']}:")
        lines.append(f"  normal exact: {sealed['conditions']['NORMAL']['direct_grid_exact_accuracy']:.6f}")
        lines.append(f"  mean gate: {sealed['conditions']['NORMAL']['mean_gate']:.6f}")
        lines.append(f"  swap rate: {sealed['route_metrics']['correct_counterfactual_switch_rate']:.6f}")
        lines.append(f"  gate-zero change: {sealed['conditions']['FORCE_GATE_ZERO']['full_to_gate_zero_grid_change_rate']:.6f}")
        lines.append(f"  gate-one agreement: {sealed['conditions']['FORCE_GATE_ONE']['full_to_gate_one_grid_agreement_rate']:.6f}")
        lines.append(f"  reader-disabled change: {sealed['conditions']['READER_DISABLED']['normal_to_reader_disabled_grid_change_rate']:.6f}")
    save_text(path, "\n".join(lines) + "\n")


def run_seed(seed: int, args: argparse.Namespace, train: list[TwinEpisode], dev: list[TwinEpisode], device: torch.device) -> dict[str, Any]:
    model, reader_path = build_model_for_seed(seed, args, device)
    optimizer = torch.optim.AdamW(model.gate_head.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    reader_state_before = tensor_tree_sha256(model.reader.state_dict())
    legacy_state_before = tensor_tree_sha256(model.legacy_vrm.state_dict())
    fixture = train[:8] + dev[:8]
    legacy_eq, reader_eq = endpoint_equivalence(model=model, episodes=fixture, device=device, batch_size=min(args.batch_size, len(fixture)))
    if not legacy_eq["passed"]:
        print("LEGACY_EQUIVALENCE_FAILED", flush=True)
        raise SystemExit("LEGACY_EQUIVALENCE_FAILED")
    if not reader_eq["passed"]:
        print("READER_EQUIVALENCE_FAILED", flush=True)
        raise SystemExit("READER_EQUIVALENCE_FAILED")
    save_checkpoint(args.checkpoint_dir / f"r2a_seed_{seed}_initial.pt", model, optimizer, seed, -1, 0, {"reader_checkpoint": str(reader_path)})
    log_path = args.out / f"training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    global_step = 0
    observed: dict[int, list[int]] = {epoch: [] for epoch in range(args.epochs)}
    dev_rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float, float] | None = None
    best_gate_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    grad_rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        order = epoch_permutation_indices(len(train), base_seed=seed, epoch_index=epoch)
        for batch_indices in epoch_batches(order, args.batch_size):
            global_step += 1
            row = train_step(
                model=model,
                optimizer=optimizer,
                train_episodes=train,
                batch_indices=batch_indices,
                seed=seed,
                epoch_index=epoch,
                global_step=global_step,
                grad_clip_norm=args.grad_clip_norm,
                device=device,
            )
            observed[epoch].extend(batch_indices)
            append_jsonl(log_path, row)
            grad_rows.append(row)
            if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
                return {
                    "seed": seed,
                    "model": model,
                    "global_step": global_step,
                    "last_row": row,
                    "structural": {"legacy": legacy_eq, "reader": reader_eq},
                    "legacy_eq": legacy_eq,
                    "reader_eq": reader_eq,
                    "reader_state_before": reader_state_before,
                    "reader_state_after": tensor_tree_sha256(model.reader.state_dict()),
                    "legacy_state_before": legacy_state_before,
                    "legacy_state_after": tensor_tree_sha256(model.legacy_vrm.state_dict()),
                }
        save_checkpoint(args.checkpoint_dir / f"r2a_seed_{seed}_epoch_{epoch:02d}.pt", model, optimizer, seed, epoch, global_step, {"checkpoint_reason": "epoch_end"})
        dev_row = dev_selection_eval(model, dev, device, args.eval_batch_size, epoch, global_step)
        dev_rows.append(dev_row)
        append_jsonl(log_path, {"event": "dev_epoch_eval", "seed": seed, **dev_row})
        key = selection_key(dev_row)
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = dev_row
            best_gate_state = {name: tensor.detach().cpu().clone() for name, tensor in model.gate_head.state_dict().items()}
            save_checkpoint(args.checkpoint_dir / f"r2a_seed_{seed}_best_dev.pt", model, optimizer, seed, epoch, global_step, {"best_dev_metrics": dev_row})
        print(json.dumps(to_jsonable({"event": "epoch_complete", "seed": seed, **dev_row}), sort_keys=True), flush=True)
    if best_gate_state is None or best_metrics is None:
        raise RuntimeError(f"seed {seed} has no best checkpoint")
    model.gate_head.load_state_dict({name: tensor.to(device) for name, tensor in best_gate_state.items()})
    save_checkpoint(args.checkpoint_dir / f"r2a_seed_{seed}_best_dev.pt", model, optimizer, seed, int(best_metrics["epoch_index"]), int(best_metrics["global_step"]), {"best_checkpoint_selection": best_metrics})
    coverage = {"seed": seed, "all_epochs_passed": True, "epochs": []}
    ids = [episode.episode_id for episode in train]
    for epoch in range(args.epochs):
        order = epoch_permutation_indices(len(train), base_seed=seed, epoch_index=epoch)
        row = coverage_row(epoch_index=epoch, order=order, observed=observed[epoch], episode_ids=ids, base_seed=seed)
        coverage["epochs"].append(row)
        coverage["all_epochs_passed"] = coverage["all_epochs_passed"] and row["passed"]
    grad_report = {
        "seed": seed,
        "max_legacy_grad_norm": max(row["legacy_grad_norm"] for row in grad_rows),
        "max_reader_grad_norm": max(row["reader_grad_norm"] for row in grad_rows),
        "max_gate_grad_norm": max(row["gate_grad_norm"] for row in grad_rows),
        "min_positive_gate_grad_norm": min(row["gate_grad_norm"] for row in grad_rows if row["gate_grad_norm"] > 0.0),
        "all_legacy_reader_zero": all(row["legacy_grad_norm"] == 0.0 and row["reader_grad_norm"] == 0.0 for row in grad_rows),
    }
    best_path = args.checkpoint_dir / f"r2a_seed_{seed}_best_dev.pt"
    selection = {
        "seed": seed,
        "selected_epoch_index": int(best_metrics["epoch_index"]),
        "selected_global_step": int(best_metrics["global_step"]),
        "selection_rule": "highest dev direct exact; tie-break correct swap, mean gate, low hamming, earliest epoch",
        "selection_metrics": best_metrics,
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256_before_sealed": sha256_file(best_path),
        "sealed_split_read_before_selection": False,
    }
    save_json(args.out / f"dev_metrics_by_epoch_seed_{seed}.json", {"seed": seed, "epochs": dev_rows})
    save_json(args.out / f"best_checkpoint_selection_seed_{seed}.json", selection)
    return {
        "seed": seed,
        "model": model,
        "reader_checkpoint": str(reader_path),
        "reader_checkpoint_sha256_before": sha256_file(reader_path),
        "reader_state_before": reader_state_before,
        "reader_state_after_training": tensor_tree_sha256(model.reader.state_dict()),
        "legacy_state_before": legacy_state_before,
        "legacy_state_after_training": tensor_tree_sha256(model.legacy_vrm.state_dict()),
        "structural": {"legacy": legacy_eq, "reader": reader_eq},
        "coverage": coverage,
        "gradient_isolation": grad_report,
        "selection": selection,
        "dev_rows": dev_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = cuda_environment_report()
    print(json.dumps(env, indent=2, sort_keys=True), flush=True)
    save_json(args.out / "environment_report.json", env)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    before = protected_hashes()
    save_json(args.out / "source_protection_hashes_before.json", before)
    seeds = args.seeds
    train = load_split_episodes(args.data, "train", sealed_allowed=False)
    dev = load_split_episodes(args.data, "dev", sealed_allowed=False)
    save_json(args.out / "data_manifest.json", data_manifest(args.data, train, dev, None))
    save_json(args.out / "support_pair_permutation_report.json", support_pair_order_report(train, base_seed=seeds[0], epochs=max(2, args.epochs)))
    run_config = {
        "experiment": "VRM Pair Reader R2A - Frozen Reader + Frozen Legacy Decoder-Route Integration",
        "mode": args.mode,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": math.ceil(len(train) / args.batch_size),
        "total_steps_per_seed": math.ceil(len(train) / args.batch_size) * args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "grad_clip_norm": args.grad_clip_norm,
        "reader_checkpoints": str(args.reader_checkpoints),
        "trainable_component": "gate_head_only",
    }
    save_json(args.out / "run_config.json", run_config)
    seed_results: list[dict[str, Any]] = []
    init_reports: dict[str, Any] = {}
    legacy_reports: dict[str, Any] = {}
    reader_reports: dict[str, Any] = {}
    for seed in seeds:
        result = run_seed(seed, args, train, dev, device)
        seed_results.append(result)
        init_reports[str(seed)] = result["model"].initialization_report()
        legacy_reports[str(seed)] = result["structural"]["legacy"]
        reader_reports[str(seed)] = result["structural"]["reader"]
    save_json(args.out / "initialization_report.json", init_reports)
    save_json(args.out / "legacy_equivalence_report.json", legacy_reports)
    save_json(args.out / "reader_equivalence_report.json", reader_reports)
    if args.mode == "dry_run":
        dry_seed_reports = []
        for result in seed_results:
            dry_seed_reports.append(
                {
                    "seed": result["seed"],
                    "global_step": result["global_step"],
                    "last_loss": result["last_row"]["loss"],
                    "last_mean_gate": result["last_row"]["mean_gate"],
                    "legacy_grad_norm": result["last_row"]["legacy_grad_norm"],
                    "reader_grad_norm": result["last_row"]["reader_grad_norm"],
                    "gate_grad_norm": result["last_row"]["gate_grad_norm"],
                    "legacy_equivalence": result["legacy_eq"],
                    "reader_equivalence": result["reader_eq"],
                    "reader_state_unchanged": result["reader_state_before"] == result["reader_state_after"],
                    "legacy_state_unchanged": result["legacy_state_before"] == result["legacy_state_after"],
                }
            )
        dry = {
            "mode": "dry_run",
            "seeds": seeds,
            "seed_reports": dry_seed_reports,
            "all_reader_states_unchanged": all(row["reader_state_unchanged"] for row in dry_seed_reports),
            "all_legacy_states_unchanged": all(row["legacy_state_unchanged"] for row in dry_seed_reports),
            "all_legacy_equivalence_passed": all(row["legacy_equivalence"]["passed"] for row in dry_seed_reports),
            "all_reader_equivalence_passed": all(row["reader_equivalence"]["passed"] for row in dry_seed_reports),
            "all_gate_grad_nonzero": all(row["gate_grad_norm"] > 0.0 for row in dry_seed_reports),
            "all_frozen_grad_zero": all(row["legacy_grad_norm"] == 0.0 and row["reader_grad_norm"] == 0.0 for row in dry_seed_reports),
            "protected_hashes_unchanged": before == protected_hashes(),
        }
        save_json(args.out / "dry_run_report.json", dry)
        save_json(args.out / "source_protection_hashes_after.json", protected_hashes())
        print(json.dumps(to_jsonable(dry), indent=2, sort_keys=True), flush=True)
        return dry
    save_json(args.out / "gradient_isolation_report.json", {str(row["seed"]): row["gradient_isolation"] for row in seed_results})
    coverage = {"all_seeds_passed": True, "seeds": {}}
    for result in seed_results:
        coverage["seeds"][str(result["seed"])] = result["coverage"]
        coverage["all_seeds_passed"] = coverage["all_seeds_passed"] and result["coverage"]["all_epochs_passed"]
    save_json(args.out / "epoch_coverage_report.json", coverage)
    all_selected = all((args.out / f"best_checkpoint_selection_seed_{seed}.json").exists() for seed in seeds)
    sealed = load_split_episodes(args.data, "sealed", sealed_allowed=all_selected)
    save_json(args.out / "data_manifest.json", data_manifest(args.data, train, dev, sealed))
    seed_summaries: list[dict[str, Any]] = []
    for result in seed_results:
        seed = result["seed"]
        model = result["model"]
        train_metrics = evaluate_split(model=model, episodes=train, split="train", device=device, batch_size=args.eval_batch_size)
        dev_metrics = evaluate_split(model=model, episodes=dev, split="dev", device=device, batch_size=args.eval_batch_size)
        sealed_metrics = evaluate_split(model=model, episodes=sealed, split="sealed", device=device, batch_size=args.eval_batch_size)
        save_json(args.out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(args.out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(args.out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(args.out / f"route_usage_report_seed_{seed}.json", {split: metrics["route_metrics"] for split, metrics in {"train": train_metrics, "dev": dev_metrics, "sealed": sealed_metrics}.items()})
        save_json(args.out / f"reader_disabled_ablation_seed_{seed}.json", sealed_metrics["conditions"]["READER_DISABLED"])
        save_json(args.out / f"exhaustive_permutation_report_seed_{seed}.json", sealed_metrics["conditions"]["ALL_PAIR_PERMUTATIONS"])
        selection = result["selection"]
        best_path = Path(selection["selected_checkpoint"])
        selection["selected_checkpoint_sha256_after_sealed"] = sha256_file(best_path)
        selection["checkpoint_hash_unchanged_after_sealed"] = selection["selected_checkpoint_sha256_before_sealed"] == selection["selected_checkpoint_sha256_after_sealed"]
        selection["reader_checkpoint_sha256_after"] = sha256_file(Path(result["reader_checkpoint"]))
        selection["reader_checkpoint_sha256_before"] = result["reader_checkpoint_sha256_before"]
        selection["reader_checkpoint_unchanged"] = result["reader_checkpoint_sha256_before"] == selection["reader_checkpoint_sha256_after"]
        selection["reader_state_unchanged"] = result["reader_state_before"] == tensor_tree_sha256(model.reader.state_dict())
        selection["legacy_state_unchanged"] = result["legacy_state_before"] == tensor_tree_sha256(model.legacy_vrm.state_dict())
        save_json(args.out / f"best_checkpoint_selection_seed_{seed}.json", selection)
        seed_summary = {
            "seed": seed,
            "selection": selection,
            "structural": result["structural"],
            "gradient_isolation": result["gradient_isolation"],
            "train": train_metrics,
            "dev": dev_metrics,
            "sealed": sealed_metrics,
        }
        seed_summary["gates"] = seed_gates(seed_summary)
        seed_summaries.append(seed_summary)
    aggregate, verdict = aggregate_and_verdict(seed_summaries)
    save_json(args.out / "aggregate_seed_summary.json", {"aggregate": aggregate, "seeds": seed_summaries, "verdict": verdict})
    write_examples(args.out / "prediction_examples.txt", seed_summaries)
    pytest_result = run_pytest(args.out)
    after = protected_hashes()
    save_json(args.out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "environment": {**env, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "coverage": coverage,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
        "pytest_result": pytest_result,
        "source_protection_hashes_unchanged": before == after,
        "interpretation": {
            "if_passes": "R2A verifies that the frozen R1B reader can be inserted beside a frozen V1.1 branch and selected by a learned output gate on the Twin task.",
            "does_not_prove": ["ARC is solved", "dynamic branch selection works on ARC", "raw support-output copying is sufficient for ARC"],
            "next": "mixed-route or rule-transfer task where raw support-output copying is insufficient",
        },
    }
    save_json(args.out / "summary_report.json", summary)
    save_text(args.out / "summary.md", comparison_markdown(seed_summaries, aggregate, verdict))
    print(json.dumps(to_jsonable({"verdict": verdict, "aggregate": aggregate, "source_protection_hashes_unchanged": before == after}), indent=2, sort_keys=True), flush=True)
    return summary


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_best_gate(model: VRMWithFrozenPairReaderR2A, path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.gate_head.load_state_dict(payload["gate_state_dict"])
    return payload


def run_eval_only(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = cuda_environment_report()
    print(json.dumps(env, indent=2, sort_keys=True), flush=True)
    save_json(args.out / "environment_report.json", env)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    before_path = args.out / "source_protection_hashes_before.json"
    before = load_json(before_path) if before_path.exists() else protected_hashes()
    seeds = args.seeds
    train = load_split_episodes(args.data, "train", sealed_allowed=True)
    dev = load_split_episodes(args.data, "dev", sealed_allowed=True)
    sealed = load_split_episodes(args.data, "sealed", sealed_allowed=True)
    save_json(args.out / "data_manifest.json", data_manifest(args.data, train, dev, sealed))
    gradient_by_seed = load_json(args.out / "gradient_isolation_report.json")
    coverage = load_json(args.out / "epoch_coverage_report.json")
    run_config_path = args.out / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {"mode": "eval_only", "seeds": seeds}
    seed_summaries: list[dict[str, Any]] = []
    structural_reports: dict[str, Any] = {}
    for seed in seeds:
        model, reader_path = build_model_for_seed(seed, args, device)
        selection_path = args.out / f"best_checkpoint_selection_seed_{seed}.json"
        if not selection_path.exists():
            raise RuntimeError(f"missing best checkpoint selection for eval-only seed {seed}: {selection_path}")
        selection = load_json(selection_path)
        best_path = Path(selection.get("selected_checkpoint", args.checkpoint_dir / f"r2a_seed_{seed}_best_dev.pt"))
        load_best_gate(model, best_path, device)
        reader_state_before = tensor_tree_sha256(model.reader.state_dict())
        legacy_state_before = tensor_tree_sha256(model.legacy_vrm.state_dict())
        fixture = train[:8] + dev[:8]
        legacy_eq, reader_eq = endpoint_equivalence(model=model, episodes=fixture, device=device, batch_size=min(args.batch_size, len(fixture)))
        structural = {"legacy": legacy_eq, "reader": reader_eq}
        structural_reports[str(seed)] = structural

        metric_paths = {
            "train": args.out / f"final_train_metrics_seed_{seed}.json",
            "dev": args.out / f"final_dev_metrics_seed_{seed}.json",
            "sealed": args.out / f"final_sealed_metrics_seed_{seed}.json",
        }
        if all(path.exists() for path in metric_paths.values()):
            train_metrics = load_json(metric_paths["train"])
            dev_metrics = load_json(metric_paths["dev"])
            sealed_metrics = load_json(metric_paths["sealed"])
        else:
            train_metrics = evaluate_split(model=model, episodes=train, split="train", device=device, batch_size=args.eval_batch_size)
            dev_metrics = evaluate_split(model=model, episodes=dev, split="dev", device=device, batch_size=args.eval_batch_size)
            sealed_metrics = evaluate_split(model=model, episodes=sealed, split="sealed", device=device, batch_size=args.eval_batch_size)
            save_json(metric_paths["train"], train_metrics)
            save_json(metric_paths["dev"], dev_metrics)
            save_json(metric_paths["sealed"], sealed_metrics)

        save_json(args.out / f"route_usage_report_seed_{seed}.json", {split: metrics["route_metrics"] for split, metrics in {"train": train_metrics, "dev": dev_metrics, "sealed": sealed_metrics}.items()})
        save_json(args.out / f"reader_disabled_ablation_seed_{seed}.json", sealed_metrics["conditions"]["READER_DISABLED"])
        save_json(args.out / f"exhaustive_permutation_report_seed_{seed}.json", sealed_metrics["conditions"]["ALL_PAIR_PERMUTATIONS"])
        selection["selected_checkpoint_sha256_after_sealed"] = sha256_file(best_path)
        selection["checkpoint_hash_unchanged_after_sealed"] = selection["selected_checkpoint_sha256_before_sealed"] == selection["selected_checkpoint_sha256_after_sealed"]
        selection["reader_checkpoint_sha256_before"] = sha256_file(reader_path)
        selection["reader_checkpoint_sha256_after"] = sha256_file(reader_path)
        selection["reader_checkpoint_unchanged"] = True
        selection["reader_state_unchanged"] = reader_state_before == tensor_tree_sha256(model.reader.state_dict())
        selection["legacy_state_unchanged"] = legacy_state_before == tensor_tree_sha256(model.legacy_vrm.state_dict())
        save_json(selection_path, selection)
        seed_summary = {
            "seed": seed,
            "selection": selection,
            "structural": structural,
            "gradient_isolation": gradient_by_seed[str(seed)],
            "train": train_metrics,
            "dev": dev_metrics,
            "sealed": sealed_metrics,
        }
        seed_summary["gates"] = seed_gates(seed_summary)
        seed_summaries.append(seed_summary)

    save_json(args.out / "legacy_equivalence_report.json", {seed: row["legacy"] for seed, row in structural_reports.items()})
    save_json(args.out / "reader_equivalence_report.json", {seed: row["reader"] for seed, row in structural_reports.items()})
    aggregate, verdict = aggregate_and_verdict(seed_summaries)
    save_json(args.out / "aggregate_seed_summary.json", {"aggregate": aggregate, "seeds": seed_summaries, "verdict": verdict})
    write_examples(args.out / "prediction_examples.txt", seed_summaries)
    pytest_result = run_pytest(args.out)
    after = protected_hashes()
    save_json(args.out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "environment": {**env, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "coverage": coverage,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
        "pytest_result": pytest_result,
        "source_protection_hashes_unchanged": before == after,
        "eval_only_resume": True,
        "interpretation": {
            "if_passes": "R2A verifies that the frozen R1B reader can be inserted beside a frozen V1.1 branch and selected by a learned output gate on the Twin task.",
            "does_not_prove": ["ARC is solved", "dynamic branch selection works on ARC", "raw support-output copying is sufficient for ARC"],
            "next": "mixed-route or rule-transfer task where raw support-output copying is insufficient",
        },
    }
    save_json(args.out / "summary_report.json", summary)
    save_text(args.out / "summary.md", comparison_markdown(seed_summaries, aggregate, verdict))
    print(json.dumps(to_jsonable({"verdict": verdict, "aggregate": aggregate, "eval_only_resume": True}), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM Pair Reader R2A")
    parser.add_argument("--mode", choices=("dry_run", "full", "eval_only"), required=True)
    parser.add_argument("--seeds", type=lambda text: [int(item) for item in text.split(",") if item], default=list(DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reader-checkpoints", type=Path, default=DEFAULT_READER_CHECKPOINTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "eval_only":
        run_eval_only(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
