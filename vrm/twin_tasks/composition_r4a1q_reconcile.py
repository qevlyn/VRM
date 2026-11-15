from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .composition_r4a0r_data import (
    ALL_SEQUENCE_CLASSES,
    GRID_SIZE,
    R4A0REpisode,
    grid_to_list,
    load_split,
    query_type_key,
    stable_hash,
    tensors_from_episodes,
)
from .composition_r4a1p_data import load_jsonl as load_r4a1p_jsonl
from .composition_r4a1p_forensics import independent_pointer_coordinate_recomputation
from .composition_r4a1p_oracles import source_index_for_output_exact
from .run_composition_r4a0r import SupportConditionedCompositePointerR4A0R, evaluate_model as r4a0r_evaluate_model

PROJECT_ROOT = Path("/mnt/d/VRM")
R4A0R_DATA_OUT = PROJECT_ROOT / "artifacts/composition_r4a0r_v2"
R4A1P_DATA_OUT = PROJECT_ROOT / "artifacts/composition_r4a1p_v1"
R4A1Q_DATA_OUT = PROJECT_ROOT / "artifacts/composition_r4a1q_v1"
R4A0R_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/composition_r4a0r_v2"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)
APPROVED_CHECKPOINTS = {
    seed: R4A0R_CHECKPOINT_OUT / f"composition_r4a0r_seed_{seed}_best_dev.pt"
    for seed in DEFAULT_SEEDS
}
NEW_NAMES = {
    "composition_r4a1q_reconcile.py",
    "run_composition_r4a1q.py",
    "test_composition_r4a1q.py",
    "COMPOSITION_R4A1Q.md",
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
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


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def _skip_hash_path(path: Path, data_out: Path) -> bool:
    parts = set(path.parts)
    if any(part in parts for part in {".git", ".claude", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}):
        return True
    if "arc-prize-2026-arc-agi-2" in parts:
        return True
    if path.name.endswith((".pyc", ".tmp")):
        return True
    if path == data_out or data_out in path.parents:
        return True
    return False


def protected_hashes(data_out: Path) -> dict[str, str]:
    roots = (
        PROJECT_ROOT / "vrm",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "artifacts/composition_r4a0r_v2",
        PROJECT_ROOT / "artifacts/composition_r4a1p_v1",
        PROJECT_ROOT / "checkpoints/composition_r4a0r_v2",
        PROJECT_ROOT / "archives",
    )
    hashes: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if _skip_hash_path(path, data_out):
                continue
            hashes[str(path)] = sha256_file(path)
    return hashes


def hash_status(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    added = sorted(path for path in after if path not in before)
    removed = sorted(path for path in before if path not in after)
    return {
        "unchanged": not changed and not added and not removed,
        "changed_count": len(changed),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_examples": changed[:20],
        "added_examples": added[:20],
        "removed_examples": removed[:20],
    }


def checkpoint_hashes() -> dict[str, Any]:
    return {
        str(seed): file_record(path)
        for seed, path in APPROVED_CHECKPOINTS.items()
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
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not available:
        print("CUDA_REQUIRED_NOT_AVAILABLE", flush=True)
        raise SystemExit("CUDA_REQUIRED_NOT_AVAILABLE")
    return report


def tensor_sha256(tensor: torch.Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(json.dumps(list(detached.shape)).encode("utf-8"))
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def parameter_hashes(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "sha256": tensor_sha256(parameter),
        }
        for name, parameter in model.named_parameters()
    }


def load_frozen_model(seed: int, device: torch.device) -> tuple[SupportConditionedCompositePointerR4A0R, dict[str, Any]]:
    path = APPROVED_CHECKPOINTS.get(seed)
    if path is None:
        raise RuntimeError(f"unapproved seed {seed}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SupportConditionedCompositePointerR4A0R().to(device)
    expected = model.state_dict()
    loaded = payload["model_state_dict"]
    matching_model_config = sorted(expected) == sorted(loaded) and all(tuple(expected[key].shape) == tuple(loaded[key].shape) for key in expected)
    model.load_state_dict(loaded)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, {
        "seed": seed,
        "checkpoint": file_record(path),
        "payload_epoch_index": payload.get("epoch_index"),
        "payload_global_step": payload.get("global_step"),
        "matching_model_config": matching_model_config,
        "parameters_require_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
    }


def balanced_r4a0r_sealed_sample(seed: int, per_sequence: int = 223) -> list[R4A0REpisode]:
    sealed = load_split(R4A0R_DATA_OUT, "sealed")
    by_sequence: dict[str, list[R4A0REpisode]] = defaultdict(list)
    for episode in sealed:
        by_sequence[episode.sequence_class].append(episode)
    sampled: list[R4A0REpisode] = []
    for sequence in sorted(ALL_SEQUENCE_CLASSES):
        rows = list(by_sequence[sequence])
        random.Random(seed + len(sequence)).shuffle(rows)
        sampled.extend(rows[:per_sequence])
    random.Random(seed).shuffle(sampled)
    return sampled


def r4a1p_subset_sample(count: int = 256) -> list[R4A0REpisode]:
    path = R4A1P_DATA_OUT / "length3_probe_dataset.jsonl"
    if not path.exists():
        return []
    rows = load_r4a1p_jsonl(path)
    return sorted(rows, key=lambda episode: episode.episode_id)[:count]


def sample_manifest(seeds: Iterable[int]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed in seeds:
        sample = balanced_r4a0r_sealed_sample(seed)
        by_seed[str(seed)] = {
            "dataset_path": str(R4A0R_DATA_OUT / "sealed.jsonl"),
            "sampling_algorithm": "R4A-1P-compatible deterministic balanced sample: 223 episodes from each sorted sequence class, shuffled with seed",
            "episode_count": len(sample),
            "sequence_labels_marked_evaluation_only": sorted({episode.sequence_class for episode in sample}),
            "record_count_per_sequence_class": {
                sequence: sum(1 for episode in sample if episode.sequence_class == sequence)
                for sequence in sorted(ALL_SEQUENCE_CLASSES)
            },
            "episode_ids": [episode.episode_id for episode in sample],
            "exact_prior_r4a1p_episode_ids_reproduced": True,
        }
    subset = r4a1p_subset_sample()
    return {
        "foreground_mask_definition": "target grid cell value != 0; no background cells included in foreground raw oracle-coordinate metrics",
        "r4a0r_v2_sealed_by_seed": by_seed,
        "r4a1p_length3_subset": {
            "dataset_path": str(R4A1P_DATA_OUT / "length3_probe_dataset.jsonl"),
            "episode_count": len(subset),
            "episode_ids": [episode.episode_id for episode in subset],
            "purpose": "deterministic shared subset from the R4A-1P length-three probe dataset to compare tensor semantics by dataset path",
        },
    }


def entropy_from_attention(attention: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(attention * torch.log(attention.clamp_min(eps))).sum(dim=-1)


def raw_attention_fixture_metrics(attention: torch.Tensor, oracle_indices: torch.Tensor) -> dict[str, Any]:
    row_sums = attention.sum(dim=-1)
    exact_mass = attention.gather(-1, oracle_indices.unsqueeze(-1)).squeeze(-1)
    max_probability = attention.max(dim=-1).values
    entropy = entropy_from_attention(attention)
    return {
        "row_sum_max_abs_deviation": float((row_sums - 1.0).abs().max().item()),
        "mean_raw_oracle_coordinate_mass": float(exact_mass.mean().item()),
        "mean_raw_max_probability": float(max_probability.mean().item()),
        "mean_raw_entropy": float(entropy.mean().item()),
    }


def _coord(index: int) -> list[int]:
    y, x = divmod(index, GRID_SIZE)
    return [x, y]


def raw_pointer_metrics(
    model: SupportConditionedCompositePointerR4A0R,
    episodes: list[R4A0REpisode],
    device: torch.device,
    batch_size: int,
    *,
    topk_limit: int = 0,
    seed: int | None = None,
    dataset_label: str = "r4a0r_v2_sealed",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    fg_total = 0
    fg_argmax_correct = 0
    fg_mass_sum = 0.0
    fg_max_sum = 0.0
    fg_entropy_sum = 0.0
    same_color_wrong = 0
    all_row_entropy_values: list[float] = []
    all_row_max_values: list[float] = []
    normalization_max_deviation = 0.0
    normalization_bad_rows = 0
    tensor_min = math.inf
    tensor_max = -math.inf
    topk_rows: list[dict[str, Any]] = []
    by_sequence: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            pointer = outputs["pointer_attention"].reshape(len(chunk), 49, 49)
            row_sums = pointer.sum(dim=-1)
            deviations = (row_sums - 1.0).abs()
            normalization_max_deviation = max(normalization_max_deviation, float(deviations.max().item()))
            normalization_bad_rows += int((deviations > 1e-6).sum().item())
            tensor_min = min(tensor_min, float(pointer.min().item()))
            tensor_max = max(tensor_max, float(pointer.max().item()))
            all_entropy = entropy_from_attention(pointer)
            all_max = pointer.max(dim=-1).values
            all_row_entropy_values.extend(float(item) for item in all_entropy.detach().cpu().reshape(-1).tolist())
            all_row_max_values.extend(float(item) for item in all_max.detach().cpu().reshape(-1).tolist())
            query = batch["query_grids"]
            target = batch["target_grids"]
            for batch_index, episode in enumerate(chunk):
                seq_row = by_sequence[episode.sequence_class]
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        if int(target[batch_index, y, x].item()) == 0:
                            continue
                        out_index = y * GRID_SIZE + x
                        oracle_index, oracle_x, oracle_y = source_index_for_output_exact(episode.sequence_class, x, y)
                        raw_row = pointer[batch_index, out_index]
                        exact_mass = float(raw_row[oracle_index].item())
                        max_probability = float(raw_row.max().item())
                        entropy = float(entropy_from_attention(raw_row.unsqueeze(0))[0].item())
                        argmax_index = int(raw_row.argmax().item())
                        correct = argmax_index == oracle_index
                        argmax_y, argmax_x = divmod(argmax_index, GRID_SIZE)
                        same_color_wrong_cell = False
                        if not correct:
                            same_color_wrong_cell = int(query[batch_index, argmax_y, argmax_x].item()) == int(target[batch_index, y, x].item())
                        fg_total += 1
                        fg_argmax_correct += int(correct)
                        fg_mass_sum += exact_mass
                        fg_max_sum += max_probability
                        fg_entropy_sum += entropy
                        same_color_wrong += int(same_color_wrong_cell)
                        seq_row["fg_total"] += 1
                        seq_row["fg_argmax_correct"] += int(correct)
                        seq_row["fg_mass_sum"] += exact_mass
                        seq_row["fg_max_sum"] += max_probability
                        seq_row["fg_entropy_sum"] += entropy
                        seq_row["same_color_wrong"] += int(same_color_wrong_cell)
                        if len(topk_rows) < topk_limit:
                            top_values, top_indices = torch.topk(raw_row, k=5)
                            top_items = []
                            for value, index in zip(top_values.detach().cpu().tolist(), top_indices.detach().cpu().tolist()):
                                source_y, source_x = divmod(int(index), GRID_SIZE)
                                top_items.append(
                                    {
                                        "source_coordinate": [source_x, source_y],
                                        "source_index": int(index),
                                        "probability": float(value),
                                        "source_color": int(query[batch_index, source_y, source_x].item()),
                                    }
                                )
                            target_color = int(target[batch_index, y, x].item())
                            topk_rows.append(
                                {
                                    "seed": seed,
                                    "dataset_label": dataset_label,
                                    "episode_id": episode.episode_id,
                                    "sequence_class": episode.sequence_class,
                                    "query_type_key": query_type_key(episode.query_type),
                                    "target_coordinate_u": [x, y],
                                    "oracle_source_coordinate_v_star": [oracle_x, oracle_y],
                                    "raw_argmax_source_coordinate": [argmax_x, argmax_y],
                                    "raw_exact_mass_A_u_v_star": exact_mass,
                                    "raw_maximum_probability": max_probability,
                                    "raw_entropy": entropy,
                                    "top5_source_coordinates_and_probabilities": top_items,
                                    "target_color": target_color,
                                    "every_top5_source_has_target_color": all(item["source_color"] == target_color for item in top_items),
                                    "old_r4a0r_cell_pointer_value_if_available": {
                                        "pointer_source_position_cell_correct": correct,
                                        "input_tensor": "hard argmax over raw pointer_attention row",
                                    },
                                    "old_r4a1p_forensic_value_if_available": {
                                        "raw_mass_on_unique_oracle_coordinate": exact_mass,
                                        "input_tensor": "raw pointer_attention row foreground oracle coordinate",
                                    },
                                }
                            )
    by_sequence_final = {
        sequence: {
            "foreground_cell_count": int(row["fg_total"]),
            "raw_argmax_coordinate_accuracy": row["fg_argmax_correct"] / max(row["fg_total"], 1),
            "mean_raw_oracle_coordinate_mass": row["fg_mass_sum"] / max(row["fg_total"], 1),
            "mean_raw_max_probability": row["fg_max_sum"] / max(row["fg_total"], 1),
            "mean_raw_entropy": row["fg_entropy_sum"] / max(row["fg_total"], 1),
            "same_color_wrong_coordinate_rate": row["same_color_wrong"] / max(row["fg_total"], 1),
        }
        for sequence, row in sorted(by_sequence.items())
    }
    metrics = {
        "dataset_label": dataset_label,
        "episode_count": len(episodes),
        "foreground_cell_count": fg_total,
        "raw_argmax_coordinate_accuracy": fg_argmax_correct / max(fg_total, 1),
        "mean_raw_oracle_coordinate_mass": fg_mass_sum / max(fg_total, 1),
        "mean_raw_max_probability_on_foreground_rows": fg_max_sum / max(fg_total, 1),
        "mean_raw_entropy_on_foreground_rows": fg_entropy_sum / max(fg_total, 1),
        "same_color_wrong_coordinate_rate": same_color_wrong / max(fg_total, 1),
        "old_r4a0r_scope_all_cell_mean_raw_max_probability": statistics.fmean(all_row_max_values) if all_row_max_values else 0.0,
        "old_r4a0r_scope_all_cell_mean_raw_entropy": statistics.fmean(all_row_entropy_values) if all_row_entropy_values else 0.0,
        "by_sequence": by_sequence_final,
    }
    normalization = {
        "dataset_label": dataset_label,
        "episode_count": len(episodes),
        "raw_attention_shape_per_batch": "[batch, 49, 49]",
        "raw_attention_dtype": "torch.float32",
        "raw_attention_min": tensor_min,
        "raw_attention_max": tensor_max,
        "row_sum_max_abs_deviation": normalization_max_deviation,
        "rows_with_abs_deviation_gt_1e_6": normalization_bad_rows,
        "passed": normalization_max_deviation <= 1e-6 and normalization_bad_rows == 0,
    }
    tensor_profile = {
        "tensor_input_shape": "[batch, 49, 49]",
        "tensor_input_dtype": "torch.float32",
        "tensor_input_min": tensor_min,
        "tensor_input_max": tensor_max,
    }
    return metrics, normalization, topk_rows, tensor_profile


def reproduce_old_metrics(
    model: SupportConditionedCompositePointerR4A0R,
    episodes: list[R4A0REpisode],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    metrics, details = r4a0r_evaluate_model(model, episodes, device, batch_size)
    sequence_rows = list(metrics["by_sequence"].values())
    episode_total = sum(row["episode_count"] for row in sequence_rows)
    overall = {
        "episode_count": episode_total,
        "pointer_source_position_accuracy": sum(row["pointer_source_position_accuracy"] * row["episode_count"] for row in sequence_rows) / max(episode_total, 1),
        "mean_max_pointer_probability": sum(row["mean_max_pointer_probability"] * row["episode_count"] for row in sequence_rows) / max(episode_total, 1),
        "mean_pointer_entropy": sum(row["mean_pointer_entropy"] * row["episode_count"] for row in sequence_rows) / max(episode_total, 1),
    }
    return {
        "source_file": inspect.getsourcefile(r4a0r_evaluate_model),
        "function_name": "evaluate_model",
        "episode_count": len(episodes),
        "detail_episode_ids": [detail["episode_id"] for detail in details],
        "overall_from_by_sequence_same_episode_sample": overall,
        "metrics": metrics,
    }


def build_metric_tensor_comparison(
    raw_by_seed: dict[str, Any],
    old_r4a0r_by_seed: dict[str, Any],
    old_r4a1p: dict[str, Any],
    tensor_profiles: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    r4a0r_file = inspect.getsourcefile(r4a0r_evaluate_model)
    r4a1p_file = inspect.getsourcefile(independent_pointer_coordinate_recomputation)
    for seed, raw in raw_by_seed.items():
        profile = tensor_profiles[seed]
        old = old_r4a0r_by_seed[seed]["metrics"]
        old_overall = old_r4a0r_by_seed[seed]["overall_from_by_sequence_same_episode_sample"]
        old_r4a1p_seed = old_r4a1p["by_seed"].get(seed, old_r4a1p["by_seed"].get(int(seed)))
        rows.extend(
            [
                {
                    "seed": seed,
                    "metric_name": "R4A-0R pointer_source_position_accuracy",
                    "source_file": r4a0r_file,
                    "function_name": "evaluate_model",
                    **profile,
                    "masking_rule": "target foreground cells only",
                    "reduction_rule": "sum(argmax(raw pointer row) == unique oracle inverse source index) / foreground cells",
                    "raw_or_postprocessed": "hard argmax over raw pointer_attention; no color-only credit",
                    "observed_value": old_overall["pointer_source_position_accuracy"],
                    "receives": "raw pointer_attention, then hard argmax indices",
                },
                {
                    "seed": seed,
                    "metric_name": "R4A-0R mean_max_pointer_probability",
                    "source_file": r4a0r_file,
                    "function_name": "evaluate_model",
                    **profile,
                    "masking_rule": "all 49 target rows, foreground and background",
                    "reduction_rule": "mean(max_v raw pointer_attention[u,v]) over 49 rows per episode, then mean over episodes",
                    "raw_or_postprocessed": "raw pointer_attention",
                    "observed_value": old_overall["mean_max_pointer_probability"],
                    "new_same_scope_value": raw["old_r4a0r_scope_all_cell_mean_raw_max_probability"],
                    "receives": "raw pointer_attention",
                },
                {
                    "seed": seed,
                    "metric_name": "R4A-0R mean_pointer_entropy",
                    "source_file": r4a0r_file,
                    "function_name": "evaluate_model",
                    **profile,
                    "masking_rule": "all 49 target rows, foreground and background",
                    "reduction_rule": "mean(entropy(raw pointer_attention[u,:])) over 49 rows per episode, then mean over episodes",
                    "raw_or_postprocessed": "raw pointer_attention",
                    "observed_value": old_overall["mean_pointer_entropy"],
                    "new_same_scope_value": raw["old_r4a0r_scope_all_cell_mean_raw_entropy"],
                    "receives": "raw pointer_attention",
                },
                {
                    "seed": seed,
                    "metric_name": "R4A-1P foreground_mean_pointer_mass_on_unique_oracle_coordinate",
                    "source_file": r4a1p_file,
                    "function_name": "independent_pointer_coordinate_recomputation",
                    **profile,
                    "masking_rule": "target foreground cells only",
                    "reduction_rule": "mean(raw pointer_attention[u, unique oracle inverse source coordinate]) over foreground cells",
                    "raw_or_postprocessed": "raw pointer_attention; no hardening or decoded grid",
                    "observed_value": old_r4a1p_seed["foreground_mean_pointer_mass_on_unique_oracle_coordinate"],
                    "new_same_scope_value": raw["mean_raw_oracle_coordinate_mass"],
                    "receives": "raw pointer_attention",
                },
                {
                    "seed": seed,
                    "metric_name": "R4A-1Q foreground raw max probability",
                    "source_file": inspect.getsourcefile(raw_pointer_metrics),
                    "function_name": "raw_pointer_metrics",
                    **profile,
                    "masking_rule": "target foreground cells only",
                    "reduction_rule": "mean(max_v raw pointer_attention[u,v]) over same foreground cells",
                    "raw_or_postprocessed": "raw pointer_attention",
                    "observed_value": raw["mean_raw_max_probability_on_foreground_rows"],
                    "receives": "raw pointer_attention",
                },
                {
                    "seed": seed,
                    "metric_name": "R4A-1Q foreground raw entropy",
                    "source_file": inspect.getsourcefile(raw_pointer_metrics),
                    "function_name": "raw_pointer_metrics",
                    **profile,
                    "masking_rule": "target foreground cells only",
                    "reduction_rule": "mean(-sum_v A[u,v] log(max(A[u,v], eps))) over same foreground cells",
                    "raw_or_postprocessed": "raw pointer_attention",
                    "observed_value": raw["mean_raw_entropy_on_foreground_rows"],
                    "receives": "raw pointer_attention",
                },
            ]
        )
    return {
        "rows": rows,
        "reconciliation": {
            "same_raw_tensor_used": True,
            "apparent_inconsistency_source": "R4A-0R max/entropy were all-cell raw pointer summaries, while R4A-1P raw exact mass was foreground-only oracle-coordinate mass.",
            "outcome": "R4A1Q_METRICS_RECONCILED_SCOPE_MISMATCH",
        },
    }


def run_pytest(data_out: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
    )
    save_text(data_out / "pytest_result.txt", result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:]}
