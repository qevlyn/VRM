from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from vrm.data import MAX_GRID_HEIGHT, MAX_GRID_WIDTH, move_tensor_batch
from vrm.metrics import factorized_shape_color_loss
from vrm.model import VRM, VRMConfig

from .epoch_sampler import (
    coverage_row,
    epoch_batches,
    epoch_permutation_indices,
    epoch_training_view,
    sha256_json,
    support_pair_order_report,
)
from .vrm_probe_data import (
    CONDITION_NORMAL,
    CONDITION_PAIR_ORDER_SHUFFLED,
    CONDITION_TWIN_SWAP_SUPPORTS,
    TwinEpisode,
    build_opposite_lookup,
    collate_twin_episodes,
    condition_view,
    group_by_base,
    model_inputs_for_twin_batch,
    read_jsonl,
    records_to_episodes,
    support_input_bag_signature,
    support_output_bag_signature,
    support_signature,
)
from .vrm_probe_train import (
    compute_lr,
    configure_determinism,
    finite_gradients,
    finite_parameters,
    initialize_fresh_model,
    resolve_device,
)

DEFAULT_DATA = Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1")
DEFAULT_OUT = Path("/mnt/d/VRM/artifacts/twin_vrm_epoch_baseline_v1")
DEFAULT_CHECKPOINT_DIR = Path("/mnt/d/VRM/checkpoints/twin_vrm_epoch_baseline_v1")
TWIN_PROBE_CONFIG = Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0/probe_config.json")
SEED = 20260628
CONDITIONS = (CONDITION_NORMAL, CONDITION_TWIN_SWAP_SUPPORTS, CONDITION_PAIR_ORDER_SHUFFLED)
PROTECTED_PATHS = (
    Path("/mnt/d/VRM/vrm/model.py"),
    Path("/mnt/d/VRM/vrm/data.py"),
    Path("/mnt/d/VRM/vrm/metrics.py"),
    Path("/mnt/d/VRM/vrm/train.py"),
    Path("/mnt/d/VRM/vrm/twin_tasks/vrm_probe_data.py"),
    Path("/mnt/d/VRM/vrm/twin_tasks/vrm_probe_metrics.py"),
    Path("/mnt/d/VRM/vrm/twin_tasks/vrm_probe_train.py"),
    Path("/mnt/d/VRM/vrm/twin_tasks/run_vrm_probe.py"),
    Path("/mnt/d/VRM/tests/test_twin_vrm_probe.py"),
    Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0/probe_config.json"),
    Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0/summary_report.json"),
    Path("/mnt/d/VRM/checkpoints/twin_vrm_probe_v0/twin_vrm_probe_best_dev_candidate.pt"),
    Path("/mnt/d/VRM/archives/VRM_TWIN_PROBE_V0_FROZEN_20260628.zip"),
)
PROTECTED_DIRS = (
    Path("/mnt/d/VRM/artifacts/twin_task_pilot_v1"),
    Path("/mnt/d/VRM/artifacts/twin_vrm_probe_v0"),
    Path("/mnt/d/VRM/checkpoints/twin_vrm_probe_v0"),
)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    seen: set[str] = set()
    for path in PROTECTED_PATHS:
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


def load_split_episodes(data: Path, split: str, *, sealed_allowed: bool) -> list[TwinEpisode]:
    if split == "sealed" and not sealed_allowed:
        raise RuntimeError("sealed split read attempted before best checkpoint selection was written")
    return records_to_episodes(read_jsonl(Path(data) / f"{split}.jsonl"))


def load_probe_config(path: Path = TWIN_PROBE_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_probe(probe_config: dict[str, Any]) -> VRMConfig:
    cfg = probe_config["model_config"]
    return VRMConfig(
        hidden_size=int(cfg["hidden_size"]),
        num_heads=int(cfg["num_heads"]),
        pair_encoder_layers=int(cfg["pair_encoder_layers"]),
        rewrite_layers=int(cfg["rewrite_layers"]),
        rewrite_steps=int(cfg["rewrite_steps"]),
        rule_slots=int(cfg["rule_slots"]),
        dropout=float(cfg["dropout"]),
    )


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


def architecture_parity_report(probe_config: dict[str, Any], config: VRMConfig, args: argparse.Namespace) -> dict[str, Any]:
    used_model = asdict(config)
    fields: dict[str, dict[str, Any]] = {}
    for name in ("hidden_size", "num_heads", "pair_encoder_layers", "rewrite_layers", "rewrite_steps", "rule_slots", "dropout"):
        fields[name] = {
            "twin_probe_v0": probe_config["model_config"][name],
            "epoch_baseline_v1": used_model[name],
            "matches": probe_config["model_config"][name] == used_model[name],
        }
    optim_pairs = {
        "optimizer_type": ("AdamW", "AdamW"),
        "adamw_betas": (probe_config["betas"], [args.beta1, args.beta2]),
        "weight_decay": (probe_config["weight_decay"], args.weight_decay),
        "peak_learning_rate": (probe_config["learning_rate"], args.learning_rate),
        "minimum_learning_rate": (probe_config["min_learning_rate"], args.min_learning_rate),
        "gradient_clip_norm": (probe_config["grad_clip_norm"], args.grad_clip_norm),
    }
    for name, (prior, current) in optim_pairs.items():
        fields[name] = {"twin_probe_v0": prior, "epoch_baseline_v1": current, "matches": prior == current}
    return {
        "source_config": str(TWIN_PROBE_CONFIG),
        "all_required_fields_match": all(row["matches"] for row in fields.values()),
        "allowed_differences": [
            "epoch-based sampler without replacement",
            "total scheduler horizon derived from full epochs",
            "new artifact/checkpoint paths",
            "additional diagnostic metrics",
        ],
        "fields": fields,
    }


def grid_tensor(grid: Any) -> torch.Tensor:
    return torch.tensor(grid.to_list(), dtype=torch.long)


def prediction_rect(prediction: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return prediction.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)[:h, :w].detach().cpu().to(torch.long)


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
    score = 0.5 if decision is None else (1.0 if decision == episode.label else 0.0)
    return {
        "distance_to_candidate_0": d0,
        "distance_to_candidate_1": d1,
        "candidate_decision": decision,
        "candidate_decision_name": "TIE" if decision is None else str(decision),
        "candidate_nearest_score": score,
        "candidate_tie": decision is None,
        "candidate_exact_match": bool(torch.equal(pred, c0) or torch.equal(pred, c1)),
    }


def derived_pair_proxy(pred: torch.Tensor, episode: TwinEpisode) -> dict[str, Any]:
    matches = []
    for idx, pair in enumerate(episode.supports):
        if torch.equal(pred, grid_tensor(pair.output)):
            matches.append(idx)
    if len(matches) != 1:
        return {
            "matches_any_support_output": bool(matches),
            "support_output_match_tie": len(matches) > 1,
            "derived_relevant_pair_output": False,
            "matched_support_pair_index": None,
        }
    match_idx = matches[0]
    relevant = episode.supports[match_idx].input == episode.query_input
    return {
        "matches_any_support_output": True,
        "support_output_match_tie": False,
        "derived_relevant_pair_output": bool(relevant),
        "matched_support_pair_index": match_idx,
    }


def foreground_metric(pred: torch.Tensor, target: torch.Tensor, background_color: int = 0) -> dict[str, Any]:
    mask = target != background_color
    total = int(mask.sum().item())
    correct = int((pred[mask] == target[mask]).sum().item()) if total else 0
    return {
        "foreground_color_correct": correct,
        "foreground_color_total": total,
        "foreground_color_accuracy": correct / max(total, 1),
    }


def evaluate_condition(
    *,
    model: VRM,
    episodes: list[TwinEpisode],
    split: str,
    condition: str,
    device: torch.device,
    seed: int,
    batch_size: int,
    shape_loss_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    opposite = build_opposite_lookup(episodes)
    rows: list[dict[str, Any]] = []
    totals = {"loss": 0.0, "all_correct": 0, "all_total": 0, "fg_correct": 0, "fg_total": 0}
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            source_batch = episodes[start : start + batch_size]
            view_batch = [
                condition_view(episode, condition=condition, opposite_by_episode=opposite, seed=seed)
                for episode in source_batch
            ]
            batch = move_tensor_batch(collate_twin_episodes(view_batch), device)
            outputs = model(**model_inputs_for_twin_batch(batch))
            loss, _ = factorized_shape_color_loss(outputs, batch, shape_loss_weight)
            predictions = outputs["color_log_probs"].argmax(dim=-1).detach().cpu().to(torch.long)
            pred_h = (outputs["height_logits"].argmax(dim=-1) + 1).detach().cpu().to(torch.long)
            pred_w = (outputs["width_logits"].argmax(dim=-1) + 1).detach().cpu().to(torch.long)
            totals["loss"] += float(loss.detach().cpu().item()) * len(source_batch)
            for local_idx, episode in enumerate(source_batch):
                h, w = episode.target_output.height, episode.target_output.width
                pred = prediction_rect(predictions[local_idx], h, w)
                target = grid_tensor(episode.target_output)
                all_correct = int((pred == target).sum().item())
                all_total = int(target.numel())
                fg = foreground_metric(pred, target)
                fg_total = fg["foreground_color_total"]
                fg_correct = fg["foreground_color_correct"]
                hamming = all_total - all_correct
                shape_match = int(pred_h[local_idx].item()) == h and int(pred_w[local_idx].item()) == w
                cand = candidate_metrics(pred, episode)
                proxy = derived_pair_proxy(pred, episode)
                row = {
                    "split": split,
                    "condition": condition,
                    "episode_id": episode.episode_id,
                    "base_id": episode.base_id,
                    "twin": episode.twin,
                    "label": episode.label,
                    "alternate_label": episode.alternate_label,
                    "prediction_sha256": sha256_json(pred.tolist()),
                    "target_sha256": sha256_json(target.tolist()),
                    "predicted_shape": [int(pred_h[local_idx].item()), int(pred_w[local_idx].item())],
                    "target_shape": [h, w],
                    "direct_grid_exact": bool(shape_match and hamming == 0),
                    "shape_match": bool(shape_match),
                    "all_cell_color_correct": all_correct,
                    "all_cell_color_total": all_total,
                    "foreground_color_correct": fg_correct,
                    "foreground_color_total": fg_total,
                    "target_rectangle_hamming_error": hamming,
                    **cand,
                    **proxy,
                }
                rows.append(row)
                totals["all_correct"] += all_correct
                totals["all_total"] += all_total
                totals["fg_correct"] += fg_correct
                totals["fg_total"] += fg_total
    count = len(rows)
    hamming_values = [row["target_rectangle_hamming_error"] for row in rows]
    foreground_counts = [row["foreground_color_total"] for row in rows]
    agg = {
        "split": split,
        "condition": condition,
        "episode_count": count,
        "loss": totals["loss"] / max(count, 1),
        "direct_grid_exact_accuracy": sum(row["direct_grid_exact"] for row in rows) / max(count, 1),
        "shape_match_rate": sum(row["shape_match"] for row in rows) / max(count, 1),
        "all_cell_color_accuracy": totals["all_correct"] / max(totals["all_total"], 1),
        "foreground_color_accuracy": totals["fg_correct"] / max(totals["fg_total"], 1),
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
        "derived_relevant_pair_output_rate": sum(row["derived_relevant_pair_output"] for row in rows) / max(count, 1),
        "predictions_matching_any_support_output_rate": sum(row["matches_any_support_output"] for row in rows) / max(count, 1),
    }
    return agg, rows


def by_episode_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["episode_id"]: row for row in rows}


def derived_correspondence_metrics(episodes: list[TwinEpisode], per_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
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
    for twins in by_base.values():
        if set(twins) != {"A", "B"}:
            continue
        a, b = twins["A"], twins["B"]
        na, nb = normal[a.episode_id], normal[b.episode_id]
        sa, sb = swapped[a.episode_id], swapped[b.episode_id]
        if na["prediction_sha256"] == nb["prediction_sha256"]:
            twin_same_grid += 1
        if na["candidate_decision"] == nb["candidate_decision"]:
            twin_same_candidate += 1
        normal_correct = na["candidate_decision"] == a.label and nb["candidate_decision"] == b.label
        if normal_correct:
            joint_normal += 1
        if normal_correct and sa["candidate_decision"] == a.alternate_label and sb["candidate_decision"] == b.alternate_label:
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
    }


def evaluate_split_all_conditions(
    *,
    model: VRM,
    episodes: list[TwinEpisode],
    split: str,
    device: torch.device,
    seed: int,
    batch_size: int,
    shape_loss_weight: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    conditions: dict[str, Any] = {}
    per_condition: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        agg, rows = evaluate_condition(
            model=model,
            episodes=episodes,
            split=split,
            condition=condition,
            device=device,
            seed=seed,
            batch_size=batch_size,
            shape_loss_weight=shape_loss_weight,
        )
        conditions[condition] = agg
        per_condition[condition] = rows
    return {"conditions": conditions, "derived_metrics": derived_correspondence_metrics(episodes, per_condition)}, per_condition


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["candidate_nearest_score"]),
        float(row["twin_pair_joint_normal_accuracy"]),
        float(row["foreground_color_accuracy"]),
        -float(row["mean_target_rectangle_hamming_error"]),
    )


def save_checkpoint(
    path: Path,
    *,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    config: VRMConfig,
    global_step: int,
    epoch_index: int,
    next_batch_offset: int,
    current_epoch_order: list[int],
    total_steps: int,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lr = compute_lr(global_step, total_steps, args.learning_rate, args.warmup_steps, args.min_learning_rate)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": global_step,
            "current_epoch": epoch_index,
            "next_batch_offset": next_batch_offset,
            "scheduler_state": {
                "global_step": global_step,
                "total_steps": total_steps,
                "peak_learning_rate": args.learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "warmup_steps": args.warmup_steps,
                "current_learning_rate": lr,
            },
            "current_epoch_permutation": current_epoch_order,
            "current_epoch_permutation_sha256": sha256_json(current_epoch_order),
            "extra": extra or {},
        },
        path,
    )


def load_baseline_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if "twin_vrm_epoch_baseline" not in path.name:
        raise ValueError(f"refusing to resume from non-baseline checkpoint: {path}")
    return torch.load(path, map_location=device)


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
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


def verdict_from_metrics(final: dict[str, Any]) -> str:
    train = final["train"]
    dev = final["dev"]
    sealed = final["sealed"]
    train_high = train["conditions"][CONDITION_NORMAL]["candidate_nearest_score"] >= 0.90
    dev_high = dev["conditions"][CONDITION_NORMAL]["candidate_nearest_score"] >= 0.90
    sealed_high = sealed["conditions"][CONDITION_NORMAL]["candidate_nearest_score"] >= 0.90
    strong_switch = dev["derived_metrics"]["correct_counterfactual_switch_rate"] >= 0.80 and sealed["derived_metrics"]["correct_counterfactual_switch_rate"] >= 0.80
    near_one_third_fg = all(
        split["conditions"][CONDITION_NORMAL]["foreground_color_accuracy"] <= 0.45
        for split in (train, dev, sealed)
    )
    twin_same_high = all(split["derived_metrics"]["twin_same_predicted_grid_rate"] >= 0.80 for split in (train, dev, sealed))
    switch_low = all(split["derived_metrics"]["correct_counterfactual_switch_rate"] <= 0.20 for split in (train, dev, sealed))
    order_grid_change = 1.0 - sealed["derived_metrics"]["normal_vs_pair_order_same_grid_rate"]
    swap_grid_change = sealed["derived_metrics"]["normal_to_swapped_grid_change_rate"]
    if train_high and dev_high and sealed_high and strong_switch:
        return "FAIR_BASELINE_CORRESPONDENCE_SUPPORTED"
    if train_high and (not dev_high or not sealed_high):
        return "TRAIN_FIT_BUT_GENERALIZATION_NOT_DEMONSTRATED"
    if near_one_third_fg and twin_same_high and switch_low:
        return "CAPACITY_NOT_DEMONSTRATED_AFTER_FULL_COVERAGE"
    if order_grid_change > swap_grid_change + 0.10:
        return "ORDER_ARTIFACT_SUSPECTED"
    return "INCONCLUSIVE"


def comparison_markdown(final: dict[str, Any], verdict: str) -> str:
    headers = [
        "split",
        "condition",
        "direct exact",
        "foreground color acc",
        "candidate score",
        "tie rate",
        "derived relevant-pair output rate",
        "any-support-output match rate",
        "twin same-grid rate",
        "candidate change rate",
        "correct mapping switch rate",
        "pair joint normal acc",
        "mean Hamming",
    ]
    lines = [
        "# VRM Twin Epoch Baseline v1 comparison",
        "",
        "This is a fair-training-protocol baseline for the unchanged VRM V1.1-shaped architecture. It is not ARC training and not an ARC-solving claim.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for split in ("train", "dev", "sealed"):
        derived = final[split]["derived_metrics"]
        for condition, row in final[split]["conditions"].items():
            if condition == CONDITION_TWIN_SWAP_SUPPORTS:
                candidate_change = derived["normal_to_swapped_candidate_change_rate"]
                correct_switch = derived["correct_counterfactual_switch_rate"]
            elif condition == CONDITION_PAIR_ORDER_SHUFFLED:
                candidate_change = 1.0 - derived["normal_vs_pair_order_same_candidate_rate"]
                correct_switch = None
            else:
                candidate_change = None
                correct_switch = None
            values = [
                split,
                condition,
                f"{row['direct_grid_exact_accuracy']:.6f}",
                f"{row['foreground_color_accuracy']:.6f}",
                f"{row['candidate_nearest_score']:.6f}",
                f"{row['candidate_tie_rate']:.6f}",
                f"{row['derived_relevant_pair_output_rate']:.6f}",
                f"{row['predictions_matching_any_support_output_rate']:.6f}",
                f"{derived['twin_same_predicted_grid_rate']:.6f}",
                "NA" if candidate_change is None else f"{candidate_change:.6f}",
                "NA" if correct_switch is None else f"{correct_switch:.6f}",
                f"{derived['twin_pair_joint_normal_accuracy']:.6f}",
                f"{row['mean_target_rectangle_hamming_error']:.6f}",
            ]
            lines.append("|" + "|".join(values) + "|")
    lines.extend(
        [
            "",
            "For `TWIN_SWAP_SUPPORTS`, candidate change and correct mapping switch measure movement toward the opposite twin answer.",
            "For `PAIR_ORDER_SHUFFLED`, candidate change is relative to `NORMAL` and should remain low if predictions are order-stable.",
            "",
            f"Conclusion: `{verdict}`",
            "",
        ]
    )
    return "\n".join(lines)


def train_step(
    *,
    model: VRM,
    optimizer: torch.optim.Optimizer,
    episodes: list[TwinEpisode],
    batch_indices: list[int],
    epoch_index: int,
    global_step: int,
    total_steps: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    views = [epoch_training_view(episodes[idx], base_seed=args.seed, epoch_index=epoch_index)[0] for idx in batch_indices]
    batch = move_tensor_batch(collate_twin_episodes(views), device)
    lr = compute_lr(global_step, total_steps, args.learning_rate, args.warmup_steps, args.min_learning_rate)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**model_inputs_for_twin_batch(batch))
    loss, components = factorized_shape_color_loss(outputs, batch, args.shape_loss_weight)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    loss.backward()
    if not finite_gradients(model):
        raise RuntimeError("non-finite gradients")
    grad_before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm))
    optimizer.step()
    if not finite_parameters(model):
        raise RuntimeError("non-finite parameters")
    return {
        "event": "train_step",
        "global_step": global_step,
        "epoch_index": epoch_index,
        "episode_indices": batch_indices,
        "learning_rate": lr,
        "train_loss": float(loss.detach().cpu().item()),
        "color_loss": float(components["color_loss"].detach().cpu().item()),
        "height_loss": float(components["height_loss"].detach().cpu().item()),
        "width_loss": float(components["width_loss"].detach().cpu().item()),
        "grad_norm_before_clip": grad_before,
    }


def full_or_dry(args: argparse.Namespace) -> dict[str, Any]:
    out = args.out
    ckpt_dir = args.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    before = protected_hashes()
    save_json(out / "source_protection_hashes_before.json", before)
    determinism = configure_determinism(args.seed)
    device = resolve_device(args.device)
    probe_config = load_probe_config()
    config = config_from_probe(probe_config)
    parity = architecture_parity_report(probe_config, config, args)
    save_json(out / "architecture_parity_report.json", parity)
    train_episodes = load_split_episodes(args.data, "train", sealed_allowed=False)
    dev_episodes = load_split_episodes(args.data, "dev", sealed_allowed=False)
    steps_per_epoch = math.ceil(len(train_episodes) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    run_config = {
        "mode": args.mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_episode_count": len(train_episodes),
        "dev_episode_count": len(dev_episodes),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "data": str(args.data),
        "out": str(out),
        "checkpoint_dir": str(ckpt_dir),
        "determinism": determinism,
        "runtime": runtime_report(device),
    }
    save_json(out / "run_config.json", run_config)
    model, init_report = initialize_fresh_model(config, device, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    save_json(out / "initialization_report.json", init_report)
    support_report_epochs = args.epochs if args.mode == "full" else max(2, args.epochs)
    save_json(
        out / "support_pair_order_report.json",
        support_pair_order_report(train_episodes, base_seed=args.seed, epochs=support_report_epochs),
    )
    log_path = out / "training_log.jsonl"
    if args.resume_from is None and log_path.exists():
        log_path.unlink()
    start_epoch = 0
    start_offset = 0
    global_step = 0
    current_order: list[int] = []
    resume_report = {"resume_used": False}
    observed_by_epoch: dict[int, list[int]] = {idx: [] for idx in range(args.epochs)}
    dev_rows: list[dict[str, Any]] = []
    dry_train_losses: list[float] = []
    best_dev: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    if args.resume_from is not None:
        payload = load_baseline_checkpoint(args.resume_from, device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        global_step = int(payload["global_step"])
        start_epoch = int(payload["current_epoch"])
        start_offset = int(payload["next_batch_offset"])
        current_order = list(payload["current_epoch_permutation"])
        resume_report = {
            "resume_used": True,
            "resume_from": str(args.resume_from),
            "global_step": global_step,
            "current_epoch": start_epoch,
            "next_batch_offset": start_offset,
            "current_epoch_permutation_sha256": payload["current_epoch_permutation_sha256"],
            "remaining_episode_indices_sha256": sha256_json(current_order[start_offset:]),
            "will_not_restart_repeat_or_skip": True,
        }
        retained_rows = [row for row in load_jsonl(log_path) if int(row.get("global_step", row.get("step", -1))) <= global_step]
        rewrite_jsonl(log_path, retained_rows)
        for row in retained_rows:
            if row.get("event") == "train_step":
                observed_by_epoch[int(row["epoch_index"])].extend(int(idx) for idx in row["episode_indices"])
            elif row.get("event") == "dev_epoch_eval":
                dev_rows.append(row)
                if best_dev is None or selection_key(row) > selection_key(best_dev):
                    best_dev = dict(row)
        best_path = ckpt_dir / "twin_vrm_epoch_baseline_best_dev_epoch.pt"
        if best_path.exists():
            best_payload = load_baseline_checkpoint(best_path, device)
            best_state = {name: value.detach().cpu().clone() for name, value in best_payload["model_state_dict"].items()}
    save_json(out / "resume_integrity_report.json", resume_report)
    if global_step == 0:
        save_checkpoint(
            ckpt_dir / "twin_vrm_epoch_baseline_step00000.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            global_step=0,
            epoch_index=0,
            next_batch_offset=0,
            current_epoch_order=epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=0),
            total_steps=total_steps,
            args=args,
            extra={"initial_state": True},
        )
    start_time = time.perf_counter()
    max_batches_remaining = args.max_batches
    for epoch_index in range(start_epoch, args.epochs):
        order = current_order if epoch_index == start_epoch and current_order else epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=epoch_index)
        offset = start_offset if epoch_index == start_epoch else 0
        for batch_indices in epoch_batches(order, args.batch_size, offset):
            global_step += 1
            row = train_step(
                model=model,
                optimizer=optimizer,
                episodes=train_episodes,
                batch_indices=batch_indices,
                epoch_index=epoch_index,
                global_step=global_step,
                total_steps=total_steps,
                args=args,
                device=device,
            )
            if args.mode == "dry_run":
                dry_train_losses.append(float(row["train_loss"]))
            observed_by_epoch[epoch_index].extend(batch_indices)
            offset += len(batch_indices)
            append_jsonl(log_path, row)
            if args.mode == "full" and global_step % args.checkpoint_every_steps == 0:
                save_checkpoint(
                    ckpt_dir / f"twin_vrm_epoch_baseline_step{global_step:05d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    epoch_index=epoch_index,
                    next_batch_offset=offset,
                    current_epoch_order=order,
                    total_steps=total_steps,
                    args=args,
                    extra={"checkpoint_reason": "periodic_500_steps"},
                )
            if max_batches_remaining is not None:
                max_batches_remaining -= 1
                if max_batches_remaining <= 0:
                    dry_support = support_pair_order_report(train_episodes[:20], base_seed=args.seed, epochs=max(2, args.epochs))
                    dry_report = {
                        "mode": "dry_run",
                        "batches_run": args.max_batches,
                        "global_step": global_step,
                        "finite_parameters": finite_parameters(model),
                        "finite_train_losses": all(math.isfinite(loss) for loss in dry_train_losses),
                        "last_train_loss": dry_train_losses[-1] if dry_train_losses else None,
                        "fresh_initialization": init_report["fresh_random_initialization"],
                        "loaded_checkpoint": init_report["loaded_checkpoint"],
                        "candidate_metadata_excluded": True,
                        "target_excluded_from_forward": True,
                        "support_pair_order_report_passed": dry_support["passed"],
                        "support_pair_order_diagnostic_epochs": max(2, args.epochs),
                        "protected_hashes_unchanged": before == protected_hashes(),
                    }
                    save_json(out / "dry_run_report.json", dry_report)
                    save_json(out / "source_protection_hashes_after.json", protected_hashes())
                    print(json.dumps(to_jsonable(dry_report), indent=2, sort_keys=True), flush=True)
                    return dry_report
        normal_agg, normal_rows = evaluate_condition(
            model=model,
            episodes=dev_episodes,
            split="dev",
            condition=CONDITION_NORMAL,
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
            shape_loss_weight=args.shape_loss_weight,
        )
        normal_agg["epoch_index"] = epoch_index
        normal_agg["global_step"] = global_step
        normal_agg["twin_pair_joint_normal_accuracy"] = derived_correspondence_metrics(
            dev_episodes,
            {
                CONDITION_NORMAL: normal_rows,
                CONDITION_TWIN_SWAP_SUPPORTS: normal_rows,
                CONDITION_PAIR_ORDER_SHUFFLED: normal_rows,
            },
        )["twin_pair_joint_normal_accuracy"]
        normal_agg["selection_key"] = list(selection_key(normal_agg))
        dev_rows.append(normal_agg)
        append_jsonl(log_path, {"event": "dev_epoch_eval", **normal_agg})
        if best_dev is None or selection_key(normal_agg) > selection_key(best_dev):
            best_dev = dict(normal_agg)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if args.mode == "full":
                save_checkpoint(
                    ckpt_dir / "twin_vrm_epoch_baseline_best_dev_epoch.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    epoch_index=epoch_index,
                    next_batch_offset=len(order),
                    current_epoch_order=order,
                    total_steps=total_steps,
                    args=args,
                    extra={"provisional_best_dev_metrics": normal_agg},
                )
        if args.mode == "full":
            save_checkpoint(
                ckpt_dir / f"twin_vrm_epoch_baseline_epoch{epoch_index + 1:02d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                global_step=global_step,
                epoch_index=epoch_index,
                next_batch_offset=len(order),
                current_epoch_order=order,
                total_steps=total_steps,
                args=args,
                extra={"checkpoint_reason": "epoch_end", "dev_normal_metrics": normal_agg},
            )
        print(json.dumps(to_jsonable({"event": "epoch_complete", **normal_agg}), sort_keys=True), flush=True)
    episode_ids = [episode.episode_id for episode in train_episodes]
    coverage = {
        "all_epochs_passed": True,
        "epochs": [],
    }
    for epoch_index in range(args.epochs):
        order = epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=epoch_index)
        row = coverage_row(epoch_index=epoch_index, order=order, observed=observed_by_epoch[epoch_index], episode_ids=episode_ids, base_seed=args.seed)
        coverage["epochs"].append(row)
        coverage["all_epochs_passed"] = coverage["all_epochs_passed"] and row["passed"]
    save_json(out / "epoch_coverage_report.json", coverage)
    save_json(out / "dev_metrics_by_epoch.json", {"epochs": dev_rows})
    if best_dev is None or best_state is None:
        raise RuntimeError("no dev-selected checkpoint")
    best_selection = {
        "selected_epoch_index": best_dev["epoch_index"],
        "selected_global_step": best_dev["global_step"],
        "selection_metrics": best_dev,
        "selection_rule": "highest dev NORMAL candidate_nearest_score; tie-break twin_pair_joint_normal_accuracy, foreground_color_accuracy, lowest hamming",
        "sealed_split_read_before_selection": False,
    }
    save_json(out / "best_checkpoint_selection.json", best_selection)
    model.load_state_dict({name: tensor.to(device) for name, tensor in best_state.items()})
    save_checkpoint(
        ckpt_dir / "twin_vrm_epoch_baseline_best_dev_epoch.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        global_step=int(best_dev["global_step"]),
        epoch_index=int(best_dev["epoch_index"]),
        next_batch_offset=len(train_episodes),
        current_epoch_order=epoch_permutation_indices(len(train_episodes), base_seed=args.seed, epoch_index=int(best_dev["epoch_index"])),
        total_steps=total_steps,
        args=args,
        extra={"best_checkpoint_selection": best_selection},
    )
    selected_hash_before = sha256_file(ckpt_dir / "twin_vrm_epoch_baseline_best_dev_epoch.pt")
    sealed_episodes = load_split_episodes(args.data, "sealed", sealed_allowed=(out / "best_checkpoint_selection.json").exists())
    final: dict[str, Any] = {}
    per_episode_payload: dict[str, Any] = {}
    for split, episodes in (("train", train_episodes), ("dev", dev_episodes), ("sealed", sealed_episodes)):
        metrics, per_condition = evaluate_split_all_conditions(
            model=model,
            episodes=episodes,
            split=split,
            device=device,
            seed=args.seed,
            batch_size=args.eval_batch_size,
            shape_loss_weight=args.shape_loss_weight,
        )
        final[split] = metrics
        per_episode_payload[split] = per_condition
        save_json(out / f"final_{split}_metrics.json", metrics)
        save_json(out / f"per_episode_{split}.json", per_condition)
    selected_hash_after = sha256_file(ckpt_dir / "twin_vrm_epoch_baseline_best_dev_epoch.pt")
    verdict = verdict_from_metrics(final)
    save_text(out / "train_dev_sealed_comparison.md", comparison_markdown(final, verdict))
    pytest_result = run_pytest(out)
    after = protected_hashes()
    save_json(out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "runtime_seconds": time.perf_counter() - start_time,
        "runtime": runtime_report(device),
        "pytest_result": pytest_result,
        "run_config": run_config,
        "architecture_parity": parity,
        "coverage_passed": coverage["all_epochs_passed"],
        "support_pair_order_passed": json.loads((out / "support_pair_order_report.json").read_text())["passed"],
        "best_checkpoint_selection": best_selection,
        "selected_checkpoint_hash_before_sealed": selected_hash_before,
        "selected_checkpoint_hash_after_sealed": selected_hash_after,
        "sealed_evaluation_left_selected_checkpoint_unchanged": selected_hash_before == selected_hash_after,
        "source_protection_hashes_unchanged": before == after,
        "final_metrics": final,
    }
    save_json(out / "summary_report.json", summary)
    save_text(out / "summary.md", comparison_markdown(final, verdict) + f"\nRuntime seconds: {summary['runtime_seconds']:.3f}\n")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VRM Twin Epoch Baseline v1")
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=500)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--shape-loss-weight", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    full_or_dry(args)


if __name__ == "__main__":
    main()
