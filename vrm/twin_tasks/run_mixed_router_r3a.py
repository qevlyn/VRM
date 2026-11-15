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

from .mixed_router_r3a_data import (
    GLOBAL_SEED,
    ROUTE_FAMILIES,
    ROUTE_HARD_NEGATIVE,
    ROUTE_RETRIEVAL,
    ROUTE_TRANSFORM,
    TRUE_ROUTE_READER,
    TRUE_ROUTE_TRANSFORM,
    R3AEpisode,
    counterfactual_group_report,
    grid_to_list,
    load_split,
    query_type_key,
    reorder_episode,
    route_balance_report,
    save_dataset,
    stable_hash,
    stable_int,
    support_signature,
    tensors_from_episodes,
)
from .mixed_router_r3a_oracles import certify_transform_oracle, transform_oracle_probs
from .mixed_router_r3a_reader import PairKeyValueReaderR3A, reader_output_loss
from .mixed_router_r3a_router import MixedRouteRouterR3A, grad_norm, router_output_loss

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/mixed_router_r3a_v1"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/mixed_router_r3a_v1"
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
)
PROTECTED_DIRS = (
    PROJECT_ROOT / "artifacts/twin_task_pilot_v1",
    PROJECT_ROOT / "artifacts/twin_vrm_probe_v0",
    PROJECT_ROOT / "artifacts/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1",
    PROJECT_ROOT / "artifacts/twin_pair_reader_r1b",
    PROJECT_ROOT / "artifacts/twin_vrm_pair_reader_r2a",
    PROJECT_ROOT / "checkpoints/twin_vrm_probe_v0",
    PROJECT_ROOT / "checkpoints/twin_vrm_epoch_baseline_v1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1",
    PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b",
    PROJECT_ROOT / "checkpoints/twin_vrm_pair_reader_r2a",
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


def tensor_grid_to_list(grid: torch.Tensor) -> list[list[int]]:
    return [[int(value) for value in row] for row in grid.detach().cpu().long().tolist()]


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
        hashes[key] = sha256_file(path) if path.exists() else None
        seen.add(key)
    for directory in PROTECTED_DIRS:
        if not directory.exists():
            hashes[str(directory)] = None
            continue
        for path in sorted(child for child in directory.rglob("*") if child.is_file()):
            key = str(path)
            if key in seen:
                continue
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
    return perms[stable_int(f"{seed}|{epoch}|{episode_id}|support_order") % len(perms)]


def training_view(episode: R3AEpisode, seed: int, epoch: int) -> R3AEpisode:
    return reorder_episode(episode, support_order(episode.episode_id, seed, epoch))


def transform_specs(episodes: list[R3AEpisode]) -> list[str]:
    return [episode.transform_spec for episode in episodes]


def batch_for(episodes: list[R3AEpisode], device: torch.device) -> dict[str, torch.Tensor]:
    return tensors_from_episodes(episodes, device)


def true_route_is_reader(episode: R3AEpisode) -> bool:
    return episode.true_route == TRUE_ROUTE_READER


def save_reader_checkpoint(path: Path, model: PairKeyValueReaderR3A, optimizer: torch.optim.Optimizer, seed: int, epoch: int, step: int, extra: dict[str, Any] | None = None) -> None:
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


def load_reader_checkpoint(path: Path, device: torch.device) -> PairKeyValueReaderR3A:
    model = PairKeyValueReaderR3A().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model


def save_router_checkpoint(path: Path, model: MixedRouteRouterR3A, optimizer: torch.optim.Optimizer, seed: int, epoch: int, step: int, extra: dict[str, Any] | None = None) -> None:
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


def load_router_gate(model: MixedRouteRouterR3A, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.gate_head.load_state_dict(payload["gate_state_dict"])


def reader_metrics(model: PairKeyValueReaderR3A, episodes: list[R3AEpisode], device: torch.device, batch_size: int, *, permutations: bool = False) -> dict[str, Any]:
    model.eval()
    count = 0
    exact = 0
    fg_correct = 0
    fg_total = 0
    top1 = 0
    rel_mass: list[float] = []
    perm_same_grid = 0
    perm_same_logical = 0
    perm_total = 0
    perms = list(itertools.permutations(range(3)))
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = batch_for(batch_eps, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            decoded = out["decoded_grid"]
            targets = batch["target_grids"]
            count += len(batch_eps)
            exact += int((decoded == targets).flatten(1).all(dim=1).sum().item())
            fg = targets != 0
            fg_correct += int(((decoded == targets) & fg).sum().item())
            fg_total += int(fg.sum().item())
            alpha = out["retrieval_weights"]
            for idx, episode in enumerate(batch_eps):
                if episode.relevant_pair_index is not None:
                    top1 += int(int(alpha[idx].argmax().item()) == episode.relevant_pair_index)
                    rel_mass.append(float(alpha[idx, episode.relevant_pair_index].item()))
            if permutations:
                base_decoded = decoded.detach().cpu()
                base_top = alpha.argmax(dim=-1).detach().cpu().tolist()
                for perm in perms:
                    perm_eps = [reorder_episode(ep, perm) for ep in batch_eps]
                    perm_batch = batch_for(perm_eps, device)
                    perm_out = model(perm_batch["query_grids"], perm_batch["support_input_grids"], perm_batch["support_output_grids"])
                    perm_decoded = perm_out["decoded_grid"].detach().cpu()
                    perm_alpha = perm_out["retrieval_weights"].detach().cpu()
                    for idx, episode in enumerate(batch_eps):
                        perm_total += 1
                        perm_same_grid += int(torch.equal(base_decoded[idx], perm_decoded[idx]))
                        logical_alpha = torch.zeros_like(perm_alpha[idx])
                        for new_pos, old_pos in enumerate(perm):
                            logical_alpha[old_pos] = perm_alpha[idx, new_pos]
                        perm_same_logical += int(int(logical_alpha.argmax().item()) == base_top[idx])
    return {
        "episode_count": count,
        "direct_grid_exact_accuracy": exact / max(count, 1),
        "foreground_color_accuracy": fg_correct / max(fg_total, 1),
        "retrieval_top1_accuracy": top1 / max(count, 1),
        "mean_relevant_pair_mass": statistics.fmean(rel_mass) if rel_mass else 0.0,
        "all_permutation_same_final_grid_rate": perm_same_grid / max(perm_total, 1) if permutations else None,
        "all_permutation_same_logical_pair_rate": perm_same_logical / max(perm_total, 1) if permutations else None,
        "permutation_presentation_count": perm_total if permutations else 0,
    }


def train_reader_seed(seed: int, args: argparse.Namespace, train: list[R3AEpisode], dev: list[R3AEpisode], sealed: list[R3AEpisode], device: torch.device) -> dict[str, Any]:
    train_retrieval = [episode for episode in train if episode.route_family == ROUTE_RETRIEVAL]
    dev_retrieval = [episode for episode in dev if episode.route_family == ROUTE_RETRIEVAL]
    sealed_retrieval = [episode for episode in sealed if episode.route_family == ROUTE_RETRIEVAL]
    model = PairKeyValueReaderR3A().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    save_reader_checkpoint(args.checkpoint_out / f"reader_r3a_seed_{seed}_initial.pt", model, optimizer, seed, -1, 0)
    log_path = args.data_out / f"reader_training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    dev_rows: list[dict[str, Any]] = []
    global_step = 0
    best_key: tuple[float, float, float, int] | None = None
    best_epoch = -1
    for epoch in range(args.reader_epochs):
        order = epoch_order(len(train_retrieval), seed, epoch)
        observed: list[int] = []
        for indices in batches(order, args.batch_size):
            global_step += 1
            view = [training_view(train_retrieval[idx], seed, epoch) for idx in indices]
            batch = batch_for(view, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = reader_output_loss(out["reader_probs"], batch["target_grids"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            observed.extend(indices)
            append_jsonl(log_path, {"event": "reader_train_step", "seed": seed, "epoch_index": epoch, "global_step": global_step, "loss": float(loss.detach().cpu().item())})
            if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
                break
        save_reader_checkpoint(args.checkpoint_out / f"reader_r3a_seed_{seed}_epoch_{epoch:02d}.pt", model, optimizer, seed, epoch, global_step)
        eval_dev = dev_retrieval[:512] if args.mode == "dry_run" else dev_retrieval
        metrics = reader_metrics(model, eval_dev, device, args.batch_size)
        row = {"epoch_index": epoch, "global_step": global_step, **metrics}
        dev_rows.append(row)
        key = (row["direct_grid_exact_accuracy"], row["foreground_color_accuracy"], row["mean_relevant_pair_mass"], -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            save_reader_checkpoint(args.checkpoint_out / f"reader_r3a_seed_{seed}_best_dev.pt", model, optimizer, seed, epoch, global_step, {"best_dev_metrics": row})
        if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
            break
    save_json(args.data_out / f"reader_dev_metrics_by_epoch_seed_{seed}.json", {"seed": seed, "epochs": dev_rows})
    best_path = args.checkpoint_out / f"reader_r3a_seed_{seed}_best_dev.pt"
    best_model = load_reader_checkpoint(best_path, device)
    cert_episodes = sealed_retrieval[:512] if args.mode == "dry_run" else sealed_retrieval
    cert = reader_metrics(best_model, cert_episodes, device, args.batch_size, permutations=True)
    cert["passed"] = (
        cert["direct_grid_exact_accuracy"] >= 0.99
        and cert["foreground_color_accuracy"] >= 0.99
        and cert["retrieval_top1_accuracy"] >= 0.99
        and cert["mean_relevant_pair_mass"] >= 0.99
        and cert["all_permutation_same_final_grid_rate"] >= 0.999
        and cert["all_permutation_same_logical_pair_rate"] >= 0.999
    )
    selection = {
        "seed": seed,
        "selected_epoch_index": best_epoch,
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": sha256_file(best_path),
        "selection_rule": "highest dev exact, foreground, relevant mass, earliest epoch",
    }
    save_json(args.data_out / f"reader_best_checkpoint_selection_seed_{seed}.json", selection)
    save_json(args.data_out / f"reader_sealed_certification_seed_{seed}.json", cert)
    return {"seed": seed, "model": best_model, "selection": selection, "certification": cert}


def family_metric_template() -> dict[str, Any]:
    return {
        "count": 0,
        "exact": 0,
        "fg_correct": 0,
        "fg_total": 0,
        "all_correct": 0,
        "all_total": 0,
        "hamming_sum": 0.0,
        "route_correct": 0,
        "gate_values": [],
        "reader_exact": 0,
        "transform_exact": 0,
        "reader_top1": 0,
        "reader_relevant_mass": [],
        "max_score": [],
        "margin": [],
        "entropy": [],
        "min_distance": [],
        "mean_distance": [],
    }


def finalize_family_metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    final: dict[str, Any] = {}
    for family, row in rows.items():
        gates = row["gate_values"]
        final[family] = {
            "episode_count": row["count"],
            "direct_grid_exact_accuracy": row["exact"] / max(row["count"], 1),
            "foreground_color_accuracy": row["fg_correct"] / max(row["fg_total"], 1),
            "all_cell_color_accuracy": row["all_correct"] / max(row["all_total"], 1),
            "target_rectangle_hamming_error": row["hamming_sum"] / max(row["count"], 1),
            "route_accuracy": row["route_correct"] / max(row["count"], 1),
            "mean_gate": statistics.fmean(gates) if gates else 0.0,
            "median_gate": statistics.median(gates) if gates else 0.0,
            "gate_p10": sorted(gates)[int(0.1 * (len(gates) - 1))] if gates else 0.0,
            "gate_p90": sorted(gates)[int(0.9 * (len(gates) - 1))] if gates else 0.0,
            "fraction_gate_ge_0_5": sum(1 for value in gates if value >= 0.5) / max(len(gates), 1),
            "fraction_gate_ge_0_9": sum(1 for value in gates if value >= 0.9) / max(len(gates), 1),
            "reader_exact_accuracy": row["reader_exact"] / max(row["count"], 1),
            "transform_oracle_exact_accuracy": row["transform_exact"] / max(row["count"], 1),
            "reader_top1_accuracy": row["reader_top1"] / max(row["count"], 1),
            "reader_relevant_pair_mass": statistics.fmean(row["reader_relevant_mass"]) if row["reader_relevant_mass"] else 0.0,
            "max_score_mean": statistics.fmean(row["max_score"]) if row["max_score"] else 0.0,
            "margin_mean": statistics.fmean(row["margin"]) if row["margin"] else 0.0,
            "alpha_entropy_mean": statistics.fmean(row["entropy"]) if row["entropy"] else 0.0,
            "min_q_key_distance_mean": statistics.fmean(row["min_distance"]) if row["min_distance"] else 0.0,
            "mean_q_key_distance_mean": statistics.fmean(row["mean_distance"]) if row["mean_distance"] else 0.0,
        }
    return final


def normal_router_metrics(model: MixedRouteRouterR3A, episodes: list[R3AEpisode], device: torch.device, batch_size: int) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows = {family: family_metric_template() for family in ROUTE_FAMILIES}
    predictions: dict[str, torch.Tensor] = {}
    row_details: list[dict[str, Any]] = []
    route_matrix = {
        TRUE_ROUTE_READER: {TRUE_ROUTE_READER: 0, TRUE_ROUTE_TRANSFORM: 0},
        TRUE_ROUTE_TRANSFORM: {TRUE_ROUTE_READER: 0, TRUE_ROUTE_TRANSFORM: 0},
    }
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = batch_for(batch_eps, device)
            transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs(batch_eps))
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs)
            decoded = out["decoded_grid"]
            reader_decoded = out["reader_decoded_grid"]
            transform_decoded = out["transform_decoded_grid"]
            targets = batch["target_grids"]
            alpha = out["reader_alpha"]
            logits = out["retrieval_logits"]
            distances = out["qk_distances"]
            for idx, episode in enumerate(batch_eps):
                family = episode.route_family
                row = rows[family]
                row["count"] += 1
                exact = bool((decoded[idx] == targets[idx]).all().item())
                reader_exact = bool((reader_decoded[idx] == targets[idx]).all().item())
                transform_exact = bool((transform_decoded[idx] == targets[idx]).all().item())
                fg = targets[idx] != 0
                row["exact"] += int(exact)
                row["fg_correct"] += int(((decoded[idx] == targets[idx]) & fg).sum().item())
                row["fg_total"] += int(fg.sum().item())
                row["all_correct"] += int((decoded[idx] == targets[idx]).sum().item())
                row["all_total"] += int(targets[idx].numel())
                row["hamming_sum"] += float((decoded[idx] != targets[idx]).sum().item())
                gate = float(out["gate"][idx].item())
                row["gate_values"].append(gate)
                predicted_route = TRUE_ROUTE_READER if gate >= 0.5 else TRUE_ROUTE_TRANSFORM
                route_matrix[episode.true_route][predicted_route] += 1
                row["route_correct"] += int(predicted_route == episode.true_route)
                row["reader_exact"] += int(reader_exact)
                row["transform_exact"] += int(transform_exact)
                if episode.relevant_pair_index is not None:
                    row["reader_top1"] += int(int(alpha[idx].argmax().item()) == episode.relevant_pair_index)
                    row["reader_relevant_mass"].append(float(alpha[idx, episode.relevant_pair_index].item()))
                top2 = torch.topk(logits[idx], k=2).values
                row["max_score"].append(float(logits[idx].max().item()))
                row["margin"].append(float((top2[0] - top2[1]).item()))
                row["entropy"].append(float((-(alpha[idx].clamp_min(1e-8) * torch.log(alpha[idx].clamp_min(1e-8))).sum()).item()))
                row["min_distance"].append(float(distances[idx].min().item()))
                row["mean_distance"].append(float(distances[idx].mean().item()))
                predictions[episode.episode_id] = decoded[idx].detach().cpu()
                row_details.append(
                    {
                        "episode_id": episode.episode_id,
                        "counterfactual_group_id": episode.counterfactual_group_id,
                        "query_type_key": query_type_key(episode.query_type),
                        "route_family": family,
                        "true_route": episode.true_route,
                        "predicted_route": predicted_route,
                        "gate": gate,
                        "exact": exact,
                    }
                )
    family_metrics = finalize_family_metrics(rows)
    total = sum(row["episode_count"] for row in family_metrics.values())
    macro_exact = statistics.fmean(row["direct_grid_exact_accuracy"] for row in family_metrics.values())
    overall_route = sum(rows[family]["route_correct"] for family in ROUTE_FAMILIES) / max(total, 1)
    hard_rejection = family_metrics[ROUTE_HARD_NEGATIVE]["route_accuracy"]
    return {
        "overall": {
            "episode_count": total,
            "direct_grid_exact_accuracy": sum(rows[family]["exact"] for family in ROUTE_FAMILIES) / max(total, 1),
            "macro_direct_grid_exact_accuracy": macro_exact,
            "route_accuracy": overall_route,
            "route_confusion_matrix": route_matrix,
            "retrieval_route_recall": family_metrics[ROUTE_RETRIEVAL]["route_accuracy"],
            "transform_route_recall": (
                rows[ROUTE_TRANSFORM]["route_correct"] + rows[ROUTE_HARD_NEGATIVE]["route_correct"]
            ) / max(rows[ROUTE_TRANSFORM]["count"] + rows[ROUTE_HARD_NEGATIVE]["count"], 1),
            "hard_negative_rejection_rate": hard_rejection,
        },
        "families": family_metrics,
    }, predictions, row_details


def force_route_metrics(model: MixedRouteRouterR3A, episodes: list[R3AEpisode], device: torch.device, batch_size: int, gate: float) -> dict[str, Any]:
    counts = {family: 0 for family in ROUTE_FAMILIES}
    exact = {family: 0 for family in ROUTE_FAMILIES}
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = batch_for(batch_eps, device)
            transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs(batch_eps))
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs, force_gate=gate)
            decoded = out["decoded_grid"]
            for idx, episode in enumerate(batch_eps):
                counts[episode.route_family] += 1
                exact[episode.route_family] += int(bool((decoded[idx] == batch["target_grids"][idx]).all().item()))
    return {
        "force_gate": gate,
        "exact_accuracy_by_family": {family: exact[family] / max(counts[family], 1) for family in ROUTE_FAMILIES},
    }


def disabled_metrics(model: MixedRouteRouterR3A, episodes: list[R3AEpisode], normal_predictions: dict[str, torch.Tensor], device: torch.device, batch_size: int, *, reader_disabled: bool = False, transform_disabled: bool = False) -> dict[str, Any]:
    counts = {family: 0 for family in ROUTE_FAMILIES}
    changed = {family: 0 for family in ROUTE_FAMILIES}
    gate_same = True
    alpha_same = True
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = batch_for(batch_eps, device)
            transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs(batch_eps))
            base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs)
            out = model(
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                transform_probs,
                reader_disabled=reader_disabled,
                transform_disabled=transform_disabled,
            )
            gate_same = gate_same and bool(torch.equal(base["gate"], out["gate"]))
            alpha_same = alpha_same and bool(torch.equal(base["reader_alpha"], out["reader_alpha"]))
            for idx, episode in enumerate(batch_eps):
                counts[episode.route_family] += 1
                changed[episode.route_family] += int(not torch.equal(normal_predictions[episode.episode_id], out["decoded_grid"][idx].detach().cpu()))
    return {
        "reader_disabled": reader_disabled,
        "transform_disabled": transform_disabled,
        "grid_change_rate_by_family": {family: changed[family] / max(counts[family], 1) for family in ROUTE_FAMILIES},
        "gate_unchanged": gate_same,
        "reader_alpha_unchanged": alpha_same,
    }


def permutation_metrics(model: MixedRouteRouterR3A, episodes: list[R3AEpisode], device: torch.device, batch_size: int) -> dict[str, Any]:
    perms = list(itertools.permutations(range(3)))
    same_grid = 0
    same_route = 0
    same_logical = 0
    total = 0
    max_final_diff = 0.0
    max_alpha_diff = 0.0
    max_gate_diff = 0.0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_eps = episodes[start : start + batch_size]
            batch = batch_for(batch_eps, device)
            transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs(batch_eps))
            base = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs)
            base_route = (base["gate"] >= 0.5).detach().cpu()
            base_top = base["reader_alpha"].argmax(dim=-1).detach().cpu()
            base_probs = base["final_probs"].detach().cpu()
            base_alpha = base["reader_alpha"].detach().cpu()
            base_gate = base["gate"].detach().cpu()
            base_decoded = base["decoded_grid"].detach().cpu()
            for perm in perms:
                perm_eps = [reorder_episode(ep, perm) for ep in batch_eps]
                perm_batch = batch_for(perm_eps, device)
                perm_transform_probs = transform_oracle_probs(perm_batch["query_grids"], transform_specs(perm_eps))
                out = model(perm_batch["query_grids"], perm_batch["support_input_grids"], perm_batch["support_output_grids"], perm_transform_probs)
                probs = out["final_probs"].detach().cpu()
                alpha = out["reader_alpha"].detach().cpu()
                gate = out["gate"].detach().cpu()
                decoded = out["decoded_grid"].detach().cpu()
                route = (gate >= 0.5).detach().cpu()
                for idx in range(len(batch_eps)):
                    total += 1
                    same_grid += int(torch.equal(base_decoded[idx], decoded[idx]))
                    same_route += int(bool(base_route[idx].item()) == bool(route[idx].item()))
                    logical_alpha = torch.zeros_like(alpha[idx])
                    for new_pos, old_pos in enumerate(perm):
                        logical_alpha[old_pos] = alpha[idx, new_pos]
                    same_logical += int(int(logical_alpha.argmax().item()) == int(base_top[idx].item()))
                    max_final_diff = max(max_final_diff, float((base_probs[idx] - probs[idx]).abs().max().item()))
                    max_alpha_diff = max(max_alpha_diff, float((base_alpha[idx] - logical_alpha).abs().max().item()))
                    max_gate_diff = max(max_gate_diff, float((base_gate[idx] - gate[idx]).abs().max().item()))
    return {
        "permutation_presentation_count": total,
        "all_permutation_same_final_grid_rate": same_grid / max(total, 1),
        "all_permutation_same_route_rate": same_route / max(total, 1),
        "all_permutation_same_logical_reader_pair_rate": same_logical / max(total, 1),
        "max_abs_final_probability_difference_after_logical_unpermute": max_final_diff,
        "max_abs_alpha_difference_after_logical_unpermute": max_alpha_diff,
        "max_abs_gate_difference_after_logical_unpermute": max_gate_diff,
    }


def same_query_counterfactual_report(row_details: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in row_details:
        groups.setdefault(row["counterfactual_group_id"], []).append(row)
    correct = 0
    sensitive = 0
    drops: list[float] = []
    samples: list[dict[str, Any]] = []
    for group_id, rows in groups.items():
        by_family = {row["route_family"]: row for row in rows}
        if set(by_family) != set(ROUTE_FAMILIES):
            continue
        retrieval = by_family[ROUTE_RETRIEVAL]
        transform = by_family[ROUTE_TRANSFORM]
        hard = by_family[ROUTE_HARD_NEGATIVE]
        ok = (
            retrieval["predicted_route"] == TRUE_ROUTE_READER
            and transform["predicted_route"] == TRUE_ROUTE_TRANSFORM
            and hard["predicted_route"] == TRUE_ROUTE_TRANSFORM
        )
        correct += int(ok)
        sensitive += int(len({retrieval["predicted_route"], transform["predicted_route"], hard["predicted_route"]}) > 1)
        drops.append(float(retrieval["gate"]) - (float(transform["gate"]) + float(hard["gate"])) / 2.0)
        if len(samples) < 20:
            samples.append(
                {
                    "counterfactual_group_id": group_id,
                    "retrieval_gate": retrieval["gate"],
                    "transform_only_gate": transform["gate"],
                    "hard_negative_gate": hard["gate"],
                    "route_flip_correct": ok,
                }
            )
    count = len(groups)
    return {
        "counterfactual_group_count": count,
        "route_flip_correct_rate": correct / max(count, 1),
        "same_query_support_sensitive_route_rate": sensitive / max(count, 1),
        "retrieval_to_transform_gate_drop": statistics.fmean(drops) if drops else 0.0,
        "examples": samples,
    }


def heldout_query_type_report(row_details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in row_details:
        grouped.setdefault(row["query_type_key"], []).append(row)
    rows = []
    for key, items in grouped.items():
        families = {item["route_family"] for item in items}
        exact = sum(int(item["exact"]) for item in items) / max(len(items), 1)
        route = sum(int(item["predicted_route"] == item["true_route"]) for item in items) / max(len(items), 1)
        hard_items = [item for item in items if item["route_family"] == ROUTE_HARD_NEGATIVE]
        hard_rejection = sum(int(item["predicted_route"] == TRUE_ROUTE_TRANSFORM) for item in hard_items) / max(len(hard_items), 1)
        rows.append(
            {
                "query_type_key": key,
                "episode_count": len(items),
                "families": sorted(families),
                "all_three_families_present": families == set(ROUTE_FAMILIES),
                "direct_grid_exact_accuracy": exact,
                "route_accuracy": route,
                "hard_negative_rejection_rate": hard_rejection,
            }
        )
    return {
        "query_type_count": len(rows),
        "worst_direct_grid_exact_accuracy": min(row["direct_grid_exact_accuracy"] for row in rows),
        "worst_route_accuracy": min(row["route_accuracy"] for row in rows),
        "worst_hard_negative_rejection_rate": min(row["hard_negative_rejection_rate"] for row in rows),
        "all_query_types_have_at_least_30_episodes": all(row["episode_count"] >= 30 for row in rows),
        "all_query_types_have_all_families": all(row["all_three_families_present"] for row in rows),
        "rows": rows,
    }


def evaluate_router_split(model: MixedRouteRouterR3A, episodes: list[R3AEpisode], split: str, device: torch.device, batch_size: int) -> dict[str, Any]:
    normal, normal_predictions, row_details = normal_router_metrics(model, episodes, device, batch_size)
    force_reader = force_route_metrics(model, episodes, device, batch_size, gate=1.0)
    force_transform = force_route_metrics(model, episodes, device, batch_size, gate=0.0)
    reader_disabled = disabled_metrics(model, episodes, normal_predictions, device, batch_size, reader_disabled=True)
    transform_disabled = disabled_metrics(model, episodes, normal_predictions, device, batch_size, transform_disabled=True)
    perms = permutation_metrics(model, episodes, device, batch_size)
    return {
        "split": split,
        "conditions": {
            "NORMAL": normal,
            "ALL_PAIR_PERMUTATIONS": perms,
            "FORCE_READER_ROUTE": force_reader,
            "FORCE_TRANSFORM_ROUTE": force_transform,
            "READER_DISABLED": reader_disabled,
            "TRANSFORM_DISABLED": transform_disabled,
        },
        "heldout_query_type_report": heldout_query_type_report(row_details),
        "same_query_counterfactual_report": same_query_counterfactual_report(row_details),
    }


def router_selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, float, int]:
    normal = metrics["conditions"]["NORMAL"]
    macro = normal["overall"]["macro_direct_grid_exact_accuracy"]
    route_acc = normal["overall"]["route_accuracy"]
    hard_exact = normal["families"][ROUTE_HARD_NEGATIVE]["direct_grid_exact_accuracy"]
    confusion = 1.0 - route_acc
    return (macro, route_acc, hard_exact, -confusion, -epoch)


def train_router_seed(seed: int, args: argparse.Namespace, reader_path: Path, train: list[R3AEpisode], dev: list[R3AEpisode], device: torch.device) -> dict[str, Any]:
    reader = load_reader_checkpoint(reader_path, device)
    model = MixedRouteRouterR3A(reader).to(device)
    optimizer = torch.optim.AdamW(model.gate_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    save_router_checkpoint(args.checkpoint_out / f"router_r3a_seed_{seed}_initial.pt", model, optimizer, seed, -1, 0)
    log_path = args.data_out / f"router_training_log_seed_{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()
    dev_rows: list[dict[str, Any]] = []
    observed_by_epoch: dict[int, list[int]] = {}
    global_step = 0
    best_key: tuple[float, float, float, float, int] | None = None
    best_epoch = -1
    grad_rows: list[dict[str, Any]] = []
    for epoch in range(args.router_epochs):
        order = epoch_order(len(train), seed, epoch + 1000)
        observed: list[int] = []
        for indices in batches(order, args.batch_size):
            global_step += 1
            view = [training_view(train[idx], seed, epoch) for idx in indices]
            batch = batch_for(view, device)
            transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs(view))
            optimizer.zero_grad(set_to_none=True)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs)
            loss = router_output_loss(out["final_probs"], batch["target_grids"])
            loss.backward()
            reader_grad = grad_norm(model.reader.parameters())
            gate_grad = grad_norm(model.gate_head.parameters())
            torch.nn.utils.clip_grad_norm_(model.gate_head.parameters(), 1.0)
            optimizer.step()
            observed.extend(indices)
            row = {
                "event": "router_train_step",
                "seed": seed,
                "epoch_index": epoch,
                "global_step": global_step,
                "loss": float(loss.detach().cpu().item()),
                "reader_grad_norm": reader_grad,
                "gate_grad_norm": gate_grad,
                "mean_gate": float(out["gate"].detach().mean().cpu().item()),
            }
            grad_rows.append(row)
            append_jsonl(log_path, row)
            if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
                break
        observed_by_epoch[epoch] = observed
        save_router_checkpoint(args.checkpoint_out / f"router_r3a_seed_{seed}_epoch_{epoch:02d}.pt", model, optimizer, seed, epoch, global_step)
        eval_dev = dev[:512] if args.mode == "dry_run" else dev
        dev_metrics = evaluate_router_split(model, eval_dev, "dev", device, args.batch_size)
        row = {
            "epoch_index": epoch,
            "global_step": global_step,
            "macro_direct_grid_exact_accuracy": dev_metrics["conditions"]["NORMAL"]["overall"]["macro_direct_grid_exact_accuracy"],
            "route_accuracy": dev_metrics["conditions"]["NORMAL"]["overall"]["route_accuracy"],
            "hard_negative_transform_exact": dev_metrics["conditions"]["NORMAL"]["families"][ROUTE_HARD_NEGATIVE]["direct_grid_exact_accuracy"],
            "route_confusion_rate": 1.0 - dev_metrics["conditions"]["NORMAL"]["overall"]["route_accuracy"],
            "mean_gate_by_family": {
                family: dev_metrics["conditions"]["NORMAL"]["families"][family]["mean_gate"]
                for family in ROUTE_FAMILIES
            },
            "selection_key": router_selection_key(dev_metrics, epoch),
        }
        dev_rows.append(row)
        key = router_selection_key(dev_metrics, epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            save_router_checkpoint(args.checkpoint_out / f"router_r3a_seed_{seed}_best_dev.pt", model, optimizer, seed, epoch, global_step, {"best_dev_metrics": row})
        if args.mode == "dry_run" and args.max_batches is not None and global_step >= args.max_batches:
            break
    save_json(args.data_out / f"router_dev_metrics_by_epoch_seed_{seed}.json", {"seed": seed, "epochs": dev_rows})
    selection = {
        "seed": seed,
        "selected_epoch_index": best_epoch,
        "selected_checkpoint": str(args.checkpoint_out / f"router_r3a_seed_{seed}_best_dev.pt"),
        "selected_checkpoint_sha256_before_sealed": sha256_file(args.checkpoint_out / f"router_r3a_seed_{seed}_best_dev.pt"),
        "selection_rule": "macro exact, route accuracy, hard-negative exact, low confusion, earliest epoch",
    }
    save_json(args.data_out / f"router_best_checkpoint_selection_seed_{seed}.json", selection)
    grad_report = {
        "seed": seed,
        "max_reader_grad_norm": max(row["reader_grad_norm"] for row in grad_rows),
        "max_gate_grad_norm": max(row["gate_grad_norm"] for row in grad_rows),
        "min_positive_gate_grad_norm": min(row["gate_grad_norm"] for row in grad_rows if row["gate_grad_norm"] > 0.0),
        "reader_frozen_grad_zero": all(row["reader_grad_norm"] == 0.0 for row in grad_rows),
    }
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
    return {"seed": seed, "selection": selection, "grad_report": grad_report, "coverage": coverage}


def load_best_router(seed: int, args: argparse.Namespace, device: torch.device) -> MixedRouteRouterR3A:
    reader_path = args.checkpoint_out / f"reader_r3a_seed_{seed}_best_dev.pt"
    reader = load_reader_checkpoint(reader_path, device)
    model = MixedRouteRouterR3A(reader).to(device)
    load_router_gate(model, args.checkpoint_out / f"router_r3a_seed_{seed}_best_dev.pt", device)
    model.eval()
    return model


def seed_gates(seed_summary: dict[str, Any]) -> dict[str, Any]:
    sealed = seed_summary["sealed"]
    normal = sealed["conditions"]["NORMAL"]
    families = normal["families"]
    overall = normal["overall"]
    perms = sealed["conditions"]["ALL_PAIR_PERMUTATIONS"]
    force_reader = sealed["conditions"]["FORCE_READER_ROUTE"]["exact_accuracy_by_family"]
    force_transform = sealed["conditions"]["FORCE_TRANSFORM_ROUTE"]["exact_accuracy_by_family"]
    reader_disabled = sealed["conditions"]["READER_DISABLED"]["grid_change_rate_by_family"]
    transform_disabled = sealed["conditions"]["TRANSFORM_DISABLED"]["grid_change_rate_by_family"]
    heldout = sealed["heldout_query_type_report"]
    counter = sealed["same_query_counterfactual_report"]
    defs = {
        "reader_certification": (seed_summary["reader_certification"]["passed"], "is", True),
        "macro_direct_exact": (overall["macro_direct_grid_exact_accuracy"], ">=", 0.99),
        "overall_route_accuracy": (overall["route_accuracy"], ">=", 0.99),
        "retrieval_exact": (families[ROUTE_RETRIEVAL]["direct_grid_exact_accuracy"], ">=", 0.99),
        "retrieval_route_recall": (families[ROUTE_RETRIEVAL]["route_accuracy"], ">=", 0.99),
        "retrieval_mean_gate": (families[ROUTE_RETRIEVAL]["mean_gate"], ">=", 0.95),
        "transform_exact": (families[ROUTE_TRANSFORM]["direct_grid_exact_accuracy"], ">=", 0.99),
        "transform_route_recall": (families[ROUTE_TRANSFORM]["route_accuracy"], ">=", 0.99),
        "transform_mean_gate": (families[ROUTE_TRANSFORM]["mean_gate"], "<=", 0.05),
        "hard_exact": (families[ROUTE_HARD_NEGATIVE]["direct_grid_exact_accuracy"], ">=", 0.99),
        "hard_negative_rejection": (families[ROUTE_HARD_NEGATIVE]["route_accuracy"], ">=", 0.99),
        "hard_mean_gate": (families[ROUTE_HARD_NEGATIVE]["mean_gate"], "<=", 0.05),
        "heldout_macro_exact": (heldout["worst_direct_grid_exact_accuracy"], ">=", 0.99),
        "heldout_route_accuracy": (heldout["worst_route_accuracy"], ">=", 0.99),
        "heldout_hard_rejection": (heldout["worst_hard_negative_rejection_rate"], ">=", 0.99),
        "permutation_grid": (perms["all_permutation_same_final_grid_rate"], ">=", 0.999),
        "permutation_route": (perms["all_permutation_same_route_rate"], ">=", 0.999),
        "permutation_logical_pair": (perms["all_permutation_same_logical_reader_pair_rate"], ">=", 0.999),
        "permutation_final_diff": (perms["max_abs_final_probability_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_alpha_diff": (perms["max_abs_alpha_difference_after_logical_unpermute"], "<=", 1e-5),
        "permutation_gate_diff": (perms["max_abs_gate_difference_after_logical_unpermute"], "<=", 1e-5),
        "force_reader_retrieval": (force_reader[ROUTE_RETRIEVAL], ">=", 0.99),
        "force_reader_transform": (force_reader[ROUTE_TRANSFORM], "<=", 0.01),
        "force_reader_hard": (force_reader[ROUTE_HARD_NEGATIVE], "<=", 0.01),
        "force_transform_retrieval": (force_transform[ROUTE_RETRIEVAL], "<=", 0.01),
        "force_transform_transform": (force_transform[ROUTE_TRANSFORM], ">=", 0.99),
        "force_transform_hard": (force_transform[ROUTE_HARD_NEGATIVE], ">=", 0.99),
        "reader_disabled_retrieval": (reader_disabled[ROUTE_RETRIEVAL], ">=", 0.95),
        "reader_disabled_transform": (reader_disabled[ROUTE_TRANSFORM], "<=", 0.01),
        "reader_disabled_hard": (reader_disabled[ROUTE_HARD_NEGATIVE], "<=", 0.01),
        "transform_disabled_retrieval": (transform_disabled[ROUTE_RETRIEVAL], "<=", 0.01),
        "transform_disabled_transform": (transform_disabled[ROUTE_TRANSFORM], ">=", 0.95),
        "transform_disabled_hard": (transform_disabled[ROUTE_HARD_NEGATIVE], ">=", 0.95),
        "same_query_route_flip": (counter["route_flip_correct_rate"], ">=", 0.99),
        "same_query_sensitive": (counter["same_query_support_sensitive_route_rate"], ">=", 0.99),
    }
    gates = {}
    for name, (value, op, threshold) in defs.items():
        if op == "is":
            passed = value is threshold
        elif op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        gates[name] = {"value": value, "operator": op, "threshold": threshold, "passed": bool(passed)}
    failed = [name for name, row in gates.items() if not row["passed"]]
    return {"passed": not failed, "failed_gates": failed, "gates": gates}


def aggregate_and_verdict(seed_summaries: list[dict[str, Any]], transform_cert: dict[str, Any]) -> tuple[dict[str, Any], str]:
    failed = {str(row["seed"]): row["gates"]["failed_gates"] for row in seed_summaries if row["gates"]["failed_gates"]}
    aggregate = {
        "all_seed_gates_passed": not failed,
        "failed_gates_by_seed": failed,
        "mean_macro_exact": statistics.fmean(row["sealed"]["conditions"]["NORMAL"]["overall"]["macro_direct_grid_exact_accuracy"] for row in seed_summaries),
        "min_macro_exact": min(row["sealed"]["conditions"]["NORMAL"]["overall"]["macro_direct_grid_exact_accuracy"] for row in seed_summaries),
        "mean_route_accuracy": statistics.fmean(row["sealed"]["conditions"]["NORMAL"]["overall"]["route_accuracy"] for row in seed_summaries),
        "min_route_accuracy": min(row["sealed"]["conditions"]["NORMAL"]["overall"]["route_accuracy"] for row in seed_summaries),
        "mean_same_query_flip": statistics.fmean(row["sealed"]["same_query_counterfactual_report"]["route_flip_correct_rate"] for row in seed_summaries),
        "min_same_query_flip": min(row["sealed"]["same_query_counterfactual_report"]["route_flip_correct_rate"] for row in seed_summaries),
    }
    if not transform_cert["passed"]:
        return aggregate, "TRANSFORM_ORACLE_VALIDATION_FAILED"
    if not failed:
        return aggregate, "R3A_PASS"
    failed_names = {gate for gates in failed.values() for gate in gates}
    if "reader_certification" in failed_names:
        return aggregate, "READER_EXPERT_NOT_COMPETENT"
    if any(name in failed_names for name in ("transform_mean_gate", "hard_mean_gate")):
        return aggregate, "ALWAYS_READER_COLLAPSE"
    if "retrieval_mean_gate" in failed_names:
        return aggregate, "ALWAYS_TRANSFORM_COLLAPSE"
    if "hard_negative_rejection" in failed_names:
        return aggregate, "HARD_NEGATIVE_REJECTION_FAILURE"
    if any(name.startswith("same_query") for name in failed_names):
        return aggregate, "QUERY_SHORTCUT_FAILURE"
    if any(name.startswith("heldout") for name in failed_names):
        return aggregate, "HELDOUT_ROUTING_GENERALIZATION_FAILURE"
    if any(name.startswith("permutation") for name in failed_names):
        return aggregate, "PERMUTATION_LEAK_OR_REORDER_BUG"
    if any("disabled" in name or name.startswith("force_") for name in failed_names):
        return aggregate, "CAUSAL_ROUTE_ABLATION_FAILURE"
    return aggregate, "INCONCLUSIVE"


def prediction_detail(model: MixedRouteRouterR3A, episode: R3AEpisode, device: torch.device) -> dict[str, Any]:
    batch = batch_for([episode], device)
    with torch.no_grad():
        transform_probs = transform_oracle_probs(batch["query_grids"], transform_specs([episode]))
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], transform_probs)
    decoded = out["decoded_grid"][0].detach().cpu()
    target = batch["target_grids"][0].detach().cpu()
    gate = float(out["gate"][0].detach().cpu().item())
    predicted_route = TRUE_ROUTE_READER if gate >= 0.5 else TRUE_ROUTE_TRANSFORM
    return {
        "reader_alpha": [float(value) for value in out["reader_alpha"][0].detach().cpu().tolist()],
        "raw_scores": [float(value) for value in out["raw_scores"][0].detach().cpu().tolist()],
        "retrieval_logits": [float(value) for value in out["retrieval_logits"][0].detach().cpu().tolist()],
        "reader_output": tensor_grid_to_list(out["reader_decoded_grid"][0]),
        "transform_oracle_output": tensor_grid_to_list(out["transform_decoded_grid"][0]),
        "gate_value": gate,
        "final_output": tensor_grid_to_list(decoded),
        "predicted_route": predicted_route,
        "route_correct": predicted_route == episode.true_route,
        "final_exact": bool(torch.equal(decoded, target)),
        "all_route_conditions_passed": bool(torch.equal(decoded, target) and predicted_route == episode.true_route),
    }


def append_episode_example(lines: list[str], model: MixedRouteRouterR3A, episode: R3AEpisode, device: torch.device) -> None:
    detail = prediction_detail(model, episode, device)
    lines.append(f"## {episode.route_family} success")
    lines.append(f"episode_id: {episode.episode_id}")
    lines.append(f"route family: {episode.route_family}")
    lines.append(f"query_type: {query_type_key(episode.query_type)}")
    lines.append(f"predicted route: {detail['predicted_route']}")
    lines.append(f"true route for evaluation only: {episode.true_route}")
    lines.append(f"all route conditions passed: {detail['all_route_conditions_passed']}")
    lines.append("query grid: " + json.dumps(grid_to_list(episode.query)))
    for idx, pair in enumerate(episode.supports):
        lines.append(f"support {idx} input: " + json.dumps(grid_to_list(pair.input)))
        lines.append(f"support {idx} output: " + json.dumps(grid_to_list(pair.output)))
    lines.append("reader alpha: " + json.dumps(detail["reader_alpha"]))
    lines.append("reader raw scores: " + json.dumps(detail["raw_scores"]))
    lines.append("reader retrieval logits: " + json.dumps(detail["retrieval_logits"]))
    lines.append("reader output: " + json.dumps(detail["reader_output"]))
    lines.append("transform oracle output: " + json.dumps(detail["transform_oracle_output"]))
    lines.append(f"gate value: {detail['gate_value']:.12g}")
    lines.append("final output: " + json.dumps(detail["final_output"]))
    lines.append("target: " + json.dumps(grid_to_list(episode.target)))
    lines.append("")


def write_prediction_examples(path: Path, episodes: list[R3AEpisode], seed_summary: dict[str, Any], model: MixedRouteRouterR3A, device: torch.device) -> None:
    lines = ["# Mixed Router R3A Prediction Examples", ""]
    rows = seed_summary["sealed"]["same_query_counterfactual_report"]["examples"][:4]
    by_family = {family: [episode for episode in episodes if episode.route_family == family][:4] for family in ROUTE_FAMILIES}
    for family, items in by_family.items():
        for episode in items:
            append_episode_example(lines, model, episode, device)
    lines.append("## Same-query route-flip groups")
    for row in rows:
        group_items = [episode for episode in episodes if episode.counterfactual_group_id == row["counterfactual_group_id"]]
        lines.append(f"### {row['counterfactual_group_id']}")
        lines.append("summary: " + json.dumps(row, sort_keys=True))
        for episode in sorted(group_items, key=lambda item: ROUTE_FAMILIES.index(item.route_family)):
            detail = prediction_detail(model, episode, device)
            lines.append(
                json.dumps(
                    {
                        "route_family": episode.route_family,
                        "query_type": query_type_key(episode.query_type),
                        "gate_value": detail["gate_value"],
                        "reader_alpha": detail["reader_alpha"],
                        "raw_scores": detail["raw_scores"],
                        "reader_output": detail["reader_output"],
                        "transform_oracle_output": detail["transform_oracle_output"],
                        "final_output": detail["final_output"],
                        "target": grid_to_list(episode.target),
                        "predicted_route": detail["predicted_route"],
                        "true_route_for_evaluation_only": episode.true_route,
                        "all_route_conditions_passed": detail["all_route_conditions_passed"],
                    },
                    sort_keys=True,
                )
            )
    lines.append("")
    lines.append("## Held-out query-type examples")
    for row in seed_summary["sealed"]["heldout_query_type_report"]["rows"][:4]:
        lines.append(json.dumps(row, sort_keys=True))
    lines.append("")
    lines.append("## Failures")
    lines.append("No failures were observed for the selected seed summary." if not seed_summary["gates"]["failed_gates"] else json.dumps(seed_summary["gates"]["failed_gates"]))
    save_text(path, "\n".join(lines) + "\n")


def summary_markdown(seed_summaries: list[dict[str, Any]], aggregate: dict[str, Any], verdict: str) -> str:
    lines = ["# Mixed Router R3A", "", "|seed|macro exact|route acc|retrieval gate|transform gate|hard gate|same-query flip|", "|---|---|---|---|---|---|---|"]
    for row in seed_summaries:
        normal = row["sealed"]["conditions"]["NORMAL"]
        families = normal["families"]
        lines.append(
            f"|{row['seed']}|{normal['overall']['macro_direct_grid_exact_accuracy']:.6f}|{normal['overall']['route_accuracy']:.6f}|"
            f"{families[ROUTE_RETRIEVAL]['mean_gate']:.6f}|{families[ROUTE_TRANSFORM]['mean_gate']:.6f}|"
            f"{families[ROUTE_HARD_NEGATIVE]['mean_gate']:.6f}|{row['sealed']['same_query_counterfactual_report']['route_flip_correct_rate']:.6f}|"
        )
    lines.extend(
        [
            "",
            f"Aggregate min/mean macro exact: {aggregate['min_macro_exact']:.6f} / {aggregate['mean_macro_exact']:.6f}",
            f"Aggregate min/mean route accuracy: {aggregate['min_route_accuracy']:.6f} / {aggregate['mean_route_accuracy']:.6f}",
            f"Verdict: `{verdict}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_pytest(out: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    save_text(out / "pytest_result.txt", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
    return result.stdout.strip()


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
    sealed = load_split(args.data_out, "sealed")
    episodes_by_split = {"train": train, "dev": dev, "sealed": sealed}
    save_json(args.data_out / "route_balance_report.json", route_balance_report(episodes_by_split))
    save_json(args.data_out / "counterfactual_group_report.json", counterfactual_group_report(episodes_by_split))
    transform_cert = certify_transform_oracle(episodes_by_split)
    save_json(args.data_out / "transform_oracle_certification.json", transform_cert)
    if not transform_cert["passed"]:
        print("TRANSFORM_ORACLE_VALIDATION_FAILED", flush=True)
        raise SystemExit("TRANSFORM_ORACLE_VALIDATION_FAILED")
    run_config = {
        "experiment": "R3A - Oracle Mixed-Route Router Benchmark",
        "mode": args.mode,
        "seeds": args.seeds,
        "reader_epochs": args.reader_epochs,
        "router_epochs": args.router_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "data_out": str(args.data_out),
        "checkpoint_out": str(args.checkpoint_out),
    }
    save_json(args.data_out / "run_config.json", run_config)
    device = torch.device("cuda")
    reader_results = []
    for seed in args.seeds:
        result = train_reader_seed(seed, args, train, dev, sealed, device)
        reader_results.append(result)
        if not result["certification"]["passed"]:
            print("READER_EXPERT_NOT_COMPETENT", flush=True)
            raise SystemExit("READER_EXPERT_NOT_COMPETENT")
    router_results = []
    for seed in args.seeds:
        router_results.append(train_router_seed(seed, args, args.checkpoint_out / f"reader_r3a_seed_{seed}_best_dev.pt", train, dev, device))
    gradient_report = {str(row["seed"]): row["grad_report"] for row in router_results}
    coverage = {"all_seeds_passed": all(row["coverage"]["all_epochs_passed"] for row in router_results), "seeds": {str(row["seed"]): row["coverage"] for row in router_results}}
    save_json(args.data_out / "gradient_isolation_report.json", gradient_report)
    save_json(args.data_out / "epoch_coverage_report.json", coverage)
    if args.mode == "dry_run":
        dry = {
            "mode": "dry_run",
            "reader_results": [{k: v for k, v in row.items() if k != "model"} for row in reader_results],
            "router_results": router_results,
            "gradient_isolation": gradient_report,
            "coverage": coverage,
            "protected_hashes_unchanged": before == protected_hashes(),
        }
        save_json(args.data_out / "dry_run_report.json", dry)
        save_json(args.data_out / "source_protection_hashes_after.json", protected_hashes())
        print(json.dumps(to_jsonable(dry), indent=2, sort_keys=True), flush=True)
        return dry
    seed_summaries = []
    for router_result in router_results:
        seed = router_result["seed"]
        model = load_best_router(seed, args, device)
        train_metrics = evaluate_router_split(model, train, "train", device, args.batch_size)
        dev_metrics = evaluate_router_split(model, dev, "dev", device, args.batch_size)
        sealed_metrics = evaluate_router_split(model, sealed, "sealed", device, args.batch_size)
        save_json(args.data_out / f"router_final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(args.data_out / f"router_final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(args.data_out / f"router_final_sealed_metrics_seed_{seed}.json", sealed_metrics)
        save_json(args.data_out / f"exhaustive_permutation_report_seed_{seed}.json", sealed_metrics["conditions"]["ALL_PAIR_PERMUTATIONS"])
        save_json(args.data_out / f"reader_disabled_ablation_seed_{seed}.json", sealed_metrics["conditions"]["READER_DISABLED"])
        save_json(args.data_out / f"transform_disabled_ablation_seed_{seed}.json", sealed_metrics["conditions"]["TRANSFORM_DISABLED"])
        save_json(args.data_out / f"same_query_counterfactual_report_seed_{seed}.json", sealed_metrics["same_query_counterfactual_report"])
        save_json(args.data_out / f"heldout_query_type_report_seed_{seed}.json", sealed_metrics["heldout_query_type_report"])
        reader_cert = next(row["certification"] for row in reader_results if row["seed"] == seed)
        seed_summary = {
            "seed": seed,
            "reader_certification": reader_cert,
            "router_selection": router_result["selection"],
            "gradient_isolation": router_result["grad_report"],
            "train": train_metrics,
            "dev": dev_metrics,
            "sealed": sealed_metrics,
        }
        seed_summary["gates"] = seed_gates(seed_summary)
        seed_summaries.append(seed_summary)
    aggregate, verdict = aggregate_and_verdict(seed_summaries, transform_cert)
    save_json(args.data_out / "aggregate_seed_summary.json", {"aggregate": aggregate, "seeds": seed_summaries, "verdict": verdict})
    example_model = load_best_router(seed_summaries[0]["seed"], args, device)
    write_prediction_examples(args.data_out / "prediction_examples.txt", sealed, seed_summaries[0], example_model, device)
    pytest_result = run_pytest(args.data_out)
    after = protected_hashes()
    save_json(args.data_out / "source_protection_hashes_after.json", after)
    summary = {
        "verdict": verdict,
        "environment": {**env, "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()},
        "run_config": run_config,
        "dataset_manifest": generated["manifest"],
        "transform_oracle_certification": transform_cert,
        "coverage": coverage,
        "gradient_isolation": gradient_report,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
        "pytest_result": pytest_result,
        "source_protection_hashes_unchanged": before == after,
        "interpretation": {
            "evidence_sensitive": verdict == "R3A_PASS",
            "does_not_prove": ["ARC is solved", "a learned transform expert works", "test-time adaptation works"],
            "next": "R3B should replace the transform oracle with a learned transform expert.",
        },
    }
    save_json(args.data_out / "summary_report.json", summary)
    save_text(args.data_out / "summary.md", summary_markdown(seed_summaries, aggregate, verdict))
    print(json.dumps(to_jsonable({"verdict": verdict, "aggregate": aggregate, "source_protection_hashes_unchanged": before == after}), indent=2, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed Router R3A")
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=lambda text: [int(item) for item in text.split(",") if item], default=list(DEFAULT_SEEDS))
    parser.add_argument("--reader-epochs", type=int, default=10)
    parser.add_argument("--router-epochs", type=int, default=8)
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
