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

from vrm.data import ARC_COLOR_COUNT

from .mixed_router_r3b2_data import (
    CORE_FAMILIES,
    FAMILY_HARD_NEGATIVE,
    FAMILY_RETRIEVAL,
    FAMILY_TRANSFORM,
    TRUE_ROUTE_READER,
    TRUE_ROUTE_TRANSFORM,
    R3B2Episode,
    boundary_candidate_pool_manifest,
    counterfactual_group_manifest,
    family_balance_report,
    generate_compatibility_set,
    grid_to_list,
    load_boundary_pool,
    load_split,
    query_type_key,
    reorder_episode,
    save_dataset,
    stable_hash,
    support_signature,
    tensors_from_episodes,
)
from .mixed_router_r3b2_gate import (
    EVIDENCE_FEATURE_NAMES,
    MixedRouterR3B2,
    grad_norm,
    nll_per_episode,
    router_output_loss,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
DEFAULT_OUT = PROJECT_ROOT / "artifacts/mixed_router_r3b2_v1"
DEFAULT_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/mixed_router_r3b2_v1"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)

READER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/twin_pair_reader_r1b"
TRANSFORM_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints/transform_r3b0_v1"

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


def _is_new_output_path(path: Path) -> bool:
    resolved = path.resolve()
    return (
        resolved == DEFAULT_OUT
        or DEFAULT_OUT.resolve() in resolved.parents
        or resolved == DEFAULT_CHECKPOINT_OUT
        or DEFAULT_CHECKPOINT_OUT.resolve() in resolved.parents
    )


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
            if _is_new_output_path(path):
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


def configure_seed(seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {"seed": seed, "torch_deterministic_algorithms": "enabled_warn_only"}


def reader_checkpoint_for_seed(seed: int) -> Path:
    path = READER_CHECKPOINT_DIR / f"pair_reader_r1b_seed_{seed}_best_dev.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def transform_checkpoint_for_seed(seed: int) -> Path:
    path = TRANSFORM_CHECKPOINT_DIR / f"transform_r3b0_seed_{seed}_best_dev.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def batches(order: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def support_order(episode_id: str, seed: int, epoch: int) -> tuple[int, int, int]:
    perms = list(itertools.permutations(range(3)))
    index = int(hashlib.sha256(f"{seed}|{epoch}|{episode_id}".encode("utf-8")).hexdigest()[:8], 16) % len(perms)
    return tuple(int(value) for value in perms[index])


def training_view(episode: R3B2Episode, seed: int, epoch: int) -> R3B2Episode:
    return reorder_episode(episode, support_order(episode.episode_id, seed, epoch))


def save_checkpoint(
    path: Path,
    router: MixedRouterR3B2,
    optimizer: torch.optim.Optimizer,
    seed: int,
    epoch: int,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate_state_dict": router.gate.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "epoch_index": epoch,
            "global_step": step,
            "reader_checkpoint": str(router.reader_expert.checkpoint_path),
            "transform_checkpoint": str(router.transform_expert.checkpoint_path),
            "extra": extra or {},
        },
        path,
    )


def load_selected_router(seed: int, checkpoint_path: Path, device: torch.device) -> MixedRouterR3B2:
    router = MixedRouterR3B2(reader_checkpoint_for_seed(seed), transform_checkpoint_for_seed(seed), device).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    router.gate.load_state_dict(payload["gate_state_dict"])
    router.eval()
    router.freeze_experts()
    return router


def exact_mask(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (decoded == target).flatten(1).all(dim=1)


def per_episode_hamming(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (decoded != target).flatten(1).sum(dim=1).to(torch.float32)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    sorted_values = sorted(values)
    return {
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p10": sorted_values[int(0.1 * (len(sorted_values) - 1))],
        "p90": sorted_values[int(0.9 * (len(sorted_values) - 1))],
    }


def metric_row() -> dict[str, Any]:
    return {
        "count": 0,
        "soft_exact": 0,
        "hard_exact": 0,
        "soft_fg_correct": 0,
        "hard_fg_correct": 0,
        "fg_total": 0,
        "soft_all_correct": 0,
        "hard_all_correct": 0,
        "all_total": 0,
        "soft_hamming": 0.0,
        "hard_hamming": 0.0,
        "reader_exact": 0,
        "transform_exact": 0,
        "reader_route": 0,
        "transform_route": 0,
        "semantic_route_correct": 0,
        "semantic_route_total": 0,
        "route_utility_correct": 0,
        "route_utility_total": 0,
        "routing_regret": [],
        "gate": [],
        "reader_nll": [],
        "transform_nll": [],
        "selected_nll": [],
        "soft_hard_disagree": 0,
    }


def finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    count = max(row["count"], 1)
    gate_stats = quantiles(row["gate"])
    return {
        "episode_count": row["count"],
        "SOFT_MIX": {
            "direct_grid_exact_accuracy": row["soft_exact"] / count,
            "foreground_color_accuracy": row["soft_fg_correct"] / max(row["fg_total"], 1),
            "all_cell_color_accuracy": row["soft_all_correct"] / max(row["all_total"], 1),
            "target_rectangle_hamming_error": row["soft_hamming"] / count,
        },
        "HARD_ROUTE": {
            "direct_grid_exact_accuracy": row["hard_exact"] / count,
            "foreground_color_accuracy": row["hard_fg_correct"] / max(row["fg_total"], 1),
            "all_cell_color_accuracy": row["hard_all_correct"] / max(row["all_total"], 1),
            "target_rectangle_hamming_error": row["hard_hamming"] / count,
        },
        "reader_branch_exact": row["reader_exact"] / count,
        "transform_branch_exact": row["transform_exact"] / count,
        "hard_route_final_exact": row["hard_exact"] / count,
        "mean_gate": gate_stats["mean"],
        "median_gate": gate_stats["median"],
        "p10_gate": gate_stats["p10"],
        "p90_gate": gate_stats["p90"],
        "fraction_gate_ge_0_5": row["reader_route"] / count,
        "fraction_gate_ge_0_9": sum(1 for value in row["gate"] if value >= 0.9) / count,
        "fraction_gate_le_0_1": sum(1 for value in row["gate"] if value <= 0.1) / count,
        "route_accuracy_semantic_family": row["semantic_route_correct"] / max(row["semantic_route_total"], 1),
        "reader_route_recall": row["reader_route"] / count,
        "transform_route_recall": row["transform_route"] / count,
        "reader_nll": statistics.fmean(row["reader_nll"]) if row["reader_nll"] else 0.0,
        "transform_nll": statistics.fmean(row["transform_nll"]) if row["transform_nll"] else 0.0,
        "selected_branch_nll": statistics.fmean(row["selected_nll"]) if row["selected_nll"] else 0.0,
        "mean_routing_regret": statistics.fmean(row["routing_regret"]) if row["routing_regret"] else 0.0,
        "route_utility_agreement": row["route_utility_correct"] / max(row["route_utility_total"], 1),
        "soft_vs_hard_final_disagreement": row["soft_hard_disagree"] / count,
    }


def evaluate_router(
    router: MixedRouterR3B2,
    episodes: list[R3B2Episode],
    device: torch.device,
    batch_size: int,
    *,
    split: str,
    return_details: bool = False,
    reader_disabled: bool = False,
    transform_disabled: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {family: metric_row() for family in sorted({episode.route_family for episode in episodes})}
    overall = metric_row()
    details: list[dict[str, Any]] = []
    confusion = {family: {"reader": 0, "transform": 0} for family in rows}
    router.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_episodes = episodes[start : start + batch_size]
            batch = tensors_from_episodes(batch_episodes, device)
            outputs = router(
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                reader_disabled=reader_disabled,
                transform_disabled=transform_disabled,
            )
            target = batch["target_grids"]
            soft_decoded = outputs["soft_decoded_grid"]
            hard_decoded = outputs["hard_decoded_grid"]
            reader_decoded = outputs["reader_decoded_grid"]
            transform_decoded = outputs["transform_decoded_grid"]
            soft_exact = exact_mask(soft_decoded, target)
            hard_exact = exact_mask(hard_decoded, target)
            reader_exact = exact_mask(reader_decoded, target)
            transform_exact = exact_mask(transform_decoded, target)
            reader_nll = nll_per_episode(outputs["reader_probs"], target)
            transform_nll = nll_per_episode(outputs["transform_probs"], target)
            hard_reader = outputs["hard_route_reader"]
            selected_nll = torch.where(hard_reader, reader_nll, transform_nll)
            regret = selected_nll - torch.minimum(reader_nll, transform_nll)
            fg = target != 0
            all_total = target[0].numel()
            for index, episode in enumerate(batch_episodes):
                family = episode.route_family
                for row in (rows[family], overall):
                    row["count"] += 1
                    fg_total = int(fg[index].sum().item())
                    row["fg_total"] += fg_total
                    row["all_total"] += all_total
                    row["soft_exact"] += int(bool(soft_exact[index].item()))
                    row["hard_exact"] += int(bool(hard_exact[index].item()))
                    row["soft_fg_correct"] += int(((soft_decoded[index] == target[index]) & fg[index]).sum().item())
                    row["hard_fg_correct"] += int(((hard_decoded[index] == target[index]) & fg[index]).sum().item())
                    row["soft_all_correct"] += int((soft_decoded[index] == target[index]).sum().item())
                    row["hard_all_correct"] += int((hard_decoded[index] == target[index]).sum().item())
                    row["soft_hamming"] += float(per_episode_hamming(soft_decoded[index : index + 1], target[index : index + 1]).item())
                    row["hard_hamming"] += float(per_episode_hamming(hard_decoded[index : index + 1], target[index : index + 1]).item())
                    row["reader_exact"] += int(bool(reader_exact[index].item()))
                    row["transform_exact"] += int(bool(transform_exact[index].item()))
                    selected_reader = bool(hard_reader[index].item())
                    row["reader_route"] += int(selected_reader)
                    row["transform_route"] += int(not selected_reader)
                    if episode.true_route is not None:
                        row["semantic_route_total"] += 1
                        row["semantic_route_correct"] += int(
                            (episode.true_route == TRUE_ROUTE_READER and selected_reader)
                            or (episode.true_route == TRUE_ROUTE_TRANSFORM and not selected_reader)
                        )
                    rnll = float(reader_nll[index].item())
                    tnll = float(transform_nll[index].item())
                    snll = float(selected_nll[index].item())
                    reg = float(regret[index].item())
                    row["reader_nll"].append(rnll)
                    row["transform_nll"].append(tnll)
                    row["selected_nll"].append(snll)
                    row["routing_regret"].append(reg)
                    if abs(rnll - tnll) > 1e-8:
                        row["route_utility_total"] += 1
                        row["route_utility_correct"] += int((selected_reader and rnll < tnll) or ((not selected_reader) and tnll < rnll))
                    gate = float(outputs["gate"][index].item())
                    row["gate"].append(gate)
                    row["soft_hard_disagree"] += int(not torch.equal(soft_decoded[index], hard_decoded[index]))
                confusion[family]["reader" if bool(hard_reader[index].item()) else "transform"] += 1
                if return_details:
                    details.append(
                        {
                            "episode_id": episode.episode_id,
                            "split": split,
                            "family": family,
                            "query_type_key": query_type_key(episode.query_type),
                            "true_route": episode.true_route,
                            "gate": float(outputs["gate"][index].item()),
                            "selected_branch": "reader" if bool(hard_reader[index].item()) else "transform",
                            "reader_exact": bool(reader_exact[index].item()),
                            "transform_exact": bool(transform_exact[index].item()),
                            "soft_exact": bool(soft_exact[index].item()),
                            "hard_exact": bool(hard_exact[index].item()),
                            "reader_nll": float(reader_nll[index].item()),
                            "transform_nll": float(transform_nll[index].item()),
                            "selected_branch_nll": float(selected_nll[index].item()),
                            "routing_regret": float(regret[index].item()),
                            "route_utility_agreement": None if abs(float(reader_nll[index].item()) - float(transform_nll[index].item())) <= 1e-8 else (
                                (bool(hard_reader[index].item()) and float(reader_nll[index].item()) < float(transform_nll[index].item()))
                                or ((not bool(hard_reader[index].item())) and float(transform_nll[index].item()) < float(reader_nll[index].item()))
                            ),
                            "boundary_kind": episode.boundary_kind,
                            "boundary_hamming_level": episode.boundary_hamming_level,
                            "soft_hard_disagree": not torch.equal(soft_decoded[index], hard_decoded[index]),
                        }
                    )
    finalized = {
        "split": split,
        "families": {family: finalize_row(row) for family, row in rows.items()},
        "overall": finalize_row(overall),
        "route_confusion_matrix": confusion,
    }
    core_present = [family for family in CORE_FAMILIES if family in finalized["families"]]
    finalized["macro_hard_route_exact"] = statistics.fmean(
        finalized["families"][family]["HARD_ROUTE"]["direct_grid_exact_accuracy"] for family in core_present
    ) if core_present else finalized["overall"]["HARD_ROUTE"]["direct_grid_exact_accuracy"]
    finalized["macro_route_utility_agreement"] = statistics.fmean(
        finalized["families"][family]["route_utility_agreement"] for family in core_present
    ) if core_present else finalized["overall"]["route_utility_agreement"]
    return finalized, details


def compatibility_certification(
    seeds: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    episodes = generate_compatibility_set(1500)
    report: dict[str, Any] = {
        "example_count_per_family": 1500,
        "total_examples": len(episodes),
        "fresh_r3b2_generation_path": True,
        "seeds": {},
        "passed": True,
    }
    for seed in seeds:
        router = MixedRouterR3B2(reader_checkpoint_for_seed(seed), transform_checkpoint_for_seed(seed), device).to(device)
        metrics, details = evaluate_router(router, episodes, device, batch_size, split="compatibility", return_details=True)
        family = metrics["families"]
        seed_report = {
            "reader_exact_on_retrieval_clear": family[FAMILY_RETRIEVAL]["reader_branch_exact"],
            "transform_exact_on_retrieval_clear": family[FAMILY_RETRIEVAL]["transform_branch_exact"],
            "transform_exact_on_transform_clear": family[FAMILY_TRANSFORM]["transform_branch_exact"],
            "reader_exact_on_transform_clear": family[FAMILY_TRANSFORM]["reader_branch_exact"],
            "transform_exact_on_transform_hard_negative": family[FAMILY_HARD_NEGATIVE]["transform_branch_exact"],
            "reader_exact_on_transform_hard_negative": family[FAMILY_HARD_NEGATIVE]["reader_branch_exact"],
            "per_example_branch_nll_recorded": all("reader_nll" in detail and "transform_nll" in detail for detail in details),
        }
        seed_report["passed"] = (
            seed_report["reader_exact_on_retrieval_clear"] >= 0.99
            and seed_report["transform_exact_on_retrieval_clear"] <= 0.01
            and seed_report["transform_exact_on_transform_clear"] >= 0.99
            and seed_report["reader_exact_on_transform_clear"] <= 0.05
            and seed_report["transform_exact_on_transform_hard_negative"] >= 0.99
            and seed_report["reader_exact_on_transform_hard_negative"] <= 0.10
        )
        report["seeds"][seed] = seed_report
        report["passed"] = bool(report["passed"] and seed_report["passed"])
    return report


def gradient_isolation_report(router: MixedRouterR3B2, episodes: list[R3B2Episode], device: torch.device) -> dict[str, Any]:
    router.train()
    batch = tensors_from_episodes(episodes[:128], device)
    optimizer = torch.optim.AdamW(router.gate_parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    optimizer.zero_grad(set_to_none=True)
    outputs = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss = router_output_loss(outputs["soft_probs"], batch["target_grids"])
    loss.backward()
    gate_norm = grad_norm(router.gate_parameters())
    expert_norm = grad_norm(router.expert_parameters())
    report = {
        "loss": float(loss.detach().item()),
        "gate_grad_norm": gate_norm,
        "expert_grad_norm": expert_norm,
        "reader_parameters_require_grad_false": all(not parameter.requires_grad for parameter in router.reader_expert.parameters()),
        "transform_parameters_require_grad_false": all(not parameter.requires_grad for parameter in router.transform_expert.parameters()),
        "gate_grad_nonzero": gate_norm > 0.0,
        "expert_grad_zero": expert_norm == 0.0,
    }
    optimizer.zero_grad(set_to_none=True)
    router.eval()
    return report


def train_one_seed(
    seed: int,
    train_episodes: list[R3B2Episode],
    dev_episodes: list[R3B2Episode],
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_out: Path,
    data_out: Path,
) -> dict[str, Any]:
    configure_seed(seed)
    router = MixedRouterR3B2(reader_checkpoint_for_seed(seed), transform_checkpoint_for_seed(seed), device).to(device)
    optimizer = torch.optim.AdamW(router.gate_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    initial_path = checkpoint_out / f"mixed_router_r3b2_seed_{seed}_initial.pt"
    save_checkpoint(initial_path, router, optimizer, seed, -1, 0, {"stage": "initial"})
    training_log = data_out / f"training_log_seed_{seed}.jsonl"
    if training_log.exists():
        training_log.unlink()
    dev_by_epoch: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    global_step = 0
    epochs = 1 if args.mode == "dry_run" else args.epochs
    max_batches = args.max_batches if args.mode == "dry_run" else None
    for epoch in range(epochs):
        router.train()
        order = epoch_order(len(train_episodes), seed, epoch)
        observed_ids: list[str] = []
        loss_values: list[float] = []
        for batch_index, batch_indices in enumerate(batches(order, args.batch_size)):
            if max_batches is not None and batch_index >= max_batches:
                break
            source = [train_episodes[index] for index in batch_indices]
            view = [training_view(episode, seed, epoch) for episode in source]
            observed_ids.extend(episode.episode_id for episode in source)
            batch = tensors_from_episodes(view, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            loss = router_output_loss(outputs["soft_probs"], batch["target_grids"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.gate_parameters(), 1.0)
            optimizer.step()
            global_step += 1
            loss_values.append(float(loss.detach().item()))
        unique = len(set(observed_ids))
        coverage = {
            "seed": seed,
            "epoch_index": epoch,
            "observed": len(observed_ids),
            "unique": unique,
            "duplicates": len(observed_ids) - unique,
            "missing": len(train_episodes) - unique if max_batches is None else "dry_run_partial",
            "full_coverage": max_batches is None and len(observed_ids) == len(train_episodes) and unique == len(train_episodes),
        }
        coverage_rows.append(coverage)
        router.eval()
        dev_metrics, _ = evaluate_router(router, dev_episodes, device, args.batch_size, split="dev")
        dev_row = {
            "seed": seed,
            "epoch_index": epoch,
            "global_step": global_step,
            "mean_loss": statistics.fmean(loss_values) if loss_values else 0.0,
            "coverage": coverage,
            "dev_metrics": dev_metrics,
        }
        dev_by_epoch.append(dev_row)
        append_jsonl(training_log, dev_row)
        save_checkpoint(checkpoint_out / f"mixed_router_r3b2_seed_{seed}_epoch_{epoch:02d}.pt", router, optimizer, seed, epoch, global_step, {"dev_metrics": dev_metrics})
    save_json(data_out / f"dev_metrics_by_epoch_seed_{seed}.json", dev_by_epoch)
    if args.mode == "dry_run":
        best = dev_by_epoch[-1]
    else:
        def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
            metrics = row["dev_metrics"]
            hard_negative = metrics["families"][FAMILY_HARD_NEGATIVE]
            return (
                metrics["macro_hard_route_exact"],
                metrics["macro_route_utility_agreement"],
                hard_negative["transform_route_recall"],
                -metrics["overall"]["mean_routing_regret"],
                -row["epoch_index"],
            )
        best = max(dev_by_epoch, key=selection_key)
    best_path = checkpoint_out / f"mixed_router_r3b2_seed_{seed}_best_dev.pt"
    payload = torch.load(checkpoint_out / f"mixed_router_r3b2_seed_{seed}_epoch_{best['epoch_index']:02d}.pt", map_location=device, weights_only=False)
    torch.save(payload, best_path)
    selection = {
        "seed": seed,
        "selected_epoch": best["epoch_index"],
        "selected_global_step": best["global_step"],
        "best_checkpoint": str(best_path),
        "selection_order": [
            "highest dev HARD_ROUTE macro exact across core families",
            "highest dev HARD_ROUTE route-utility agreement",
            "highest dev hard-negative transform-route recall",
            "lowest dev HARD_ROUTE routing regret",
            "earliest epoch",
        ],
        "selected_dev_metrics": best["dev_metrics"],
    }
    save_json(data_out / f"best_checkpoint_selection_seed_{seed}.json", selection)
    return {
        "seed": seed,
        "initial_checkpoint": str(initial_path),
        "best_checkpoint": str(best_path),
        "selection": selection,
        "coverage": coverage_rows,
        "dev_by_epoch": dev_by_epoch,
    }


def endpoint_equivalence(router: MixedRouterR3B2, episodes: list[R3B2Episode], device: torch.device, batch_size: int) -> dict[str, Any]:
    max_reader = 0.0
    max_transform = 0.0
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch = tensors_from_episodes(episodes[start : start + batch_size], device)
            reader_forced = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], force_gate=1.0)
            transform_forced = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], force_gate=0.0)
            max_reader = max(max_reader, float((reader_forced["soft_probs"] - reader_forced["reader_probs"]).abs().max().item()))
            max_transform = max(max_transform, float((transform_forced["soft_probs"] - transform_forced["transform_probs"]).abs().max().item()))
    return {
        "force_g_1_reader_max_abs_difference": max_reader,
        "force_g_0_transform_max_abs_difference": max_transform,
        "passed": max_reader <= 1e-6 and max_transform <= 1e-6,
    }


def disabled_ablation(
    router: MixedRouterR3B2,
    episodes: list[R3B2Episode],
    device: torch.device,
    batch_size: int,
    *,
    disable: str,
) -> dict[str, Any]:
    rows = {family: {"count": 0, "grid_change": 0} for family in CORE_FAMILIES}
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            batch_episodes = episodes[start : start + batch_size]
            batch = tensors_from_episodes(batch_episodes, device)
            normal = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            disabled = router(
                batch["query_grids"],
                batch["support_input_grids"],
                batch["support_output_grids"],
                reader_disabled=disable == "reader",
                transform_disabled=disable == "transform",
            )
            same_gate = float((normal["gate"] - disabled["gate"]).abs().max().item())
            for index, episode in enumerate(batch_episodes):
                row = rows[episode.route_family]
                row["count"] += 1
                row["grid_change"] += int(not torch.equal(normal["hard_decoded_grid"][index], disabled["hard_decoded_grid"][index]))
                row["max_gate_diff"] = max(row.get("max_gate_diff", 0.0), same_gate)
    families = {
        family: {
            "episode_count": row["count"],
            "normal_to_disabled_hard_route_grid_change_rate": row["grid_change"] / max(row["count"], 1),
            "max_gate_difference": row.get("max_gate_diff", 0.0),
        }
        for family, row in rows.items()
    }
    return {"disabled_branch": disable, "families": families}


def exhaustive_permutation_report(
    router: MixedRouterR3B2,
    episodes: list[R3B2Episode],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    same_grid = 0
    same_route = 0
    same_pair = 0
    total = 0
    max_prob = 0.0
    max_gate = 0.0
    max_alpha = 0.0
    max_context = 0.0
    max_pointer = 0.0
    perms = list(itertools.permutations(range(3)))
    baseline: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            base_batch = tensors_from_episodes(episodes[start : start + batch_size], device)
            base = router(base_batch["query_grids"], base_batch["support_input_grids"], base_batch["support_output_grids"])
            baseline.append(
                {
                    "hard_decoded_grid": base["hard_decoded_grid"].detach().cpu(),
                    "hard_route_reader": base["hard_route_reader"].detach().cpu(),
                    "gate": base["gate"].detach().cpu(),
                    "reader_alpha": base["reader_alpha"].detach().cpu(),
                    "soft_probs": base["soft_probs"].detach().cpu(),
                    "transform_task_context": base["transform_task_context"].detach().cpu(),
                    "transform_pointer_attention": base["transform_pointer_attention"].detach().cpu(),
                }
            )
        for order in perms:
            for chunk_index, start in enumerate(range(0, len(episodes), batch_size)):
                source = episodes[start : start + batch_size]
                views = [reorder_episode(episode, order) for episode in source]
                batch = tensors_from_episodes(views, device)
                out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                base = baseline[chunk_index]
                count = len(source)
                total += count
                out_hard = out["hard_decoded_grid"].detach().cpu()
                out_route = out["hard_route_reader"].detach().cpu()
                out_gate = out["gate"].detach().cpu()
                out_alpha = out["reader_alpha"].detach().cpu()
                out_soft = out["soft_probs"].detach().cpu()
                out_context = out["transform_task_context"].detach().cpu()
                out_pointer = out["transform_pointer_attention"].detach().cpu()
                same_grid += int((out_hard == base["hard_decoded_grid"]).flatten(1).all(dim=1).sum().item())
                same_route += int((out_route == base["hard_route_reader"]).sum().item())
                base_pair = base["reader_alpha"].argmax(dim=-1)
                out_pair = out_alpha.argmax(dim=-1)
                logical_pair = torch.tensor([order[int(value)] for value in out_pair.tolist()], dtype=torch.long)
                same_pair += int((base_pair == logical_pair).sum().item())
                restored_alpha = torch.zeros_like(out_alpha)
                for new_index, old_index in enumerate(order):
                    restored_alpha[:, old_index] = out_alpha[:, new_index]
                max_prob = max(max_prob, float((base["soft_probs"] - out_soft).abs().max().item()))
                max_gate = max(max_gate, float((base["gate"] - out_gate).abs().max().item()))
                max_alpha = max(max_alpha, float((base["reader_alpha"] - restored_alpha).abs().max().item()))
                max_context = max(max_context, float((base["transform_task_context"] - out_context).abs().max().item()))
                max_pointer = max(max_pointer, float((base["transform_pointer_attention"] - out_pointer).abs().max().item()))
    return {
        "presentations": total,
        "same_hard_route_final_grid_rate": same_grid / max(total, 1),
        "same_hard_route_rate": same_route / max(total, 1),
        "same_reader_logical_selected_pair_rate": same_pair / max(total, 1),
        "max_probability_difference": max_prob,
        "max_gate_difference": max_gate,
        "max_reader_alpha_difference": max_alpha,
        "max_transform_context_difference": max_context,
        "max_transform_pointer_difference": max_pointer,
        "passed": (
            same_grid / max(total, 1) >= 0.999
            and same_route / max(total, 1) >= 0.999
            and max(max_prob, max_gate, max_alpha, max_context, max_pointer) <= 1e-5
        ),
    }


def same_query_counterfactual_report(details: list[dict[str, Any]], episodes: list[R3B2Episode]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for detail in details:
        group_id = episode_by_id[detail["episode_id"]].counterfactual_group_id
        by_group.setdefault(group_id, []).append(detail)
    correct = 0
    sensitive = 0
    output_switch = 0
    total = 0
    gate_drops: list[float] = []
    examples: list[dict[str, Any]] = []
    for group_id, rows in by_group.items():
        families = {row["family"]: row for row in rows}
        if not all(family in families for family in CORE_FAMILIES):
            continue
        total += 1
        retrieval = families[FAMILY_RETRIEVAL]
        transform = families[FAMILY_TRANSFORM]
        hard = families[FAMILY_HARD_NEGATIVE]
        pattern_correct = (
            retrieval["selected_branch"] == "reader"
            and transform["selected_branch"] == "transform"
            and hard["selected_branch"] == "transform"
        )
        correct += int(pattern_correct)
        route_set = {row["selected_branch"] for row in families.values()}
        sensitive += int(len(route_set) > 1)
        output_switch += int(all(row["hard_exact"] for row in families.values()))
        gate_drops.append(float(retrieval["gate"]) - statistics.fmean([float(transform["gate"]), float(hard["gate"])]))
        if len(examples) < 12:
            examples.append(
                {
                    "group_id": group_id,
                    "gates": {family: families[family]["gate"] for family in CORE_FAMILIES},
                    "routes": {family: families[family]["selected_branch"] for family in CORE_FAMILIES},
                }
            )
    return {
        "group_count": total,
        "route_flip_correctness": correct / max(total, 1),
        "same_query_support_sensitive_route_rate": sensitive / max(total, 1),
        "output_switch_correctness": output_switch / max(total, 1),
        "mean_retrieval_to_transform_gate_drop": statistics.fmean(gate_drops) if gate_drops else 0.0,
        "examples": examples,
        "passed": correct / max(total, 1) >= 0.99 and sensitive / max(total, 1) >= 0.99,
    }


def heldout_query_type_report(details: list[dict[str, Any]], episodes: list[R3B2Episode]) -> dict[str, Any]:
    by_episode = {episode.episode_id: episode for episode in episodes}
    rows: dict[str, dict[str, Any]] = {}
    for detail in details:
        episode = by_episode[detail["episode_id"]]
        key = query_type_key(episode.query_type)
        row = rows.setdefault(key, {"count": 0, "hard_exact": 0, "utility_correct": 0, "utility_total": 0, "families": set(), "hard_negative_transform": 0, "hard_negative_count": 0})
        row["count"] += 1
        row["hard_exact"] += int(detail["hard_exact"])
        row["families"].add(detail["family"])
        if detail["route_utility_agreement"] is not None:
            row["utility_total"] += 1
            row["utility_correct"] += int(detail["route_utility_agreement"])
        if detail["family"] == FAMILY_HARD_NEGATIVE:
            row["hard_negative_count"] += 1
            row["hard_negative_transform"] += int(detail["selected_branch"] == "transform")
    finalized = {}
    for key, row in rows.items():
        finalized[key] = {
            "episode_count": row["count"],
            "families": sorted(row["families"]),
            "all_core_families_represented": set(row["families"]) == set(CORE_FAMILIES),
            "at_least_30_episodes": row["count"] >= 30,
            "hard_route_exact": row["hard_exact"] / max(row["count"], 1),
            "route_utility_agreement": row["utility_correct"] / max(row["utility_total"], 1),
            "hard_negative_transform_route_recall": row["hard_negative_transform"] / max(row["hard_negative_count"], 1),
        }
    return {
        "query_type_count": len(finalized),
        "query_types": finalized,
        "worst_hard_route_exact": min(row["hard_route_exact"] for row in finalized.values()),
        "worst_route_utility_agreement": min(row["route_utility_agreement"] for row in finalized.values()),
        "all_query_types_have_at_least_30_and_all_families": all(row["at_least_30_episodes"] and row["all_core_families_represented"] for row in finalized.values()),
        "macro_hard_route_exact": statistics.fmean(row["hard_route_exact"] for row in finalized.values()),
        "macro_route_utility_agreement": statistics.fmean(row["route_utility_agreement"] for row in finalized.values()),
        "hard_negative_transform_route_recall": statistics.fmean(row["hard_negative_transform_route_recall"] for row in finalized.values()),
    }


def boundary_partition(details: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    one_good = [row for row in details if bool(row["reader_exact"]) ^ bool(row["transform_exact"])]
    gaps = sorted(abs(float(row["reader_nll"]) - float(row["transform_nll"])) for row in one_good if abs(float(row["reader_nll"]) - float(row["transform_nll"])) > 1e-8)
    q1 = gaps[max(0, int(0.25 * (len(gaps) - 1)))] if gaps else 0.0
    partitions: dict[str, list[dict[str, Any]]] = {
        "CLEAR_READER": [],
        "CLEAR_TRANSFORM": [],
        "BOUNDARY_ONE_GOOD": [],
        "BOTH_GOOD": [],
        "BOTH_BAD": [],
    }
    for row in details:
        reader_exact = bool(row["reader_exact"])
        transform_exact = bool(row["transform_exact"])
        gap = abs(float(row["reader_nll"]) - float(row["transform_nll"]))
        if reader_exact and transform_exact:
            partitions["BOTH_GOOD"].append(row)
        elif not reader_exact and not transform_exact:
            partitions["BOTH_BAD"].append(row)
        elif gap <= q1 and gap > 1e-8:
            partitions["BOUNDARY_ONE_GOOD"].append(row)
        elif reader_exact:
            partitions["CLEAR_READER"].append(row)
        else:
            partitions["CLEAR_TRANSFORM"].append(row)
    return partitions


def summarize_partition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    utility_rows = [row for row in rows if row["route_utility_agreement"] is not None]
    selected_reader = sum(1 for row in rows if row["selected_branch"] == "reader")
    regrets = [float(row["routing_regret"]) for row in rows]
    random_regrets = []
    for row in rows:
        random_regrets.append(0.5 * (float(row["reader_nll"]) + float(row["transform_nll"])) - min(float(row["reader_nll"]), float(row["transform_nll"])))
    return {
        "count": count,
        "hard_route_final_exact": sum(1 for row in rows if row["hard_exact"]) / max(count, 1),
        "route_utility_agreement": sum(1 for row in utility_rows if row["route_utility_agreement"]) / max(len(utility_rows), 1),
        "mean_routing_regret": statistics.fmean(regrets) if regrets else 0.0,
        "random_50_50_branch_selection_baseline_regret": statistics.fmean(random_regrets) if random_regrets else 0.0,
        "mean_gate": statistics.fmean(float(row["gate"]) for row in rows) if rows else 0.0,
        "gate_distribution": quantiles([float(row["gate"]) for row in rows]),
        "soft_vs_hard_final_disagreement": sum(1 for row in rows if row["soft_hard_disagree"]) / max(count, 1),
        "selected_branch_frequencies": {
            "reader": selected_reader,
            "transform": count - selected_reader,
        },
    }


def boundary_audit_report(details: list[dict[str, Any]]) -> dict[str, Any]:
    partitions = boundary_partition(details)
    report = {name: summarize_partition(rows) for name, rows in partitions.items()}
    boundary = report["BOUNDARY_ONE_GOOD"]
    if boundary["count"] >= 300:
        status = "CONFIRMED" if (
            boundary["route_utility_agreement"] >= 0.95
            and boundary["mean_routing_regret"] < boundary["random_50_50_branch_selection_baseline_regret"]
        ) else "FAILED"
    else:
        status = "INSUFFICIENT_BOUNDARY_CASES"
    return {
        "partition_reports": report,
        "boundary_audit_status": status,
        "partition_uses_branch_outcomes_after_frozen_inference_only": True,
        "partitions_contribute_to_training": False,
    }


def write_prediction_examples(
    path: Path,
    router: MixedRouterR3B2,
    sealed: list[R3B2Episode],
    boundary: list[R3B2Episode],
    boundary_report: dict[str, Any],
    device: torch.device,
) -> None:
    selected: list[tuple[str, R3B2Episode]] = []
    for family, limit in ((FAMILY_RETRIEVAL, 4), (FAMILY_TRANSFORM, 4), (FAMILY_HARD_NEGATIVE, 4)):
        selected.extend((family, episode) for episode in [row for row in sealed if row.route_family == family][:limit])
    groups: dict[str, list[R3B2Episode]] = {}
    for episode in sealed:
        groups.setdefault(episode.counterfactual_group_id, []).append(episode)
    selected.extend(("same_query_group", episode) for rows in list(groups.values())[:4] for episode in rows)
    selected.extend(("heldout_query_type", episode) for episode in sealed[:4])
    boundary_partitions = boundary_partition(evaluate_router(router, boundary, device, 256, split="boundary", return_details=True)[1])
    for partition, limit in (("BOUNDARY_ONE_GOOD", 4), ("BOTH_GOOD", 2), ("BOTH_BAD", 2)):
        ids = {row["episode_id"] for row in boundary_partitions.get(partition, [])[:limit]}
        selected.extend((partition, episode) for episode in boundary if episode.episode_id in ids)
    lines: list[str] = []
    failures_added = 0
    for label, episode in selected:
        batch = tensors_from_episodes([episode], device)
        with torch.no_grad():
            out = router(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            rnll = float(nll_per_episode(out["reader_probs"], batch["target_grids"])[0].item())
            tnll = float(nll_per_episode(out["transform_probs"], batch["target_grids"])[0].item())
            selected_nll = rnll if bool(out["hard_route_reader"][0].item()) else tnll
        route = "reader" if bool(out["hard_route_reader"][0].item()) else "transform"
        record = {
            "label": label,
            "evaluation_only_family": episode.route_family,
            "evaluation_only_boundary_kind": episode.boundary_kind,
            "query_grid": grid_to_list(episode.query),
            "support_pairs": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in episode.supports],
            "reader_alpha": out["reader_alpha"][0].detach().cpu().tolist(),
            "reader_evidence_summary": {
                "min_q_to_key_distance": float(out["reader_qk_distances"][0].min().item()),
                "mean_q_to_key_distance": float(out["reader_qk_distances"][0].float().mean().item()),
            },
            "transform_context_pointer_summary": {
                "context_norm": float(out["transform_task_context"][0].norm().item()),
                "mean_max_pointer_probability": float(out["transform_pointer_attention"][0].max(dim=-1).values.mean().item()),
            },
            "reader_output": out["reader_decoded_grid"][0].detach().cpu().tolist(),
            "transform_output": out["transform_decoded_grid"][0].detach().cpu().tolist(),
            "soft_mixture_output": out["soft_decoded_grid"][0].detach().cpu().tolist(),
            "hard_route_output": out["hard_decoded_grid"][0].detach().cpu().tolist(),
            "target": grid_to_list(episode.target),
            "reader_nll": rnll,
            "transform_nll": tnll,
            "selected_branch": route,
            "gate_value": float(out["gate"][0].item()),
            "routing_regret": selected_nll - min(rnll, tnll),
        }
        exact = torch.equal(out["hard_decoded_grid"][0], batch["target_grids"][0])
        if not exact and failures_added >= 8:
            continue
        failures_added += int(not exact)
        lines.append(json.dumps(to_jsonable(record), sort_keys=True))
    save_text(path, "\n\n".join(lines) + "\n")


def run_pytest(data_out: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    save_text(data_out / "pytest_result.txt", result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:]}


def failed_gates_for_seed(
    sealed_metrics: dict[str, Any],
    heldout: dict[str, Any],
    endpoint: dict[str, Any],
    permutation: dict[str, Any],
    reader_disabled: dict[str, Any],
    transform_disabled: dict[str, Any],
    same_query: dict[str, Any],
) -> list[str]:
    failed: list[str] = []
    families = sealed_metrics["families"]
    if sealed_metrics["macro_hard_route_exact"] < 0.99:
        failed.append("macro_hard_route_exact")
    retrieval = families[FAMILY_RETRIEVAL]
    transform = families[FAMILY_TRANSFORM]
    hard = families[FAMILY_HARD_NEGATIVE]
    if retrieval["HARD_ROUTE"]["direct_grid_exact_accuracy"] < 0.99 or retrieval["reader_route_recall"] < 0.99 or retrieval["mean_gate"] < 0.95:
        failed.append("retrieval_core")
    if transform["HARD_ROUTE"]["direct_grid_exact_accuracy"] < 0.99 or transform["transform_route_recall"] < 0.99 or transform["mean_gate"] > 0.05:
        failed.append("transform_core")
    if hard["HARD_ROUTE"]["direct_grid_exact_accuracy"] < 0.99 or hard["transform_route_recall"] < 0.99 or hard["mean_gate"] > 0.05:
        failed.append("hard_negative_core")
    if sealed_metrics["overall"]["route_utility_agreement"] < 0.99:
        failed.append("route_utility_agreement")
    if sealed_metrics["overall"]["mean_routing_regret"] > 1e-4:
        failed.append("routing_regret")
    if heldout["macro_hard_route_exact"] < 0.99 or heldout["macro_route_utility_agreement"] < 0.99 or heldout["hard_negative_transform_route_recall"] < 0.99:
        failed.append("heldout_query_types")
    if not endpoint["passed"]:
        failed.append("endpoint_equivalence")
    if not permutation["passed"]:
        failed.append("pair_permutation")
    reader_disabled_families = reader_disabled["families"]
    transform_disabled_families = transform_disabled["families"]
    if reader_disabled_families[FAMILY_RETRIEVAL]["normal_to_disabled_hard_route_grid_change_rate"] < 0.95:
        failed.append("reader_disabled_retrieval")
    if reader_disabled_families[FAMILY_TRANSFORM]["normal_to_disabled_hard_route_grid_change_rate"] > 0.01:
        failed.append("reader_disabled_transform")
    if reader_disabled_families[FAMILY_HARD_NEGATIVE]["normal_to_disabled_hard_route_grid_change_rate"] > 0.01:
        failed.append("reader_disabled_hard_negative")
    if transform_disabled_families[FAMILY_RETRIEVAL]["normal_to_disabled_hard_route_grid_change_rate"] > 0.01:
        failed.append("transform_disabled_retrieval")
    if transform_disabled_families[FAMILY_TRANSFORM]["normal_to_disabled_hard_route_grid_change_rate"] < 0.95:
        failed.append("transform_disabled_transform")
    if transform_disabled_families[FAMILY_HARD_NEGATIVE]["normal_to_disabled_hard_route_grid_change_rate"] < 0.95:
        failed.append("transform_disabled_hard_negative")
    if not same_query["passed"]:
        failed.append("same_query_counterfactual")
    return failed


def verdict_from_results(aggregate: dict[str, Any]) -> str:
    if not aggregate["compatibility_passed"]:
        return "EXPERT_COMPATIBILITY_FAILURE"
    failed = aggregate["failed_gates_by_seed"]
    if not any(failed.values()):
        return "R3B2_PASS_BOUNDARY_CONFIRMED" if aggregate["boundary_status"] == "CONFIRMED" else "R3B2_PASS_BOUNDARY_INCONCLUSIVE"
    flattened = {item for rows in failed.values() for item in rows}
    if any(item.startswith("retrieval_core") for item in flattened):
        return "ALWAYS_TRANSFORM_COLLAPSE"
    if any(item.startswith("transform_core") for item in flattened):
        return "ALWAYS_READER_COLLAPSE"
    if any(item.startswith("hard_negative_core") for item in flattened):
        return "HARD_NEGATIVE_ROUTING_FAILURE"
    if "heldout_query_types" in flattened:
        return "HELDOUT_ROUTING_GENERALIZATION_FAILURE"
    if "pair_permutation" in flattened:
        return "PAIR_ORDER_LEAK_OR_REORDER_BUG"
    if any(item.startswith("reader_disabled") or item.startswith("transform_disabled") or item == "endpoint_equivalence" for item in flattened):
        return "CAUSAL_ROUTE_ABLATION_FAILURE"
    if "same_query_counterfactual" in flattened:
        return "QUERY_SHORTCUT_FAILURE"
    return "INCONCLUSIVE"


def write_summary(data_out: Path, aggregate: dict[str, Any], final_metrics: dict[int, dict[str, Any]]) -> None:
    lines = ["# Mixed Router R3B2", ""]
    lines.append("|seed|sealed macro hard exact|retrieval gate|transform gate|hard-neg gate|route utility|boundary status|failed gates|")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for seed, rows in final_metrics.items():
        sealed = rows["sealed_metrics"]
        lines.append(
            "|{seed}|{macro:.6f}|{rg:.6f}|{tg:.6f}|{hg:.6f}|{util:.6f}|{boundary}|{failed}|".format(
                seed=seed,
                macro=sealed["macro_hard_route_exact"],
                rg=sealed["families"][FAMILY_RETRIEVAL]["mean_gate"],
                tg=sealed["families"][FAMILY_TRANSFORM]["mean_gate"],
                hg=sealed["families"][FAMILY_HARD_NEGATIVE]["mean_gate"],
                util=sealed["overall"]["route_utility_agreement"],
                boundary=rows["boundary_audit"]["boundary_audit_status"],
                failed=",".join(rows["failed_gates"]) or "none",
            )
        )
    lines.extend(["", f"Verdict: `{aggregate['verdict']}`", ""])
    save_text(data_out / "summary.md", "\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=8)
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

    before = protected_hashes()
    save_json(data_out / "source_protection_hashes_before.json", before)
    environment = cuda_environment_report()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    save_json(data_out / "environment_report.json", environment)
    save_json(
        data_out / "run_config.json",
        {
            "mode": args.mode,
            "seeds": seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "data_out": str(data_out),
            "checkpoint_out": str(checkpoint_out),
            "reader_checkpoint_dir": str(READER_CHECKPOINT_DIR),
            "transform_checkpoint_dir": str(TRANSFORM_CHECKPOINT_DIR),
            "evidence_feature_names": EVIDENCE_FEATURE_NAMES,
            "cuda_required": True,
            "cpu_fallback": False,
        },
    )

    dataset = save_dataset(data_out)
    train_episodes = dataset["episodes_by_split"]["train"]
    dev_episodes = dataset["episodes_by_split"]["dev"]
    sealed_episodes = dataset["episodes_by_split"]["sealed"]
    boundary_pool = dataset["boundary_pool"]

    compatibility = compatibility_certification(seeds, device, args.batch_size)
    save_json(data_out / "compatibility_certification.json", compatibility)
    if not compatibility["passed"]:
        after = protected_hashes()
        save_json(data_out / "source_protection_hashes_after.json", after)
        aggregate = {
            "compatibility_passed": False,
            "hash_status": hash_status(before, after),
            "verdict": "EXPERT_COMPATIBILITY_FAILURE",
        }
        save_json(data_out / "aggregate_seed_summary.json", aggregate)
        print("EXPERT_COMPATIBILITY_FAILURE", flush=True)
        return 2

    init_reports: dict[int, Any] = {}
    gradient_report: dict[int, Any] = {}
    for seed in seeds:
        router = MixedRouterR3B2(reader_checkpoint_for_seed(seed), transform_checkpoint_for_seed(seed), device).to(device)
        sample = tensors_from_episodes(train_episodes[: args.batch_size], device)
        with torch.no_grad():
            sample_gate = router(sample["query_grids"], sample["support_input_grids"], sample["support_output_grids"])["gate"]
        init_reports[seed] = router.initialization_report(sample_gate)
        gradient_report[seed] = gradient_isolation_report(router, train_episodes, device)
    save_json(data_out / "initialization_report.json", init_reports)
    save_json(data_out / "gradient_isolation_report.json", gradient_report)

    pytest_result = {"passed": None, "note": "not_run_in_dry_run"}
    if args.mode == "full":
        pytest_result = run_pytest(data_out)
    else:
        save_text(data_out / "pytest_result.txt", "not run in dry_run mode\n")

    train_results: dict[int, Any] = {}
    coverage_report: dict[int, Any] = {}
    for seed in seeds:
        train_result = train_one_seed(seed, train_episodes, dev_episodes, device, args, checkpoint_out, data_out)
        train_results[seed] = train_result
        coverage_report[seed] = train_result["coverage"]
    save_json(data_out / "epoch_coverage_report.json", coverage_report)

    if args.mode == "dry_run":
        after = protected_hashes()
        dry_report = {
            "mode": "dry_run",
            "seeds": seeds,
            "max_batches": args.max_batches,
            "compatibility_passed": compatibility["passed"],
            "gradient_isolation": gradient_report,
            "coverage": coverage_report,
            "source_protection_hashes_unchanged": hash_status(before, after)["unchanged"],
        }
        save_json(data_out / "dry_run_report.json", dry_report)
        save_json(data_out / "source_protection_hashes_after.json", after)
        print(json.dumps({"dry_run_passed": True, "compatibility_passed": True}, indent=2, sort_keys=True), flush=True)
        return 0

    final_metrics: dict[int, dict[str, Any]] = {}
    all_boundary_statuses: list[str] = []
    for seed in seeds:
        selected = train_results[seed]["selection"]
        router = load_selected_router(seed, Path(selected["best_checkpoint"]), device)
        train_metrics, _ = evaluate_router(router, train_episodes, device, args.batch_size, split="train")
        dev_metrics, _ = evaluate_router(router, dev_episodes, device, args.batch_size, split="dev")
        sealed_metrics, sealed_details = evaluate_router(router, sealed_episodes, device, args.batch_size, split="sealed", return_details=True)
        save_json(data_out / f"final_train_metrics_seed_{seed}.json", train_metrics)
        save_json(data_out / f"final_dev_metrics_seed_{seed}.json", dev_metrics)
        save_json(data_out / f"final_sealed_metrics_seed_{seed}.json", sealed_metrics)

        endpoint = endpoint_equivalence(router, sealed_episodes, device, args.batch_size)
        permutation = exhaustive_permutation_report(router, sealed_episodes, device, args.batch_size)
        reader_disabled = disabled_ablation(router, sealed_episodes, device, args.batch_size, disable="reader")
        transform_disabled = disabled_ablation(router, sealed_episodes, device, args.batch_size, disable="transform")
        same_query = same_query_counterfactual_report(sealed_details, sealed_episodes)
        heldout = heldout_query_type_report(sealed_details, sealed_episodes)
        boundary_metrics, boundary_details = evaluate_router(router, boundary_pool, device, args.batch_size, split="boundary", return_details=True)
        boundary_audit = boundary_audit_report(boundary_details)
        all_boundary_statuses.append(boundary_audit["boundary_audit_status"])

        save_json(data_out / f"exhaustive_permutation_report_seed_{seed}.json", permutation)
        save_json(data_out / f"reader_disabled_ablation_seed_{seed}.json", reader_disabled)
        save_json(data_out / f"transform_disabled_ablation_seed_{seed}.json", transform_disabled)
        save_json(data_out / f"same_query_counterfactual_report_seed_{seed}.json", same_query)
        save_json(data_out / f"heldout_query_type_report_seed_{seed}.json", heldout)
        save_json(data_out / f"boundary_partition_report_seed_{seed}.json", boundary_audit["partition_reports"])
        save_json(data_out / f"boundary_routing_audit_seed_{seed}.json", boundary_audit | {"boundary_metrics": boundary_metrics})

        failed = failed_gates_for_seed(sealed_metrics, heldout, endpoint, permutation, reader_disabled, transform_disabled, same_query)
        final_metrics[seed] = {
            "selection": selected,
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "sealed_metrics": sealed_metrics,
            "endpoint_equivalence": endpoint,
            "permutation": permutation,
            "reader_disabled": reader_disabled,
            "transform_disabled": transform_disabled,
            "same_query": same_query,
            "heldout": heldout,
            "boundary_audit": boundary_audit,
            "failed_gates": failed,
        }
        if seed == seeds[0]:
            write_prediction_examples(data_out / "prediction_examples.txt", router, sealed_episodes, boundary_pool, boundary_audit, device)

    after = protected_hashes()
    save_json(data_out / "source_protection_hashes_after.json", after)
    boundary_status = "CONFIRMED" if all(status == "CONFIRMED" for status in all_boundary_statuses) else "INSUFFICIENT_BOUNDARY_CASES"
    aggregate = {
        "compatibility_passed": True,
        "pytest_result": pytest_result,
        "gradient_isolation_passed": all(row["gate_grad_nonzero"] and row["expert_grad_zero"] for row in gradient_report.values()),
        "source_protection_hashes": hash_status(before, after),
        "failed_gates_by_seed": {seed: rows["failed_gates"] for seed, rows in final_metrics.items()},
        "boundary_status": boundary_status,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "final_metrics_by_seed": final_metrics,
    }
    aggregate["verdict"] = verdict_from_results(aggregate)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    save_json(data_out / "summary_report.json", aggregate)
    write_summary(data_out, aggregate, final_metrics)
    print(json.dumps({"verdict": aggregate["verdict"], "failed_gates_by_seed": aggregate["failed_gates_by_seed"], "boundary_status": boundary_status}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
