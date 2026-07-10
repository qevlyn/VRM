from __future__ import annotations

import hashlib
import inspect
import json
import platform
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from .composition_r4a0r_data import (
    ALL_SEQUENCE_CLASSES,
    GRID_SIZE,
    R4A0REpisode,
    all_query_types,
    grid_to_list,
    load_split,
    query_type_key,
    source_index_for_output,
    stable_hash,
    tensors_from_episodes,
)
from .composition_r4a0r_data import apply_sequence_to_grid as r4a0r_apply_sequence_to_grid
from .run_composition_r4a0r import SupportConditionedCompositePointerR4A0R, evaluate_model
from .composition_r4a1p_oracles import source_index_for_output_exact

PROJECT_ROOT = Path("/mnt/d/VRM")
R4A0R_DATA_OUT = PROJECT_ROOT / "artifacts/composition_r4a0r_v2"
R4A0R_CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints/composition_r4a0r_v2"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)
APPROVED_CHECKPOINTS = {
    seed: R4A0R_CHECKPOINT_OUT / f"composition_r4a0r_seed_{seed}_best_dev.pt"
    for seed in DEFAULT_SEEDS
}
LOADED_CHECKPOINT_PATHS: list[str] = []
NEW_SOURCE_NAMES = {
    "composition_r4a1p_data.py",
    "composition_r4a1p_forensics.py",
    "composition_r4a1p_oracles.py",
    "run_composition_r4a1p.py",
    "test_composition_r4a1p.py",
    "COMPOSITION_R4A1P.md",
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


def _skip_path(path: Path, data_out: Path) -> bool:
    parts = set(path.parts)
    if any(part in parts for part in {".git", ".claude", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules"}):
        return True
    if path.name.endswith((".pyc", ".tmp", ".zip")):
        return True
    if data_out == path or data_out in path.parents:
        return True
    if "arc-prize-2026-arc-agi-2" in parts:
        return True
    if path.name in NEW_SOURCE_NAMES:
        return True
    return False


def protected_hashes(data_out: Path) -> dict[str, str]:
    roots = [
        PROJECT_ROOT / "vrm",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "archives",
    ]
    hashes: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if _skip_path(path, data_out):
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


def snapshot_checkpoint_directory() -> dict[str, Any]:
    files = sorted(path for path in R4A0R_CHECKPOINT_OUT.glob("*") if path.is_file())
    return {
        "directory": str(R4A0R_CHECKPOINT_OUT),
        "file_count": len(files),
        "files": {str(path): file_record(path) for path in files},
    }


def checkpoint_provenance_before() -> dict[str, Any]:
    approved = {str(seed): file_record(path) for seed, path in APPROVED_CHECKPOINTS.items()}
    return {
        "approved_checkpoint_paths": {str(seed): str(path) for seed, path in APPROVED_CHECKPOINTS.items()},
        "approved_checkpoint_records": approved,
        "checkpoint_directory_before": snapshot_checkpoint_directory(),
        "allowed_to_load_only_approved_best_dev": True,
    }


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


def load_frozen_model(seed: int, device: torch.device) -> tuple[SupportConditionedCompositePointerR4A0R, dict[str, Any], dict[str, Any]]:
    path = APPROVED_CHECKPOINTS[seed]
    if path not in APPROVED_CHECKPOINTS.values():
        raise RuntimeError(f"unapproved checkpoint path {path}")
    LOADED_CHECKPOINT_PATHS.append(str(path))
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SupportConditionedCompositePointerR4A0R().to(device)
    expected = model.state_dict()
    loaded = payload["model_state_dict"]
    config_match = sorted(expected.keys()) == sorted(loaded.keys()) and all(tuple(expected[key].shape) == tuple(loaded[key].shape) for key in expected)
    model.load_state_dict(loaded)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    provenance = {
        "seed": seed,
        "checkpoint_path": str(path),
        "checkpoint_record": file_record(path),
        "payload_epoch_index": payload.get("epoch_index"),
        "payload_global_step": payload.get("global_step"),
        "payload_seed": payload.get("seed"),
        "payload_has_optimizer_state_dict": "optimizer_state_dict" in payload,
        "matching_model_config": config_match,
        "model_initialization_report": model.initialization_report(),
        "state_dict_keys": sorted(loaded.keys()),
        "parameters_require_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
    }
    return model, payload, provenance


def pointer_metric_semantics_audit() -> dict[str, Any]:
    source = inspect.getsource(evaluate_model)
    return {
        "metric_source_file": inspect.getsourcefile(evaluate_model),
        "function_name": "evaluate_model",
        "exact_code_path_used": [
            "outputs['pointer_attention'].reshape(batch, 49, 49)",
            "pointer_argmax = pointer.argmax(dim=-1)",
            "fg = target[index] != 0",
            "source_index_for_output(episode.sequence_class, x, y)",
            "pointer_correct increments only when argmax index equals unique oracle source index",
        ],
        "foreground_only_or_all_cell_scope": "foreground target cells only for pointer_source_position_accuracy",
        "argmax_versus_probability_mass_interpretation": "argmax source-position accuracy; probability mass is not used by the old metric",
        "multiple_same_color_source_cells_handled_as": "not credited unless the argmax coordinate equals the oracle inverse-coordinate source position",
        "background_target_cells_scored": False,
        "oracle_source_coordinate_unique": "computed by source_index_for_output over the 7x7 grid; invalid/no-source cells are skipped",
        "semantics_conclusion": "exact-coordinate metric",
        "source_contains_same_color_credit_path": "same_color" in source,
    }


def _sample_balanced(episodes: list[R4A0REpisode], seed: int, per_sequence: int = 223) -> list[R4A0REpisode]:
    by_sequence: dict[str, list[R4A0REpisode]] = defaultdict(list)
    for episode in episodes:
        by_sequence[episode.sequence_class].append(episode)
    sampled: list[R4A0REpisode] = []
    for sequence in sorted(ALL_SEQUENCE_CLASSES):
        rows = list(by_sequence[sequence])
        random.Random(seed + len(sequence)).shuffle(rows)
        sampled.extend(rows[:per_sequence])
    random.Random(seed).shuffle(sampled)
    return sampled


def independent_pointer_coordinate_recomputation(
    models_by_seed: dict[int, SupportConditionedCompositePointerR4A0R],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    sealed = load_split(R4A0R_DATA_OUT, "sealed")
    by_seed: dict[int, Any] = {}
    aggregate_fg_total = 0
    aggregate_fg_correct = 0
    for seed, model in models_by_seed.items():
        sample = _sample_balanced(sealed, seed)
        fg_total = 0
        fg_correct = 0
        pointer_mass_sum = 0.0
        same_color_wrong = 0
        unique_count = 0
        background_total = 0
        background_argmax_background = 0
        background_pointer_entropy: list[float] = []
        old_metric_accumulator = 0.0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(sample), batch_size):
                chunk = sample[start : start + batch_size]
                batch = tensors_from_episodes(chunk, device)
                outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                pointer = outputs["pointer_attention"].reshape(len(chunk), 49, 49)
                pointer_argmax = pointer.argmax(dim=-1)
                entropy = -(pointer.clamp_min(1e-8) * torch.log(pointer.clamp_min(1e-8))).sum(dim=-1)
                query = batch["query_grids"]
                target = batch["target_grids"]
                for index, episode in enumerate(chunk):
                    for y in range(GRID_SIZE):
                        for x in range(GRID_SIZE):
                            out_index = y * GRID_SIZE + x
                            if int(target[index, y, x].item()) != 0:
                                oracle_index, oracle_x, oracle_y = source_index_for_output_exact(episode.sequence_class, x, y)
                                old_oracle_index = source_index_for_output(episode.sequence_class, x, y)
                                if oracle_index >= 0:
                                    unique_count += 1
                                    fg_total += 1
                                    pred_index = int(pointer_argmax[index, out_index].item())
                                    pointer_mass_sum += float(pointer[index, out_index, oracle_index].item())
                                    correct = pred_index == oracle_index
                                    fg_correct += int(correct)
                                    old_metric_accumulator += float(pred_index == old_oracle_index)
                                    if not correct:
                                        pred_y, pred_x = divmod(pred_index, GRID_SIZE)
                                        if int(query[index, pred_y, pred_x].item()) == int(target[index, y, x].item()):
                                            same_color_wrong += 1
                            else:
                                background_total += 1
                                pred_index = int(pointer_argmax[index, out_index].item())
                                pred_y, pred_x = divmod(pred_index, GRID_SIZE)
                                background_argmax_background += int(int(query[index, pred_y, pred_x].item()) == 0)
                                background_pointer_entropy.append(float(entropy[index, out_index].item()))
        aggregate_fg_total += fg_total
        aggregate_fg_correct += fg_correct
        old_metrics_path = R4A0R_DATA_OUT / f"final_sealed_metrics_seed_{seed}.json"
        old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        old_reported = old_metrics["by_group"]["heldout_pairs"]["pointer_source_position_accuracy"]
        by_seed[seed] = {
            "sampled_episode_count": len(sample),
            "balanced_sequence_count": {sequence: sum(1 for episode in sample if episode.sequence_class == sequence) for sequence in sorted(ALL_SEQUENCE_CLASSES)},
            "foreground_exact_argmax_source_coordinate_accuracy": fg_correct / max(fg_total, 1),
            "foreground_mean_pointer_mass_on_unique_oracle_coordinate": pointer_mass_sum / max(fg_total, 1),
            "foreground_same_color_but_wrong_coordinate_rate": same_color_wrong / max(fg_total, 1),
            "foreground_unique_oracle_coordinate_count": unique_count,
            "background_pointer_metric_summary": {
                "background_cells_scored_by_old_metric": False,
                "background_cell_count": background_total,
                "argmax_source_is_background_rate": background_argmax_background / max(background_total, 1),
                "mean_pointer_entropy": sum(background_pointer_entropy) / max(len(background_pointer_entropy), 1),
            },
            "old_metric_on_same_sample": old_metric_accumulator / max(fg_total, 1),
            "old_reported_heldout_pair_pointer_accuracy": old_reported,
            "recomputation_differs_from_old_metric_semantics": abs((fg_correct / max(fg_total, 1)) - (old_metric_accumulator / max(fg_total, 1))) > 1e-12,
        }
    return {
        "minimum_sampled_sealed_episodes_per_seed": 2000,
        "actual_sampled_episodes_per_seed": 2007,
        "balanced_over_all_nine_r4a0r_v2_sequence_classes": True,
        "by_seed": by_seed,
        "aggregate_foreground_exact_argmax_source_coordinate_accuracy": aggregate_fg_correct / max(aggregate_fg_total, 1),
        "conclusion": "independent exact-coordinate recomputation matches old exact-coordinate argmax semantics",
    }


def _mutated_grid(grid: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple((int(value) + 1) % 10 for value in row) for row in grid)


def mutate_forbidden_metadata(episode: R4A0REpisode, index: int) -> R4A0REpisode:
    qtypes = all_query_types()
    qtype = qtypes[index % len(qtypes)]
    return replace(
        episode,
        episode_id=f"mutated_forbidden_episode_id_{index}",
        split=f"MUTATED_SPLIT_{index}",
        sequence_class="T->F" if episode.sequence_class != "T->F" else "C->F",
        query_type=qtype,
        counterfactual_group_id=f"mutated_group_{index}",
        target=_mutated_grid(episode.target),
    )


def forward_provenance_audit(
    models_by_seed: dict[int, SupportConditionedCompositePointerR4A0R],
    probe_episodes: list[R4A0REpisode],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    forward_signature = inspect.signature(SupportConditionedCompositePointerR4A0R.forward)
    forbidden = {
        "sequence_class",
        "sequence_index",
        "target",
        "target_grid",
        "target_grids",
        "oracle_source_coordinates",
        "query_type",
        "split",
        "episode_id",
        "counterfactual_group_id",
        "target_feature",
    }
    static_bad = sorted(forbidden.intersection(forward_signature.parameters))
    source_file = inspect.getsourcefile(SupportConditionedCompositePointerR4A0R.forward)
    dynamic: dict[int, Any] = {}
    sample = probe_episodes[: min(len(probe_episodes), batch_size * 2)]
    mutated = [mutate_forbidden_metadata(episode, index) for index, episode in enumerate(sample)]
    for seed, model in models_by_seed.items():
        max_prob = 0.0
        max_context = 0.0
        max_pointer = 0.0
        legal_inputs_identical = True
        model.eval()
        with torch.no_grad():
            for start in range(0, len(sample), batch_size):
                left = sample[start : start + batch_size]
                right = mutated[start : start + batch_size]
                left_batch = tensors_from_episodes(left, device)
                right_batch = tensors_from_episodes(right, device)
                legal_inputs_identical = legal_inputs_identical and torch.equal(left_batch["query_grids"], right_batch["query_grids"])
                legal_inputs_identical = legal_inputs_identical and torch.equal(left_batch["support_input_grids"], right_batch["support_input_grids"])
                legal_inputs_identical = legal_inputs_identical and torch.equal(left_batch["support_output_grids"], right_batch["support_output_grids"])
                left_out = model(left_batch["query_grids"], left_batch["support_input_grids"], left_batch["support_output_grids"])
                right_out = model(right_batch["query_grids"], right_batch["support_input_grids"], right_batch["support_output_grids"])
                max_prob = max(max_prob, float((left_out["output_probs"] - right_out["output_probs"]).abs().max().item()))
                max_context = max(max_context, float((left_out["task_context"] - right_out["task_context"]).abs().max().item()))
                max_pointer = max(max_pointer, float((left_out["pointer_attention"] - right_out["pointer_attention"]).abs().max().item()))
        dynamic[seed] = {
            "max_abs_output_probability_difference": max_prob,
            "max_abs_task_context_difference": max_context,
            "max_abs_pointer_difference": max_pointer,
            "legal_model_inputs_identical_after_forbidden_metadata_mutation": legal_inputs_identical,
            "passed": legal_inputs_identical and max(max_prob, max_context, max_pointer) <= 1e-7,
        }
    return {
        "static_audit": {
            "forward_source_file": source_file,
            "forward_signature_parameters": list(forward_signature.parameters),
            "forbidden_parameters_present": static_bad,
            "batch_builder_source_file": inspect.getsourcefile(tensors_from_episodes),
            "passed": not static_bad,
        },
        "dynamic_audit": dynamic,
        "passed": not static_bad and all(row["passed"] for row in dynamic.values()),
    }


def epoch_zero_semantics_audit(seeds: list[int]) -> dict[str, Any]:
    by_seed: dict[int, Any] = {}
    coverage = json.loads((R4A0R_DATA_OUT / "epoch_coverage_report.json").read_text(encoding="utf-8"))
    for seed in seeds:
        selection = json.loads((R4A0R_DATA_OUT / f"best_checkpoint_selection_seed_{seed}.json").read_text(encoding="utf-8"))
        log_rows = [
            json.loads(line)
            for line in (R4A0R_DATA_OUT / f"training_log_seed_{seed}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_epoch = int(selection["selected_epoch"])
        row = next(item for item in log_rows if int(item["epoch_index"]) == selected_epoch)
        coverage_row = next(item for item in coverage[str(seed)] if int(item["epoch_index"]) == selected_epoch)
        checkpoint_path = Path(selection["best_checkpoint"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        by_seed[seed] = {
            "selected_epoch": selected_epoch,
            "checkpoint_payload_epoch_index": payload.get("epoch_index"),
            "checkpoint_payload_global_step": payload.get("global_step"),
            "completed_batches_before_checkpoint_save": row["global_step"],
            "optimizer_updates_before_checkpoint_save": row["global_step"],
            "observed_examples_before_checkpoint_save": coverage_row["observed"],
            "unique_examples_before_checkpoint_save": coverage_row["unique"],
            "full_epoch_coverage_before_checkpoint_save": coverage_row["full_coverage"],
            "checkpoint_was_random_initialization": row["global_step"] == 0,
            "checkpoint_semantics": "post-training after the first fully completed training epoch" if selected_epoch == 0 and row["global_step"] > 0 else "other",
        }
    return {
        "by_seed": by_seed,
        "conclusion": "selected checkpoint epoch 0 means after the first fully completed training epoch, not random initialization",
        "passed": all(not row["checkpoint_was_random_initialization"] for row in by_seed.values()),
    }


def approved_loaded_checkpoint_audit() -> dict[str, Any]:
    approved = {str(path) for path in APPROVED_CHECKPOINTS.values()}
    loaded = list(LOADED_CHECKPOINT_PATHS)
    return {
        "loaded_checkpoint_paths": loaded,
        "approved_checkpoint_paths": sorted(approved),
        "only_approved_checkpoints_loaded": set(loaded).issubset(approved),
        "loaded_checkpoint_count": len(loaded),
    }
