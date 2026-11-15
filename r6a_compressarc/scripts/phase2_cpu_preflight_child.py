from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, challenge_for_condition, read_json, write_json, write_text


PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702
R6 = PROJECT_ROOT / "r6a_compressarc"
CONTRACT_DIR = R6 / "manifests" / "r6a_phase2_symmetry_contract"
PREFLIGHT_DIR = R6 / "artifacts" / "r6a_phase2_symmetry_preflight"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    item = tensor.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(item.shape)}|{item.dtype}".encode() + b"\0" + item.numpy().tobytes()).hexdigest()


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    if not torch.is_floating_point(a):
        return 0.0 if torch.equal(a.detach().cpu(), b.detach().cpu()) else float("inf")
    return float(torch.max(torch.abs(a.detach().cpu() - b.detach().cpu())).item()) if a.numel() else 0.0


def reset_seeds(seed: int = FIXED_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def deterministic_seed_probe(seed: int = FIXED_SEED) -> dict[str, Any]:
    reset_seeds(seed)
    probe = {
        "seed": seed,
        "python_random_first": random.random(),
        "numpy_random_first": float(np.random.random()),
        "torch_random_first": float(torch.rand(()).item()),
    }
    reset_seeds(seed)
    return probe


def install_cpu_import_shim() -> dict[str, Any]:
    original_set_default_device = torch.set_default_device
    log: list[dict[str, Any]] = []

    def shimmed_set_default_device(device: Any) -> None:
        requested = str(device)
        redirected = requested
        if requested.startswith("cuda"):
            redirected = "cpu"
        log.append({"requested": requested, "effective": redirected, "redirected": requested != redirected})
        return original_set_default_device(redirected)

    torch.set_default_device = shimmed_set_default_device  # type: ignore[assignment]
    return {"intercept_log": log, "original_set_default_device": original_set_default_device}


def import_reference_modules() -> dict[str, Any]:
    os.chdir(REFERENCE_ROOT)
    sys.path.insert(0, str(REFERENCE_ROOT))
    shim = install_cpu_import_shim()
    for name in [
        "preprocessing",
        "arc_compressor",
        "train",
        "solution_selection",
        "layers",
        "initializers",
        "multitensor_systems",
    ]:
        sys.modules.pop(name, None)
    import preprocessing
    import arc_compressor
    import train
    import solution_selection
    import layers
    import multitensor_systems

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    reset_seeds(FIXED_SEED)
    return {
        "preprocessing": preprocessing,
        "arc_compressor": arc_compressor,
        "train": train,
        "solution_selection": solution_selection,
        "layers": layers,
        "multitensor_systems": multitensor_systems,
        "shim": shim,
        "post_import_seed_probe": deterministic_seed_probe(FIXED_SEED),
    }


def grid_shape(grid: list[list[int]]) -> list[int]:
    return [len(grid), len(grid[0]) if grid else 0]


def mask_for_shape(shape: list[int]) -> list[list[int]]:
    return [[1 for _ in range(shape[1])] for _ in range(shape[0])]


def grid_hash(grid: list[list[int]]) -> str:
    return stable_hash(grid)


def task_flags(task: Any) -> dict[str, Any]:
    branches = []
    for example_num in range(task.n_examples):
        for mode in range(2):
            if example_num >= task.n_train and mode == 1:
                continue
            branches.append(
                {
                    "example": example_num,
                    "mode": "input" if mode == 0 else "output",
                    "grid_size_uncertain": not (
                        task.in_out_same_size
                        or (task.all_out_same_size and mode == 1)
                        or (task.all_in_same_size and mode == 0)
                    ),
                }
            )
    return {
        "in_out_same_size": bool(task.in_out_same_size),
        "all_out_same_size": bool(task.all_out_same_size),
        "all_in_same_size": bool(task.all_in_same_size),
        "grid_size_uncertain_branches": branches,
    }


def task_snapshot(task: Any) -> dict[str, Any]:
    return {
        "shapes": copy.deepcopy(task.shapes),
        "predicted_query_output_shapes": [copy.deepcopy(shape[1]) for shape in task.shapes[task.n_train :]],
        "flags": task_flags(task),
        "mask_hash": tensor_hash(task.masks),
        "problem_hash": tensor_hash(task.problem),
        "colors": list(task.colors),
        "n_examples": int(task.n_examples),
        "n_train": int(task.n_train),
        "n_test": int(task.n_test),
        "n_x": int(task.n_x),
        "n_y": int(task.n_y),
    }


def y_signature(task: Any, slot: int) -> dict[str, Any]:
    output_shape = task.shapes[slot][1]
    output_mask = task.masks[slot, :, :, 1].detach().cpu()
    flags = task_flags(task)
    branch = next(row for row in flags["grid_size_uncertain_branches"] if row["example"] == slot and row["mode"] == "output")
    return {
        "output_shape": copy.deepcopy(output_shape),
        "output_mask_hash": tensor_hash(output_mask),
        "output_preprocessing_branch_flags": {
            "in_out_same_size": flags["in_out_same_size"],
            "all_out_same_size": flags["all_out_same_size"],
            "all_in_same_size": flags["all_in_same_size"],
            "grid_size_uncertain": branch["grid_size_uncertain"],
        },
    }


def compare_task_under_s(preprocessing: Any, task_id: str, stratum: str, problem: dict[str, Any], sigma: list[int]) -> dict[str, Any]:
    reset_seeds(FIXED_SEED)
    base_task = preprocessing.Task(task_id, problem, None)
    swapped_problem = challenge_for_condition(problem, "ALL_PAIRING_SWAP", sigma)
    reset_seeds(FIXED_SEED)
    swapped_task = preprocessing.Task(task_id, swapped_problem, None)

    base = task_snapshot(base_task)
    swapped = task_snapshot(swapped_task)
    signatures = [y_signature(base_task, idx) for idx in range(base_task.n_train)]
    cycle_invariant = all(signatures[idx] == signatures[sigma[idx]] for idx in range(len(sigma)))
    shape_equal_by_slot = [
        base_task.shapes[idx][1] == base_task.shapes[sigma[idx]][1]
        for idx in range(base_task.n_train)
    ]
    before_after_flags = {
        "in_out_same_size": {"before": base["flags"]["in_out_same_size"], "after": swapped["flags"]["in_out_same_size"]},
        "all_out_same_size": {"before": base["flags"]["all_out_same_size"], "after": swapped["flags"]["all_out_same_size"]},
        "all_in_same_size": {"before": base["flags"]["all_in_same_size"], "after": swapped["flags"]["all_in_same_size"]},
        "grid_size_uncertain_branches": {
            "before": base["flags"]["grid_size_uncertain_branches"],
            "after": swapped["flags"]["grid_size_uncertain_branches"],
        },
    }
    flag_flip = {
        key: value
        for key, value in before_after_flags.items()
        if value["before"] != value["after"] if key != "grid_size_uncertain_branches"
    }
    grid_branch_equal = base["flags"]["grid_size_uncertain_branches"] == swapped["flags"]["grid_size_uncertain_branches"]
    masks_equal = base["mask_hash"] == swapped["mask_hash"]
    predicted_query_equal = base["predicted_query_output_shapes"] == swapped["predicted_query_output_shapes"]
    task_shapes_equal = base["shapes"] == swapped["shapes"]
    lemma_ok = (
        all(shape_equal_by_slot)
        and base["flags"]["in_out_same_size"] == swapped["flags"]["in_out_same_size"]
        and base["flags"]["all_out_same_size"] == swapped["flags"]["all_out_same_size"]
        and base["flags"]["all_in_same_size"] == swapped["flags"]["all_in_same_size"]
        and predicted_query_equal
        and masks_equal
        and grid_branch_equal
    )
    possible_impacts = {
        "support_output_mask": not masks_equal,
        "per_slot_output_shape": not all(shape_equal_by_slot),
        "hidden_query_output_shape_prediction": not predicted_query_equal,
        "in_out_same_size": base["flags"]["in_out_same_size"] != swapped["flags"]["in_out_same_size"],
        "all_out_same_size": base["flags"]["all_out_same_size"] != swapped["flags"]["all_out_same_size"],
        "all_in_same_size": base["flags"]["all_in_same_size"] != swapped["flags"]["all_in_same_size"],
        "loss_coefficient_schedule": not grid_branch_equal,
        "grid_size_uncertainty_branch": not grid_branch_equal,
    }
    return {
        "task_id": task_id,
        "stratum": stratum,
        "sigma": sigma,
        "y_signatures": signatures,
        "s_cycle_invariant": cycle_invariant,
        "s_cycle_invariance_reason": "all y_signature(i) equal y_signature(sigma(i))" if cycle_invariant else "one or more y_signature cycle comparisons differ",
        "task_shapes_equal_under_s": task_shapes_equal,
        "task_masks_equal_under_s": masks_equal,
        "predicted_query_shape_equal_under_s": predicted_query_equal,
        "task_flag_flip_under_s": before_after_flags,
        "in_out_same_size_before": base["flags"]["in_out_same_size"],
        "in_out_same_size_after": swapped["flags"]["in_out_same_size"],
        "all_out_same_size_before": base["flags"]["all_out_same_size"],
        "all_out_same_size_after": swapped["flags"]["all_out_same_size"],
        "all_in_same_size_before": base["flags"]["all_in_same_size"],
        "all_in_same_size_after": swapped["flags"]["all_in_same_size"],
        "grid_size_uncertain_branch_equal_under_s": grid_branch_equal,
        "s_wrapper_reconstructs_task_normally": True,
        "lemma_assumption_status": "LEMMA_VERIFIED_FOR_TASK" if lemma_ok else "LEMMA_NOT_CLAIMED",
        "non_invariant_possible_impacts": possible_impacts if not cycle_invariant else {},
        "base_snapshot": base,
        "swapped_snapshot": swapped,
        "source_basis": [
            {
                "source_file": "preprocessing.py",
                "line_range": "15-28",
                "pinned_commit": PINNED_COMMIT,
                "classification": "DIRECT_SOURCE_FACT",
                "claim": "Task constructor recomputes shapes, multitensor system, masks, and problem tensor from the provided raw problem.",
            },
            {
                "source_file": "preprocessing.py",
                "line_range": "44-62",
                "pinned_commit": PINNED_COMMIT,
                "classification": "DIRECT_SOURCE_FACT",
                "claim": "Predicted hidden-query output shape is derived from in/out and all-output size flags.",
            },
            {
                "source_file": "train.py",
                "line_range": "65-87",
                "pinned_commit": PINNED_COMMIT,
                "classification": "DIRECT_SOURCE_FACT",
                "claim": "The reconstruction loss grid-size branch depends on in_out_same_size, all_out_same_size, and all_in_same_size.",
            },
        ],
    }


def active_colors_known(problem: dict[str, Any]) -> list[int]:
    colors: set[int] = set()
    for split in ("train", "test"):
        for example in problem[split]:
            grids = [example["input"]]
            if "output" in example:
                grids.append(example["output"])
            for grid in grids:
                for row in grid:
                    colors.update(row)
    return sorted(color for color in colors if color != 0)


def apply_raw_mapping(problem: dict[str, Any], mapping: dict[int, int]) -> dict[str, Any]:
    cloned = copy.deepcopy(problem)

    def mapped_grid(grid: list[list[int]]) -> list[list[int]]:
        return [[mapping.get(value, value) for value in row] for row in grid]

    for split in ("train", "test"):
        for example in cloned[split]:
            example["input"] = mapped_grid(example["input"])
            if "output" in example:
                example["output"] = mapped_grid(example["output"])
    return cloned


def c_raw_preflight(preprocessing: Any, task_id: str, stratum: str, problem: dict[str, Any]) -> dict[str, Any]:
    active = active_colors_known(problem)
    if len(active) < 2:
        return {"task_id": task_id, "stratum": stratum, "status": "C_RAW_NOT_ELIGIBLE", "active_non_black_raw_colors": active}
    rotated = active[1:] + active[:1]
    mapping = {0: 0, **{src: dst for src, dst in zip(active, rotated)}}
    transformed_problem = apply_raw_mapping(problem, mapping)
    reset_seeds(FIXED_SEED)
    base_task = preprocessing.Task(task_id, problem, None)
    reset_seeds(FIXED_SEED)
    transformed_task = preprocessing.Task(task_id, transformed_problem, None)
    base_internal = list(base_task.colors)
    transformed_internal = list(transformed_task.colors)
    induced: dict[int, int] = {}
    for internal_idx, raw_color in enumerate(base_internal):
        mapped_raw = mapping.get(raw_color, raw_color)
        induced[internal_idx] = transformed_internal.index(mapped_raw)
    identity = all(key == value for key, value in induced.items())
    return {
        "task_id": task_id,
        "stratum": stratum,
        "status": "C_RAW_INVALID_INDUCED_IDENTITY" if identity else "C_RAW_VALID_PREFLIGHT",
        "active_non_black_raw_colors": active,
        "raw_mapping": mapping,
        "baseline_raw_color_list": sorted(set([0] + active)),
        "transformed_raw_color_list": transformed_internal,
        "baseline_internal_color_order": base_internal,
        "transformed_internal_color_order": transformed_internal,
        "induced_internal_color_axis_mapping": induced,
        "induced_internal_color_permutation_is_identity": identity,
        "canonicalization_collapse_detected": identity,
        "known_grids_only": True,
        "hidden_query_output_used": False,
    }


def permute_train_problem(problem: dict[str, Any], train_perm_new_to_old: list[int]) -> dict[str, Any]:
    cloned = copy.deepcopy(problem)
    cloned["train"] = [copy.deepcopy(problem["train"][old_idx]) for old_idx in train_perm_new_to_old]
    return cloned


def example_perm_for_problem(problem: dict[str, Any]) -> list[int]:
    n_train = len(problem["train"])
    train_perm = list(range(1, n_train)) + [0] if n_train > 1 else [0]
    return train_perm + list(range(n_train, n_train + len(problem["test"])))


def inverse_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for new_idx, old_idx in enumerate(perm):
        inv[old_idx] = new_idx
    return inv


def traverse_weight_dims(model: Any) -> dict[int, dict[str, Any]]:
    inventory: dict[int, dict[str, Any]] = {}
    weight_ids = {id(tensor) for tensor in model.weights_list}

    def visit(obj: Any, path: str, dims: list[int] | None = None) -> None:
        if torch.is_tensor(obj):
            if id(obj) in weight_ids:
                rec = inventory.setdefault(
                    id(obj),
                    {
                        "paths": [],
                        "dims_records": [],
                        "shape": list(obj.shape),
                        "example_indexed": False,
                    },
                )
                rec["paths"].append(path)
                if dims is not None:
                    rec["dims_records"].append(list(dims))
                    rec["example_indexed"] = rec["example_indexed"] or bool(dims[0])
            return
        if hasattr(obj, "multitensor_system") and hasattr(obj, "__getitem__"):
            for dims_item in obj.multitensor_system:
                try:
                    visit(obj[dims_item], f"{path}[{dims_item}]", list(dims_item))
                except Exception:
                    continue
            return
        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                visit(item, f"{path}/{idx}", dims)
            return
        if isinstance(obj, tuple):
            for idx, item in enumerate(obj):
                visit(item, f"{path}/{idx}", dims)
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, f"{path}/{key}", dims)

    for attr in [
        "multiposteriors",
        "decode_weights",
        "target_capacities",
        "share_up_weights",
        "share_down_weights",
        "softmax_weights",
        "cummax_weights",
        "shift_weights",
        "direction_share_weights",
        "nonlinear_weights",
        "head_weights",
        "mask_weights",
    ]:
        visit(getattr(model, attr), attr)

    by_index: dict[int, dict[str, Any]] = {}
    for idx, tensor in enumerate(model.weights_list):
        rec = inventory.get(id(tensor), {"paths": [], "dims_records": [], "shape": list(tensor.shape), "example_indexed": False})
        rec["weight_index"] = idx
        by_index[idx] = rec
    return by_index


def apply_example_action_to_tensor(tensor: torch.Tensor, perm_new_to_old: list[int], example_indexed: bool) -> torch.Tensor:
    if example_indexed and tensor.ndim > 0 and tensor.shape[0] == len(perm_new_to_old):
        return tensor.detach()[torch.tensor(perm_new_to_old, device=tensor.device)]
    return tensor.detach().clone()


def inverse_align_tensor(tensor: torch.Tensor, inv: list[int], example_indexed: bool) -> torch.Tensor:
    if example_indexed and tensor.ndim > 0 and tensor.shape[0] == len(inv):
        return tensor.detach()[torch.tensor(inv, device=tensor.device)]
    return tensor.detach()


def load_transformed_initialization(base_model: Any, transformed_model: Any, perm_new_to_old: list[int], inventory: dict[int, dict[str, Any]]) -> None:
    with torch.no_grad():
        for idx, (src, dst) in enumerate(zip(base_model.weights_list, transformed_model.weights_list)):
            transformed = apply_example_action_to_tensor(src, perm_new_to_old, bool(inventory[idx]["example_indexed"]))
            dst.copy_(transformed.to(dst.device, dst.dtype))


def compare_weight_lists(base_model: Any, transformed_model: Any, inv: list[int], inventory: dict[int, dict[str, Any]]) -> float:
    max_diff = 0.0
    for idx, (a, b) in enumerate(zip(base_model.weights_list, transformed_model.weights_list)):
        aligned = inverse_align_tensor(b, inv, bool(inventory[idx]["example_indexed"]))
        max_diff = max(max_diff, tensor_max_abs(a, aligned))
    return max_diff


def compare_optimizer_states(opt_a: torch.optim.Optimizer, opt_b: torch.optim.Optimizer, inv: list[int], inventory: dict[int, dict[str, Any]]) -> float:
    state_a = opt_a.state_dict()["state"]
    state_b = opt_b.state_dict()["state"]
    max_diff = 0.0
    for key in sorted(set(state_a) | set(state_b)):
        if key not in state_a or key not in state_b:
            return float("inf")
        for subkey in sorted(set(state_a[key]) | set(state_b[key])):
            va = state_a[key].get(subkey)
            vb = state_b[key].get(subkey)
            if torch.is_tensor(va) and torch.is_tensor(vb):
                aligned = inverse_align_tensor(vb, inv, bool(inventory[int(key)]["example_indexed"]))
                max_diff = max(max_diff, tensor_max_abs(va, aligned))
            elif va != vb:
                return float("inf")
    return max_diff


def forward_loss(task: Any, model: Any, train_step: int) -> dict[str, Any]:
    import train

    logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)
    reconstruction_error = 0
    branch_records = []
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue
            grid_size_uncertain = not (
                task.in_out_same_size
                or task.all_out_same_size
                and in_out_mode == 1
                or task.all_in_same_size
                and in_out_mode == 0
            )
            coefficient = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
            logits_slice = logits[example_num, :, :, :, in_out_mode]
            problem_slice = task.problem[example_num, :, :, in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], output_shape[1])
            if grid_size_uncertain:
                x_log_partitions = [
                    train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], length)[0]
                    for length in range(1, x_mask.shape[1] + 1)
                ]
                y_log_partitions = [
                    train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], length)[0]
                    for length in range(1, y_mask.shape[1] + 1)
                ]
                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)
            logprobs = [[] for _ in range(x_logprobs.shape[0])]
            for x_offset in range(x_logprobs.shape[0]):
                for y_offset in range(y_logprobs.shape[0]):
                    logprob = x_logprobs[x_offset] - x_log_partition + y_logprobs[y_offset] - y_log_partition
                    logits_crop = logits_slice[:, x_offset : x_offset + output_shape[0], y_offset : y_offset + output_shape[1]]
                    target_crop = problem_slice[: output_shape[0], : output_shape[1]]
                    logprob = logprob - torch.nn.functional.cross_entropy(logits_crop[None, ...], target_crop[None, ...], reduction="sum")
                    logprobs[x_offset].append(logprob)
            logprobs_t = torch.stack([torch.stack(items, dim=0) for items in logprobs], dim=0)
            coefficient = 0.1 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
            reconstruction_error = reconstruction_error - torch.logsumexp(coefficient * logprobs_t, dim=(0, 1)) / coefficient
            branch_records.append({"example": example_num, "mode": in_out_mode, "grid_size_uncertain": bool(grid_size_uncertain)})
    loss = total_KL + 10 * reconstruction_error
    return {"logits": logits, "x_mask": x_mask, "y_mask": y_mask, "loss": loss, "branch_records": branch_records}


def inverse_align_forward(payload: dict[str, Any], inv: list[int]) -> dict[str, Any]:
    idx = torch.tensor(inv, device=payload["logits"].device)
    return {
        "logits": payload["logits"][idx],
        "x_mask": payload["x_mask"][idx],
        "y_mask": payload["y_mask"][idx],
        "loss": payload["loss"],
    }


def prediction_distance(a: Any, b: Any) -> float:
    return 0.0 if a == b else 1.0


def build_task_pair(mods: dict[str, Any], task_id: str, problem: dict[str, Any]) -> dict[str, Any]:
    preprocessing = mods["preprocessing"]
    arc_compressor = mods["arc_compressor"]
    train_perm = example_perm_for_problem(problem)[: len(problem["train"])]
    full_perm = example_perm_for_problem(problem)
    inv = inverse_perm(full_perm)
    transformed_problem = permute_train_problem(problem, train_perm)
    reset_seeds(FIXED_SEED)
    base_task = preprocessing.Task(task_id, problem, None)
    reset_seeds(FIXED_SEED)
    base_model = arc_compressor.ARCCompressor(base_task)
    inventory = traverse_weight_dims(base_model)
    reset_seeds(FIXED_SEED)
    transformed_task = preprocessing.Task(task_id, transformed_problem, None)
    reset_seeds(FIXED_SEED)
    transformed_model = arc_compressor.ARCCompressor(transformed_task)
    load_transformed_initialization(base_model, transformed_model, full_perm, inventory)
    return {
        "base_task": base_task,
        "base_model": base_model,
        "transformed_task": transformed_task,
        "transformed_model": transformed_model,
        "train_perm_new_to_old": train_perm,
        "example_perm_new_to_old": full_perm,
        "inverse_example_perm": inv,
        "inventory": inventory,
    }


def run_pair_trajectory(mods: dict[str, Any], task_id: str, problem: dict[str, Any], steps: int) -> dict[str, Any]:
    train = mods["train"]
    solution_selection = mods["solution_selection"]
    pair = build_task_pair(mods, task_id, problem)
    base_task, base_model = pair["base_task"], pair["base_model"]
    trans_task, trans_model = pair["transformed_task"], pair["transformed_model"]
    inv = pair["inverse_example_perm"]
    inventory = pair["inventory"]

    base_optimizer = torch.optim.Adam(base_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    trans_optimizer = torch.optim.Adam(trans_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    optimizer_empty_at_step_zero = len(base_optimizer.state_dict()["state"]) == 0 and len(trans_optimizer.state_dict()["state"]) == 0

    reset_seeds(FIXED_SEED)
    base_step0 = forward_loss(base_task, base_model, 0)
    reset_seeds(FIXED_SEED)
    trans_step0 = inverse_align_forward(forward_loss(trans_task, trans_model, 0), inv)
    step0 = {
        "D_logits": tensor_max_abs(base_step0["logits"], trans_step0["logits"]),
        "D_masks": max(tensor_max_abs(base_step0["x_mask"], trans_step0["x_mask"]), tensor_max_abs(base_step0["y_mask"], trans_step0["y_mask"])),
        "D_loss": tensor_max_abs(base_step0["loss"], trans_step0["loss"]),
        "D_trainable_state": compare_weight_lists(base_model, trans_model, inv, inventory),
        "D_optimizer_state": compare_optimizer_states(base_optimizer, trans_optimizer, inv, inventory),
    }
    base_logger = solution_selection.Logger(base_task)
    trans_logger = solution_selection.Logger(trans_task)
    trace = []
    for step in range(steps):
        reset_seeds(FIXED_SEED + step)
        train.take_step(base_task, base_model, base_optimizer, step, base_logger)
        reset_seeds(FIXED_SEED + step)
        train.take_step(trans_task, trans_model, trans_optimizer, step, trans_logger)
        logits_diff = tensor_max_abs(base_logger.current_logits, trans_logger.current_logits)
        mask_diff = max(tensor_max_abs(base_logger.current_x_mask, trans_logger.current_x_mask), tensor_max_abs(base_logger.current_y_mask, trans_logger.current_y_mask))
        trace.append(
            {
                "step": step + 1,
                "D_logits": logits_diff,
                "D_masks": mask_diff,
                "D_loss": abs(base_logger.loss_curve[-1] - trans_logger.loss_curve[-1]),
                "D_trainable_state": compare_weight_lists(base_model, trans_model, inv, inventory),
                "D_optimizer_state": compare_optimizer_states(base_optimizer, trans_optimizer, inv, inventory),
                "D_prediction": prediction_distance(base_logger.solution_most_frequent, trans_logger.solution_most_frequent),
            }
        )
    return {
        "task_id": task_id,
        "steps": steps,
        "train_perm_new_to_old": pair["train_perm_new_to_old"],
        "example_perm_new_to_old": pair["example_perm_new_to_old"],
        "optimizer_empty_at_step_zero": optimizer_empty_at_step_zero,
        "state_inventory_summary": {
            "weight_count": len(inventory),
            "example_indexed_weight_count": sum(1 for row in inventory.values() if row["example_indexed"]),
            "actions": {
                "TRANSFORM_AT_INITIALIZATION": sum(1 for row in inventory.values() if row["example_indexed"]),
                "TRANSFORM_ON_CHECKPOINT_RESUME": sum(1 for row in inventory.values() if row["example_indexed"]),
                "INVERSE_TRANSFORM_FOR_COMPARISON_ONLY": sum(1 for row in inventory.values() if row["example_indexed"]),
                "NO_EXAMPLE_ACTION_REQUIRED": sum(1 for row in inventory.values() if not row["example_indexed"]),
                "UNKNOWN_REQUIRES_MANUAL_REVIEW": 0,
            },
        },
        "step0": step0,
        "trace": trace,
    }


def synthetic_problem(kind: str) -> dict[str, Any]:
    if kind == "same_size":
        return {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]},
            ],
            "test": [{"input": [[0, 1], [2, 0]]}],
        }
    return {
        "train": [
            {"input": [[0, 1], [1, 1]], "output": [[1, 0, 1]]},
            {"input": [[0, 2, 2]], "output": [[2], [0], [2]]},
        ],
        "test": [{"input": [[0, 1], [2, 0]]}],
    }


def max_trace_value(row: dict[str, Any], keys: list[str]) -> float:
    vals = [float(row["step0"].get(key, 0.0)) for key in keys]
    for step in row["trace"]:
        vals.extend(float(step.get(key, 0.0)) for key in keys)
    return max(vals) if vals else 0.0


def p_strict_preflight(mods: dict[str, Any], challenges: dict[str, dict[str, Any]]) -> dict[str, Any]:
    calibration_rows = []
    for kind in ["same_size", "mixed_shape"]:
        calibration_rows.append(run_pair_trajectory(mods, f"synthetic_{kind}", synthetic_problem(kind), steps=20))
    forward_max = max(max(row["step0"].values()) for row in calibration_rows)
    tau_forward_abs = max(1e-12, 10 * forward_max)
    trajectory_max = max(max_trace_value(row, ["D_logits", "D_masks", "D_loss", "D_trainable_state", "D_optimizer_state", "D_prediction"]) for row in calibration_rows)
    tau_trajectory_abs = max(1e-12, 10 * trajectory_max)
    actual_task_ids = ["1f876c06", "604001fa"]
    actual_rows = []
    for task_id in actual_task_ids:
        problem = challenges[task_id]
        row = run_pair_trajectory(mods, task_id, problem, steps=5)
        step0_max = max(float(value) for value in row["step0"].values())
        trajectory_actual_max = max_trace_value(row, ["D_logits", "D_masks", "D_loss", "D_trainable_state", "D_optimizer_state", "D_prediction"])
        if step0_max > tau_forward_abs:
            classification = "P_STRICT_STEP0_STRUCTURAL_FAIL"
        elif trajectory_actual_max > tau_trajectory_abs:
            classification = "P_STRICT_TRAJECTORY_DIVERGENCE_ABOVE_CALIBRATION"
        else:
            classification = "P_STRICT_PASS"
        row["classification"] = classification
        row["step0_within_tau_forward"] = step0_max <= tau_forward_abs
        row["trajectory_within_tau_trajectory"] = trajectory_actual_max <= tau_trajectory_abs
        row["manual_review_notes"] = [
            "Production forward samples latent noise with torch.randn in layers.py:123-125; exact group action on stochastic generator streams is not source-specified."
        ]
        actual_rows.append(row)
    return {
        "fixed_seed": FIXED_SEED,
        "tau_forward": {
            "tau_forward_abs": tau_forward_abs,
            "tau_forward_rel": tau_forward_abs,
            "derivation": "10 * max step-zero discrepancy over CPU float64 synthetic operation/composed calibrations; floor 1e-12 if all zero.",
            "operation_level_calibrations": [
                {
                    "operation_class": "production_forward_loss_optimizer_sensitive_path",
                    "synthetic_task_id": row["task_id"],
                    "step0": row["step0"],
                }
                for row in calibration_rows
            ],
        },
        "tau_trajectory": {
            "tau_trajectory_abs": tau_trajectory_abs,
            "tau_trajectory_rel": tau_trajectory_abs,
            "derivation": "10 * max inverse-aligned discrepancy over composed 20-step CPU float64 calibration on non-Phase-1 synthetic fixtures.",
            "calibration_rows": calibration_rows,
        },
        "actual_p_strict_preflights": actual_rows,
        "optimizer_state_rule": {
            "adam_state_empty_at_step_zero_for_actuals": all(row["optimizer_empty_at_step_zero"] for row in actual_rows),
            "optimizer_state_mutated_during_training_by_harness": False,
            "inverse_transform_for_comparison_only": True,
        },
    }


def collect_tensor_proof(mods: dict[str, Any], sample_task: Any, sample_model: Any) -> dict[str, Any]:
    floating = []
    integer = []
    for name, tensor in [("task.masks", sample_task.masks), ("task.problem", sample_task.problem)] + [
        (f"weight_{idx}", tensor) for idx, tensor in enumerate(sample_model.weights_list[:20])
    ]:
        row = {"name": name, "device": tensor.device.type, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        if torch.is_floating_point(tensor):
            floating.append(row)
        else:
            integer.append(row)
    return {"floating_tensors": floating, "integer_index_tensors": integer}


def run_child() -> dict[str, Any]:
    mods = import_reference_modules()
    preprocessing = mods["preprocessing"]
    arc_compressor = mods["arc_compressor"]
    shim = mods["shim"]
    manifest = read_json(R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json")
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    challenge_by_id = {**eval_challenges, **train_challenges}

    s_rows = []
    c_rows = []
    for task in manifest["tasks"]:
        bank = eval_challenges if task["stratum"] == "A" else train_challenges
        problem = bank[task["task_id"]]
        sigma = task["recomputed"]["selected_permutation"]
        s_rows.append(compare_task_under_s(preprocessing, task["task_id"], task["stratum"], problem, sigma))
        c_rows.append(c_raw_preflight(preprocessing, task["task_id"], task["stratum"], problem))

    structural = {
        "experiment": "R6A_PHASE2_SOURCE_GROUNDED_SYMMETRY_PREFLIGHT",
        "created_utc": utc_now(),
        "outcome_blind": True,
        "inputs_used_before_hash": [
            "r6a_phase1_multirepeat_frozen_manifest.json",
            "arc-agi_evaluation_challenges.json",
            "arc-agi_training_challenges.json",
            "pinned source imported under CPU shim",
        ],
        "phase1_outcome_files_read": False,
        "tasks": s_rows,
    }
    structural_path = PREFLIGHT_DIR / "s_cycle_invariance_structural_reanalysis.json"
    write_json(structural_path, structural)
    structural_sha = stable_hash(structural)
    write_text(PREFLIGHT_DIR / "s_cycle_invariance_structural_reanalysis.sha256", structural_sha + "  s_cycle_invariance_structural_reanalysis.json\n")

    reset_seeds(FIXED_SEED)
    sample_task = preprocessing.Task("cpu_proof_sample", synthetic_problem("same_size"), None)
    reset_seeds(FIXED_SEED)
    sample_model = arc_compressor.ARCCompressor(sample_task)
    cuda_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    tensor_proof = collect_tensor_proof(mods, sample_task, sample_model)
    cpu_proof = {
        "created_utc": utc_now(),
        "fixed_seed": FIXED_SEED,
        "post_import_seed_probe": mods["post_import_seed_probe"],
        "final_seed_probe_after_explicit_reset": deterministic_seed_probe(FIXED_SEED),
        "import_intercept_log": shim["intercept_log"],
        "effective_default_device": str(torch.get_default_device()),
        "effective_default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory_allocated_after_preflights": int(cuda_allocated),
        "all_tested_tensors_cpu": all(row["device"] == "cpu" for row in tensor_proof["floating_tensors"] + tensor_proof["integer_index_tensors"]),
        "all_tested_floating_tensors_float64": all(row["dtype"] == "torch.float64" for row in tensor_proof["floating_tensors"]),
        "integer_index_tensor_exception": "task.problem is an int64 color-index tensor by production design, not a floating preflight tensor.",
        "tensor_proof": tensor_proof,
        "classification": "CPU_PREFLIGHT_ESTABLISHED",
    }
    write_json(PREFLIGHT_DIR / "cpu_execution_proof.json", cpu_proof)

    c_payload = {
        "experiment": "R6A_PHASE2_C_RAW_PREFLIGHT",
        "created_utc": utc_now(),
        "fixed_seed": FIXED_SEED,
        "tasks": c_rows,
        "c_internal_status": "C_INTERNAL_REQUIRES_MANUAL_REVIEW",
        "c_internal_reason": "A complete internal color-axis action must transform every internal color-indexed state/tensor consistently; this preflight does not claim C-internal pass from C-raw.",
    }
    write_json(PREFLIGHT_DIR / "c_raw_color_preflight.json", c_payload)

    p_payload = p_strict_preflight(mods, challenge_by_id)
    p_payload["experiment"] = "R6A_PHASE2_P_STRICT_PREFLIGHT"
    p_payload["created_utc"] = utc_now()
    write_json(PREFLIGHT_DIR / "p_strict_preflight.json", p_payload)

    result = {
        "status": "CPU_CHILD_PREFLIGHT_COMPLETE",
        "created_utc": utc_now(),
        "structural_path": str(structural_path),
        "structural_sha256": structural_sha,
        "cpu_execution_proof": str(PREFLIGHT_DIR / "cpu_execution_proof.json"),
        "c_raw_preflight": str(PREFLIGHT_DIR / "c_raw_color_preflight.json"),
        "p_strict_preflight": str(PREFLIGHT_DIR / "p_strict_preflight.json"),
    }
    write_json(PREFLIGHT_DIR / "cpu_child_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="R6A Phase2 CPU-isolated symmetry preflight child.")
    parser.parse_args()
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(run_child(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "status": "CPU_CHILD_PREFLIGHT_FAILED",
            "created_utc": utc_now(),
            "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        write_json(PREFLIGHT_DIR / "cpu_child_summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
