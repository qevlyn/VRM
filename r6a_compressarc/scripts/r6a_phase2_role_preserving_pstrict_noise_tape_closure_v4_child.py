from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, read_json, write_json


PREFIX = "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
TARGET_TASKS = ["1f876c06", "604001fa"]
ABS_TOL = 1e-8
REL_TOL = 1e-8


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reset_seeds(seed: int = FIXED_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    item = tensor.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(item.shape)}|{item.dtype}".encode() + b"\0" + item.numpy().tobytes()).hexdigest()


def max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | bool]:
    if tuple(a.shape) != tuple(b.shape):
        return {"pass": False, "max_abs": float("inf"), "max_rel": float("inf")}
    ac = a.detach().cpu()
    bc = b.detach().cpu()
    if not torch.is_floating_point(ac):
        ok = torch.equal(ac, bc)
        return {"pass": bool(ok), "max_abs": 0.0 if ok else float("inf"), "max_rel": 0.0 if ok else float("inf")}
    diff = torch.abs(ac - bc)
    denom = torch.maximum(torch.abs(ac), torch.abs(bc))
    allowed = ABS_TOL + REL_TOL * denom
    rel = diff / torch.maximum(denom, torch.tensor(1e-300, dtype=denom.dtype))
    return {
        "pass": bool(torch.all(diff <= allowed).item()),
        "max_abs": float(torch.max(diff).item()) if diff.numel() else 0.0,
        "max_rel": float(torch.max(rel).item()) if rel.numel() else 0.0,
    }


def install_cpu_import_shim() -> dict[str, Any]:
    original = torch.set_default_device
    intercept_log: list[dict[str, Any]] = []

    def shim(device: Any) -> None:
        requested = str(device)
        effective = "cpu" if requested.startswith("cuda") else requested
        intercept_log.append({"requested": requested, "effective": effective, "redirected": requested != effective})
        return original(effective)

    torch.set_default_device = shim  # type: ignore[assignment]
    return {"original": original, "intercept_log": intercept_log}


def deterministic_seed_probe() -> dict[str, Any]:
    reset_seeds(FIXED_SEED)
    probe = {
        "seed": FIXED_SEED,
        "python_random_first": random.random(),
        "numpy_random_first": float(np.random.random()),
        "torch_random_first": float(torch.rand(()).item()),
    }
    reset_seeds(FIXED_SEED)
    return probe


def import_pinned_modules() -> dict[str, Any]:
    os.chdir(REFERENCE_ROOT)
    sys.path.insert(0, str(REFERENCE_ROOT))
    shim = install_cpu_import_shim()
    for name in ["preprocessing", "arc_compressor", "train", "solution_selection", "layers", "initializers", "multitensor_systems"]:
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
        "seed_probe": deterministic_seed_probe(),
    }


def support_perm(problem: dict[str, Any]) -> list[int]:
    n_train = len(problem["train"])
    return [(idx + 1) % n_train for idx in range(n_train)] + list(range(n_train, n_train + len(problem["test"])))


def inverse_perm(perm_new_to_old: list[int]) -> list[int]:
    inv = [0] * len(perm_new_to_old)
    for new_idx, old_idx in enumerate(perm_new_to_old):
        inv[old_idx] = new_idx
    return inv


def transformed_problem(problem: dict[str, Any], perm_new_to_old: list[int]) -> dict[str, Any]:
    n_train = len(problem["train"])
    out = copy.deepcopy(problem)
    out["train"] = [copy.deepcopy(problem["train"][old_idx]) for old_idx in perm_new_to_old[:n_train]]
    out["test"] = copy.deepcopy(problem["test"])
    return out


def is_multitensor(obj: Any) -> bool:
    return hasattr(obj, "multitensor_system") and hasattr(obj, "__getitem__")


def iter_model_containers(model: Any) -> list[tuple[str, Any]]:
    names = [
        "multiposteriors",
        "target_capacities",
        "decode_weights",
        "share_up_weights",
        "share_down_weights",
        "softmax_weights",
        "cummax_weights",
        "shift_weights",
        "direction_share_weights",
        "nonlinear_weights",
        "head_weights",
        "mask_weights",
    ]
    return [(name, getattr(model, name)) for name in names]


def build_state_inventory(model: Any, task: Any) -> tuple[list[dict[str, Any]], dict[int, int]]:
    weight_positions: dict[int, list[int]] = {}
    for idx, tensor in enumerate(model.weights_list):
        weight_positions.setdefault(id(tensor), []).append(idx)
    records_by_obj: dict[int, dict[str, Any]] = {}

    def visit(obj: Any, path: str, dims: list[int] | None) -> None:
        if torch.is_tensor(obj):
            oid = id(obj)
            has_real_example_axis = bool(dims and dims[0] == 1 and obj.ndim > 0 and obj.shape[0] == task.n_examples)
            rec = records_by_obj.setdefault(
                oid,
                {
                    "object_id": oid,
                    "structural_paths": [],
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "device": obj.device.type,
                    "requires_grad": bool(obj.requires_grad),
                    "has_real_example_axis": has_real_example_axis,
                    "appears_in_weights_list": oid in weight_positions,
                    "optimizer_parameter_positions": weight_positions.get(oid, []),
                    "dims_records": [],
                },
            )
            rec["structural_paths"].append(path)
            if dims is not None:
                rec["dims_records"].append(list(dims))
            rec["has_real_example_axis"] = rec["has_real_example_axis"] or has_real_example_axis
            return
        if is_multitensor(obj):
            for dims_item in obj.multitensor_system:
                visit(obj[dims_item], f"{path}[{tuple(dims_item)}]", list(dims_item))
            return
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                visit(item, f"{path}/{idx}", dims)
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, f"{path}/{key}", dims)

    for name, container in iter_model_containers(model):
        visit(container, name, None)
    records = list(records_by_obj.values())
    records.sort(key=lambda row: min(row["structural_paths"]))
    alias_id_by_obj: dict[int, int] = {}
    for alias_id, row in enumerate(records):
        row["alias_class_id"] = alias_id
        alias_id_by_obj[int(row["object_id"])] = alias_id
    return records, alias_id_by_obj


def snapshot_unique_tensors(records: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for rec in records:
        obj = _OBJECT_REGISTRY[int(rec["object_id"])]
        out[int(rec["object_id"])] = obj.detach().clone()
    return out


_OBJECT_REGISTRY: dict[int, torch.Tensor] = {}


def register_objects(model: Any, records: list[dict[str, Any]]) -> None:
    wanted = {int(row["object_id"]) for row in records}

    def visit(obj: Any) -> None:
        if torch.is_tensor(obj):
            if id(obj) in wanted:
                _OBJECT_REGISTRY[id(obj)] = obj
            return
        if is_multitensor(obj):
            for dims_item in obj.multitensor_system:
                visit(obj[dims_item])
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
            return
        if isinstance(obj, dict):
            for item in obj.values():
                visit(item)

    for _, container in iter_model_containers(model):
        visit(container)


def restore_snapshot(records: list[dict[str, Any]], snapshot: dict[int, torch.Tensor], perm_new_to_old: list[int] | None = None) -> None:
    with torch.no_grad():
        for rec in records:
            obj = _OBJECT_REGISTRY[int(rec["object_id"])]
            value = snapshot[int(rec["object_id"])]
            if perm_new_to_old is not None and rec["has_real_example_axis"]:
                index = torch.tensor(perm_new_to_old, dtype=torch.long, device=value.device)
                value = value[index]
            obj.copy_(value.to(obj.device, obj.dtype))


def restore_transformed_from_canonical(
    canonical_records: list[dict[str, Any]],
    transformed_records: list[dict[str, Any]],
    canonical_snapshot: dict[int, torch.Tensor],
    perm_new_to_old: list[int],
) -> None:
    if len(canonical_records) != len(transformed_records):
        raise RuntimeError("P_STRICT_SNAPSHOT_RESTORE_TOPOLOGY_MISMATCH")
    index = torch.tensor(perm_new_to_old, dtype=torch.long)
    with torch.no_grad():
        for canonical_rec, transformed_rec in zip(canonical_records, transformed_records):
            source = canonical_snapshot[int(canonical_rec["object_id"])]
            target = _OBJECT_REGISTRY[int(transformed_rec["object_id"])]
            value = source
            if transformed_rec["has_real_example_axis"]:
                value = value[index.to(value.device)]
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError("P_STRICT_SNAPSHOT_RESTORE_SHAPE_MISMATCH")
            target.copy_(value.to(target.device, target.dtype))


def compare_records(
    canonical_records: list[dict[str, Any]],
    transformed_records: list[dict[str, Any]],
    invperm_old_to_new: list[int],
) -> dict[str, Any]:
    failures = []
    max_abs = 0.0
    max_rel = 0.0
    for crec, trec in zip(canonical_records, transformed_records):
        a = _OBJECT_REGISTRY[int(crec["object_id"])]
        b = _OBJECT_REGISTRY[int(trec["object_id"])]
        if trec["has_real_example_axis"]:
            b = b[torch.tensor(invperm_old_to_new, dtype=torch.long, device=b.device)]
        comp = max_abs_rel(a, b)
        max_abs = max(max_abs, float(comp["max_abs"]))
        max_rel = max(max_rel, float(comp["max_rel"]))
        if not comp["pass"]:
            failures.append(
                {
                    "canonical_path": crec["structural_paths"][0],
                    "transformed_path": trec["structural_paths"][0],
                    "max_abs": comp["max_abs"],
                    "max_rel": comp["max_rel"],
                    "canonical_checksum": tensor_checksum(a),
                    "transformed_checksum": tensor_checksum(b),
                }
            )
            break
    return {"pass": not failures, "first_failure": failures[0] if failures else None, "max_abs": max_abs, "max_rel": max_rel}


def expected_noise_leaves(model: Any, task: Any) -> list[dict[str, Any]]:
    leaves = []
    ordinal = 0
    for dims in model.multitensor_system:
        posterior = model.multiposteriors[dims]
        mean = posterior[0]
        leaves.append(
            {
                "leaf_ordinal": ordinal,
                "dims": list(dims),
                "requested_shape": list(mean.shape),
                "dtype": str(mean.dtype),
                "device": mean.device.type,
                "real_example_axis": bool(dims[0] == 1 and mean.ndim > 0 and mean.shape[0] == task.n_examples),
            }
        )
        ordinal += 1
    return leaves


class NoiseTape:
    def __init__(self, layers: Any, task_id: str, stage: str, step: int, expected: list[dict[str, Any]], perm_new_to_old: list[int]):
        self.layers = layers
        self.task_id = task_id
        self.stage = stage
        self.step = step
        self.expected = expected
        self.perm_new_to_old = perm_new_to_old
        self.original = layers.torch.randn
        self.active = False
        self.mode = "record"
        self.forward_invocation_index = 0
        self.cursor = 0
        self.stored: list[torch.Tensor] = []
        self.trace: list[dict[str, Any]] = []
        self.unhandled: list[dict[str, Any]] = []

    @contextmanager
    def context(self, mode: str, stored: list[torch.Tensor] | None = None, transform_for_permuted: bool = False):
        self.mode = mode
        self.cursor = 0
        self.active = True
        self.forward_invocation_index += 1
        if stored is not None:
            self.stored = stored

        def wrapped(*args, **kwargs):
            return self._randn(*args, transform_for_permuted=transform_for_permuted, **kwargs)

        self.layers.torch.randn = wrapped
        try:
            yield self
            if self.cursor != len(self.expected):
                self.unhandled.append({"type": "missing_expected_calls", "consumed": self.cursor, "expected": len(self.expected)})
                raise RuntimeError("P_STRICT_STOCHASTIC_TRACE_INCOMPLETE")
        finally:
            self.layers.torch.randn = self.original
            self.active = False

    def _randn(self, *args, transform_for_permuted: bool = False, **kwargs):
        stack = inspect.stack()
        caller = next((frame for frame in stack[1:] if not frame.filename.endswith(Path(__file__).name)), stack[1])
        if not self.active or caller.function != "channel_layer" or not caller.filename.endswith("layers.py"):
            self.unhandled.append({"type": "unexpected_random_call", "caller": caller.function, "filename": caller.filename})
            raise RuntimeError("P_STRICT_STOCHASTIC_TRACE_INCOMPLETE")
        if self.cursor >= len(self.expected):
            self.unhandled.append({"type": "too_many_random_calls", "cursor": self.cursor})
            raise RuntimeError("P_STRICT_STOCHASTIC_TRACE_INCOMPLETE")
        exp = self.expected[self.cursor]
        requested_shape = list(args[0]) if args else []
        if requested_shape != exp["requested_shape"]:
            self.unhandled.append({"type": "shape_mismatch", "requested": requested_shape, "expected": exp})
            raise RuntimeError("P_STRICT_STOCHASTIC_TRACE_INCOMPLETE")
        requested_dtype = str(kwargs.get("dtype", torch.get_default_dtype()))
        requested_device = torch.device(kwargs.get("device", torch.get_default_device())).type
        if requested_dtype != exp["dtype"] or requested_device != exp["device"]:
            self.unhandled.append(
                {
                    "type": "dtype_or_device_mismatch",
                    "requested_dtype": requested_dtype,
                    "requested_device": requested_device,
                    "expected": exp,
                }
            )
            raise RuntimeError("P_STRICT_STOCHASTIC_TRACE_INCOMPLETE")
        if self.mode == "record":
            epsilon = self.original(*args, **kwargs)
            self.stored.append(epsilon.detach().clone())
        elif self.mode == "replay":
            epsilon = self.stored[self.cursor].detach().clone()
            if transform_for_permuted and exp["real_example_axis"]:
                epsilon = epsilon[torch.tensor(self.perm_new_to_old, dtype=torch.long, device=epsilon.device)]
        elif self.mode == "zero":
            epsilon = torch.zeros(tuple(requested_shape), dtype=torch.get_default_dtype(), device=torch.get_default_device())
        else:
            raise ValueError(self.mode)
        record = {
            "task_id": self.task_id,
            "audit_stage": self.stage,
            "optimizer_step": self.step,
            "forward_invocation_index": self.forward_invocation_index,
            **exp,
            "mode": self.mode,
            "returned_checksum": tensor_checksum(epsilon),
        }
        self.trace.append(record)
        self.cursor += 1
        return epsilon


def snapshot_output(obj: Any, task: Any, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_tensor(path: str, tensor: torch.Tensor, dims: list[int] | None) -> None:
        rows.append(
            {
                "path": path,
                "dims": dims,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": tensor.device.type,
                "real_example_axis": bool(dims and dims[0] == 1 and tensor.ndim > 0 and tensor.shape[0] == task.n_examples)
                or bool(dims is None and tensor.ndim > 0 and tensor.shape[0] == task.n_examples),
                "tensor": tensor.detach().clone(),
                "checksum": tensor_checksum(tensor),
            }
        )

    def visit(path: str, item: Any, dims: list[int] | None = None) -> None:
        if torch.is_tensor(item):
            add_tensor(path, item, dims)
            return
        if is_multitensor(item):
            for dims_item in item.multitensor_system:
                visit(f"{path}[{tuple(dims_item)}]", item[dims_item], list(dims_item))
            return
        if isinstance(item, (list, tuple)):
            for idx, sub in enumerate(item):
                visit(f"{path}/{idx}", sub, dims)
            return
        if isinstance(item, dict):
            for key, sub in item.items():
                visit(f"{path}/{key}", sub, dims)

    visit(name, obj, None)
    return rows


class Observer:
    def __init__(self, layers: Any, task: Any):
        self.layers = layers
        self.task = task
        self.records: list[dict[str, Any]] = []
        self.originals: dict[str, Any] = {}
        self.names = ["decode_latents", "share_up", "softmax", "cummax", "shift", "direction_share", "nonlinear", "share_down", "normalize", "postprocess_mask"]

    def __enter__(self):
        for name in self.names:
            original = getattr(self.layers, name)
            self.originals[name] = original

            def make_wrapper(fn, fname):
                def wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    self.records.append({"function": fname, "call_index": len(self.records), "outputs": snapshot_output(result, self.task, fname)})
                    return result

                return wrapper

            setattr(self.layers, name, make_wrapper(original, name))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            setattr(self.layers, name, original)
        return False


def align_tensor_for_compare(row: dict[str, Any], invperm_old_to_new: list[int]) -> torch.Tensor:
    tensor = row["tensor"]
    if row["real_example_axis"]:
        return tensor[torch.tensor(invperm_old_to_new, dtype=torch.long, device=tensor.device)]
    return tensor


def compare_observer_records(a: list[dict[str, Any]], b: list[dict[str, Any]], invperm_old_to_new: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        return {"pass": False, "first_failure": {"reason": "record_count_mismatch", "canonical": len(a), "transformed": len(b)}}
    for rec_a, rec_b in zip(a, b):
        if rec_a["function"] != rec_b["function"]:
            return {"pass": False, "first_failure": {"reason": "function_sequence_mismatch", "canonical": rec_a["function"], "transformed": rec_b["function"]}}
        if len(rec_a["outputs"]) != len(rec_b["outputs"]):
            return {"pass": False, "first_failure": {"reason": "output_count_mismatch", "function": rec_a["function"]}}
        for out_a, out_b in zip(rec_a["outputs"], rec_b["outputs"]):
            comp = max_abs_rel(out_a["tensor"], align_tensor_for_compare(out_b, invperm_old_to_new))
            if not comp["pass"]:
                return {
                    "pass": False,
                    "first_failure": {
                        "stage": "observer",
                        "function": rec_a["function"],
                        "structural_path": out_a["path"],
                        "max_abs": comp["max_abs"],
                        "max_rel": comp["max_rel"],
                        "canonical_checksum": out_a["checksum"],
                        "inverse_aligned_transformed_checksum": tensor_checksum(align_tensor_for_compare(out_b, invperm_old_to_new)),
                    },
                }
    return {"pass": True, "first_failure": None}


def compute_loss_components(task: Any, logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor, KL_amounts: list[torch.Tensor], train_step: int) -> dict[str, Any]:
    import train

    logits_with_black = torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)
    total_KL = sum(torch.sum(item) for item in KL_amounts)
    reconstruction_error = 0
    for example_num in range(task.n_examples):
        for in_out_mode in range(2):
            if example_num >= task.n_train and in_out_mode == 1:
                continue
            grid_size_uncertain = not (
                task.in_out_same_size
                or (task.all_out_same_size and in_out_mode == 1)
                or (task.all_in_same_size and in_out_mode == 0)
            )
            coefficient = 0.01 ** max(0, 1 - train_step / 100) if grid_size_uncertain else 1
            logits_slice = logits_with_black[example_num, :, :, :, in_out_mode]
            problem_slice = task.problem[example_num, :, :, in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            x_log_partition, x_logprobs = train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], output_shape[0])
            y_log_partition, y_logprobs = train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], output_shape[1])
            if grid_size_uncertain:
                x_log_partition = torch.logsumexp(torch.stack([train.mask_select_logprobs(coefficient * x_mask[example_num, :, in_out_mode], length)[0] for length in range(1, x_mask.shape[1] + 1)], dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack([train.mask_select_logprobs(coefficient * y_mask[example_num, :, in_out_mode], length)[0] for length in range(1, y_mask.shape[1] + 1)], dim=0), dim=0)
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
    loss = total_KL + 10 * reconstruction_error
    return {"logits_with_black": logits_with_black, "total_KL": total_KL, "reconstruction_error": reconstruction_error, "loss": loss, "KL_amounts": KL_amounts}


def predicted_grid(task: Any, logits_with_black: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor) -> Any:
    import solution_selection

    logger = solution_selection.Logger(task)
    try:
        solution, uncertainty = logger._postprocess_solution(logits_with_black[task.n_train :, :, :, :, 1], x_mask[task.n_train :, :, 1], y_mask[task.n_train :, 1])
        return {"status": "OK", "solution": solution, "uncertainty": float(uncertainty)}
    except Exception as exc:
        return {"status": "POSTPROCESS_EXCEPTION", "error_type": type(exc).__name__, "message": str(exc)}


def forward_readout(model: Any, task: Any, layers: Any, stage: str, step: int, tape_mode: str, expected: list[dict[str, Any]], perm_new_to_old: list[int], stored: list[torch.Tensor] | None = None, transform_noise: bool = False, observe: bool = False) -> dict[str, Any]:
    tape = NoiseTape(layers, task.task_name, stage, step, expected, perm_new_to_old)
    observer_records: list[dict[str, Any]] = []
    context = Observer(layers, task) if observe else None
    if context:
        context.__enter__()
    try:
        with tape.context(tape_mode, stored=stored, transform_for_permuted=transform_noise):
            logits, x_mask, y_mask, KL_amounts, KL_names = model.forward()
    finally:
        if context:
            observer_records = context.records
            context.__exit__(None, None, None)
    loss = compute_loss_components(task, logits, x_mask, y_mask, KL_amounts, step)
    return {
        "logits": loss["logits_with_black"],
        "x_mask": x_mask,
        "y_mask": y_mask,
        "KL_amounts": KL_amounts,
        "KL_names": KL_names,
        "total_KL": loss["total_KL"],
        "reconstruction_error": loss["reconstruction_error"],
        "loss": loss["loss"],
        "prediction": predicted_grid(task, loss["logits_with_black"], x_mask, y_mask),
        "noise_trace": tape.trace,
        "stored_noise": tape.stored,
        "unhandled_noise_calls": tape.unhandled,
        "observer_records": observer_records,
    }


def compare_readouts(a: dict[str, Any], b: dict[str, Any], invperm_old_to_new: list[int], include_observer: bool = False) -> dict[str, Any]:
    checks = []
    for name in ["logits", "x_mask", "y_mask"]:
        bt = b[name][torch.tensor(invperm_old_to_new, dtype=torch.long, device=b[name].device)]
        comp = max_abs_rel(a[name], bt)
        checks.append((name, comp, a[name], bt))
    for name in ["total_KL", "reconstruction_error", "loss"]:
        comp = max_abs_rel(a[name], b[name])
        checks.append((name, comp, a[name], b[name]))
    if len(a["KL_amounts"]) != len(b["KL_amounts"]):
        return {"pass": False, "first_failure": {"object": "KL_amounts", "reason": "length_mismatch"}}
    for idx, (ka, kb) in enumerate(zip(a["KL_amounts"], b["KL_amounts"])):
        comp = max_abs_rel(ka, kb)
        checks.append((f"KL_amounts/{idx}", comp, ka, kb))
    if a["prediction"] != b["prediction"]:
        return {"pass": False, "first_failure": {"object": "prediction", "reason": "discrete_prediction_mismatch"}}
    for name, comp, ca, tb in checks:
        if not comp["pass"]:
            return {
                "pass": False,
                "first_failure": {
                    "first_failing_object": name,
                    "structural_path": name,
                    "source_function_or_module": "ARCCompressor.forward/train loss",
                    "max_abs": comp["max_abs"],
                    "max_rel": comp["max_rel"],
                    "canonical_checksum": tensor_checksum(ca if torch.is_tensor(ca) else ca.detach()),
                    "inverse_aligned_transformed_checksum": tensor_checksum(tb if torch.is_tensor(tb) else tb.detach()),
                    "preceding_observed_objects_passed": True,
                },
            }
    if include_observer:
        obs = compare_observer_records(a["observer_records"], b["observer_records"], invperm_old_to_new)
        if not obs["pass"]:
            return obs
    return {"pass": True, "first_failure": None}


def task_reconstruction_checks(base_task: Any, trans_task: Any, perm_new_to_old: list[int], invperm_old_to_new: list[int]) -> dict[str, Any]:
    role_membership = ["train"] * base_task.n_train + ["test"] * base_task.n_test
    transformed_role_membership = ["train"] * trans_task.n_train + ["test"] * trans_task.n_test
    mask_aligned = trans_task.masks[torch.tensor(invperm_old_to_new, dtype=torch.long)]
    checks = {
        "same_n_train": base_task.n_train == trans_task.n_train,
        "same_n_test": base_task.n_test == trans_task.n_test,
        "same_train_test_membership_at_every_index": role_membership == transformed_role_membership and all(role_membership[old_idx] == transformed_role_membership[new_idx] for new_idx, old_idx in enumerate(perm_new_to_old)),
        "same_color_list": base_task.colors == trans_task.colors,
        "same_in_out_same_size": base_task.in_out_same_size == trans_task.in_out_same_size,
        "same_all_in_same_size": base_task.all_in_same_size == trans_task.all_in_same_size,
        "same_all_out_same_size": base_task.all_out_same_size == trans_task.all_out_same_size,
        "same_predicted_query_output_shape": [row[1] for row in base_task.shapes[base_task.n_train :]] == [row[1] for row in trans_task.shapes[trans_task.n_train :]],
        "equal_task_masks_after_inverse_alignment": torch.equal(base_task.masks, mask_aligned),
    }
    return {"pass": all(checks.values()), "checks": checks}


def write_inventory_files(all_records: list[dict[str, Any]], alias_graph: dict[str, Any], snapshot_audit: dict[str, Any]) -> None:
    csv_path = ARTIFACT / "forward_reachable_state_inventory.csv"
    header = ["task_id", "variant", "alias_class_id", "object_id", "structural_path", "shape", "dtype", "device", "requires_grad", "has_real_example_axis", "appears_in_weights_list", "optimizer_parameter_positions"]
    lines = [",".join(header)]
    for row in all_records:
        for path in row["structural_paths"]:
            lines.append(
                ",".join(
                    json.dumps(item) if isinstance(item, (list, bool)) else str(item)
                    for item in [
                        row["task_id"],
                        row["variant"],
                        row["alias_class_id"],
                        row["object_id"],
                        path,
                        row["shape"],
                        row["dtype"],
                        row["device"],
                        row["requires_grad"],
                        row["has_real_example_axis"],
                        row["appears_in_weights_list"],
                        row["optimizer_parameter_positions"],
                    ]
                )
            )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ARTIFACT / "alias_graph.json", alias_graph)
    write_json(ARTIFACT / "snapshot_restore_audit.json", snapshot_audit)


def run_task(mods: dict[str, Any], task_id: str, problem: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    preprocessing = mods["preprocessing"]
    arc_compressor = mods["arc_compressor"]
    layers = mods["layers"]
    if len(problem["train"]) < 2:
        return {"task_id": task_id, "eligible": False, "final_label": "P_ROLE_PERM_NOT_ELIGIBLE"}, [], {}, {}
    perm_new_to_old = support_perm(problem)
    inv = inverse_perm(perm_new_to_old)
    reset_seeds()
    base_task = preprocessing.Task(task_id, problem, None)
    reset_seeds()
    base_model = arc_compressor.ARCCompressor(base_task)
    reset_seeds()
    trans_task = preprocessing.Task(task_id, transformed_problem(problem, perm_new_to_old), None)
    reset_seeds()
    trans_model = arc_compressor.ARCCompressor(trans_task)
    base_records, _ = build_state_inventory(base_model, base_task)
    trans_records, _ = build_state_inventory(trans_model, trans_task)
    register_objects(base_model, base_records)
    register_objects(trans_model, trans_records)
    for row in base_records:
        row["task_id"] = task_id
        row["variant"] = "canonical"
    for row in trans_records:
        row["task_id"] = task_id
        row["variant"] = "transformed"
    base_snapshot = snapshot_unique_tensors(base_records)
    restore_transformed_from_canonical(base_records, trans_records, base_snapshot, perm_new_to_old)
    recon = task_reconstruction_checks(base_task, trans_task, perm_new_to_old, inv)
    snapshot_audit = {
        "task_id": task_id,
        "canonical_unique_forward_reachable_tensors": len(base_records),
        "transformed_unique_forward_reachable_tensors": len(trans_records),
        "canonical_alias_path_count": sum(len(row["structural_paths"]) for row in base_records),
        "transformed_alias_path_count": sum(len(row["structural_paths"]) for row in trans_records),
        "alias_topology_preserved_by_path_counts": [len(row["structural_paths"]) for row in base_records] == [len(row["structural_paths"]) for row in trans_records],
        "restore_used_in_place_copy": True,
        "rebinding_used": False,
        "example_axis_tensor_count": sum(1 for row in base_records if row["has_real_example_axis"]),
        "non_example_axis_tensor_count": sum(1 for row in base_records if not row["has_real_example_axis"]),
    }
    alias_graph = {
        "task_id": task_id,
        "canonical_alias_classes": [{"alias_class_id": row["alias_class_id"], "paths": row["structural_paths"]} for row in base_records],
        "transformed_alias_classes": [{"alias_class_id": row["alias_class_id"], "paths": row["structural_paths"]} for row in trans_records],
    }
    result: dict[str, Any] = {
        "task_id": task_id,
        "eligible": True,
        "perm_new_to_old": perm_new_to_old,
        "invperm_old_to_new": inv,
        "task_reconstruction": recon,
        "stages": {},
        "first_divergence": None,
    }
    runtime_traces: list[dict[str, Any]] = []
    unhandled_calls: list[dict[str, Any]] = []

    def capture(readout: dict[str, Any]) -> None:
        runtime_traces.extend(readout.get("noise_trace", []))
        unhandled_calls.extend(readout.get("unhandled_noise_calls", []))
        if isinstance(readout.get("prediction"), dict) and readout["prediction"].get("status") != "OK":
            result.setdefault("manual_review_reasons", [])
            reason = {
                "stage": readout["noise_trace"][0]["audit_stage"] if readout.get("noise_trace") else None,
                "prediction_status": readout["prediction"],
            }
            if reason not in result["manual_review_reasons"]:
                result["manual_review_reasons"].append(reason)

    def finalize(label: str | None = None):
        if label is not None:
            result["final_label"] = label
        result["noise_runtime_trace"] = runtime_traces
        result["unhandled_noise_calls"] = unhandled_calls
        return result, base_records + trans_records, alias_graph, snapshot_audit

    if not recon["pass"]:
        return finalize("P_ROLE_PRESERVING_TASK_RECONSTRUCTION_MISMATCH")

    expected = expected_noise_leaves(base_model, base_task)
    result["noise_tape_expected_leaves"] = expected
    reset_seeds()
    restore_snapshot(base_records, base_snapshot)
    unmodified = forward_readout(base_model, base_task, layers, "A0_unmodified", 0, "record", expected, perm_new_to_old, observe=False)
    capture(unmodified)
    reset_seeds()
    restore_snapshot(base_records, base_snapshot)
    record = forward_readout(base_model, base_task, layers, "A0_record", 0, "record", expected, perm_new_to_old, observe=True)
    capture(record)
    restore_snapshot(base_records, base_snapshot)
    replay = forward_readout(base_model, base_task, layers, "A0_replay", 0, "replay", expected, perm_new_to_old, stored=record["stored_noise"], observe=True)
    capture(replay)
    a0a = compare_readouts(unmodified, record, list(range(base_task.n_examples)))
    a0b = compare_readouts(record, replay, list(range(base_task.n_examples)), include_observer=True)
    a0_ok = a0a["pass"] and a0b["pass"] and [row["leaf_ordinal"] for row in record["noise_trace"]] == [row["leaf_ordinal"] for row in replay["noise_trace"]]
    result["stages"]["A0"] = {"pass": a0_ok, "unmodified_vs_record": a0a, "record_vs_replay": a0b}
    if not a0_ok:
        result["first_divergence"] = a0a.get("first_failure") or a0b.get("first_failure")
        return finalize("P_STRICT_RUNNER_INSTRUMENTATION_INVALID")

    restore_snapshot(base_records, base_snapshot)
    restore_transformed_from_canonical(base_records, trans_records, base_snapshot, perm_new_to_old)
    cz = forward_readout(base_model, base_task, layers, "A1_zero", 0, "zero", expected, perm_new_to_old, observe=True)
    capture(cz)
    tz = forward_readout(trans_model, trans_task, layers, "A1_zero", 0, "zero", expected, perm_new_to_old, transform_noise=True, observe=True)
    capture(tz)
    state_comp = compare_records(base_records, trans_records, inv)
    a1 = compare_readouts(cz, tz, inv, include_observer=True)
    a1_ok = a1["pass"] and state_comp["pass"] and torch.equal(base_task.masks, trans_task.masks[torch.tensor(inv, dtype=torch.long)])
    result["stages"]["A1"] = {"pass": a1_ok, "readout": a1, "state": state_comp}
    if not a1_ok:
        result["first_divergence"] = a1.get("first_failure") or state_comp.get("first_failure")
        return finalize("P_STRICT_DETERMINISTIC_STEP0_MISMATCH")

    restore_snapshot(base_records, base_snapshot)
    restore_transformed_from_canonical(base_records, trans_records, base_snapshot, perm_new_to_old)
    c_record = forward_readout(base_model, base_task, layers, "A2_coupled", 0, "record", expected, perm_new_to_old, observe=True)
    capture(c_record)
    t_replay = forward_readout(trans_model, trans_task, layers, "A2_coupled", 0, "replay", expected, perm_new_to_old, stored=c_record["stored_noise"], transform_noise=True, observe=True)
    capture(t_replay)
    a2 = compare_readouts(c_record, t_replay, inv, include_observer=True)
    result["stages"]["A2"] = {"pass": a2["pass"], "readout": a2}
    if not a2["pass"]:
        result["first_divergence"] = a2.get("first_failure")
        return finalize("P_STRICT_STOCHASTIC_COUPLING_MISMATCH")

    def backward_once(stage: str, step: int, do_step: bool = False):
        restore_snapshot(base_records, base_snapshot)
        restore_transformed_from_canonical(base_records, trans_records, base_snapshot, perm_new_to_old)
        opt_a = torch.optim.Adam(base_model.weights_list, lr=0.01, betas=(0.5, 0.9))
        opt_b = torch.optim.Adam(trans_model.weights_list, lr=0.01, betas=(0.5, 0.9))
        opt_a.zero_grad()
        opt_b.zero_grad()
        read_a = forward_readout(base_model, base_task, layers, stage, step, "record", expected, perm_new_to_old)
        read_b = forward_readout(trans_model, trans_task, layers, stage, step, "replay", expected, perm_new_to_old, stored=read_a["stored_noise"], transform_noise=True)
        capture(read_a)
        capture(read_b)
        read_a["loss"].backward()
        read_b["loss"].backward()
        grad_fail = None
        for crec, trec in zip(base_records, trans_records):
            a = _OBJECT_REGISTRY[int(crec["object_id"])]
            b = _OBJECT_REGISTRY[int(trec["object_id"])]
            if a.grad is None and b.grad is None:
                continue
            if a.grad is None or b.grad is None:
                grad_fail = {"structural_path": crec["structural_paths"][0], "reason": "grad_presence_mismatch"}
                break
            bg = b.grad[torch.tensor(inv, dtype=torch.long, device=b.grad.device)] if trec["has_real_example_axis"] else b.grad
            comp = max_abs_rel(a.grad, bg)
            if not comp["pass"]:
                grad_fail = {"structural_path": crec["structural_paths"][0], "max_abs": comp["max_abs"], "max_rel": comp["max_rel"]}
                break
        if do_step:
            opt_a.step()
            opt_b.step()
        return read_a, read_b, grad_fail, opt_a, opt_b

    read_a, read_b, grad_fail, _, _ = backward_once("B_backward", 0, do_step=False)
    b_read = compare_readouts(read_a, read_b, inv)
    b_ok = b_read["pass"] and grad_fail is None
    result["stages"]["B"] = {"pass": b_ok, "readout": b_read, "gradient_first_failure": grad_fail}
    if not b_ok:
        result["first_divergence"] = grad_fail or b_read.get("first_failure")
        return finalize("P_STRICT_BACKWARD_MISMATCH")

    read_a, read_b, grad_fail, opt_a, opt_b = backward_once("C_adam", 0, do_step=True)
    c_next_a = forward_readout(base_model, base_task, layers, "C_next_forward", 1, "record", expected, perm_new_to_old)
    c_next_b = forward_readout(trans_model, trans_task, layers, "C_next_forward", 1, "replay", expected, perm_new_to_old, stored=c_next_a["stored_noise"], transform_noise=True)
    capture(c_next_a)
    capture(c_next_b)
    state_after = compare_records(base_records, trans_records, inv)
    optimizer_state = compare_optimizer_states(opt_a, opt_b, base_model, trans_model, base_records, trans_records, inv)
    c_next = compare_readouts(c_next_a, c_next_b, inv)
    c_ok = grad_fail is None and state_after["pass"] and optimizer_state["pass"] and c_next["pass"]
    result["stages"]["C"] = {"pass": c_ok, "state_after": state_after, "optimizer_state": optimizer_state, "next_forward_readout": c_next, "gradient_first_failure": grad_fail}
    if not c_ok:
        result["first_divergence"] = grad_fail or state_after.get("first_failure") or optimizer_state.get("first_failure") or c_next.get("first_failure")
        return finalize("P_STRICT_ADAM_UPDATE_MISMATCH")

    restore_snapshot(base_records, base_snapshot)
    restore_transformed_from_canonical(base_records, trans_records, base_snapshot, perm_new_to_old)
    opt_a = torch.optim.Adam(base_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    opt_b = torch.optim.Adam(trans_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    trajectory = []
    d_ok = True
    for step in range(20):
        opt_a.zero_grad()
        opt_b.zero_grad()
        ra = forward_readout(base_model, base_task, layers, "D_trajectory", step, "record", expected, perm_new_to_old)
        rb = forward_readout(trans_model, trans_task, layers, "D_trajectory", step, "replay", expected, perm_new_to_old, stored=ra["stored_noise"], transform_noise=True)
        capture(ra)
        capture(rb)
        cmp = compare_readouts(ra, rb, inv)
        ra["loss"].backward()
        rb["loss"].backward()
        opt_a.step()
        opt_b.step()
        state_cmp = compare_records(base_records, trans_records, inv)
        opt_cmp = compare_optimizer_states(opt_a, opt_b, base_model, trans_model, base_records, trans_records, inv)
        row = {"step": step + 1, "readout": cmp, "state": state_cmp, "optimizer": opt_cmp, "predicted_grid_equal": ra["prediction"] == rb["prediction"]}
        trajectory.append(row)
        if not (cmp["pass"] and state_cmp["pass"] and opt_cmp["pass"] and row["predicted_grid_equal"]):
            d_ok = False
            result["first_divergence"] = cmp.get("first_failure") or state_cmp.get("first_failure") or opt_cmp.get("first_failure")
            break
    result["stages"]["D"] = {"pass": d_ok, "trace": trajectory}
    if d_ok and result.get("manual_review_reasons"):
        return finalize("P_STRICT_REQUIRES_MANUAL_REVIEW")
    return finalize("P_ROLE_PRESERVING_STRICT_GROUP_ACTION_CONFIRMED" if d_ok else "P_STRICT_TRAJECTORY_MISMATCH")


def compare_optimizer_states(opt_a: torch.optim.Optimizer, opt_b: torch.optim.Optimizer, model_a: Any, model_b: Any, records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], inv: list[int]) -> dict[str, Any]:
    state_a = opt_a.state
    state_b = opt_b.state
    reachable_a = {id(_OBJECT_REGISTRY[int(row["object_id"])]): row for row in records_a if row["appears_in_weights_list"]}
    pos_to_b = {pos: row for row in records_b for pos in row["optimizer_parameter_positions"]}
    for idx, param_a in enumerate(model_a.weights_list):
        if id(param_a) not in reachable_a:
            continue
        param_b = model_b.weights_list[idx]
        sa = state_a.get(param_a, {})
        sb = state_b.get(param_b, {})
        for key in sorted(set(sa) | set(sb)):
            va = sa.get(key)
            vb = sb.get(key)
            if torch.is_tensor(va) and torch.is_tensor(vb):
                rec_b = pos_to_b.get(idx)
                aligned = vb[torch.tensor(inv, dtype=torch.long, device=vb.device)] if rec_b and rec_b["has_real_example_axis"] and vb.ndim > 0 and vb.shape[0] == len(inv) else vb
                comp = max_abs_rel(va, aligned)
                if not comp["pass"]:
                    return {"pass": False, "first_failure": {"optimizer_position": idx, "state_key": key, "max_abs": comp["max_abs"], "max_rel": comp["max_rel"]}}
            elif va != vb:
                return {"pass": False, "first_failure": {"optimizer_position": idx, "state_key": key, "reason": "non_tensor_state_mismatch"}}
    return {"pass": True, "first_failure": None}


def run_child() -> dict[str, Any]:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    mods = import_pinned_modules()
    preprocessing = mods["preprocessing"]
    arc_compressor = mods["arc_compressor"]
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    problems = {"1f876c06": train_challenges["1f876c06"], "604001fa": eval_challenges["604001fa"]}

    sample_task = preprocessing.Task("cpu_proof", {"train": [{"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]}, {"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]}], "test": [{"input": [[0, 1], [2, 0]]}]}, None)
    sample_model = arc_compressor.ARCCompressor(sample_task)
    tensor_rows = [
        {"name": "task.masks", "device": sample_task.masks.device.type, "dtype": str(sample_task.masks.dtype)},
        {"name": "task.problem", "device": sample_task.problem.device.type, "dtype": str(sample_task.problem.dtype)},
    ] + [{"name": f"weight_{idx}", "device": tensor.device.type, "dtype": str(tensor.dtype)} for idx, tensor in enumerate(sample_model.weights_list[:20])]
    cpu_proof = {
        "created_utc": utc_now(),
        "fixed_seed": FIXED_SEED,
        "post_import_seed_probe": mods["seed_probe"],
        "final_seed_probe_after_imports": deterministic_seed_probe(),
        "import_intercept_log": mods["shim"]["intercept_log"],
        "effective_default_device": str(torch.get_default_device()),
        "effective_default_dtype": str(torch.get_default_dtype()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory_allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "all_runtime_tensors_under_test_cpu": all(row["device"] == "cpu" for row in tensor_rows),
        "all_new_floating_preflight_tensors_float64": all(row["dtype"] == "torch.float64" for row in tensor_rows if row["name"] != "task.problem"),
        "tensor_rows": tensor_rows,
    }
    write_json(ARTIFACT / "cpu_execution_proof.json", cpu_proof)

    task_results = []
    all_inventory: list[dict[str, Any]] = []
    alias_graph = {"tasks": []}
    snapshot_audit = {"tasks": []}
    noise_manifest = {"created_utc": utc_now(), "tasks": []}
    noise_runtime = {"created_utc": utc_now(), "trace": []}
    unhandled = {"created_utc": utc_now(), "unhandled_calls": []}
    for task_id in TARGET_TASKS:
        try:
            result, inv_records, graph, snap = run_task(mods, task_id, problems[task_id])
        except RuntimeError as exc:
            label = "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE" if "P_STRICT_STOCHASTIC_TRACE_INCOMPLETE" in str(exc) else "P_STRICT_REQUIRES_MANUAL_REVIEW"
            result = {
                "task_id": task_id,
                "eligible": True,
                "final_label": label,
                "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
            inv_records = []
            graph = {"task_id": task_id, "error": result["failure"]}
            snap = {"task_id": task_id, "error": result["failure"]}
        task_trace = result.pop("noise_runtime_trace", [])
        task_unhandled = result.pop("unhandled_noise_calls", [])
        expected_leaves = result.pop("noise_tape_expected_leaves", [])
        result["noise_runtime_trace_count"] = len(task_trace)
        result["unhandled_noise_call_count"] = len(task_unhandled)
        task_results.append(result)
        all_inventory.extend(inv_records)
        alias_graph["tasks"].append(graph)
        snapshot_audit["tasks"].append(snap)
        noise_runtime["trace"].extend(task_trace)
        unhandled["unhandled_calls"].extend(task_unhandled)
        noise_manifest["tasks"].append(
            {
                "task_id": task_id,
                "perm_new_to_old": result.get("perm_new_to_old"),
                "final_label": result.get("final_label"),
                "expected_noise_leaves": expected_leaves,
                "runtime_trace_count": len(task_trace),
                "unhandled_call_count": len(task_unhandled),
            }
        )
    write_inventory_files(all_inventory, alias_graph, snapshot_audit)
    write_json(ARTIFACT / "closure_results.json", {"created_utc": utc_now(), "tasks": task_results})
    write_json(ARTIFACT / "noise_tape_manifest.json", noise_manifest)
    write_json(ARTIFACT / "noise_tape_runtime_trace.json", noise_runtime)
    write_json(ARTIFACT / "noise_tape_unhandled_call_report.json", unhandled)
    summary = {
        "status": "CHILD_COMPLETE",
        "created_utc": utc_now(),
        "tasks": [{"task_id": row["task_id"], "final_label": row["final_label"]} for row in task_results],
    }
    write_json(ARTIFACT / "child_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    try:
        print(json.dumps(run_child(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {"status": "CHILD_FAILED", "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        write_json(ARTIFACT / "child_summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
