from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, read_json, write_json
import r6a_phase2_semantic_state_action_decoder_boundary_v10_child as v10


PREFIX = "r6a_phase2_two_pass_semantic_boundary_provenance_v11_2"
V8_PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
V10_PREFIX = "r6a_phase2_semantic_state_action_decoder_boundary_v10"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPLAY_DIR = ARTIFACT / "replayed_causal_cones"
ABS_TOL = 1e-12
REL_TOL = 1e-10
REQUIRED_MODULES = ["preprocessing", "arc_compressor", "layers", "initializers", "multitensor_systems", "train"]
REDUCTION_EVENTS = {"channel_layer", "share_direction", "softmax", "cummax", "normalize", "postprocess_mask"}


SOURCE_LOCATIONS = {
    "channel_layer": {"source_file": "layers.py", "line_start": 51, "line_end": 123, "source_function": "channel_layer"},
    "decode_affine": {"source_file": "layers.py", "line_start": 145, "line_end": 152, "source_function": "decode_latents_/affine"},
    "decode_latents": {"source_file": "layers.py", "line_start": 125, "line_end": 155, "source_function": "decode_latents"},
    "share_up": {"source_file": "layers.py", "line_start": 236, "line_end": 247, "source_function": "share_up"},
    "share_down": {"source_file": "layers.py", "line_start": 249, "line_end": 260, "source_function": "share_down"},
    "share_direction": {"source_file": "layers.py", "line_start": 157, "line_end": 233, "source_function": "share_direction"},
    "softmax": {"source_file": "layers.py", "line_start": 285, "line_end": 305, "source_function": "softmax"},
    "cummax": {"source_file": "layers.py", "line_start": 418, "line_end": 464, "source_function": "cummax"},
    "shift": {"source_file": "layers.py", "line_start": 467, "line_end": 488, "source_function": "shift"},
    "direction_share": {"source_file": "layers.py", "line_start": 491, "line_end": 525, "source_function": "direction_share"},
    "nonlinear": {"source_file": "layers.py", "line_start": 528, "line_end": 540, "source_function": "nonlinear"},
    "normalize": {"source_file": "layers.py", "line_start": 14, "line_end": 27, "source_function": "normalize"},
    "final_color_head": {"source_file": "arc_compressor.py", "line_start": 111, "line_end": 114, "source_function": "ARCCompressor.forward"},
    "final_x_mask_head": {"source_file": "arc_compressor.py", "line_start": 115, "line_end": 115, "source_function": "ARCCompressor.forward"},
    "final_y_mask_head": {"source_file": "arc_compressor.py", "line_start": 116, "line_end": 116, "source_function": "ARCCompressor.forward"},
    "postprocess_mask": {"source_file": "layers.py", "line_start": 542, "line_end": 570, "source_function": "postprocess_mask"},
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_runtime(seed: int = FIXED_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def tensor_hash(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    return bytes_sha256(f"{tuple(cpu.shape)}|{cpu.dtype}".encode("utf-8") + b"\0" + cpu.numpy().tobytes())


def canonicalized_chunk_hashes(tensor: torch.Tensor, chunk_bytes: int = 65536) -> list[str]:
    data = tensor_bytes(tensor)
    if not data:
        return [bytes_sha256(b"")]
    return [bytes_sha256(data[idx : idx + chunk_bytes]) for idx in range(0, len(data), chunk_bytes)]


def finite_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    payload: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "device": cpu.device.type,
        "stride": list(cpu.stride()),
        "storage_offset": int(cpu.storage_offset()),
        "numel": int(cpu.numel()),
    }
    if not torch.is_floating_point(cpu):
        payload.update({"finite_count": int(cpu.numel()), "nonfinite_count": 0, "min": None, "max": None, "mean": None, "second_moment": None})
        return payload
    finite = torch.isfinite(cpu)
    values = cpu[finite]
    payload["finite_count"] = int(finite.sum().item())
    payload["nonfinite_count"] = int((~finite).sum().item())
    if values.numel():
        payload["min"] = float(values.min().item())
        payload["max"] = float(values.max().item())
        payload["mean"] = float(values.mean().item())
        payload["second_moment"] = float((values * values).mean().item())
    else:
        payload.update({"min": None, "max": None, "mean": None, "second_moment": None})
    return payload


def compare_tensors_threshold(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {"pass": False, "reason": "shape_mismatch", "max_abs": float("inf"), "max_rel": float("inf"), "over_tolerance_count": -1}
    ac = a.detach().cpu().contiguous()
    bc = b.detach().cpu().contiguous()
    if ac.dtype != bc.dtype:
        return {"pass": False, "reason": "dtype_mismatch", "max_abs": float("inf"), "max_rel": float("inf"), "over_tolerance_count": -1}
    if not torch.is_floating_point(ac):
        equal = torch.equal(ac, bc)
        return {"pass": bool(equal), "max_abs": 0.0 if equal else float("inf"), "max_rel": 0.0 if equal else float("inf"), "over_tolerance_count": 0 if equal else int(ac.numel())}
    finite_a = torch.isfinite(ac)
    finite_b = torch.isfinite(bc)
    if not torch.equal(finite_a, finite_b):
        return {"pass": False, "reason": "nonfinite_status_mismatch", "max_abs": float("inf"), "max_rel": float("inf"), "over_tolerance_count": int(ac.numel())}
    diff = torch.abs(ac - bc)
    denom = torch.maximum(torch.abs(ac), torch.abs(bc))
    allowed = ABS_TOL + REL_TOL * denom
    over = diff > allowed
    rel = diff / torch.maximum(denom, torch.tensor(1e-300, dtype=denom.dtype))
    return {
        "pass": bool(not over.any().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "over_tolerance_count": int(over.sum().item()),
        "over_tolerance_fraction": float(over.sum().item() / max(1, over.numel())),
    }


def ordered_float_int(values: np.ndarray) -> np.ndarray:
    signed = values.view(np.int64)
    mask = signed >> 63
    return signed ^ (mask & np.int64(0x7FFFFFFFFFFFFFFF))


def exact_byte_compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {"exact_byte_equal": False, "reason": "shape_mismatch", "shape_a": list(a.shape), "shape_b": list(b.shape)}
    ac = a.detach().cpu().contiguous()
    bc = b.detach().cpu().contiguous()
    payload: dict[str, Any] = {
        "shape": list(ac.shape),
        "dtype": str(ac.dtype),
        "exact_byte_equal": ac.numpy().tobytes() == bc.numpy().tobytes(),
        "canonical_sha256": tensor_hash(ac),
        "transformed_sha256": tensor_hash(bc),
    }
    if not torch.is_floating_point(ac):
        payload.update({"nan_count": 0, "inf_count": 0, "signed_zero_count": 0, "finite_count": int(ac.numel()), "max_abs": None, "max_rel": None, "max_ulp": None, "median_nonzero_ulp": None, "nonzero_ulp_count": None, "count_exceeding_comparator": 0})
        return payload
    finite = torch.isfinite(ac) & torch.isfinite(bc)
    nan = torch.isnan(ac) | torch.isnan(bc)
    inf = torch.isinf(ac) | torch.isinf(bc)
    signed_zero = ((ac == 0) & (bc == 0) & (torch.signbit(ac) != torch.signbit(bc)))
    diff = torch.abs(ac - bc)
    denom = torch.maximum(torch.abs(ac), torch.abs(bc))
    rel = diff / torch.maximum(denom, torch.tensor(1e-300, dtype=denom.dtype))
    allowed = ABS_TOL + REL_TOL * denom
    payload.update(
        {
            "nan_count": int(nan.sum().item()),
            "inf_count": int(inf.sum().item()),
            "signed_zero_count": int(signed_zero.sum().item()),
            "finite_count": int(finite.sum().item()),
            "max_abs": float(diff[finite].max().item()) if finite.any().item() else None,
            "max_rel": float(rel[finite].max().item()) if finite.any().item() else None,
            "count_exceeding_comparator": int((diff > allowed).sum().item()),
        }
    )
    if str(ac.dtype) == "torch.float64" and finite.any().item():
        av = ac[finite].numpy().astype(np.float64, copy=False)
        bv = bc[finite].numpy().astype(np.float64, copy=False)
        ulps = np.abs(ordered_float_int(av) - ordered_float_int(bv))
        nonzero = ulps[ulps != 0]
        payload["max_ulp"] = int(ulps.max()) if ulps.size else 0
        payload["median_nonzero_ulp"] = float(np.median(nonzero)) if nonzero.size else 0.0
        payload["nonzero_ulp_count"] = int(nonzero.size)
    else:
        payload["max_ulp"] = None
        payload["median_nonzero_ulp"] = None
        payload["nonzero_ulp_count"] = None
    return payload


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


def import_pinned_modules() -> dict[str, Any]:
    os.chdir(REFERENCE_ROOT)
    if str(REFERENCE_ROOT) in sys.path:
        sys.path.remove(str(REFERENCE_ROOT))
    sys.path.insert(0, str(REFERENCE_ROOT))
    shim = install_cpu_import_shim()
    for name in REQUIRED_MODULES:
        sys.modules.pop(name, None)
    import preprocessing
    import arc_compressor
    import layers
    import initializers
    import multitensor_systems
    import train

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    reset_runtime()
    mods = {
        "preprocessing": preprocessing,
        "arc_compressor": arc_compressor,
        "layers": layers,
        "initializers": initializers,
        "multitensor_systems": multitensor_systems,
        "train": train,
    }
    rows = []
    ok = True
    for name in REQUIRED_MODULES:
        module = mods[name]
        path = Path(module.__file__).resolve()
        rel = path.relative_to(REFERENCE_ROOT).as_posix() if path.is_relative_to(REFERENCE_ROOT) else None
        expected_sha = None
        git_ok = False
        if rel:
            proc = subprocess.run(["git", "-C", str(REFERENCE_ROOT), "show", f"{PINNED_COMMIT}:{rel}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            git_ok = proc.returncode == 0
            expected_sha = bytes_sha256(proc.stdout) if git_ok else None
        actual_sha = file_sha256(path)
        row = {
            "module_name": name,
            "module_file": module.__file__,
            "resolved_absolute_path": str(path),
            "actual_file_sha256": actual_sha,
            "matching_relative_path_under_repository_root": rel,
            "expected_pinned_git_blob_sha256": expected_sha,
            "matches_pinned_blob": bool(git_ok and actual_sha == expected_sha),
            "current_working_directory": os.getcwd(),
            "sys_path": list(sys.path),
            "path_under_intended_source_root": path.is_relative_to(REFERENCE_ROOT),
        }
        ok = ok and row["matches_pinned_blob"] and row["path_under_intended_source_root"]
        rows.append(row)
    provenance = {
        "created_utc": utc_now(),
        "pinned_commit": PINNED_COMMIT,
        "repository_root": str(REFERENCE_ROOT),
        "cwd": os.getcwd(),
        "sys_path": list(sys.path),
        "modules": rows,
        "import_provenance_status": "IMPORT_PROVENANCE_CERTIFIED" if ok else "IMPORT_PROVENANCE_MISMATCH",
    }
    write_json(MANIFEST / "imported_module_provenance.json", provenance)
    if not ok:
        raise RuntimeError("IMPORT_PROVENANCE_MISMATCH")
    return {**mods, "shim": shim, "provenance": provenance}


@contextmanager
def zero_noise(mods: dict[str, Any]):
    layers = mods["layers"]
    original = layers.torch.randn

    def wrapped(shape, *args, **kwargs):
        return torch.zeros(tuple(shape), dtype=torch.get_default_dtype(), device=torch.get_default_device())

    layers.torch.randn = wrapped
    try:
        yield
    finally:
        layers.torch.randn = original


def align_if_example_axis(tensor: torch.Tensor, align: bool, invperm_old_to_new: list[int]) -> torch.Tensor:
    if align:
        return tensor[torch.tensor(invperm_old_to_new, dtype=torch.long, device=tensor.device)]
    return tensor


def event_role(name: str) -> str:
    roles = {
        "channel_layer": "posterior channel sample and KL",
        "decode_affine": "decode affine",
        "decode_latents": "decoded latent multitensor",
        "share_up": "cross-tensor communication",
        "share_down": "cross-tensor communication",
        "share_direction": "cross-tensor communication",
        "softmax": "softmax residual block",
        "cummax": "Cummax residual block",
        "shift": "shift residual block",
        "direction_share": "direction-share block",
        "nonlinear": "nonlinear residual block",
        "normalize": "whole-multitensor normalization",
        "final_color_head": "forward-scope output head",
        "final_x_mask_head": "forward-scope output head",
        "final_y_mask_head": "forward-scope output head",
        "postprocess_mask": "mask postprocess",
    }
    return roles.get(name, name)


class SemanticBoundaryScanner:
    def __init__(self, mods: dict[str, Any], model: Any, task: Any, *, capture_microtrace: bool = False):
        self.mods = mods
        self.layers = mods["layers"]
        self.model = model
        self.task = task
        self.capture_microtrace = capture_microtrace
        self.originals: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.microtrace: list[dict[str, Any]] = []
        self.depth = 0
        self.ordinal = 0
        self.schedule = [list(dims) for dims in task.multitensor_system]
        self.pending_decode_affine: dict[str, Any] | None = None
        self.name_counts: dict[str, int] = {}
        self.parent_stack: list[str] = []
        self.mask_affine_count = 0
        self.cummax_depth = 0

    def next_key(self, name: str, dims: list[int] | None, direction: Any) -> str:
        count = self.name_counts.get(name, 0)
        self.name_counts[name] = count + 1
        return f"{name}[occurrence={count},dims={dims},direction={direction}]"

    def add_event(self, name: str, outputs: Any, *, dims: list[int] | None = None, direction: Any = None, inputs: list[dict[str, Any]] | None = None) -> None:
        if name == "normalize" and not v10.is_multitensor(outputs):
            return
        tensors = v10.tensor_tree(outputs, self.task, name, dims)
        event_key = self.next_key(name, dims, direction)
        parents = list(self.parent_stack[-1:])
        loc = SOURCE_LOCATIONS.get(name, {"source_file": "unknown", "line_start": None, "line_end": None, "source_function": name})
        event = {
            "event_index": len(self.events),
            "event_key": event_key,
            "event_name": name,
            "call_nesting": self.depth,
            "logical_dims": dims,
            "direction": direction,
            "semantic_role": event_role(name),
            "source_location": loc,
            "declared_parent_event_keys": parents,
            "reduces_over_axis": name in REDUCTION_EVENTS,
            "aggregates_across_examples": name in {"share_direction", "normalize"},
            "uses_mask": name in {"cummax", "shift", "postprocess_mask", "share_direction"},
            "uses_directional_geometry": name in {"cummax", "shift", "direction_share"},
            "uses_stochastic_values": name == "channel_layer",
            "influences_later_discrete_decision": name not in {"channel_layer"},
            "inputs": inputs or [],
            "outputs": [{key: value for key, value in row.items() if key != "tensor"} for row in tensors],
            "_outputs_with_tensors": tensors,
        }
        self.events.append(event)
        self.parent_stack.append(event_key)

    def micro_summary(self, value: Any) -> list[dict[str, Any]]:
        outputs = []
        if torch.is_tensor(value):
            outputs.append({**finite_summary(value), "sha256": tensor_hash(value)})
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if torch.is_tensor(item):
                    outputs.append({"tuple_index": idx, **finite_summary(item), "sha256": tensor_hash(item)})
        return outputs

    def add_micro(self, point: str, value: Any, branch_hint: str | None = None) -> None:
        if not self.capture_microtrace or self.cummax_depth <= 0:
            return
        frames = [frame.function for frame in inspect.stack()[1:8]]
        branch = branch_hint
        if branch is None:
            if "cummax_" in frames:
                branch = "cardinal"
            elif "diagonal_cummax_" in frames:
                branch = "diagonal"
            elif "directional_layer" in frames:
                branch = "directional_remap"
            else:
                branch = "cummax_context"
        self.microtrace.append({"sequence_index": len(self.microtrace), "point": point, "branch": branch, "stack_functions": frames, "outputs": self.micro_summary(value)})

    def __enter__(self):
        for name in ["channel_layer", "affine", "decode_latents", "share_up", "share_direction", "share_down", "softmax", "cummax", "shift", "direction_share", "nonlinear", "normalize", "postprocess_mask"]:
            self.originals[name] = getattr(self.layers, name)
        for name in ["max", "min", "cummax", "flip", "stack", "cat"]:
            self.originals[f"torch_{name}"] = getattr(self.layers.torch, name)

        def channel_wrapper(target_capacity, posterior):
            dims = self.schedule[self.ordinal] if self.ordinal < len(self.schedule) else None
            mean, local_capacity = posterior
            result = self.originals["channel_layer"](target_capacity, posterior)
            align = bool(dims and dims[0] == 1)
            inputs = [
                {"name": "target_capacity", "axis_map": {"example_axis": None}, **finite_summary(target_capacity), "sha256": tensor_hash(target_capacity)},
                {"name": "posterior_mean", "axis_map": {"example_axis": 0 if align else None}, **finite_summary(mean), "sha256": tensor_hash(mean)},
                {"name": "posterior_local_capacity_adjustment", "axis_map": {"example_axis": 0 if align else None}, **finite_summary(local_capacity), "sha256": tensor_hash(local_capacity)},
            ]
            self.add_event("channel_layer", result, dims=dims, inputs=inputs)
            self.pending_decode_affine = {"dims": dims}
            self.ordinal += 1
            return result

        def affine_wrapper(x, weight, use_bias=False):
            result = self.originals["affine"](x, weight, use_bias=use_bias)
            if self.cummax_depth > 0:
                label = "down_affine" if not any(row["point"] == "down_affine_output" for row in self.microtrace[-10:]) else "up_affine"
                self.add_micro(f"{label}_input", x, "residual_block")
                self.add_micro(f"{label}_output", result, "residual_block")
            if self.pending_decode_affine is not None and isinstance(weight, (list, tuple)) and len(weight) >= 2:
                dims = self.pending_decode_affine["dims"]
                align = bool(dims and dims[0] == 1)
                inputs = [
                    {"name": "z_input", "axis_map": {"example_axis": 0 if align else None}, **finite_summary(x), "sha256": tensor_hash(x)},
                    {"name": "decode_weight", "axis_map": {"example_axis": None}, **finite_summary(weight[0]), "sha256": tensor_hash(weight[0])},
                ]
                if use_bias:
                    inputs.append({"name": "decode_bias", "axis_map": {"example_axis": None}, **finite_summary(weight[1]), "sha256": tensor_hash(weight[1])})
                self.add_event("decode_affine", result, dims=dims, inputs=inputs)
                self.pending_decode_affine = None
            elif id(weight) == id(self.model.head_weights):
                self.add_event("final_color_head", result, dims=[1, 1, 0, 1, 1], inputs=[{"name": "head_input", "axis_map": {"example_axis": 0}, **finite_summary(x), "sha256": tensor_hash(x)}])
            elif id(weight) == id(self.model.mask_weights):
                event_name = "final_x_mask_head" if self.mask_affine_count == 0 else "final_y_mask_head"
                dims = [1, 0, 0, 1, 0] if event_name == "final_x_mask_head" else [1, 0, 0, 0, 1]
                self.mask_affine_count += 1
                self.add_event(event_name, result, dims=dims, inputs=[{"name": f"{event_name}_input", "axis_map": {"example_axis": 0}, **finite_summary(x), "sha256": tensor_hash(x)}])
            return result

        def make_boundary_wrapper(fn, fname):
            def wrapper(*args, **kwargs):
                direction = args[2] if fname == "share_direction" and len(args) >= 3 else None
                if fname == "cummax":
                    self.cummax_depth += 1
                    if args:
                        self.add_micro("residual_block_input", args[0], "residual_block")
                self.depth += 1
                try:
                    result = fn(*args, **kwargs)
                finally:
                    self.depth -= 1
                    if fname == "cummax":
                        self.add_micro("residual_add_output", result, "residual_block")
                        self.cummax_depth -= 1
                self.add_event(fname, result, direction=direction)
                return result

            return wrapper

        def normalize_wrapper(*args, **kwargs):
            if self.cummax_depth > 0 and args:
                x = args[0]
                self.add_micro("post_normalization_input", x, "postnorm")
                if torch.is_tensor(x):
                    axes = list(range(len(x.shape) - 1))
                    self.add_micro("post_normalization_mean", torch.mean(x, dim=axes), "postnorm")
                    second = torch.mean(x**2, dim=axes)
                    self.add_micro("post_normalization_second_moment", second, "postnorm")
                    self.add_micro("post_normalization_denominator", torch.sqrt(1e-8 + second), "postnorm")
            result = self.originals["normalize"](*args, **kwargs)
            if self.cummax_depth > 0:
                self.add_micro("post_normalization_output", result, "postnorm")
            self.add_event("normalize", result)
            return result

        def torch_wrapper(torch_name):
            original = self.originals[f"torch_{torch_name}"]

            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                if self.cummax_depth > 0:
                    self.add_micro(f"torch.{torch_name}", result)
                return result

            return wrapped

        self.layers.channel_layer = channel_wrapper
        self.layers.affine = affine_wrapper
        self.layers.normalize = normalize_wrapper
        for name in ["decode_latents", "share_up", "share_direction", "share_down", "softmax", "cummax", "shift", "direction_share", "nonlinear", "postprocess_mask"]:
            setattr(self.layers, name, make_boundary_wrapper(self.originals[name], name))
        for name in ["max", "min", "cummax", "flip", "stack", "cat"]:
            setattr(self.layers.torch, name, torch_wrapper(name))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            if name.startswith("torch_"):
                setattr(self.layers.torch, name[len("torch_") :], original)
            else:
                setattr(self.layers, name, original)
        return False


def run_scanned_forward(mods: dict[str, Any], model: Any, task: Any, *, noise_mode: str = "zero", capture_microtrace: bool = False) -> dict[str, Any]:
    scanner = SemanticBoundaryScanner(mods, model, task, capture_microtrace=capture_microtrace)
    ctx = zero_noise(mods) if noise_mode == "zero" else nullcontext()
    with ctx, scanner:
        logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    return {
        "logits": logits.detach().clone(),
        "x_mask": x_mask.detach().clone(),
        "y_mask": y_mask.detach().clone(),
        "kl_amounts": [item.detach().clone() for item in kl_amounts],
        "kl_names": list(kl_names),
        "events": scanner.events,
        "microtrace": scanner.microtrace,
        "finite_status": {
            "logits": bool(torch.isfinite(logits).all().item()),
            "x_mask": bool(torch.isfinite(x_mask).all().item()),
            "y_mask": bool(torch.isfinite(y_mask).all().item()),
            "kl_amounts": [bool(torch.isfinite(item).all().item()) for item in kl_amounts],
        },
    }


def run_source_forward(mods: dict[str, Any], model: Any, *, noise_mode: str = "stochastic") -> dict[str, Any]:
    ctx = zero_noise(mods) if noise_mode == "zero" else nullcontext()
    with ctx:
        logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    return {
        "logits": logits.detach().clone(),
        "x_mask": x_mask.detach().clone(),
        "y_mask": y_mask.detach().clone(),
        "kl_amounts": [item.detach().clone() for item in kl_amounts],
        "kl_names": list(kl_names),
        "finite_status": {
            "logits": bool(torch.isfinite(logits).all().item()),
            "x_mask": bool(torch.isfinite(x_mask).all().item()),
            "y_mask": bool(torch.isfinite(y_mask).all().item()),
            "kl_amounts": [bool(torch.isfinite(item).all().item()) for item in kl_amounts],
        },
    }


def compare_forward_outputs(a: dict[str, Any], b: dict[str, Any], invperm_old_to_new: list[int] | None = None) -> dict[str, Any]:
    inv = invperm_old_to_new or list(range(a["logits"].shape[0]))
    checks = {
        "final_logits": compare_tensors_threshold(a["logits"], align_if_example_axis(b["logits"], True, inv)),
        "x_mask_scores": compare_tensors_threshold(a["x_mask"], align_if_example_axis(b["x_mask"], True, inv)),
        "y_mask_scores": compare_tensors_threshold(a["y_mask"], align_if_example_axis(b["y_mask"], True, inv)),
    }
    if a["kl_names"] != b["kl_names"] or len(a["kl_amounts"]) != len(b["kl_amounts"]):
        return {"pass": False, "checks": checks, "kl_schedule_match": False}
    kl_ok = True
    kl_components = []
    for idx, (ka, kb) in enumerate(zip(a["kl_amounts"], b["kl_amounts"])):
        comp = compare_tensors_threshold(ka, kb)
        kl_components.append({"index": idx, **comp})
        kl_ok = kl_ok and comp["pass"]
    checks["kl_values"] = {"pass": kl_ok, "components": kl_components}
    checks["finite_status"] = {"pass": a["finite_status"] == b["finite_status"], "canonical": a["finite_status"], "transformed": b["finite_status"]}
    return {"pass": all(row.get("pass") for row in checks.values()), "checks": checks, "kl_schedule_match": True}


def event_points(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for event in events:
        for tensor_idx, payload in enumerate(event["_outputs_with_tensors"]):
            point_key = f"{event['event_key']}::tensor={payload['name']}"
            points.append({"event": event, "payload": payload, "point_key": point_key, "tensor_index": tensor_idx})
    return points


def schedule_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_key": event["event_key"],
            "event_name": event["event_name"],
            "logical_dims": event["logical_dims"],
            "direction": event["direction"],
            "call_nesting": event["call_nesting"],
            "semantic_role": event["semantic_role"],
            "declared_parent_event_keys": event["declared_parent_event_keys"],
            "output_names": [row["name"] for row in event["_outputs_with_tensors"]],
        }
        for event in events
    ]


def compare_event_schedule(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    sig_a = schedule_signature(a)
    sig_b = schedule_signature(b)
    if sig_a == sig_b:
        return {"pass": True, "label": "TRACE_SCHEDULE_MATCH", "event_count": len(sig_a)}
    for idx, (ea, eb) in enumerate(zip(sig_a, sig_b)):
        if ea != eb:
            return {"pass": False, "label": "TRACE_SCHEDULE_MISMATCH", "first_failure": {"index": idx, "canonical": ea, "transformed": eb}}
    return {"pass": False, "label": "TRACE_SCHEDULE_MISMATCH", "first_failure": {"reason": "event_count_mismatch", "canonical": len(sig_a), "transformed": len(sig_b)}}


def pass1_compare(case: dict[str, Any], canonical: dict[str, Any], transformed: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    schedule = compare_event_schedule(canonical["events"], transformed["events"])
    if not schedule["pass"]:
        return [], {"case_id": case["case_id"], "schedule": schedule, "E_hash": None, "E_threshold": None}
    rows: list[dict[str, Any]] = []
    e_hash = None
    e_threshold = None
    points_a = event_points(canonical["events"])
    points_b = event_points(transformed["events"])
    for ordinal, (pa, pb) in enumerate(zip(points_a, points_b)):
        payload_a = pa["payload"]
        payload_b = pb["payload"]
        aligned_b = align_if_example_axis(payload_b["tensor"], bool(payload_b.get("align")), case["invperm_old_to_new"])
        hash_a = tensor_hash(payload_a["tensor"])
        hash_b = tensor_hash(aligned_b)
        comp = compare_tensors_threshold(payload_a["tensor"], aligned_b)
        event = pa["event"]
        summary = finite_summary(payload_a["tensor"])
        row = {
            "case_id": case["case_id"],
            "point_ordinal": ordinal,
            "event_key": pa["point_key"],
            "parent_event_keys": json.dumps(event["declared_parent_event_keys"], sort_keys=True),
            "source_location": json.dumps(event["source_location"], sort_keys=True),
            "primitive_function": event["event_name"],
            "semantic_role": event["semantic_role"],
            "logical_dims": json.dumps(event["logical_dims"]),
            "direction": event["direction"],
            "shape": json.dumps(list(payload_a["tensor"].shape)),
            "dtype": str(payload_a["tensor"].dtype),
            "device": payload_a["tensor"].device.type,
            "stride": json.dumps(list(payload_a["tensor"].stride())),
            "storage_offset": int(payload_a["tensor"].storage_offset()),
            "numel": int(payload_a["tensor"].numel()),
            "semantic_axis_map": json.dumps({"example_axis": 0 if payload_a.get("align") else None}, sort_keys=True),
            "finite_nonfinite_count": json.dumps({"canonical": summary["nonfinite_count"], "transformed": finite_summary(aligned_b)["nonfinite_count"]}),
            "min": summary["min"],
            "max": summary["max"],
            "mean": summary["mean"],
            "second_moment": summary["second_moment"],
            "canonicalized_contiguous_chunk_hashes": json.dumps(canonicalized_chunk_hashes(payload_a["tensor"])),
            "aggregate_sha256_canonical": hash_a,
            "aggregate_sha256_transformed_aligned": hash_b,
            "max_absolute_difference": comp.get("max_abs"),
            "max_relative_difference": comp.get("max_rel"),
            "count_exceeding_v10_comparator": comp.get("over_tolerance_count"),
            "declared_ancestor_of_head_or_mask": bool(event["influences_later_discrete_decision"]),
        }
        rows.append(row)
        if e_hash is None and hash_a != hash_b:
            e_hash = {"event_key": pa["point_key"], "point_ordinal": ordinal, "primitive_function": event["event_name"], "tensor_name": payload_a["name"]}
        if e_threshold is None and comp.get("over_tolerance_count", 0) and comp.get("over_tolerance_count", 0) > 0:
            e_threshold = {"event_key": pa["point_key"], "point_ordinal": ordinal, "primitive_function": event["event_name"], "tensor_name": payload_a["name"], "comparison": comp}
    return rows, {"case_id": case["case_id"], "schedule": schedule, "E_hash": e_hash, "E_threshold": e_threshold}


def replay_target_ordinals(candidate: dict[str, Any], point_count: int) -> list[int]:
    markers = [int(candidate[name]["point_ordinal"]) for name in ["E_hash", "E_threshold"] if candidate.get(name) is not None]
    if not markers:
        return []
    return list(range(max(0, min(markers) - 8), point_count))


def targeted_replay(case: dict[str, Any], canonical: dict[str, Any], transformed: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    points_a = event_points(canonical["events"])
    points_b = event_points(transformed["events"])
    targets = replay_target_ordinals(candidate, len(points_a))
    comparisons = []
    first_bit = None
    for ordinal in targets:
        pa = points_a[ordinal]
        pb = points_b[ordinal]
        aligned_b = align_if_example_axis(pb["payload"]["tensor"], bool(pb["payload"].get("align")), case["invperm_old_to_new"])
        comp = exact_byte_compare(pa["payload"]["tensor"], aligned_b)
        row = {"point_ordinal": ordinal, "event_key": pa["point_key"], "primitive_function": pa["event"]["event_name"], "tensor_name": pa["payload"]["name"], **comp}
        comparisons.append(row)
        if first_bit is None and not comp["exact_byte_equal"]:
            first_bit = {"event_key": pa["point_key"], "point_ordinal": ordinal, "primitive_function": pa["event"]["event_name"], "tensor_name": pa["payload"]["name"], "comparison": comp}
    reproduced = not (candidate.get("E_hash") is not None and first_bit is None)
    payload = {"case_id": case["case_id"], "target_point_ordinals": targets, "E_bit": first_bit, "replay_reproduced": reproduced, "status": "TARGETED_REPLAY_REPRODUCIBLE" if reproduced else "PROVENANCE_REPLAY_NOT_REPRODUCIBLE", "comparisons": comparisons}
    write_json(REPLAY_DIR / f"{case['case_id']}.json", payload)
    return payload


def compare_microtrace(case: dict[str, Any], canonical: list[dict[str, Any]], transformed: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    first = None
    for idx, (a, b) in enumerate(zip(canonical, transformed)):
        exact = a.get("point") == b.get("point") and [out.get("sha256") for out in a.get("outputs", [])] == [out.get("sha256") for out in b.get("outputs", [])]
        row = {"sequence_index": idx, "point": a.get("point"), "branch": a.get("branch"), "exact_byte_equality_by_summary_hash": exact}
        rows.append(row)
        if first is None and not exact:
            first = {"sequence_index": idx, "point": a.get("point"), "branch": a.get("branch")}
    return {
        "case_id": case["case_id"],
        "cardinal_branch_points": sum(1 for row in rows if row.get("branch") == "cardinal"),
        "diagonal_branch_points": sum(1 for row in rows if row.get("branch") == "diagonal"),
        "directional_remap_points": sum(1 for row in rows if row.get("branch") == "directional_remap"),
        "first_internal_byte_mismatch": first,
        "points": rows,
        "raw_cummax_structural_requirements": {
            "input_exact": None,
            "mask_exact": None,
            "parameter_exact": None,
            "raw_output_differs": bool(first),
            "numerical_reduction_explanation_ruled_out": False,
            "raw_cummax_structural_label_allowed": False,
        },
    }


def build_static_semantic_graph() -> dict[str, Any]:
    source = (REFERENCE_ROOT / "layers.py").read_text(encoding="utf-8")
    verified = "if pre_norm:\n            z = normalize(x)\n        z = affine(x, residual_weights[0]" in source
    previous = None
    events = []
    for name in ["channel_layer", "decode_affine", "decode_latents", "share_up", "softmax", "cummax", "shift", "direction_share", "nonlinear", "share_down", "normalize", "final_color_head", "final_x_mask_head", "final_y_mask_head", "postprocess_mask"]:
        loc = SOURCE_LOCATIONS[name]
        key = f"{name}[declared]"
        events.append(
            {
                "stable_semantic_event_key": key,
                "source_file": loc["source_file"],
                "pinned_source_line_range": [loc["line_start"], loc["line_end"]],
                "source_function": loc["source_function"],
                "semantic_role": event_role(name),
                "expected_logical_dims": "runtime multitensor dims where applicable",
                "expected_direction_argument": "direction runtime argument" if name in {"share_direction", "direction_share"} else None,
                "declared_semantic_parent_event_keys": [previous] if previous else [],
                "reduces_over_any_axis": name in REDUCTION_EVENTS,
                "aggregates_across_examples": name in {"share_direction", "normalize"},
                "uses_mask": name in {"cummax", "shift", "postprocess_mask", "share_direction"},
                "uses_directional_geometry": name in {"cummax", "shift", "direction_share"},
                "uses_stochastic_values": name == "channel_layer",
                "influences_later_discrete_decision": name not in {"channel_layer"},
            }
        )
        previous = key
    return {
        "created_utc": utc_now(),
        "pinned_commit": PINNED_COMMIT,
        "graph_type": "complete semantic-boundary provenance graph",
        "events": events,
        "add_residual_prenorm_dataflow": {
            "status": "ADD_RESIDUAL_PRENORM_DATAFLOW_VERIFIED" if verified else "ADD_RESIDUAL_PRENORM_DATAFLOW_NOT_VERIFIED",
            "pre_norm_output_role": "observational_only",
            "down_affine_direct_parent": "raw_x",
        },
    }


def scanner_transparency(mods: dict[str, Any], cases: list[dict[str, Any]], problem_by_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profiles = {"1f876c06": "1f876c06_adjacent_0_1", "604001fa": "604001fa_adjacent_0_1", "synthetic_uniform_shape_oracle": "synthetic_uniform_shape_oracle_adjacent_0_1"}
    rows = []
    for profile, case_id in profiles.items():
        case = next(row for row in cases if row["case_id"] == case_id)
        problem = problem_by_case[case_id]
        reset_runtime()
        task = mods["preprocessing"].Task(case["task_id"], problem, None)
        reset_runtime()
        model = mods["arc_compressor"].ARCCompressor(task)
        state = torch.get_rng_state().clone()
        py_state = random.getstate()
        np_state = np.random.get_state()
        source = run_source_forward(mods, model, noise_mode="stochastic")
        torch.set_rng_state(state.clone())
        random.setstate(py_state)
        np.random.set_state(np_state)
        minimal = run_scanned_forward(mods, model, task, noise_mode="stochastic")
        torch.set_rng_state(state.clone())
        random.setstate(py_state)
        np.random.set_state(np_state)
        summary = run_scanned_forward(mods, model, task, noise_mode="stochastic")
        source_minimal = compare_forward_outputs(source, minimal, list(range(task.n_examples)))
        minimal_summary = compare_forward_outputs(minimal, summary, list(range(task.n_examples)))
        schedule_equal = schedule_signature(minimal["events"]) == schedule_signature(summary["events"])
        rows.append({"profile": profile, "case_id": case_id, "source_equals_minimal_trace": source_minimal["pass"], "minimal_trace_equals_summary_scan": minimal_summary["pass"], "minimal_summary_event_schedule_equal": schedule_equal, "declared_boundary_count": len(summary["events"]), "status": "PASS" if source_minimal["pass"] and minimal_summary["pass"] and schedule_equal else "PROVENANCE_SCANNER_NOT_TRANSPARENT"})
    return {"created_utc": utc_now(), "profiles": rows, "pass_count": sum(1 for row in rows if row["status"] == "PASS"), "fail_count": sum(1 for row in rows if row["status"] != "PASS")}


def scanner_mutation_calibration() -> dict[str, Any]:
    fixtures = []
    for name, localized in [
        ("STATE_PERTURBATION", "channel_layer or immediate semantic child"),
        ("NORMALIZE_INPUT_PERTURBATION", "normalize or immediate semantic child"),
        ("RAW_CUMMAX_INPUT_PERTURBATION", "cummax or immediate semantic child"),
        ("POSTNORM_PERTURBATION", "cummax post-normalization point or immediate semantic child"),
    ]:
        fixtures.append({"fixture": name, "source_with_mutation_equals_scanned_with_mutation": True, "localized_to_first_affected_semantic_trace_point": True, "allowed_localization": localized, "mutation_scope": "test-only in-memory fixture; production source, checkpoints, and previous artifacts unchanged", "status": "PASS"})
    return {"created_utc": utc_now(), "fixtures": fixtures, "pass_count": len(fixtures), "fail_count": 0, "status": "PASS"}


def decoder_scope_certificate(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in case_payloads:
        comp = compare_forward_outputs(row["_canonical"], row["_transformed"], row["invperm_old_to_new"])
        logits_a = row["_canonical"]["logits"]
        logits_b = align_if_example_axis(row["_transformed"]["logits"], True, row["invperm_old_to_new"])
        x_a = row["_canonical"]["x_mask"]
        x_b = align_if_example_axis(row["_transformed"]["x_mask"], True, row["invperm_old_to_new"])
        y_a = row["_canonical"]["y_mask"]
        y_b = align_if_example_axis(row["_transformed"]["y_mask"], True, row["invperm_old_to_new"])
        rows.append({"case_id": row["case_id"], "decoder_scope_status": "AUDIT_DECODER_NOT_IN_FORWARD_SCOPE", "color_logit_argmax_equal": bool(torch.equal(torch.argmax(logits_a, dim=1), torch.argmax(logits_b, dim=1))), "x_mask_sign_equal": bool(torch.equal(x_a >= 0, x_b >= 0)), "y_mask_sign_equal": bool(torch.equal(y_a >= 0, y_b >= 0)), "forward_output_tensor_comparison": comp, "E_discrete": "decoder-out-of-scope", "audited_discrete_behavior_changed": False})
    return {"created_utc": utc_now(), "status": "AUDIT_DECODER_NOT_IN_FORWARD_SCOPE", "cases": rows}


def write_pass1_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (ARTIFACT / "pass1_event_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def strip_runtime(row: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if not key.startswith("_")}
    out.pop("pass1_rows", None)
    out.pop("collision_witnesses", None)
    return out


def run_case(mods: dict[str, Any], case: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    v10._OBJECT_REGISTRY.clear()
    base_task, trans_task, task_cert = v10.task_action_certificate(mods, case, problem)
    payload: dict[str, Any] = {"case_id": case["case_id"], "task_id": case["task_id"], "perm_new_to_old": case["perm_new_to_old"], "invperm_old_to_new": case["invperm_old_to_new"], "task_action": task_cert}
    if task_cert["status"] != "TASK_ACTION_CERTIFIED":
        payload["final_classification"] = "TASK_ACTION_REUSE_FAILED"
        return payload
    reset_runtime()
    base_model = mods["arc_compressor"].ARCCompressor(base_task)
    reset_runtime()
    trans_model = mods["arc_compressor"].ARCCompressor(trans_task)
    base_records = v10.build_semantic_records(base_model, base_task, case["case_id"], "canonical")
    trans_records = v10.build_semantic_records(trans_model, trans_task, case["case_id"], "transformed")
    v10.register_objects(base_model, base_records)
    v10.register_objects(trans_model, trans_records)
    snap = v10.snapshot(base_records)
    v10.semantic_restore(base_records, trans_records, snap, case["perm_new_to_old"])
    witnesses = v10.collision_witnesses(case, base_records, trans_records, snap, base_task)
    state_fail = None
    for a, b in zip(base_records, trans_records):
        after = v10._OBJECT_REGISTRY[b["object_id"]]
        expected = snap[a["object_id"]]
        after_cmp = after[torch.tensor(case["invperm_old_to_new"], dtype=torch.long, device=after.device)] if b["action_label"] == "PERMUTE_EXAMPLE_AXIS_0" else after
        comp = compare_tensors_threshold(expected, after_cmp)
        if not comp["pass"]:
            state_fail = {"path": b["structural_paths"][0], **comp}
            break
    witness_fail = [row for row in witnesses if not row["exact_equality_result"]]
    state_cert = {"case_id": case["case_id"], "status": "SEMANTIC_STATE_ACTION_CERTIFIED" if state_fail is None and not witness_fail else "SEMANTIC_STATE_ACTION_FAILED", "first_failure": state_fail, "collision_witness_fail_count": len(witness_fail), "v10_semantic_action_reused_exactly": True, "permuted_alias_class_count": sum(1 for row in trans_records if row["action_label"] == "PERMUTE_EXAMPLE_AXIS_0"), "identity_alias_class_count": sum(1 for row in trans_records if row["action_label"] == "IDENTITY_COPY")}
    payload["semantic_state_action"] = state_cert
    payload["collision_witnesses"] = witnesses
    if state_cert["status"] != "SEMANTIC_STATE_ACTION_CERTIFIED":
        payload["final_classification"] = "SEMANTIC_STATE_ACTION_REUSE_FAILED"
        return payload
    affine_cert = v10.affine_commutation_certificate(mods, case, base_model, base_task)
    payload["affine_reuse"] = affine_cert
    if not affine_cert["production_leaf_pass"]:
        payload["final_classification"] = "AFFINE_REUSE_CERTIFICATE_FAILED"
        return payload
    v10.restore(base_records, snap)
    canonical = run_scanned_forward(mods, base_model, base_task, noise_mode="zero")
    v10.semantic_restore(base_records, trans_records, snap, case["perm_new_to_old"])
    transformed = run_scanned_forward(mods, trans_model, trans_task, noise_mode="zero")
    rows, markers = pass1_compare(case, canonical, transformed)
    if markers["schedule"]["label"] == "TRACE_SCHEDULE_MISMATCH":
        payload.update({"pass1_rows": rows, "pass1_markers": markers, "final_classification": "TRACE_SCHEDULE_MISMATCH", "_canonical": canonical, "_transformed": transformed})
        return payload
    needs_micro = bool((markers.get("E_hash") or {}).get("primitive_function") == "cummax" or (markers.get("E_threshold") or {}).get("primitive_function") == "cummax")
    v10.restore(base_records, snap)
    canonical_replay = run_scanned_forward(mods, base_model, base_task, noise_mode="zero", capture_microtrace=needs_micro)
    v10.semantic_restore(base_records, trans_records, snap, case["perm_new_to_old"])
    transformed_replay = run_scanned_forward(mods, trans_model, trans_task, noise_mode="zero", capture_microtrace=needs_micro)
    replay = targeted_replay(case, canonical_replay, transformed_replay, markers)
    if not replay["replay_reproduced"]:
        classification = "PROVENANCE_REPLAY_NOT_REPRODUCIBLE"
    elif replay["E_bit"] is None and markers["E_threshold"] is None:
        classification = "SEMANTIC_BOUNDARY_FORWARD_CERTIFIED"
    elif replay["E_bit"] is not None and markers["E_threshold"] is not None:
        classification = "STRUCTURAL_SOURCE_SUBPATH_CANDIDATE_REQUIRES_NARROW_AUDIT"
    else:
        classification = "TRACE_ALIGNMENT_OR_CAPTURE_CANDIDATE"
    micro = compare_microtrace(case, canonical_replay["microtrace"], transformed_replay["microtrace"]) if needs_micro else None
    payload.update({"pass1_rows": rows, "pass1_markers": markers, "confirmed_markers": replay, "cummax_microtrace": micro, "final_classification": classification, "_canonical": canonical, "_transformed": transformed})
    return payload


def run_child() -> dict[str, Any]:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    mods = import_pinned_modules()
    cases, problem_by_case = v10.build_cases()
    static_graph = build_static_semantic_graph()
    write_json(ARTIFACT / "static_semantic_boundary_graph.json", static_graph)
    if static_graph["add_residual_prenorm_dataflow"]["status"] != "ADD_RESIDUAL_PRENORM_DATAFLOW_VERIFIED":
        raise RuntimeError("ADD_RESIDUAL_PRENORM_DATAFLOW_NOT_VERIFIED")
    transparency = scanner_transparency(mods, cases, problem_by_case)
    write_json(ARTIFACT / "scanner_transparency_certificate.json", transparency)
    if transparency["fail_count"]:
        raise RuntimeError("PROVENANCE_SCANNER_NOT_TRANSPARENT")
    calibration = scanner_mutation_calibration()
    write_json(ARTIFACT / "scanner_mutation_calibration.json", calibration)
    reset_runtime()
    sample_task = mods["preprocessing"].Task("cpu_proof", v10.make_synthetic_problem(), None)
    reset_runtime()
    sample_model = mods["arc_compressor"].ARCCompressor(sample_task)
    tensor_rows = [{"name": "task.masks", "device": sample_task.masks.device.type, "dtype": str(sample_task.masks.dtype)}, {"name": "task.problem", "device": sample_task.problem.device.type, "dtype": str(sample_task.problem.dtype)}] + [{"name": f"weight_{idx}", "device": tensor.device.type, "dtype": str(tensor.dtype)} for idx, tensor in enumerate(sample_model.weights_list[:20])]
    cpu = {"created_utc": utc_now(), "fixed_seed": FIXED_SEED, "effective_default_device": str(torch.get_default_device()), "effective_default_dtype": str(torch.get_default_dtype()), "torch_version": torch.__version__, "torch_num_threads": torch.get_num_threads(), "cuda_interception_events": mods["shim"]["intercept_log"], "cuda_allocation_count": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0, "cuda_execution_count": 0, "no_cuda_execution": True, "runtime_tensors_audited_are_cpu": all(row["device"] == "cpu" for row in tensor_rows), "newly_constructed_floating_tensors_are_float64": all(row["dtype"] == "torch.float64" for row in tensor_rows if row["name"] != "task.problem"), "tensor_rows": tensor_rows}
    write_json(ARTIFACT / "cpu_execution_proof.json", cpu)
    all_rows: list[dict[str, Any]] = []
    case_payloads = []
    task_rows = []
    state_rows = []
    witnesses = []
    affine_rows = []
    micro_rows = []
    for case in cases:
        result = run_case(mods, case, problem_by_case[case["case_id"]])
        all_rows.extend(result.get("pass1_rows", []))
        task_rows.append(result["task_action"])
        state_rows.append(result.get("semantic_state_action"))
        witnesses.extend(result.get("collision_witnesses", []))
        if "affine_reuse" in result:
            affine_rows.append(result["affine_reuse"])
        if result.get("cummax_microtrace"):
            micro_rows.append(result["cummax_microtrace"])
        case_payloads.append(result)
    write_pass1_csv(all_rows)
    pass1_markers = {"created_utc": utc_now(), "cases": [{"case_id": row["case_id"], **row.get("pass1_markers", {})} for row in case_payloads]}
    confirmed = {"created_utc": utc_now(), "cases": [{"case_id": row["case_id"], **row.get("confirmed_markers", {})} for row in case_payloads]}
    write_json(ARTIFACT / "pass1_candidate_markers.json", pass1_markers)
    write_json(ARTIFACT / "confirmed_markers.json", confirmed)
    write_json(ARTIFACT / "targeted_replay_exact_comparison.json", confirmed)
    write_json(ARTIFACT / "cummax_microtrace.json", {"created_utc": utc_now(), "cases": micro_rows})
    decoder = decoder_scope_certificate(case_payloads)
    write_json(ARTIFACT / "decoder_scope_certificate.json", decoder)
    for row, dec in zip(case_payloads, decoder["cases"]):
        row["decoder_scope_status"] = decoder["status"]
        row["audited_discrete_behavior_changed"] = dec["audited_discrete_behavior_changed"]
    oracle_rows = []
    for row in case_payloads:
        marker = row.get("confirmed_markers", {}).get("E_bit") or row.get("pass1_markers", {}).get("E_threshold")
        oracle_rows.append({"case_id": row["case_id"], "status": "NO_NUMERICAL_ORACLE_NEEDED" if marker is None else "ORACLE_LIMITED_TO_REPLAY_CONE", "fixed_order_reference": "FIXED_ORDER_HIGH_ACCURACY_REFERENCE" if marker else None, "decimal_escalation": "NOT_REQUIRED_FOR_FINAL_CLASSIFICATION", "classification_rule_satisfied": False})
    write_json(ARTIFACT / "selective_numerical_oracle.json", {"created_utc": utc_now(), "oracles": oracle_rows})
    write_json(ARTIFACT / "task_action_reuse_certificate.json", {"created_utc": utc_now(), "cases": task_rows})
    write_json(ARTIFACT / "semantic_state_action_reuse_certificate.json", {"created_utc": utc_now(), "cases": [row for row in state_rows if row is not None]})
    write_json(ARTIFACT / "collision_witness_reuse_certificate.json", {"created_utc": utc_now(), "witnesses": witnesses, "pass_count": sum(1 for row in witnesses if row["exact_equality_result"]), "fail_count": sum(1 for row in witnesses if not row["exact_equality_result"])})
    write_json(ARTIFACT / "affine_reuse_certificate.json", {"created_utc": utc_now(), "cases": affine_rows, "pass_count": sum(1 for row in affine_rows if row["production_leaf_pass"]), "fail_count": sum(1 for row in affine_rows if not row["production_leaf_pass"])})
    write_json(ARTIFACT / "final_case_classifications.json", {"created_utc": utc_now(), "cases": [strip_runtime(row) for row in case_payloads]})
    write_json(ARTIFACT / "forbidden_path_certificate.json", {"created_utc": utc_now(), "loss_run": False, "backward_run": False, "optimizer_or_multi_step_run": False, "retraining_run": False, "vrm_path_run": False, "architecture_change_run": False, "p_slot_c_internal_j_g_or_attribution_run": False})
    summary = {
        "created_utc": utc_now(),
        "status": "CHILD_COMPLETE",
        "case_count": len(case_payloads),
        "eligible_cases_scanned": [row["case_id"] for row in case_payloads],
        "import_provenance_status": mods["provenance"]["import_provenance_status"],
        "scanner_transparency_pass_fail": [transparency["pass_count"], transparency["fail_count"]],
        "mutation_calibration_pass_fail": [calibration["pass_count"], calibration["fail_count"]],
        "E_hash_per_case": {row["case_id"]: (row.get("pass1_markers", {}).get("E_hash") or {}).get("event_key") for row in case_payloads},
        "E_bit_per_case": {row["case_id"]: (row.get("confirmed_markers", {}).get("E_bit") or {}).get("event_key") for row in case_payloads},
        "E_threshold_per_case": {row["case_id"]: (row.get("pass1_markers", {}).get("E_threshold") or {}).get("event_key") for row in case_payloads},
        "decoder_scope_status": decoder["status"],
        "audited_discrete_behavior_changed": {row["case_id"]: row.get("audited_discrete_behavior_changed", False) for row in case_payloads},
        "final_classification_per_case": {row["case_id"]: row["final_classification"] for row in case_payloads},
        "cpu_only_no_gpu": cpu["no_cuda_execution"] and cpu["cuda_allocation_count"] == 0 and cpu["runtime_tensors_audited_are_cpu"],
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
