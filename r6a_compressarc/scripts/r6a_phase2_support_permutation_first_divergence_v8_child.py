from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, read_json, write_json


PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
DETAIL = ARTIFACT / "first_divergence_detail"
TASK_DIFF = ARTIFACT / "task_action_diff"
TARGET_TASKS = ["1f876c06", "604001fa"]
ABS_TOL = 1e-12
REL_TOL = 1e-10
REQUIRED_MODULES = ["preprocessing", "arc_compressor", "layers", "initializers", "multitensor_systems", "train"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def reset_determinism(seed: int = FIXED_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def seed_probe() -> dict[str, Any]:
    reset_determinism()
    probe = {
        "seed": FIXED_SEED,
        "python_random_first": random.random(),
        "numpy_random_first": float(np.random.random()),
        "torch_random_first": float(torch.rand(()).item()),
    }
    reset_determinism()
    return probe


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    item = tensor.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(item.shape)}|{item.dtype}".encode() + b"\0" + item.numpy().tobytes()).hexdigest()


def json_checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": value.device.type, "sha256": tensor_checksum(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def max_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int | None:
    ac = a.detach().cpu().contiguous()
    bc = b.detach().cpu().contiguous()
    if ac.dtype != torch.float64 or bc.dtype != torch.float64 or ac.shape != bc.shape:
        return None
    finite = torch.isfinite(ac) & torch.isfinite(bc)
    if not bool(finite.any().item()):
        return 0
    ai = ac[finite].view(torch.int64)
    bi = bc[finite].view(torch.int64)
    return int(torch.max(torch.abs(ai - bi)).item()) if ai.numel() else 0


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {
            "pass": False,
            "reason": "shape_mismatch",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "max_ulp": None,
            "over_tolerance_count": None,
            "over_tolerance_fraction": None,
            "canonical_checksum": tensor_checksum(a),
            "inverse_aligned_transformed_checksum": tensor_checksum(b),
        }
    ac = a.detach().cpu()
    bc = b.detach().cpu()
    if ac.dtype != bc.dtype:
        return {"pass": False, "reason": "dtype_mismatch", "dtype_a": str(ac.dtype), "dtype_b": str(bc.dtype)}
    if not torch.is_floating_point(ac):
        ok = torch.equal(ac, bc)
        return {
            "pass": bool(ok),
            "max_abs": 0.0 if ok else float("inf"),
            "max_rel": 0.0 if ok else float("inf"),
            "max_ulp": None,
            "over_tolerance_count": 0 if ok else int(ac.numel()),
            "over_tolerance_fraction": 0.0 if ok else 1.0,
            "canonical_checksum": tensor_checksum(a),
            "inverse_aligned_transformed_checksum": tensor_checksum(b),
        }
    finite_a = torch.isfinite(ac)
    finite_b = torch.isfinite(bc)
    if not torch.equal(finite_a, finite_b):
        return {
            "pass": False,
            "reason": "nonfinite_status_mismatch",
            "nonfinite_a": int((~finite_a).sum().item()),
            "nonfinite_b": int((~finite_b).sum().item()),
            "canonical_checksum": tensor_checksum(a),
            "inverse_aligned_transformed_checksum": tensor_checksum(b),
        }
    diff = torch.abs(ac - bc)
    denom = torch.maximum(torch.abs(ac), torch.abs(bc))
    allowed = ABS_TOL + REL_TOL * denom
    over = diff > allowed
    rel = diff / torch.maximum(denom, torch.tensor(1e-300, dtype=denom.dtype))
    return {
        "pass": bool(not over.any().item()),
        "max_abs": float(torch.max(diff).item()) if diff.numel() else 0.0,
        "max_rel": float(torch.max(rel).item()) if rel.numel() else 0.0,
        "max_ulp": max_ulp_distance(ac, bc),
        "over_tolerance_count": int(over.sum().item()),
        "over_tolerance_fraction": float(over.sum().item() / max(1, over.numel())),
        "canonical_checksum": tensor_checksum(a),
        "inverse_aligned_transformed_checksum": tensor_checksum(b),
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


def import_pinned_modules() -> dict[str, Any]:
    os.chdir(REFERENCE_ROOT)
    sys.path.insert(0, str(REFERENCE_ROOT))
    import_context = {
        "repository_root": str(REFERENCE_ROOT),
        "current_working_directory": os.getcwd(),
        "sys_path": list(sys.path),
        "sys_path_0": sys.path[0],
    }
    shim = install_cpu_import_shim()
    for name in REQUIRED_MODULES + ["solution_selection", "visualization"]:
        sys.modules.pop(name, None)
    import preprocessing
    import arc_compressor
    import layers
    import initializers
    import multitensor_systems
    import train
    import solution_selection

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    reset_determinism()
    mods = {
        "preprocessing": preprocessing,
        "arc_compressor": arc_compressor,
        "layers": layers,
        "initializers": initializers,
        "multitensor_systems": multitensor_systems,
        "train": train,
        "solution_selection": solution_selection,
    }
    provenance_rows = []
    all_match = True
    for name in REQUIRED_MODULES:
        module = mods[name]
        module_path = Path(module.__file__).resolve()
        rel = module_path.relative_to(REFERENCE_ROOT).as_posix() if module_path.is_relative_to(REFERENCE_ROOT) else None
        expected_bytes = b""
        expected_sha = None
        git_show_ok = False
        if rel is not None:
            proc = subprocess.run(
                ["git", "-C", str(REFERENCE_ROOT), "show", f"{PINNED_COMMIT}:{rel}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            git_show_ok = proc.returncode == 0
            expected_bytes = proc.stdout if git_show_ok else b""
            expected_sha = bytes_sha256(expected_bytes) if git_show_ok else None
        actual_sha = file_sha256(module_path)
        row = {
            "module_name": name,
            "module_file": module.__file__,
            "resolved_absolute_path": str(module_path),
            "actual_file_sha256": actual_sha,
            "path_under_intended_source_root": module_path.is_relative_to(REFERENCE_ROOT),
            "matching_relative_path_in_pinned_git_commit": rel,
            "expected_pinned_git_blob_sha256": expected_sha,
            "git_show_ok": git_show_ok,
            "matches_pinned_blob": git_show_ok and actual_sha == expected_sha,
        }
        all_match = all_match and row["path_under_intended_source_root"] and row["matches_pinned_blob"]
        provenance_rows.append(row)
    provenance = {
        "created_utc": utc_now(),
        "pinned_commit": PINNED_COMMIT,
        "import_context": import_context,
        "modules": provenance_rows,
        "import_provenance_status": "IMPORT_PROVENANCE_CERTIFIED" if all_match else "IMPORT_PROVENANCE_MISMATCH",
    }
    write_json(MANIFEST / "imported_module_provenance.json", provenance)
    if not all_match:
        raise RuntimeError("IMPORT_PROVENANCE_MISMATCH")
    return {**mods, "shim": shim, "seed_probe": seed_probe(), "import_context": import_context, "provenance": provenance}


def static_stochastic_inventory() -> dict[str, Any]:
    targets = ["arc_compressor.py", "layers.py", "multitensor_systems.py", "train.py"]
    calls = []
    for rel in targets:
        path = REFERENCE_ROOT / rel
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                text = ast.get_source_segment(source, node.func) or ""
                if text in {"torch.randn", "torch.rand", "torch.randn_like", "np.random.random", "np.random.randn", "numpy.random.random"} or "random" in text:
                    calls.append({"source_file": rel, "line": node.lineno, "callable": text, "source_line": lines[node.lineno - 1].strip()})
    represented = all(row["callable"] == "torch.randn" and row["source_file"] == "layers.py" for row in calls)
    return {
        "created_utc": utc_now(),
        "searched_files": targets,
        "stochastic_calls": calls,
        "runtime_wrappers": ["layers.torch.randn"],
        "all_static_calls_represented_by_runtime_schedule": represented,
    }


def support_transposition_perm(n_train: int, n_test: int, idx: int) -> list[int]:
    perm = list(range(n_train + n_test))
    perm[idx], perm[idx + 1] = perm[idx + 1], perm[idx]
    return perm


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


def make_synthetic_problem() -> dict[str, Any]:
    train = [
        {"input": [[0, 1, 2], [1, 2, 0], [2, 0, 1]], "output": [[1, 2, 0], [2, 0, 1], [0, 1, 2]]},
        {"input": [[2, 0, 1], [0, 1, 2], [1, 2, 0]], "output": [[0, 1, 2], [1, 2, 0], [2, 0, 1]]},
        {"input": [[1, 2, 0], [2, 0, 1], [0, 1, 2]], "output": [[2, 0, 1], [0, 1, 2], [1, 2, 0]]},
    ]
    test = [{"input": [[0, 2, 1], [1, 0, 2], [2, 1, 0]]}]
    return {"train": train, "test": test}


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


_OBJECT_REGISTRY: dict[int, torch.Tensor] = {}


def build_state_inventory(model: Any, task: Any, case_id: str, variant: str) -> list[dict[str, Any]]:
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
                    "case_id": case_id,
                    "variant": variant,
                    "object_id": oid,
                    "structural_paths": [],
                    "logical_dims": [],
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "device": obj.device.type,
                    "requires_grad": bool(obj.requires_grad),
                    "has_real_example_axis": has_real_example_axis,
                    "appears_in_weights_list": oid in weight_positions,
                    "optimizer_parameter_positions": weight_positions.get(oid, []),
                },
            )
            rec["structural_paths"].append(path)
            if dims is not None:
                rec["logical_dims"].append(list(dims))
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
    for alias_id, row in enumerate(records):
        row["alias_class_id"] = alias_id
    return records


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


def snapshot_records(records: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
    return {int(row["object_id"]): _OBJECT_REGISTRY[int(row["object_id"])].detach().clone() for row in records}


def restore_records(records: list[dict[str, Any]], snapshot: dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for row in records:
            target = _OBJECT_REGISTRY[int(row["object_id"])]
            target.copy_(snapshot[int(row["object_id"])].to(target.device, target.dtype))


def restore_transformed_from_canonical(canonical: list[dict[str, Any]], transformed: list[dict[str, Any]], snapshot: dict[int, torch.Tensor], perm_new_to_old: list[int]) -> None:
    if len(canonical) != len(transformed):
        raise RuntimeError("STATE_ACTION_REQUIRES_MANUAL_REVIEW")
    index = torch.tensor(perm_new_to_old, dtype=torch.long)
    with torch.no_grad():
        for a, b in zip(canonical, transformed):
            source = snapshot[int(a["object_id"])]
            value = source[index.to(source.device)] if b["has_real_example_axis"] else source
            target = _OBJECT_REGISTRY[int(b["object_id"])]
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError("STATE_ACTION_FAILED")
            target.copy_(value.to(target.device, target.dtype))


def compare_state(canonical: list[dict[str, Any]], transformed: list[dict[str, Any]], invperm_old_to_new: list[int]) -> dict[str, Any]:
    for a, b in zip(canonical, transformed):
        av = _OBJECT_REGISTRY[int(a["object_id"])]
        bv = _OBJECT_REGISTRY[int(b["object_id"])]
        if b["has_real_example_axis"]:
            bv = bv[torch.tensor(invperm_old_to_new, dtype=torch.long, device=bv.device)]
        comp = compare_tensors(av, bv)
        if not comp["pass"]:
            return {"pass": False, "first_failure": {"canonical_path": a["structural_paths"][0], "transformed_path": b["structural_paths"][0], **comp}}
    return {"pass": True, "first_failure": None}


def alias_signature(records: list[dict[str, Any]]) -> list[list[str]]:
    return [sorted(row["structural_paths"]) for row in records]


def task_action_certificate(mods: dict[str, Any], case: dict[str, Any], problem: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    preprocessing = mods["preprocessing"]
    perm = case["perm_new_to_old"]
    inv = case["invperm_old_to_new"]
    transformed = transformed_problem(problem, perm)
    reset_determinism()
    base_task = preprocessing.Task(case["task_id"], problem, None)
    reset_determinism()
    trans_task = preprocessing.Task(case["task_id"], transformed, None)
    role = ["train"] * base_task.n_train + ["test"] * base_task.n_test
    trans_role = ["train"] * trans_task.n_train + ["test"] * trans_task.n_test
    aligned_problem = trans_task.problem[torch.tensor(inv, dtype=torch.long)]
    aligned_masks = trans_task.masks[torch.tensor(inv, dtype=torch.long)]
    aligned_shapes = [trans_task.shapes[inv_idx] for inv_idx in inv]
    checks = {
        "n_train": base_task.n_train == trans_task.n_train,
        "n_test": base_task.n_test == trans_task.n_test,
        "raw_test_query_entries_byte_identical": problem["test"] == transformed["test"],
        "test_output_hidden_present_status": all(("output" in a) == ("output" in b) for a, b in zip(problem["test"], transformed["test"])),
        "colors": base_task.colors == trans_task.colors,
        "n_colors": base_task.n_colors == trans_task.n_colors,
        "n_x": base_task.n_x == trans_task.n_x,
        "n_y": base_task.n_y == trans_task.n_y,
        "shapes": base_task.shapes == aligned_shapes,
        "problem": torch.equal(base_task.problem, aligned_problem),
        "masks": torch.equal(base_task.masks, aligned_masks),
        "in_out_same_size": base_task.in_out_same_size == trans_task.in_out_same_size,
        "all_in_same_size": base_task.all_in_same_size == trans_task.all_in_same_size,
        "all_out_same_size": base_task.all_out_same_size == trans_task.all_out_same_size,
        "predicted_query_output_shape": [row[1] for row in base_task.shapes[base_task.n_train :]] == [row[1] for row in trans_task.shapes[trans_task.n_train :]],
        "support_pairing_after_inverse_alignment": all(transformed["train"][inv[old_idx]] == problem["train"][old_idx] for old_idx in range(base_task.n_train)),
        "train_test_membership": role == trans_role and all(role[old_idx] == trans_role[new_idx] for new_idx, old_idx in enumerate(perm)),
    }
    failures = [name for name, ok in checks.items() if not ok]
    cert = {
        "case_id": case["case_id"],
        "task_id": case["task_id"],
        "case_kind": case["case_kind"],
        "status": "TASK_ACTION_CERTIFIED" if not failures else "TASK_ACTION_FAILED",
        "checks": checks,
        "first_mismatched_field": failures[0] if failures else None,
        "canonical_problem_checksum": tensor_checksum(base_task.problem),
        "inverse_aligned_transformed_problem_checksum": tensor_checksum(aligned_problem),
        "canonical_masks_checksum": tensor_checksum(base_task.masks),
        "inverse_aligned_transformed_masks_checksum": tensor_checksum(aligned_masks),
        "mismatch_occurs_before_model_construction": bool(failures),
        "construction_function_path": "preprocessing.Task",
    }
    if failures:
        write_json(TASK_DIFF / f"{case['case_id']}.json", cert)
    return base_task, trans_task, cert


def posterior_leaf_schedule(model: Any, task: Any, case_id: str) -> list[dict[str, Any]]:
    rows = []
    for ordinal, dims in enumerate(list(model.multitensor_system)):
        posterior = model.multiposteriors[dims]
        mean = posterior[0]
        local_capacity = posterior[1]
        target_capacity = model.target_capacities[dims]
        rows.append(
            {
                "case_id": case_id,
                "ordinal": ordinal,
                "dims": list(dims),
                "posterior_structural_path": f"multiposteriors[{tuple(dims)}]",
                "mean_object_id": id(mean),
                "local_capacity_object_id": id(local_capacity),
                "target_capacity_object_id": id(target_capacity),
                "mean_shape": list(mean.shape),
                "true_example_axis": bool(dims[0] == 1 and mean.ndim > 0 and mean.shape[0] == task.n_examples),
            }
        )
    return rows


def tensor_rows(obj: Any, task: Any, root: str, dims_hint: list[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(path: str, tensor: torch.Tensor, dims: list[int] | None) -> None:
        strict_example_axis = bool(dims and dims[0] == 1 and tensor.ndim > 0 and tensor.shape[0] == task.n_examples)
        boundary_alignment_axis = bool(strict_example_axis or (dims is None and tensor.ndim > 0 and tensor.shape[0] == task.n_examples))
        rows.append(
            {
                "path": path,
                "dims": dims,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": tensor.device.type,
                "finite": bool(torch.isfinite(tensor.detach()).all().item()) if torch.is_floating_point(tensor) else True,
                "true_example_axis": strict_example_axis,
                "align_as_example_axis": boundary_alignment_axis,
                "alignment_basis": "strict_dims_and_shape" if strict_example_axis else ("boundary_tensor_leading_example_axis" if boundary_alignment_axis else "not_example_aligned"),
                "tensor": tensor.detach().clone(),
                "checksum": tensor_checksum(tensor),
            }
        )

    def visit(item: Any, path: str, dims: list[int] | None = None) -> None:
        if torch.is_tensor(item):
            add(path, item, dims)
            return
        if is_multitensor(item):
            for dims_item in item.multitensor_system:
                visit(item[dims_item], f"{path}[{tuple(dims_item)}]", list(dims_item))
            return
        if isinstance(item, (list, tuple)):
            for idx, sub in enumerate(item):
                visit(sub, f"{path}/{idx}", dims)
            return
        if isinstance(item, dict):
            for key, sub in item.items():
                visit(sub, f"{path}/{key}", dims)

    visit(obj, root, dims_hint)
    return rows


class BoundaryRecorder:
    def __init__(self, mods: dict[str, Any], task: Any):
        self.mods = mods
        self.layers = mods["layers"]
        self.task = task
        self.records: list[dict[str, Any]] = []
        self.originals: dict[str, Any] = {}
        self.depth = 0
        self.names = ["decode_latents", "share_up", "share_direction", "softmax", "cummax", "shift", "direction_share", "nonlinear", "share_down", "normalize", "affine", "postprocess_mask"]

    def __enter__(self):
        for name in self.names:
            original = getattr(self.layers, name)
            self.originals[name] = original

            def make_wrapper(fn, fname):
                def wrapper(*args, **kwargs):
                    call_index = len(self.records)
                    self.depth += 1
                    try:
                        result = fn(*args, **kwargs)
                    finally:
                        self.depth -= 1
                    direction = None
                    if fname == "share_direction" and len(args) >= 3:
                        direction = args[2]
                    self.records.append(
                        {
                            "event_index": call_index,
                            "event_name": fname,
                            "call_nesting": self.depth,
                            "direction": direction,
                            "outputs": [{k: v for k, v in row.items() if k != "tensor"} for row in tensor_rows(result, self.task, fname)],
                            "_outputs_with_tensors": tensor_rows(result, self.task, fname),
                        }
                    )
                    return result

                return wrapper

            setattr(self.layers, name, make_wrapper(original, name))
        return self

    def record_channel(self, event_name: str, ordinal: int, dims: list[int], result: Any) -> None:
        self.records.append(
            {
                "event_index": len(self.records),
                "event_name": event_name,
                "posterior_leaf_ordinal": ordinal,
                "dims": dims,
                "call_nesting": self.depth,
                "direction": None,
                "outputs": [{k: v for k, v in row.items() if k != "tensor"} for row in tensor_rows(result, self.task, event_name, dims)],
                "_outputs_with_tensors": tensor_rows(result, self.task, event_name, dims),
            }
        )

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            setattr(self.layers, name, original)
        return False


class NoiseController:
    def __init__(
        self,
        mods: dict[str, Any],
        task: Any,
        case_id: str,
        schedule: list[dict[str, Any]],
        mode: str,
        perm_new_to_old: list[int],
        stored: list[torch.Tensor] | None = None,
        transform_for_permuted: bool = False,
        boundary_recorder: BoundaryRecorder | None = None,
    ):
        self.mods = mods
        self.layers = mods["layers"]
        self.task = task
        self.case_id = case_id
        self.schedule = schedule
        self.mode = mode
        self.perm_new_to_old = perm_new_to_old
        self.stored = stored or []
        self.transform_for_permuted = transform_for_permuted
        self.boundary_recorder = boundary_recorder
        self.original_channel = self.layers.channel_layer
        self.original_randn = self.layers.torch.randn
        self.active_leaf: dict[str, Any] | None = None
        self.cursor = 0
        self.randn_seen_for_active = False
        self.trace: list[dict[str, Any]] = []
        self.unhandled: list[dict[str, Any]] = []
        self.recorded: list[torch.Tensor] = []

    def __enter__(self):
        def channel_wrapper(target_capacity, posterior):
            if self.cursor >= len(self.schedule):
                self.unhandled.append({"type": "channel_after_schedule_exhaustion", "cursor": self.cursor})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            expected = self.schedule[self.cursor]
            mean, local_capacity = posterior
            checks = {
                "mean_object_id": id(mean) == expected["mean_object_id"],
                "local_capacity_object_id": id(local_capacity) == expected["local_capacity_object_id"],
                "target_capacity_object_id": id(target_capacity) == expected["target_capacity_object_id"],
            }
            if not all(checks.values()):
                self.unhandled.append({"type": "posterior_leaf_identity_mismatch", "cursor": self.cursor, "checks": checks, "expected": expected})
                raise RuntimeError("TRACE_SCHEDULE_MISMATCH")
            self.active_leaf = expected
            self.randn_seen_for_active = False
            result = self.original_channel(target_capacity, posterior)
            if not self.randn_seen_for_active:
                self.unhandled.append({"type": "skipped_noise_request", "cursor": self.cursor, "expected": expected})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            if self.boundary_recorder is not None:
                self.boundary_recorder.record_channel("channel_layer", expected["ordinal"], expected["dims"], result)
            self.trace.append({"event": "channel_layer", "case_id": self.case_id, "ordinal": expected["ordinal"], "dims": expected["dims"], "mode": self.mode})
            self.cursor += 1
            self.active_leaf = None
            return result

        def randn_wrapper(*args, **kwargs):
            if self.active_leaf is None:
                self.unhandled.append({"type": "randn_without_active_leaf"})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            if self.randn_seen_for_active:
                self.unhandled.append({"type": "duplicate_noise_request", "active_leaf": self.active_leaf})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            self.randn_seen_for_active = True
            expected_shape = list(self.active_leaf["mean_shape"])
            requested_shape = list(args[0]) if args else []
            if requested_shape != expected_shape:
                self.unhandled.append({"type": "unexpected_requested_shape", "requested": requested_shape, "expected": expected_shape})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            dtype = torch.float64
            device = torch.device("cpu")
            if self.mode == "record":
                epsilon = self.original_randn(*args, **kwargs)
                if tuple(epsilon.shape) != tuple(expected_shape) or str(epsilon.dtype) != "torch.float64" or epsilon.device.type != "cpu":
                    self.unhandled.append({"type": "record_epsilon_dtype_device_shape_mismatch"})
                    raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
                self.recorded.append(epsilon.detach().clone())
            elif self.mode == "replay":
                epsilon = self.stored[self.active_leaf["ordinal"]].detach().clone().to(device=device, dtype=dtype)
                if self.transform_for_permuted and self.active_leaf["true_example_axis"]:
                    epsilon = epsilon[torch.tensor(self.perm_new_to_old, dtype=torch.long, device=epsilon.device)]
            elif self.mode == "zero":
                epsilon = torch.zeros(tuple(expected_shape), dtype=dtype, device=device)
            else:
                raise ValueError(self.mode)
            if tuple(epsilon.shape) != tuple(expected_shape) or str(epsilon.dtype) != "torch.float64" or epsilon.device.type != "cpu":
                self.unhandled.append({"type": "returned_epsilon_dtype_device_shape_mismatch"})
                raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
            self.trace.append(
                {
                    "event": "randn",
                    "case_id": self.case_id,
                    "ordinal": self.active_leaf["ordinal"],
                    "dims": self.active_leaf["dims"],
                    "mode": self.mode,
                    "requested_shape": requested_shape,
                    "dtype": str(epsilon.dtype),
                    "device": epsilon.device.type,
                    "true_example_axis": self.active_leaf["true_example_axis"],
                    "checksum": tensor_checksum(epsilon),
                }
            )
            return epsilon

        self.layers.channel_layer = channel_wrapper
        self.layers.torch.randn = randn_wrapper
        return self

    def __exit__(self, exc_type, exc, tb):
        self.layers.channel_layer = self.original_channel
        self.layers.torch.randn = self.original_randn
        if exc_type is None and self.cursor != len(self.schedule):
            self.unhandled.append({"type": "missing_channel_calls", "consumed": self.cursor, "expected": len(self.schedule)})
            raise RuntimeError("STOCHASTIC_TRACE_INCOMPLETE")
        return False


def add_black_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(logits[:, :1, :, :]), logits], dim=1)


def predicted_grid_status(mods: dict[str, Any], task: Any, logits: torch.Tensor, x_mask: torch.Tensor, y_mask: torch.Tensor) -> dict[str, Any]:
    logger = mods["solution_selection"].Logger(task)
    try:
        solution, uncertainty = logger._postprocess_solution(add_black_logits(logits)[task.n_train :, :, :, :, 1], x_mask[task.n_train :, :, 1], y_mask[task.n_train :, 1])
        return {"status": "OK", "solution": solution, "uncertainty": float(uncertainty)}
    except Exception as exc:
        return {"status": "POSTPROCESS_EXCEPTION", "error_type": type(exc).__name__, "message": str(exc)}


def run_forward(
    mods: dict[str, Any],
    model: Any,
    task: Any,
    case_id: str,
    mode: str,
    schedule: list[dict[str, Any]],
    perm_new_to_old: list[int],
    stored: list[torch.Tensor] | None = None,
    transform_noise: bool = False,
    observe: bool = False,
) -> dict[str, Any]:
    boundary = BoundaryRecorder(mods, task) if observe else None
    noise = None if mode == "source" else NoiseController(mods, task, case_id, schedule, mode, perm_new_to_old, stored=stored, transform_for_permuted=transform_noise, boundary_recorder=boundary)
    if boundary is not None:
        boundary.__enter__()
    if noise is not None:
        noise.__enter__()
    try:
        logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    finally:
        if noise is not None:
            noise.__exit__(None, None, None)
        if boundary is not None:
            boundary.__exit__(None, None, None)
    readout = {
        "logits": logits.detach().clone(),
        "x_mask": x_mask.detach().clone(),
        "y_mask": y_mask.detach().clone(),
        "kl_amounts": [item.detach().clone() for item in kl_amounts],
        "kl_names": list(kl_names),
        "prediction": predicted_grid_status(mods, task, logits.detach(), x_mask.detach(), y_mask.detach()),
        "finite_status": {
            "logits": bool(torch.isfinite(logits).all().item()),
            "x_mask": bool(torch.isfinite(x_mask).all().item()),
            "y_mask": bool(torch.isfinite(y_mask).all().item()),
            "kl_amounts": [bool(torch.isfinite(item).all().item()) for item in kl_amounts],
        },
        "noise_trace": [] if noise is None else noise.trace,
        "unhandled_noise_calls": [] if noise is None else noise.unhandled,
        "stored_noise": [] if noise is None else noise.recorded,
        "boundary_events": [] if boundary is None else boundary.records,
    }
    return readout


def align_tensor_for_case(tensor: torch.Tensor, invperm_old_to_new: list[int]) -> torch.Tensor:
    if tensor.ndim > 0 and tensor.shape[0] == len(invperm_old_to_new):
        return tensor[torch.tensor(invperm_old_to_new, dtype=torch.long, device=tensor.device)]
    return tensor


def compare_forward_readouts(a: dict[str, Any], b: dict[str, Any], invperm_old_to_new: list[int]) -> dict[str, Any]:
    checks: list[tuple[str, dict[str, Any]]] = []
    checks.append(("logits", compare_tensors(a["logits"], align_tensor_for_case(b["logits"], invperm_old_to_new))))
    checks.append(("x_mask", compare_tensors(a["x_mask"], align_tensor_for_case(b["x_mask"], invperm_old_to_new))))
    checks.append(("y_mask", compare_tensors(a["y_mask"], align_tensor_for_case(b["y_mask"], invperm_old_to_new))))
    if a["kl_names"] != b["kl_names"] or len(a["kl_amounts"]) != len(b["kl_amounts"]):
        return {"pass": False, "first_failure": {"object": "kl_schedule", "canonical": a["kl_names"], "transformed": b["kl_names"]}}
    for idx, (ka, kb) in enumerate(zip(a["kl_amounts"], b["kl_amounts"])):
        checks.append((f"KL_amounts/{idx}", compare_tensors(ka, kb)))
    if a["prediction"] != b["prediction"]:
        return {"pass": False, "first_failure": {"object": "prediction", "reason": "discrete_prediction_or_status_mismatch", "canonical": a["prediction"], "transformed": b["prediction"]}}
    if a["finite_status"] != b["finite_status"]:
        return {"pass": False, "first_failure": {"object": "finite_status", "canonical": a["finite_status"], "transformed": b["finite_status"]}}
    for name, comp in checks:
        if not comp.get("pass"):
            return {"pass": False, "first_failure": {"first_failing_object": name, "source_function_or_module": "ARCCompressor.forward", **comp}}
    return {"pass": True, "first_failure": None}


def compare_boundary_events(a: list[dict[str, Any]], b: list[dict[str, Any]], invperm_old_to_new: list[int]) -> dict[str, Any]:
    if len(a) != len(b):
        return {"pass": False, "label": "TRACE_SCHEDULE_MISMATCH", "first_failure": {"reason": "event_count_mismatch", "canonical": len(a), "transformed": len(b)}}
    trace_rows = []
    for idx, (ea, eb) in enumerate(zip(a, b)):
        schedule_keys = ["event_name", "call_nesting", "direction", "posterior_leaf_ordinal", "dims"]
        for key in schedule_keys:
            if ea.get(key) != eb.get(key):
                return {"pass": False, "label": "TRACE_SCHEDULE_MISMATCH", "first_failure": {"event_index": idx, "key": key, "canonical": ea.get(key), "transformed": eb.get(key)}}
        rows_a = ea["_outputs_with_tensors"]
        rows_b = eb["_outputs_with_tensors"]
        if len(rows_a) != len(rows_b):
            return {"pass": False, "label": "TRACE_SCHEDULE_MISMATCH", "first_failure": {"event_index": idx, "reason": "output_count_mismatch"}}
        for out_idx, (ra, rb) in enumerate(zip(rows_a, rows_b)):
            tb = rb["tensor"]
            if rb.get("align_as_example_axis"):
                tb = tb[torch.tensor(invperm_old_to_new, dtype=torch.long, device=tb.device)]
            comp = compare_tensors(ra["tensor"], tb)
            trace_rows.append(
                {
                    "event_index": idx,
                    "event_name": ea["event_name"],
                    "output_index": out_idx,
                    "output_path": ra["path"],
                    "pass": comp.get("pass"),
                    "max_abs": comp.get("max_abs"),
                    "max_rel": comp.get("max_rel"),
                    "max_ulp": comp.get("max_ulp"),
                    "over_tolerance_count": comp.get("over_tolerance_count"),
                }
            )
            if not comp.get("pass"):
                label = label_for_boundary(ea["event_name"], ea.get("direction"))
                return {
                    "pass": False,
                    "label": label,
                    "first_failure": {
                        "event_index": idx,
                        "event_name": ea["event_name"],
                        "output_index": out_idx,
                        "output_path": ra["path"],
                        "input_certified_matched_by_prior_boundaries": True,
                        **comp,
                    },
                    "trace_rows": trace_rows,
                }
    return {"pass": True, "label": "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED", "first_failure": None, "trace_rows": trace_rows}


def label_for_boundary(name: str, direction: Any = None) -> str:
    mapping = {
        "channel_layer": "FIRST_DIVERGENCE_CHANNEL_LAYER",
        "decode_latents": "FIRST_DIVERGENCE_DECODE_LATENTS",
        "softmax": "FIRST_DIVERGENCE_SOFTMAX",
        "cummax": "FIRST_DIVERGENCE_CUMMAX",
        "shift": "FIRST_DIVERGENCE_SHIFT",
        "direction_share": "FIRST_DIVERGENCE_DIRECTION_SHARE",
        "nonlinear": "FIRST_DIVERGENCE_NONLINEAR",
        "normalize": "FIRST_DIVERGENCE_NORMALIZE",
        "affine": "FIRST_DIVERGENCE_AFFINE",
        "postprocess_mask": "FIRST_DIVERGENCE_POSTPROCESS",
        "share_up": "FIRST_DIVERGENCE_SHARE_DIRECTION_UP",
        "share_down": "FIRST_DIVERGENCE_SHARE_DIRECTION_DOWN",
    }
    if name == "share_direction":
        return "FIRST_DIVERGENCE_SHARE_DIRECTION_UP" if direction == 1 else "FIRST_DIVERGENCE_SHARE_DIRECTION_DOWN"
    return mapping.get(name, "FIRST_DIVERGENCE_LOCALIZED_TO_SOURCE_PRIMITIVE_REQUIRES_MANUAL_REVIEW")


@contextmanager
def channel_line_trace(mods: dict[str, Any], task: Any):
    layers = mods["layers"]
    code = layers.channel_layer.__code__
    selected = {"mean", "local_capacity_adjustment", "target_capacity", "normalized_mean", "noise_std", "signal_std", "z", "KL"}
    rows: list[dict[str, Any]] = []

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code is code:
            for name in selected:
                if name in frame.f_locals and torch.is_tensor(frame.f_locals[name]):
                    value = frame.f_locals[name]
                    dims = None
                    true_axis = bool(value.ndim > 0 and value.shape[0] == task.n_examples)
                    rows.append(
                        {
                            "line": frame.f_lineno,
                            "variable": name,
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "device": value.device.type,
                            "true_example_axis": true_axis,
                            "tensor": value.detach().clone(),
                            "checksum": tensor_checksum(value),
                        }
                    )
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        yield rows
    finally:
        sys.settrace(old)


def run_tier2_channel_trace(mods: dict[str, Any], base_model: Any, trans_model: Any, base_task: Any, trans_task: Any, case: dict[str, Any], schedule: list[dict[str, Any]], base_records: list[dict[str, Any]], trans_records: list[dict[str, Any]], snapshot: dict[int, torch.Tensor]) -> dict[str, Any]:
    inv = case["invperm_old_to_new"]
    perm = case["perm_new_to_old"]
    restore_records(base_records, snapshot)
    read_observer = run_forward(mods, base_model, base_task, case["case_id"], "zero", schedule, perm, observe=True)
    restore_records(base_records, snapshot)
    with channel_line_trace(mods, base_task) as line_rows_a:
        read_trace = run_forward(mods, base_model, base_task, case["case_id"], "zero", schedule, perm, observe=True)
    transparency = compare_forward_readouts(read_observer, read_trace, list(range(base_task.n_examples)))
    if not transparency["pass"]:
        return {"status": "LINE_TRACE_NOT_TRANSPARENT", "transparency": transparency}
    restore_records(base_records, snapshot)
    with channel_line_trace(mods, base_task) as rows_a:
        run_forward(mods, base_model, base_task, case["case_id"], "zero", schedule, perm, observe=False)
    restore_transformed_from_canonical(base_records, trans_records, snapshot, perm)
    trans_schedule = [
        {**row, "mean_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][0]), "local_capacity_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][1]), "target_capacity_object_id": id(trans_model.target_capacities[tuple(row["dims"])])}
        for row in schedule
    ]
    with channel_line_trace(mods, trans_task) as rows_b:
        run_forward(mods, trans_model, trans_task, case["case_id"], "zero", trans_schedule, perm, transform_noise=True, observe=False)
    detail_rows = []
    first = None
    for idx, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        if ra["variable"] != rb["variable"]:
            first = {"reason": "line_trace_variable_schedule_mismatch", "index": idx}
            break
        tb = rb["tensor"]
        if rb["true_example_axis"]:
            tb = tb[torch.tensor(inv, dtype=torch.long, device=tb.device)]
        comp = compare_tensors(ra["tensor"], tb)
        detail = {"index": idx, "line": ra["line"], "variable": ra["variable"], **{k: v for k, v in comp.items() if k != "pass"}, "pass": comp.get("pass")}
        detail_rows.append(detail)
        if not comp.get("pass") and first is None:
            first = detail
            break
    payload = {
        "status": "TIER2_CHANNEL_LAYER_TRACE_COMPLETED",
        "transparency": transparency,
        "first_local_variable_divergence": first,
        "line_trace_row_count_canonical": len(rows_a),
        "line_trace_row_count_transformed": len(rows_b),
        "detail_rows_prefix": detail_rows[:100],
    }
    write_json(DETAIL / f"{case['case_id']}_tier2_channel_layer.json", payload)
    return payload


def apply_synthetic_fingerprints(model: Any, task: Any) -> dict[str, Any]:
    touched = []
    with torch.no_grad():
        for dims in model.multitensor_system:
            if dims[0] != 1:
                continue
            mean, local_capacity = model.multiposteriors[dims]
            if mean.ndim == 0 or mean.shape[0] != task.n_examples:
                continue
            pattern = torch.linspace(-1.5, 1.5, task.n_examples, dtype=mean.dtype, device=mean.device)
            while pattern.ndim < mean.ndim:
                pattern = pattern.unsqueeze(-1)
            mean.copy_(pattern.expand_as(mean) * 1e-4)
            local_capacity.copy_(pattern.expand_as(local_capacity) * 1e-5)
            touched.append({"dims": list(dims), "mean_checksum": tensor_checksum(mean), "local_capacity_checksum": tensor_checksum(local_capacity)})
    finite = all(torch.isfinite(model.multiposteriors[dims][0]).all().item() and torch.isfinite(model.multiposteriors[dims][1]).all().item() for dims in model.multitensor_system)
    return {"fingerprinted_leaf_count": len(touched), "touched": touched, "all_fingerprinted_tensors_finite": bool(finite)}


def optimizer_partition(model: Any, records: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    reachable_ids = {int(row["object_id"]) for row in records if row["appears_in_weights_list"]}
    record_by_id = {int(row["object_id"]): row for row in records}
    forward_reachable = []
    stale = []
    unknown = []
    for idx, param in enumerate(model.weights_list):
        row = {
            "optimizer_position": idx,
            "object_id": id(param),
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "device": param.device.type,
            "requires_grad": bool(param.requires_grad),
            "gradient_probe": "NOT_EXECUTED_UNLESS_FORWARD_CERTIFIED",
        }
        if id(param) in reachable_ids:
            forward_reachable.append({**row, "alias_class_id": record_by_id[id(param)]["alias_class_id"]})
        elif torch.is_tensor(param):
            stale.append(row)
        else:
            unknown.append(row)
    return {
        "case_id": case_id,
        "forward_reachable_optimizer_parameters": forward_reachable,
        "stale_or_unreachable_optimizer_parameters": stale,
        "unknown_optimizer_parameters": unknown,
        "stale_gradient_probe_status": "BLOCKED_UNTIL_FORWARD_CERTIFIED",
        "stale_optimizer_parameter_active": False,
    }


class CaptureNoOpLogger:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def log(self, train_step, logits, x_mask, y_mask, KL_amounts, KL_names, total_KL, reconstruction_error, loss):
        self.events.append(
            {
                "train_step": int(train_step),
                "logits": logits.detach().clone(),
                "x_mask": x_mask.detach().clone(),
                "y_mask": y_mask.detach().clone(),
                "KL_amounts": [item.detach().clone() for item in KL_amounts],
                "KL_names": list(KL_names),
                "total_KL": total_KL.detach().clone(),
                "reconstruction_error": reconstruction_error.detach().clone(),
                "loss": loss.detach().clone(),
            }
        )


class CaptureAdamProxy:
    def __init__(self, optimizer: torch.optim.Optimizer, mode: str, records: list[dict[str, Any]]):
        self.optimizer = optimizer
        self.mode = mode
        self.records = records
        self.step_called = 0
        self.gradient_snapshot: dict[int, torch.Tensor | None] = {}
        self.state_after_step: dict[int, dict[str, Any]] = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def step(self):
        self.step_called += 1
        for row in self.records:
            obj = _OBJECT_REGISTRY[int(row["object_id"])]
            self.gradient_snapshot[int(row["object_id"])] = None if obj.grad is None else obj.grad.detach().clone()
        if self.mode == "one_step":
            out = self.optimizer.step()
            for row in self.records:
                obj = _OBJECT_REGISTRY[int(row["object_id"])]
                state = self.optimizer.state.get(obj, {})
                self.state_after_step[int(row["object_id"])] = {
                    key: value.detach().clone() if torch.is_tensor(value) else value for key, value in state.items()
                }
            return out
        return None


def run_take_step_certificate(mods: dict[str, Any], base_model: Any, trans_model: Any, base_task: Any, trans_task: Any, case: dict[str, Any], schedule: list[dict[str, Any]], base_records: list[dict[str, Any]], trans_records: list[dict[str, Any]], snapshot: dict[int, torch.Tensor], mode: str) -> dict[str, Any]:
    train = mods["train"]
    perm = case["perm_new_to_old"]
    inv = case["invperm_old_to_new"]
    restore_records(base_records, snapshot)
    restore_transformed_from_canonical(base_records, trans_records, snapshot, perm)
    opt_a = torch.optim.Adam(base_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    opt_b = torch.optim.Adam(trans_model.weights_list, lr=0.01, betas=(0.5, 0.9))
    proxy_a = CaptureAdamProxy(opt_a, mode, base_records)
    proxy_b = CaptureAdamProxy(opt_b, mode, trans_records)
    logger_a = CaptureNoOpLogger()
    logger_b = CaptureNoOpLogger()
    with NoiseController(mods, base_task, case["case_id"], schedule, "record", perm) as tape_a:
        train.take_step(base_task, base_model, proxy_a, 0, logger_a)
    trans_schedule = [
        {**row, "mean_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][0]), "local_capacity_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][1]), "target_capacity_object_id": id(trans_model.target_capacities[tuple(row["dims"])])}
        for row in schedule
    ]
    with NoiseController(mods, trans_task, case["case_id"], trans_schedule, "replay", perm, stored=tape_a.recorded, transform_for_permuted=True) as tape_b:
        train.take_step(trans_task, trans_model, proxy_b, 0, logger_b)
    if len(logger_a.events) != 1 or len(logger_b.events) != 1:
        return {"status": "LOGGER_EVENT_COUNT_FAILED", "logger_event_count_canonical": len(logger_a.events), "logger_event_count_transformed": len(logger_b.events)}
    event_a, event_b = logger_a.events[0], logger_b.events[0]
    loss_checks = {
        "total_KL": compare_tensors(event_a["total_KL"], event_b["total_KL"]),
        "reconstruction_error": compare_tensors(event_a["reconstruction_error"], event_b["reconstruction_error"]),
        "loss": compare_tensors(event_a["loss"], event_b["loss"]),
    }
    grad_fail = None
    for row_a, row_b in zip(base_records, trans_records):
        ga = proxy_a.gradient_snapshot.get(int(row_a["object_id"]))
        gb = proxy_b.gradient_snapshot.get(int(row_b["object_id"]))
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            grad_fail = {"path": row_a["structural_paths"][0], "reason": "gradient_presence_mismatch"}
            break
        if row_b["has_real_example_axis"]:
            gb = gb[torch.tensor(inv, dtype=torch.long, device=gb.device)]
        comp = compare_tensors(ga, gb)
        if not comp["pass"]:
            grad_fail = {"path": row_a["structural_paths"][0], **comp}
            break
    adam_fail = None
    if mode == "one_step":
        for row_a, row_b in zip(base_records, trans_records):
            sa = proxy_a.state_after_step.get(int(row_a["object_id"]), {})
            sb = proxy_b.state_after_step.get(int(row_b["object_id"]), {})
            for key in sorted(set(sa) | set(sb)):
                va, vb = sa.get(key), sb.get(key)
                if torch.is_tensor(va) and torch.is_tensor(vb):
                    if row_b["has_real_example_axis"] and vb.ndim > 0 and vb.shape[0] == len(inv):
                        vb = vb[torch.tensor(inv, dtype=torch.long, device=vb.device)]
                    comp = compare_tensors(va, vb)
                    if not comp["pass"]:
                        adam_fail = {"path": row_a["structural_paths"][0], "state_key": key, **comp}
                        break
                elif va != vb:
                    adam_fail = {"path": row_a["structural_paths"][0], "state_key": key, "reason": "non_tensor_state_mismatch"}
                    break
            if adam_fail:
                break
    return {
        "status": "TAKE_STEP_CERTIFIED" if all(v["pass"] for v in loss_checks.values()) and grad_fail is None and adam_fail is None else "TAKE_STEP_FAILED",
        "logger_event_count": 1,
        "noise_trace_complete": len(tape_a.unhandled) == 0 and len(tape_b.unhandled) == 0,
        "loss_checks": loss_checks,
        "gradient_first_failure": grad_fail,
        "adam_first_failure": adam_fail,
        "mode": mode,
    }


def run_case(mods: dict[str, Any], case: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    arc_compressor = mods["arc_compressor"]
    _OBJECT_REGISTRY.clear()
    base_task, trans_task, task_cert = task_action_certificate(mods, case, problem)
    result: dict[str, Any] = {"case": case, "task_action": task_cert, "final_label": None}
    if task_cert["status"] != "TASK_ACTION_CERTIFIED":
        result["final_label"] = "TASK_ACTION_FAILED"
        return result
    reset_determinism()
    base_model = arc_compressor.ARCCompressor(base_task)
    reset_determinism()
    trans_model = arc_compressor.ARCCompressor(trans_task)
    base_records = build_state_inventory(base_model, base_task, case["case_id"], "canonical")
    trans_records = build_state_inventory(trans_model, trans_task, case["case_id"], "transformed")
    register_objects(base_model, base_records)
    register_objects(trans_model, trans_records)
    alias_equal = alias_signature(base_records) == alias_signature(trans_records)
    if not alias_equal:
        result["alias_topology"] = {"status": "ALIAS_TOPOLOGY_CONSTRUCTION_MISMATCH"}
        result["final_label"] = "ALIAS_TOPOLOGY_CONSTRUCTION_MISMATCH"
        result["_inventory"] = base_records + trans_records
        return result
    if case["case_kind"] == "synthetic_oracle":
        result["synthetic_fingerprint"] = apply_synthetic_fingerprints(base_model, base_task)
        if not result["synthetic_fingerprint"]["all_fingerprinted_tensors_finite"]:
            result["final_label"] = "SYNTHETIC_FINGERPRINT_NONFINITE"
            result["_inventory"] = base_records + trans_records
            return result
    snapshot = snapshot_records(base_records)
    restore_transformed_from_canonical(base_records, trans_records, snapshot, case["perm_new_to_old"])
    state_cmp = compare_state(base_records, trans_records, case["invperm_old_to_new"])
    result["state_action"] = {
        "case_id": case["case_id"],
        "status": "STATE_ACTION_CERTIFIED" if state_cmp["pass"] else "STATE_ACTION_FAILED",
        "alias_topology_status": "ALIAS_TOPOLOGY_CERTIFIED",
        "state_compare": state_cmp,
        "canonical_unique_forward_reachable_tensors": len(base_records),
        "transformed_unique_forward_reachable_tensors": len(trans_records),
        "example_axis_tensor_count": sum(1 for row in base_records if row["has_real_example_axis"]),
        "non_example_axis_tensor_count": sum(1 for row in base_records if not row["has_real_example_axis"]),
        "restore_used_in_place_copy": True,
        "rebinding_used": False,
    }
    result["optimizer_partition"] = optimizer_partition(base_model, base_records, case["case_id"])
    result["_inventory"] = base_records + trans_records
    if not state_cmp["pass"]:
        result["final_label"] = "STATE_ACTION_FAILED"
        return result
    schedule = posterior_leaf_schedule(base_model, base_task, case["case_id"])
    result["posterior_leaf_schedule"] = schedule
    perm = case["perm_new_to_old"]
    rng_state = torch.get_rng_state().clone()
    restore_records(base_records, snapshot)
    torch.set_rng_state(rng_state.clone())
    source = run_forward(mods, base_model, base_task, case["case_id"], "source", schedule, perm)
    restore_records(base_records, snapshot)
    torch.set_rng_state(rng_state.clone())
    record = run_forward(mods, base_model, base_task, case["case_id"], "record", schedule, perm)
    restore_records(base_records, snapshot)
    torch.set_rng_state(rng_state.clone())
    replay = run_forward(mods, base_model, base_task, case["case_id"], "replay", schedule, perm, stored=record["stored_noise"])
    restore_records(base_records, snapshot)
    torch.set_rng_state(rng_state.clone())
    observed = run_forward(mods, base_model, base_task, case["case_id"], "replay", schedule, perm, stored=record["stored_noise"], observe=True)
    a0_source_record = compare_forward_readouts(source, record, list(range(base_task.n_examples)))
    a0_record_replay = compare_forward_readouts(record, replay, list(range(base_task.n_examples)))
    a0_replay_observed = compare_forward_readouts(replay, observed, list(range(base_task.n_examples)))
    result["a0"] = {
        "source_vs_record": a0_source_record,
        "record_vs_replay": a0_record_replay,
        "replay_vs_observed_replay": a0_replay_observed,
        "source_record_replay_observed_pass": a0_source_record["pass"] and a0_record_replay["pass"] and a0_replay_observed["pass"],
    }
    result["noise_runtime_trace"] = record["noise_trace"] + replay["noise_trace"] + observed["noise_trace"]
    result["noise_unhandled"] = record["unhandled_noise_calls"] + replay["unhandled_noise_calls"] + observed["unhandled_noise_calls"]
    if not a0_source_record["pass"]:
        result["final_label"] = "NOISETAPE_RECORD_NOT_TRANSPARENT"
        return result
    if not a0_record_replay["pass"]:
        result["final_label"] = "NOISETAPE_REPLAY_NOT_TRANSPARENT"
        return result
    if not a0_replay_observed["pass"]:
        result["final_label"] = "OBSERVER_INSTRUMENTATION_NOT_TRANSPARENT"
        return result
    restore_records(base_records, snapshot)
    cz = run_forward(mods, base_model, base_task, case["case_id"], "zero", schedule, perm, observe=True)
    restore_transformed_from_canonical(base_records, trans_records, snapshot, perm)
    trans_schedule = [
        {**row, "mean_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][0]), "local_capacity_object_id": id(trans_model.multiposteriors[tuple(row["dims"])][1]), "target_capacity_object_id": id(trans_model.target_capacities[tuple(row["dims"])])}
        for row in schedule
    ]
    tz = run_forward(mods, trans_model, trans_task, case["case_id"], "zero", trans_schedule, perm, transform_noise=True, observe=True)
    result["noise_runtime_trace"].extend(cz["noise_trace"] + tz["noise_trace"])
    result["noise_unhandled"].extend(cz["unhandled_noise_calls"] + tz["unhandled_noise_calls"])
    boundary_cmp = compare_boundary_events(cz["boundary_events"], tz["boundary_events"], case["invperm_old_to_new"])
    final_cmp = compare_forward_readouts(cz, tz, case["invperm_old_to_new"])
    result["primitive_commutation"] = {"tier1": {k: v for k, v in boundary_cmp.items() if k != "trace_rows"}, "final_forward_compare": final_cmp}
    result["first_divergence_trace_rows"] = boundary_cmp.get("trace_rows", [])
    if not boundary_cmp["pass"]:
        label = boundary_cmp["label"]
        result["primitive_commutation"]["tier2"] = (
            run_tier2_channel_trace(mods, base_model, trans_model, base_task, trans_task, case, schedule, base_records, trans_records, snapshot)
            if label == "FIRST_DIVERGENCE_CHANNEL_LAYER"
            else {"status": "FIRST_DIVERGENCE_LOCALIZED_TO_SOURCE_PRIMITIVE_REQUIRES_MANUAL_REVIEW", "reason": "line tracing implemented only for channel_layer in this audit"}
        )
        result["final_label"] = label
        return result
    if not final_cmp["pass"]:
        result["final_label"] = "FIRST_DIVERGENCE_HEAD"
        return result
    result["final_label"] = "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED"
    backward = run_take_step_certificate(mods, base_model, trans_model, base_task, trans_task, case, schedule, base_records, trans_records, snapshot, "backward_only")
    adam = run_take_step_certificate(mods, base_model, trans_model, base_task, trans_task, case, schedule, base_records, trans_records, snapshot, "one_step")
    result["loss_certificate"] = backward
    result["backward_certificate"] = backward
    result["adam_certificate"] = adam
    if backward["status"] != "TAKE_STEP_CERTIFIED":
        result["final_label"] = "BACKWARD_ACTION_FAILED" if backward.get("gradient_first_failure") else "LOSS_ACTION_FAILED"
    elif adam["status"] != "TAKE_STEP_CERTIFIED":
        result["final_label"] = "ADAM_ACTION_FAILED"
    else:
        result["final_label"] = "FULL_ONE_STEP_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED"
    return result


def build_cases(problems: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    case_rows = []
    problem_by_case: dict[str, dict[str, Any]] = {}
    for task_id in TARGET_TASKS:
        problem = problems[task_id]
        n_train = len(problem["train"])
        n_test = len(problem["test"])
        for idx in range(n_train - 1):
            perm = support_transposition_perm(n_train, n_test, idx)
            inv = inverse_perm(perm)
            case_id = f"{task_id}_adjacent_{idx}_{idx+1}"
            row = {
                "case_id": case_id,
                "case_kind": "real_task",
                "task_id": task_id,
                "n_train": n_train,
                "n_test": n_test,
                "n_examples": n_train + n_test,
                "transposition": [idx, idx + 1],
                "perm_new_to_old": perm,
                "invperm_old_to_new": inv,
                "test_query_slots_fixed_by_perm": all(perm[i] == i for i in range(n_train, n_train + n_test)),
                "test_query_slots_fixed_by_invperm": all(inv[i] == i for i in range(n_train, n_train + n_test)),
                "support_pairs_before": copy.deepcopy(problem["train"]),
                "support_pairs_after": copy.deepcopy(transformed_problem(problem, perm)["train"]),
                "permutation_non_identity": perm != list(range(n_train + n_test)),
            }
            case_rows.append(row)
            problem_by_case[case_id] = problem
    synthetic = make_synthetic_problem()
    for idx in [0, 1]:
        perm = support_transposition_perm(3, 1, idx)
        inv = inverse_perm(perm)
        case_id = f"synthetic_uniform_shape_oracle_adjacent_{idx}_{idx+1}"
        row = {
            "case_id": case_id,
            "case_kind": "synthetic_oracle",
            "task_id": "synthetic_uniform_shape_oracle",
            "n_train": 3,
            "n_test": 1,
            "n_examples": 4,
            "transposition": [idx, idx + 1],
            "perm_new_to_old": perm,
            "invperm_old_to_new": inv,
            "test_query_slots_fixed_by_perm": perm[3] == 3,
            "test_query_slots_fixed_by_invperm": inv[3] == 3,
            "support_pairs_before": copy.deepcopy(synthetic["train"]),
            "support_pairs_after": copy.deepcopy(transformed_problem(synthetic, perm)["train"]),
            "permutation_non_identity": True,
        }
        case_rows.append(row)
        problem_by_case[case_id] = synthetic
    spec = {
        "task_id": "synthetic_uniform_shape_oracle",
        "problem": synthetic,
        "requirements": {
            "train_support_pairs": 3,
            "test_query_inputs": 1,
            "known_grid_shape": [3, 3],
            "same_active_non_black_raw_color_set": [1, 2],
            "hidden_test_output": True,
        },
        "transpositions": [[0, 1], [1, 2]],
    }
    return case_rows, problem_by_case, spec


def write_inventory_csv(rows: list[dict[str, Any]]) -> None:
    path = ARTIFACT / "forward_reachable_state_inventory.csv"
    header = ["case_id", "variant", "alias_class_id", "object_id", "structural_path", "logical_dims", "shape", "dtype", "device", "requires_grad", "has_real_example_axis", "appears_in_weights_list", "optimizer_parameter_positions"]
    lines = [",".join(header)]
    for row in rows:
        for structural_path in row["structural_paths"]:
            lines.append(
                ",".join(
                    json.dumps(item) if isinstance(item, (list, bool)) else str(item)
                    for item in [
                        row["case_id"],
                        row["variant"],
                        row["alias_class_id"],
                        row["object_id"],
                        structural_path,
                        row["logical_dims"],
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_first_divergence_csv(rows: list[dict[str, Any]]) -> None:
    path = ARTIFACT / "first_divergence_trace.csv"
    header = ["case_id", "event_index", "event_name", "output_index", "output_path", "pass", "max_abs", "max_rel", "max_ulp", "over_tolerance_count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})


def run_child() -> dict[str, Any]:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    DETAIL.mkdir(parents=True, exist_ok=True)
    TASK_DIFF.mkdir(parents=True, exist_ok=True)
    mods = import_pinned_modules()
    stochastic_inventory = static_stochastic_inventory()
    write_json(ARTIFACT / "static_stochastic_call_inventory.json", stochastic_inventory)
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    problems = {"1f876c06": train_challenges["1f876c06"], "604001fa": eval_challenges["604001fa"]}
    cases, problem_by_case, synthetic_spec = build_cases(problems)
    write_json(ARTIFACT / "permutation_registry.json", {"created_utc": utc_now(), "cases": cases})
    write_json(ARTIFACT / "synthetic_oracle_spec.json", synthetic_spec)

    sample_task = mods["preprocessing"].Task("cpu_proof", make_synthetic_problem(), None)
    sample_model = mods["arc_compressor"].ARCCompressor(sample_task)
    tensor_proof = [
        {"name": "task.masks", "device": sample_task.masks.device.type, "dtype": str(sample_task.masks.dtype)},
        {"name": "task.problem", "device": sample_task.problem.device.type, "dtype": str(sample_task.problem.dtype)},
    ] + [{"name": f"weight_{idx}", "device": tensor.device.type, "dtype": str(tensor.dtype)} for idx, tensor in enumerate(sample_model.weights_list[:20])]
    cpu_proof = {
        "created_utc": utc_now(),
        "fixed_seed": FIXED_SEED,
        "post_import_seed_probe": mods["seed_probe"],
        "final_seed_probe_after_imports": seed_probe(),
        "effective_default_device": str(torch.get_default_device()),
        "effective_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "import_intercept_log": mods["shim"]["intercept_log"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_memory_allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "no_cuda_execution": True,
        "all_runtime_tensors_under_audit_cpu": all(row["device"] == "cpu" for row in tensor_proof),
        "all_new_floating_preflight_tensors_float64": all(row["dtype"] == "torch.float64" for row in tensor_proof if row["name"] != "task.problem"),
        "no_production_source_file_changed": all(row["matches_pinned_blob"] for row in mods["provenance"]["modules"]),
        "tensor_proof": tensor_proof,
    }
    write_json(ARTIFACT / "cpu_execution_proof.json", cpu_proof)

    results = []
    all_inventory = []
    all_trace_rows = []
    all_noise_trace = []
    all_unhandled = []
    all_schedules = []
    task_certs = []
    state_certs = []
    alias_certs = []
    optimizer_parts = []
    loss_certs = []
    backward_certs = []
    adam_certs = []
    for case in cases:
        try:
            result = run_case(mods, case, problem_by_case[case["case_id"]])
        except RuntimeError as exc:
            label = str(exc) if str(exc) in {"STOCHASTIC_TRACE_INCOMPLETE", "TRACE_SCHEDULE_MISMATCH"} else "FIRST_DIVERGENCE_LOCALIZED_TO_SOURCE_PRIMITIVE_REQUIRES_MANUAL_REVIEW"
            result = {"case": case, "final_label": label, "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        task_certs.append(result.get("task_action", {"case_id": case["case_id"], "status": result.get("final_label")}))
        if "state_action" in result:
            state_certs.append(result["state_action"])
            alias_certs.append({"case_id": case["case_id"], "status": result["state_action"]["alias_topology_status"]})
        if "optimizer_partition" in result:
            optimizer_parts.append(result["optimizer_partition"])
        if "posterior_leaf_schedule" in result:
            all_schedules.extend(result["posterior_leaf_schedule"])
        if "loss_certificate" in result:
            loss_certs.append({"case_id": case["case_id"], **result["loss_certificate"]})
            backward_certs.append({"case_id": case["case_id"], **result["backward_certificate"]})
        if "adam_certificate" in result:
            adam_certs.append({"case_id": case["case_id"], **result["adam_certificate"]})
        all_inventory.extend(result.pop("_inventory", []))
        for row in result.get("first_divergence_trace_rows", []):
            all_trace_rows.append({"case_id": case["case_id"], **row})
        all_noise_trace.extend(result.get("noise_runtime_trace", []))
        all_unhandled.extend(result.get("noise_unhandled", []))
        results.append(result)

    write_inventory_csv(all_inventory)
    write_first_divergence_csv(all_trace_rows)
    write_json(ARTIFACT / "task_action_certificate.json", {"created_utc": utc_now(), "cases": task_certs})
    write_json(ARTIFACT / "state_action_certificate.json", {"created_utc": utc_now(), "cases": state_certs})
    write_json(ARTIFACT / "alias_graph_certificate.json", {"created_utc": utc_now(), "cases": alias_certs})
    write_json(ARTIFACT / "optimizer_parameter_partition.json", {"created_utc": utc_now(), "cases": optimizer_parts})
    write_json(ARTIFACT / "posterior_leaf_schedule.json", {"created_utc": utc_now(), "leaves": all_schedules})
    write_json(ARTIFACT / "noise_tape_manifest.json", {"created_utc": utc_now(), "static_stochastic_inventory": stochastic_inventory, "case_count": len(cases)})
    write_json(ARTIFACT / "noise_tape_runtime_trace.json", {"created_utc": utc_now(), "trace": all_noise_trace})
    write_json(ARTIFACT / "noise_tape_unhandled_call_report.json", {"created_utc": utc_now(), "unhandled_calls": all_unhandled})
    write_json(ARTIFACT / "primitive_commutation_certificate.json", {"created_utc": utc_now(), "cases": [{ "case_id": row["case"]["case_id"], "final_label": row["final_label"], "primitive_commutation": row.get("primitive_commutation") } for row in results]})
    write_json(ARTIFACT / "synthetic_oracle_results.json", {"created_utc": utc_now(), "cases": [row for row in results if row["case"]["case_kind"] == "synthetic_oracle"]})
    write_json(ARTIFACT / "loss_certificate.json", {"created_utc": utc_now(), "status": "NOT_REACHED_FORWARD_FAILURE" if not loss_certs else "SOME_CASES_REACHED", "cases": loss_certs})
    write_json(ARTIFACT / "backward_certificate.json", {"created_utc": utc_now(), "status": "NOT_REACHED_FORWARD_FAILURE" if not backward_certs else "SOME_CASES_REACHED", "cases": backward_certs})
    write_json(ARTIFACT / "adam_certificate.json", {"created_utc": utc_now(), "status": "NOT_REACHED_FORWARD_FAILURE" if not adam_certs else "SOME_CASES_REACHED", "cases": adam_certs})
    write_json(ARTIFACT / "first_divergence_cases.json", {"created_utc": utc_now(), "cases": results})
    summary = {
        "created_utc": utc_now(),
        "status": "CHILD_COMPLETE",
        "case_count": len(results),
        "real_case_count": sum(1 for row in results if row["case"]["case_kind"] == "real_task"),
        "synthetic_case_count": sum(1 for row in results if row["case"]["case_kind"] == "synthetic_oracle"),
        "final_labels": {row["case"]["case_id"]: row["final_label"] for row in results},
        "loss_backward_adam_reached": bool(loss_certs or backward_certs or adam_certs),
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
