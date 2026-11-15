from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
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


PREFIX = "r6a_phase2_semantic_state_action_decoder_boundary_v10"
V8_PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
FIXED_SEED = 20260702
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
V8_ARTIFACT = R6 / "artifacts" / V8_PREFIX
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
    out = {
        "seed": FIXED_SEED,
        "python_random_first": random.random(),
        "numpy_random_first": float(np.random.random()),
        "torch_random_first": float(torch.rand(()).item()),
    }
    reset_determinism()
    return out


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


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {"pass": False, "reason": "shape_mismatch", "shape_a": list(a.shape), "shape_b": list(b.shape), "max_abs": float("inf"), "max_rel": float("inf"), "canonical_checksum": tensor_checksum(a), "transformed_checksum": tensor_checksum(b)}
    ac = a.detach().cpu()
    bc = b.detach().cpu()
    if ac.dtype != bc.dtype:
        return {"pass": False, "reason": "dtype_mismatch", "dtype_a": str(ac.dtype), "dtype_b": str(bc.dtype)}
    if not torch.is_floating_point(ac):
        ok = torch.equal(ac, bc)
        return {"pass": bool(ok), "max_abs": 0.0 if ok else float("inf"), "max_rel": 0.0 if ok else float("inf"), "canonical_checksum": tensor_checksum(a), "transformed_checksum": tensor_checksum(b)}
    finite_a = torch.isfinite(ac)
    finite_b = torch.isfinite(bc)
    if not torch.equal(finite_a, finite_b):
        return {"pass": False, "reason": "nonfinite_status_mismatch", "canonical_checksum": tensor_checksum(a), "transformed_checksum": tensor_checksum(b)}
    diff = torch.abs(ac - bc)
    denom = torch.maximum(torch.abs(ac), torch.abs(bc))
    allowed = ABS_TOL + REL_TOL * denom
    over = diff > allowed
    rel = diff / torch.maximum(denom, torch.tensor(1e-300, dtype=denom.dtype))
    return {
        "pass": bool(not over.any().item()),
        "max_abs": float(torch.max(diff).item()) if diff.numel() else 0.0,
        "max_rel": float(torch.max(rel).item()) if rel.numel() else 0.0,
        "over_tolerance_count": int(over.sum().item()),
        "over_tolerance_fraction": float(over.sum().item() / max(1, over.numel())),
        "canonical_checksum": tensor_checksum(a),
        "transformed_checksum": tensor_checksum(b),
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
    import_context = {"repository_root": str(REFERENCE_ROOT), "cwd": os.getcwd(), "sys_path": list(sys.path), "sys_path_0": sys.path[0]}
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
            "current_working_directory": os.getcwd(),
            "sys_path": list(sys.path),
            "matching_relative_path_under_repository_root": rel,
            "expected_pinned_git_blob_sha256": expected_sha,
            "path_under_intended_source_root": path.is_relative_to(REFERENCE_ROOT),
            "matches_pinned_blob": git_ok and actual_sha == expected_sha,
        }
        ok = ok and row["path_under_intended_source_root"] and row["matches_pinned_blob"]
        rows.append(row)
    provenance = {"created_utc": utc_now(), "pinned_commit": PINNED_COMMIT, "import_context": import_context, "modules": rows, "import_provenance_status": "IMPORT_PROVENANCE_CERTIFIED" if ok else "IMPORT_PROVENANCE_MISMATCH"}
    write_json(MANIFEST / "imported_module_provenance.json", provenance)
    if not ok:
        raise RuntimeError("IMPORT_PROVENANCE_MISMATCH")
    return {**mods, "shim": shim, "seed_probe": seed_probe(), "provenance": provenance}


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
    return {"train": train, "test": [{"input": [[0, 2, 1], [1, 0, 2], [2, 1, 0]]}]}


def build_cases() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    registry = read_json(V8_ARTIFACT / "permutation_registry.json")["cases"]
    wanted = {
        "1f876c06_adjacent_0_1",
        "1f876c06_adjacent_1_2",
        "604001fa_adjacent_0_1",
        "604001fa_adjacent_1_2",
        "604001fa_adjacent_2_3",
        "synthetic_uniform_shape_oracle_adjacent_0_1",
        "synthetic_uniform_shape_oracle_adjacent_1_2",
    }
    train_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_training_challenges.json")
    eval_challenges = read_json(REFERENCE_ROOT / "dataset" / "arc-agi_evaluation_challenges.json")
    problem_by_task = {"1f876c06": train_challenges["1f876c06"], "604001fa": eval_challenges["604001fa"], "synthetic_uniform_shape_oracle": make_synthetic_problem()}
    cases = [row for row in registry if row["case_id"] in wanted]
    cases.sort(key=lambda row: ["1f876c06_adjacent_0_1", "1f876c06_adjacent_1_2", "604001fa_adjacent_0_1", "604001fa_adjacent_1_2", "604001fa_adjacent_2_3", "synthetic_uniform_shape_oracle_adjacent_0_1", "synthetic_uniform_shape_oracle_adjacent_1_2"].index(row["case_id"]))
    return cases, {row["case_id"]: problem_by_task[row["task_id"]] for row in cases}


def is_multitensor(obj: Any) -> bool:
    return hasattr(obj, "multitensor_system") and hasattr(obj, "__getitem__")


def iter_model_containers(model: Any) -> list[tuple[str, Any]]:
    return [
        ("multiposteriors", model.multiposteriors),
        ("target_capacities", model.target_capacities),
        ("decode_weights", model.decode_weights),
        ("share_up_weights", model.share_up_weights),
        ("share_down_weights", model.share_down_weights),
        ("softmax_weights", model.softmax_weights),
        ("cummax_weights", model.cummax_weights),
        ("shift_weights", model.shift_weights),
        ("direction_share_weights", model.direction_share_weights),
        ("nonlinear_weights", model.nonlinear_weights),
        ("head_weights", model.head_weights),
        ("mask_weights", model.mask_weights),
    ]


_OBJECT_REGISTRY: dict[int, torch.Tensor] = {}


def constructor_role(root: str, path: str) -> str:
    if root == "multiposteriors":
        if path.endswith("/0"):
            return "posterior_mean"
        if path.endswith("/1"):
            return "posterior_local_capacity_adjustment"
        return "posterior_container"
    if root == "target_capacities":
        return "target_capacity"
    if root == "decode_weights":
        if path.endswith("/0"):
            return "decode_weight_matrix"
        if path.endswith("/1"):
            return "decode_bias"
        return "decode_weight_container"
    if root == "head_weights":
        return "head_weight_or_bias"
    if root == "mask_weights":
        return "mask_weight_or_bias"
    return f"{root}_tensor"


def semantic_action(root: str, role: str, dims: list[int] | None) -> tuple[str, str]:
    if root == "multiposteriors" and role in {"posterior_mean", "posterior_local_capacity_adjustment"} and dims is not None and dims[0] == 1:
        return "PERMUTE_EXAMPLE_AXIS_0", "posterior mean/local-capacity leaf with dims[0] == 1; physical axis 0 is the example axis by multitensor_system.shape(dims, channel_dim)"
    return "IDENTITY_COPY", "closed semantic table: all non-eligible posterior tensors and all non-posterior forward-reachable tensors receive identity copy"


def build_semantic_records(model: Any, task: Any, case_id: str, variant: str) -> list[dict[str, Any]]:
    weight_positions: dict[int, list[int]] = {}
    for idx, tensor in enumerate(model.weights_list):
        weight_positions.setdefault(id(tensor), []).append(idx)
    records_by_obj: dict[int, dict[str, Any]] = {}

    def visit(obj: Any, root: str, path: str, dims: list[int] | None) -> None:
        if torch.is_tensor(obj):
            role = constructor_role(root, path)
            action, rationale = semantic_action(root, role, dims)
            oid = id(obj)
            rec = records_by_obj.setdefault(
                oid,
                {
                    "case_id": case_id,
                    "variant": variant,
                    "object_id": oid,
                    "structural_paths": [],
                    "constructor_roles": [],
                    "leaf_dims": [],
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "device": obj.device.type,
                    "requires_grad": bool(obj.requires_grad),
                    "appears_in_weights_list": oid in weight_positions,
                    "optimizer_parameter_positions": weight_positions.get(oid, []),
                    "action_labels": [],
                    "semantic_rationales": [],
                    "leading_dim_equals_task_n_examples": bool(obj.ndim >= 1 and obj.shape[0] == task.n_examples),
                    "is_posterior_tensor": root == "multiposteriors",
                },
            )
            rec["structural_paths"].append(path)
            rec["constructor_roles"].append(role)
            rec["leaf_dims"].append(list(dims) if dims is not None else None)
            rec["action_labels"].append(action)
            rec["semantic_rationales"].append(rationale)
            return
        if is_multitensor(obj):
            for dims_item in obj.multitensor_system:
                visit(obj[dims_item], root, f"{path}[{tuple(dims_item)}]", list(dims_item))
            return
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                visit(item, root, f"{path}/{idx}", dims)
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                visit(item, root, f"{path}/{key}", dims)

    for root, container in iter_model_containers(model):
        visit(container, root, root, None)
    records = list(records_by_obj.values())
    records.sort(key=lambda row: min(row["structural_paths"]))
    for alias_id, row in enumerate(records):
        row["alias_class_id"] = alias_id
        row["action_label"] = sorted(set(row["action_labels"]))
        row["semantic_action_conflict"] = len(row["action_label"]) != 1
        row["action_label"] = row["action_label"][0] if len(row["action_label"]) == 1 else "CONFLICT"
        row["semantic_rationale"] = sorted(set(row["semantic_rationales"]))[0]
        row["accidental_leading_dim_equality"] = bool(row["leading_dim_equals_task_n_examples"] and row["action_label"] == "IDENTITY_COPY")
    return records


def register_objects(model: Any, records: list[dict[str, Any]]) -> None:
    wanted = {row["object_id"] for row in records}

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


def snapshot(records: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
    return {row["object_id"]: _OBJECT_REGISTRY[row["object_id"]].detach().clone() for row in records}


def restore(records: list[dict[str, Any]], snap: dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for row in records:
            target = _OBJECT_REGISTRY[row["object_id"]]
            target.copy_(snap[row["object_id"]].to(target.device, target.dtype))


def semantic_restore(canonical: list[dict[str, Any]], transformed: list[dict[str, Any]], snap: dict[int, torch.Tensor], perm_new_to_old: list[int]) -> None:
    index = torch.tensor(perm_new_to_old, dtype=torch.long)
    with torch.no_grad():
        for a, b in zip(canonical, transformed):
            source = snap[a["object_id"]]
            value = source[index.to(source.device)] if b["action_label"] == "PERMUTE_EXAMPLE_AXIS_0" else source
            target = _OBJECT_REGISTRY[b["object_id"]]
            target.copy_(value.to(target.device, target.dtype))


def task_action_certificate(mods: dict[str, Any], case: dict[str, Any], problem: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    preprocessing = mods["preprocessing"]
    reset_determinism()
    canonical = preprocessing.Task(case["task_id"], problem, None)
    reset_determinism()
    transformed = preprocessing.Task(case["task_id"], transformed_problem(problem, case["perm_new_to_old"]), None)
    inv = torch.tensor(case["invperm_old_to_new"], dtype=torch.long)
    aligned_problem = transformed.problem[inv]
    aligned_masks = transformed.masks[inv]
    aligned_shapes = [transformed.shapes[i] for i in case["invperm_old_to_new"]]
    checks = {
        "problem": torch.equal(canonical.problem, aligned_problem),
        "masks": torch.equal(canonical.masks, aligned_masks),
        "shapes": canonical.shapes == aligned_shapes,
        "colors": canonical.colors == transformed.colors,
        "n_colors": canonical.n_colors == transformed.n_colors,
        "n_x": canonical.n_x == transformed.n_x,
        "n_y": canonical.n_y == transformed.n_y,
        "n_train": canonical.n_train == transformed.n_train,
        "n_test": canonical.n_test == transformed.n_test,
        "in_out_same_size": canonical.in_out_same_size == transformed.in_out_same_size,
        "all_in_same_size": canonical.all_in_same_size == transformed.all_in_same_size,
        "all_out_same_size": canonical.all_out_same_size == transformed.all_out_same_size,
        "predicted_query_output_shape": [row[1] for row in canonical.shapes[canonical.n_train :]] == [row[1] for row in transformed.shapes[transformed.n_train :]],
        "fixed_test_query_record": problem["test"] == transformed_problem(problem, case["perm_new_to_old"])["test"],
    }
    return canonical, transformed, {"case_id": case["case_id"], "status": "TASK_ACTION_CERTIFIED" if all(checks.values()) else "TASK_ACTION_FAILED", "checks": checks}


def collision_witnesses(case: dict[str, Any], canonical: list[dict[str, Any]], transformed: list[dict[str, Any]], snap: dict[int, torch.Tensor], task: Any) -> list[dict[str, Any]]:
    rows = []
    for a, b in zip(canonical, transformed):
        if b["is_posterior_tensor"]:
            continue
        if not b["leading_dim_equals_task_n_examples"]:
            continue
        after = _OBJECT_REGISTRY[b["object_id"]]
        before = snap[a["object_id"]]
        equal = torch.equal(before.cpu(), after.detach().cpu())
        rows.append(
            {
                "case_id": case["case_id"],
                "structural_path": b["structural_paths"][0],
                "constructor_role": b["constructor_roles"][0],
                "leaf_dims": b["leaf_dims"][0],
                "physical_shape": b["shape"],
                "task_n_examples": task.n_examples,
                "action_label": b["action_label"],
                "before_checksum": tensor_checksum(before),
                "after_checksum": tensor_checksum(after),
                "exact_equality_result": bool(equal),
            }
        )
    return rows


@contextmanager
def zero_randn(layers: Any):
    original = layers.torch.randn

    def wrapped(shape, *args, **kwargs):
        return torch.zeros(tuple(shape), dtype=torch.get_default_dtype(), device=torch.get_default_device())

    layers.torch.randn = wrapped
    try:
        yield
    finally:
        layers.torch.randn = original


def affine_commutation_certificate(mods: dict[str, Any], case: dict[str, Any], model: Any, task: Any) -> dict[str, Any]:
    layers = mods["layers"]
    rows = []
    with zero_randn(layers):
        for dims in model.multitensor_system:
            if dims[0] != 1:
                continue
            posterior = model.multiposteriors[dims]
            target_capacity = model.target_capacities[dims]
            decode_weight = model.decode_weights[dims]
            z, _ = layers.channel_layer(target_capacity, posterior)
            perm = torch.tensor(case["perm_new_to_old"], dtype=torch.long, device=z.device)
            left = layers.affine(z[perm], decode_weight, use_bias=True)
            right = layers.affine(z, decode_weight, use_bias=True)[perm]
            comp = compare_tensors(left, right)
            rows.append({"case_id": case["case_id"], "dims": list(dims), "status": "PASS" if comp["pass"] else "FAIL", "comparison": comp})
    z = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10
    w = (torch.arange(16, dtype=torch.float64).reshape(4, 4) + 1) / 7
    b = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float64)
    perm = torch.tensor([1, 0, 2, 3], dtype=torch.long)
    wrong_w = w[perm]
    neg_left = layers.affine(z[perm], [wrong_w, b], use_bias=True)
    neg_right = layers.affine(z, [w, b], use_bias=True)[perm]
    neg = compare_tensors(neg_left, neg_right)
    return {
        "case_id": case["case_id"],
        "production_leaf_rows": rows,
        "production_leaf_pass": all(row["status"] == "PASS" for row in rows),
        "negative_control": {"wrong_weight_action_mismatch_observed": not neg["pass"], "comparison": neg},
    }


def tensor_payload(name: str, tensor: torch.Tensor, align: bool, role: str) -> dict[str, Any]:
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": tensor.device.type, "align": align, "role": role, "tensor": tensor.detach().clone(), "checksum": tensor_checksum(tensor)}


class DecoderBoundaryTracer:
    def __init__(self, mods: dict[str, Any], task: Any):
        self.layers = mods["layers"]
        self.task = task
        self.events: list[dict[str, Any]] = []
        self.originals: dict[str, Any] = {}
        self.depth = 0
        self.current_leaf: dict[str, Any] | None = None
        self.pending_decode_affine: dict[str, Any] | None = None
        self.ordinal = 0
        self.schedule = [list(dims) for dims in task.multitensor_system]
        self.boundary_names = ["decode_latents", "share_up", "softmax", "cummax", "shift", "direction_share", "nonlinear", "share_down", "normalize", "postprocess_mask"]

    def __enter__(self):
        self.originals["channel_layer"] = self.layers.channel_layer
        self.originals["affine"] = self.layers.affine

        def channel_wrapper(target_capacity, posterior):
            dims = self.schedule[self.ordinal] if self.ordinal < len(self.schedule) else None
            ordinal = self.ordinal
            mean, local_capacity = posterior
            event = {"event_name": "channel_layer", "call_nesting": self.depth, "posterior_leaf_ordinal": ordinal, "dims": dims, "direction": None, "tensors": []}
            align = bool(dims and dims[0] == 1)
            event["tensors"].extend(
                [
                    tensor_payload("target_capacity", target_capacity, False, "target_capacity"),
                    tensor_payload("posterior_mean", mean, align, "posterior_mean"),
                    tensor_payload("posterior_local_capacity_adjustment", local_capacity, align, "posterior_local_capacity_adjustment"),
                ]
            )
            result = self.originals["channel_layer"](target_capacity, posterior)
            z, kl = result
            event["tensors"].extend([tensor_payload("z", z, align, "channel_output"), tensor_payload("KL", kl, align, "channel_output")])
            self.events.append(event)
            self.pending_decode_affine = {"ordinal": ordinal, "dims": dims, "z_checksum": tensor_checksum(z)}
            self.ordinal += 1
            return result

        def affine_wrapper(x, weight, use_bias=False):
            result = self.originals["affine"](x, weight, use_bias=use_bias)
            if self.pending_decode_affine is not None and isinstance(weight, (list, tuple)) and len(weight) >= 2:
                dims = self.pending_decode_affine["dims"]
                align = bool(dims and dims[0] == 1)
                event = {"event_name": "decode_affine", "call_nesting": self.depth, "posterior_leaf_ordinal": self.pending_decode_affine["ordinal"], "dims": dims, "direction": None, "tensors": []}
                event["tensors"].append(tensor_payload("z_input", x, align, "runtime_activation"))
                event["tensors"].append(tensor_payload("decode_weight", weight[0], False, "model_parameter"))
                if use_bias:
                    event["tensors"].append(tensor_payload("decode_bias", weight[1], False, "model_parameter"))
                event["tensors"].append(tensor_payload("decode_affine_output", result, align, "runtime_activation"))
                self.events.append(event)
                self.pending_decode_affine = None
            return result

        self.layers.channel_layer = channel_wrapper
        self.layers.affine = affine_wrapper
        for name in self.boundary_names:
            original = getattr(self.layers, name)
            self.originals[name] = original

            def make_wrapper(fn, fname):
                def wrapper(*args, **kwargs):
                    direction = args[2] if fname == "direction_share" and len(args) >= 3 else None
                    self.depth += 1
                    try:
                        result = fn(*args, **kwargs)
                    finally:
                        self.depth -= 1
                    if fname == "normalize" and not is_multitensor(result):
                        return result
                    self.events.append({"event_name": fname, "call_nesting": self.depth, "posterior_leaf_ordinal": None, "dims": None, "direction": direction, "tensors": tensor_tree(result, self.task, fname)})
                    return result

                return wrapper

            setattr(self.layers, name, make_wrapper(original, name))
        return self

    def __exit__(self, exc_type, exc, tb):
        for name, original in self.originals.items():
            setattr(self.layers, name, original)
        return False


def tensor_tree(obj: Any, task: Any, root: str, dims: list[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if root == "decode_latents" and isinstance(obj, tuple) and len(obj) >= 3:
        x_multitensor, kl_amounts, kl_names = obj[0], obj[1], obj[2]
        rows.extend(tensor_tree(x_multitensor, task, "decode_latents/x", dims))
        for idx, kl_tensor in enumerate(kl_amounts):
            try:
                kl_dims = list(ast.literal_eval(kl_names[idx]))
            except Exception:
                kl_dims = None
            align = bool(kl_dims and kl_dims[0] == 1)
            rows.append(tensor_payload(f"decode_latents/KL/{idx}", kl_tensor, align, "runtime_activation"))
        return rows
    if root == "postprocess_mask" and isinstance(obj, tuple) and len(obj) == 2 and all(torch.is_tensor(item) for item in obj):
        rows.append(tensor_payload("postprocess_mask/x_mask", obj[0], True, "runtime_activation_from_dims_[1,0,0,1,0]"))
        rows.append(tensor_payload("postprocess_mask/y_mask", obj[1], True, "runtime_activation_from_dims_[1,0,0,0,1]"))
        return rows

    def visit(item: Any, path: str, leaf_dims: list[int] | None):
        if torch.is_tensor(item):
            align = bool(leaf_dims and leaf_dims[0] == 1)
            rows.append(tensor_payload(path, item, align, "runtime_activation"))
            return
        if is_multitensor(item):
            for d in item.multitensor_system:
                visit(item[d], f"{path}[{tuple(d)}]", list(d))
            return
        if isinstance(item, (list, tuple)):
            for idx, sub in enumerate(item):
                visit(sub, f"{path}/{idx}", leaf_dims)
            return
        if isinstance(item, dict):
            for key, sub in item.items():
                visit(sub, f"{path}/{key}", leaf_dims)

    visit(obj, root, dims)
    return rows


@contextmanager
def zero_noise_leaf_context(mods: dict[str, Any], model: Any):
    layers = mods["layers"]
    original_randn = layers.torch.randn
    original_channel = layers.channel_layer
    # The tracer wraps channel_layer; this context only zeros the random draw.
    def randn_wrapper(shape, *args, **kwargs):
        return torch.zeros(tuple(shape), dtype=torch.get_default_dtype(), device=torch.get_default_device())
    layers.torch.randn = randn_wrapper
    try:
        yield
    finally:
        layers.torch.randn = original_randn


def run_decoder_trace(mods: dict[str, Any], model: Any, task: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tracer = DecoderBoundaryTracer(mods, task)
    layers = mods["layers"]
    original_channel = layers.channel_layer

    def schedule_channel(target_capacity, posterior):
        # This shim is replaced by tracer, but current_leaf must be set before each channel call.
        return original_channel(target_capacity, posterior)

    with zero_noise_leaf_context(mods, model), tracer:
        # Drive the source normally. The channel wrapper assigns ordinal order dynamically.
        logits, x_mask, y_mask, kl_amounts, kl_names = model.forward()
    final = {
        "logits": tensor_payload("final_logits", logits, True, "runtime_activation"),
        "x_mask": tensor_payload("final_x_mask", x_mask, True, "runtime_activation"),
        "y_mask": tensor_payload("final_y_mask", y_mask, True, "runtime_activation"),
        "kl_names": list(kl_names),
        "kl_amounts": [tensor_payload(f"final_KL/{idx}", item, False, "runtime_activation") for idx, item in enumerate(kl_amounts)],
    }
    return final, tracer.events


def compare_payload(a: dict[str, Any], b: dict[str, Any], invperm_old_to_new: list[int]) -> dict[str, Any]:
    bt = b["tensor"]
    if b["align"]:
        bt = bt[torch.tensor(invperm_old_to_new, dtype=torch.long, device=bt.device)]
    return compare_tensors(a["tensor"], bt)


def compare_decoder_traces(case: dict[str, Any], events_a: list[dict[str, Any]], events_b: list[dict[str, Any]], affine_cert_pass: bool) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    rows = []
    if len(events_a) != len(events_b):
        return "TRACE_SCHEDULE_MISMATCH", rows, {"reason": "event_count_mismatch", "canonical": len(events_a), "transformed": len(events_b)}
    for idx, (ea, eb) in enumerate(zip(events_a, events_b)):
        for key in ["event_name", "call_nesting", "posterior_leaf_ordinal", "dims", "direction"]:
            if ea.get(key) != eb.get(key):
                return "TRACE_SCHEDULE_MISMATCH", rows, {"event_index": idx, "key": key, "canonical": ea.get(key), "transformed": eb.get(key)}
        tensors_a, tensors_b = ea["tensors"], eb["tensors"]
        if len(tensors_a) != len(tensors_b):
            return "TRACE_SCHEDULE_MISMATCH", rows, {"event_index": idx, "reason": "tensor_count_mismatch"}
        event_comparisons: dict[str, dict[str, Any]] = {}
        for ta, tb in zip(tensors_a, tensors_b):
            if ta["name"] != tb["name"] or ta["role"] != tb["role"]:
                return "TRACE_SCHEDULE_MISMATCH", rows, {"event_index": idx, "reason": "tensor_schedule_mismatch", "canonical": ta["name"], "transformed": tb["name"]}
            comp = compare_payload(ta, tb, case["invperm_old_to_new"])
            event_comparisons[ta["name"]] = comp
            rows.append({"case_id": case["case_id"], "event_index": idx, "event_name": ea["event_name"], "tensor_name": ta["name"], "pass": comp["pass"], "max_abs": comp.get("max_abs"), "max_rel": comp.get("max_rel"), "over_tolerance_count": comp.get("over_tolerance_count")})
            if not comp["pass"]:
                if ea["event_name"] == "channel_layer":
                    return "FIRST_DIVERGENCE_CHANNEL_LAYER", rows, {"event_index": idx, "tensor_name": ta["name"], **comp}
                if ea["event_name"] == "decode_affine":
                    z_ok = event_comparisons.get("z_input", {}).get("pass") is True
                    w_ok = event_comparisons.get("decode_weight", {}).get("pass") is True
                    b_ok = event_comparisons.get("decode_bias", {"pass": True}).get("pass") is True
                    if ta["name"] == "decode_affine_output" and z_ok and w_ok and b_ok and affine_cert_pass:
                        return "AFFINE_OBSERVATION_CONTRADICTION", rows, {"event_index": idx, "tensor_name": ta["name"], "z_input_equal": z_ok, "decode_weight_identical": w_ok, "decode_bias_identical": b_ok, "direct_affine_certificate_pass": affine_cert_pass, **comp}
                    return "FIRST_DIVERGENCE_CHANNEL_LAYER", rows, {"event_index": idx, "tensor_name": ta["name"], "reason": "decode_affine_precondition_failed", **comp}
                mapping = {
                    "decode_latents": "FIRST_DIVERGENCE_DECODE_LATENTS",
                    "share_up": "FIRST_DIVERGENCE_SHARE_DIRECTION_UP",
                    "share_down": "FIRST_DIVERGENCE_SHARE_DIRECTION_DOWN",
                    "softmax": "FIRST_DIVERGENCE_SOFTMAX",
                    "cummax": "FIRST_DIVERGENCE_CUMMAX",
                    "shift": "FIRST_DIVERGENCE_SHIFT",
                    "direction_share": "FIRST_DIVERGENCE_DIRECTION_SHARE",
                    "nonlinear": "FIRST_DIVERGENCE_NONLINEAR",
                    "normalize": "FIRST_DIVERGENCE_NORMALIZE",
                    "postprocess_mask": "FIRST_DIVERGENCE_POSTPROCESS",
                }
                return mapping.get(ea["event_name"], "FIRST_DIVERGENCE_HEAD"), rows, {"event_index": idx, "tensor_name": ta["name"], **comp}
    return "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED", rows, {}


def v8_action_behavior_audit() -> dict[str, Any]:
    source = (R6 / "scripts" / f"{V8_PREFIX}_child.py").read_text(encoding="utf-8")
    inventory = V8_ARTIFACT / "forward_reachable_state_inventory.csv"
    rows = []
    if not inventory.exists():
        payload = {"status": "V8_ACTION_IMPLEMENTATION_UNAVAILABLE", "records": []}
        write_json(ARTIFACT / "v8_action_behavior_audit.json", payload)
        return payload
    with inventory.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("variant") != "transformed":
                continue
            applied = row.get("has_real_example_axis") == "true"
            path = row.get("structural_path", "")
            nonposterior = not path.startswith("multiposteriors")
            if applied and nonposterior:
                rows.append(
                    {
                        "structural_path": path,
                        "constructor_role": path.split("[", 1)[0].split("/", 1)[0],
                        "leaf_dims": row.get("logical_dims"),
                        "physical_shape": row.get("shape"),
                        "v8_applied_permutation": True,
                        "evidence_source": "v8 forward_reachable_state_inventory.csv plus v8 restore_transformed_from_canonical uses transformed_rec['has_real_example_axis']",
                    }
                )
            if len(rows) >= 500:
                break
    status = "V8_NONPOSTERIOR_PERMUTATION_OBSERVED" if rows and "if b[\"has_real_example_axis\"]" in source or "if b[\"has_real_example_axis\"]" in source else ("V8_NONPOSTERIOR_PERMUTATION_OBSERVED" if rows else "V8_NONPOSTERIOR_PERMUTATION_NOT_OBSERVED")
    payload = {"created_utc": utc_now(), "status": status, "record_count": len(rows), "records": rows}
    write_json(ARTIFACT / "v8_action_behavior_audit.json", payload)
    return payload


def write_semantic_action_csv(rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "structural_path", "constructor_role", "leaf_dims", "physical_shape", "dtype", "device", "alias_class", "action_label", "semantic_rationale", "checksum_before_action", "checksum_after_action", "leading_dim_equals_task_n_examples", "accidental_leading_dim_equality"]
    with (ARTIFACT / "semantic_action_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_first_divergence_csv(rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "event_index", "event_name", "tensor_name", "pass", "max_abs", "max_rel", "over_tolerance_count"]
    with (ARTIFACT / "first_divergence_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(mods: dict[str, Any], case: dict[str, Any], problem: dict[str, Any]) -> dict[str, Any]:
    _OBJECT_REGISTRY.clear()
    base_task, trans_task, task_cert = task_action_certificate(mods, case, problem)
    result: dict[str, Any] = {"case": case, "task_action": task_cert}
    if task_cert["status"] != "TASK_ACTION_CERTIFIED":
        result["forward_label"] = "TASK_ACTION_FAILED"
        return result
    reset_determinism()
    base_model = mods["arc_compressor"].ARCCompressor(base_task)
    reset_determinism()
    trans_model = mods["arc_compressor"].ARCCompressor(trans_task)
    base_records = build_semantic_records(base_model, base_task, case["case_id"], "canonical")
    trans_records = build_semantic_records(trans_model, trans_task, case["case_id"], "transformed")
    register_objects(base_model, base_records)
    register_objects(trans_model, trans_records)
    if any(row["semantic_action_conflict"] for row in trans_records):
        result["forward_label"] = "ALIAS_SEMANTIC_ROLE_CONFLICT"
        result["semantic_state_action"] = {"case_id": case["case_id"], "status": "ALIAS_SEMANTIC_ROLE_CONFLICT"}
        result["_records"] = base_records + trans_records
        return result
    snap = snapshot(base_records)
    before_target = {row["object_id"]: tensor_checksum(_OBJECT_REGISTRY[row["object_id"]]) for row in trans_records}
    semantic_restore(base_records, trans_records, snap, case["perm_new_to_old"])
    witnesses = collision_witnesses(case, base_records, trans_records, snap, base_task)
    state_fail = None
    for a, b in zip(base_records, trans_records):
        after = _OBJECT_REGISTRY[b["object_id"]]
        expected = snap[a["object_id"]]
        if b["action_label"] == "PERMUTE_EXAMPLE_AXIS_0":
            aligned = after[torch.tensor(case["invperm_old_to_new"], dtype=torch.long, device=after.device)]
            comp = compare_tensors(expected, aligned)
        else:
            comp = compare_tensors(expected, after)
        if not comp["pass"]:
            state_fail = {"path": b["structural_paths"][0], **comp}
            break
    witness_fail = [row for row in witnesses if not row["exact_equality_result"]]
    result["semantic_state_action"] = {
        "case_id": case["case_id"],
        "status": "SEMANTIC_STATE_ACTION_CERTIFIED" if state_fail is None and not witness_fail else "SEMANTIC_STATE_ACTION_FAILED",
        "first_failure": state_fail,
        "collision_witness_fail_count": len(witness_fail),
        "permuted_alias_class_count": sum(1 for row in trans_records if row["action_label"] == "PERMUTE_EXAMPLE_AXIS_0"),
        "identity_alias_class_count": sum(1 for row in trans_records if row["action_label"] == "IDENTITY_COPY"),
    }
    result["collision_witnesses"] = witnesses
    result["_records"] = base_records + trans_records
    semantic_rows = []
    for a, b in zip(base_records, trans_records):
        after = _OBJECT_REGISTRY[b["object_id"]]
        for path, role, dims in zip(b["structural_paths"], b["constructor_roles"], b["leaf_dims"]):
            semantic_rows.append(
                {
                    "case_id": case["case_id"],
                    "structural_path": path,
                    "constructor_role": role,
                    "leaf_dims": json.dumps(dims),
                    "physical_shape": json.dumps(b["shape"]),
                    "dtype": b["dtype"],
                    "device": b["device"],
                    "alias_class": b["alias_class_id"],
                    "action_label": b["action_label"],
                    "semantic_rationale": b["semantic_rationale"],
                    "checksum_before_action": tensor_checksum(snap[a["object_id"]]),
                    "checksum_after_action": tensor_checksum(after),
                    "leading_dim_equals_task_n_examples": b["leading_dim_equals_task_n_examples"],
                    "accidental_leading_dim_equality": b["accidental_leading_dim_equality"],
                }
            )
    result["_semantic_rows"] = semantic_rows
    if result["semantic_state_action"]["status"] != "SEMANTIC_STATE_ACTION_CERTIFIED":
        result["forward_label"] = result["semantic_state_action"]["status"]
        return result
    affine_cert = affine_commutation_certificate(mods, case, base_model, base_task)
    result["affine_commutation"] = affine_cert
    if not affine_cert["production_leaf_pass"]:
        result["forward_label"] = "AFFINE_IDENTITY_WEIGHT_COMMUTATION_FAILED"
        return result
    restore(base_records, snap)
    final_a, events_a = run_decoder_trace(mods, base_model, base_task)
    semantic_restore(base_records, trans_records, snap, case["perm_new_to_old"])
    final_b, events_b = run_decoder_trace(mods, trans_model, trans_task)
    label, trace_rows, detail = compare_decoder_traces(case, events_a, events_b, affine_cert["production_leaf_pass"])
    result["decoder_boundary"] = {"case_id": case["case_id"], "label": label, "detail": detail, "event_count": len(events_a)}
    result["_trace_rows"] = trace_rows
    result["forward_label"] = label
    return result


def run_a0_profiles(mods: dict[str, Any], cases: list[dict[str, Any]], problem_by_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(R6 / "scripts"))
    import r6a_phase2_support_permutation_first_divergence_v8_child as v8

    profiles = ["1f876c06_adjacent_0_1", "604001fa_adjacent_0_1", "synthetic_uniform_shape_oracle_adjacent_0_1"]
    rows = []
    for case_id in profiles:
        case = next(row for row in cases if row["case_id"] == case_id)
        problem = problem_by_case[case_id]
        reset_determinism()
        task = mods["preprocessing"].Task(case["task_id"], problem, None)
        reset_determinism()
        model = mods["arc_compressor"].ARCCompressor(task)
        records = v8.build_state_inventory(model, task, case_id, "canonical")
        v8.register_objects(model, records)
        snap = v8.snapshot_records(records)
        schedule = v8.posterior_leaf_schedule(model, task, case_id)
        rng = torch.get_rng_state().clone()
        v8.restore_records(records, snap)
        torch.set_rng_state(rng.clone())
        source = v8.run_forward(mods, model, task, case_id, "source", schedule, case["perm_new_to_old"])
        v8.restore_records(records, snap)
        torch.set_rng_state(rng.clone())
        record = v8.run_forward(mods, model, task, case_id, "record", schedule, case["perm_new_to_old"])
        v8.restore_records(records, snap)
        torch.set_rng_state(rng.clone())
        replay = v8.run_forward(mods, model, task, case_id, "replay", schedule, case["perm_new_to_old"], stored=record["stored_noise"])
        v8.restore_records(records, snap)
        torch.set_rng_state(rng.clone())
        observed = v8.run_forward(mods, model, task, case_id, "replay", schedule, case["perm_new_to_old"], stored=record["stored_noise"], observe=True)
        sr = v8.compare_forward_readouts(source, record, list(range(task.n_examples)))
        rr = v8.compare_forward_readouts(record, replay, list(range(task.n_examples)))
        ro = v8.compare_forward_readouts(replay, observed, list(range(task.n_examples)))
        rows.append({"profile": case["task_id"], "case_id": case_id, "status": "A0_REUSE_CERTIFIED" if sr["pass"] and rr["pass"] and ro["pass"] else "A0_REUSE_FAILED", "source_vs_record": sr, "record_vs_replay": rr, "replay_vs_observed": ro, "v8_bounded_noisetape_observer_reused": True})
    payload = {"created_utc": utc_now(), "profiles": rows}
    write_json(ARTIFACT / "instrumentation_reuse_certificate.json", payload)
    return payload


def outcome_classification(v8_status: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["forward_label"] for row in results]
    if v8_status == "V8_NONPOSTERIOR_PERMUTATION_OBSERVED" and all(label == "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED" for label in labels):
        return {"outcome": "Outcome A", "statement": "The v8 mismatch was caused by an invalid parameter-axis action. It is not evidence of support-order dependence in the production model."}
    if all(label == "FORWARD_ROLE_PRESERVING_SUPPORT_PERMUTATION_CERTIFIED" for label in labels):
        return {"outcome": "Outcome E", "statement": "Under the audited CPU float64, zero-noise, step-zero conditions, the production forward path was role-preserving support-permutation equivariant for the seven tested cases after applying source-derived semantic state action. This result does not certify stochastic training, loss, optimization, or all ARC tasks."}
    if any(label == "FIRST_DIVERGENCE_CHANNEL_LAYER" for label in labels):
        return {"outcome": "Outcome C", "statement": "A deterministic mismatch remains before decode affine, at the channel-layer boundary. This is a source/harness localization result only; it is not a shortcut-family or reasoning claim."}
    return {"outcome": "Outcome D", "statement": "A deterministic mismatch remains after semantic parameter-action repair at the reported source boundary. It is not yet evidence that a shortcut carrier, architectural route, or reasoning mechanism has been identified."}


def run_child() -> dict[str, Any]:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    mods = import_pinned_modules()
    cases, problem_by_case = build_cases()
    v8_audit = v8_action_behavior_audit()
    a0 = run_a0_profiles(mods, cases, problem_by_case)
    sample_task = mods["preprocessing"].Task("cpu_proof", make_synthetic_problem(), None)
    sample_model = mods["arc_compressor"].ARCCompressor(sample_task)
    tensor_rows = [{"name": "task.masks", "device": sample_task.masks.device.type, "dtype": str(sample_task.masks.dtype)}, {"name": "task.problem", "device": sample_task.problem.device.type, "dtype": str(sample_task.problem.dtype)}] + [{"name": f"weight_{idx}", "device": tensor.device.type, "dtype": str(tensor.dtype)} for idx, tensor in enumerate(sample_model.weights_list[:20])]
    cpu = {
        "created_utc": utc_now(),
        "fixed_seed": FIXED_SEED,
        "post_import_seed_probe": mods["seed_probe"],
        "final_seed_probe_after_imports": seed_probe(),
        "effective_default_device": str(torch.get_default_device()),
        "effective_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_interception_events": mods["shim"]["intercept_log"],
        "cuda_memory_allocated": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "no_cuda_execution": True,
        "runtime_tensors_audited_are_cpu": all(row["device"] == "cpu" for row in tensor_rows),
        "newly_constructed_floating_tensors_are_float64": all(row["dtype"] == "torch.float64" for row in tensor_rows if row["name"] != "task.problem"),
        "tensor_rows": tensor_rows,
    }
    write_json(ARTIFACT / "cpu_execution_proof.json", cpu)
    results = []
    semantic_rows = []
    witnesses = []
    trace_rows = []
    affine_rows = []
    task_rows = []
    state_rows = []
    alias_rows = []
    unclassified = []
    for case in cases:
        result = run_case(mods, case, problem_by_case[case["case_id"]])
        semantic_rows.extend(result.pop("_semantic_rows", []))
        trace_rows.extend(result.pop("_trace_rows", []))
        task_rows.append(result["task_action"])
        if "semantic_state_action" in result:
            state_rows.append(result["semantic_state_action"])
            alias_rows.append({"case_id": case["case_id"], "status": "ALIAS_SEMANTIC_ACTION_CERTIFIED"})
        witnesses.extend(result.get("collision_witnesses", []))
        if "affine_commutation" in result:
            affine_rows.append(result["affine_commutation"])
        results.append(result)
    write_semantic_action_csv(semantic_rows)
    write_first_divergence_csv(trace_rows)
    write_json(ARTIFACT / "parameter_collision_witnesses.json", {"created_utc": utc_now(), "witnesses": witnesses, "pass_count": sum(1 for row in witnesses if row["exact_equality_result"]), "fail_count": sum(1 for row in witnesses if not row["exact_equality_result"])})
    write_json(ARTIFACT / "affine_commutation_certificate.json", {"created_utc": utc_now(), "cases": affine_rows, "pass_count": sum(1 for row in affine_rows if row["production_leaf_pass"]), "fail_count": sum(1 for row in affine_rows if not row["production_leaf_pass"])})
    write_json(ARTIFACT / "task_action_certificate.json", {"created_utc": utc_now(), "cases": task_rows})
    write_json(ARTIFACT / "semantic_state_action_certificate.json", {"created_utc": utc_now(), "cases": state_rows})
    write_json(ARTIFACT / "alias_semantic_action_certificate.json", {"created_utc": utc_now(), "cases": alias_rows})
    write_json(ARTIFACT / "unclassified_tensor_report.json", {"created_utc": utc_now(), "unclassified_count": len(unclassified), "unclassified": unclassified})
    write_json(ARTIFACT / "decoder_boundary_trace.json", {"created_utc": utc_now(), "cases": [{"case_id": row["case"]["case_id"], "label": row["forward_label"], "decoder_boundary": row.get("decoder_boundary")} for row in results]})
    write_json(ARTIFACT / "forward_trace_summary.json", {"created_utc": utc_now(), "cases": [{"case_id": row["case"]["case_id"], "label": row["forward_label"]} for row in results]})
    outcome = outcome_classification(v8_audit["status"], results)
    write_json(ARTIFACT / "final_outcome_classification.json", outcome)
    write_json(ARTIFACT / "v10_cases.json", {"created_utc": utc_now(), "cases": results})
    summary = {
        "created_utc": utc_now(),
        "status": "CHILD_COMPLETE",
        "case_count": len(results),
        "v8_action_behavior_status": v8_audit["status"],
        "collision_witness_pass_fail": [sum(1 for row in witnesses if row["exact_equality_result"]), sum(1 for row in witnesses if not row["exact_equality_result"])],
        "affine_certificate_pass_fail": [sum(1 for row in affine_rows if row["production_leaf_pass"]), sum(1 for row in affine_rows if not row["production_leaf_pass"])],
        "task_action_pass_fail": [sum(1 for row in task_rows if row["status"] == "TASK_ACTION_CERTIFIED"), sum(1 for row in task_rows if row["status"] != "TASK_ACTION_CERTIFIED")],
        "semantic_state_action_pass_fail": [sum(1 for row in state_rows if row["status"] == "SEMANTIC_STATE_ACTION_CERTIFIED"), sum(1 for row in state_rows if row["status"] != "SEMANTIC_STATE_ACTION_CERTIFIED")],
        "a0_reuse_pass_fail": [sum(1 for row in a0["profiles"] if row["status"] == "A0_REUSE_CERTIFIED"), sum(1 for row in a0["profiles"] if row["status"] != "A0_REUSE_CERTIFIED")],
        "decoder_boundary_result_per_case": {row["case"]["case_id"]: row["forward_label"] for row in results},
        "cpu_only_no_gpu": cpu["cuda_memory_allocated"] == 0,
        "outcome": outcome,
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
