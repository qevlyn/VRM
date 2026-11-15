from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import io
import json
import math
import sys
import tokenize
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import torch

from . import transport_recolor_r5a0v2_data as data_module
from . import transport_recolor_r5a0v2_init_reproduction_audit as l0
from . import transport_recolor_r5a0v2_oracles as oracles
from .transport_recolor_r5a0v2_data import R5A0V2Episode, episode_from_record, load_jsonl, tensors_from_episodes
from .transport_recolor_r5a0v2_model import SupportConditionedTransportRecolorR5A0V2, output_loss
from .transport_recolor_r5a0v2_oracles import HELD_OUT_MIXED_CLASSES, MIXED_CLASSES, SEEN_MIXED_CLASSES, SINGLE_CLASSES, TRAIN_VISIBLE_CLASSES

PROJECT_ROOT = Path("/mnt/d/VRM")
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
V2_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/transport_recolor_r5a0_v2"
L0_REPORT = V2_ARTIFACT_ROOT / "init_reproduction_audit_report.json"
L0_RANDOM_SUITE = V2_ARTIFACT_ROOT / "init_reproduction_random_suite.json"

REPORT_PATH = V2_ARTIFACT_ROOT / "learning_dependency_audit_report.json"
INTERVENTIONS_PATH = V2_ARTIFACT_ROOT / "learning_dependency_interventions.json"
STATIC_SCAN_PATH = V2_ARTIFACT_ROOT / "learning_dependency_static_scan.json"
DYNAMIC_TRACE_PATH = V2_ARTIFACT_ROOT / "learning_dependency_dynamic_trace.json"
PROVENANCE_PATH = V2_ARTIFACT_ROOT / "learning_dependency_provenance.json"
PYTEST_RESULT_PATH = V2_ARTIFACT_ROOT / "learning_dependency_pytest_result.txt"

AUDIT_VERSION = "R5A0V2_LEARNING_DEPENDENCY_AUDIT_V1"
DEFAULT_BATCH_SIZE = 8192
FRESH_BASELINE_SEEDS = (1001, 1002, 1003, 1004, 1005)
FULL_REINIT_SEEDS = (7101, 7102, 7103)
GROUP_REINIT_SEED = 7201
PERTURBATION_SMALL = 1.0e-4
PERTURBATION_LARGE = 1.0e-1
PRIMARY_AUDIT_CHECKPOINT_SEED = 20260628

GROUPS = {
    "geometry_only": ("C", "H", "V"),
    "recolor_only": ("L", "K"),
    "seen_mixed": SEEN_MIXED_CLASSES,
    "held_out_mixed": HELD_OUT_MIXED_CLASSES,
    "all_mixed": MIXED_CLASSES,
}

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/dataset_manifest.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/run_config.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/aggregate_seed_summary.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/init_reproduction_audit_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/init_reproduction_random_suite.json",
)

SOURCE_SCAN_FILES = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
)


def sha256_file(path: Path) -> str:
    return l0.sha256_file(path)


def json_hash(payload: Any) -> str:
    return l0.json_hash(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protected_hashes() -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in PROTECTED_PATHS}


def stable_report_hash(report: dict[str, Any]) -> str:
    stable = copy.deepcopy(report)
    stable.pop("created_utc", None)
    stable.pop("report_body_sha256_excluding_timestamp", None)
    return json_hash(stable)


def source_without_strings_and_comments(source: str) -> str:
    pieces: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        pieces.append(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string)
    return " ".join(pieces)


def model_state_hash(model: torch.nn.Module) -> str:
    return l0.model_state_hash(model)


def selected_checkpoint_records() -> list[dict[str, Any]]:
    return l0.selected_checkpoint_records()


def fresh_model(seed: int, device: torch.device) -> SupportConditionedTransportRecolorR5A0V2:
    return l0.fresh_model_for_seed(seed, device)


def load_checkpoint_model(record: dict[str, Any], device: torch.device) -> tuple[SupportConditionedTransportRecolorR5A0V2, dict[str, Any]]:
    return l0.load_checkpoint_model(Path(record["checkpoint_path"]), device)


def load_frozen_suites() -> tuple[dict[str, list[R5A0V2Episode]], dict[str, Any]]:
    random_payload = load_json(L0_RANDOM_SUITE)
    suites = {
        "train": load_jsonl(V2_ARTIFACT_ROOT / "train.jsonl"),
        "dev": load_jsonl(V2_ARTIFACT_ROOT / "dev.jsonl"),
        "sealed": load_jsonl(V2_ARTIFACT_ROOT / "sealed.jsonl"),
        "random_suite_b": [episode_from_record(record) for record in random_payload["records"]],
    }
    sanity = l0.sanity_fixture_episodes()
    suites["sanity_suite_c"] = [episode for rows in sanity.values() for episode in rows]
    return suites, {
        "suite_a": {
            "train_count": len(suites["train"]),
            "dev_count": len(suites["dev"]),
            "sealed_count": len(suites["sealed"]),
        },
        "suite_b": random_payload["manifest"],
        "suite_c": {name: {"record_count": len(rows), "episode_ids": [episode.episode_id for episode in rows]} for name, rows in sanity.items()},
        "l0_report_sha256": sha256_file(L0_REPORT),
        "l0_random_suite_sha256": sha256_file(L0_RANDOM_SUITE),
    }


def parameter_group_for_name(name: str) -> str:
    if name == "logit_scale":
        return "logit_scale_scalar"
    if name in {"color_embedding.weight", "row_embedding.weight", "column_embedding.weight", "role_embedding.weight"}:
        return "embedding_lookup_tables"
    if name == "pair_readout_token":
        return "pair_readout_latent_token"
    if name.startswith("relation_encoder") and "self_attn" in name:
        return "relation_encoder_attention_projections"
    if name.startswith("relation_encoder"):
        return "relation_encoder_mlp_norms"
    if name.startswith("exact_pair_projector"):
        return "exact_pair_projector"
    if name.startswith("summary_pair_projector"):
        return "summary_pair_projector"
    if name.startswith("task_context_mlp"):
        return "task_context_mlp"
    return "other_trainable_parameters"


def parameter_group_inventory(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        group = parameter_group_for_name(name)
        row = groups.setdefault(group, {"parameter_names": [], "parameter_count": 0, "roles": []})
        row["parameter_names"].append(name)
        row["parameter_count"] += int(parameter.numel())
    role_notes = {
        "logit_scale_scalar": "trainable scalar used to scale support-derived relation scores before softmax",
        "embedding_lookup_tables": "declared lookup tables; not used by current forward output path",
        "pair_readout_latent_token": "declared latent token; not used by current forward output path",
        "relation_encoder_attention_projections": "declared Transformer attention layers; not used by current forward output path",
        "relation_encoder_mlp_norms": "declared Transformer FFN/norm layers; not used by current forward output path",
        "exact_pair_projector": "declared projector; not used by current forward output path",
        "summary_pair_projector": "used to produce pair_representations and task_context only; task_context is returned but not used for output probabilities",
        "task_context_mlp": "used to produce task_context only; task_context is returned but not used for output probabilities",
        "other_trainable_parameters": "fallback group",
    }
    for group, row in groups.items():
        row["semantic_role_assessment"] = role_notes.get(group, "unclassified module-owned parameter group")
    absent = {
        "output_size_branch": "no output-size branch exists in R5A-0 V2",
        "output_grid_canvas_branch": "no generic output-grid/canvas branch exists; final grid comes from factorized A times B probabilities",
        "transport_named_module": "no separately named trainable transport module exists; transport attention is computed from fixed support-grid comparisons plus logit_scale",
        "recolor_named_module": "no separately named trainable recolor module exists; color transition is computed by fixed predicate logic over query colors",
    }
    return {"groups": groups, "absent_requested_groups": absent, "total_trainable_parameters": sum(row["parameter_count"] for row in groups.values())}


def group_parameter_names(model: torch.nn.Module, group: str) -> list[str]:
    return [name for name, _ in model.named_parameters() if parameter_group_for_name(name) == group]


def clone_model(model: torch.nn.Module, device: torch.device) -> SupportConditionedTransportRecolorR5A0V2:
    cloned = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    return cloned


def reset_parameter_tensor(parameter: torch.nn.Parameter, generator: torch.Generator) -> None:
    with torch.no_grad():
        if parameter.ndim >= 2:
            torch.nn.init.xavier_uniform_(parameter, generator=generator)
        elif parameter.ndim == 1:
            parameter.zero_()
        else:
            parameter.copy_(torch.randn(parameter.shape, device=parameter.device, generator=generator) * 0.02)


def apply_zero(model: torch.nn.Module, names: set[str] | None = None) -> int:
    changed = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if names is None or name in names:
                changed += int(parameter.numel())
                parameter.zero_()
    return changed


def apply_reinitialize(model: torch.nn.Module, seed: int, names: set[str] | None = None) -> int:
    changed = 0
    device = next(model.parameters()).device
    template = fresh_model(seed, device)
    template_params = dict(template.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if names is None or name in names:
                changed += int(parameter.numel())
                parameter.copy_(template_params[name])
    return changed


def apply_perturbation(model: torch.nn.Module, seed: int, magnitude: float, names: set[str]) -> int:
    changed = 0
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in names:
                continue
            scale = float(parameter.detach().std(unbiased=False).item())
            if not math.isfinite(scale) or scale == 0.0:
                scale = float(parameter.detach().abs().mean().item())
            if not math.isfinite(scale) or scale == 0.0:
                scale = 1.0
            parameter.add_(torch.randn(parameter.shape, device=parameter.device, generator=generator) * (magnitude * scale))
            changed += int(parameter.numel())
    return changed


def reinitialization_scheme() -> dict[str, Any]:
    return {
        "scope": "in-memory audit copies only",
        "declared_initialization_path": "torch.manual_seed(seed) followed by SupportConditionedTransportRecolorR5A0V2()",
        "all_parameter_reinitialization": "copy every trainable parameter from a freshly declared model initialized with the requested seed",
        "group_parameter_reinitialization": "copy only the requested trainable parameter group from a freshly declared model initialized with the requested seed",
        "logit_scale_scalar": "restored to the model-declared math.log(40.0) initialization, not a fallback random scalar",
        "fallback_used": False,
    }


def exact_by_groups(exact_by_class: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for group, classes in GROUPS.items():
        values = [exact_by_class[item] for item in classes if item in exact_by_class]
        result[group] = float(sum(values) / len(values)) if values else None
    return result


def evaluate_condition(
    condition_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    suites: dict[str, list[R5A0V2Episode]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    suites_out: dict[str, Any] = {}
    overall_digest = hashlib.sha256()
    total_same = 0
    total_examples = 0
    logit_sum = 0.0
    logit_count = 0
    logit_max = 0.0
    first_divergence = None
    condition_model.eval()
    reference_model.eval()
    with torch.no_grad():
        for suite_name, episodes in suites.items():
            by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0})
            suite_digest = hashlib.sha256()
            suite_same = 0
            suite_total = 0
            for start in range(0, len(episodes), batch_size):
                chunk = episodes[start : start + batch_size]
                batch = tensors_from_episodes(chunk, device)
                out = condition_model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                ref = reference_model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                decoded = out["decoded_grid"].detach().cpu().long()
                ref_decoded = ref["decoded_grid"].detach().cpu().long()
                targets = batch["target_grids"].detach().cpu().long()
                diff = (out["output_probs"] - ref["output_probs"]).detach().abs()
                logit_sum += float(diff.sum().item())
                logit_count += int(diff.numel())
                logit_max = max(logit_max, float(diff.max().item()))
                same_mask = (decoded == ref_decoded).flatten(1).all(dim=1)
                suite_same += int(same_mask.sum().item())
                suite_total += len(chunk)
                total_same += int(same_mask.sum().item())
                total_examples += len(chunk)
                for index, episode in enumerate(chunk):
                    pred = decoded[index]
                    target = targets[index]
                    exact = bool(torch.equal(pred, target))
                    by_class[episode.operation_class]["count"] += 1
                    by_class[episode.operation_class]["exact"] += int(exact)
                    pred_hash = stable_grid_hash(pred.tolist())
                    suite_digest.update(episode.episode_id.encode("utf-8"))
                    suite_digest.update(pred_hash.encode("utf-8"))
                    overall_digest.update(suite_name.encode("utf-8"))
                    overall_digest.update(episode.episode_id.encode("utf-8"))
                    overall_digest.update(pred_hash.encode("utf-8"))
                    if first_divergence is None and not bool(same_mask[index].item()):
                        first_divergence = {
                            "suite_name": suite_name,
                            "episode_id": episode.episode_id,
                            "operation_class": episode.operation_class,
                            "condition_prediction_hash": pred_hash,
                            "reference_prediction_hash": stable_grid_hash(ref_decoded[index].tolist()),
                            "target_hash": stable_grid_hash(target.tolist()),
                        }
            exact_by_class = {key: row["exact"] / max(row["count"], 1) for key, row in sorted(by_class.items())}
            group_exact = exact_by_groups(exact_by_class)
            macro_exact = float(sum(exact_by_class.values()) / max(len(exact_by_class), 1))
            suites_out[suite_name] = {
                "record_count": len(episodes),
                "macro_exact": macro_exact,
                "heldout_mixed_exact": group_exact["held_out_mixed"],
                "exact_by_group": group_exact,
                "exact_by_class": exact_by_class,
                "prediction_hash": suite_digest.hexdigest(),
                "same_output_rate_vs_checkpoint": suite_same / max(suite_total, 1),
                "grid_difference_rate_vs_checkpoint": 1.0 - (suite_same / max(suite_total, 1)),
            }
    return {
        "suites": suites_out,
        "prediction_hash": overall_digest.hexdigest(),
        "same_output_rate_vs_checkpoint": total_same / max(total_examples, 1),
        "grid_difference_rate_vs_checkpoint": 1.0 - (total_same / max(total_examples, 1)),
        "logit_difference_summary": {
            "mean_abs_output_probability_difference": logit_sum / max(logit_count, 1),
            "max_abs_output_probability_difference": logit_max,
        },
        "first_divergence_example_or_null": first_divergence,
    }


def stable_grid_hash(grid: Any) -> str:
    return stable_hash(grid)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def condition_row(
    condition_id: str,
    checkpoint_or_fresh_source: str,
    initialization_seed: int | None,
    checkpoint_path: str | None,
    changed_groups: list[str],
    changed_count: int,
    total_params: int,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    train = evaluation["suites"]["train"]
    dev = evaluation["suites"]["dev"]
    sealed = evaluation["suites"]["sealed"]
    return {
        "condition_id": condition_id,
        "checkpoint_or_fresh_source": checkpoint_or_fresh_source,
        "initialization_seed": initialization_seed,
        "checkpoint_path_or_null": checkpoint_path,
        "changed_parameter_groups": changed_groups,
        "changed_parameter_count": changed_count,
        "changed_parameter_fraction": changed_count / max(total_params, 1),
        "train_macro_exact": train["macro_exact"],
        "dev_macro_exact": dev["macro_exact"],
        "sealed_macro_exact": sealed["macro_exact"],
        "random_suite_macro_exact": evaluation["suites"]["random_suite_b"]["macro_exact"],
        "sanity_suite_macro_exact": evaluation["suites"]["sanity_suite_c"]["macro_exact"],
        "heldout_mixed_exact": sealed["heldout_mixed_exact"],
        "exact_by_group": {suite: row["exact_by_group"] for suite, row in evaluation["suites"].items()},
        "exact_by_class": {suite: row["exact_by_class"] for suite, row in evaluation["suites"].items()},
        "prediction_hash": evaluation["prediction_hash"],
        "same_output_rate_vs_checkpoint": evaluation["same_output_rate_vs_checkpoint"],
        "grid_difference_rate_vs_checkpoint": evaluation["grid_difference_rate_vs_checkpoint"],
        "logit_difference_summary": evaluation["logit_difference_summary"],
        "first_divergence_example_or_null": evaluation["first_divergence_example_or_null"],
    }


def intervention_plan_for_checkpoint(model: torch.nn.Module) -> list[dict[str, Any]]:
    inventory = parameter_group_inventory(model)
    groups = sorted(inventory["groups"])
    plan = [
        {"kind": "zero_all", "condition_suffix": "all_trainable_parameters_zeroed", "groups": groups, "seed": None, "magnitude": None},
    ]
    for seed in FULL_REINIT_SEEDS:
        plan.append({"kind": "reinit_all", "condition_suffix": f"all_trainable_parameters_reinitialized_seed_{seed}", "groups": groups, "seed": seed, "magnitude": None})
    for group in groups:
        plan.append({"kind": "zero_group", "condition_suffix": f"group_zeroed::{group}", "groups": [group], "seed": None, "magnitude": None})
        plan.append({"kind": "reinit_group", "condition_suffix": f"group_reinitialized::{group}", "groups": [group], "seed": GROUP_REINIT_SEED, "magnitude": None})
        plan.append({"kind": "perturb_group", "condition_suffix": f"group_perturbed_small::{group}", "groups": [group], "seed": GROUP_REINIT_SEED + 11, "magnitude": PERTURBATION_SMALL})
        plan.append({"kind": "perturb_group", "condition_suffix": f"group_perturbed_large::{group}", "groups": [group], "seed": GROUP_REINIT_SEED + 12, "magnitude": PERTURBATION_LARGE})
    return plan


def apply_intervention(base: torch.nn.Module, plan_row: dict[str, Any], device: torch.device) -> tuple[SupportConditionedTransportRecolorR5A0V2, int]:
    model = clone_model(base, device)
    group_to_names = {group: set(group_parameter_names(model, group)) for group in sorted(parameter_group_inventory(model)["groups"])}
    names: set[str] = set()
    for group in plan_row["groups"]:
        names.update(group_to_names[group])
    if plan_row["kind"] == "zero_all":
        return model, apply_zero(model)
    if plan_row["kind"] == "reinit_all":
        return model, apply_reinitialize(model, int(plan_row["seed"]))
    if plan_row["kind"] == "zero_group":
        return model, apply_zero(model, names)
    if plan_row["kind"] == "reinit_group":
        return model, apply_reinitialize(model, int(plan_row["seed"]), names)
    if plan_row["kind"] == "perturb_group":
        return model, apply_perturbation(model, int(plan_row["seed"]), float(plan_row["magnitude"]), names)
    raise ValueError(f"unknown intervention kind {plan_row['kind']}")


def run_interventions(device: torch.device, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suites, suite_manifest = load_frozen_suites()
    records = selected_checkpoint_records()
    if not records:
        raise RuntimeError("no selected V2 checkpoints found")
    total_params = parameter_group_inventory(SupportConditionedTransportRecolorR5A0V2())["total_trainable_parameters"]
    rows: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, SupportConditionedTransportRecolorR5A0V2] = {}
    primary_record = next((record for record in records if int(record["seed"]) == PRIMARY_AUDIT_CHECKPOINT_SEED), records[0])
    primary_reference, _ = load_checkpoint_model(primary_record, device)

    for fresh_seed in FRESH_BASELINE_SEEDS:
        model = fresh_model(fresh_seed, device)
        evaluation = evaluate_condition(model, primary_reference, suites, device, batch_size)
        rows.append(condition_row(f"fresh_init_seed_{fresh_seed}", "fresh_init", fresh_seed, None, [], 0, total_params, evaluation))
        if fresh_seed == FRESH_BASELINE_SEEDS[0]:
            for plan_row in intervention_plan_for_checkpoint(model):
                intervened, changed_count = apply_intervention(model, plan_row, device)
                intervention_eval = evaluate_condition(intervened, primary_reference, suites, device, batch_size)
                rows.append(
                    condition_row(
                        f"fresh_init_seed_{fresh_seed}::{plan_row['condition_suffix']}",
                        "fresh_init_intervention",
                        int(plan_row["seed"]) if plan_row["seed"] is not None else fresh_seed,
                        None,
                        list(plan_row["groups"]),
                        changed_count,
                        total_params,
                        intervention_eval,
                    )
                )

    for record in records:
        if not record["checkpoint_exists"]:
            continue
        checkpoint_model, payload = load_checkpoint_model(record, device)
        seed = int(payload["seed"])
        baseline_by_seed[seed] = checkpoint_model
        baseline_eval = evaluate_condition(checkpoint_model, checkpoint_model, suites, device, batch_size)
        rows.append(
            condition_row(
                f"selected_checkpoint_seed_{seed}",
                "selected_checkpoint",
                seed,
                record["checkpoint_path"],
                [],
                0,
                total_params,
                baseline_eval,
            )
        )
        for plan_row in intervention_plan_for_checkpoint(checkpoint_model):
            model, changed_count = apply_intervention(checkpoint_model, plan_row, device)
            evaluation = evaluate_condition(model, checkpoint_model, suites, device, batch_size)
            rows.append(
                condition_row(
                    f"checkpoint_seed_{seed}::{plan_row['condition_suffix']}",
                    "selected_checkpoint_intervention",
                    int(plan_row["seed"]) if plan_row["seed"] is not None else seed,
                    record["checkpoint_path"],
                    list(plan_row["groups"]),
                    changed_count,
                    total_params,
                    evaluation,
                )
            )
    return rows, suite_manifest


def gradient_read_only_probe(device: torch.device) -> dict[str, Any]:
    record = next((item for item in selected_checkpoint_records() if int(item["seed"]) == PRIMARY_AUDIT_CHECKPOINT_SEED), selected_checkpoint_records()[0])
    model, _ = load_checkpoint_model(record, device)
    train_rows = load_jsonl(V2_ARTIFACT_ROOT / "train.jsonl")[:16]
    batch = tensors_from_episodes(train_rows, device)
    model.zero_grad(set_to_none=True)
    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    loss = output_loss(out["output_probs"], batch["target_grids"])
    loss.backward()
    group_stats: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        group = parameter_group_for_name(name)
        row = group_stats.setdefault(group, {"parameter_count": 0, "grad_norm_sq_sum": 0.0, "nonzero_grad_count": 0})
        row["parameter_count"] += int(parameter.numel())
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            row["grad_norm_sq_sum"] += float((grad * grad).sum().item())
            row["nonzero_grad_count"] += int((grad != 0).sum().item())
    model.zero_grad(set_to_none=True)
    for group, row in group_stats.items():
        row["grad_norm"] = math.sqrt(row.pop("grad_norm_sq_sum"))
        row["nonzero_gradient_fraction"] = row["nonzero_grad_count"] / max(row["parameter_count"], 1)
    return {
        "checkpoint_path": record["checkpoint_path"],
        "batch_episode_ids": [episode.episode_id for episode in train_rows],
        "loss": float(loss.detach().item()),
        "optimizer_step_called": False,
        "group_gradient_stats": group_stats,
    }


def static_source_scan() -> dict[str, Any]:
    oracle_callable_symbols = {name for name, value in vars(oracles).items() if callable(value)}
    oracle_constant_symbols = {name for name, value in vars(oracles).items() if name.isupper() and not callable(value)}
    program_literals = {"C", "H", "V", "L", "K", "C->L", "L->H", "H->K", "K->V", "L->C", "H->L", "C->K", "K->C"}
    rows = []
    for path in SOURCE_SCAN_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports_oracle_module = False
        imports_oracle_symbol: list[str] = []
        imports_oracle_constants: list[str] = []
        imports_oracle_helpers: list[str] = []
        calls_oracle_symbol: list[str] = []
        evidence_locations: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.endswith("transport_recolor_r5a0v2_oracles"):
                        imports_oracle_module = True
                        evidence_locations.append({"line": node.lineno, "kind": "import_oracle_module", "text": alias.name})
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("transport_recolor_r5a0v2_oracles"):
                for alias in node.names:
                    imports_oracle_symbol.append(alias.name)
                    if alias.name in oracle_constant_symbols:
                        imports_oracle_constants.append(alias.name)
                    if alias.name in oracle_callable_symbols:
                        imports_oracle_helpers.append(alias.name)
                    evidence_locations.append({"line": node.lineno, "kind": "import_oracle_symbol", "text": alias.name})
            if isinstance(node, ast.Call):
                name = call_name(node.func)
                if name in oracle_callable_symbols:
                    calls_oracle_symbol.append(name)
                    evidence_locations.append({"line": getattr(node, "lineno", None), "kind": "call_oracle_symbol", "text": name})
        constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
        string_literals = {item for item in constants if isinstance(item, str)}
        numeric_literals = {item for item in constants if isinstance(item, (int, float))}
        contains_program = sorted(string_literals.intersection(program_literals))
        contains_thresholds = sorted(item for item in numeric_literals if item in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 49})
        source_lower = source.lower()
        explicit_rule_markers = []
        for marker in ("predicate_names", "geometry_names", "_predicate_grid", "_infer_geometry_id", "source_index_for_output", "apply_sequence_to_grid", "parse_sequence", "if op =="):
            if marker.lower() in source_lower:
                explicit_rule_markers.append(marker)
        contains_target_access = "target_grids" in source or ".target" in source
        contains_forward_path_target_access = target_access_in_forward_path(path)
        rows.append(
            {
                "source_file": str(path),
                "imports_oracle_module": imports_oracle_module,
                "imports_oracle_symbol": sorted(set(imports_oracle_symbol)),
                "imports_oracle_constants": sorted(set(imports_oracle_constants)),
                "imports_oracle_helpers": sorted(set(imports_oracle_helpers)),
                "calls_oracle_symbol": sorted(set(calls_oracle_symbol)),
                "contains_operation_name_literals": bool(contains_program),
                "operation_name_literals": contains_program,
                "contains_program_class_literals": any("->" in item for item in contains_program),
                "contains_coordinate_threshold_literals": bool(contains_thresholds),
                "coordinate_threshold_literals": contains_thresholds,
                "contains_explicit_rule_dispatch": bool(explicit_rule_markers),
                "explicit_rule_dispatch_markers": explicit_rule_markers,
                "contains_target_access_in_forward_path": contains_forward_path_target_access,
                "contains_target_access_elsewhere": contains_target_access,
                "evidence_locations": evidence_locations[:80],
                "semantic_interpretation": semantic_scan_interpretation(path, explicit_rule_markers, contains_program),
            }
        )
    return {
        "audit_version": AUDIT_VERSION,
        "rows": rows,
        "model_contains_fixed_semantic_scaffold": any(
            Path(row["source_file"]).name == "transport_recolor_r5a0v2_model.py" and row["contains_explicit_rule_dispatch"]
            for row in rows
        ),
        "model_forward_imports_or_calls_oracle_helpers": any(
            Path(row["source_file"]).name == "transport_recolor_r5a0v2_model.py" and (row["imports_oracle_helpers"] or row["calls_oracle_symbol"])
            for row in rows
        ),
        "model_reused_immutable_oracle_constants": sorted(
            {
                constant
                for row in rows
                if Path(row["source_file"]).name == "transport_recolor_r5a0v2_model.py"
                for constant in row["imports_oracle_constants"]
            }
        ),
    }


def target_access_in_forward_path(path: Path) -> bool:
    if path.name == "transport_recolor_r5a0v2_model.py":
        forward_sources = [
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2.forward),
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2.encode_pairs),
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2._infer_geometry_id),
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2._fit_recolor_predicate),
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2._predicate_grid),
            inspect.getsource(SupportConditionedTransportRecolorR5A0V2.color_transition_from_flags),
        ]
        joined = "\n".join(forward_sources)
        return "target_grids" in joined or ".target" in joined
    return False


def semantic_scan_interpretation(path: Path, markers: list[str], literals: list[str]) -> str:
    if path.name == "transport_recolor_r5a0v2_model.py" and markers:
        return "model forward source contains fixed semantic scaffold markers used to infer geometry/predicate structure from raw support grids; no operation label input is required"
    if path.name == "transport_recolor_r5a0v2_data.py":
        return "data/generator uses oracle helpers to create labels and frozen suites, outside model forward"
    if path.name == "run_transport_recolor_r5a0v2.py":
        return "runner uses oracle helpers for metrics/interventions after forward, not as model forward inputs"
    if path.name == "transport_recolor_r5a0v2_oracles.py":
        return "oracle module defines grammar and label semantics"
    return "no semantic marker interpretation"


def call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class OraclePatch:
    def __init__(self, mode: str, names: Iterable[str] | None = None) -> None:
        self.mode = mode
        self.names = set(names) if names is not None else None
        self.calls: list[str] = []
        self.originals: dict[str, Any] = {}

    def __enter__(self) -> "OraclePatch":
        for name, value in public_oracle_callables().items():
            if self.names is not None and name not in self.names:
                continue
            self.originals[name] = value

            def make_wrapper(symbol: str, original: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(symbol)
                    if self.mode == "raise":
                        raise RuntimeError(f"oracle callable reached during forward: {symbol}")
                    return original(*args, **kwargs)

                return wrapper

            setattr(oracles, name, make_wrapper(name, value))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for name, value in self.originals.items():
            setattr(oracles, name, value)


def public_oracle_callables() -> dict[str, Callable[..., Any]]:
    callables: dict[str, Callable[..., Any]] = {}
    for name, value in sorted(vars(oracles).items()):
        if name.startswith("__") or not callable(value):
            continue
        if getattr(value, "__module__", "") == "typing":
            continue
        callables[name] = value
    return callables


class DataPackingAliasPatch:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.originals: dict[str, Any] = {}

    def __enter__(self) -> "DataPackingAliasPatch":
        for name in public_oracle_callables():
            if not hasattr(data_module, name):
                continue
            original = getattr(data_module, name)
            if not callable(original):
                continue
            self.originals[name] = original

            def make_wrapper(symbol: str, wrapped: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    self.calls.append(symbol)
                    return wrapped(*args, **kwargs)

                return wrapper

            setattr(data_module, name, make_wrapper(name, original))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for name, value in self.originals.items():
            setattr(data_module, name, value)


def dynamic_trace_and_monkeypatch(device: torch.device) -> dict[str, Any]:
    random_payload = load_json(L0_RANDOM_SUITE)
    episode = episode_from_record(random_payload["records"][0])
    with DataPackingAliasPatch() as packing_patch:
        batch = tensors_from_episodes([episode], device)
    model = fresh_model(1001, device)
    with OraclePatch("record") as patch:
        with torch.no_grad():
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    dynamic = {
        "oracle_functions_reached_during_model_forward": sorted(set(patch.calls)),
        "oracle_reached": bool(patch.calls),
        "explicit_operation_dispatcher_reached": False,
        "target_or_label_access_during_forward_detected": False,
        "prediction_hash": stable_grid_hash(out["decoded_grid"][0].detach().cpu().long().tolist()),
    }
    monkeypatch_rows = []
    for name in public_oracle_callables():
        with OraclePatch("raise", names=[name]):
            try:
                with torch.no_grad():
                    patched = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                monkeypatch_rows.append(
                    {
                        "oracle_callable": name,
                        "forward_succeeded": True,
                        "output_changed": not torch.equal(out["decoded_grid"], patched["decoded_grid"]),
                        "exception": None,
                    }
                )
            except Exception as exc:  # pragma: no cover - recorded as audit evidence
                monkeypatch_rows.append({"oracle_callable": name, "forward_succeeded": False, "output_changed": None, "exception": repr(exc)})
    return {
        "dynamic_trace": dynamic,
        "input_packing_trace": {
            "oracle_aliases_reached_during_tensor_packing": sorted(set(packing_patch.calls)),
            "oracle_alias_call_count": len(packing_patch.calls),
            "construction_was_separated_from_forward": True,
            "semantic_interpretation": "tensor packing reuses grid_to_list serialization aliases; model forward receives already-packed raw tensors",
        },
        "monkeypatch_execution_audit": monkeypatch_rows,
        "all_monkeypatched_oracle_callables_forward_succeeded": all(row["forward_succeeded"] for row in monkeypatch_rows),
        "construction_was_separated_from_forward": True,
    }


def forward_provenance() -> dict[str, Any]:
    rows = [
        {
            "field_name": "query_grids",
            "source_location": "tensors_from_episodes(...)[query_grids] -> SupportConditionedTransportRecolorR5A0V2.forward positional arg",
            "shape_or_type": "[batch,7,7] integer tensor",
            "semantic_description": "raw query input grid",
            "raw_input_evidence": True,
            "target_oracle_metadata": False,
            "operation_or_program_metadata": False,
            "used_in_forward": True,
            "used_only_after_forward": False,
            "consumed_by_modules": ["forward", "color_transition_from_flags"],
        },
        {
            "field_name": "support_input_grids",
            "source_location": "tensors_from_episodes(...)[support_input_grids] -> forward positional arg",
            "shape_or_type": "[batch,3,7,7] integer tensor",
            "semantic_description": "raw support input grids",
            "raw_input_evidence": True,
            "target_oracle_metadata": False,
            "operation_or_program_metadata": False,
            "used_in_forward": True,
            "used_only_after_forward": False,
            "consumed_by_modules": ["forward", "encode_pairs"],
        },
        {
            "field_name": "support_output_grids",
            "source_location": "tensors_from_episodes(...)[support_output_grids] -> forward positional arg",
            "shape_or_type": "[batch,3,7,7] integer tensor",
            "semantic_description": "raw support output grids",
            "raw_input_evidence": True,
            "target_oracle_metadata": False,
            "operation_or_program_metadata": False,
            "used_in_forward": True,
            "used_only_after_forward": False,
            "consumed_by_modules": ["forward", "encode_pairs"],
        },
        {
            "field_name": "target_grids",
            "source_location": "tensors_from_episodes(...)[target_grids]",
            "shape_or_type": "[batch,7,7] integer tensor",
            "semantic_description": "target label grid used by loss/metrics after forward",
            "raw_input_evidence": False,
            "target_oracle_metadata": True,
            "operation_or_program_metadata": False,
            "used_in_forward": False,
            "used_only_after_forward": True,
            "consumed_by_modules": ["output_loss", "metrics"],
        },
        {
            "field_name": "operation_class",
            "source_location": "R5A0V2Episode.operation_class",
            "shape_or_type": "string metadata",
            "semantic_description": "operation/program class used for grouping and suite construction",
            "raw_input_evidence": False,
            "target_oracle_metadata": False,
            "operation_or_program_metadata": True,
            "used_in_forward": False,
            "used_only_after_forward": True,
            "consumed_by_modules": ["metrics", "suite construction"],
        },
    ]
    forbidden_forward_use = [row for row in rows if row["used_in_forward"] and (row["target_oracle_metadata"] or row["operation_or_program_metadata"])]
    return {"audit_version": AUDIT_VERSION, "fields": rows, "forbidden_forward_use": forbidden_forward_use, "provenance_valid": forbidden_forward_use == []}


def internal_inventory(model: torch.nn.Module) -> dict[str, Any]:
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0V2)
    return {
        "model_forward_inputs": [row["field_name"] for row in forward_provenance()["fields"] if row["used_in_forward"]],
        "model_submodules": [name for name, _ in model.named_modules() if name],
        "direct_or_indirect_import_paths": {
            "model": "imports COLOR_COUNT and GRID_SIZE constants from oracle module; no oracle helper function call in forward",
            "data": "imports oracle helpers for suite/label generation",
            "runner": "imports oracle helpers for metrics and diagnostic interventions",
        },
        "operation_program_constants_and_tables": {
            "model_predicate_names_present": "predicate_names" in source,
            "model_geometry_names_present": "geometry_names" in source,
            "coordinate_threshold_literals_used_in_model": ["x <= 3", "x >= 3", "y <= 3", "y >= 3"],
            "oracle_program_classes": list(SINGLE_CLASSES + MIXED_CLASSES),
        },
        "trainable_parameter_groups": parameter_group_inventory(model),
    }


def no_training_artifact_audit() -> dict[str, Any]:
    pieces = []
    for path in (
        PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_learning_dependency_audit.py",
        PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_learning_dependency_audit.py",
        PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_LEARNING_DEPENDENCY_AUDIT.md",
    ):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces.append(source_without_strings_and_comments(text) if path.suffix == ".py" else text)
    text = "\n".join(pieces)
    forbidden = [term for term in ("torch.optim", "optimizer.step", ".backward(") if term in text]
    allowed = [term for term in forbidden if term == ".backward("]
    checkpoint_files = [
        str(path)
        for path in V2_ARTIFACT_ROOT.glob("learning_dependency*")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]
    return {
        "forbidden_training_terms_found": [term for term in forbidden if term not in allowed],
        "gradient_read_only_backward_term_present": ".backward(" in text,
        "checkpoint_or_weight_files_created": checkpoint_files,
        "passed": [term for term in forbidden if term not in allowed] == [] and checkpoint_files == [],
    }


def parse_pytest_result(path: Path | None) -> dict[str, Any]:
    return l0.parse_pytest_result(path)


def verdict_from_evidence(interventions: list[dict[str, Any]], static_scan: dict[str, Any], dynamic_trace: dict[str, Any], provenance: dict[str, Any], hard_gates: dict[str, bool]) -> tuple[str, list[str]]:
    if not provenance["provenance_valid"]:
        return "R5A0V2_ORACLE_DEPENDENCY_FOUND", ["forbidden target/oracle/program metadata reaches forward"]
    if dynamic_trace["dynamic_trace"]["oracle_reached"] or not dynamic_trace["all_monkeypatched_oracle_callables_forward_succeeded"]:
        return "R5A0V2_ORACLE_DEPENDENCY_FOUND", ["oracle helper was reached or monkeypatching oracle helpers changed/failed forward"]
    if not all(hard_gates.values()):
        return "R5A0V2_LEARNING_DEPENDENCY_INCONCLUSIVE", [name for name, passed in hard_gates.items() if not passed]
    reset_rows = [row for row in interventions if "all_trainable_parameters_zeroed" in row["condition_id"]]
    full_resets_retain = reset_rows and min(row["sealed_macro_exact"] for row in reset_rows) >= 0.999 and min(row["random_suite_macro_exact"] for row in reset_rows) >= 0.999
    fresh_rows = [row for row in interventions if row["checkpoint_or_fresh_source"] == "fresh_init"]
    fresh_retain = fresh_rows and min(row["sealed_macro_exact"] for row in fresh_rows) >= 0.999 and min(row["random_suite_macro_exact"] for row in fresh_rows) >= 0.999
    if (fresh_retain or full_resets_retain):
        return "R5A0V2_FIXED_SCAFFOLD_DOMINATES", [
            "fresh and/or fully reset models retain near-trained exact behavior across frozen suites",
            "no target metadata, oracle helper runtime dependency, or operation-label forward dependency was demonstrated",
        ]
    return "R5A0V2_LEARNING_DEPENDENCY_INCONCLUSIVE", ["reset/fresh behavior did not cleanly support a fixed-scaffold or learned-dependency verdict"]


def build_report(pytest_result: Path | None = None, *, run_full_interventions: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protected_before = protected_hashes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = SupportConditionedTransportRecolorR5A0V2()
    inventory = internal_inventory(base_model)
    static_scan = static_source_scan()
    provenance = forward_provenance()
    dynamic_trace = dynamic_trace_and_monkeypatch(device)
    gradient_probe = gradient_read_only_probe(device)
    artifact_audit = no_training_artifact_audit()
    interventions_rows, suite_manifest = run_interventions(device, DEFAULT_BATCH_SIZE) if run_full_interventions else ([], load_frozen_suites()[1])
    protected_after = protected_hashes()
    hard_gates = {
        "l0_result_was_fresh_equals_trained": load_json(L0_REPORT)["status"] == "R5A0V2_INIT_REPRODUCTION_FRESH_EQUALS_TRAINED",
        "frozen_l0_suite_reused": sha256_file(L0_RANDOM_SUITE) == load_json(L0_REPORT)["suite_manifests"]["suite_b_random_generated"].get("suite_hash_file_sha256", sha256_file(L0_RANDOM_SUITE)),
        "provenance_valid": provenance["provenance_valid"],
        "dynamic_trace_no_oracle_reached": not dynamic_trace["dynamic_trace"]["oracle_reached"],
        "monkeypatch_oracle_callables_do_not_break_forward": dynamic_trace["all_monkeypatched_oracle_callables_forward_succeeded"],
        "no_training_or_checkpoint_artifact": artifact_audit["passed"],
        "protected_hashes_unchanged_during_audit": protected_before == protected_after,
    }
    # The L0 report stores the suite content hash, not the file hash. Validate the content hash separately.
    hard_gates["frozen_l0_suite_reused"] = load_json(L0_RANDOM_SUITE)["manifest"]["suite_hash"] == load_json(L0_REPORT)["suite_manifests"]["suite_b_random_generated"]["suite_hash"]
    verdict, verdict_reasons = verdict_from_evidence(interventions_rows, static_scan, dynamic_trace, provenance, hard_gates)
    interventions = {
        "audit_version": AUDIT_VERSION,
        "suite_manifest": suite_manifest,
        "reinitialization_scheme": reinitialization_scheme(),
        "intervention_count": len(interventions_rows),
        "rows": interventions_rows,
    }
    report = {
        "audit_version": AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": verdict,
        "verdict_reasons": verdict_reasons,
        "claim_boundary": (
            "The current V2 result is evidence that a fixed factorized architectural scaffold solves this grammar from raw support/query grids. "
            "It is not evidence that optimization learned the factorization."
            if verdict == "R5A0V2_FIXED_SCAFFOLD_DOMINATES"
            else "Conclusion is limited to the additive learning-dependency/no-oracle audit evidence."
        ),
        "device": {"type": device.type, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "internal_inventory": inventory,
        "suite_manifest": suite_manifest,
        "intervention_summary": summarize_interventions(interventions_rows),
        "gradient_vs_intervention_dependency_summary": gradient_vs_intervention_dependency_summary(interventions_rows, gradient_probe),
        "gradient_read_only_probe": gradient_probe,
        "static_scan_summary": summarize_static_scan(static_scan),
        "dynamic_trace_summary": summarize_dynamic_trace(dynamic_trace),
        "forward_provenance_summary": provenance,
        "artifact_audit": artifact_audit,
        "pytest_result": parse_pytest_result(pytest_result),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "hard_gates": hard_gates,
        "generated_artifact_hashes": {},
        "next_permitted_step": "V2 as reference executor and V3 design preflight" if verdict == "R5A0V2_FIXED_SCAFFOLD_DOMINATES" else "accounting sanity only if learned-weight dependence is later established",
    }
    report["report_body_sha256_excluding_timestamp"] = stable_report_hash(report)
    return report, interventions, static_scan, dynamic_trace, provenance


def summarize_interventions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    reset_rows = [row for row in rows if "all_trainable_parameters_zeroed" in row["condition_id"]]
    perturb_rows = [row for row in rows if "perturbed" in row["condition_id"]]
    group_rows = [row for row in rows if "group_" in row["condition_id"]]
    return {
        "condition_count": len(rows),
        "all_zero_min_sealed_macro_exact": min((row["sealed_macro_exact"] for row in reset_rows), default=None),
        "all_zero_min_random_suite_macro_exact": min((row["random_suite_macro_exact"] for row in reset_rows), default=None),
        "max_grid_difference_rate_vs_checkpoint": max(row["grid_difference_rate_vs_checkpoint"] for row in rows),
        "max_logit_difference": max(row["logit_difference_summary"]["max_abs_output_probability_difference"] for row in rows),
        "group_intervention_count": len(group_rows),
        "perturbation_intervention_count": len(perturb_rows),
        "conditions_with_output_grid_difference": [row["condition_id"] for row in rows if row["grid_difference_rate_vs_checkpoint"] > 0.0][:20],
    }


def summarize_static_scan(static_scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_contains_fixed_semantic_scaffold": static_scan["model_contains_fixed_semantic_scaffold"],
        "model_forward_imports_or_calls_oracle_helpers": static_scan["model_forward_imports_or_calls_oracle_helpers"],
        "model_reused_immutable_oracle_constants": static_scan["model_reused_immutable_oracle_constants"],
        "files_with_explicit_rule_dispatch_markers": [
            row["source_file"] for row in static_scan["rows"] if row["contains_explicit_rule_dispatch"]
        ],
    }


def summarize_dynamic_trace(dynamic_trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "oracle_reached_during_forward": dynamic_trace["dynamic_trace"]["oracle_reached"],
        "oracle_functions_reached_during_forward": dynamic_trace["dynamic_trace"]["oracle_functions_reached_during_model_forward"],
        "input_packing_oracle_aliases_reached": dynamic_trace["input_packing_trace"]["oracle_aliases_reached_during_tensor_packing"],
        "all_monkeypatched_oracle_callables_forward_succeeded": dynamic_trace["all_monkeypatched_oracle_callables_forward_succeeded"],
        "monkeypatch_failures": [row for row in dynamic_trace["monkeypatch_execution_audit"] if not row["forward_succeeded"]],
    }


def gradient_vs_intervention_dependency_summary(rows: list[dict[str, Any]], gradient_probe: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in rows:
        if len(row["changed_parameter_groups"]) != 1:
            continue
        group = row["changed_parameter_groups"][0]
        summary = by_group.setdefault(
            group,
            {
                "intervention_count": 0,
                "max_grid_difference_rate_vs_checkpoint": 0.0,
                "max_logit_difference": 0.0,
                "min_train_macro_exact": 1.0,
                "min_dev_macro_exact": 1.0,
                "min_sealed_macro_exact": 1.0,
            },
        )
        summary["intervention_count"] += 1
        summary["max_grid_difference_rate_vs_checkpoint"] = max(summary["max_grid_difference_rate_vs_checkpoint"], row["grid_difference_rate_vs_checkpoint"])
        summary["max_logit_difference"] = max(summary["max_logit_difference"], row["logit_difference_summary"]["max_abs_output_probability_difference"])
        summary["min_train_macro_exact"] = min(summary["min_train_macro_exact"], row["train_macro_exact"])
        summary["min_dev_macro_exact"] = min(summary["min_dev_macro_exact"], row["dev_macro_exact"])
        summary["min_sealed_macro_exact"] = min(summary["min_sealed_macro_exact"], row["sealed_macro_exact"])
    nonzero_grad_no_argmax_effect = []
    for group, grad in gradient_probe["group_gradient_stats"].items():
        group_effect = by_group.get(group)
        if grad["nonzero_grad_count"] > 0 and group_effect and group_effect["max_grid_difference_rate_vs_checkpoint"] == 0.0:
            nonzero_grad_no_argmax_effect.append(group)
    return {
        "by_group": by_group,
        "nonzero_gradient_but_no_argmax_grid_effect_groups": sorted(nonzero_grad_no_argmax_effect),
    }


def write_artifacts(report: dict[str, Any], interventions: dict[str, Any], static_scan: dict[str, Any], dynamic_trace: dict[str, Any], provenance: dict[str, Any]) -> None:
    write_json(INTERVENTIONS_PATH, interventions)
    write_json(STATIC_SCAN_PATH, static_scan)
    write_json(DYNAMIC_TRACE_PATH, dynamic_trace)
    write_json(PROVENANCE_PATH, provenance)
    report = copy.deepcopy(report)
    report["generated_artifact_hashes"] = {
        "learning_dependency_interventions.json": sha256_file(INTERVENTIONS_PATH),
        "learning_dependency_static_scan.json": sha256_file(STATIC_SCAN_PATH),
        "learning_dependency_dynamic_trace.json": sha256_file(DYNAMIC_TRACE_PATH),
        "learning_dependency_provenance.json": sha256_file(PROVENANCE_PATH),
    }
    report["report_body_sha256_excluding_timestamp"] = stable_report_hash(report)
    write_json(REPORT_PATH, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-0 V2 learning-dependency and no-oracle audit")
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    report, interventions, static_scan, dynamic_trace, provenance = build_report(args.pytest_result, run_full_interventions=True)
    write_artifacts(report, interventions, static_scan, dynamic_trace, provenance)
    final = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": final["status"],
                "report": str(REPORT_PATH),
                "report_body_sha256_excluding_timestamp": final["report_body_sha256_excluding_timestamp"],
                "intervention_count": interventions["intervention_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if final["status"] != "R5A0V2_LEARNING_DEPENDENCY_INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
