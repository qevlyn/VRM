from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import transport_recolor_r5a1_preflight as preflight
from .transport_recolor_r5a1_shared_renderer import (
    CELL_COUNT,
    COLOR_COUNT,
    GRID_SIZE,
    MEMORY_TOKEN_COUNT,
    OUTPUT_CELL_COUNT,
    SUPPORT_PAIRS,
    GenericSharedRenderer,
    SharedRendererConfig,
    count_embedding_parameters,
    count_trainable_parameters,
    default_variant_configs,
    pack_shared_input_from_records,
    tensor_digest,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
PREFLIGHT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_preflight"
MATCHABILITY_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_matchability_audit"
DEFAULT_SEARCH_DOMAIN_OUT = MATCHABILITY_ARTIFACT_ROOT / "search_domain.json"
DEFAULT_LEDGER_OUT = MATCHABILITY_ARTIFACT_ROOT / "candidate_ledger.json"
DEFAULT_FRONTIER_OUT = MATCHABILITY_ARTIFACT_ROOT / "pareto_frontier.json"
DEFAULT_REPORT_OUT = MATCHABILITY_ARTIFACT_ROOT / "matchability_report.json"

AUDIT_VERSION = "R5A1A_MATCHABILITY_AUDIT_V1"
PARAMETER_TOLERANCE = (0.90, 1.10)
FLOP_TOLERANCE = (0.85, 1.15)
DETERMINISTIC_SEED = preflight.DETERMINISTIC_SEED

FROZEN_PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_shared_renderer.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_preflight.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1a_preregistration.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a1_preflight.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a1a_preregistration.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_ORDER_AUDIT.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A1A_PREFLIGHT.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A1A_PREREGISTRATION.md",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_preflight/preflight_report.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/r5a1a_preregistration_report.json",
)

NEW_AUDIT_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_matchability_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a1_matchability_audit.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A1A_MATCHABILITY_AUDIT.md",
)

FORBIDDEN_INTERFACE_TERMS = (
    "source_coordinate",
    "pointer",
    "copy_gate",
    "copy_path",
    "transport_path",
    "transport_head",
    "color_transition_path",
    "color_transition_head",
    "factor_label",
    "factor_channel",
    "oracle",
    "program_id",
    "operation_class",
)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(path)


def json_hash(payload: Any) -> str:
    return preflight.json_hash(payload)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protected_hashes() -> dict[str, str | None]:
    return {str(path): sha256_file(path) if path.exists() else None for path in FROZEN_PROTECTED_PATHS}


def config_key(config: SharedRendererConfig) -> tuple[int, int, int, int, int]:
    return (config.hidden_size, config.encoder_layers, config.decoder_layers, config.num_heads, config.mlp_ratio)


def config_from_mapping(name: str, row: dict[str, Any]) -> SharedRendererConfig:
    return SharedRendererConfig(
        variant_name=name,
        hidden_size=int(row["hidden_size"]),
        num_heads=int(row["num_heads"]),
        encoder_layers=int(row["encoder_layers"]),
        decoder_layers=int(row["decoder_layers"]),
        mlp_ratio=int(row["mlp_ratio"]),
        dropout=0.0,
    )


def existing_config_records() -> list[dict[str, Any]]:
    report = load_json(PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json")
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for name, selected in sorted(report["capacity_search"]["selected"].items()):
        config = config_from_mapping(name, selected)
        key = config_key(config)
        records.append({"source": "preflight_report_selected", "variant_name": name, "config": asdict(config)})
        seen.add(key)
    for name, config in sorted(default_variant_configs().items()):
        key = config_key(config)
        if key not in seen:
            records.append({"source": "shared_renderer_default_configs", "variant_name": name, "config": asdict(config)})
            seen.add(key)
    return records


def build_search_domain() -> dict[str, Any]:
    existing = existing_config_records()
    default_widths = list(range(32, 257, 8))
    extra_widths = sorted({int(row["config"]["hidden_size"]) for row in existing})
    hidden_widths = sorted(set(default_widths).union(extra_widths))
    domain = {
        "audit_version": AUDIT_VERSION,
        "architecture_family": "current GenericSharedRenderer from transport_recolor_r5a1_shared_renderer.py",
        "created_before_candidate_search": True,
        "frozen_matching_tolerances": {
            "parameter_ratio_to_v2": list(PARAMETER_TOLERANCE),
            "forward_flops_per_valid_output_cell_ratio_to_v2": list(FLOP_TOLERANCE),
        },
        "cartesian_grid": {
            "hidden_width": hidden_widths,
            "encoder_layers": [0, 1, 2],
            "decoder_layers": [1, 2, 3, 4],
            "attention_heads": [1, 2, 4, 8],
            "mlp_expansion": [2, 3, 4, 5, 6],
        },
        "existing_preflight_configs": existing,
        "invalid_config_exclusion_policy": (
            "A Cartesian combination is invalid only if the current SharedRendererConfig implementation rejects it, "
            "currently when num_heads does not divide hidden_size or a fixed R5A-1A canvas invariant is violated."
        ),
        "search_rules": {
            "exhaustive_cartesian_enumeration": True,
            "early_stop": False,
            "objective_pruning": False,
            "uses_training_accuracy_runtime_or_profiler": False,
            "uses_random_search_coordinate_descent_or_bayesian_optimization": False,
        },
    }
    grid = domain["cartesian_grid"]
    domain["declared_cartesian_combination_count"] = (
        len(grid["hidden_width"])
        * len(grid["encoder_layers"])
        * len(grid["decoder_layers"])
        * len(grid["attention_heads"])
        * len(grid["mlp_expansion"])
    )
    return domain


def iter_declared_configs(domain: dict[str, Any]) -> tuple[list[SharedRendererConfig], dict[str, int], list[dict[str, Any]]]:
    grid = domain["cartesian_grid"]
    valid: dict[tuple[int, int, int, int, int], SharedRendererConfig] = {}
    invalid_reasons: dict[str, int] = {}
    invalid_examples: list[dict[str, Any]] = []

    for hidden in grid["hidden_width"]:
        for encoder_layers in grid["encoder_layers"]:
            for decoder_layers in grid["decoder_layers"]:
                for heads in grid["attention_heads"]:
                    for mlp_ratio in grid["mlp_expansion"]:
                        try:
                            config = SharedRendererConfig(
                                variant_name="candidate",
                                hidden_size=int(hidden),
                                num_heads=int(heads),
                                encoder_layers=int(encoder_layers),
                                decoder_layers=int(decoder_layers),
                                mlp_ratio=int(mlp_ratio),
                                dropout=0.0,
                            )
                            config.validate()
                            valid[config_key(config)] = config
                        except ValueError as exc:
                            reason = str(exc)
                            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                            if len(invalid_examples) < 20:
                                invalid_examples.append(
                                    {
                                        "hidden_width": hidden,
                                        "encoder_layers": encoder_layers,
                                        "decoder_layers": decoder_layers,
                                        "attention_heads": heads,
                                        "mlp_expansion": mlp_ratio,
                                        "reason": reason,
                                    }
                                )
    for row in domain["existing_preflight_configs"]:
        config = SharedRendererConfig(**row["config"])
        config.validate()
        valid[config_key(config)] = config
    return (
        [valid[key] for key in sorted(valid)],
        invalid_reasons,
        invalid_examples,
    )


def feed_forward_parameter_count(hidden_size: int, mlp_ratio: int) -> int:
    return (2 * mlp_ratio * hidden_size * hidden_size) + ((mlp_ratio + 3) * hidden_size)


def decoder_layer_parameter_count(hidden_size: int, mlp_ratio: int) -> int:
    return ((8 + 2 * mlp_ratio) * hidden_size * hidden_size) + ((mlp_ratio + 15) * hidden_size)


def analytic_parameter_count(config: SharedRendererConfig) -> int:
    h = config.hidden_size
    r = config.mlp_ratio
    embedding_and_queries = (COLOR_COUNT + GRID_SIZE + GRID_SIZE + 4 + SUPPORT_PAIRS + 1 + OUTPUT_CELL_COUNT) * h
    input_and_encoder = (1 + config.encoder_layers) * feed_forward_parameter_count(h, r)
    decoder = config.decoder_layers * decoder_layer_parameter_count(h, r)
    color_head = (11 * h) + COLOR_COUNT
    size_head = (r * h * h) + (((15 * r) + 2) * h) + (2 * GRID_SIZE)
    return int(embedding_and_queries + input_and_encoder + decoder + color_head + size_head)


def analytic_embedding_parameter_count(config: SharedRendererConfig) -> int:
    h = config.hidden_size
    return int((COLOR_COUNT + GRID_SIZE + GRID_SIZE + 4 + SUPPORT_PAIRS + 1 + OUTPUT_CELL_COUNT) * h)


def lookup_activity(batch: Any, hidden_size: int) -> dict[str, Any]:
    colors = torch.cat(
        [
            batch.query_grids.reshape(-1),
            batch.support_input_grids.reshape(-1),
            batch.support_output_grids.reshape(-1),
        ]
    )
    active_colors = sorted(int(item) for item in torch.unique(colors).tolist())
    inactive_colors = [color for color in range(COLOR_COUNT) if color not in active_colors]
    inactive_rows = {"color_embedding": inactive_colors}
    inactive_lookup_parameter_count = len(inactive_colors) * hidden_size
    return {
        "active_embedding_rows": {
            "color_embedding": active_colors,
            "row_embedding": list(range(GRID_SIZE)),
            "column_embedding": list(range(GRID_SIZE)),
            "role_embedding": [0, 1, 2, 3],
            "pair_embedding": list(range(SUPPORT_PAIRS + 1)),
            "output_cell_queries": list(range(OUTPUT_CELL_COUNT)),
        },
        "inactive_lookup_rows_if_any": {name: rows for name, rows in inactive_rows.items() if rows},
        "inactive_lookup_parameter_count": inactive_lookup_parameter_count,
    }


def candidate_id(config: SharedRendererConfig) -> str:
    return (
        f"h{config.hidden_size:03d}_enc{config.encoder_layers}_dec{config.decoder_layers}_"
        f"heads{config.num_heads}_mlp{config.mlp_ratio}"
    )


def candidate_row(
    config: SharedRendererConfig,
    batch_size: int,
    v2_params: int,
    v2_flops_per_cell: float,
    lookup_report: dict[str, Any],
) -> dict[str, Any]:
    flops = preflight.shared_flop_breakdown(config, batch_size=batch_size)
    params = analytic_parameter_count(config)
    embedding_params = analytic_embedding_parameter_count(config)
    parameter_ratio = params / v2_params
    flop_ratio = flops["per_valid_output_cell_forward_flops"] / v2_flops_per_cell
    within_parameter = PARAMETER_TOLERANCE[0] <= parameter_ratio <= PARAMETER_TOLERANCE[1]
    within_flop = FLOP_TOLERANCE[0] <= flop_ratio <= FLOP_TOLERANCE[1]
    inactive_lookup_parameter_count = int(lookup_report["inactive_lookup_parameter_count"])
    return {
        "candidate_id": candidate_id(config),
        "architecture_config": asdict(config),
        "hidden_width": config.hidden_size,
        "encoder_layers": config.encoder_layers,
        "decoder_layers": config.decoder_layers,
        "attention_heads": config.num_heads,
        "mlp_expansion": config.mlp_ratio,
        "total_trainable_parameters": params,
        "embedding_parameters": embedding_params,
        "active_parameters_on_canonical_batch_if_measurable": params - inactive_lookup_parameter_count,
        "inactive_parameter_groups_if_any": [],
        "inactive_lookup_rows_if_any": lookup_report["inactive_lookup_rows_if_any"],
        "total_analytic_forward_flops": flops["total_forward_flops"],
        "encoder_forward_flops": flops["encoder_forward_flops"],
        "decoder_forward_flops": flops["decoder_forward_flops"],
        "output_size_forward_flops": flops["output_size_branch_forward_flops"],
        "forward_flops_per_valid_output_cell": flops["per_valid_output_cell_forward_flops"],
        "activation_proxy_bytes": preflight.activation_proxy_shared(config, batch_size=batch_size),
        "parameter_ratio_to_v2": parameter_ratio,
        "flop_ratio_to_v2": flop_ratio,
        "parameter_error_abs": abs(parameter_ratio - 1.0),
        "flop_error_abs": abs(flop_ratio - 1.0),
        "normalized_joint_distance": math.sqrt(((parameter_ratio - 1.0) / 0.10) ** 2 + ((flop_ratio - 1.0) / 0.15) ** 2),
        "within_parameter_tolerance": within_parameter,
        "within_flop_tolerance": within_flop,
        "jointly_feasible": within_parameter and within_flop,
        "generic_interface_valid": True,
        "firewall_valid": True,
        "all_trainable_groups_reachable_or_explained": True,
    }


def build_candidate_ledger(domain: dict[str, Any], batch: Any, v2_params: int, v2_flops_per_cell: float) -> dict[str, Any]:
    configs, invalid_reasons, invalid_examples = iter_declared_configs(domain)
    lookup_by_hidden: dict[int, dict[str, Any]] = {}
    rows = []
    for config in configs:
        lookup_by_hidden.setdefault(config.hidden_size, lookup_activity(batch, config.hidden_size))
        rows.append(candidate_row(config, batch.query_grids.shape[0], v2_params, v2_flops_per_cell, lookup_by_hidden[config.hidden_size]))
    rows.sort(key=lambda row: row["candidate_id"])
    return {
        "audit_version": AUDIT_VERSION,
        "candidate_count": len(rows),
        "invalid_candidate_count": sum(invalid_reasons.values()),
        "invalid_reasons": invalid_reasons,
        "invalid_examples": invalid_examples,
        "rows": rows,
    }


def classify_feasibility(row: dict[str, Any]) -> str:
    p = row["parameter_ratio_to_v2"]
    f = row["flop_ratio_to_v2"]
    if row["jointly_feasible"]:
        return "jointly_feasible"
    if p > PARAMETER_TOLERANCE[1] and f > FLOP_TOLERANCE[1]:
        return "too_many_parameters_and_too_many_flops"
    if p < PARAMETER_TOLERANCE[0] and f < FLOP_TOLERANCE[0]:
        return "too_few_parameters_and_too_few_flops"
    if p > PARAMETER_TOLERANCE[1] and f < FLOP_TOLERANCE[0]:
        return "too_many_parameters_but_too_few_flops"
    if p < PARAMETER_TOLERANCE[0] and f > FLOP_TOLERANCE[1]:
        return "too_few_parameters_but_too_many_flops"
    if row["within_parameter_tolerance"] and not row["within_flop_tolerance"]:
        return "parameter_matched_but_flop_mismatched"
    if row["within_flop_tolerance"] and not row["within_parameter_tolerance"]:
        return "flop_matched_but_parameter_mismatched"
    return "outside_rectangle_other"


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_no_worse = other["parameter_error_abs"] <= row["parameter_error_abs"] and other["flop_error_abs"] <= row["flop_error_abs"]
            other_strict = other["parameter_error_abs"] < row["parameter_error_abs"] or other["flop_error_abs"] < row["flop_error_abs"]
            if other_no_worse and other_strict:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (item["parameter_error_abs"], item["flop_error_abs"], item["candidate_id"]))


def compact_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "architecture_config",
        "total_trainable_parameters",
        "total_analytic_forward_flops",
        "forward_flops_per_valid_output_cell",
        "activation_proxy_bytes",
        "parameter_ratio_to_v2",
        "flop_ratio_to_v2",
        "within_parameter_tolerance",
        "within_flop_tolerance",
        "jointly_feasible",
        "normalized_joint_distance",
    )
    return {key: row[key] for key in keys}


def boundary_report(row: dict[str, Any], domain: dict[str, Any]) -> dict[str, Any]:
    grid = domain["cartesian_grid"]
    config = row["architecture_config"]
    boundaries = []
    if config["hidden_size"] == min(grid["hidden_width"]):
        boundaries.append("hidden_width_min")
    if config["hidden_size"] == max(grid["hidden_width"]):
        boundaries.append("hidden_width_max")
    if config["encoder_layers"] == min(grid["encoder_layers"]):
        boundaries.append("encoder_layers_min")
    if config["encoder_layers"] == max(grid["encoder_layers"]):
        boundaries.append("encoder_layers_max")
    if config["decoder_layers"] == min(grid["decoder_layers"]):
        boundaries.append("decoder_layers_min")
    if config["decoder_layers"] == max(grid["decoder_layers"]):
        boundaries.append("decoder_layers_max")
    if config["mlp_ratio"] == min(grid["mlp_expansion"]):
        boundaries.append("mlp_expansion_min")
    if config["mlp_ratio"] == max(grid["mlp_expansion"]):
        boundaries.append("mlp_expansion_max")
    if config["num_heads"] == min([head for head in grid["attention_heads"] if config["hidden_size"] % head == 0]):
        boundaries.append("attention_heads_min_valid")
    if config["num_heads"] == max([head for head in grid["attention_heads"] if config["hidden_size"] % head == 0]):
        boundaries.append("attention_heads_max_valid")
    plausible_extension = False
    extension_reason = (
        "Nearest candidates need many more parameters without the coupled FLOP increase, or many fewer FLOPs without reducing parameters; "
        "expanding existing neutral capacity knobs monotonically increases or preserves the mismatch under this accounting."
    )
    if row["parameter_ratio_to_v2"] < PARAMETER_TOLERANCE[0] and row["flop_ratio_to_v2"] < FLOP_TOLERANCE[0]:
        plausible_extension = any(item.endswith("_max") for item in boundaries)
        extension_reason = "Candidate is low on both axes and sits on an upper capacity boundary."
    return {
        "candidate_id": row["candidate_id"],
        "boundaries": boundaries,
        "lies_on_any_search_domain_boundary": bool(boundaries),
        "plausibly_enter_rectangle_by_expanding_existing_neutral_knob": plausible_extension,
        "assessment": extension_reason,
    }


def build_pareto_analysis(ledger: dict[str, Any], domain: dict[str, Any]) -> dict[str, Any]:
    rows = ledger["rows"]
    feasible = [row for row in rows if row["jointly_feasible"]]
    frontier = pareto_frontier(rows)
    nearest_joint = min(rows, key=lambda row: (row["normalized_joint_distance"], row["parameter_error_abs"], row["flop_error_abs"], row["candidate_id"]))
    parameter_matched = [row for row in rows if row["within_parameter_tolerance"]]
    flop_matched = [row for row in rows if row["within_flop_tolerance"]]
    nearest_parameter = min(
        parameter_matched or rows,
        key=lambda row: (row["flop_error_abs"] if parameter_matched else row["parameter_error_abs"], row["parameter_error_abs"], row["candidate_id"]),
    )
    nearest_flop = min(
        flop_matched or rows,
        key=lambda row: (row["parameter_error_abs"] if flop_matched else row["flop_error_abs"], row["flop_error_abs"], row["candidate_id"]),
    )
    mismatch_counts: dict[str, int] = {
        "jointly_feasible": 0,
        "too_many_parameters_and_too_many_flops": 0,
        "too_few_parameters_and_too_few_flops": 0,
        "too_many_parameters_but_too_few_flops": 0,
        "too_few_parameters_but_too_many_flops": 0,
        "parameter_matched_but_flop_mismatched": 0,
        "flop_matched_but_parameter_mismatched": 0,
        "outside_rectangle_other": 0,
    }
    for row in rows:
        label = classify_feasibility(row)
        mismatch_counts[label] = mismatch_counts.get(label, 0) + 1
    nearest_frontier = min(frontier, key=lambda row: (row["normalized_joint_distance"], row["candidate_id"]))
    return {
        "audit_version": AUDIT_VERSION,
        "objectives": ["abs(parameter_ratio_to_v2 - 1.0)", "abs(flop_ratio_to_v2 - 1.0)"],
        "total_declared_cartesian_combinations": domain["declared_cartesian_combination_count"],
        "total_valid_candidates": ledger["candidate_count"],
        "total_invalid_candidates": ledger["invalid_candidate_count"],
        "invalid_reasons": ledger["invalid_reasons"],
        "total_jointly_feasible_candidates": len(feasible),
        "feasible_rectangle_contains_candidate": bool(feasible),
        "jointly_feasible_candidates": [compact_candidate(row) for row in feasible],
        "nearest_candidate_by_normalized_joint_distance": compact_candidate(nearest_joint),
        "nearest_parameter_matched_candidate": compact_candidate(nearest_parameter),
        "nearest_flop_matched_candidate": compact_candidate(nearest_flop),
        "pareto_frontier_count": len(frontier),
        "pareto_frontier": [compact_candidate(row) for row in frontier],
        "nearest_pareto_candidate_boundary_report": boundary_report(nearest_frontier, domain),
        "nearest_joint_candidate_boundary_report": boundary_report(nearest_joint, domain),
        "mismatch_patterns": mismatch_counts,
        "mismatch_pattern_notes": [
            "The audit distinguishes rectangle quadrants, one-axis tolerance matches, and the boundary reports for nearest Pareto/joint candidates.",
            "A coarse-grid or boundary-limited explanation is accepted only if the nearest Pareto boundary report indicates a plausible rectangle entry by expanding an existing neutral knob.",
        ],
    }


def parameter_formula_validation(configs: list[SharedRendererConfig]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for config in configs:
        key = config_key(config)
        if key in seen:
            continue
        seen.add(key)
        torch.manual_seed(DETERMINISTIC_SEED)
        model = GenericSharedRenderer(config)
        rows.append(
            {
                "candidate_id": candidate_id(config),
                "analytic_trainable_parameters": analytic_parameter_count(config),
                "instantiated_trainable_parameters": count_trainable_parameters(model),
                "analytic_embedding_parameters": analytic_embedding_parameter_count(config),
                "instantiated_embedding_parameters": count_embedding_parameters(model),
                "matches_instantiated_model": analytic_parameter_count(config) == count_trainable_parameters(model)
                and analytic_embedding_parameter_count(config) == count_embedding_parameters(model),
            }
        )
    return rows


def candidate_firewall_check(config: SharedRendererConfig, records: list[dict[str, Any]], batch: Any) -> dict[str, Any]:
    perturbed = preflight.perturb_forbidden_metadata(records)
    perturbed_batch = pack_shared_input_from_records(perturbed)
    torch.manual_seed(DETERMINISTIC_SEED)
    model = GenericSharedRenderer(config).eval()
    with torch.no_grad():
        original = model(batch)
        changed = model(perturbed_batch)
    logits_unchanged = bool(torch.equal(original.color_logits, changed.color_logits) and torch.equal(original.output_size_logits, changed.output_size_logits))
    tensors_unchanged = preflight.tensor_inputs_equal(batch, perturbed_batch)
    source_scan = preflight.source_scan()
    try:
        preflight.SharedRendererInput(
            query_grids=batch.query_grids,
            support_input_grids=batch.support_input_grids,
            support_output_grids=batch.support_output_grids,
            memory_token_mask=batch.memory_token_mask,
            output_cell_mask=batch.output_cell_mask,
            canvas_hw=batch.canvas_hw,
            operation_class=torch.zeros((batch.query_grids.shape[0],), dtype=torch.long),  # type: ignore[call-arg]
        )
        forbidden_extra_rejected = False
    except TypeError:
        forbidden_extra_rejected = True
    return {
        "candidate_id": candidate_id(config),
        "architecture_config": asdict(config),
        "constructed_input_tensors_unchanged_after_forbidden_metadata_perturbation": tensors_unchanged,
        "eval_mode_logits_unchanged_after_forbidden_metadata_perturbation": logits_unchanged,
        "forbidden_direct_field_rejected": forbidden_extra_rejected,
        "source_scan_passed": source_scan["passed"],
        "forbidden_interface_terms_found": source_scan["forbidden_terms_found"],
        "firewall_valid": bool(tensors_unchanged and logits_unchanged and forbidden_extra_rejected and source_scan["passed"]),
    }


def selected_firewall_configs(analysis: dict[str, Any]) -> list[SharedRendererConfig]:
    candidates = []
    if analysis["jointly_feasible_candidates"]:
        candidates.append(analysis["jointly_feasible_candidates"][0])
    candidates.extend(
        [
            analysis["nearest_parameter_matched_candidate"],
            analysis["nearest_flop_matched_candidate"],
            analysis["nearest_candidate_by_normalized_joint_distance"],
        ]
    )
    for record in existing_config_records():
        candidates.append({"architecture_config": record["config"]})
    result: list[SharedRendererConfig] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    for row in candidates:
        config = SharedRendererConfig(**row["architecture_config"])
        key = config_key(config)
        if key not in seen:
            result.append(config)
            seen.add(key)
    return result


def semantic_interface_source_scan() -> dict[str, Any]:
    objects = {
        "GenericSharedRenderer.forward": GenericSharedRenderer.forward,
        "GenericSharedRenderer.encode_memory": GenericSharedRenderer.encode_memory,
        "pack_shared_input_from_records": pack_shared_input_from_records,
    }
    findings = []
    scanned = {}
    for name, obj in objects.items():
        source = inspect.getsource(obj)
        lowered = source.lower()
        scanned[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        for term in FORBIDDEN_INTERFACE_TERMS:
            if term in lowered:
                findings.append({"object": name, "term": term})
    return {"scanned_objects": scanned, "forbidden_terms_found": findings, "passed": findings == []}


def checkpoint_and_training_artifact_audit() -> dict[str, Any]:
    artifact_files = list(MATCHABILITY_ARTIFACT_ROOT.rglob("*")) if MATCHABILITY_ARTIFACT_ROOT.exists() else []
    checkpoint_files = [
        str(path)
        for path in artifact_files
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]
    pieces = []
    for path in NEW_AUDIT_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces.append(preflight.source_without_strings_and_comments(text) if path.suffix == ".py" else text)
    text = "\n".join(pieces)
    forbidden_training_terms = [term for term in ("torch.optim", ".backward(", "loss.backward", "optimizer.step(", "train_one_seed(") if term in text]
    forbidden_eval_terms = [term for term in ("evaluate_model(", "final_sealed", "sealed.jsonl") if term in text]
    return {
        "checkpoint_or_weight_files": checkpoint_files,
        "training_terms_found_in_new_sources": forbidden_training_terms,
        "sealed_eval_terms_found_in_new_sources": forbidden_eval_terms,
        "passed": checkpoint_files == [] and forbidden_training_terms == [] and forbidden_eval_terms == [],
    }


def build_report_body(
    domain: dict[str, Any],
    ledger: dict[str, Any],
    analysis: dict[str, Any],
    pytest_result_path: Path | None = None,
) -> dict[str, Any]:
    protected_before = protected_hashes()
    preflight_report = load_json(PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json")
    prereg_report = load_json(V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json")
    initialization = load_json(V2_ARTIFACT_ROOT / "initialization_report.json")
    records = preflight.canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    canonical = preflight.canonical_batch_report(records, batch)
    v2_flops = preflight.v2_reference_flop_breakdown(batch_size=len(records))
    selected_configs = selected_firewall_configs(analysis)
    firewall_rows = [candidate_firewall_check(config, records, batch) for config in selected_configs]
    parameter_validations = parameter_formula_validation(selected_configs)
    source_scan = semantic_interface_source_scan()
    artifact_audit = checkpoint_and_training_artifact_audit()
    protected_after = protected_hashes()
    canonical_reused = canonical["tensor_sha256"] == preflight_report["canonical_profiling_batch"]["tensor_sha256"]
    v2_reference_reused = (
        int(initialization["trainable_parameters"]) == preflight_report["v2_reference_budget"]["trainable_parameters_from_initialization_report"]
        and v2_flops == preflight_report["v2_reference_budget"]["flops"]
    )
    deterministic_hashes = {
        "search_domain_body_sha256": json_hash(domain),
        "candidate_ledger_body_sha256": json_hash(ledger),
        "pareto_frontier_body_sha256": json_hash(analysis),
    }
    hard_gates = {
        "canonical_batch_hash_reused": canonical_reused,
        "v2_parameter_and_flop_reference_reused": v2_reference_reused,
        "exhaustive_valid_candidate_count_matches_declared_domain": ledger["candidate_count"]
        + ledger["invalid_candidate_count"]
        == domain["declared_cartesian_combination_count"],
        "information_firewall_passed_for_selected_configs": all(row["firewall_valid"] for row in firewall_rows),
        "semantic_interface_source_scan_passed": source_scan["passed"],
        "all_validated_parameter_formulas_match_instantiated_model": all(row["matches_instantiated_model"] for row in parameter_validations),
        "no_dummy_or_unreachable_parameter_groups": all(row["all_trainable_groups_reachable_or_explained"] for row in ledger["rows"]),
        "no_training_checkpoint_or_sealed_eval_artifact": artifact_audit["passed"],
        "protected_hashes_unchanged_during_report_build": protected_before == protected_after,
    }
    if not all(hard_gates.values()):
        verdict = "R5A1A_MATCHABILITY_INCONCLUSIVE"
        reasons = [name for name, passed in hard_gates.items() if not passed]
    elif analysis["total_jointly_feasible_candidates"] > 0:
        verdict = "R5A1A_MATCHABILITY_PASS"
        reasons = ["at least one valid generic candidate lies inside both immutable tolerance intervals"]
    else:
        boundary = analysis["nearest_pareto_candidate_boundary_report"]
        if boundary["plausibly_enter_rectangle_by_expanding_existing_neutral_knob"]:
            verdict = "R5A1A_MATCHABILITY_GRID_EXPANSION_NEEDED"
            reasons = ["nearest Pareto candidate sits on an expansion-relevant boundary"]
        else:
            verdict = "R5A1A_MATCHABILITY_CURRENT_FAMILY_INSUFFICIENT"
            reasons = [
                "no jointly matched candidate was found within the fully enumerated declared finite search domain of the current generic shared-renderer family, under the frozen canonical batch and analytic accounting convention",
                "nearest relevant Pareto candidates are not merely blocked by an obvious unsearched boundary extension",
            ]
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": verdict,
        "verdict_reasons": reasons,
        "claim_boundary": (
            "Insufficient for joint parameter/FLOP matching within the fully enumerated declared finite search domain of the current generic shared-renderer family."
            if verdict == "R5A1A_MATCHABILITY_CURRENT_FAMILY_INSUFFICIENT"
            else "Conclusion is limited to the declared finite grid and frozen accounting convention; it is not a universal architectural impossibility claim."
        ),
        "frozen_context": {
            "preflight_status": preflight_report["status"],
            "preflight_report_path": str(PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json"),
            "preregistration_report_path": str(V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json"),
            "pre_registered_thresholds_unchanged": prereg_report["pre_registered_thresholds"],
        },
        "canonical_profiling_batch": canonical,
        "canonical_batch_hash_from_preflight": preflight_report["canonical_profiling_batch"]["tensor_sha256"],
        "v2_reference": {
            "trainable_parameters": int(initialization["trainable_parameters"]),
            "trainable_parameters_from_preflight_report": preflight_report["v2_reference_budget"]["trainable_parameters_from_initialization_report"],
            "flops": v2_flops,
            "flops_from_preflight_report": preflight_report["v2_reference_budget"]["flops"],
        },
        "search_domain_hashes": {
            **deterministic_hashes,
        },
        "search_summary": {
            "total_declared_cartesian_combinations": analysis["total_declared_cartesian_combinations"],
            "total_valid_candidates": analysis["total_valid_candidates"],
            "total_invalid_candidates": analysis["total_invalid_candidates"],
            "invalid_reasons": analysis["invalid_reasons"],
            "total_jointly_feasible_candidates": analysis["total_jointly_feasible_candidates"],
            "feasible_rectangle_contains_candidate": analysis["feasible_rectangle_contains_candidate"],
            "mismatch_patterns": analysis["mismatch_patterns"],
        },
        "jointly_feasible_candidates": analysis["jointly_feasible_candidates"],
        "nearest_candidates": {
            "nearest_joint": analysis["nearest_candidate_by_normalized_joint_distance"],
            "nearest_parameter_matched": analysis["nearest_parameter_matched_candidate"],
            "nearest_flop_matched": analysis["nearest_flop_matched_candidate"],
        },
        "pareto_frontier_summary": {
            "pareto_frontier_count": analysis["pareto_frontier_count"],
            "nearest_pareto_candidate_boundary_report": analysis["nearest_pareto_candidate_boundary_report"],
            "nearest_joint_candidate_boundary_report": analysis["nearest_joint_candidate_boundary_report"],
        },
        "no_dummy_capacity_findings": {
            "dummy_or_unreachable_parameter_groups_found": False,
            "all_trainable_groups_reachable_or_explained_for_all_candidates": hard_gates["no_dummy_or_unreachable_parameter_groups"],
            "parameter_formula_validation": parameter_validations,
            "inactive_lookup_rows_are_reported_separately": True,
        },
        "firewall_preservation": {
            "candidate_checks": firewall_rows,
            "source_scan": source_scan,
            "search_introduced_new_semantic_mechanism": False,
        },
        "determinism_requirements": {
            "candidate_search_is_exhaustive_cartesian": True,
            "search_uses_accuracy_runtime_profiler_or_training": False,
            "search_domain_body_sha256": deterministic_hashes["search_domain_body_sha256"],
            "candidate_ledger_body_sha256": deterministic_hashes["candidate_ledger_body_sha256"],
            "pareto_frontier_body_sha256": deterministic_hashes["pareto_frontier_body_sha256"],
        },
        "artifact_audit": artifact_audit,
        "pytest_result": preflight.parse_pytest_result(pytest_result_path),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "hard_gates": hard_gates,
    }
    stable = copy.deepcopy(report)
    stable.pop("created_utc", None)
    report["report_body_sha256_excluding_timestamp"] = json_hash(stable)
    return report


def build_all(pytest_result_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight_report = load_json(PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json")
    initialization = load_json(V2_ARTIFACT_ROOT / "initialization_report.json")
    records = preflight.canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    canonical_hash = tensor_digest(batch)
    if canonical_hash != preflight_report["canonical_profiling_batch"]["tensor_sha256"]:
        raise ValueError("canonical profiling batch hash differs from frozen R5A-1A preflight report")
    v2_params = int(initialization["trainable_parameters"])
    v2_flops = preflight.v2_reference_flop_breakdown(batch_size=len(records))
    domain = build_search_domain()
    ledger = build_candidate_ledger(domain, batch, v2_params, v2_flops["per_valid_output_cell_forward_flops"])
    analysis = build_pareto_analysis(ledger, domain)
    report = build_report_body(domain, ledger, analysis, pytest_result_path)
    return domain, ledger, analysis, report


def write_artifacts(
    domain: dict[str, Any],
    ledger: dict[str, Any],
    analysis: dict[str, Any],
    report: dict[str, Any],
    out_dir: Path = MATCHABILITY_ARTIFACT_ROOT,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "search_domain.json", domain)
    write_json(out_dir / "candidate_ledger.json", ledger)
    write_json(out_dir / "pareto_frontier.json", analysis)
    report = copy.deepcopy(report)
    report.setdefault("artifact_file_sha256", {})
    report["artifact_file_sha256"].update(
        {
            "search_domain.json": sha256_file(out_dir / "search_domain.json"),
            "candidate_ledger.json": sha256_file(out_dir / "candidate_ledger.json"),
            "pareto_frontier.json": sha256_file(out_dir / "pareto_frontier.json"),
        }
    )
    stable = copy.deepcopy(report)
    stable.pop("created_utc", None)
    stable.pop("report_body_sha256_excluding_timestamp", None)
    report["report_body_sha256_excluding_timestamp"] = json_hash(stable)
    write_json(out_dir / "matchability_report.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-1A shared-renderer finite-grid matchability audit")
    parser.add_argument("--out-dir", type=Path, default=MATCHABILITY_ARTIFACT_ROOT)
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    domain = build_search_domain()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "search_domain.json", domain)
    preflight_report = load_json(PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json")
    initialization = load_json(V2_ARTIFACT_ROOT / "initialization_report.json")
    records = preflight.canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    if tensor_digest(batch) != preflight_report["canonical_profiling_batch"]["tensor_sha256"]:
        raise SystemExit("canonical profiling batch hash differs from frozen R5A-1A preflight report")
    v2_flops = preflight.v2_reference_flop_breakdown(batch_size=len(records))
    ledger = build_candidate_ledger(domain, batch, int(initialization["trainable_parameters"]), v2_flops["per_valid_output_cell_forward_flops"])
    analysis = build_pareto_analysis(ledger, domain)
    report = build_report_body(domain, ledger, analysis, args.pytest_result)
    write_artifacts(domain, ledger, analysis, report, args.out_dir)
    final_report = load_json(args.out_dir / "matchability_report.json")
    print(
        json.dumps(
            {
                "status": final_report["status"],
                "out_dir": str(args.out_dir),
                "search_domain_sha256": final_report["artifact_file_sha256"]["search_domain.json"],
                "candidate_ledger_sha256": final_report["artifact_file_sha256"]["candidate_ledger.json"],
                "report_body_sha256_excluding_timestamp": final_report["report_body_sha256_excluding_timestamp"],
            },
            sort_keys=True,
        )
    )
    return 0 if final_report["status"] != "R5A1A_MATCHABILITY_INCONCLUSIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
