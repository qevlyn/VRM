from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import re
import sys
import textwrap
import tokenize
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .transport_recolor_r5a0v2_model import SupportConditionedTransportRecolorR5A0V2
from .transport_recolor_r5a1_shared_renderer import (
    ALLOWED_INPUT_FIELDS,
    CELL_COUNT,
    COLOR_COUNT,
    FORBIDDEN_FIELD_NAMES,
    GRID_SIZE,
    MEMORY_TOKEN_COUNT,
    OUTPUT_CELL_COUNT,
    QUERY_TOKEN_COUNT,
    SUPPORT_PAIRS,
    SUPPORT_ROLES,
    SUPPORT_TOKEN_COUNT,
    GenericSharedRenderer,
    SharedRendererConfig,
    SharedRendererInput,
    count_embedding_parameters,
    count_trainable_parameters,
    pack_shared_input_from_records,
    tensor_digest,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
PREFLIGHT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a1a_preflight"
DEFAULT_OUT = PREFLIGHT_ARTIFACT_ROOT / "preflight_report.json"
PREFLIGHT_VERSION = "R5A1A_SHARED_RENDERER_PREFLIGHT_V1"
DETERMINISTIC_SEED = 20260630

V2_PROTECTED_SOURCE_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_ORDER_AUDIT.md",
)

R5A1A_SOURCE_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_shared_renderer.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a1_preflight.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a1_preflight.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A1A_PREFLIGHT.md",
)

FORBIDDEN_METADATA_TERMS = (
    "operation_class",
    "sequence_class",
    "evaluation_only_operation_label",
    "split",
    "seed",
    "signature",
    "oracle",
    "target",
    "transport",
    "color_transition",
    "pointer",
    "copy",
    "factor",
    "program",
    "class_label",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_report_hash(report: dict[str, Any]) -> str:
    stable = copy.deepcopy(report)
    stable.pop("created_utc", None)
    stable.pop("report_sha256_excluding_timestamp", None)
    return json_hash(stable)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_records(limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with (V2_ARTIFACT_ROOT / "train.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("canonical R5A-1A preflight may read only V2 training-side rows")
            records.append(row)
            if limit is not None and len(records) >= limit:
                break
    return records


def canonical_records(batch_size: int = 4) -> list[dict[str, Any]]:
    records = load_train_records()
    selected: list[dict[str, Any]] = []
    seen_ops: set[str] = set()
    for row in sorted(records, key=lambda item: item["episode_id"]):
        op = str(row.get("operation_class"))
        if op not in seen_ops:
            selected.append(row)
            seen_ops.add(op)
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        selected = sorted(records, key=lambda item: item["episode_id"])[:batch_size]
    return selected


def tensor_inventory(batch: SharedRendererInput) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ALLOWED_INPUT_FIELDS:
        tensor = getattr(batch, name)
        entry: dict[str, Any] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        if tensor.dtype == torch.bool:
            entry["true_count"] = int(tensor.sum().item())
            entry["false_count"] = int((~tensor).sum().item())
        result[name] = entry
    return result


def canonical_batch_report(records: list[dict[str, Any]], batch: SharedRendererInput) -> dict[str, Any]:
    bsz = len(records)
    return {
        "selection_method": (
            "sorted V2 train.jsonl by episode_id, then selected the first distinct operation_class rows until batch_size=4; "
            "falls back to first four sorted train rows only if distinct operation coverage is unavailable"
        ),
        "training_side_only": True,
        "sealed_rows_read": False,
        "batch_size": bsz,
        "stable_episode_ids": [row["episode_id"] for row in records],
        "operation_classes_for_selection_audit_only": [row.get("operation_class") for row in records],
        "max_structural_workload": {
            "support_pair_count": SUPPORT_PAIRS,
            "support_role_count_per_pair": SUPPORT_ROLES,
            "support_token_count_per_example": SUPPORT_TOKEN_COUNT,
            "query_token_count_per_example": QUERY_TOKEN_COUNT,
            "memory_token_count_per_example": MEMORY_TOKEN_COUNT,
            "output_cell_query_count_per_example": OUTPUT_CELL_COUNT,
            "valid_output_cell_count_per_example": OUTPUT_CELL_COUNT,
            "canvas_hw": [GRID_SIZE, GRID_SIZE],
            "color_count": COLOR_COUNT,
        },
        "tensor_inventory": tensor_inventory(batch),
        "padding": {
            "memory_token_padding_false_count": int((~batch.memory_token_mask).sum().item()),
            "output_cell_padding_false_count": int((~batch.output_cell_mask).sum().item()),
            "all_examples_use_full_7x7_canvas": bool((batch.canvas_hw == GRID_SIZE).all().item()),
        },
        "tensor_sha256": tensor_digest(batch),
    }


def perturb_forbidden_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    perturbed = copy.deepcopy(records)
    for index, row in enumerate(perturbed):
        row["episode_id"] = f"perturbed_episode_{index}"
        row["split"] = "forbidden_split_value"
        row["operation_class"] = "FORBIDDEN_OPERATION"
        row["sequence_class"] = "FORBIDDEN_SEQUENCE"
        row["evaluation_only_operation_label"] = "FORBIDDEN_LABEL"
        row["query_type"] = {
            "primary_shape": "forbidden_shape",
            "primary_color": 8,
            "anchor_region": "forbidden_anchor",
            "distractor_count": 99,
        }
        row["query_type_key"] = "forbidden_query_type_key"
        row["counterfactual_group_id"] = "forbidden_group"
        row["target"] = [[8 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        row["seed"] = 123456 + index
        row["signature_code"] = f"forbidden_signature_{index}"
        row["oracle_source_coordinate"] = [index, index]
        row["oracle_transport_target"] = [GRID_SIZE - 1 - index, GRID_SIZE - 1 - index]
        row["oracle_color_transition_target"] = {"from": 1, "to": 8}
        row["factor_metadata"] = {"geometry": "forbidden", "recolor": "forbidden"}
        row["program"] = ["FORBIDDEN_PROGRAM_STEP"]
    return perturbed


def tensor_inputs_equal(left: SharedRendererInput, right: SharedRendererInput) -> bool:
    return all(torch.equal(getattr(left, name), getattr(right, name)) for name in ALLOWED_INPUT_FIELDS)


def module_source_hashes(paths: tuple[Path, ...]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for path in paths:
        result[str(path)] = sha256_file(path) if path.exists() else None
    return result


def source_code_version_hash() -> str:
    present = {str(path): sha256_file(path) for path in R5A1A_SOURCE_PATHS if path.exists()}
    return json_hash(present)


def parse_pytest_result(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": False, "path": None, "summary": None}
    if not path.exists():
        return {"provided": True, "passed": False, "path": str(path), "summary": "missing"}
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    lowered = text.lower()
    passed = bool(re.search(r"\b\d+\s+passed\b", lowered)) and not any(term in lowered for term in (" failed", " error", " errors", " interrupted"))
    last_line = text.splitlines()[-1] if text else ""
    return {
        "provided": True,
        "passed": passed,
        "path": str(path),
        "summary": last_line,
        "sha256": sha256_file(path),
    }


def linear_flops(batch_items: int, in_features: int, out_features: int) -> int:
    return int(2 * batch_items * in_features * out_features)


def ffn_flops(batch_size: int, token_count: int, hidden_size: int, mlp_ratio: int) -> int:
    inner = hidden_size * mlp_ratio
    return linear_flops(batch_size * token_count, hidden_size, inner) + linear_flops(batch_size * token_count, inner, hidden_size)


def self_attention_flops(batch_size: int, token_count: int, hidden_size: int) -> int:
    qkv = 3 * linear_flops(batch_size * token_count, hidden_size, hidden_size)
    out = linear_flops(batch_size * token_count, hidden_size, hidden_size)
    qk = 2 * batch_size * token_count * token_count * hidden_size
    av = 2 * batch_size * token_count * token_count * hidden_size
    return int(qkv + out + qk + av)


def cross_attention_flops(batch_size: int, query_tokens: int, memory_tokens: int, hidden_size: int) -> int:
    q_projection = linear_flops(batch_size * query_tokens, hidden_size, hidden_size)
    kv_projection = 2 * linear_flops(batch_size * memory_tokens, hidden_size, hidden_size)
    out_projection = linear_flops(batch_size * query_tokens, hidden_size, hidden_size)
    qk = 2 * batch_size * query_tokens * memory_tokens * hidden_size
    av = 2 * batch_size * query_tokens * memory_tokens * hidden_size
    return int(q_projection + kv_projection + out_projection + qk + av)


def shared_flop_breakdown(config: SharedRendererConfig, batch_size: int) -> dict[str, Any]:
    h = config.hidden_size
    encoder = ffn_flops(batch_size, MEMORY_TOKEN_COUNT, h, config.mlp_ratio)
    encoder += config.encoder_layers * ffn_flops(batch_size, MEMORY_TOKEN_COUNT, h, config.mlp_ratio)
    decoder_layers = 0
    for _ in range(config.decoder_layers):
        decoder_layers += self_attention_flops(batch_size, OUTPUT_CELL_COUNT, h)
        decoder_layers += cross_attention_flops(batch_size, OUTPUT_CELL_COUNT, MEMORY_TOKEN_COUNT, h)
        decoder_layers += ffn_flops(batch_size, OUTPUT_CELL_COUNT, h, config.mlp_ratio)
    color_head = linear_flops(batch_size * OUTPUT_CELL_COUNT, h, COLOR_COUNT)
    decoder = decoder_layers + color_head
    pooled_size_branch = linear_flops(batch_size, h, h * config.mlp_ratio)
    pooled_size_branch += linear_flops(batch_size, h * config.mlp_ratio, 2 * GRID_SIZE)
    total = encoder + decoder + pooled_size_branch
    return {
        "total_forward_flops": int(total),
        "encoder_forward_flops": int(encoder),
        "decoder_forward_flops": int(decoder),
        "output_size_branch_forward_flops": int(pooled_size_branch),
        "per_valid_output_cell_forward_flops": float(total / (batch_size * OUTPUT_CELL_COUNT)),
        "excluded_or_separately_reported": {
            "embedding_lookup_arithmetic": 0,
            "softmax_masking_norm_activation_index_reshape": "excluded consistently from totals",
            "embedding_parameter_count_reported_separately": True,
        },
    }


def v2_reference_flop_breakdown(batch_size: int) -> dict[str, Any]:
    hidden = 128
    pairs = SUPPORT_PAIRS
    summary_pair_projector = linear_flops(batch_size * pairs, 9, hidden)
    summary_pair_projector += linear_flops(batch_size * pairs, hidden, hidden)
    task_context = linear_flops(batch_size, hidden, hidden) + linear_flops(batch_size, hidden, hidden)
    factorized_output_einsum = 2 * batch_size * CELL_COUNT * CELL_COUNT * COLOR_COUNT
    total = summary_pair_projector + task_context + factorized_output_einsum
    return {
        "total_forward_flops": int(total),
        "encoder_forward_flops": int(summary_pair_projector + task_context),
        "decoder_forward_flops": int(factorized_output_einsum),
        "output_size_branch_forward_flops": 0,
        "per_valid_output_cell_forward_flops": float(total / (batch_size * OUTPUT_CELL_COUNT)),
        "notes": [
            "R5A-0 V2 uses explicit score comparisons, argmax, predicate logic, softmax, and indexing; these are excluded or separately reported under the same convention.",
            "Only linear/matmul/einsum arithmetic is counted in the main FLOP total.",
        ],
    }


def activation_proxy_shared(config: SharedRendererConfig, batch_size: int) -> int:
    h = config.hidden_size
    memory_states = batch_size * MEMORY_TOKEN_COUNT * h * (1 + config.encoder_layers)
    decoder_states = batch_size * OUTPUT_CELL_COUNT * h * (1 + config.decoder_layers)
    logits = batch_size * OUTPUT_CELL_COUNT * COLOR_COUNT + batch_size * 2 * GRID_SIZE
    masks = batch_size * (MEMORY_TOKEN_COUNT + OUTPUT_CELL_COUNT)
    return int((memory_states + decoder_states + logits) * 4 + masks)


def activation_proxy_v2(batch_size: int) -> int:
    hidden = 128
    relation = batch_size * CELL_COUNT * CELL_COUNT * 3
    pair_representations = batch_size * SUPPORT_PAIRS * hidden
    task_context = batch_size * hidden
    transport_attention = batch_size * CELL_COUNT * CELL_COUNT
    color_transition = batch_size * CELL_COUNT * CELL_COUNT * COLOR_COUNT
    output_probs = batch_size * CELL_COUNT * COLOR_COUNT
    return int((relation + pair_representations + task_context + transport_attention + color_transition + output_probs) * 4)


def instantiate_variant(config: SharedRendererConfig, batch: SharedRendererInput) -> dict[str, Any]:
    torch.manual_seed(DETERMINISTIC_SEED)
    model = GenericSharedRenderer(config).eval()
    with torch.no_grad():
        output = model(batch)
    return {
        "config": asdict(config),
        "trainable_parameters": count_trainable_parameters(model),
        "embedding_parameters": count_embedding_parameters(model),
        "forward_shapes": {
            "color_logits": list(output.color_logits.shape),
            "output_size_logits": list(output.output_size_logits.shape),
            "memory_token_mask": list(output.memory_token_mask.shape),
            "output_cell_mask": list(output.output_cell_mask.shape),
        },
        "output_mask_true_count": int(output.output_cell_mask.sum().item()),
        "size_logits_shape_semantics": "[batch, height_or_width_axis, possible_size_1_to_7]",
    }


def config_from_candidate(name: str, candidate: dict[str, Any]) -> SharedRendererConfig:
    return SharedRendererConfig(
        variant_name=name,
        hidden_size=int(candidate["hidden_size"]),
        num_heads=int(candidate["num_heads"]),
        encoder_layers=int(candidate["encoder_layers"]),
        decoder_layers=int(candidate["decoder_layers"]),
        mlp_ratio=int(candidate["mlp_ratio"]),
        dropout=0.0,
    )


def candidate_row(config: SharedRendererConfig, batch_size: int, v2_params: int, v2_flops_per_cell: float, v2_decoder_flops: int) -> dict[str, Any]:
    torch.manual_seed(DETERMINISTIC_SEED)
    model = GenericSharedRenderer(config)
    params = count_trainable_parameters(model)
    embeddings = count_embedding_parameters(model)
    flops = shared_flop_breakdown(config, batch_size)
    row = {
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "encoder_layers": config.encoder_layers,
        "decoder_layers": config.decoder_layers,
        "mlp_ratio": config.mlp_ratio,
        "trainable_parameters": params,
        "embedding_parameters": embeddings,
        "forward_flops": flops["total_forward_flops"],
        "decoder_forward_flops": flops["decoder_forward_flops"],
        "per_valid_output_cell_forward_flops": flops["per_valid_output_cell_forward_flops"],
        "parameter_ratio_vs_v2": float(params / v2_params),
        "total_flops_ratio_vs_v2": float(flops["per_valid_output_cell_forward_flops"] / v2_flops_per_cell),
        "flops_per_cell_ratio_vs_v2": float(flops["per_valid_output_cell_forward_flops"] / v2_flops_per_cell),
        "decoder_flops_ratio_vs_v2": float(flops["decoder_forward_flops"] / v2_decoder_flops) if v2_decoder_flops else math.inf,
    }
    row["matched_param_tolerance_met"] = 0.90 <= row["parameter_ratio_vs_v2"] <= 1.10
    row["matched_flop_tolerance_met"] = 0.85 <= row["flops_per_cell_ratio_vs_v2"] <= 1.15
    row["small_target_met"] = params < v2_params and row["per_valid_output_cell_forward_flops"] < v2_flops_per_cell
    row["large_target_met"] = 1.5 <= row["total_flops_ratio_vs_v2"] <= 2.5 or 1.5 <= row["parameter_ratio_vs_v2"] <= 2.5
    row["matched_objective"] = abs(math.log(max(row["parameter_ratio_vs_v2"], 1e-12))) + abs(math.log(max(row["flops_per_cell_ratio_vs_v2"], 1e-12)))
    return row


def capacity_search(batch_size: int, v2_params: int, v2_flops: dict[str, Any]) -> dict[str, Any]:
    ledger: list[dict[str, Any]] = []
    hidden_grid = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 136, 144, 152, 160, 192)
    for hidden in hidden_grid:
        for heads in (1, 2, 4, 8):
            if hidden % heads != 0 or heads > hidden:
                continue
            for encoder_layers in (0, 1):
                for decoder_layers in (1, 2, 3):
                    for mlp_ratio in (2, 4):
                        config = SharedRendererConfig(
                            "candidate",
                            hidden_size=hidden,
                            num_heads=heads,
                            encoder_layers=encoder_layers,
                            decoder_layers=decoder_layers,
                            mlp_ratio=mlp_ratio,
                        )
                        ledger.append(
                            candidate_row(
                                config,
                                batch_size=batch_size,
                                v2_params=v2_params,
                                v2_flops_per_cell=float(v2_flops["per_valid_output_cell_forward_flops"]),
                                v2_decoder_flops=int(v2_flops["decoder_forward_flops"]),
                            )
                        )
    ledger.sort(
        key=lambda row: (
            row["hidden_size"],
            row["num_heads"],
            row["encoder_layers"],
            row["decoder_layers"],
            row["mlp_ratio"],
        )
    )
    matched_candidates = [row for row in ledger if row["matched_param_tolerance_met"] and row["matched_flop_tolerance_met"]]
    if matched_candidates:
        matched = min(matched_candidates, key=lambda row: (row["matched_objective"], row["trainable_parameters"], row["forward_flops"]))
    else:
        parameter_band = [row for row in ledger if row["matched_param_tolerance_met"]]
        if parameter_band:
            matched = min(
                parameter_band,
                key=lambda row: (
                    abs(math.log(max(row["flops_per_cell_ratio_vs_v2"], 1e-12))),
                    abs(math.log(max(row["parameter_ratio_vs_v2"], 1e-12))),
                    row["forward_flops"],
                ),
            )
        else:
            matched = min(ledger, key=lambda row: (row["matched_objective"], row["trainable_parameters"], row["forward_flops"]))
    small_candidates = [row for row in ledger if row["small_target_met"]]
    small = min(small_candidates, key=lambda row: (-row["hidden_size"], row["forward_flops"])) if small_candidates else min(ledger, key=lambda row: (row["trainable_parameters"], row["forward_flops"]))
    large_param_candidates = [row for row in ledger if 1.5 <= row["parameter_ratio_vs_v2"] <= 2.5]
    large_candidates = large_param_candidates or [row for row in ledger if row["large_target_met"]]
    large = (
        min(
            large_candidates,
            key=lambda row: (
                abs(row["parameter_ratio_vs_v2"] - 2.0) if large_param_candidates else abs(row["total_flops_ratio_vs_v2"] - 2.0),
                abs(row["total_flops_ratio_vs_v2"] - 2.0),
                row["trainable_parameters"],
            ),
        )
        if large_candidates
        else max(ledger, key=lambda row: row["forward_flops"])
    )
    selected = {
        "shared_small": small,
        "shared_matched": matched,
        "shared_large": large,
    }
    return {
        "deterministic": True,
        "search_uses_accuracy_or_runtime": False,
        "grid": {
            "hidden_size": list(hidden_grid),
            "num_heads": [1, 2, 4, 8],
            "encoder_layers": [0, 1],
            "decoder_layers": [1, 2, 3],
            "mlp_ratio": [2, 4],
        },
        "candidate_count": len(ledger),
        "selected": selected,
        "matched_tolerances": {
            "parameter_ratio_range": [0.90, 1.10],
            "flops_per_valid_output_cell_ratio_range": [0.85, 1.15],
            "satisfied": bool(selected["shared_matched"]["matched_param_tolerance_met"] and selected["shared_matched"]["matched_flop_tolerance_met"]),
        },
        "small_target": {
            "below_v2_params_and_flops_per_valid_output_cell": bool(selected["shared_small"]["small_target_met"]),
        },
        "large_target": {
            "overall_forward_flops_or_parameter_ratio_range": [1.5, 2.5],
            "satisfied": bool(selected["shared_large"]["large_target_met"]),
        },
        "infeasibility_reasons": [] if matched_candidates else [
            "No semantically valid candidate in the deterministic discrete grid met both +/-10% parameter tolerance and +/-15% FLOPs-per-valid-output-cell tolerance.",
            "The selected shared_matched row is the nearest pre-training analytic budget candidate and the status is therefore INCONCLUSIVE unless the immutable preregistration explicitly permits this deviation.",
        ],
        "full_candidate_ledger": ledger,
    }


def field_inventory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = records[0]
    rows = [
        {
            "field_name": "query",
            "source_location": "record['query']",
            "shape_or_type": f"{GRID_SIZE}x{GRID_SIZE} int grid",
            "semantic_description": "raw query input grid",
            "status": "allowed",
            "reason": "raw query information is allowed for generic support/query encoding",
        },
        {
            "field_name": "supports[].input",
            "source_location": "record['supports'][i]['input']",
            "shape_or_type": f"{SUPPORT_PAIRS} x {GRID_SIZE}x{GRID_SIZE} int grid",
            "semantic_description": "raw support input grids",
            "status": "allowed",
            "reason": "raw support inputs are allowed",
        },
        {
            "field_name": "supports[].output",
            "source_location": "record['supports'][i]['output']",
            "shape_or_type": f"{SUPPORT_PAIRS} x {GRID_SIZE}x{GRID_SIZE} int grid",
            "semantic_description": "raw support output grids",
            "status": "allowed",
            "reason": "raw support outputs are allowed",
        },
    ]
    for key in sorted(first):
        if key in {"query", "supports"}:
            continue
        status = "forbidden" if key in {"operation_class", "sequence_class", "split", "target", "evaluation_only_operation_label"} else "unused"
        if key in {"query_type", "query_type_key", "counterfactual_group_id", "episode_id"}:
            status = "unused"
        rows.append(
            {
                "field_name": key,
                "source_location": f"record[{key!r}]",
                "shape_or_type": type(first[key]).__name__,
                "semantic_description": "V2 metadata or supervision surface not passed to the shared renderer",
                "status": status,
                "reason": "not present in SharedRendererInput allowlist and not consumed by the packer",
            }
        )
    for synthetic in (
        "seed",
        "signature_code",
        "oracle_source_coordinate",
        "oracle_transport_target",
        "oracle_color_transition_target",
        "factor_metadata",
        "program",
    ):
        rows.append(
            {
                "field_name": synthetic,
                "source_location": "synthetic perturbation audit field",
                "shape_or_type": "metadata",
                "semantic_description": "forbidden oracle/program/factor metadata or derived equivalent",
                "status": "forbidden",
                "reason": "explicitly excluded by the R5A-1A information firewall",
            }
        )
    return rows


def source_scan() -> dict[str, Any]:
    objects = {
        "GenericSharedRenderer.forward": GenericSharedRenderer.forward,
        "GenericSharedRenderer.encode_memory": GenericSharedRenderer.encode_memory,
        "pack_shared_input_from_records": pack_shared_input_from_records,
    }
    findings: list[dict[str, str]] = []
    scanned: dict[str, dict[str, Any]] = {}
    for name, obj in objects.items():
        source = textwrap.dedent(inspect.getsource(obj))
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source))
        scanned[name] = {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "token_count": len(tokens),
        }
        for term in FORBIDDEN_METADATA_TERMS:
            if term in tokens or term in source:
                findings.append({"object": name, "term": term})
    return {
        "method": "source string plus identifier-token scan over forward, memory encoding, and data-packing functions",
        "scanned_objects": scanned,
        "forbidden_terms_found": findings,
        "passed": findings == [],
    }


def firewall_audit(records: list[dict[str, Any]], batch: SharedRendererInput) -> dict[str, Any]:
    perturbed = perturb_forbidden_metadata(records)
    perturbed_batch = pack_shared_input_from_records(perturbed)
    torch.manual_seed(DETERMINISTIC_SEED)
    model_a = GenericSharedRenderer(SharedRendererConfig("audit", hidden_size=8, num_heads=1, encoder_layers=0, decoder_layers=1)).eval()
    with torch.no_grad():
        original_logits = model_a(batch)
        perturbed_logits = model_a(perturbed_batch)
    try:
        SharedRendererInput(
            query_grids=batch.query_grids,
            support_input_grids=batch.support_input_grids,
            support_output_grids=batch.support_output_grids,
            memory_token_mask=batch.memory_token_mask,
            output_cell_mask=batch.output_cell_mask,
            canvas_hw=batch.canvas_hw,
            operation_class=torch.zeros((len(records),), dtype=torch.long),  # type: ignore[call-arg]
        )
        forbidden_rejection = False
        forbidden_rejection_message = "unexpectedly accepted extra field"
    except TypeError as exc:
        forbidden_rejection = True
        forbidden_rejection_message = str(exc)
    scan = source_scan()
    logits_unchanged = bool(
        torch.equal(original_logits.color_logits, perturbed_logits.color_logits)
        and torch.equal(original_logits.output_size_logits, perturbed_logits.output_size_logits)
    )
    constructed_inputs_unchanged = tensor_inputs_equal(batch, perturbed_batch)
    return {
        "allowlisted_input_dataclass": "SharedRendererInput",
        "allowed_input_fields": list(ALLOWED_INPUT_FIELDS),
        "forbidden_field_names": list(FORBIDDEN_FIELD_NAMES),
        "field_inventory": field_inventory(records),
        "forward_accepts_only_allowlisted_dataclass": True,
        "forbidden_direct_field_rejected": forbidden_rejection,
        "forbidden_direct_field_rejection_message": forbidden_rejection_message,
        "metadata_perturbation": {
            "perturbed_fields": list(FORBIDDEN_METADATA_TERMS),
            "constructed_input_tensors_unchanged": constructed_inputs_unchanged,
            "constructed_input_sha256_original": tensor_digest(batch),
            "constructed_input_sha256_perturbed": tensor_digest(perturbed_batch),
            "eval_mode_logits_unchanged": logits_unchanged,
        },
        "source_scan": scan,
        "no_explicit_factor_copy_or_oracle_interface": bool(
            forbidden_rejection and constructed_inputs_unchanged and logits_unchanged and scan["passed"]
        ),
    }


def no_training_or_checkpoint_audit() -> dict[str, Any]:
    new_sources = [path for path in R5A1A_SOURCE_PATHS if path.exists()]
    pieces = []
    for path in new_sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces.append(source_without_strings_and_comments(text) if path.suffix == ".py" else text)
    source_text = "\n".join(pieces)
    lowered = source_text.lower()
    checkpoint_files = []
    if PREFLIGHT_ARTIFACT_ROOT.exists():
        checkpoint_files = [
            str(path)
            for path in PREFLIGHT_ARTIFACT_ROOT.rglob("*")
            if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
        ]
    training_terms = [term for term in ("torch.optim", ".backward(", "loss.backward", "train_one_seed(", "optimizer.step(") if term in lowered]
    sealed_eval_terms = [term for term in ("sealed.jsonl", "final_sealed", "evaluate_model(") if term in lowered]
    return {
        "training_started": False,
        "checkpoint_files_created": checkpoint_files,
        "training_loop_terms_found_in_new_sources": training_terms,
        "sealed_evaluation_terms_found_in_new_sources": sealed_eval_terms,
        "passed": checkpoint_files == [] and training_terms == [] and sealed_eval_terms == [],
    }


def source_without_strings_and_comments(source: str) -> str:
    pieces: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in {tokenize.STRING, tokenize.COMMENT}:
            pieces.append(" ")
        else:
            pieces.append(token.string)
    return " ".join(pieces)


def build_report(pytest_result_path: Path | None = None) -> dict[str, Any]:
    before_hashes = module_source_hashes(V2_PROTECTED_SOURCE_PATHS)
    dataset_manifest = load_json(V2_ARTIFACT_ROOT / "dataset_manifest.json")
    initialization = load_json(V2_ARTIFACT_ROOT / "initialization_report.json")
    preregistration = load_json(V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json")
    records = canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    canonical = canonical_batch_report(records, batch)
    v2_params = int(initialization["trainable_parameters"])
    v2_model = SupportConditionedTransportRecolorR5A0V2().eval()
    v2_actual_params = count_trainable_parameters(v2_model)
    v2_embedding_params = count_embedding_parameters(v2_model)
    v2_flops = v2_reference_flop_breakdown(batch_size=len(records))
    search = capacity_search(batch_size=len(records), v2_params=v2_params, v2_flops=v2_flops)
    variants: dict[str, dict[str, Any]] = {}
    for name, selected in search["selected"].items():
        config = config_from_candidate(name, selected)
        instantiated = instantiate_variant(config, batch)
        flops = shared_flop_breakdown(config, batch_size=len(records))
        instantiated["flops"] = flops
        instantiated["activation_proxy_bytes"] = activation_proxy_shared(config, batch_size=len(records))
        variants[name] = instantiated
    firewall = firewall_audit(records, batch)
    no_training = no_training_or_checkpoint_audit()
    after_hashes = module_source_hashes(V2_PROTECTED_SOURCE_PATHS)
    pytest_result = parse_pytest_result(pytest_result_path)
    variant_forwards_ok = all(
        row["forward_shapes"]["color_logits"] == [len(records), GRID_SIZE, GRID_SIZE, COLOR_COUNT]
        and row["forward_shapes"]["output_size_logits"] == [len(records), 2, GRID_SIZE]
        for row in variants.values()
    )
    canonical_second = canonical_batch_report(records, pack_shared_input_from_records(records))
    flops_second = shared_flop_breakdown(config_from_candidate("shared_matched", search["selected"]["shared_matched"]), batch_size=len(records))
    deterministic_gates = {
        "canonical_batch_hash_repeats": canonical["tensor_sha256"] == canonical_second["tensor_sha256"],
        "flop_counter_repeats": variants["shared_matched"]["flops"] == flops_second,
    }
    matched_prereg_text = json.dumps(preregistration.get("planned_shared_variants", {}).get("shared_matched", {})).lower()
    prereg_allows_matched_deviation = (
        "flop" in matched_prereg_text
        and "unless implementation arithmetic makes that impossible" in matched_prereg_text
    )
    gates = {
        "v2_source_hashes_unchanged_during_preflight": before_hashes == after_hashes,
        "all_three_variants_instantiate_and_forward": variant_forwards_ok and set(variants) == {"shared_small", "shared_matched", "shared_large"},
        "no_explicit_factor_copy_or_oracle_interface": firewall["no_explicit_factor_copy_or_oracle_interface"],
        "information_firewall_passed": firewall["no_explicit_factor_copy_or_oracle_interface"],
        "canonical_batch_deterministic": deterministic_gates["canonical_batch_hash_repeats"],
        "flop_counter_deterministic": deterministic_gates["flop_counter_repeats"],
        "matched_budget_tolerances_satisfied": search["matched_tolerances"]["satisfied"],
        "matched_budget_deviation_permitted_by_immutable_prereg": bool(
            (not search["matched_tolerances"]["satisfied"]) and prereg_allows_matched_deviation
        ),
        "full_pytest_passed": pytest_result["passed"],
        "no_training_checkpoint_or_sealed_eval": no_training["passed"],
    }
    hard_failures = [
        name
        for name in (
            "v2_source_hashes_unchanged_during_preflight",
            "all_three_variants_instantiate_and_forward",
            "no_explicit_factor_copy_or_oracle_interface",
            "information_firewall_passed",
            "canonical_batch_deterministic",
            "flop_counter_deterministic",
            "no_training_checkpoint_or_sealed_eval",
        )
        if not gates[name]
    ]
    if hard_failures:
        verdict = "R5A1A_PREFLIGHT_FAIL"
        reasons = [f"hard gate failed: {name}" for name in hard_failures]
    elif gates["matched_budget_tolerances_satisfied"] and gates["full_pytest_passed"]:
        verdict = "R5A1A_PREFLIGHT_PASS"
        reasons = ["all hard gates passed, matched budget tolerances satisfied, and full pytest passed"]
    else:
        verdict = "R5A1A_PREFLIGHT_INCONCLUSIVE"
        reasons = []
        if not gates["matched_budget_tolerances_satisfied"]:
            reasons.extend(search["infeasibility_reasons"])
            if not gates["matched_budget_deviation_permitted_by_immutable_prereg"]:
                reasons.append("the immutable preregistration does not provide a waiver for the new combined parameter and FLOPs-per-valid-output-cell tolerance miss")
        if not gates["full_pytest_passed"]:
            reasons.append("full pytest result was not provided or did not pass at report-build time")
    report: dict[str, Any] = {
        "preflight_version": PREFLIGHT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": verdict,
        "verdict_reasons": reasons,
        "neutrality_statement": (
            "This R5A-1A preflight does not state or assume that the generic shared renderer is expected to pass or fail future training. "
            "It only fixes and audits the control architecture family, input firewall, deterministic batch, and analytic capacity ledger."
        ),
        "pre_registered_future_outcome_categories": preregistration["pre_registered_outcome_categories"],
        "thresholds_from_immutable_preregistration": preregistration["pre_registered_thresholds"],
        "source_artifact_anchors": {
            "dataset_manifest": {
                "path": str(V2_ARTIFACT_ROOT / "dataset_manifest.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "dataset_manifest.json"),
                "split_episode_counts": dataset_manifest["split_episode_counts"],
            },
            "initialization_report": {
                "path": str(V2_ARTIFACT_ROOT / "initialization_report.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "initialization_report.json"),
            },
            "r5a1a_preregistration_report": {
                "path": str(V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json"),
            },
            "train_jsonl": {
                "path": str(V2_ARTIFACT_ROOT / "train.jsonl"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "train.jsonl"),
            },
        },
        "code_version_hash": source_code_version_hash(),
        "protected_v2_source_hashes_before": before_hashes,
        "protected_v2_source_hashes_after": after_hashes,
        "canonical_profiling_batch": canonical,
        "allowed_raw_information": {
            "query_input_grid": True,
            "support_input_output_grids": True,
            "raw_grid_tokenization": True,
            "positional_encoding": True,
            "pair_and_example_membership": True,
            "masks": True,
            "generic_canvas_bounds": True,
            "target_grid_or_ground_truth_output_size": False,
            "operation_program_split_seed_signature_oracle_factor_metadata": False,
        },
        "shared_renderer_architecture_family": {
            "allowed_components": [
                "generic color and position embeddings",
                "generic support/query memory encoder",
                "generic learned output-cell query tokens",
                "generic self-attention and cross-attention",
                "generic MLPs",
                "final per-cell color logits",
                "generic pooled output-size logits based only on allowed support/query tensors",
            ],
            "forbidden_components": [
                "explicit source-coordinate prediction",
                "explicit pointer distribution or pointer supervision",
                "copy gate or source-copy routing",
                "explicit semantic transport representation",
                "explicit color-transition representation",
                "factor-labelled latent channels",
                "separate geometry or recolor heads",
                "auxiliary geometry/recolor losses",
                "operation/program/class/split/seed/signature/oracle metadata or derived equivalents",
            ],
            "generic_attention_policy": "Generic attention is allowed; it is not classified as forbidden merely because it may internally learn correspondence.",
        },
        "information_firewall_audit": firewall,
        "flop_counting_convention": {
            "linear_and_matmul": "2*M*N*K",
            "attention": "includes Q/K/V projections, output projection, QK^T, and attention-value products",
            "mlp": "all linears counted",
            "embedding_lookup_arithmetic": 0,
            "excluded_from_main_total": ["softmax", "masking", "normalization", "activation", "indexing", "reshape", "comparisons"],
            "embedding_parameter_count_reported_separately": True,
            "counter_type": "analytic formula, not runtime profiler",
        },
        "v2_reference_budget": {
            "architecture": initialization["architecture"],
            "trainable_parameters_from_initialization_report": v2_params,
            "trainable_parameters_recounted": v2_actual_params,
            "embedding_parameters_recounted": v2_embedding_params,
            "flops": v2_flops,
            "activation_proxy_bytes": activation_proxy_v2(batch_size=len(records)),
        },
        "capacity_search": search,
        "variants": variants,
        "determinism_audit": deterministic_gates,
        "no_training_checkpoint_or_sealed_eval_audit": no_training,
        "pytest_result": pytest_result,
        "preflight_gates": gates,
    }
    report["report_sha256_excluding_timestamp"] = stable_report_hash(report)
    return report


def write_report(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-1A shared-renderer architecture preflight")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.pytest_result)
    write_report(report, args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out), "report_sha256_excluding_timestamp": report["report_sha256_excluding_timestamp"]}, sort_keys=True))
    return 0 if report["status"] != "R5A1A_PREFLIGHT_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
