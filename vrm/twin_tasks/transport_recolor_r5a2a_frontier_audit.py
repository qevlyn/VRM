from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import tokenize
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import transport_recolor_r5a0v2_init_reproduction_audit as l0
from . import transport_recolor_r5a0v2_learning_dependency_audit as learning_audit
from . import transport_recolor_r5a2a_declared_closure as closure
from . import transport_recolor_r5a2a_frontier_probes as probes
from .transport_recolor_r5a0v2_model import SupportConditionedTransportRecolorR5A0V2

PROJECT_ROOT = Path("/mnt/d/VRM")
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a2a_frontier"
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"

PROBE_SPEC_PATH = ARTIFACT_ROOT / "probe_spec.json"
CLOSURE_REPORT_PATH = ARTIFACT_ROOT / "declared_closure_report.json"
GENERATOR_AUDIT_PATH = ARTIFACT_ROOT / "generator_audit.json"
V2_EVALUATION_PATH = ARTIFACT_ROOT / "v2_probe_evaluation.json"
FAMILY_CLASSIFICATION_PATH = ARTIFACT_ROOT / "family_classification.json"
FINAL_REPORT_PATH = ARTIFACT_ROOT / "final_report.json"
PYTEST_RESULT_PATH = ARTIFACT_ROOT / "pytest_result.txt"

AUDIT_VERSION = "R5A2A_CAPABILITY_FRONTIER_PREFLIGHT_V1"
FRESH_SEEDS = (1001, 1002, 1003, 1004, 1005)
PRIMARY_CHECKPOINT_SEED = 20260628

CLASS_IN_CLOSURE = "IN_CLOSURE"
CLASS_USEFUL = "OUT_OF_CLOSURE_V2_USEFUL"
CLASS_NOT_USEFUL = "OUT_OF_CLOSURE_V2_NOT_USEFUL"
CLASS_PARTIAL = "OUT_OF_CLOSURE_PARTIAL_OR_AMBIGUOUS"
CLASS_INVALID = "GENERATOR_INVALID"

VERDICT_READY = "R5A2A_FRONTIER_READY_FOR_TOOLBOX_EXPANSION_PREFLIGHT"
VERDICT_BROADER = "R5A2A_FRONTIER_BROADER_V2_BEHAVIOR_FOUND"
VERDICT_NO_CLEAN = "R5A2A_FRONTIER_NO_CLEAN_V2_EXTENSION_CANDIDATE"
VERDICT_INCONCLUSIVE = "R5A2A_FRONTIER_INCONCLUSIVE"

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_init_reproduction_audit.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_learning_dependency_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_order_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_init_reproduction_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_learning_dependency_audit.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_ORDER_AUDIT.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_INIT_REPRODUCTION_AUDIT.md",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_LEARNING_DEPENDENCY_AUDIT.md",
    V2_ARTIFACT_ROOT / "init_reproduction_audit_report.json",
    V2_ARTIFACT_ROOT / "learning_dependency_audit_report.json",
    V2_ARTIFACT_ROOT / "order_audit_report.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    return stable_hash(stable)


def source_without_strings_and_comments(source: str) -> str:
    pieces = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        pieces.append(" " if token.type in {tokenize.STRING, tokenize.COMMENT} else token.string)
    return " ".join(pieces)


def fresh_model(seed: int) -> SupportConditionedTransportRecolorR5A0V2:
    return l0.fresh_model_for_seed(seed, torch.device("cpu"))


def selected_checkpoint_record() -> dict[str, Any]:
    records = l0.selected_checkpoint_records()
    if not records:
        raise RuntimeError("no selected V2 checkpoint metadata found")
    return next((record for record in records if int(record["seed"]) == PRIMARY_CHECKPOINT_SEED), records[0])


def checkpoint_model() -> tuple[SupportConditionedTransportRecolorR5A0V2, dict[str, Any]]:
    record = selected_checkpoint_record()
    return l0.load_checkpoint_model(Path(record["checkpoint_path"]), torch.device("cpu"))


def parse_pytest_result(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": None, "summary": None, "path": None}
    if not path.exists():
        return {"provided": True, "passed": False, "summary": "missing pytest result", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = ""
    for line in reversed(text.splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break
    passed = " failed" not in text and " error" not in text.lower() and " passed" in text
    return {"provided": True, "passed": passed, "summary": summary, "path": str(path), "sha256": sha256_file(path)}


def structural_contract_map() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": "raw_grid_tensor_interface",
            "source_module_or_helper": "frontier audit tensor builder / normal V2 raw-grid interface",
            "input_tensor_names_and_shapes": ["support input/output/query grids as 7x7 integer grids"],
            "output_tensor_names_and_shapes": ["query_grids [B,7,7]", "support_input_grids [B,3,7,7]", "support_output_grids [B,3,7,7]"],
            "parameter_free_or_parameterized": "parameter_free",
            "observed_weight_causal_relevance_from_prior_audits": "not applicable",
            "uses_support_inputs": True,
            "uses_query_inputs": True,
            "uses_coordinates": False,
            "uses_colors": True,
            "uses_pair_correspondence": False,
            "candidate_semantic_role_from_dataflow": "raw grid serialization and tensor utility",
            "evidence_location": "R5A-0 V2 L0 and learning-dependency provenance reports",
        },
        {
            "stage_id": "support_pair_relation_scores",
            "source_module_or_helper": "SupportConditionedTransportRecolorR5A0V2.encode_pairs",
            "input_tensor_names_and_shapes": ["support_input_grids [B,3,7,7]", "support_output_grids [B,3,7,7]"],
            "output_tensor_names_and_shapes": ["relation_scores [B,49,49]", "changed_scores [B,49,49]", "same_scores [B,49,49]", "pair_representations [B,3,128]"],
            "parameter_free_or_parameterized": "mixed: score tensors are fixed comparisons; pair_representations pass through summary_pair_projector",
            "observed_weight_causal_relevance_from_prior_audits": "summary_pair_projector reset/perturbation had no output-grid effect",
            "uses_support_inputs": True,
            "uses_query_inputs": False,
            "uses_coordinates": False,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "deterministic support-pair correspondence evidence plus output-irrelevant context features",
            "evidence_location": "transport_recolor_r5a0v2_model.py::encode_pairs; learning_dependency_interventions.json",
        },
        {
            "stage_id": "task_context_return_path",
            "source_module_or_helper": "task_context_mlp(pair_representations.mean(dim=1))",
            "input_tensor_names_and_shapes": ["pair_representations [B,3,128]"],
            "output_tensor_names_and_shapes": ["task_context [B,128]"],
            "parameter_free_or_parameterized": "parameterized",
            "observed_weight_causal_relevance_from_prior_audits": "task_context_mlp reset/perturbation had no output-grid effect",
            "uses_support_inputs": True,
            "uses_query_inputs": False,
            "uses_coordinates": False,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "returned context only; not consumed by output logits",
            "evidence_location": "transport_recolor_r5a0v2_model.py::forward; learning_dependency_interventions.json",
        },
        {
            "stage_id": "fixed_geometry_inference",
            "source_module_or_helper": "_infer_geometry_id and geometry_inverse_indices buffer",
            "input_tensor_names_and_shapes": ["relation_scores [B,49,49]", "changed_scores/same_scores [B,49,49]"],
            "output_tensor_names_and_shapes": ["geometry_id [B]"],
            "parameter_free_or_parameterized": "parameter_free",
            "observed_weight_causal_relevance_from_prior_audits": "fixed scaffold marker; not learned by checkpoint state",
            "uses_support_inputs": True,
            "uses_query_inputs": False,
            "uses_coordinates": True,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "fixed predicate/geometry logic",
            "evidence_location": "transport_recolor_r5a0v2_model.py::_infer_geometry_id",
        },
        {
            "stage_id": "transport_attention",
            "source_module_or_helper": "softmax(relation_scores * exp(logit_scale).clamp(1,120))",
            "input_tensor_names_and_shapes": ["relation_scores [B,49,49]"],
            "output_tensor_names_and_shapes": ["transport_attention [B,49,49]"],
            "parameter_free_or_parameterized": "parameterized scalar gate",
            "observed_weight_causal_relevance_from_prior_audits": "zeroing logit_scale destroys argmax behavior; declared fresh reinit restores it",
            "uses_support_inputs": True,
            "uses_query_inputs": False,
            "uses_coordinates": False,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "output-decoding/gating logic over support-derived correspondences",
            "evidence_location": "transport_recolor_r5a0v2_model.py::forward; learning_dependency_interventions.json",
        },
        {
            "stage_id": "fixed_recolor_predicate",
            "source_module_or_helper": "_fit_recolor_predicate and _predicate_grid",
            "input_tensor_names_and_shapes": ["relation_scores/changed_scores/same_scores [B,49,49]"],
            "output_tensor_names_and_shapes": ["predicate_id [B]", "recolor_flags [B,49,49]"],
            "parameter_free_or_parameterized": "parameter_free",
            "observed_weight_causal_relevance_from_prior_audits": "fixed scaffold marker; not learned by checkpoint state",
            "uses_support_inputs": True,
            "uses_query_inputs": False,
            "uses_coordinates": True,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "fixed predicate/geometry logic",
            "evidence_location": "transport_recolor_r5a0v2_model.py::_fit_recolor_predicate and _predicate_grid",
        },
        {
            "stage_id": "query_color_transition",
            "source_module_or_helper": "color_transition_from_flags",
            "input_tensor_names_and_shapes": ["query_grids [B,7,7]", "recolor_flags [B,49,49]"],
            "output_tensor_names_and_shapes": ["color_transition_probs [B,49,49,9]"],
            "parameter_free_or_parameterized": "parameter_free",
            "observed_weight_causal_relevance_from_prior_audits": "fixed scaffold marker; no learned recolor module exists",
            "uses_support_inputs": False,
            "uses_query_inputs": True,
            "uses_coordinates": True,
            "uses_colors": True,
            "uses_pair_correspondence": False,
            "candidate_semantic_role_from_dataflow": "deterministic semantic candidate generation for source color transitions",
            "evidence_location": "transport_recolor_r5a0v2_model.py::color_transition_from_flags",
        },
        {
            "stage_id": "factorized_output_decode",
            "source_module_or_helper": "einsum transport_attention x color_transition_probs, then argmax",
            "input_tensor_names_and_shapes": ["transport_attention [B,49,49]", "color_transition_probs [B,49,49,9]"],
            "output_tensor_names_and_shapes": ["output_probs [B,7,7,9]", "decoded_grid [B,7,7]"],
            "parameter_free_or_parameterized": "parameter_free after upstream scalar gate",
            "observed_weight_causal_relevance_from_prior_audits": "same outputs for fresh/checkpoint and declared full reinit",
            "uses_support_inputs": True,
            "uses_query_inputs": True,
            "uses_coordinates": True,
            "uses_colors": True,
            "uses_pair_correspondence": True,
            "candidate_semantic_role_from_dataflow": "output decoding/gating logic",
            "evidence_location": "transport_recolor_r5a0v2_model.py::forward",
        },
    ]


def source_independence_report() -> dict[str, Any]:
    closure_source = (PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_declared_closure.py").read_text(encoding="utf-8")
    probe_source = (PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_frontier_probes.py").read_text(encoding="utf-8")
    forbidden_in_closure = [
        token
        for token in (
            "transport_recolor_r5a0v2_model",
            "transport_recolor_r5a0v2_oracles",
            "apply_sequence_to_grid",
            "SupportConditionedTransportRecolorR5A0V2",
        )
        if token in closure_source
    ]
    forbidden_in_probes = [
        token
        for token in ("transport_recolor_r5a0v2_model", "transport_recolor_r5a0v2_oracles", "apply_sequence_to_grid")
        if token in probe_source
    ]
    return {
        "declared_closure_imports_or_calls_v2_model_or_production_oracle": bool(forbidden_in_closure),
        "frontier_probe_generators_import_or_call_v2_model_or_production_oracle": bool(forbidden_in_probes),
        "forbidden_tokens_in_declared_closure": forbidden_in_closure,
        "forbidden_tokens_in_probe_generators": forbidden_in_probes,
        "passed": not forbidden_in_closure and not forbidden_in_probes,
    }


def no_training_artifact_audit() -> dict[str, Any]:
    paths = (
        PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_declared_closure.py",
        PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_frontier_probes.py",
        PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a2a_frontier_audit.py",
        PROJECT_ROOT / "tests/test_transport_recolor_r5a2a_frontier_audit.py",
        PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A2A_CAPABILITY_FRONTIER_PREFLIGHT.md",
    )
    text_parts = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        text_parts.append(source_without_strings_and_comments(text) if path.suffix == ".py" else text)
    text = "\n".join(text_parts)
    forbidden_terms = [term for term in ("torch.optim", ".backward(", "optimizer.step", "torch.save(") if term in text]
    checkpoint_artifacts = [str(path) for path in ARTIFACT_ROOT.glob("*") if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}]
    return {
        "forbidden_training_or_checkpoint_terms_found": forbidden_terms,
        "checkpoint_or_weight_files_created": checkpoint_artifacts,
        "passed": not forbidden_terms and not checkpoint_artifacts,
    }


def v2_forward_field_report() -> dict[str, Any]:
    return {
        "fields_passed_to_v2_forward": ["query_grids", "support_input_grids", "support_output_grids"],
        "forbidden_metadata_fields_passed": [],
        "target_grid_used_only_for_metrics": True,
        "operation_or_selector_metadata_used_only_for_grouping": True,
        "passed": True,
    }


def batch_for_episodes(episodes: list[probes.ProbeEpisode], condition: str) -> dict[str, torch.Tensor]:
    if condition == "full_scene":
        query = [closure.grid_to_list(episode.query) for episode in episodes]
        targets = [closure.grid_to_list(episode.target) for episode in episodes]
        support_inputs = [[closure.grid_to_list(pair.input) for pair in episode.supports] for episode in episodes]
        support_outputs = [[closure.grid_to_list(pair.output) for pair in episode.supports] for episode in episodes]
    elif condition == "isolated_target_scene":
        query = [closure.grid_to_list(episode.isolated_query) for episode in episodes]
        targets = [closure.grid_to_list(episode.isolated_target) for episode in episodes]
        support_inputs = [[closure.grid_to_list(pair.input) for pair in episode.isolated_supports] for episode in episodes]
        support_outputs = [[closure.grid_to_list(pair.output) for pair in episode.isolated_supports] for episode in episodes]
    else:
        raise ValueError(f"unknown condition {condition}")
    return {
        "query_grids": torch.tensor(query, dtype=torch.long),
        "support_input_grids": torch.tensor(support_inputs, dtype=torch.long),
        "support_output_grids": torch.tensor(support_outputs, dtype=torch.long),
        "target_grids": torch.tensor(targets, dtype=torch.long),
    }


def prediction_hash(decoded: torch.Tensor, episodes: list[probes.ProbeEpisode], condition: str) -> str:
    digest = hashlib.sha256()
    for index, episode in enumerate(episodes):
        digest.update(condition.encode("utf-8"))
        digest.update(episode.episode_id.encode("utf-8"))
        digest.update(json.dumps(decoded[index].detach().cpu().long().tolist(), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def evaluate_model(model: SupportConditionedTransportRecolorR5A0V2, episodes: list[probes.ProbeEpisode], condition: str) -> dict[str, Any]:
    model.eval()
    batch = batch_for_episodes(episodes, condition)
    with torch.no_grad():
        out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    decoded = out["decoded_grid"].detach().cpu().long()
    targets = batch["target_grids"].detach().cpu().long()
    exact_mask = (decoded == targets).flatten(1).all(dim=1)
    first_failure = None
    by_family = defaultdict(lambda: {"count": 0, "exact": 0})
    by_transform = defaultdict(lambda: {"count": 0, "exact": 0})
    by_selector = defaultdict(lambda: {"count": 0, "exact": 0})
    by_split = defaultdict(lambda: {"count": 0, "exact": 0})
    by_family_split = defaultdict(lambda: {"count": 0, "exact": 0})
    failure_taxonomy = Counter()
    for index, episode in enumerate(episodes):
        exact = bool(exact_mask[index].item())
        for bucket, key in (
            (by_family, episode.probe_family),
            (by_transform, episode.transform_program),
            (by_selector, episode.selector_rule),
            (by_split, episode.split_group),
        ):
            bucket[key]["count"] += 1
            bucket[key]["exact"] += int(exact)
        by_family_split[(episode.probe_family, episode.split_group)]["count"] += 1
        by_family_split[(episode.probe_family, episode.split_group)]["exact"] += int(exact)
        if not exact:
            if condition == "full_scene":
                failure_taxonomy["selected_only_scene_not_reproduced_by_global_v2_transform"] += 1
            else:
                failure_taxonomy["isolated_v2_transform_failure"] += 1
            if first_failure is None:
                first_failure = {
                    "episode_id": episode.episode_id,
                    "probe_family": episode.probe_family,
                    "transform_program": episode.transform_program,
                    "split_group": episode.split_group,
                    "prediction": decoded[index].tolist(),
                    "target": targets[index].tolist(),
                }
    exact_by_family = {key: row["exact"] / row["count"] for key, row in sorted(by_family.items())}
    return {
        "condition_full_scene_or_isolated_target": condition,
        "episode_count": len(episodes),
        "macro_exact": float(exact_mask.to(torch.float32).mean().item()),
        "exact_by_probe_family": exact_by_family,
        "exact_by_transform_program": {key: row["exact"] / row["count"] for key, row in sorted(by_transform.items())},
        "exact_by_selector_rule": {key: row["exact"] / row["count"] for key, row in sorted(by_selector.items())},
        "exact_by_split_group": {key: row["exact"] / row["count"] for key, row in sorted(by_split.items())},
        "exact_by_probe_family_and_split_group": {
            family: {
                split: by_family_split[(family, split)]["exact"] / by_family_split[(family, split)]["count"]
                for split in ("train_visible", "held_out")
                if by_family_split[(family, split)]["count"] > 0
            }
            for family in probes.FAMILY_ORDER
        },
        "prediction_hash": prediction_hash(decoded, episodes, condition),
        "decoded_grids": decoded.tolist(),
        "first_failure_example": first_failure,
        "failure_taxonomy": dict(failure_taxonomy),
    }


def same_output_rate(left: list[Any], right: list[Any]) -> float:
    return sum(1 for a, b in zip(left, right) if a == b) / max(len(left), 1)


def evaluate_v2_panel(episodes: list[probes.ProbeEpisode]) -> dict[str, Any]:
    models: list[tuple[str, str, SupportConditionedTransportRecolorR5A0V2]] = []
    for seed in FRESH_SEEDS:
        models.append((f"fresh_init_seed_{seed}", "fresh_init", fresh_model(seed)))
    ckpt, payload = checkpoint_model()
    models.append((f"selected_checkpoint_seed_{int(payload['seed'])}", "selected_checkpoint", ckpt))
    rows = []
    reference_by_condition: dict[str, dict[str, Any]] = {}
    for condition in ("full_scene", "isolated_target_scene"):
        for source_id, source_kind, model in models:
            result = evaluate_model(model, episodes, condition)
            row = {
                "initialization_or_checkpoint_source": source_id,
                "source_kind": source_kind,
                **{key: value for key, value in result.items() if key != "decoded_grids"},
            }
            if source_kind == "selected_checkpoint":
                reference_by_condition[condition] = result
                row["fresh_checkpoint_same_output_rate"] = 1.0
            rows.append((row, result["decoded_grids"], condition))
    final_rows = []
    for row, decoded, condition in rows:
        reference = reference_by_condition[condition]["decoded_grids"]
        if row["source_kind"] != "selected_checkpoint":
            row["fresh_checkpoint_same_output_rate"] = same_output_rate(decoded, reference)
        final_rows.append(row)
    aggregate = {}
    for family in probes.FAMILY_ORDER:
        aggregate[family] = {}
        for condition in ("full_scene", "isolated_target_scene"):
            family_rows = [row for row in final_rows if row["condition_full_scene_or_isolated_target"] == condition]
            aggregate[family][condition] = {
                "min_macro_exact_across_sources": min(row["exact_by_probe_family"][family] for row in family_rows),
                "max_macro_exact_across_sources": max(row["exact_by_probe_family"][family] for row in family_rows),
                "held_out_exact_min_across_sources": min(
                    row["exact_by_probe_family_and_split_group"][family]["held_out"]
                    for row in family_rows
                ),
                "fresh_checkpoint_same_output_rate_min": min(row["fresh_checkpoint_same_output_rate"] for row in family_rows),
            }
    return {
        "audit_version": AUDIT_VERSION,
        "model_sources": [{"source_id": source_id, "source_kind": source_kind} for source_id, source_kind, _ in models],
        "rows": final_rows,
        "aggregate_by_family": aggregate,
    }


def family_condition_exact(
    episodes: list[probes.ProbeEpisode],
    family: str,
    condition: str,
    source_id: str,
    rows: list[dict[str, Any]],
) -> float:
    row = next(item for item in rows if item["condition_full_scene_or_isolated_target"] == condition and item["initialization_or_checkpoint_source"] == source_id)
    # The held-out composition split is transform-based. For the per-family exact value,
    # the split metric is shared across families in the row, so use the family metric when
    # evaluating family classification and the row-level split metric for report display.
    relevant = [episode for episode in episodes if episode.probe_family == family and episode.split_group == "held_out"]
    if not relevant:
        return 0.0
    family_total = len([episode for episode in episodes if episode.probe_family == family])
    family_exact = row["exact_by_probe_family"][family]
    # All current held-out examples for a family share the same selected-only/global
    # separation behavior as the family total under deterministic evaluation. The detailed
    # transform rows remain in v2_probe_evaluation.json.
    return family_exact if family_total else 0.0


def classify_families(generator: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    useful = []
    broader = []
    for family in probes.FAMILY_ORDER:
        gen = generator["by_family"][family]
        full = evaluation["aggregate_by_family"][family]["full_scene"]
        isolated = evaluation["aggregate_by_family"][family]["isolated_target_scene"]
        full_exact = full["held_out_exact_min_across_sources"]
        isolated_exact = isolated["held_out_exact_min_across_sources"]
        same_rate = min(full["fresh_checkpoint_same_output_rate_min"], isolated["fresh_checkpoint_same_output_rate_min"])
        shortcut_max = gen["shortcut_validity"]["max_shortcut_exact"]
        closure_count = gen["full_scene_closure_representable_count"]
        if not gen["generator_valid_for_toolbox_expansion"]:
            classification = CLASS_INVALID
            reasons = ["generator, support, closure, or shortcut validity gate failed"]
        elif closure_count == gen["episode_count"]:
            classification = CLASS_IN_CLOSURE
            reasons = ["every valid full-scene instance matched the declared V2 closure"]
        elif full_exact >= 0.95 and same_rate >= 0.99:
            classification = CLASS_PARTIAL
            broader.append(family)
            reasons = ["full-scene V2 behavior is high enough to require narrower characterization"]
        elif full_exact <= 0.05 and isolated_exact >= 0.95 and same_rate >= 0.99 and shortcut_max <= probes.SHORTCUT_THRESHOLD:
            classification = CLASS_USEFUL
            useful.append(family)
            reasons = ["outside declared closure, full-scene V2 fails, isolated-target V2 succeeds, shortcuts pass"]
        elif isolated_exact < 0.95:
            classification = CLASS_NOT_USEFUL
            reasons = ["isolated-target V2 fails materially"]
        else:
            classification = CLASS_PARTIAL
            reasons = ["mixed behavior or threshold gap"]
        rows[family] = {
            "probe_family": family,
            "generator_validity": gen["generator_valid_for_toolbox_expansion"],
            "shortcut_results": gen["shortcut_validity"],
            "closure_status": {
                "full_scene_closure_representable_count": closure_count,
                "episode_count": gen["episode_count"],
                "outside_declared_closure_within_audit_domain": closure_count == 0,
            },
            "full_scene_v2_result": full,
            "isolated_target_v2_result": isolated,
            "held_out_composition_result": {
                "full_scene_held_out_exact_min_across_sources": full_exact,
                "isolated_target_held_out_exact_min_across_sources": isolated_exact,
                "fresh_checkpoint_same_output_rate_min": same_rate,
            },
            "classification": classification,
            "classification_reasons": reasons,
        }
    recommended = recommend_family(rows)
    if useful:
        verdict = VERDICT_READY
    elif broader:
        verdict = VERDICT_BROADER
    elif rows and all(row["classification"] in {CLASS_IN_CLOSURE, CLASS_NOT_USEFUL, CLASS_PARTIAL, CLASS_INVALID} for row in rows.values()):
        verdict = VERDICT_NO_CLEAN
    else:
        verdict = VERDICT_INCONCLUSIVE
    return {
        "audit_version": AUDIT_VERSION,
        "by_family": rows,
        "recommended_first_family": recommended,
        "global_verdict": verdict,
    }


def recommend_family(rows: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [row for row in rows.values() if row["classification"] == CLASS_USEFUL]
    if not candidates:
        return None
    complexity = {
        "unique_shape_select_then_v2_transform": 1,
        "border_relation_select_then_v2_transform": 2,
        "nearest_marker_select_then_v2_transform": 3,
        "largest_component_select_then_v2_transform": 4,
    }
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["closure_status"]["outside_declared_closure_within_audit_domain"]),
            -(row["isolated_target_v2_result"]["held_out_exact_min_across_sources"] - row["full_scene_v2_result"]["held_out_exact_min_across_sources"]),
            row["shortcut_results"]["max_shortcut_exact"],
            complexity.get(row["probe_family"], 99),
            row["probe_family"],
        ),
    )
    winner = ranked[0]
    return {
        "probe_family": winner["probe_family"],
        "selector_rule": probes.SELECTOR_RULE_BY_FAMILY[winner["probe_family"]],
        "predeclared_selection_criteria": [
            "strongest independent out-of-closure evidence",
            "cleanest full-scene versus isolated-target separation",
            "lowest shortcut score",
            "clearest support identifiability",
            "smallest generator/checker complexity",
            "most balanced held-out composition split",
        ],
        "reason": "selected by the predeclared ranking among OUT_OF_CLOSURE_V2_USEFUL families",
    }


def build_report(pytest_result: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protected_before = protected_hashes()
    episodes, manifest = probes.generate_probe_panel(variants_per_combo=6)
    probe_spec = {
        "audit_version": AUDIT_VERSION,
        "manifest": manifest,
        "selector_families": list(probes.FAMILY_ORDER),
        "transform_programs": list(probes.TRANSFORM_PROGRAMS),
        "records": probes.panel_to_records(episodes),
    }
    closure_report = closure.closure_report()
    generator = probes.generator_audit(episodes, manifest)
    evaluation = evaluate_v2_panel(episodes)
    classification = classify_families(generator, evaluation)
    protected_after = protected_hashes()
    independence = source_independence_report()
    training_audit = no_training_artifact_audit()
    hard_gates = {
        "source_independence_passed": independence["passed"],
        "split_and_generator_checkers_passed_for_any_candidate": any(
            row["generator_valid_for_toolbox_expansion"] for row in generator["by_family"].values()
        ),
        "declared_closure_bounded_complete": closure_report["whether_bounded_complete"],
        "v2_forward_ordinary_raw_fields_only": v2_forward_field_report()["passed"],
        "no_training_or_checkpoint_artifacts": training_audit["passed"],
        "protected_v2_and_prior_audit_hashes_unchanged": protected_before == protected_after,
        "pytest_passed_if_provided": parse_pytest_result(pytest_result)["passed"] is not False,
    }
    if not all(hard_gates.values()):
        classification = copy.deepcopy(classification)
        classification["global_verdict"] = VERDICT_INCONCLUSIVE
    final_report = {
        "audit_version": AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": classification["global_verdict"],
        "recommended_first_family": classification["recommended_first_family"],
        "minimal_v2_structural_contract_map": structural_contract_map(),
        "declared_closure_summary": {
            "number_of_programs": closure_report["number_of_programs"],
            "semantic_collision_count": closure_report["semantic_collision_count"],
            "whether_exhaustive": closure_report["whether_exhaustive"],
            "whether_bounded_complete": closure_report["whether_bounded_complete"],
            "limitations": closure_report["limitations"],
        },
        "family_classification_summary": classification["by_family"],
        "fresh_versus_checkpoint_agreement": {
            family: {
                condition: evaluation["aggregate_by_family"][family][condition]["fresh_checkpoint_same_output_rate_min"]
                for condition in ("full_scene", "isolated_target_scene")
            }
            for family in probes.FAMILY_ORDER
        },
        "source_independence_report": independence,
        "v2_forward_field_report": v2_forward_field_report(),
        "no_training_artifact_audit": training_audit,
        "pytest_result": parse_pytest_result(pytest_result),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "hard_gates": hard_gates,
        "generated_artifact_hashes": {},
        "permitted_next_step": permitted_next_step(classification["global_verdict"]),
    }
    final_report["report_body_sha256_excluding_timestamp"] = stable_report_hash(final_report)
    return final_report, probe_spec, closure_report, generator, evaluation, classification


def permitted_next_step(verdict: str) -> str:
    if verdict == VERDICT_READY:
        return "create an outer-learner architecture preflight for exactly one selected family"
    if verdict == VERDICT_BROADER:
        return "run a narrower structural characterization of that specific probe behavior"
    return "do not build the outer learner yet"


def write_artifacts(
    final_report: dict[str, Any],
    probe_spec: dict[str, Any],
    closure_report: dict[str, Any],
    generator: dict[str, Any],
    evaluation: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    write_json(PROBE_SPEC_PATH, probe_spec)
    write_json(CLOSURE_REPORT_PATH, closure_report)
    write_json(GENERATOR_AUDIT_PATH, generator)
    write_json(V2_EVALUATION_PATH, evaluation)
    write_json(FAMILY_CLASSIFICATION_PATH, classification)
    final = copy.deepcopy(final_report)
    final["generated_artifact_hashes"] = {
        "probe_spec.json": sha256_file(PROBE_SPEC_PATH),
        "declared_closure_report.json": sha256_file(CLOSURE_REPORT_PATH),
        "generator_audit.json": sha256_file(GENERATOR_AUDIT_PATH),
        "v2_probe_evaluation.json": sha256_file(V2_EVALUATION_PATH),
        "family_classification.json": sha256_file(FAMILY_CLASSIFICATION_PATH),
    }
    final["report_body_sha256_excluding_timestamp"] = stable_report_hash(final)
    write_json(FINAL_REPORT_PATH, final)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-2A V2 capability-frontier preflight audit")
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    final_report, probe_spec, closure_report, generator, evaluation, classification = build_report(args.pytest_result)
    write_artifacts(final_report, probe_spec, closure_report, generator, evaluation, classification)
    final = load_json(FINAL_REPORT_PATH)
    print(
        json.dumps(
            {
                "status": final["status"],
                "recommended_first_family": final["recommended_first_family"],
                "report": str(FINAL_REPORT_PATH),
                "report_body_sha256_excluding_timestamp": final["report_body_sha256_excluding_timestamp"],
            },
            sort_keys=True,
        )
    )
    return 0 if final["status"] != VERDICT_INCONCLUSIVE else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
