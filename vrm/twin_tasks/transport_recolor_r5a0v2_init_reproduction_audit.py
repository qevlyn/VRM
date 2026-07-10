from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import random
import re
import sys
import tokenize
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from .transport_recolor_r5a0v2_data import (
    R5A0V2Episode,
    all_query_types,
    episode_from_record,
    grid_to_list,
    load_jsonl,
    make_episode,
    partition_query_types,
    query_type_key,
    stable_hash,
    tensors_from_episodes,
)
from .transport_recolor_r5a0v2_model import SupportConditionedTransportRecolorR5A0V2
from .transport_recolor_r5a0v2_oracles import (
    GRID_SIZE,
    HELD_OUT_MIXED_CLASSES,
    MIXED_CLASSES,
    SEALED_CLASSES,
    SEEN_MIXED_CLASSES,
    SINGLE_CLASSES,
    TRAIN_VISIBLE_CLASSES,
    apply_sequence_to_grid,
)
from . import run_transport_recolor_r5a0v2 as runner

PROJECT_ROOT = Path("/mnt/d/VRM")
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
V2_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/transport_recolor_r5a0_v2"
REPORT_PATH = V2_ARTIFACT_ROOT / "init_reproduction_audit_report.json"
RANDOM_SUITE_PATH = V2_ARTIFACT_ROOT / "init_reproduction_random_suite.json"
PYTEST_RESULT_PATH = V2_ARTIFACT_ROOT / "init_reproduction_pytest_result.txt"

AUDIT_VERSION = "R5A0V2_INIT_REPRODUCTION_AUDIT_V1"
RANDOM_SUITE_SEED = 20260703
FRESH_INIT_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEFAULT_BATCH_SIZE = 512

GROUPS = {
    "geometry_only": tuple(item for item in ("C", "H", "V") if item in SEALED_CLASSES),
    "recolor_only": tuple(item for item in ("L", "K") if item in SEALED_CLASSES),
    "seen_mixed": SEEN_MIXED_CLASSES,
    "held_out_mixed": HELD_OUT_MIXED_CLASSES,
    "all_mixed": MIXED_CLASSES,
}

PROTECTED_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_model.py",
    PROJECT_ROOT / "vrm/twin_tasks/run_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py",
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2.md",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/dataset_manifest.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/run_config.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/aggregate_seed_summary.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/best_checkpoint_selection_seed_20260628.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/best_checkpoint_selection_seed_20260629.json",
    PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2/best_checkpoint_selection_seed_20260630.json",
)

NEW_AUDIT_SOURCE_PATHS = (
    PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_init_reproduction_audit.py",
    PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2_init_reproduction_audit.py",
    PROJECT_ROOT / "docs/TRANSPORT_RECOLOR_R5A0V2_INIT_REPRODUCTION_AUDIT.md",
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


def write_json(path: Path, payload: Any, *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
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


def source_code_sha256(paths: Iterable[Path]) -> str:
    return json_hash({str(path): sha256_file(path) for path in paths if path.exists()})


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_state_hash(model: torch.nn.Module) -> str:
    return state_dict_hash(model.state_dict())


def fresh_model_for_seed(seed: int, device: torch.device) -> SupportConditionedTransportRecolorR5A0V2:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    return model


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[SupportConditionedTransportRecolorR5A0V2, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = SupportConditionedTransportRecolorR5A0V2().to(device).eval()
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def selected_checkpoint_records() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(V2_ARTIFACT_ROOT.glob("best_checkpoint_selection_seed_*.json")):
        record = load_json(path)
        checkpoint = Path(record["checkpoint_path"])
        rows.append(
            {
                "seed": int(record["seed"]),
                "selection_metadata_path": str(path),
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "selected_epoch": int(record["selected_epoch"]),
                "selection_key": record["selection_key"],
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
            }
        )
    return rows


def existing_split_suites() -> dict[str, list[R5A0V2Episode]]:
    suites = {
        "suite_a_train": load_jsonl(V2_ARTIFACT_ROOT / "train.jsonl"),
        "suite_a_dev": load_jsonl(V2_ARTIFACT_ROOT / "dev.jsonl"),
        "suite_a_sealed": load_jsonl(V2_ARTIFACT_ROOT / "sealed.jsonl"),
    }
    suites["suite_a_all_existing_splits"] = suites["suite_a_train"] + suites["suite_a_dev"] + suites["suite_a_sealed"]
    return suites


def build_random_suite_records(seed: int = RANDOM_SUITE_SEED, variants_per_class_query: int = 2) -> dict[str, Any]:
    qtypes = list(all_query_types())
    random.Random(seed).shuffle(qtypes)
    records = []
    for operation_class in SEALED_CLASSES:
        for qtype in qtypes:
            for variant in range(variants_per_class_query):
                local_index = seed + variant
                episode = make_episode("init_reproduction_random_suite", qtype, operation_class, local_index)
                records.append(episode.to_record() | {"audit_variant": variant})
    class_counts: dict[str, int] = defaultdict(int)
    anchors: set[str] = set()
    colors: set[int] = set()
    distractors: set[int] = set()
    shapes: set[str] = set()
    for record in records:
        class_counts[str(record["operation_class"])] += 1
        query_type = record["query_type"]
        anchors.add(str(query_type["anchor_region"]))
        colors.add(int(query_type["primary_color"]))
        distractors.add(int(query_type["distractor_count"]))
        shapes.add(str(query_type["primary_shape"]))
    manifest = {
        "suite_name": "suite_b_random_generated",
        "suite_seed": seed,
        "generator": "transport_recolor_r5a0v2_data.make_episode over shuffled all_query_types",
        "generator_source_hash": source_code_sha256((PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_data.py", PROJECT_ROOT / "vrm/twin_tasks/transport_recolor_r5a0v2_oracles.py")),
        "operation_classes": list(SEALED_CLASSES),
        "variants_per_class_query": variants_per_class_query,
        "record_count": len(records),
        "class_balance": dict(sorted(class_counts.items())),
        "coverage": {
            "query_type_count": len(qtypes),
            "primary_shapes": sorted(shapes),
            "primary_colors": sorted(colors),
            "anchor_regions": sorted(anchors),
            "distractor_counts": sorted(distractors),
            "coordinate_predicate_branches": ["x<=3", "x>=3", "y<=3", "y>=3", "x<=3_and_y<=3", "x<=3_and_y>=3", "x>=3_and_y<=3", "x>=3_and_y>=3"],
        },
    }
    suite_hash = json_hash({"manifest": manifest, "records": records})
    return {"manifest": manifest | {"suite_hash": suite_hash}, "records": records}


def random_suite_episodes(suite_payload: dict[str, Any]) -> list[R5A0V2Episode]:
    return [episode_from_record(record) for record in suite_payload["records"]]


def sanity_fixture_episodes() -> dict[str, list[R5A0V2Episode]]:
    qtype = partition_query_types()["train"][0]
    return {
        "suite_c_sanity_final_output_c_to_k": [make_episode("unit", qtype, "C->K", 0)],
        "suite_c_sanity_reported_metrics_k_to_c": [make_episode("unit", qtype, "K->C", 0)],
        "suite_c_sanity_intervention_c_to_l": [make_episode("unit", qtype, "C->L", 0)],
        "suite_c_sanity_recombination_contexts": [
            make_episode("unit", qtype, "C", 0),
            make_episode("unit", qtype, "L->H", 0),
        ],
    }


def macro_exact_for_classes(exact_by_class: dict[str, float], classes: Iterable[str]) -> float | None:
    values = [exact_by_class[item] for item in classes if item in exact_by_class]
    if not values:
        return None
    return float(sum(values) / len(values))


def condition_label(condition: str, init_seed: int | None, checkpoint_path: str | None) -> str:
    if checkpoint_path:
        return f"{condition}:seed={init_seed}:checkpoint={Path(checkpoint_path).name}"
    return f"{condition}:seed={init_seed}"


def evaluate_suite(
    model: torch.nn.Module,
    episodes: list[R5A0V2Episode],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0})
    digest = hashlib.sha256()
    per_example = []
    first_failure = None
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            decoded = out["decoded_grid"].detach().cpu().long()
            targets = batch["target_grids"].detach().cpu().long()
            for index, episode in enumerate(chunk):
                pred = decoded[index]
                target = targets[index]
                exact = bool(torch.equal(pred, target))
                row = by_class[episode.operation_class]
                row["count"] += 1
                row["exact"] += int(exact)
                pred_list = pred.tolist()
                pred_hash = stable_hash(pred_list)
                target_hash = stable_hash(target.tolist())
                digest.update(episode.episode_id.encode("utf-8"))
                digest.update(episode.operation_class.encode("utf-8"))
                digest.update(pred_hash.encode("utf-8"))
                per_example.append(
                    {
                        "episode_id": episode.episode_id,
                        "operation_class": episode.operation_class,
                        "prediction_hash": pred_hash,
                        "target_hash": target_hash,
                        "exact": exact,
                    }
                )
                if not exact and first_failure is None:
                    first_failure = {
                        "episode_id": episode.episode_id,
                        "operation_class": episode.operation_class,
                        "prediction_grid": pred_list,
                        "target_grid": target.tolist(),
                    }
    exact_by_class = {key: row["exact"] / max(row["count"], 1) for key, row in sorted(by_class.items())}
    exact_by_group = {key: macro_exact_for_classes(exact_by_class, classes) for key, classes in GROUPS.items()}
    macro_exact = macro_exact_for_classes(exact_by_class, exact_by_class.keys())
    return {
        "record_count": len(episodes),
        "exact_by_group": exact_by_group,
        "exact_by_class": exact_by_class,
        "class_counts": {key: row["count"] for key, row in sorted(by_class.items())},
        "macro_exact": macro_exact,
        "heldout_mixed_exact": exact_by_group["held_out_mixed"],
        "prediction_hash": digest.hexdigest(),
        "first_failure_if_any": first_failure,
        "per_example": per_example,
    }


def compact_suite_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "per_example"}


def run_condition(
    condition: str,
    model: torch.nn.Module,
    model_hash: str,
    init_seed: int | None,
    checkpoint_path: str | None,
    suites: dict[str, list[R5A0V2Episode]],
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    full_results = {}
    for suite_name, episodes in suites.items():
        result = evaluate_suite(model, episodes, device, batch_size)
        label = condition_label(condition, init_seed, checkpoint_path)
        rows.append(
            {
                "condition": condition,
                "condition_label": label,
                "suite_name": suite_name,
                "init_seed": init_seed,
                "checkpoint_path_or_null": checkpoint_path,
                "model_state_hash_before_eval": model_hash,
                "number_of_optimizer_steps_this_process": 0,
                **compact_suite_result(result),
            }
        )
        full_results[f"{label}|{suite_name}"] = result
    return rows, full_results


def compare_results(checkpoint_row: dict[str, Any], fresh_row: dict[str, Any], checkpoint_full: dict[str, Any], fresh_full: dict[str, Any]) -> dict[str, Any]:
    total = min(len(checkpoint_full["per_example"]), len(fresh_full["per_example"]))
    same = 0
    first_divergence = None
    for left, right in zip(checkpoint_full["per_example"], fresh_full["per_example"]):
        is_same = left["prediction_hash"] == right["prediction_hash"]
        same += int(is_same)
        if not is_same and first_divergence is None:
            first_divergence = {
                "episode_id": left["episode_id"],
                "operation_class": left["operation_class"],
                "checkpoint_prediction_hash": left["prediction_hash"],
                "fresh_prediction_hash": right["prediction_hash"],
                "target_hash": left["target_hash"],
            }
    classes = sorted(set(checkpoint_row["exact_by_class"]) | set(fresh_row["exact_by_class"]))
    return {
        "checkpoint_condition_label": checkpoint_row["condition_label"],
        "fresh_condition_label": fresh_row["condition_label"],
        "suite_name": checkpoint_row["suite_name"],
        "same_output_rate": same / max(total, 1),
        "grid_difference_rate": 1.0 - (same / max(total, 1)),
        "macro_exact_delta": none_safe_delta(checkpoint_row["macro_exact"], fresh_row["macro_exact"]),
        "heldout_mixed_exact_delta": none_safe_delta(checkpoint_row["heldout_mixed_exact"], fresh_row["heldout_mixed_exact"]),
        "per_class_exact_delta": {
            cls: none_safe_delta(checkpoint_row["exact_by_class"].get(cls), fresh_row["exact_by_class"].get(cls))
            for cls in classes
        },
        "first_divergence_example_if_any": first_divergence,
    }


def none_safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def forward_input_provenance() -> list[dict[str, Any]]:
    return [
        {
            "field_name": "query_grids",
            "source_location": "tensors_from_episodes(...)[\"query_grids\"] -> model positional arg 1",
            "shape_or_type": "[batch, 7, 7] int tensor",
            "semantic_description": "raw query input grid",
            "is_raw_grid_or_mask_or_position": True,
            "is_target_oracle_metadata": False,
            "is_operation_program_class_metadata": False,
            "used_in_model_forward": True,
        },
        {
            "field_name": "support_input_grids",
            "source_location": "tensors_from_episodes(...)[\"support_input_grids\"] -> model positional arg 2",
            "shape_or_type": "[batch, 3, 7, 7] int tensor",
            "semantic_description": "raw support input grids",
            "is_raw_grid_or_mask_or_position": True,
            "is_target_oracle_metadata": False,
            "is_operation_program_class_metadata": False,
            "used_in_model_forward": True,
        },
        {
            "field_name": "support_output_grids",
            "source_location": "tensors_from_episodes(...)[\"support_output_grids\"] -> model positional arg 3",
            "shape_or_type": "[batch, 3, 7, 7] int tensor",
            "semantic_description": "raw support output grids",
            "is_raw_grid_or_mask_or_position": True,
            "is_target_oracle_metadata": False,
            "is_operation_program_class_metadata": False,
            "used_in_model_forward": True,
        },
        {
            "field_name": "target_grids",
            "source_location": "tensors_from_episodes(...)[\"target_grids\"]",
            "shape_or_type": "[batch, 7, 7] int tensor",
            "semantic_description": "label grid used only after forward for exact metric",
            "is_raw_grid_or_mask_or_position": False,
            "is_target_oracle_metadata": True,
            "is_operation_program_class_metadata": False,
            "used_in_model_forward": False,
        },
        {
            "field_name": "operation_class",
            "source_location": "R5A0V2Episode.operation_class",
            "shape_or_type": "string metadata",
            "semantic_description": "class label used only for metric grouping and suite construction",
            "is_raw_grid_or_mask_or_position": False,
            "is_target_oracle_metadata": False,
            "is_operation_program_class_metadata": True,
            "used_in_model_forward": False,
        },
    ]


def provenance_invalid(provenance: list[dict[str, Any]]) -> bool:
    for row in provenance:
        if row["used_in_model_forward"] and (row["is_target_oracle_metadata"] or row["is_operation_program_class_metadata"]):
            return True
    return False


def model_forward_metadata_report() -> dict[str, Any]:
    signature = inspect.signature(SupportConditionedTransportRecolorR5A0V2.forward)
    source = inspect.getsource(SupportConditionedTransportRecolorR5A0V2.forward)
    forbidden_forward_terms = [
        term
        for term in ("operation_class", "sequence_class", "target_grids", "oracle", "geometry_id_target", "predicate_id_target")
        if term in source
    ]
    return {
        "forward_signature": str(signature),
        "forbidden_metadata_terms_in_forward_source": forbidden_forward_terms,
        "forward_accepts_operation_program_class_or_target_metadata": bool(forbidden_forward_terms),
    }


def original_alleged_result_source() -> dict[str, Any]:
    test_path = PROJECT_ROOT / "tests/test_transport_recolor_r5a0v2.py"
    source = test_path.read_text(encoding="utf-8")
    fixtures = []
    for name in (
        "test_final_output_probabilities_equal_sum_v_a_times_b",
        "test_reported_metrics_match_hand_built_oracle_fixture",
        "test_interventions_alter_only_requested_factor_shapes",
        "test_recombined_output_is_computed_exactly_from_selected_a_and_b",
    ):
        if name in source:
            fixtures.append(name)
    return {
        "assessment": (
            "The visible pre-existing fresh-init success claim traces to hand-written pytest sanity fixtures, especially "
            "tests/test_transport_recolor_r5a0v2.py::test_reported_metrics_match_hand_built_oracle_fixture, "
            "which constructs one generated unit episode with make_episode and evaluates a freshly instantiated model without loading a checkpoint."
        ),
        "source_type": "hand-written pytest sanity fixture using generated unit episode",
        "not_train_dev_sealed_benchmark": True,
        "not_synthetic_oracle_check_only": False,
        "checkpoint_could_be_loaded_indirectly": False,
        "fixture_names_found": fixtures,
        "source_file": str(test_path),
    }


def parse_pytest_result(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"provided": path is not None, "passed": False, "path": str(path) if path else None, "summary": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    passed = bool(re.search(r"\b\d+\s+passed\b", lowered)) and not any(term in lowered for term in (" failed", " error", " interrupted"))
    return {
        "provided": True,
        "passed": passed,
        "path": str(path),
        "summary": text.strip().splitlines()[-1] if text.strip() else "",
        "sha256": sha256_file(path),
    }


def no_training_artifact_audit() -> dict[str, Any]:
    pieces = []
    for path in NEW_AUDIT_SOURCE_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces.append(source_without_strings_and_comments(text) if path.suffix == ".py" else text)
    executable_text = "\n".join(pieces)
    forbidden_terms = [term for term in ("torch.optim", ".backward(", "loss.backward", "optimizer.step(", "save_checkpoint(") if term in executable_text]
    checkpoint_writes = [
        str(path)
        for path in V2_ARTIFACT_ROOT.glob("init_reproduction*")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}
    ]
    return {
        "optimizer_backward_or_checkpoint_write_terms_found": forbidden_terms,
        "checkpoint_or_weight_files_created": checkpoint_writes,
        "passed": forbidden_terms == [] and checkpoint_writes == [],
    }


def fresh_path_structural_report() -> dict[str, Any]:
    source = inspect.getsource(fresh_model_for_seed)
    forbidden = [term for term in ("torch.load", "load_state_dict", "checkpoint", "optimizer", "backward") if term in source]
    return {
        "fresh_model_function": "fresh_model_for_seed",
        "forbidden_checkpoint_or_training_terms": forbidden,
        "fresh_init_path_can_call_checkpoint_loading": bool(forbidden),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def build_suites(write_random_suite: bool = False) -> tuple[dict[str, list[R5A0V2Episode]], dict[str, Any]]:
    random_payload = build_random_suite_records()
    if write_random_suite:
        write_json(RANDOM_SUITE_PATH, random_payload)
    suites = {}
    suites.update(existing_split_suites())
    suites["suite_b_random_generated"] = random_suite_episodes(random_payload)
    suites.update(sanity_fixture_episodes())
    return suites, random_payload["manifest"]


def summarize_fresh_seed_similarity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fresh_rows = [row for row in rows if row["condition"] == "fresh_init"]
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fresh_rows:
        by_suite[row["suite_name"]].append(row)
    suite_summaries = {}
    anomalous = []
    for suite, suite_rows in sorted(by_suite.items()):
        macro_values = sorted({row["macro_exact"] for row in suite_rows})
        heldout_values = sorted({row["heldout_mixed_exact"] for row in suite_rows})
        prediction_hashes = sorted({row["prediction_hash"] for row in suite_rows})
        same = len(macro_values) <= 1 and len(heldout_values) <= 1 and len(prediction_hashes) <= 1
        suite_summaries[suite] = {
            "macro_exact_values": macro_values,
            "heldout_mixed_exact_values": heldout_values,
            "distinct_prediction_hash_count": len(prediction_hashes),
            "all_fresh_seeds_same_on_suite": same,
        }
        if not same:
            anomalous.append(suite)
    return {
        "all_five_fresh_seeds_behave_similarly": anomalous == [],
        "anomalous_suites": anomalous,
        "by_suite": suite_summaries,
    }


def determine_verdict(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    provenance_bad: bool,
    hard_gates: dict[str, bool],
) -> tuple[str, list[str]]:
    if provenance_bad:
        return "R5A0V2_INIT_REPRODUCTION_PROVENANCE_INVALID", ["target/oracle/class metadata reaches model forward"]
    if not all(hard_gates.values()):
        return "R5A0V2_INIT_REPRODUCTION_INCONCLUSIVE", [name for name, passed in hard_gates.items() if not passed]
    fresh_rows = [row for row in rows if row["condition"] == "fresh_init"]
    checkpoint_rows = [row for row in rows if row["condition"] == "selected_checkpoint"]
    suite_b_fresh = [row for row in fresh_rows if row["suite_name"] == "suite_b_random_generated"]
    suite_b_checkpoint = [row for row in checkpoint_rows if row["suite_name"] == "suite_b_random_generated"]
    heldout_fresh = [row for row in fresh_rows if row["suite_name"] == "suite_a_sealed"]
    heldout_checkpoint = [row for row in checkpoint_rows if row["suite_name"] == "suite_a_sealed"]
    fresh_equals = (
        suite_b_fresh
        and suite_b_checkpoint
        and heldout_fresh
        and heldout_checkpoint
        and min(row["macro_exact"] or 0.0 for row in suite_b_fresh) >= 0.999
        and min(row["macro_exact"] or 0.0 for row in suite_b_checkpoint) >= 0.999
        and min(row["heldout_mixed_exact"] or 0.0 for row in heldout_fresh) >= 0.999
        and min(row["heldout_mixed_exact"] or 0.0 for row in heldout_checkpoint) >= 0.999
        and all(item["same_output_rate"] >= 0.999 for item in comparisons)
    )
    if fresh_equals:
        return "R5A0V2_INIT_REPRODUCTION_FRESH_EQUALS_TRAINED", ["fresh initialization matches selected checkpoints across evaluated suites"]
    fixture_success = any(row["condition"] == "fresh_init" and row["suite_name"].startswith("suite_c_") and (row["macro_exact"] or 0.0) >= 0.999 for row in rows)
    split_fail = any(row["condition"] == "fresh_init" and row["suite_name"] in {"suite_a_sealed", "suite_b_random_generated"} and (row["macro_exact"] or 0.0) < 0.99 for row in rows)
    if fixture_success and split_fail:
        return "R5A0V2_INIT_REPRODUCTION_FIXTURE_ONLY", ["fresh initialization succeeds on sanity fixtures but not broad suites"]
    trained_succeeds = suite_b_checkpoint and min(row["macro_exact"] or 0.0 for row in suite_b_checkpoint) >= 0.999
    fresh_fails = suite_b_fresh and max(row["macro_exact"] or 0.0 for row in suite_b_fresh) < 0.99
    if trained_succeeds and fresh_fails:
        return "R5A0V2_INIT_REPRODUCTION_FRESH_FAILS_TRAINED_SUCCEEDS", ["fresh underperforms selected checkpoints on broad suites"]
    return "R5A0V2_INIT_REPRODUCTION_INCONCLUSIVE", ["result did not match a decisive interpretation rule"]


def build_report(pytest_result: Path | None = None, *, write_random_suite: bool = False) -> dict[str, Any]:
    protected_before = protected_hashes()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suites, random_manifest = build_suites(write_random_suite=write_random_suite)
    rows: list[dict[str, Any]] = []
    full_results: dict[str, dict[str, Any]] = {}
    checkpoint_records = selected_checkpoint_records()

    for seed in FRESH_INIT_SEEDS:
        model = fresh_model_for_seed(seed, device)
        model_hash = model_state_hash(model)
        condition_rows, condition_full = run_condition("fresh_init", model, model_hash, seed, None, suites, device, DEFAULT_BATCH_SIZE)
        rows.extend(condition_rows)
        full_results.update(condition_full)

    for record in checkpoint_records:
        if not record["checkpoint_exists"]:
            continue
        model, payload = load_checkpoint_model(Path(record["checkpoint_path"]), device)
        model_hash = model_state_hash(model)
        condition_rows, condition_full = run_condition(
            "selected_checkpoint",
            model,
            model_hash,
            int(payload["seed"]),
            record["checkpoint_path"],
            suites,
            device,
            DEFAULT_BATCH_SIZE,
        )
        rows.extend(condition_rows)
        full_results.update(condition_full)

    for record in checkpoint_records:
        if not record["checkpoint_exists"]:
            continue
        seed = int(record["seed"])
        model = fresh_model_for_seed(seed, device)
        model_hash = model_state_hash(model)
        condition_rows, condition_full = run_condition("fresh_same_seed_as_checkpoint", model, model_hash, seed, None, suites, device, DEFAULT_BATCH_SIZE)
        rows.extend(condition_rows)
        full_results.update(condition_full)

    row_by_label_suite = {(row["condition_label"], row["suite_name"]): row for row in rows}
    comparisons = []
    fresh_labels = sorted({row["condition_label"] for row in rows if row["condition"] in {"fresh_init", "fresh_same_seed_as_checkpoint"}})
    checkpoint_labels = sorted({row["condition_label"] for row in rows if row["condition"] == "selected_checkpoint"})
    for checkpoint_label in checkpoint_labels:
        for fresh_label in fresh_labels:
            for suite_name in suites:
                checkpoint_key = (checkpoint_label, suite_name)
                fresh_key = (fresh_label, suite_name)
                if checkpoint_key not in row_by_label_suite or fresh_key not in row_by_label_suite:
                    continue
                comparisons.append(
                    compare_results(
                        row_by_label_suite[checkpoint_key],
                        row_by_label_suite[fresh_key],
                        full_results[f"{checkpoint_label}|{suite_name}"],
                        full_results[f"{fresh_label}|{suite_name}"],
                    )
                )
    provenance = forward_input_provenance()
    forward_report = model_forward_metadata_report()
    provenance_bad = provenance_invalid(provenance) or forward_report["forward_accepts_operation_program_class_or_target_metadata"]
    fresh_structural = fresh_path_structural_report()
    artifact_audit = no_training_artifact_audit()
    protected_after = protected_hashes()
    checkpoint_fresh_state_differences = []
    for record in checkpoint_records:
        if not record["checkpoint_exists"]:
            continue
        checkpoint_payload = torch.load(record["checkpoint_path"], map_location="cpu")
        checkpoint_state_hash = state_dict_hash(checkpoint_payload["model_state_dict"])
        fresh = fresh_model_for_seed(int(record["seed"]), torch.device("cpu"))
        fresh_hash = model_state_hash(fresh)
        checkpoint_fresh_state_differences.append(
            {
                "seed": int(record["seed"]),
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_state_hash": checkpoint_state_hash,
                "fresh_same_seed_state_hash": fresh_hash,
                "checkpoint_differs_from_fresh_same_seed": checkpoint_state_hash != fresh_hash,
            }
        )
    fresh_state_hashes = sorted({row["model_state_hash_before_eval"] for row in rows if row["condition"] == "fresh_init" and row["suite_name"] == "suite_b_random_generated"})
    hard_gates = {
        "cuda_or_cpu_evaluation_device_established": device.type in {"cuda", "cpu"},
        "fresh_path_cannot_call_checkpoint_loading": not fresh_structural["fresh_init_path_can_call_checkpoint_loading"],
        "fresh_models_have_distinct_state_hashes": len(fresh_state_hashes) == len(FRESH_INIT_SEEDS),
        "selected_checkpoints_differ_from_fresh_same_seed_when_present": all(row["checkpoint_differs_from_fresh_same_seed"] for row in checkpoint_fresh_state_differences) if checkpoint_fresh_state_differences else True,
        "no_optimizer_backward_or_checkpoint_write": artifact_audit["passed"],
        "protected_v2_hashes_unchanged_during_audit": protected_before == protected_after,
        "provenance_valid": not provenance_bad,
        "evaluation_uses_final_grid_exact_metric": True,
    }
    verdict, verdict_reasons = determine_verdict(rows, comparisons, provenance_bad, hard_gates)
    report = {
        "audit_version": AUDIT_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": verdict,
        "verdict_reasons": verdict_reasons,
        "device": {"type": device.type, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "original_alleged_fresh_init_result_source": original_alleged_result_source(),
        "checkpoint_indirect_load_assessment": {
            "fresh_init_path_can_call_checkpoint_loading": fresh_structural["fresh_init_path_can_call_checkpoint_loading"],
            "fresh_path_structural_report": fresh_structural,
        },
        "forward_input_provenance": provenance,
        "model_forward_metadata_report": forward_report,
        "same_final_grid_exact_metric_as_published_v2": True,
        "suite_manifests": {
            "suite_a_existing_splits": {
                "train_count": len(suites["suite_a_train"]),
                "dev_count": len(suites["suite_a_dev"]),
                "sealed_count": len(suites["suite_a_sealed"]),
                "all_existing_split_count": len(suites["suite_a_all_existing_splits"]),
                "dataset_manifest_sha256": sha256_file(V2_ARTIFACT_ROOT / "dataset_manifest.json"),
            },
            "suite_b_random_generated": random_manifest,
            "suite_c_existing_sanity_fixtures": {
                key: {"record_count": len(value), "episode_ids": [episode.episode_id for episode in value]}
                for key, value in suites.items()
                if key.startswith("suite_c_")
            },
        },
        "model_conditions": {
            "fresh_init_seeds": list(FRESH_INIT_SEEDS),
            "selected_checkpoints": checkpoint_records,
            "fresh_same_seed_as_checkpoint": [row["seed"] for row in checkpoint_records if row["checkpoint_exists"]],
        },
        "run_results": rows,
        "checkpoint_vs_fresh": comparisons,
        "fresh_seed_similarity": summarize_fresh_seed_similarity(rows),
        "checkpoint_vs_fresh_state_differences": checkpoint_fresh_state_differences,
        "artifact_audit": artifact_audit,
        "pytest_result": parse_pytest_result(pytest_result),
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "hard_gates": hard_gates,
        "next_permitted_step": (
            "full R5A-0 learning-dependency/no-oracle audit"
            if verdict == "R5A0V2_INIT_REPRODUCTION_FRESH_EQUALS_TRAINED"
            else "R5A-1A.2 accounting sanity"
            if verdict == "R5A0V2_INIT_REPRODUCTION_FRESH_FAILS_TRAINED_SUCCEEDS"
            else "pause and resolve L0 audit status before continuing"
        ),
    }
    report["report_body_sha256_excluding_timestamp"] = stable_report_hash(report)
    return report


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    write_json(path, report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R5A-0 V2 L0 initialization reproduction audit")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--random-suite-out", type=Path, default=RANDOM_SUITE_PATH)
    parser.add_argument("--pytest-result", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.pytest_result, write_random_suite=False)
    suites_payload = build_random_suite_records()
    write_json(args.random_suite_out, suites_payload)
    write_report(report, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(args.out),
                "random_suite_out": str(args.random_suite_out),
                "report_body_sha256_excluding_timestamp": report["report_body_sha256_excluding_timestamp"],
                "random_suite_hash": suites_payload["manifest"]["suite_hash"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] not in {"R5A0V2_INIT_REPRODUCTION_INCONCLUSIVE", "R5A0V2_INIT_REPRODUCTION_PROVENANCE_INVALID"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
