from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .composition_r4a0r_data import R4A0REpisode, grid_to_list, query_type_key, stable_hash, tensors_from_episodes
from .composition_r4a1p_data import (
    REQUIRED_COUNTERFACTUAL_GROUPS,
    SEALED_STRATUM,
    TRAIN_VISIBLE_STRATUM,
    VISUAL_STRATA,
    build_length3_counterfactual_suite,
    build_length3_function_catalog,
    build_length3_probe_dataset,
    default_probe_query_types,
    derange_episode,
    load_jsonl,
    records_by_group,
    reorder_episode,
    support_input_bag,
    support_output_bag,
    support_signature,
    write_probe_artifacts,
)
from .composition_r4a1p_forensics import (
    APPROVED_CHECKPOINTS,
    PROJECT_ROOT,
    approved_loaded_checkpoint_audit,
    checkpoint_provenance_before,
    cuda_environment_report,
    epoch_zero_semantics_audit,
    file_record,
    forward_provenance_audit,
    hash_status,
    independent_pointer_coordinate_recomputation,
    load_frozen_model,
    parameter_hashes,
    pointer_metric_semantics_audit,
    protected_hashes,
    save_json,
    save_text,
    snapshot_checkpoint_directory,
)
from .composition_r4a1p_oracles import affine_payload, compact_word, source_index_for_output_exact

DEFAULT_OUT = PROJECT_ROOT / "artifacts/composition_r4a1p_v1"
DEFAULT_SEEDS = (20260628, 20260629, 20260630)
VERDICT_PASS = "R4A1P_FROZEN_LENGTH3_PASS"
VERDICT_LENGTH_FAIL = "R4A1P_LENGTH3_GENERALIZATION_FAILURE"
VERDICT_VISUAL_FAIL = "R4A1P_VISUAL_OOD_ONLY_FAILURE"
VERDICT_SUPPORT_FAIL = "R4A1P_SUPPORT_NOT_USED"
VERDICT_ORDER_FAIL = "R4A1P_ORDER_COUNTERFACTUAL_FAILURE"
VERDICT_PROVENANCE_FAIL = "R4A1P_PROVENANCE_AUDIT_FAILURE"
VERDICT_CATALOG_INSUFFICIENT = "R4A1P_PROBE_CATALOG_INSUFFICIENT"
VERDICT_QUERY_INSUFFICIENT = "R4A1P_PROBE_QUERY_TYPE_INSUFFICIENT"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def metric_template() -> dict[str, Any]:
    return {
        "count": 0,
        "exact": 0,
        "fg_correct": 0,
        "fg_total": 0,
        "all_correct": 0,
        "all_total": 0,
        "hamming": 0.0,
        "pointer_correct": 0,
        "pointer_total": 0,
        "pointer_mass": 0.0,
        "same_color_wrong": 0,
        "pointer_entropy": [],
        "pointer_max": [],
        "context_norm": [],
        "context_mean": [],
        "context_std": [],
    }


def finalize_metric(row: dict[str, Any]) -> dict[str, Any]:
    count = max(row["count"], 1)
    pointer_total = max(row["pointer_total"], 1)
    return {
        "episode_count": row["count"],
        "direct_grid_exact_accuracy": row["exact"] / count,
        "foreground_accuracy": row["fg_correct"] / max(row["fg_total"], 1),
        "all_cell_accuracy": row["all_correct"] / max(row["all_total"], 1),
        "hamming_error": row["hamming"] / count,
        "foreground_exact_argmax_source_coordinate_accuracy": row["pointer_correct"] / pointer_total,
        "foreground_mean_pointer_mass_on_unique_oracle_coordinate": row["pointer_mass"] / pointer_total,
        "foreground_same_color_but_wrong_coordinate_rate": row["same_color_wrong"] / pointer_total,
        "mean_pointer_entropy": statistics.fmean(row["pointer_entropy"]) if row["pointer_entropy"] else 0.0,
        "mean_max_pointer_probability": statistics.fmean(row["pointer_max"]) if row["pointer_max"] else 0.0,
        "task_context_norm": statistics.fmean(row["context_norm"]) if row["context_norm"] else 0.0,
        "task_context_mean": statistics.fmean(row["context_mean"]) if row["context_mean"] else 0.0,
        "task_context_std": statistics.fmean(row["context_std"]) if row["context_std"] else 0.0,
    }


def exact_mask(decoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (decoded == target).flatten(1).all(dim=1)


def _row_key(episode: R4A0REpisode) -> tuple[str, str]:
    return episode.split, compact_word(episode.sequence_class)


def _update_metric(
    row: dict[str, Any],
    episode: R4A0REpisode,
    decoded: torch.Tensor,
    target: torch.Tensor,
    query: torch.Tensor,
    pointer: torch.Tensor,
    pointer_entropy: torch.Tensor,
    pointer_max: torch.Tensor,
    context: torch.Tensor,
    index: int,
    exact: bool,
) -> None:
    fg = target[index] != 0
    row["count"] += 1
    row["exact"] += int(exact)
    row["fg_total"] += int(fg.sum().item())
    row["fg_correct"] += int(((decoded[index] == target[index]) & fg).sum().item())
    row["all_total"] += 49
    row["all_correct"] += int((decoded[index] == target[index]).sum().item())
    row["hamming"] += float((decoded[index] != target[index]).sum().item())
    pointer_argmax = pointer[index].argmax(dim=-1)
    for y in range(7):
        for x in range(7):
            if not bool(fg[y, x].item()):
                continue
            out_index = y * 7 + x
            oracle_index, _, _ = source_index_for_output_exact(episode.sequence_class, x, y)
            if oracle_index < 0:
                continue
            predicted_index = int(pointer_argmax[out_index].item())
            row["pointer_total"] += 1
            row["pointer_correct"] += int(predicted_index == oracle_index)
            row["pointer_mass"] += float(pointer[index, out_index, oracle_index].item())
            if predicted_index != oracle_index:
                pred_y, pred_x = divmod(predicted_index, 7)
                if int(query[index, pred_y, pred_x].item()) == int(target[index, y, x].item()):
                    row["same_color_wrong"] += 1
    row["pointer_entropy"].append(float(pointer_entropy[index].mean().item()))
    row["pointer_max"].append(float(pointer_max[index].mean().item()))
    row["context_norm"].append(float(context[index].norm().item()))
    row["context_mean"].append(float(context[index].mean().item()))
    row["context_std"].append(float(context[index].std(unbiased=False).item()))


def evaluate_probe_model(
    model: torch.nn.Module,
    episodes: list[R4A0REpisode],
    device: torch.device,
    batch_size: int,
    *,
    max_records: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = episodes[:max_records] if max_records is not None else episodes
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(metric_template)
    rows_by_t: dict[tuple[str, int], dict[str, Any]] = defaultdict(metric_template)
    rows_by_qtype: dict[tuple[str, str], dict[str, Any]] = defaultdict(metric_template)
    rows_by_stratum: dict[str, dict[str, Any]] = defaultdict(metric_template)
    details: list[dict[str, Any]] = []
    requires_grad_false = all(not parameter.requires_grad for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        grad_enabled_inside_inference = torch.is_grad_enabled()
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            outputs = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            decoded = outputs["decoded_grid"]
            target = batch["target_grids"]
            query = batch["query_grids"]
            exact = exact_mask(decoded, target)
            pointer = outputs["pointer_attention"].reshape(len(chunk), 49, 49)
            pointer_entropy = -(pointer.clamp_min(1e-8) * torch.log(pointer.clamp_min(1e-8))).sum(dim=-1).reshape(len(chunk), 7, 7)
            pointer_max = pointer.max(dim=-1).values.reshape(len(chunk), 7, 7)
            context = outputs["task_context"]
            for index, episode in enumerate(chunk):
                word = compact_word(episode.sequence_class)
                is_exact = bool(exact[index].item())
                for row in (
                    rows_by_key[_row_key(episode)],
                    rows_by_t[(episode.split, word.count("T"))],
                    rows_by_qtype[(episode.split, query_type_key(episode.query_type))],
                    rows_by_stratum[episode.split],
                ):
                    _update_metric(row, episode, decoded, target, query, pointer, pointer_entropy, pointer_max, context, index, is_exact)
                details.append(
                    {
                        "episode_id": episode.episode_id,
                        "visual_stratum": episode.split,
                        "canonical_word": word,
                        "query_type_key": query_type_key(episode.query_type),
                        "number_of_T_applications": word.count("T"),
                        "exact": is_exact,
                        "decoded_hash": stable_hash(decoded[index].detach().cpu().long().tolist()),
                        "target_hash": stable_hash(target[index].detach().cpu().long().tolist()),
                    }
                )
    by_function: dict[str, dict[str, Any]] = {stratum: {} for stratum in VISUAL_STRATA}
    for (stratum, word), row in sorted(rows_by_key.items()):
        by_function[stratum][word] = finalize_metric(row) | {"affine_signature": affine_payload(word)}
    by_t_count: dict[str, dict[str, Any]] = {stratum: {} for stratum in VISUAL_STRATA}
    for (stratum, t_count), row in sorted(rows_by_t.items()):
        by_t_count[stratum][str(t_count)] = finalize_metric(row)
    by_query_type: dict[str, dict[str, Any]] = {stratum: {} for stratum in VISUAL_STRATA}
    for (stratum, qkey), row in sorted(rows_by_qtype.items()):
        by_query_type[stratum][qkey] = finalize_metric(row)
    by_stratum = {stratum: finalize_metric(row) for stratum, row in rows_by_stratum.items()}
    macro_by_stratum = {
        stratum: statistics.fmean(item["direct_grid_exact_accuracy"] for item in by_function[stratum].values()) if by_function[stratum] else 0.0
        for stratum in VISUAL_STRATA
    }
    worst_qtype_by_stratum = {
        stratum: min((item["direct_grid_exact_accuracy"] for item in by_query_type[stratum].values()), default=0.0)
        for stratum in VISUAL_STRATA
    }
    return {
        "record_count": len(selected),
        "by_visual_stratum": by_stratum,
        "by_visual_stratum_and_canonical_function": by_function,
        "by_visual_stratum_and_T_count": by_t_count,
        "by_visual_query_type": by_query_type,
        "macro_exact_by_visual_stratum": macro_by_stratum,
        "worst_visual_query_type_exact_by_stratum": worst_qtype_by_stratum,
        "requires_grad_false_during_inference": requires_grad_false,
        "grad_enabled_inside_no_grad_inference": grad_enabled_inside_inference,
    }, details


def pair_permutation_report(model: torch.nn.Module, episodes: list[R4A0REpisode], device: torch.device, batch_size: int) -> dict[str, Any]:
    perms = list(itertools.permutations(range(3)))
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "same": 0})
    total = 0
    same_grid = 0
    max_prob = 0.0
    max_context = 0.0
    max_pointer = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            base_batch = tensors_from_episodes(chunk, device)
            base = model(base_batch["query_grids"], base_batch["support_input_grids"], base_batch["support_output_grids"])
            for order in perms:
                views = [reorder_episode(episode, order) for episode in chunk]
                batch = tensors_from_episodes(views, device)
                out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                matches = (out["decoded_grid"] == base["decoded_grid"]).flatten(1).all(dim=1)
                total += len(chunk)
                same_grid += int(matches.sum().item())
                max_prob = max(max_prob, float((out["output_probs"] - base["output_probs"]).abs().max().item()))
                max_context = max(max_context, float((out["task_context"] - base["task_context"]).abs().max().item()))
                max_pointer = max(max_pointer, float((out["pointer_attention"] - base["pointer_attention"]).abs().max().item()))
                for index, episode in enumerate(chunk):
                    key = _row_key(episode)
                    rows[key]["total"] += 1
                    rows[key]["same"] += int(bool(matches[index].item()))
    by_function: dict[str, dict[str, Any]] = {stratum: {} for stratum in VISUAL_STRATA}
    for (stratum, word), row in sorted(rows.items()):
        by_function[stratum][word] = {"presentations": row["total"], "same_final_grid_rate": row["same"] / max(row["total"], 1)}
    return {
        "support_pair_permutations": [list(order) for order in perms],
        "presentations": total,
        "same_final_grid_rate": same_grid / max(total, 1),
        "max_output_probability_difference": max_prob,
        "max_task_context_difference": max_context,
        "max_pointer_attention_difference": max_pointer,
        "by_visual_stratum_and_function": by_function,
        "passed": same_grid / max(total, 1) >= 0.999 and max(max_prob, max_context, max_pointer) <= 1e-5,
    }


def ablation_report(model: torch.nn.Module, episodes: list[R4A0REpisode], device: torch.device, batch_size: int, *, kind: str) -> dict[str, Any]:
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"count": 0, "exact": 0, "grid_change": 0, "pointer_change": 0})
    overall = {"count": 0, "exact": 0, "grid_change": 0, "pointer_change": 0}
    shifts = (0,) if kind == "zero_support" else (1, 2)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            normal_batch = tensors_from_episodes(chunk, device)
            normal = model(normal_batch["query_grids"], normal_batch["support_input_grids"], normal_batch["support_output_grids"])
            for shift in shifts:
                if kind == "zero_support":
                    views = chunk
                    batch = normal_batch
                    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
                else:
                    views = [derange_episode(episode, shift) for episode in chunk]
                    batch = tensors_from_episodes(views, device)
                    out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
                exact = exact_mask(out["decoded_grid"], normal_batch["target_grids"])
                for index, episode in enumerate(chunk):
                    key = _row_key(episode)
                    grid_change = not torch.equal(out["decoded_grid"][index], normal["decoded_grid"][index])
                    pointer_change = not torch.allclose(out["pointer_attention"][index], normal["pointer_attention"][index], atol=1e-7, rtol=0.0)
                    for row in (rows[key], overall):
                        row["count"] += 1
                        row["exact"] += int(bool(exact[index].item()))
                        row["grid_change"] += int(grid_change)
                        row["pointer_change"] += int(pointer_change)
    by_function: dict[str, dict[str, Any]] = {stratum: {} for stratum in VISUAL_STRATA}
    for (stratum, word), row in sorted(rows.items()):
        by_function[stratum][word] = {
            "episode_count": row["count"],
            f"{kind}_exact": row["exact"] / max(row["count"], 1),
            f"normal_to_{kind}_final_grid_change_rate": row["grid_change"] / max(row["count"], 1),
            f"normal_to_{kind}_pointer_change_rate": row["pointer_change"] / max(row["count"], 1),
        }
    return {
        "ablation": kind,
        "overall": {
            "episode_count": overall["count"],
            f"{kind}_exact": overall["exact"] / max(overall["count"], 1),
            f"normal_to_{kind}_final_grid_change_rate": overall["grid_change"] / max(overall["count"], 1),
            f"normal_to_{kind}_pointer_change_rate": overall["pointer_change"] / max(overall["count"], 1),
        },
        "by_visual_stratum_and_function": by_function,
    }


def length3_counterfactual_report(model: torch.nn.Module, episodes: list[R4A0REpisode], device: torch.device, batch_size: int, eligible_words: list[str]) -> dict[str, Any]:
    rows_by_word: dict[str, dict[str, int]] = {word: {"count": 0, "exact": 0} for word in eligible_words}
    decoded_by_group: dict[str, dict[str, str]] = defaultdict(dict)
    exact_by_group: dict[str, dict[str, bool]] = defaultdict(dict)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(episodes), batch_size):
            chunk = episodes[start : start + batch_size]
            batch = tensors_from_episodes(chunk, device)
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
            exact = exact_mask(out["decoded_grid"], batch["target_grids"])
            for index, episode in enumerate(chunk):
                word = compact_word(episode.sequence_class)
                group_id = str(episode.counterfactual_group_id)
                decoded_hash = stable_hash(out["decoded_grid"][index].detach().cpu().long().tolist())
                rows_by_word[word]["count"] += 1
                rows_by_word[word]["exact"] += int(bool(exact[index].item()))
                decoded_by_group[group_id][word] = decoded_hash
                exact_by_group[group_id][word] = bool(exact[index].item())
    all_context_correct = 0
    support_sensitive = 0
    group_count = 0
    pair_switch: dict[str, list[float]] = {f"{left}_vs_{right}": [] for left, right in itertools.combinations(eligible_words, 2)}
    for group_id, decoded in decoded_by_group.items():
        if set(decoded) != set(eligible_words):
            continue
        group_count += 1
        all_context_correct += int(all(exact_by_group[group_id].get(word, False) for word in eligible_words))
        support_sensitive += int(len(set(decoded.values())) == len(eligible_words))
        for left, right in itertools.combinations(eligible_words, 2):
            pair_switch[f"{left}_vs_{right}"].append(float(decoded[left] != decoded[right]))
    return {
        "group_count": group_count,
        "exact_by_canonical_length3_context": {
            word: rows_by_word[word]["exact"] / max(rows_by_word[word]["count"], 1)
            for word in eligible_words
        },
        "all_context_correct_group_rate": all_context_correct / max(group_count, 1),
        "support_sensitive_output_rate": support_sensitive / max(group_count, 1),
        "pairwise_output_switch_rates": {key: statistics.fmean(values) if values else 0.0 for key, values in pair_switch.items()},
    }


def query_type_worst_report(metrics: dict[str, Any], stratum: str) -> dict[str, Any]:
    rows = metrics["by_visual_query_type"][stratum]
    if not rows:
        return {"query_type_count": 0, "worst_query_type_exact": 0.0, "worst_query_type_key": None}
    worst_key, worst_row = min(rows.items(), key=lambda item: item[1]["direct_grid_exact_accuracy"])
    return {
        "query_type_count": len(rows),
        "worst_query_type_key": worst_key,
        "worst_query_type_exact": worst_row["direct_grid_exact_accuracy"],
    }


def failed_gates_for_seed(
    metrics: dict[str, Any],
    perm: dict[str, Any],
    zero: dict[str, Any],
    deranged: dict[str, Any],
    cf: dict[str, Any],
    eligible_words: list[str],
) -> list[str]:
    failed: list[str] = []
    train_macro = metrics["macro_exact_by_visual_stratum"][TRAIN_VISIBLE_STRATUM]
    sealed_macro = metrics["macro_exact_by_visual_stratum"][SEALED_STRATUM]
    if train_macro < 0.99:
        failed.append("length_only_train_visible_macro_exact")
    if sealed_macro < 0.99:
        failed.append("sealed_visual_macro_exact")
    for word in eligible_words:
        train_exact = metrics["by_visual_stratum_and_canonical_function"][TRAIN_VISIBLE_STRATUM][word]["direct_grid_exact_accuracy"]
        sealed_exact = metrics["by_visual_stratum_and_canonical_function"][SEALED_STRATUM][word]["direct_grid_exact_accuracy"]
        if train_exact < 0.98:
            failed.append(f"length_only_function_{word}")
        if sealed_exact < 0.98:
            failed.append(f"sealed_function_{word}")
    if query_type_worst_report(metrics, SEALED_STRATUM)["worst_query_type_exact"] < 0.98:
        failed.append("worst_heldout_visual_query_type")
    if not perm["passed"]:
        failed.append("pair_permutation")
    zero_overall = zero["overall"]
    deranged_overall = deranged["overall"]
    if zero_overall["normal_to_zero_support_final_grid_change_rate"] < 0.95 or zero_overall["zero_support_exact"] > 0.30:
        failed.append("zero_support_causal_gate")
    if deranged_overall["normal_to_deranged_pairing_final_grid_change_rate"] < 0.95 or deranged_overall["deranged_pairing_exact"] > 0.30:
        failed.append("deranged_pairing_causal_gate")
    if min(cf["exact_by_canonical_length3_context"].values()) < 0.99 or cf["all_context_correct_group_rate"] < 0.99 or cf["support_sensitive_output_rate"] < 0.99:
        failed.append("length3_order_counterfactual")
    return failed


def verdict_from_failures(failed_by_seed: dict[int, list[str]]) -> str:
    flat = {item for rows in failed_by_seed.values() for item in rows}
    if not flat:
        return VERDICT_PASS
    train_fail = any(item.startswith("length_only") for item in flat)
    sealed_fail = any(item.startswith("sealed") or item == "worst_heldout_visual_query_type" for item in flat)
    if any(item.endswith("causal_gate") for item in flat):
        return VERDICT_SUPPORT_FAIL
    if "length3_order_counterfactual" in flat:
        return VERDICT_ORDER_FAIL
    if train_fail:
        return VERDICT_LENGTH_FAIL
    if sealed_fail:
        return VERDICT_VISUAL_FAIL
    if "pair_permutation" in flat:
        return VERDICT_INCONCLUSIVE
    return VERDICT_INCONCLUSIVE


def run_pytest(data_out: Path, *, force: bool) -> dict[str, Any]:
    path = data_out / "pytest_result.txt"
    if path.exists() and not force:
        output = path.read_text(encoding="utf-8")
        passed = " passed" in output and " failed" not in output and "ERROR" not in output
        return {"returncode": 0 if passed else 1, "passed": passed, "output_tail": output[-4000:], "reused_existing_result": True}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
    save_text(path, result.stdout)
    return {"returncode": result.returncode, "passed": result.returncode == 0, "output_tail": result.stdout[-4000:], "reused_existing_result": False}


def load_or_build_catalog(data_out: Path) -> dict[str, Any]:
    path = data_out / "length3_function_catalog.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    catalog = build_length3_function_catalog(default_probe_query_types())
    save_json(path, catalog)
    return catalog


def load_or_build_data(data_out: Path, eligible_words: list[str]) -> tuple[list[R4A0REpisode], list[R4A0REpisode], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_path = data_out / "length3_probe_dataset.jsonl"
    cf_path = data_out / "length3_order_counterfactual_suite.jsonl"
    manifest_path = data_out / "length3_dataset_manifest.json"
    feasibility_path = data_out / "length3_query_type_feasibility.json"
    rejection_path = data_out / "length3_scene_rejection_report.json"
    cf_manifest_path = data_out / "length3_counterfactual_manifest.json"
    if all(path.exists() for path in (dataset_path, cf_path, manifest_path, feasibility_path, rejection_path, cf_manifest_path)):
        return (
            load_jsonl(dataset_path),
            load_jsonl(cf_path),
            json.loads(manifest_path.read_text(encoding="utf-8")),
            json.loads(feasibility_path.read_text(encoding="utf-8")),
            json.loads(rejection_path.read_text(encoding="utf-8")),
            json.loads(cf_manifest_path.read_text(encoding="utf-8")),
        )
    episodes, manifest, feasibility, rejection = build_length3_probe_dataset(eligible_words, default_probe_query_types())
    cf_rows, cf_manifest, cf_rejections = build_length3_counterfactual_suite(eligible_words, default_probe_query_types(), REQUIRED_COUNTERFACTUAL_GROUPS)
    rejection["counterfactual_scene_rejection_counts"] = cf_rejections
    write_probe_artifacts(data_out, episodes, cf_rows, manifest, feasibility, rejection, cf_manifest)
    return episodes, cf_rows, manifest, feasibility, rejection, cf_manifest


def _foreground_source_rows(episode: R4A0REpisode, pointer: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pointer_argmax = pointer.reshape(49, 49).argmax(dim=-1)
    pointer_mass = pointer.reshape(49, 49)
    for y, row in enumerate(episode.target):
        for x, value in enumerate(row):
            if value == 0:
                continue
            out_index = y * 7 + x
            oracle_index, oracle_x, oracle_y = source_index_for_output_exact(episode.sequence_class, x, y)
            pred_index = int(pointer_argmax[out_index].item())
            pred_y, pred_x = divmod(pred_index, 7)
            rows.append(
                {
                    "target_coord": [x, y],
                    "oracle_source_coord": [oracle_x, oracle_y],
                    "pointer_argmax_coord": [pred_x, pred_y],
                    "pointer_mass_on_exact_oracle_coord": float(pointer_mass[out_index, oracle_index].item()) if oracle_index >= 0 else 0.0,
                }
            )
    return rows


def describe_episode(model: torch.nn.Module, episode: R4A0REpisode, device: torch.device, *, intervention: str | None = None) -> dict[str, Any]:
    batch = tensors_from_episodes([episode], device)
    with torch.no_grad():
        if intervention == "zero_support":
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
        else:
            out = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
    return {
        "canonical_sequence": compact_word(episode.sequence_class),
        "marked_evaluation_only": True,
        "affine_signature": affine_payload(episode.sequence_class),
        "visual_stratum": episode.split,
        "query_type_key": query_type_key(episode.query_type),
        "query_grid": grid_to_list(episode.query),
        "support_pairs": [{"input": grid_to_list(pair.input), "output": grid_to_list(pair.output)} for pair in episode.supports],
        "target": grid_to_list(episode.target),
        "decoded_output": out["decoded_grid"][0].detach().cpu().long().tolist(),
        "exact": bool(torch.equal(out["decoded_grid"][0], batch["target_grids"][0])),
        "foreground_oracle_and_pointer_sources": _foreground_source_rows(episode, out["pointer_attention"][0].detach().cpu()),
        "task_context_summary": {
            "norm": float(out["task_context"][0].norm().item()),
            "mean": float(out["task_context"][0].mean().item()),
            "std": float(out["task_context"][0].std(unbiased=False).item()),
        },
    }


def write_prediction_examples(path: Path, episodes: list[R4A0REpisode], cf_rows: list[R4A0REpisode], model: torch.nn.Module, device: torch.device, details: list[dict[str, Any]]) -> None:
    examples: list[dict[str, Any]] = []
    by_function: dict[str, dict[str, list[R4A0REpisode]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        by_function[compact_word(episode.sequence_class)][episode.split].append(episode)
    for word in sorted(by_function):
        selected = by_function[word][TRAIN_VISIBLE_STRATUM][:2] + by_function[word][SEALED_STRATUM][:1]
        for episode in selected:
            examples.append({"kind": "normal_length3", "example": describe_episode(model, episode, device)})
    for episode in episodes[:6]:
        normal = describe_episode(model, episode, device)
        zero = describe_episode(model, episode, device, intervention="zero_support")
        examples.append({"kind": "normal_vs_zero_support", "normal": normal, "zero_support_output": zero["decoded_output"]})
    for episode in episodes[6:12]:
        normal = describe_episode(model, episode, device)
        deranged = describe_episode(model, derange_episode(episode, 1), device)
        examples.append({"kind": "normal_vs_deranged_pairing", "normal": normal, "deranged_output": deranged["decoded_output"]})
    for episode in episodes[12:18]:
        normal = describe_episode(model, episode, device)
        permuted = describe_episode(model, reorder_episode(episode, (2, 0, 1)), device)
        examples.append({"kind": "pair_permutation", "normal": normal, "permuted_output": permuted["decoded_output"]})
    for _, group in list(records_by_group(cf_rows).items())[:6]:
        examples.append({"kind": "same_query_length3_counterfactual_group", "members": [describe_episode(model, episode, device) for episode in sorted(group, key=lambda item: compact_word(item.sequence_class))]})
    failures = [detail for detail in details if not detail["exact"]][:12]
    examples.append({"kind": "failures_up_to_12", "failures": failures})
    save_text(path, "\n\n".join(json.dumps(item, sort_keys=True) for item in examples) + "\n")


def write_summary(data_out: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# R4A-1P Forensic Closure and Frozen Length-Three Composition Probe", ""]
    lines.append(f"Verdict: `{aggregate['verdict']}`")
    lines.append("")
    lines.append("This is an evaluation-only probe of the frozen R4A-0R V2 direct support-conditioned pointer model. No training, checkpoint selection, or architecture change occurred.")
    lines.append("")
    lines.append("|seed|train-visible macro|sealed macro|sealed worst qtype|zero exact|deranged exact|failed gates|")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for seed, row in aggregate["final_metrics_by_seed"].items():
        metrics = row["final_metrics"]
        zero = row["zero_support"]
        deranged = row["deranged_pairing"]
        lines.append(
            f"|{seed}|{metrics['macro_exact_by_visual_stratum'][TRAIN_VISIBLE_STRATUM]:.6f}|"
            f"{metrics['macro_exact_by_visual_stratum'][SEALED_STRATUM]:.6f}|"
            f"{query_type_worst_report(metrics, SEALED_STRATUM)['worst_query_type_exact']:.6f}|"
            f"{zero['overall']['zero_support_exact']:.6f}|"
            f"{deranged['overall']['deranged_pairing_exact']:.6f}|"
            f"{','.join(row['failed_gates']) or 'none'}|"
        )
    save_text(data_out / "summary.md", "\n".join(lines) + "\n")


def finish_with_verdict(data_out: Path, before_hashes: dict[str, str], verdict: str, payload: dict[str, Any], device: torch.device | None = None) -> int:
    after_hashes = protected_hashes(data_out)
    save_json(data_out / "protected_hashes_after.json", after_hashes)
    aggregate = payload | {
        "verdict": verdict,
        "protected_hash_status": hash_status(before_hashes, after_hashes),
        "loaded_checkpoint_audit": approved_loaded_checkpoint_audit(),
    }
    if device is not None:
        aggregate["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        aggregate["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    print(json.dumps({"verdict": verdict, "failed_or_stop_reason": payload.get("failed_or_stop_reason")}, indent=2, sort_keys=True), flush=True)
    return 0 if verdict in {VERDICT_PASS, VERDICT_INCONCLUSIVE} else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry_run", "full"), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    data_out = Path(args.data_out)
    data_out.mkdir(parents=True, exist_ok=True)
    save_json(data_out / "run_config.json", vars(args) | {"seeds": seeds, "evaluation_only": True, "training_started": False, "checkpoint_selection": False})

    before_hashes = protected_hashes(data_out)
    save_json(data_out / "protected_hashes_before.json", before_hashes)
    environment = cuda_environment_report()
    save_json(data_out / "environment_report.json", environment)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_before = checkpoint_provenance_before()
    save_json(data_out / "frozen_checkpoint_provenance.json", checkpoint_before)
    pointer_audit = pointer_metric_semantics_audit()
    save_json(data_out / "pointer_metric_semantics_audit.json", pointer_audit)

    catalog = load_or_build_catalog(data_out)
    eligible_words = list(catalog["eligible_canonical_words"])
    if catalog["eligible_count"] < catalog["required_minimum_eligible_count"]:
        return finish_with_verdict(data_out, before_hashes, VERDICT_CATALOG_INSUFFICIENT, {"failed_or_stop_reason": "fewer than twelve eligible canonical length-three affine functions", "catalog": catalog}, device)

    episodes, cf_rows, dataset_manifest, feasibility, rejection_report, cf_manifest = load_or_build_data(data_out, eligible_words)
    if not feasibility["passed"] or not dataset_manifest["passed"]:
        return finish_with_verdict(data_out, before_hashes, VERDICT_QUERY_INSUFFICIENT, {"failed_or_stop_reason": "not every frozen visual query type can generate enough selected length-three scenes", "length3_query_type_feasibility": feasibility}, device)
    if not cf_manifest["passed"]:
        return finish_with_verdict(data_out, before_hashes, VERDICT_INCONCLUSIVE, {"failed_or_stop_reason": "counterfactual suite generation failed", "length3_counterfactual_manifest": cf_manifest}, device)

    pytest_result = run_pytest(data_out, force=args.mode == "dry_run")
    if not pytest_result["passed"]:
        return finish_with_verdict(data_out, before_hashes, VERDICT_INCONCLUSIVE, {"failed_or_stop_reason": "pytest failed", "pytest_result": pytest_result}, device)

    models_by_seed: dict[int, torch.nn.Module] = {}
    checkpoint_payloads: dict[int, dict[str, Any]] = {}
    checkpoint_model_provenance: dict[int, dict[str, Any]] = {}
    parameter_hashes_before: dict[int, Any] = {}
    for seed in seeds:
        if seed not in APPROVED_CHECKPOINTS:
            return finish_with_verdict(data_out, before_hashes, VERDICT_INCONCLUSIVE, {"failed_or_stop_reason": f"unapproved seed {seed}"}, device)
        model, payload, provenance = load_frozen_model(seed, device)
        models_by_seed[seed] = model
        checkpoint_payloads[seed] = {"epoch_index": payload.get("epoch_index"), "global_step": payload.get("global_step"), "seed": payload.get("seed")}
        checkpoint_model_provenance[seed] = provenance
        parameter_hashes_before[seed] = parameter_hashes(model)
    checkpoint_before["loaded_checkpoint_model_provenance"] = checkpoint_model_provenance
    checkpoint_before["checkpoint_directory_after_loading"] = snapshot_checkpoint_directory()
    save_json(data_out / "frozen_checkpoint_provenance.json", checkpoint_before)
    save_json(data_out / "frozen_parameter_hashes_before.json", parameter_hashes_before)

    provenance_audit = forward_provenance_audit(models_by_seed, episodes, device, args.batch_size)
    save_json(data_out / "forward_provenance_audit.json", provenance_audit)
    if not provenance_audit["passed"]:
        return finish_with_verdict(data_out, before_hashes, VERDICT_PROVENANCE_FAIL, {"failed_or_stop_reason": "forbidden metadata reached or affected forward output", "forward_provenance_audit": provenance_audit}, device)

    epoch_audit = epoch_zero_semantics_audit(seeds)
    save_json(data_out / "epoch_zero_semantics_audit.json", epoch_audit)
    pointer_recompute = independent_pointer_coordinate_recomputation(models_by_seed, device, args.batch_size)
    save_json(data_out / "pointer_coordinate_recomputation.json", pointer_recompute)

    parameter_hashes_mid = {seed: parameter_hashes(model) for seed, model in models_by_seed.items()}
    if parameter_hashes_mid != parameter_hashes_before:
        return finish_with_verdict(data_out, before_hashes, VERDICT_INCONCLUSIVE, {"failed_or_stop_reason": "parameter mutation detected before probe evaluation"}, device)

    if args.mode == "dry_run":
        dry_metrics: dict[int, Any] = {}
        max_records = args.batch_size * args.max_batches
        for seed, model in models_by_seed.items():
            dry_metrics[seed], _ = evaluate_probe_model(model, episodes, device, args.batch_size, max_records=max_records)
        parameter_hashes_after = {seed: parameter_hashes(model) for seed, model in models_by_seed.items()}
        checkpoint_after = snapshot_checkpoint_directory()
        dry_report = {
            "mode": "dry_run",
            "cuda_passed": True,
            "frozen_checkpoint_provenance_passed": True,
            "pytest_result": pytest_result,
            "length3_catalog_passed": True,
            "length3_query_type_feasibility_passed": True,
            "forward_provenance_audit_passed": True,
            "training_started": False,
            "dry_eval_max_records_per_seed": max_records,
            "dry_metrics_by_seed": dry_metrics,
            "parameter_hashes_unchanged": parameter_hashes_after == parameter_hashes_before,
            "checkpoint_directory_unchanged": checkpoint_after == checkpoint_before["checkpoint_directory_before"],
        }
        save_json(data_out / "dry_run_report.json", dry_report)
        save_json(data_out / "frozen_parameter_hashes_after.json", parameter_hashes_after)
        checkpoint_before["checkpoint_directory_after_run"] = checkpoint_after
        save_json(data_out / "frozen_checkpoint_provenance.json", checkpoint_before)
        return finish_with_verdict(data_out, before_hashes, VERDICT_INCONCLUSIVE, dry_report, device)

    final_by_seed: dict[int, Any] = {}
    failed_by_seed: dict[int, list[str]] = {}
    for seed, model in models_by_seed.items():
        final_metrics, details = evaluate_probe_model(model, episodes, device, args.batch_size)
        perm = pair_permutation_report(model, episodes, device, args.batch_size)
        zero = ablation_report(model, episodes, device, args.batch_size, kind="zero_support")
        deranged = ablation_report(model, episodes, device, args.batch_size, kind="deranged_pairing")
        cf = length3_counterfactual_report(model, cf_rows, device, args.batch_size, eligible_words)
        failed = failed_gates_for_seed(final_metrics, perm, zero, deranged, cf, eligible_words)
        failed_by_seed[seed] = failed
        final_by_seed[seed] = {
            "checkpoint_payload": checkpoint_payloads[seed],
            "final_metrics": final_metrics,
            "worst_query_type_train_visible": query_type_worst_report(final_metrics, TRAIN_VISIBLE_STRATUM),
            "worst_query_type_sealed": query_type_worst_report(final_metrics, SEALED_STRATUM),
            "pair_permutation": perm,
            "zero_support": zero,
            "deranged_pairing": deranged,
            "length3_counterfactual": cf,
            "failed_gates": failed,
        }
        save_json(data_out / f"final_metrics_seed_{seed}.json", final_metrics)
        save_json(data_out / f"pair_permutation_report_seed_{seed}.json", perm)
        save_json(data_out / f"zero_support_report_seed_{seed}.json", zero)
        save_json(data_out / f"deranged_pairing_report_seed_{seed}.json", deranged)
        save_json(data_out / f"length3_counterfactual_report_seed_{seed}.json", cf)
        if seed == seeds[0]:
            write_prediction_examples(data_out / "prediction_examples.txt", episodes, cf_rows, model, device, details)

    parameter_hashes_after = {seed: parameter_hashes(model) for seed, model in models_by_seed.items()}
    save_json(data_out / "frozen_parameter_hashes_after.json", parameter_hashes_after)
    checkpoint_after = snapshot_checkpoint_directory()
    checkpoint_before["checkpoint_directory_after_run"] = checkpoint_after
    checkpoint_before["checkpoint_directory_unchanged_after_run"] = checkpoint_after == checkpoint_before["checkpoint_directory_before"]
    save_json(data_out / "frozen_checkpoint_provenance.json", checkpoint_before)

    verdict = verdict_from_failures(failed_by_seed)
    aggregate = {
        "verdict": verdict,
        "environment": environment,
        "eligible_canonical_words": eligible_words,
        "final_metrics_by_seed": final_by_seed,
        "failed_gates_by_seed": failed_by_seed,
        "pointer_metric_semantics_conclusion": pointer_audit["semantics_conclusion"],
        "pointer_coordinate_recomputation_conclusion": pointer_recompute["conclusion"],
        "forward_provenance_audit_passed": provenance_audit["passed"],
        "epoch_zero_semantics": epoch_audit["conclusion"],
        "parameter_hashes_unchanged": parameter_hashes_after == parameter_hashes_before,
        "checkpoint_directory_unchanged": checkpoint_after == checkpoint_before["checkpoint_directory_before"],
        "pytest_result": pytest_result,
        "dataset_manifest": dataset_manifest,
        "length3_counterfactual_manifest": cf_manifest,
        "loaded_checkpoint_audit": approved_loaded_checkpoint_audit(),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    write_summary(data_out, aggregate)
    after_hashes = protected_hashes(data_out)
    save_json(data_out / "protected_hashes_after.json", after_hashes)
    aggregate["protected_hash_status"] = hash_status(before_hashes, after_hashes)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)
    print(json.dumps({"verdict": verdict, "failed_gates_by_seed": failed_by_seed}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
