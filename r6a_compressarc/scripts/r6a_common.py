from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/mnt/d/VRM")
EXPERIMENT_ROOT = PROJECT_ROOT / "r6a_compressarc"
REFERENCE_ROOT = PROJECT_ROOT / "third_party" / "compressarc_reference"
SMOKE_BUDGET = 1500
REFERENCE_BENCHMARK_BUDGET = 2000
SELECTION_SEED = 20260701
RUN_SEED = 0


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid_shape(grid: list[list[int]]) -> list[int]:
    return [len(grid), len(grid[0]) if grid else 0]


def hamming_grid(a: list[list[int]], b: list[list[int]], *, allow_shape_change: bool) -> int:
    ah, aw = grid_shape(a)
    bh, bw = grid_shape(b)
    if not allow_shape_change and (ah, aw) != (bh, bw):
        raise ValueError("shape mismatch in shape-preserving hamming")
    height = max(ah, bh)
    width = max(aw, bw)
    total = 0
    for row in range(height):
        for col in range(width):
            av = a[row][col] if row < ah and col < aw else None
            bv = b[row][col] if row < bh and col < bw else None
            total += int(av != bv)
    return total


def derangements(n: int) -> list[tuple[int, ...]]:
    return [perm for perm in itertools.permutations(range(n)) if all(perm[idx] != idx for idx in range(n))]


def select_derangement(outputs: list[list[list[int]]], *, shape_preserving: bool) -> dict[str, Any]:
    candidates = []
    for perm in derangements(len(outputs)):
        valid = True
        score = 0
        for idx, src_idx in enumerate(perm):
            if shape_preserving and grid_shape(outputs[idx]) != grid_shape(outputs[src_idx]):
                valid = False
                break
            score += hamming_grid(outputs[idx], outputs[src_idx], allow_shape_change=not shape_preserving)
        if valid:
            candidates.append({"permutation": list(perm), "total_output_hamming": score})
    candidates.sort(key=lambda row: (-row["total_output_hamming"], row["permutation"]))
    selected = candidates[0] if candidates else None
    return {"selected": selected, "candidates": candidates}


def pairing_manifest_for_task(task_id: str, problem: dict[str, Any]) -> dict[str, Any]:
    outputs = [pair["output"] for pair in problem["train"]]
    b1 = select_derangement(outputs, shape_preserving=False)
    if b1["selected"] is None or b1["selected"]["total_output_hamming"] == 0:
        b1_status = "B1_INELIGIBLE_NO_OUTPUT_CHANGING_DERANGEMENT"
    else:
        b1_status = "B1_ELIGIBLE"
    all_outputs_identical = all(hamming_grid(outputs[0], item, allow_shape_change=True) == 0 for item in outputs[1:]) if outputs else True
    b2 = select_derangement(outputs, shape_preserving=True)
    if all_outputs_identical:
        b2_status = "CONTENT_PAIRING_INELIGIBLE_IDENTICAL_OUTPUTS"
    elif b2["selected"] is None:
        b2_status = "CONTENT_PAIRING_INELIGIBLE_NO_SHAPE_PRESERVING_DERANGEMENT"
    elif b2["selected"]["total_output_hamming"] == 0:
        b2_status = "CONTENT_PAIRING_INELIGIBLE_IDENTICAL_OUTPUTS"
    else:
        b2_status = "CONTENT_PAIRING_ELIGIBLE"
    return {
        "task_id": task_id,
        "n_train": len(problem["train"]),
        "n_test": len(problem["test"]),
        "train_input_shapes": [grid_shape(pair["input"]) for pair in problem["train"]],
        "train_output_shapes": [grid_shape(pair["output"]) for pair in problem["train"]],
        "test_input_shapes": [grid_shape(pair["input"]) for pair in problem["test"]],
        "b1_status": b1_status,
        "b1_derangements": b1["candidates"],
        "b1_selected_permutation": b1["selected"]["permutation"] if b1["selected"] else None,
        "b2_status": b2_status,
        "b2_derangements": b2["candidates"],
        "b2_selected_permutation": b2["selected"]["permutation"] if b2["selected"] else None,
    }


def challenge_for_condition(problem: dict[str, Any], condition: str, permutation: list[int] | None = None) -> dict[str, Any]:
    cloned = json.loads(json.dumps(problem))
    if condition in {"FULL", "FULL_REPEAT"}:
        return cloned
    if permutation is None:
        raise ValueError(f"{condition} requires permutation")
    original_outputs = [pair["output"] for pair in problem["train"]]
    for idx, src_idx in enumerate(permutation):
        cloned["train"][idx]["output"] = original_outputs[src_idx]
    return cloned


def evaluate_prediction(
    *,
    challenge_problem: dict[str, Any],
    solution_grids: list[list[list[int]]],
    prediction_grids: list[list[list[int]]],
) -> dict[str, Any]:
    rows = []
    for idx, target in enumerate(solution_grids):
        pred = prediction_grids[idx] if idx < len(prediction_grids) else []
        query_input = challenge_problem["test"][idx]["input"]
        target_h, target_w = grid_shape(target)
        pred_h, pred_w = grid_shape(pred)
        query_h, query_w = grid_shape(query_input)
        correct = 0
        changed_correct = 0
        changed_count = 0
        pooled_changed_entries = []
        for row in range(target_h):
            for col in range(target_w):
                target_color = target[row][col]
                pred_valid = row < pred_h and col < pred_w
                pred_color = pred[row][col] if pred_valid else None
                query_valid = row < query_h and col < query_w
                query_color = query_input[row][col] if query_valid else None
                is_changed = (not query_valid) or query_color != target_color
                correct += int(pred_color == target_color)
                if is_changed:
                    changed_count += 1
                    changed_correct += int(pred_color == target_color)
                    pooled_changed_entries.append(
                        {
                            "row": row,
                            "col": col,
                            "query_color": query_color,
                            "target_color": target_color,
                            "predicted_color": pred_color,
                            "correct": pred_color == target_color,
                        }
                    )
        total = target_h * target_w
        rows.append(
            {
                "test_index": idx,
                "predicted_grid": pred,
                "target_grid": target,
                "output_shape_exact_match": [pred_h, pred_w] == [target_h, target_w],
                "exact_grid_match": pred == target,
                "all_cell_accuracy": correct / total if total else None,
                "changed_cell_count": changed_count,
                "changed_cell_accuracy": changed_correct / changed_count if changed_count else None,
                "changed_cell_loss_or_nll": None,
                "changed_cell_entries": pooled_changed_entries,
            }
        )
    changed_values = [row["changed_cell_accuracy"] for row in rows if row["changed_cell_accuracy"] is not None]
    total_changed = sum(row["changed_cell_count"] for row in rows)
    total_changed_correct = sum(
        sum(int(entry["correct"]) for entry in row["changed_cell_entries"])
        for row in rows
    )
    return {
        "rows": rows,
        "exact_grid_match": all(row["exact_grid_match"] for row in rows),
        "output_shape_exact_match": all(row["output_shape_exact_match"] for row in rows),
        "all_cell_accuracy": sum((row["all_cell_accuracy"] or 0.0) for row in rows) / len(rows) if rows else None,
        "changed_cell_macro_accuracy": sum(changed_values) / len(changed_values) if changed_values else None,
        "changed_cell_pooled_accuracy": total_changed_correct / total_changed if total_changed else None,
        "total_changed_cells": total_changed,
    }


def deterministic_selection(candidate_ids: list[str], count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    ids = sorted(candidate_ids)
    rng.shuffle(ids)
    return ids[:count]
