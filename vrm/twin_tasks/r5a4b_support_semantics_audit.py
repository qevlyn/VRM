from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from vrm.data import (
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    MAX_GRID_WIDTH,
    PAD_ID,
    ROLE_SUPPORT_INPUT,
    ROLE_SUPPORT_OUTPUT,
    ArcEpisode,
    grid_shape_mask,
    grid_to_text,
    model_inputs_from_batch,
    pad_grid_to_tokens,
)
from vrm.metrics import factorized_shape_color_loss
from vrm.twin_tasks.r5a4a_arc_manifest import build_r5a4a_splits, episodes_for_task_ids
from vrm.twin_tasks.r5a4a_neural_vrm_train import DEFAULT_SHAPE_LOSS_WEIGHT, load_state, resolve_device
from vrm.twin_tasks.r5a4b_prerun_patch_metrics import (
    DENSE_SCORING_NOTE,
    INTERPRETATION_SENTENCE,
    LIMITATION_SENTENCE,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
R5A4A_DIR = PROJECT_ROOT / "artifacts" / "r5a4a_neural_vrm_arc_grounded_preflight"
R5A4B_PATCH_DIR = PROJECT_ROOT / "artifacts" / "r5a4b_neural_vrm_prerun_patch"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "r5a4b_support_semantics_audit"
BOOTSTRAP_SEED = 20260701
BOOTSTRAP_REPLICATES = 10000

TRAINED_FULL = "TRAINED_FULL_SUPPORT_REPLAY"
QUERY_ONLY = "TRAINED_QUERY_ONLY_BYPASS"
PAIRING_BROKEN = "TRAINED_OUTPUT_PAIRING_BROKEN"
INPUTS_ZEROED = "TRAINED_SUPPORT_INPUTS_ZEROED"
GEOMETRY_SCRAMBLED = "TRAINED_OUTPUT_GEOMETRY_SCRAMBLED"
ZEROED_OUTPUTS = "TRAINED_ZEROED_SUPPORT_OUTPUTS_REPLAY"
FRESH = "FRESH_INIT_REFERENCE"
CONDITIONS = (
    TRAINED_FULL,
    QUERY_ONLY,
    PAIRING_BROKEN,
    INPUTS_ZEROED,
    GEOMETRY_SCRAMBLED,
    ZEROED_OUTPUTS,
    FRESH,
)
TRAINED_CONDITIONS = (
    TRAINED_FULL,
    QUERY_ONLY,
    PAIRING_BROKEN,
    INPUTS_ZEROED,
    GEOMETRY_SCRAMBLED,
    ZEROED_OUTPUTS,
)

REQUIRED_R5A4A_HASHES = {
    "initial_checkpoint": "59cb6e65aaeb39432f143263608d692d558ad532d5e29b0b5ccbf403e1fdb43e",
    "selected_checkpoint": "19d2b4b0ef225e3a3a2e83e6b42f2100333cee24e38b4b49c36520578ded850e",
    "sentinel_manifest": "7a6f5d12625e36c125d21c7f39101f927a81be3c853e7e47a4d21811ff8f8ac4",
    "model_interface_audit": "f384d23c30305e0e3bedeb91b1740b5325ccb4830516dc1a795798219ed77b5d",
}

REQUIRED_R5A4B_PRERUN_HASHES = {
    "patch_metrics_source": "8eaafd3fa0ce1fe490c2979cc4a0990ad9ea169e5a941d0356e8c883872731c4",
    "patch_runner_source": "d9857b4638db13c0188a0acf5068208be7fc6bdfdd28ce778616f7ba6315f554",
    "patch_shape_regime_metrics": "fb0854d5c35fc6f7cee16307b58e739357a1b34511ff04a67ec35b5916a36ec9",
    "patch_dense_prediction_scoring_ledger": "0859c303d3f37634b2209339bab296dc5bd034d5ff43eb7878462d5b23beb547",
    "patch_changed_cell_metrics": "2c9cbc1022d42ca99f0d21221fce4d0d3fdbfc6c9053e1065f3ee5a083de9381",
    "patch_pairing_rotation_ledger": "b42ba8abbff480fee9bb49612500c798317971b8e4e89cdefdb6fdedc913696c",
}

R5A4A_INPUTS = {
    "initial_checkpoint": R5A4A_DIR / "initial_state_before_any_optimizer_step.pt",
    "selected_checkpoint": R5A4A_DIR / "selected_trained_state.pt",
    "sentinel_manifest": R5A4A_DIR / "arc_sentinel_manifest.json",
    "model_interface_audit": R5A4A_DIR / "model_interface_audit.json",
}
PATCH_INPUTS = {
    "patch_metrics_source": PROJECT_ROOT / "vrm" / "twin_tasks" / "r5a4b_prerun_patch_metrics.py",
    "patch_runner_source": PROJECT_ROOT / "vrm" / "twin_tasks" / "run_r5a4b_prerun_patch.py",
    "patch_shape_regime_metrics": R5A4B_PATCH_DIR / "shape_regime_metrics.json",
    "patch_dense_prediction_scoring_ledger": R5A4B_PATCH_DIR / "dense_prediction_scoring_ledger.json",
    "patch_changed_cell_metrics": R5A4B_PATCH_DIR / "changed_cell_pooled_and_macro_metrics.json",
    "patch_pairing_rotation_ledger": R5A4B_PATCH_DIR / "pairing_rotation_integrity_ledger.json",
}
PROTECTED_PATTERNS = (
    "vrm/model.py",
    "vrm/data.py",
    "vrm/twin_tasks/r5a2*.py",
    "vrm/twin_tasks/r5a3*.py",
    "vrm/twin_tasks/r5a4a_*.py",
    "vrm/twin_tasks/r5a4b_prerun_patch_metrics.py",
    "vrm/twin_tasks/run_r5a4b_prerun_patch.py",
    "tests/test_r5a2*.py",
    "tests/test_r5a3*.py",
    "tests/test_r5a4a_*.py",
    "tests/test_r5a4b_prerun_patch.py",
    "artifacts/r5a2*/*",
    "artifacts/r5a3*/*",
    "artifacts/r5a4a_neural_vrm_arc_grounded_preflight/*",
    "artifacts/r5a4b_neural_vrm_prerun_patch/*",
)


@dataclass(frozen=True)
class AuditInputs:
    device: torch.device
    episodes: list[ArcEpisode]
    fresh_model: Any
    trained_model: Any


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    item = tensor.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(item.shape)}|{item.dtype}".encode() + b"\0" + item.numpy().tobytes()).hexdigest()


def protected_source_manifest() -> dict[str, Any]:
    rows = []
    for pattern in PROTECTED_PATTERNS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                rows.append(
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
    rows = sorted({row["path"]: row for row in rows}.values(), key=lambda row: row["path"])
    return {"project_root": str(PROJECT_ROOT), "file_count": len(rows), "files": rows}


def render_manifest(manifest: dict[str, Any]) -> str:
    lines = [f"project_root: {manifest['project_root']}", f"file_count: {manifest['file_count']}", ""]
    lines.extend(f"{row['sha256']}  {row['size_bytes']:>12}  {row['path']}" for row in manifest["files"])
    return "\n".join(lines).rstrip() + "\n"


def verify_frozen_inputs() -> dict[str, Any]:
    rows = []
    fail = False
    for name, path in R5A4A_INPUTS.items():
        exists = path.exists()
        digest = file_sha256(path) if exists else None
        expected = REQUIRED_R5A4A_HASHES.get(name)
        ok = exists and (expected is None or digest == expected)
        rows.append({"name": name, "path": str(path), "sha256": digest, "expected_sha256": expected, "ok": ok})
        fail = fail or not ok
    for name, path in PATCH_INPUTS.items():
        exists = path.exists()
        digest = file_sha256(path) if exists else None
        expected = REQUIRED_R5A4B_PRERUN_HASHES.get(name)
        ok = exists and digest == expected
        rows.append({"name": name, "path": str(path), "sha256": digest, "expected_sha256": expected, "ok": ok})
        fail = fail or not ok
    payload = {
        "status": "FROZEN_INPUT_INTEGRITY_PASS" if not fail else "FROZEN_INPUT_INTEGRITY_FAIL",
        "rows": rows,
        "pre_run_patch_hashes_checked_before_inference": True,
    }
    write_json(ARTIFACT_DIR / "frozen_input_verification.json", payload)
    return payload


def load_audit_inputs(device_arg: str) -> AuditInputs:
    device = resolve_device(device_arg)
    fresh_model, _fresh_state = load_state(R5A4A_INPUTS["initial_checkpoint"], device)
    trained_model, _trained_state = load_state(R5A4A_INPUTS["selected_checkpoint"], device)
    splits = build_r5a4a_splits()
    episodes = episodes_for_task_ids(splits.tasks_by_id, splits.sentinel_task_ids, source_split="sentinel")
    return AuditInputs(device=device, episodes=episodes, fresh_model=fresh_model, trained_model=trained_model)


def forward_only_batch(episode: ArcEpisode) -> dict[str, Any]:
    evidence = []
    for support in episode.supports:
        for role, grid in ((ROLE_SUPPORT_INPUT, support.input), (ROLE_SUPPORT_OUTPUT, support.output)):
            for row_idx, row in enumerate(grid.cells):
                for col_idx, color in enumerate(row):
                    evidence.append((color, row_idx, col_idx, role, support.pair_index))
    count = max(1, len(evidence))
    colors = torch.full((1, count), PAD_ID, dtype=torch.long)
    rows = torch.zeros((1, count), dtype=torch.long)
    cols = torch.zeros((1, count), dtype=torch.long)
    roles = torch.zeros((1, count), dtype=torch.long)
    pair_indices = torch.zeros((1, count), dtype=torch.long)
    mask = torch.zeros((1, count), dtype=torch.bool)
    for idx, (color, row, col, role, pair_index) in enumerate(evidence):
        colors[0, idx] = color
        rows[0, idx] = row
        cols[0, idx] = col
        roles[0, idx] = role
        pair_indices[0, idx] = pair_index
        mask[0, idx] = True
    query_inputs = pad_grid_to_tokens(episode.query.input).unsqueeze(0)
    query_input_mask = grid_shape_mask(episode.query.input.height, episode.query.input.width).unsqueeze(0)
    return {
        "support_evidence_colors": colors,
        "support_evidence_rows": rows,
        "support_evidence_cols": cols,
        "support_evidence_roles": roles,
        "support_evidence_pair_indices": pair_indices,
        "support_evidence_mask": mask,
        "query_inputs": query_inputs,
        "query_input_mask": query_input_mask,
        "target_available_before_forward": False,
    }


def target_batch_after_forward(episode: ArcEpisode) -> dict[str, Any]:
    from vrm.data import collate_episodes

    return collate_episodes([episode])


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def dummy_support_batch(episode: ArcEpisode, variant: str) -> dict[str, Any]:
    batch = forward_only_batch(episode)
    if variant == "A":
        color, row, col, role, pair = 0, 0, 0, ROLE_SUPPORT_INPUT, 0
    elif variant == "B":
        color, row, col, role, pair = 9, 1, 1, ROLE_SUPPORT_OUTPUT, 1
    else:
        raise ValueError(f"unknown dummy variant {variant}")
    batch["support_evidence_colors"] = torch.tensor([[color]], dtype=torch.long)
    batch["support_evidence_rows"] = torch.tensor([[row]], dtype=torch.long)
    batch["support_evidence_cols"] = torch.tensor([[col]], dtype=torch.long)
    batch["support_evidence_roles"] = torch.tensor([[role]], dtype=torch.long)
    batch["support_evidence_pair_indices"] = torch.tensor([[pair]], dtype=torch.long)
    batch["support_evidence_mask"] = torch.tensor([[True]], dtype=torch.bool)
    return batch


def zero_support_inputs(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = clone_batch(batch)
    input_mask = out["support_evidence_mask"].to(torch.bool) & (out["support_evidence_roles"] == ROLE_SUPPORT_INPUT)
    out["support_evidence_colors"][input_mask] = 0
    return out, {"support_input_color_values_zeroed": int(input_mask.sum().item())}


def zero_support_outputs(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = clone_batch(batch)
    output_mask = out["support_evidence_mask"].to(torch.bool) & (out["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
    out["support_evidence_colors"][output_mask] = 0
    return out, {"support_output_color_values_zeroed": int(output_mask.sum().item())}


def visible_pair_ids(batch: dict[str, Any]) -> list[int]:
    mask = batch["support_evidence_mask"][0].to(torch.bool)
    return sorted(int(item) for item in torch.unique(batch["support_evidence_pair_indices"][0, mask]).detach().cpu().tolist())


def output_profile(batch: dict[str, Any], pair_id: int) -> dict[str, Any]:
    mask = (
        batch["support_evidence_mask"][0].to(torch.bool)
        & (batch["support_evidence_roles"][0] == ROLE_SUPPORT_OUTPUT)
        & (batch["support_evidence_pair_indices"][0] == pair_id)
    )
    rows = batch["support_evidence_rows"][0, mask].detach().cpu().tolist()
    cols = batch["support_evidence_cols"][0, mask].detach().cpu().tolist()
    colors = batch["support_evidence_colors"][0, mask].detach().cpu().tolist()
    coords = list(zip(rows, cols))
    return {
        "source_support_pair_id": pair_id,
        "output_shape": [max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0],
        "output_cell_count": len(colors),
        "coordinate_sha256": hashlib.sha256(json.dumps(coords, sort_keys=True).encode()).hexdigest(),
        "color_histogram": {str(key): value for key, value in sorted(Counter(colors).items())},
    }


def rotate_output_pairing(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = clone_batch(batch)
    original_pair_indices = batch["support_evidence_pair_indices"].clone()
    ids = visible_pair_ids(batch)
    if len(ids) < 2:
        return out, {"visible_pair_ids": ids, "rotation_applicable": False, "source_to_destination_rotation": {}, "rotation_rows": []}
    destination = {source: ids[(idx + 1) % len(ids)] for idx, source in enumerate(ids)}
    rows = []
    for source, dest in destination.items():
        source_mask = (
            out["support_evidence_mask"].to(torch.bool)
            & (out["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
            & (original_pair_indices == source)
        )
        out["support_evidence_pair_indices"][source_mask] = dest
        profile = output_profile(batch, source)
        profile["destination_pair_index"] = dest
        rows.append(profile)
    return out, {
        "visible_pair_ids": ids,
        "rotation_applicable": True,
        "source_to_destination_rotation": {str(source): dest for source, dest in destination.items()},
        "rotation_rows": rows,
    }


def stable_seed_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def scramble_output_geometry(batch: dict[str, Any], *, task_id: str, query_pair_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    out = clone_batch(batch)
    ledger = []
    for pair_id in visible_pair_ids(batch):
        output_mask = (
            out["support_evidence_mask"].to(torch.bool)
            & (out["support_evidence_roles"] == ROLE_SUPPORT_OUTPUT)
            & (out["support_evidence_pair_indices"] == pair_id)
        )
        positions = torch.where(output_mask[0])[0]
        before_colors = out["support_evidence_colors"][0, positions].clone()
        before_rows = out["support_evidence_rows"][0, positions].clone()
        before_cols = out["support_evidence_cols"][0, positions].clone()
        if len(positions) > 1:
            seed = stable_seed_int(f"R5A4B_OUTPUT_SCRAMBLE:{task_id}:{query_pair_index}:{pair_id}")
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(positions), generator=generator)
            if bool(torch.equal(perm, torch.arange(len(positions)))) and len(positions) > 1:
                perm = torch.roll(perm, shifts=1)
            out["support_evidence_colors"][0, positions] = before_colors[perm]
        after_colors = out["support_evidence_colors"][0, positions].clone()
        ledger.append(
            {
                "source_support_pair_id": pair_id,
                "output_shape": [int(before_rows.max().item()) + 1 if len(positions) else 0, int(before_cols.max().item()) + 1 if len(positions) else 0],
                "cell_count": int(len(positions)),
                "coordinate_sha256_before": hashlib.sha256(json.dumps(list(zip(before_rows.tolist(), before_cols.tolist())), sort_keys=True).encode()).hexdigest(),
                "coordinate_sha256_after": hashlib.sha256(json.dumps(list(zip(out["support_evidence_rows"][0, positions].tolist(), out["support_evidence_cols"][0, positions].tolist())), sort_keys=True).encode()).hexdigest(),
                "color_histogram_before": {str(k): v for k, v in sorted(Counter(before_colors.tolist()).items())},
                "color_histogram_after": {str(k): v for k, v in sorted(Counter(after_colors.tolist()).items())},
                "shape_mask_coordinates_histogram_preserved": True,
            }
        )
    return out, {"scramble_rows": ledger}


def move_forward_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_inputs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "support_evidence_colors": batch["support_evidence_colors"],
        "support_evidence_rows": batch["support_evidence_rows"],
        "support_evidence_cols": batch["support_evidence_cols"],
        "support_evidence_roles": batch["support_evidence_roles"],
        "support_evidence_pair_indices": batch["support_evidence_pair_indices"],
        "support_evidence_mask": batch["support_evidence_mask"],
        "query_inputs": batch["query_inputs"],
        "query_input_mask": batch["query_input_mask"],
    }


def query_target_tensors(target_batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    target = target_batch["query_targets"][0].detach().cpu().view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    query = target_batch["query_inputs"][0].detach().cpu().view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    target_h = int(target_batch["query_output_heights"][0].item())
    target_w = int(target_batch["query_output_widths"][0].item())
    query_h = int(target_batch["query_input_heights"][0].item())
    query_w = int(target_batch["query_input_widths"][0].item())
    return query, target, query_h, query_w, target_h, target_w


def score_outputs(
    *,
    condition: str,
    episode_index: int,
    episode: ArcEpisode,
    outputs: dict[str, torch.Tensor],
    target_batch: dict[str, Any],
    loss: torch.Tensor,
    components: dict[str, torch.Tensor],
    inference_seconds: float,
    construction_metadata: dict[str, Any],
) -> dict[str, Any]:
    query, target, query_h, query_w, target_h, target_w = query_target_tensors(target_batch)
    color_probs = outputs["color_probabilities"][0].detach().cpu()
    dense_argmax = color_probs.argmax(dim=-1).to(torch.long)
    dense_grid = dense_argmax.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    pred_h = int(outputs["height_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    pred_w = int(outputs["width_logits"].argmax(dim=-1)[0].detach().cpu().item()) + 1
    pred_shape = [pred_h, pred_w]
    same_shape = query_h == target_h and query_w == target_w

    entries = []
    changed_entries = []
    unchanged_entries = []
    all_correct = 0
    all_nll = 0.0
    changed_correct = 0
    changed_nll = 0.0
    unchanged_correct = 0
    unchanged_nll = 0.0
    predicted_change_count = 0
    true_positive_change_count = 0
    false_positive_change_count = 0
    copy_identity_changed_correct = 0
    copy_identity_unchanged_correct = 0
    for row in range(target_h):
        for col in range(target_w):
            flat = row * MAX_GRID_WIDTH + col
            target_color = int(target[row, col].item())
            query_valid = row < query_h and col < query_w
            query_color = int(query[row, col].item()) if query_valid else None
            pred_color = int(dense_grid[row, col].item())
            changed = (not query_valid) or query_color != target_color
            predicted_change = (not query_valid) or pred_color != query_color
            correct = pred_color == target_color
            prob = float(color_probs[flat, target_color].clamp_min(1e-12).item())
            nll = -math.log(prob)
            all_correct += int(correct)
            all_nll += nll
            predicted_change_count += int(predicted_change)
            if changed and predicted_change:
                true_positive_change_count += 1
            if not changed and predicted_change:
                false_positive_change_count += 1
            entry = {
                "row": row,
                "col": col,
                "target_color": target_color,
                "query_color": query_color,
                "query_valid_at_target_coordinate": query_valid,
                "dense_predicted_color": pred_color,
                "target_color_probability": prob,
                "target_color_nll": nll,
                "changed_cell": changed,
                "predicted_change": predicted_change,
                "correct_color": correct,
            }
            entries.append(entry)
            if changed:
                changed_entries.append(entry)
                changed_correct += int(correct)
                changed_nll += nll
                copy_identity_changed_correct += 0
            else:
                unchanged_entries.append(entry)
                unchanged_correct += int(correct)
                unchanged_nll += nll
                copy_identity_unchanged_correct += 1
    changed_count = len(changed_entries)
    unchanged_count = len(unchanged_entries)
    precision = true_positive_change_count / predicted_change_count if predicted_change_count else 0.0
    recall = true_positive_change_count / changed_count if changed_count else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall > 0 else (0.0 if changed_count else None)
    exact_shape = pred_h == target_h and pred_w == target_w
    return {
        "condition": condition,
        "episode_index": episode_index,
        "task_id": episode.task_id,
        "query_pair_index": episode.query_pair_index,
        "support_pair_indices": [pair.pair_index for pair in episode.supports],
        "query_shape": [query_h, query_w],
        "target_shape": [target_h, target_w],
        "predicted_shape": pred_shape,
        "same_shape_regime": same_shape,
        "exact_shape_correct": exact_shape,
        "exact_full_grid_correct": bool(exact_shape and all_correct == len(entries)),
        "dense_cell_accuracy": all_correct / max(1, len(entries)),
        "dense_color_nll": all_nll / max(1, len(entries)),
        "total_loss": float(loss.detach().cpu().item()),
        "color_nll": float(components["color_loss"].detach().cpu().item()),
        "height_loss": float(components["height_loss"].detach().cpu().item()),
        "width_loss": float(components["width_loss"].detach().cpu().item()),
        "target_rectangle_cell_count": len(entries),
        "target_rectangle_color_correct": all_correct,
        "changed_cell_count": changed_count,
        "unchanged_cell_count": unchanged_count,
        "changed_cell_color_accuracy": changed_correct / changed_count if changed_count else None,
        "changed_cell_color_nll": changed_nll / changed_count if changed_count else None,
        "predicted_change_precision": precision,
        "predicted_change_recall": recall,
        "predicted_change_f1": f1,
        "correct_changed_color_rate": changed_correct / changed_count if changed_count else None,
        "unchanged_cell_color_accuracy": unchanged_correct / unchanged_count if unchanged_count else None,
        "unchanged_cell_color_nll": unchanged_nll / unchanged_count if unchanged_count else None,
        "predicted_change_count": predicted_change_count,
        "true_positive_predicted_change_count": true_positive_change_count,
        "false_positive_predicted_change_count": false_positive_change_count,
        "changed_cell_color_correct": changed_correct,
        "unchanged_cell_color_correct": unchanged_correct,
        "target_changed_colors": [entry["target_color"] for entry in changed_entries],
        "identity_copy_changed_cell_accuracy": 0.0 if changed_count else None,
        "identity_copy_unchanged_cell_accuracy": copy_identity_unchanged_correct / unchanged_count if unchanged_count else None,
        "dense_argmax_canvas": dense_grid.tolist(),
        "dense_argmax_canvas_sha256": tensor_digest(dense_argmax),
        "dense_prediction_saved_before_target_loading": True,
        "target_coordinate_scoring_entries": entries,
        "dense_target_coordinate_color_scoring_note": DENSE_SCORING_NOTE,
        "decoded_grid_sha256": hashlib.sha256(json.dumps(dense_grid[:pred_h, :pred_w].tolist(), sort_keys=True).encode()).hexdigest(),
        "prediction_digest": tensor_digest(dense_argmax),
        "inference_seconds": inference_seconds,
        **construction_metadata,
    }


def prepare_condition_batch(condition: str, episode: ArcEpisode) -> tuple[dict[str, Any], dict[str, Any], bool]:
    batch = forward_only_batch(episode)
    if condition == TRAINED_FULL or condition == FRESH:
        return batch, {"condition_construction": "normal_visible_supports"}, False
    if condition == QUERY_ONLY:
        return dummy_support_batch(episode, "A"), {"condition_construction": "DUMMY_A_with_disable_support_evidence"}, True
    if condition == PAIRING_BROKEN:
        return rotate_output_pairing(batch) + (False,)
    if condition == INPUTS_ZEROED:
        return zero_support_inputs(batch) + (False,)
    if condition == GEOMETRY_SCRAMBLED:
        return scramble_output_geometry(batch, task_id=episode.task_id, query_pair_index=episode.query_pair_index) + (False,)
    if condition == ZEROED_OUTPUTS:
        return zero_support_outputs(batch) + (False,)
    raise ValueError(f"unknown condition {condition}")


def run_model_condition(
    *,
    condition: str,
    model: Any,
    episodes: list[ArcEpisode],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    construction = []
    firewall = []
    pairing_or_scramble = []
    model.eval()
    with torch.no_grad():
        for episode_index, episode in enumerate(episodes):
            batch, metadata, disable_support = prepare_condition_batch(condition, episode)
            construction.append(
                {
                    "condition": condition,
                    "episode_index": episode_index,
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    "target_available_before_forward": False,
                    **metadata,
                }
            )
            if condition == PAIRING_BROKEN:
                pairing_or_scramble.append(
                    {
                        "condition": condition,
                        "episode_index": episode_index,
                        "task_id": episode.task_id,
                        "query_pair_index": episode.query_pair_index,
                        **metadata,
                    }
                )
            if condition == GEOMETRY_SCRAMBLED:
                pairing_or_scramble.append(
                    {
                        "condition": condition,
                        "episode_index": episode_index,
                        "task_id": episode.task_id,
                        "query_pair_index": episode.query_pair_index,
                        **metadata,
                    }
                )
            device_batch = move_forward_batch(batch, device)
            start = time.perf_counter()
            outputs = model(**forward_inputs(device_batch), disable_support_evidence=disable_support)
            inference_seconds = time.perf_counter() - start
            dense_digest_before_target = tensor_digest(outputs["color_probabilities"][0].argmax(dim=-1).detach().cpu())
            target_batch = target_batch_after_forward(episode)
            target_device_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in target_batch.items()}
            loss, components = factorized_shape_color_loss(outputs, target_device_batch, DEFAULT_SHAPE_LOSS_WEIGHT)
            records.append(
                score_outputs(
                    condition=condition,
                    episode_index=episode_index,
                    episode=episode,
                    outputs=outputs,
                    target_batch=target_batch,
                    loss=loss,
                    components=components,
                    inference_seconds=inference_seconds,
                    construction_metadata=metadata,
                )
            )
            firewall.append(
                {
                    "condition": condition,
                    "episode_index": episode_index,
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    "sequence": [
                        "load query input and visible support pairs",
                        "construct condition-specific forward tensors",
                        "execute forward",
                        "decode dense canvas, predicted shape, and cropped output",
                        "load held-out target after inference",
                        "score outputs",
                    ],
                    "target_available_before_forward": False,
                    "dense_prediction_sha256_before_target_loading": dense_digest_before_target,
                    "hidden_target_firewall_pass": True,
                }
            )
    return records, construction, firewall, pairing_or_scramble


def compare_output_subset(outputs_a: dict[str, torch.Tensor], outputs_b: dict[str, torch.Tensor]) -> dict[str, bool]:
    dense_a = outputs_a["color_logits"].detach().cpu()
    dense_b = outputs_b["color_logits"].detach().cpu()
    decoded_a = dense_a.argmax(dim=-1)
    decoded_b = dense_b.argmax(dim=-1)
    return {
        "height_logits_equal": bool(torch.equal(outputs_a["height_logits"].detach().cpu(), outputs_b["height_logits"].detach().cpu())),
        "width_logits_equal": bool(torch.equal(outputs_a["width_logits"].detach().cpu(), outputs_b["width_logits"].detach().cpu())),
        "dense_color_logits_equal": bool(torch.equal(dense_a, dense_b)),
        "decoded_grid_equal": bool(torch.equal(decoded_a, decoded_b)),
        "predicted_shape_equal": bool(
            torch.equal(outputs_a["height_logits"].argmax(dim=-1).detach().cpu(), outputs_b["height_logits"].argmax(dim=-1).detach().cpu())
            and torch.equal(outputs_a["width_logits"].argmax(dim=-1).detach().cpu(), outputs_b["width_logits"].argmax(dim=-1).detach().cpu())
        ),
    }


def query_only_dummy_invariance(model: Any, episodes: list[ArcEpisode], device: torch.device) -> dict[str, Any]:
    selected = sorted(
        episodes,
        key=lambda ep: hashlib.sha256(f"R5A4B_DUMMY:{ep.task_id}:{ep.query_pair_index}".encode()).hexdigest(),
    )[:20]
    rows = []
    model.eval()
    with torch.no_grad():
        for idx, episode in enumerate(selected):
            a = move_forward_batch(dummy_support_batch(episode, "A"), device)
            b = move_forward_batch(dummy_support_batch(episode, "B"), device)
            out_a = model(**forward_inputs(a), disable_support_evidence=True)
            out_b = model(**forward_inputs(b), disable_support_evidence=True)
            checks = compare_output_subset(out_a, out_b)
            rows.append(
                {
                    "audit_index": idx,
                    "task_id": episode.task_id,
                    "query_pair_index": episode.query_pair_index,
                    **checks,
                    "all_equal": all(checks.values()),
                }
            )
    payload = {
        "status": "QUERY_ONLY_BYPASS_VALID" if all(row["all_equal"] for row in rows) else "QUERY_ONLY_BYPASS_INVALID",
        "fold_count": len(rows),
        "rows": rows,
    }
    write_json(ARTIFACT_DIR / "query_only_dummy_invariance.json", payload)
    return payload


def aggregate_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    total_target_cells = sum(row["target_rectangle_cell_count"] for row in records)
    total_target_correct = sum(row["target_rectangle_color_correct"] for row in records)
    return {
        "fold_count": count,
        "task_count": len({row["task_id"] for row in records}),
        "exact_full_grid_accuracy": sum(row["exact_full_grid_correct"] for row in records) / count,
        "exact_shape_accuracy": sum(row["exact_shape_correct"] for row in records) / count,
        "same_shape_exact_shape_accuracy": safe_rate([row["exact_shape_correct"] for row in records if row["same_shape_regime"]]),
        "shape_changing_exact_shape_accuracy": safe_rate([row["exact_shape_correct"] for row in records if not row["same_shape_regime"]]),
        "dense_cell_accuracy_inside_true_target_rectangle": total_target_correct / total_target_cells,
        "mean_fold_dense_cell_accuracy_inside_true_target_rectangle": statistics.fmean(row["dense_cell_accuracy"] for row in records),
        "dense_color_nll_inside_true_target_rectangle": statistics.fmean(row["color_nll"] for row in records),
        "mean_fold_dense_color_nll_inside_true_target_rectangle": statistics.fmean(row["dense_color_nll"] for row in records),
        "mean_total_loss": statistics.fmean(row["total_loss"] for row in records),
        "mean_color_nll": statistics.fmean(row["color_nll"] for row in records),
    }


def safe_rate(values: list[Any]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def metric_values(records: list[dict[str, Any]], metric: str, regime: str = "all") -> list[float]:
    rows = records
    if regime == "same_shape":
        rows = [row for row in records if row["same_shape_regime"]]
    elif regime == "shape_changing":
        rows = [row for row in records if not row["same_shape_regime"]]
    values = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def pooled_changed(records: list[dict[str, Any]], regime: str = "all") -> dict[str, Any]:
    rows = records
    if regime == "same_shape":
        rows = [row for row in rows if row["same_shape_regime"]]
    elif regime == "shape_changing":
        rows = [row for row in rows if not row["same_shape_regime"]]
    eligible = [row for row in rows if row["changed_cell_count"] > 0]
    total_changed = sum(row["changed_cell_count"] for row in eligible)
    total_unchanged = sum(row["unchanged_cell_count"] for row in eligible)
    correct_changed = sum(row["changed_cell_color_correct"] for row in eligible)
    correct_unchanged = sum(row["unchanged_cell_color_correct"] for row in eligible)
    changed_nll = sum(row["changed_cell_color_nll"] * row["changed_cell_count"] for row in eligible)
    unchanged_nll = sum(row["unchanged_cell_color_nll"] * row["unchanged_cell_count"] for row in eligible if row["unchanged_cell_color_nll"] is not None)
    predicted_change = sum(row["predicted_change_count"] for row in eligible)
    true_positive = sum(row["true_positive_predicted_change_count"] for row in eligible)
    precision = true_positive / predicted_change if predicted_change else 0.0
    recall = true_positive / total_changed if total_changed else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall > 0 else None
    counts = [row["changed_cell_count"] for row in eligible]
    return {
        "eligible_changed_cell_fold_count": len(eligible),
        "total_changed_cell_count": total_changed,
        "median_changed_cell_count_per_eligible_fold": safe_median(counts),
        "minimum_changed_cell_count": min(counts) if counts else None,
        "maximum_changed_cell_count": max(counts) if counts else None,
        "pooled_descriptive_only": {
            "changed_cell_color_accuracy": correct_changed / total_changed if total_changed else None,
            "changed_cell_color_nll": changed_nll / total_changed if total_changed else None,
            "predicted_change_precision": precision,
            "predicted_change_recall": recall,
            "predicted_change_f1": f1 if f1 is not None else 0.0,
            "correct_changed_color_rate": correct_changed / total_changed if total_changed else None,
            "unchanged_cell_color_accuracy": correct_unchanged / total_unchanged if total_unchanged else None,
            "unchanged_cell_color_nll": unchanged_nll / total_unchanged if total_unchanged else None,
        },
    }


def macro_changed(records: list[dict[str, Any]], regime: str = "all") -> dict[str, Any]:
    rows = records
    if regime == "same_shape":
        rows = [row for row in rows if row["same_shape_regime"]]
    elif regime == "shape_changing":
        rows = [row for row in rows if not row["same_shape_regime"]]
    eligible = [row for row in rows if row["changed_cell_count"] > 0]
    metrics = (
        "changed_cell_color_accuracy",
        "changed_cell_color_nll",
        "predicted_change_precision",
        "predicted_change_recall",
        "predicted_change_f1",
        "correct_changed_color_rate",
        "unchanged_cell_color_accuracy",
        "unchanged_cell_color_nll",
    )
    return {
        metric: {
            "mean": safe_mean([float(row[metric]) for row in eligible if row.get(metric) is not None]),
            "median": safe_median([float(row[metric]) for row in eligible if row.get(metric) is not None]),
            "eligible_fold_count": len([row for row in eligible if row.get(metric) is not None]),
            "primary_inferential_statistic": True,
        }
        for metric in metrics
    }


def changed_cell_summary(condition_records: dict[str, list[dict[str, Any]]], regime: str = "all") -> dict[str, Any]:
    return {
        condition: {
            **pooled_changed(records, regime=regime),
            "macro_per_fold": macro_changed(records, regime=regime),
        }
        for condition, records in condition_records.items()
    }


def shape_regime_summary(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = {}
    for condition, records in condition_records.items():
        same = [row for row in records if row["same_shape_regime"]]
        changing = [row for row in records if not row["same_shape_regime"]]
        rows[condition] = {
            "condition": condition,
            "same_shape_fold_count": len(same),
            "same_shape_correct_shape_count": sum(row["exact_shape_correct"] for row in same),
            "same_shape_shape_accuracy": safe_rate([row["exact_shape_correct"] for row in same]),
            "shape_changing_fold_count": len(changing),
            "shape_changing_correct_shape_count": sum(row["exact_shape_correct"] for row in changing),
            "shape_changing_shape_accuracy": safe_rate([row["exact_shape_correct"] for row in changing]),
            "overall_correct_shape_count": sum(row["exact_shape_correct"] for row in records),
            "overall_shape_accuracy": safe_rate([row["exact_shape_correct"] for row in records]),
        }
    return {"by_condition": rows, "note": "Do not interpret overall shape accuracy alone."}


def paired_bootstrap(values: list[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None, "bootstrap_replicates": BOOTSTRAP_REPLICATES}
    generator = torch.Generator().manual_seed(seed)
    data = torch.tensor(values, dtype=torch.float64)
    count = data.numel()
    indices = torch.randint(0, count, (BOOTSTRAP_REPLICATES, count), generator=generator)
    means = data[indices].mean(dim=1).sort().values
    return {
        "count": int(count),
        "mean": float(data.mean().item()),
        "ci95_low": float(means[int(0.025 * (BOOTSTRAP_REPLICATES - 1))].item()),
        "ci95_high": float(means[int(0.975 * (BOOTSTRAP_REPLICATES - 1))].item()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
    }


def fold_map(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["task_id"], int(row["query_pair_index"])): row for row in records}


def comparison_values(full: dict[tuple[str, int], dict[str, Any]], other: dict[tuple[str, int], dict[str, Any]], outcome: str) -> list[float]:
    keys = sorted(set(full) & set(other))
    values = []
    for key in keys:
        left = full[key]
        right = other[key]
        if outcome == "total_loss":
            values.append(right["total_loss"] - left["total_loss"])
        elif outcome == "color_nll":
            values.append(right["color_nll"] - left["color_nll"])
        elif outcome == "dense_cell_accuracy":
            values.append(left["dense_cell_accuracy"] - right["dense_cell_accuracy"])
        elif outcome == "changed_cell_macro_accuracy":
            if left["changed_cell_color_accuracy"] is not None and right["changed_cell_color_accuracy"] is not None:
                values.append(left["changed_cell_color_accuracy"] - right["changed_cell_color_accuracy"])
        elif outcome == "changed_cell_macro_nll":
            if left["changed_cell_color_nll"] is not None and right["changed_cell_color_nll"] is not None:
                values.append(right["changed_cell_color_nll"] - left["changed_cell_color_nll"])
        elif outcome == "predicted_change_macro_f1":
            if left["predicted_change_f1"] is not None and right["predicted_change_f1"] is not None:
                values.append(left["predicted_change_f1"] - right["predicted_change_f1"])
        elif outcome == "shape_accuracy":
            values.append(float(left["exact_shape_correct"]) - float(right["exact_shape_correct"]))
        elif outcome == "same_shape_shape_accuracy":
            if left["same_shape_regime"]:
                values.append(float(left["exact_shape_correct"]) - float(right["exact_shape_correct"]))
        elif outcome == "shape_changing_shape_accuracy":
            if not left["same_shape_regime"]:
                values.append(float(left["exact_shape_correct"]) - float(right["exact_shape_correct"]))
        elif outcome == "decoded_grid_change_rate":
            values.append(float(left["prediction_digest"] != right["prediction_digest"] or left["predicted_shape"] != right["predicted_shape"]))
        else:
            raise ValueError(outcome)
    return values


def paired_comparisons(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    full = fold_map(condition_records[TRAINED_FULL])
    outcomes = (
        "total_loss",
        "color_nll",
        "dense_cell_accuracy",
        "changed_cell_macro_accuracy",
        "changed_cell_macro_nll",
        "predicted_change_macro_f1",
        "shape_accuracy",
        "same_shape_shape_accuracy",
        "shape_changing_shape_accuracy",
        "decoded_grid_change_rate",
    )
    comparisons = {}
    for other_name in (FRESH, QUERY_ONLY, PAIRING_BROKEN, INPUTS_ZEROED, GEOMETRY_SCRAMBLED, ZEROED_OUTPUTS):
        other = fold_map(condition_records[other_name])
        outcome_rows = {}
        for outcome in outcomes:
            values = comparison_values(full, other, outcome)
            outcome_rows[outcome] = paired_bootstrap(
                values,
                seed=BOOTSTRAP_SEED + int(hashlib.sha256(f"{other_name}:{outcome}".encode()).hexdigest()[:6], 16),
            )
        comparisons[f"{TRAINED_FULL}_vs_{other_name}"] = {
            "paired_fold_count": len(set(full) & set(other)),
            "positive_delta_favors": TRAINED_FULL,
            "outcomes": outcome_rows,
        }
    return {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": 0.95,
        "seed": BOOTSTRAP_SEED,
        "comparisons": comparisons,
    }


def dense_ledger(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for condition, records in condition_records.items():
        for row in records:
            rows.append(
                {
                    "condition": condition,
                    "episode_index": row["episode_index"],
                    "task_id": row["task_id"],
                    "query_pair_index": row["query_pair_index"],
                    "query_shape": row["query_shape"],
                    "target_shape": row["target_shape"],
                    "predicted_height": row["predicted_shape"][0],
                    "predicted_width": row["predicted_shape"][1],
                    "dense_argmax_canvas": row["dense_argmax_canvas"],
                    "dense_argmax_canvas_sha256": row["dense_argmax_canvas_sha256"],
                    "dense_prediction_saved_before_target_loading": row["dense_prediction_saved_before_target_loading"],
                    "target_coordinate_scoring_entries": row["target_coordinate_scoring_entries"],
                    "dense_scoring_note": DENSE_SCORING_NOTE,
                }
            )
    return {"fixed_canvas_shape": [MAX_GRID_HEIGHT, MAX_GRID_WIDTH], "dense_scoring_note": DENSE_SCORING_NOTE, "rows": rows}


def condition_metrics(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"by_condition": {condition: aggregate_condition(records) for condition, records in condition_records.items()}}


def target_difficulty(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    base = condition_records[TRAINED_FULL]
    for row in base:
        colors = row["target_changed_colors"]
        if not colors:
            continue
        counts = Counter(colors)
        rows.append(
            {
                "task_id": row["task_id"],
                "query_pair_index": row["query_pair_index"],
                "changed_cell_count": len(colors),
                "distinct_target_colors_among_changed_cells": len(counts),
                "modal_target_changed_color_fraction": max(counts.values()) / len(colors),
                "all_changed_cells_have_one_target_color": len(counts) == 1,
            }
        )
    changed_counts = [row["changed_cell_count"] for row in rows]
    distinct_counts = [row["distinct_target_colors_among_changed_cells"] for row in rows]
    modal = [row["modal_target_changed_color_fraction"] for row in rows]
    return {
        "analysis_name": "TARGET_CHANGED_REGION_DIFFICULTY_DIAGNOSTICS",
        "diagnostic_only": True,
        "not_an_oracle_baseline": True,
        "not_a_fair_predictive_comparison": True,
        "not_used_in_verdict": True,
        "not_used_for_checkpoint_or_case_selection": True,
        "eligible_changed_cell_fold_count": len(rows),
        "pooled": {
            "total_changed_cell_count": sum(changed_counts),
            "single_color_changed_region_fold_rate": safe_mean([float(row["all_changed_cells_have_one_target_color"]) for row in rows]),
            "weighted_modal_target_changed_color_fraction": sum(row["modal_target_changed_color_fraction"] * row["changed_cell_count"] for row in rows) / sum(changed_counts),
        },
        "macro": {
            "mean_changed_cells_per_eligible_fold": safe_mean(changed_counts),
            "median_changed_cells_per_eligible_fold": safe_median(changed_counts),
            "mean_distinct_changed_colors_per_eligible_fold": safe_mean(distinct_counts),
            "median_distinct_changed_colors_per_eligible_fold": safe_median(distinct_counts),
            "mean_modal_target_changed_color_fraction": safe_mean(modal),
            "median_modal_target_changed_color_fraction": safe_median(modal),
        },
        "per_fold": rows,
    }


def copy_paint_diagnostics(condition_records: dict[str, list[dict[str, Any]]], outputs_by_key: dict[tuple[str, int], dict[str, torch.Tensor]]) -> dict[str, Any]:
    rows = []
    for row in condition_records[TRAINED_FULL]:
        key = (row["task_id"], int(row["query_pair_index"]))
        outputs = outputs_by_key[key]
        copy_dist = outputs["copy_distribution"][0].detach().cpu()
        paint_dist = outputs["paint_distribution"][0].detach().cpu()
        final_dist = outputs["color_probabilities"][0].detach().cpu()
        copy_gate = outputs["copy_gate"][0, :, 0].detach().cpu()
        copy_entropy = outputs["copy_entropy"][0].detach().cpu()
        paint_entropy = outputs["paint_entropy"][0].detach().cpu()
        changed_entries = [entry for entry in row["target_coordinate_scoring_entries"] if entry["changed_cell"]]
        unchanged_entries = [entry for entry in row["target_coordinate_scoring_entries"] if not entry["changed_cell"]]

        def score(entries: list[dict[str, Any]], dist: torch.Tensor) -> float | None:
            if not entries:
                return None
            return sum(int(int(dist[entry["row"] * MAX_GRID_WIDTH + entry["col"]].argmax().item()) == entry["target_color"]) for entry in entries) / len(entries)

        def mean_tensor(entries: list[dict[str, Any]], tensor: torch.Tensor) -> float | None:
            if not entries:
                return None
            return statistics.fmean(float(tensor[entry["row"] * MAX_GRID_WIDTH + entry["col"]].item()) for entry in entries)

        rows.append(
            {
                "task_id": row["task_id"],
                "query_pair_index": row["query_pair_index"],
                "changed_cell_count": len(changed_entries),
                "unchanged_cell_count": len(unchanged_entries),
                "changed": {
                    "copy_only_argmax_color_accuracy": score(changed_entries, copy_dist),
                    "paint_only_argmax_color_accuracy": score(changed_entries, paint_dist),
                    "final_mixed_argmax_color_accuracy": score(changed_entries, final_dist),
                    "mean_copy_gate": mean_tensor(changed_entries, copy_gate),
                    "mean_copy_entropy": mean_tensor(changed_entries, copy_entropy),
                    "mean_paint_entropy": mean_tensor(changed_entries, paint_entropy),
                },
                "unchanged": {
                    "copy_only_argmax_color_accuracy": score(unchanged_entries, copy_dist),
                    "paint_only_argmax_color_accuracy": score(unchanged_entries, paint_dist),
                    "final_mixed_argmax_color_accuracy": score(unchanged_entries, final_dist),
                    "mean_copy_gate": mean_tensor(unchanged_entries, copy_gate),
                    "mean_copy_entropy": mean_tensor(unchanged_entries, copy_entropy),
                    "mean_paint_entropy": mean_tensor(unchanged_entries, paint_entropy),
                },
            }
        )
    changed_gates = [row["changed"]["mean_copy_gate"] for row in rows if row["changed"]["mean_copy_gate"] is not None and row["unchanged"]["mean_copy_gate"] is not None]
    unchanged_gates = [row["unchanged"]["mean_copy_gate"] for row in rows if row["changed"]["mean_copy_gate"] is not None and row["unchanged"]["mean_copy_gate"] is not None]
    gate_deltas = [c - u for c, u in zip(changed_gates, unchanged_gates)]

    def aggregate(section: str) -> dict[str, Any]:
        metrics = (
            "copy_only_argmax_color_accuracy",
            "paint_only_argmax_color_accuracy",
            "final_mixed_argmax_color_accuracy",
            "mean_copy_gate",
            "mean_copy_entropy",
            "mean_paint_entropy",
        )
        return {
            metric: safe_mean([row[section][metric] for row in rows if row[section][metric] is not None])
            for metric in metrics
        }

    return {
        "condition": TRAINED_FULL,
        "uses_unmodified_model_returned_tensors": ["copy_distribution", "paint_distribution", "copy_gate", "color_probabilities", "copy_entropy", "paint_entropy"],
        "changed": aggregate("changed"),
        "unchanged": aggregate("unchanged"),
        "copy_gate_changed_minus_unchanged_macro_paired": paired_bootstrap(gate_deltas, seed=BOOTSTRAP_SEED + 777),
        "per_fold": rows,
    }


def identity_copy_reference(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = condition_records[TRAINED_FULL]
    changed_acc = [row["identity_copy_changed_cell_accuracy"] for row in base if row["identity_copy_changed_cell_accuracy"] is not None]
    unchanged_acc = [row["identity_copy_unchanged_cell_accuracy"] for row in base if row["identity_copy_unchanged_cell_accuracy"] is not None]
    return {
        "reference": "identity-copy query colors only",
        "descriptive_non_transform_baseline": True,
        "identity_copy_changed_cell_accuracy_is_zero_by_construction": True,
        "changed_cell_accuracy_macro": safe_mean(changed_acc),
        "unchanged_cell_accuracy_macro": safe_mean(unchanged_acc),
        "changed_cell_accuracy_pooled": 0.0,
    }


def r5a4a_replay_check(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    old = json.loads((R5A4A_DIR / "trained_full_support_metrics.json").read_text(encoding="utf-8"))["aggregate"]
    new = aggregate_condition(condition_records[TRAINED_FULL])
    comparisons = {
        "fold_count": [old["fold_count"], new["fold_count"]],
        "task_count": [old["task_count"], new["task_count"]],
        "exact_full_grid_accuracy": [old["exact_full_grid_accuracy"], new["exact_full_grid_accuracy"]],
        "exact_output_shape_accuracy": [old["exact_output_shape_accuracy"], new["exact_shape_accuracy"]],
        "cell_accuracy_inside_true_target_rectangle": [old["cell_accuracy_inside_true_target_rectangle"], new["dense_cell_accuracy_inside_true_target_rectangle"]],
        "mean_color_negative_log_likelihood_inside_true_target_rectangle": [old["mean_color_negative_log_likelihood_inside_true_target_rectangle"], new["dense_color_nll_inside_true_target_rectangle"]],
        "mean_total_existing_model_loss": [old["mean_total_existing_model_loss"], new["mean_total_loss"]],
    }
    failures = []
    for name, pair in comparisons.items():
        if isinstance(pair[0], (int, float)):
            if abs(float(pair[0]) - float(pair[1])) > 1e-8:
                failures.append(name)
        elif pair[0] != pair[1]:
            failures.append(name)
    payload = {
        "status": "R5A4A_FULL_SUPPORT_REPLAY_PASS" if not failures else "R5A4A_FULL_SUPPORT_REPLAY_MISMATCH",
        "comparisons": comparisons,
        "failures": failures,
    }
    write_json(ARTIFACT_DIR / "r5a4a_full_support_replay.json", payload)
    return payload


def verdict_from_comparisons(
    *,
    frozen_ok: bool,
    dummy_ok: bool,
    firewall_ok: bool,
    replay_ok: bool,
    determinism_ok: bool,
    full_predicted_change_f1: float | None,
    intervals: dict[str, Any],
) -> tuple[str, str]:
    if not (frozen_ok and dummy_ok and firewall_ok and replay_ok and determinism_ok):
        return "INVALID_OR_INCONCLUSIVE", "D. FIX_AUDIT_ONLY"

    comps = intervals["comparisons"]

    def positive_ci(comp: str, outcome: str) -> bool:
        row = comps[comp]["outcomes"][outcome]
        return row["ci95_low"] is not None and row["ci95_low"] > 0

    full_vs_fresh = f"{TRAINED_FULL}_vs_{FRESH}"
    full_vs_query = f"{TRAINED_FULL}_vs_{QUERY_ONLY}"
    full_vs_pair = f"{TRAINED_FULL}_vs_{PAIRING_BROKEN}"
    full_vs_geom = f"{TRAINED_FULL}_vs_{GEOMETRY_SCRAMBLED}"

    fresh_changed = positive_ci(full_vs_fresh, "changed_cell_macro_accuracy")
    query_changed = positive_ci(full_vs_query, "changed_cell_macro_accuracy") or positive_ci(full_vs_query, "changed_cell_macro_nll")
    pair_changed = positive_ci(full_vs_pair, "changed_cell_macro_accuracy") or positive_ci(full_vs_pair, "changed_cell_macro_nll")
    geom_changed = positive_ci(full_vs_geom, "changed_cell_macro_accuracy") or positive_ci(full_vs_geom, "changed_cell_macro_nll")
    f1_signal = positive_ci(full_vs_fresh, "predicted_change_macro_f1")
    if (
        fresh_changed
        and query_changed
        and pair_changed
        and geom_changed
        and f1_signal
        and full_predicted_change_f1 is not None
        and full_predicted_change_f1 > 0
    ):
        return "SUPPORT_SEMANTIC_NONCOPY_SIGNAL", "B. REPLICATE_TWO_MORE_SEEDS"
    if not fresh_changed or not pair_changed or not geom_changed or not f1_signal:
        query_or_zero = query_changed or positive_ci(f"{TRAINED_FULL}_vs_{ZEROED_OUTPUTS}", "changed_cell_macro_accuracy") or positive_ci(f"{TRAINED_FULL}_vs_{ZEROED_OUTPUTS}", "changed_cell_macro_nll")
        if query_or_zero and (not pair_changed or not geom_changed):
            return "SUPPORT_DEPENDENT_BUT_SEMANTICALLY_UNRESOLVED", "C. ARCHIVE_AS_UNRESOLVED"
        return "SHAPE_COPY_OR_MARGINAL_PRIOR_ONLY", "A. ARCHIVE_NEURAL_VRM"
    return "INVALID_OR_INCONCLUSIVE", "D. FIX_AUDIT_ONLY"


def selected_case_keys(condition_records: dict[str, list[dict[str, Any]]]) -> list[tuple[str, int, str]]:
    maps = {condition: fold_map(records) for condition, records in condition_records.items()}
    full = maps[TRAINED_FULL]
    keys = sorted(full)
    selected: list[tuple[str, int, str]] = []
    used = set()

    def add(label: str, ordered: list[tuple[str, int]]) -> None:
        for key in ordered:
            if key not in used:
                selected.append((key[0], key[1], label))
                used.add(key)
                return

    add(
        "largest_trained_vs_fresh_macro_changed_cell_improvement",
        sorted(keys, key=lambda key: -((full[key]["changed_cell_color_accuracy"] or 0.0) - (maps[FRESH][key]["changed_cell_color_accuracy"] or 0.0))),
    )
    add(
        "largest_full_vs_pairing_broken_degradation",
        sorted(keys, key=lambda key: -((full[key]["changed_cell_color_accuracy"] or 0.0) - (maps[PAIRING_BROKEN][key]["changed_cell_color_accuracy"] or 0.0))),
    )
    add(
        "largest_full_vs_geometry_scrambled_degradation",
        sorted(keys, key=lambda key: -((full[key]["changed_cell_color_accuracy"] or 0.0) - (maps[GEOMETRY_SCRAMBLED][key]["changed_cell_color_accuracy"] or 0.0))),
    )
    add(
        "largest_shape_improvement",
        sorted(keys, key=lambda key: -(float(full[key]["exact_shape_correct"]) - float(maps[FRESH][key]["exact_shape_correct"]))),
    )
    hash_order = sorted(keys, key=lambda key: hashlib.sha256(f"R5A4B_CASE:{key[0]}:{key[1]}".encode()).hexdigest())
    for key in hash_order:
        if key not in used:
            selected.append((key[0], key[1], "deterministic_hash_selection"))
            used.add(key)
            if sum(1 for *_rest, label in selected if label == "deterministic_hash_selection") == 3:
                break
    return selected


def grid_text_from_dense(row: dict[str, Any]) -> str:
    height, width = row["predicted_shape"]
    grid = row["dense_argmax_canvas"]
    return "\n".join(" ".join(str(int(cell)) for cell in grid[r][:width]) for r in range(height))


def render_case_cards(episodes: list[ArcEpisode], condition_records: dict[str, list[dict[str, Any]]]) -> str:
    episode_map = {(ep.task_id, ep.query_pair_index): ep for ep in episodes}
    maps = {condition: fold_map(records) for condition, records in condition_records.items()}
    lines = [
        "# R5A-4B representative changed-cell diagnostic case cards",
        "",
        "These deterministic case cards are diagnostics, not proof by example.",
        "",
    ]
    condition_labels = [
        (FRESH, "fresh-init prediction"),
        (TRAINED_FULL, "trained full-support prediction"),
        (QUERY_ONLY, "query-only prediction"),
        (PAIRING_BROKEN, "pairing-broken prediction"),
        (INPUTS_ZEROED, "support-input-zeroed prediction"),
        (GEOMETRY_SCRAMBLED, "geometry-scrambled prediction"),
        (ZEROED_OUTPUTS, "zeroed-output prediction"),
    ]
    for task_id, query_pair_index, label in selected_case_keys(condition_records):
        episode = episode_map[(task_id, query_pair_index)]
        lines.extend([f"## {label}: task {task_id}, held-out pair {query_pair_index}", "", "### visible support pairs", ""])
        for support in episode.supports:
            lines.extend(
                [
                    f"support {support.pair_index} input ({support.input.height}x{support.input.width})",
                    "```text",
                    grid_to_text(support.input),
                    "```",
                    f"support {support.pair_index} output ({support.output.height}x{support.output.width})",
                    "```text",
                    grid_to_text(support.output),
                    "```",
                ]
            )
        lines.extend(
            [
                f"held-out query input ({episode.query.input.height}x{episode.query.input.width})",
                "```text",
                grid_to_text(episode.query.input),
                "```",
                f"hidden target ({episode.query.output.height}x{episode.query.output.width})",
                "```text",
                grid_to_text(episode.query.output),
                "```",
            ]
        )
        for condition, title in condition_labels:
            row = maps[condition][(task_id, query_pair_index)]
            lines.extend(
                [
                    f"### {title}",
                    f"predicted shape: {row['predicted_shape'][0]}x{row['predicted_shape'][1]}",
                    f"changed-cell accuracy: {row['changed_cell_color_accuracy']}",
                    "```text",
                    grid_text_from_dense(row),
                    "```",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def deterministic_payload(condition_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        condition: [
            {
                key: value
                for key, value in row.items()
                if key not in {"inference_seconds"} and not key.endswith("_entries")
            }
            for row in records
        ]
        for condition, records in condition_records.items()
    }


def run_all_conditions(inputs: AuditInputs) -> tuple[dict[str, list[dict[str, Any]]], dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    condition_records = {}
    full_outputs_by_key = {}
    construction_ledger = []
    firewall_rows = []
    pairing_rows = []
    scramble_rows = []
    for condition in CONDITIONS:
        model = inputs.fresh_model if condition == FRESH else inputs.trained_model
        records, construction, firewall, special = run_model_condition(
            condition=condition,
            model=model,
            episodes=inputs.episodes,
            device=inputs.device,
        )
        condition_records[condition] = records
        construction_ledger.extend(construction)
        firewall_rows.extend(firewall)
        if condition == PAIRING_BROKEN:
            pairing_rows.extend(special)
        if condition == GEOMETRY_SCRAMBLED:
            scramble_rows.extend(special)
        if condition == TRAINED_FULL:
            # Re-run one pass for returned tensors needed in copy/paint diagnostics.
            inputs.trained_model.eval()
            with torch.no_grad():
                for episode in inputs.episodes:
                    batch = move_forward_batch(forward_only_batch(episode), inputs.device)
                    outputs = inputs.trained_model(**forward_inputs(batch))
                    full_outputs_by_key[(episode.task_id, int(episode.query_pair_index))] = {
                        key: value.detach().cpu()
                        for key, value in outputs.items()
                        if key in {"copy_distribution", "paint_distribution", "copy_gate", "color_probabilities", "copy_entropy", "paint_entropy"}
                    }
    return condition_records, full_outputs_by_key, construction_ledger, firewall_rows, pairing_rows, scramble_rows


def run_support_semantics_audit(device_arg: str = "auto") -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    before_manifest = protected_source_manifest()
    write_text(ARTIFACT_DIR / "protected_source_manifest_before.txt", render_manifest(before_manifest))
    frozen = verify_frozen_inputs()
    if frozen["status"] != "FROZEN_INPUT_INTEGRITY_PASS":
        report = {"verdict": "INVALID_OR_INCONCLUSIVE", "failure": "FROZEN_INPUT_INTEGRITY_FAIL"}
        write_json(ARTIFACT_DIR / "final_report.json", report)
        return report

    pre_run_commitment = {
        "audit": "R5A-4B ARC Support Semantics and Non-Copy Audit",
        "no_training": True,
        "conditions": list(CONDITIONS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "frozen_input_verification_path": str(ARTIFACT_DIR / "frozen_input_verification.json"),
        "r5a4a_artifact_dir": str(R5A4A_DIR),
        "r5a4b_prerun_patch_artifact_dir": str(R5A4B_PATCH_DIR),
    }
    write_json(ARTIFACT_DIR / "pre_run_commitment.json", pre_run_commitment)

    inputs = load_audit_inputs(device_arg)
    dummy = query_only_dummy_invariance(inputs.trained_model, inputs.episodes, inputs.device)
    if dummy["status"] != "QUERY_ONLY_BYPASS_VALID":
        report = {"verdict": "INVALID_OR_INCONCLUSIVE", "failure": "QUERY_ONLY_BYPASS_INVALID"}
        write_json(ARTIFACT_DIR / "final_report.json", report)
        return report

    condition_records, full_outputs, construction, firewall, pairing, scramble = run_all_conditions(inputs)
    write_json(ARTIFACT_DIR / "condition_construction_ledger.json", {"rows": construction})
    write_json(ARTIFACT_DIR / "support_pairing_rotation_ledger.json", {"rows": pairing})
    write_json(ARTIFACT_DIR / "output_geometry_scramble_ledger.json", {"rows": scramble})
    firewall_payload = {
        "status": "HIDDEN_TARGET_FIREWALL_PASS",
        "rows": firewall,
        "all_conditions_target_unavailable_before_forward": all(not row["target_available_before_forward"] for row in firewall),
    }
    write_json(ARTIFACT_DIR / "hidden_target_firewall.json", firewall_payload)
    write_json(ARTIFACT_DIR / "dense_prediction_scoring_ledger.json", dense_ledger(condition_records))
    metrics = condition_metrics(condition_records)
    write_json(ARTIFACT_DIR / "condition_metrics.json", metrics)
    shape_metrics = shape_regime_summary(condition_records)
    write_json(ARTIFACT_DIR / "shape_regime_metrics.json", shape_metrics)
    changed_metrics = changed_cell_summary(condition_records)
    write_json(ARTIFACT_DIR / "changed_cell_pooled_and_macro_metrics.json", changed_metrics)
    shape_slice_changed = {
        "same_shape": changed_cell_summary(condition_records, regime="same_shape"),
        "shape_changing": changed_cell_summary(condition_records, regime="shape_changing"),
        "slice_note": "same-shape and shape-changing slices are disjoint and exhaustive.",
    }
    write_json(ARTIFACT_DIR / "shape_slice_changed_cell_metrics.json", shape_slice_changed)
    copy_paint = copy_paint_diagnostics(condition_records, full_outputs)
    write_json(ARTIFACT_DIR / "copy_paint_path_diagnostics.json", copy_paint)
    identity = identity_copy_reference(condition_records)
    write_json(ARTIFACT_DIR / "identity_copy_reference_metrics.json", identity)
    difficulty = target_difficulty(condition_records)
    write_json(ARTIFACT_DIR / "target_changed_region_difficulty_diagnostics.json", difficulty)
    intervals = paired_comparisons(condition_records)
    write_json(ARTIFACT_DIR / "paired_bootstrap_intervals.json", intervals)
    write_text(ARTIFACT_DIR / "representative_changed_cell_case_cards.md", render_case_cards(inputs.episodes, condition_records))
    replay = r5a4a_replay_check(condition_records)

    rerun_records, *_ = run_all_conditions(inputs)
    determinism = {
        "status": "DETERMINISM_PASS" if deterministic_payload(condition_records) == deterministic_payload(rerun_records) else "DETERMINISM_FAIL",
        "compared_conditions": list(CONDITIONS),
        "runtime_fields_excluded": True,
    }
    write_json(ARTIFACT_DIR / "determinism_verification.json", determinism)

    after_manifest = protected_source_manifest()
    write_text(ARTIFACT_DIR / "protected_source_manifest_after.txt", render_manifest(after_manifest))
    protected_integrity = {
        "unchanged": before_manifest == after_manifest,
        "before_file_count": before_manifest["file_count"],
        "after_file_count": after_manifest["file_count"],
    }
    full_predicted_change_f1 = changed_metrics[TRAINED_FULL]["macro_per_fold"]["predicted_change_f1"]["mean"]
    verdict, next_action = verdict_from_comparisons(
        frozen_ok=frozen["status"] == "FROZEN_INPUT_INTEGRITY_PASS",
        dummy_ok=dummy["status"] == "QUERY_ONLY_BYPASS_VALID",
        firewall_ok=firewall_payload["status"] == "HIDDEN_TARGET_FIREWALL_PASS",
        replay_ok=replay["status"] == "R5A4A_FULL_SUPPORT_REPLAY_PASS",
        determinism_ok=determinism["status"] == "DETERMINISM_PASS",
        full_predicted_change_f1=full_predicted_change_f1,
        intervals=intervals,
    )
    runtime = {"wall_clock_seconds": time.perf_counter() - start, "device": str(inputs.device)}
    write_json(ARTIFACT_DIR / "runtime_profile.json", runtime)

    report = {
        "audit": "R5A-4B ARC Support Semantics and Non-Copy Audit",
        "frozen_input_verification_status": frozen["status"],
        "query_only_dummy_invariance_status": dummy["status"],
        "baseline_replay_status": replay["status"],
        "hidden_target_firewall_status": firewall_payload["status"],
        "determinism_status": determinism["status"],
        "all_fold_count": len(inputs.episodes),
        "changed_cell_eligible_fold_count": changed_metrics[TRAINED_FULL]["eligible_changed_cell_fold_count"],
        "condition_metrics": metrics,
        "changed_cell_metrics": changed_metrics,
        "shape_regime_metrics": shape_metrics,
        "paired_bootstrap_intervals": intervals,
        "copy_paint_path_diagnostics": {
            "changed": copy_paint["changed"],
            "unchanged": copy_paint["unchanged"],
            "copy_gate_changed_minus_unchanged_macro_paired": copy_paint["copy_gate_changed_minus_unchanged_macro_paired"],
        },
        "identity_copy_reference": identity,
        "target_changed_region_difficulty_diagnostics": {
            key: difficulty[key]
            for key in ("analysis_name", "diagnostic_only", "not_an_oracle_baseline", "not_used_in_verdict", "pooled", "macro")
        },
        "dense_scoring_statement": "Dense target-coordinate color scoring measures color quality at target positions after inference. It does not repair a wrong predicted shape and is not a substitute for exact-grid accuracy.",
        "limitations": [LIMITATION_SENTENCE],
        "interpretation": [INTERPRETATION_SENTENCE],
        "exact_verdict": verdict,
        "mandatory_next_action": next_action,
        "protected_source_integrity": protected_integrity,
        "artifact_dir": str(ARTIFACT_DIR),
    }
    write_json(ARTIFACT_DIR / "final_report.json", report)
    write_text(PROJECT_ROOT / "docs" / "R5A4B_SUPPORT_SEMANTICS_AUDIT.md", render_final_markdown(report))
    return report


def render_final_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# R5A-4B ARC Support Semantics and Non-Copy Audit",
        "",
        "This is an audit of the existing frozen neural VRM line. No training, model change, TTT, adapters, synthetic data, or ARC evaluation/test outputs were used.",
        "",
        f"- frozen input verification: `{report['frozen_input_verification_status']}`",
        f"- query-only dummy invariance: `{report['query_only_dummy_invariance_status']}`",
        f"- baseline replay: `{report['baseline_replay_status']}`",
        f"- exact verdict: `{report['exact_verdict']}`",
        f"- mandatory next action: `{report['mandatory_next_action']}`",
        "",
        "## Dense Scoring",
        "",
        report["dense_scoring_statement"],
        "",
        "## Limitations",
        "",
        LIMITATION_SENTENCE,
        "",
        "## Interpretation",
        "",
        INTERPRETATION_SENTENCE,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen R5A-4B support semantics audit.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(to_jsonable(run_support_semantics_audit(device_arg=args.device)), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
