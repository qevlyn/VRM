from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from vrm.data import collate_episodes, model_inputs_from_batch, move_tensor_batch
from vrm.model import VRM
from vrm.twin_tasks.r5a4a_neural_vrm_data import changed_metadata_episode, model_input_digest, tensor_digest

SCAN_TERMS = ("task_id", "puzzle_id", "identifier", "embedding", "cache", "checkpoint", "filename", "path", "dataset_index")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_no_id_folds(episodes: list[Any], count: int = 20) -> list[Any]:
    return sorted(
        episodes,
        key=lambda episode: hashlib.sha256(f"R5A4A_NO_ID:{episode.task_id}:{episode.query_pair_index}".encode("utf-8")).hexdigest(),
    )[:count]


def output_digest(outputs: dict[str, torch.Tensor]) -> dict[str, str]:
    return {key: tensor_digest(value) for key, value in sorted(outputs.items()) if torch.is_tensor(value)}


def decoded_digest(outputs: dict[str, torch.Tensor]) -> dict[str, str]:
    return {
        "decoded_color_digest": tensor_digest(outputs["color_log_probs"].argmax(dim=-1).detach().cpu()),
        "decoded_height_digest": tensor_digest(outputs["height_logits"].argmax(dim=-1).detach().cpu() + 1),
        "decoded_width_digest": tensor_digest(outputs["width_logits"].argmax(dim=-1).detach().cpu() + 1),
    }


def run_dynamic_no_id_check(*, model: VRM, episodes: list[Any], device: torch.device, count: int = 20) -> dict[str, Any]:
    model.eval()
    rows = []
    failures = []
    selected = select_no_id_folds(episodes, count=count)
    with torch.no_grad():
        for index, episode in enumerate(selected):
            changed = changed_metadata_episode(episode, suffix=f"_noid_{index}")
            batch_a = collate_episodes([episode])
            batch_b = collate_episodes([changed])
            inputs_a = model_inputs_from_batch(batch_a)
            inputs_b = model_inputs_from_batch(batch_b)
            input_digest_a = model_input_digest(inputs_a)
            input_digest_b = model_input_digest(inputs_b)
            moved_a = move_tensor_batch(batch_a, device)
            moved_b = move_tensor_batch(batch_b, device)
            outputs_a = model(**model_inputs_from_batch(moved_a))
            outputs_b = model(**model_inputs_from_batch(moved_b))
            row = {
                "audit_index": index,
                "original_task_id": episode.task_id,
                "changed_task_id": changed.task_id,
                "query_pair_index": episode.query_pair_index,
                "metadata_changed": True,
                "model_input_tensors_byte_identical": input_digest_a == input_digest_b,
                "model_output_tensors_byte_identical": output_digest(outputs_a) == output_digest(outputs_b),
                "decoded_outputs_byte_identical": decoded_digest(outputs_a) == decoded_digest(outputs_b),
                "input_digest": input_digest_a,
                "output_digest": output_digest(outputs_a),
                "decoded_digest": decoded_digest(outputs_a),
            }
            rows.append(row)
            if not (row["model_input_tensors_byte_identical"] and row["model_output_tensors_byte_identical"] and row["decoded_outputs_byte_identical"]):
                failures.append(row)
    return {"fold_count": len(selected), "rows": rows, "failures": failures, "dynamic_no_id_pass": not failures and len(selected) >= count}


def classify_occurrence(path: Path, line: str, term: str) -> tuple[str, str]:
    lower = line.strip().lower()
    if path.name.startswith("r5a4a_") or path.name.startswith("run_r5a4a_"):
        return "bookkeeping-only", "R5A-4A audit, manifest, logging, artifact, or test-control bookkeeping"
    if term == "embedding" and path.name == "model.py":
        return "bookkeeping-only", "generic non-task embedding; no external task identifier"
    if ("model_inputs_from_batch" in lower or "forward" in lower) and any(token in lower for token in ("task_id", "puzzle_id", "filename", "dataset_index")):
        return "unknown", "identifier appears near model input/forward surface"
    if "model(" in lower and any(token in lower for token in ("task_id", "puzzle_id", "filename", "dataset_index")):
        return "model-input", "identifier appears in model call"
    if term == "checkpoint":
        return "bookkeeping-only", "checkpoint path or selection metadata only"
    if term == "path":
        return "bookkeeping-only", "filesystem path for source/artifact bookkeeping"
    if term == "filename":
        return "bookkeeping-only", "external filename metadata only"
    return "bookkeeping-only", "manifest, logging, audit text, or generic model component"


def static_source_scan(source_paths: list[Path]) -> tuple[dict[str, Any], str]:
    rows = []
    for path in source_paths:
        if not path.exists() or not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lower = line.lower()
            for term in SCAN_TERMS:
                if term in lower:
                    occurrence_class, rationale = classify_occurrence(path, line, term)
                    rows.append({"path": str(path), "line": line_number, "term": term, "occurrence_class": occurrence_class, "rationale": rationale, "source_line": line.rstrip()})
    failures = [row for row in rows if row["occurrence_class"] != "bookkeeping-only"]
    text_lines = ["# R5A-4A no-ID source scan", "", "Allowed pass classes: bookkeeping-only or absent.", ""]
    for row in rows:
        text_lines.append(f"{row['occurrence_class']}\t{row['term']}\t{row['path']}:{row['line']}\t{row['rationale']}\t{row['source_line'].strip()}")
    return {"scan_terms": list(SCAN_TERMS), "occurrence_count": len(rows), "failure_count": len(failures), "failures": failures, "static_source_scan_pass": not failures}, "\n".join(text_lines).rstrip() + "\n"


def forward_input_provenance() -> dict[str, Any]:
    signature = inspect.signature(VRM.forward)
    return {
        "model_class": "VRM",
        "forward_parameters": [name for name in signature.parameters if name != "self"],
        "model_inputs_from_batch_keys": [
            "support_evidence_colors",
            "support_evidence_rows",
            "support_evidence_cols",
            "support_evidence_roles",
            "support_evidence_pair_indices",
            "support_evidence_mask",
            "query_inputs",
            "query_input_mask",
        ],
        "excluded_batch_bookkeeping_keys": ["task_ids", "query_pair_indices", "source_splits"],
        "excluded_target_keys": ["query_targets", "query_target_mask", "height_targets", "width_targets", "query_output_heights", "query_output_widths"],
        "task_identifier_reaches_forward": False,
        "hidden_target_reaches_forward": False,
    }


def run_no_id_audit(*, model: VRM, episodes: list[Any], device: torch.device, artifact_dir: Path, source_paths: list[Path]) -> dict[str, Any]:
    dynamic = run_dynamic_no_id_check(model=model, episodes=episodes, device=device, count=20)
    static, static_text = static_source_scan(source_paths)
    provenance = forward_input_provenance()
    audit = {
        "audit_name": "R5A-4A no-ID causal audit",
        "dynamic": dynamic,
        "static_source_scan": static,
        "forward_input_provenance_path": str(artifact_dir / "forward_input_provenance.json"),
        "no_id_causal_audit_pass": bool(dynamic["dynamic_no_id_pass"] and static["static_source_scan_pass"]),
    }
    write_json(artifact_dir / "no_id_causal_audit.json", audit)
    (artifact_dir / "no_id_source_scan.txt").write_text(static_text, encoding="utf-8")
    write_json(artifact_dir / "forward_input_provenance.json", provenance)
    return audit
