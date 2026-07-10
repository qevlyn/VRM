from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, file_sha256, read_json, write_json, write_text


PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT_DIR = R6 / "reports" / PREFIX
REPORT = REPORT_DIR / "R6A_PHASE2_SUPPORT_PERMUTATION_FIRST_DIVERGENCE_CERTIFICATE_V8_REPORT.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compare_manifest(before: dict[str, Any]) -> dict[str, Any]:
    rows = []
    all_match = True
    for row in before["files"]:
        path = PROJECT_ROOT / row["path"]
        exists = path.exists()
        current = {
            "path": row["path"],
            "exists_after": exists,
            "size_bytes_after": path.stat().st_size if exists else None,
            "mtime_ns_after": path.stat().st_mtime_ns if exists else None,
            "sha256_after": file_sha256(path) if exists and path.is_file() else None,
        }
        matches = exists and current["size_bytes_after"] == row["size_bytes"] and current["sha256_after"] == row["sha256"]
        rows.append({**row, **current, "matches_before": matches})
        all_match = all_match and matches
    return {"created_utc": utc_now(), "file_count": len(rows), "all_match_before": all_match, "files": rows}


def full_phase1_integrity() -> dict[str, Any]:
    original_path = R6 / "manifests" / "protected_paths_before_phase1.json"
    original = read_json(original_path)
    result = compare_manifest(original)
    result["original_manifest_path"] = str(original_path)
    result["original_protected_file_count"] = original["file_count"]
    return result


def collect_preexisting_phase2_paths() -> dict[str, Any]:
    entries = []
    for path in R6.rglob("*"):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        parts = path.relative_to(R6).parts
        is_phase2 = any(part.startswith("r6a_phase2_") for part in parts)
        if not is_phase2:
            continue
        if PREFIX in parts or path.name.startswith(PREFIX):
            continue
        stat = path.stat()
        entries.append(
            {
                "path": rel,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": stat.st_size if path.is_file() else None,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    entries.sort(key=lambda row: row["path"])
    return {"created_utc": utc_now(), "entry_count": len(entries), "file_count": sum(1 for row in entries if row["is_file"]), "entries": entries}


def compare_preexisting_phase2(before: dict[str, Any]) -> dict[str, Any]:
    after = collect_preexisting_phase2_paths()
    before_by_path = {row["path"]: row for row in before["entries"]}
    after_by_path = {row["path"]: row for row in after["entries"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = []
    for path in sorted(set(before_by_path) & set(after_by_path)):
        if before_by_path[path] != after_by_path[path]:
            changed.append({"path": path, "before": before_by_path[path], "after": after_by_path[path]})
    return {"created_utc": utc_now(), "pass": not added and not removed and not changed, "added": added, "removed": removed, "changed": changed, "after_entry_count": after["entry_count"], "after_file_count": after["file_count"]}


def prior_v4_summary() -> dict[str, Any]:
    v4_artifact = R6 / "artifacts" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4"
    v4_manifest = R6 / "manifests" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4"
    v4_report = R6 / "reports" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4" / "R6A_PHASE2_ROLE_PRESERVING_PSTRICT_CLOSURE_V4_REPORT.md"
    files = [
        v4_report,
        v4_artifact / "closure_results.json",
        v4_artifact / "prior_pstrict_scope_classification.json",
        v4_artifact / "noise_tape_runtime_trace.json",
        v4_artifact / "snapshot_restore_audit.json",
        R6 / "reports" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4" / "integrity_scope_reconciliation.json",
        v4_manifest / "preflight_integrity_before.json",
        v4_manifest / "preflight_integrity_after.json",
        v4_manifest / "write_ledger.json",
    ]
    rows = []
    for path in files:
        rows.append({"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None, "sha256": file_sha256(path) if path.exists() and path.is_file() else None})
    closure = read_json(v4_artifact / "closure_results.json") if (v4_artifact / "closure_results.json").exists() else {"tasks": []}
    prior = read_json(v4_artifact / "prior_pstrict_scope_classification.json") if (v4_artifact / "prior_pstrict_scope_classification.json").exists() else {}
    reconciliation = read_json(R6 / "reports" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4" / "integrity_scope_reconciliation.json") if (R6 / "reports" / "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4" / "integrity_scope_reconciliation.json").exists() else {}
    payload = {
        "created_utc": utc_now(),
        "read_only_v4_files": rows,
        "v4_final_labels": {row["task_id"]: row["final_label"] for row in closure.get("tasks", [])},
        "v4_prior_pstrict_scope_classification": prior.get("prior_pstrict_scope_classification"),
        "v4_701_295_reconciliation_status": {
            "count_701": reconciliation.get("count_701"),
            "count_295": reconciliation.get("count_295"),
            "is_295_subset_of_701": reconciliation.get("is_295_subset_of_701"),
        },
        "phase1_verdict_remains": "PAIRING_SENSITIVE_QUERY_PILOT",
        "read_only_summary_only": True,
    }
    write_json(ARTIFACT / "prior_v4_evidence_readonly_summary.json", payload)
    return payload


def run_child() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(ARTIFACT / f"{PREFIX}_tmp")
    (ARTIFACT / f"{PREFIX}_tmp").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(R6 / "scripts" / f"{PREFIX}_child.py")]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=3600)
    (ARTIFACT / f"{PREFIX}_child_stdout.log").write_text(proc.stdout, encoding="utf-8")
    summary = read_json(ARTIFACT / "child_summary.json") if (ARTIFACT / "child_summary.json").exists() else {}
    summary["child_returncode"] = proc.returncode
    summary["child_stdout_tail"] = proc.stdout[-4000:]
    write_json(ARTIFACT / "child_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError("child failed")
    return summary


def update_write_ledger() -> dict[str, Any]:
    roots = [MANIFEST, ARTIFACT, REPORT_DIR]
    files = []
    for root in roots:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    for path in [R6 / "scripts" / f"{PREFIX}_child.py", R6 / "scripts" / f"{PREFIX}_run.py", R6 / "tests" / f"test_{PREFIX}.py"]:
        if path.exists():
            stat = path.stat()
            files.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    files.sort(key=lambda row: row["path"])
    payload = {"created_utc": utc_now(), "new_file_count": len(files), "new_files": files}
    write_json(MANIFEST / "write_ledger.json", payload)
    return payload


def count_status(rows: list[dict[str, Any]], key: str, pass_value: str) -> dict[str, int]:
    passed = sum(1 for row in rows if row.get(key) == pass_value)
    return {"pass": passed, "fail": len(rows) - passed}


def write_report(integrity_before: dict[str, Any], integrity_after: dict[str, Any], prior: dict[str, Any], child: dict[str, Any]) -> None:
    provenance = read_json(MANIFEST / "imported_module_provenance.json")
    cpu = read_json(ARTIFACT / "cpu_execution_proof.json")
    permutation = read_json(ARTIFACT / "permutation_registry.json")
    task_action = read_json(ARTIFACT / "task_action_certificate.json")
    state_action = read_json(ARTIFACT / "state_action_certificate.json")
    alias = read_json(ARTIFACT / "alias_graph_certificate.json")
    optimizer = read_json(ARTIFACT / "optimizer_parameter_partition.json")
    schedule = read_json(ARTIFACT / "posterior_leaf_schedule.json")
    noise = read_json(ARTIFACT / "noise_tape_runtime_trace.json")
    primitive = read_json(ARTIFACT / "primitive_commutation_certificate.json")
    first_cases = read_json(ARTIFACT / "first_divergence_cases.json")
    synthetic = read_json(ARTIFACT / "synthetic_oracle_results.json")
    loss = read_json(ARTIFACT / "loss_certificate.json")
    backward = read_json(ARTIFACT / "backward_certificate.json")
    adam = read_json(ARTIFACT / "adam_certificate.json")
    final_labels = {row["case"]["case_id"]: row["final_label"] for row in first_cases["cases"]}
    first_divergences = {
        row["case"]["case_id"]: row.get("primitive_commutation", {}).get("tier1", {}).get("first_failure")
        for row in first_cases["cases"]
    }
    lines = [
        "# R6A Phase 2 Support-Permutation First-Divergence Certificate v8",
        "",
        "## Imported Module Provenance",
        "",
        json.dumps(provenance, indent=2, sort_keys=True),
        "",
        "## Protected-File Integrity Before/After",
        "",
        json.dumps({"before": integrity_before, "after": integrity_after}, indent=2, sort_keys=True),
        "",
        "## Read-Only v4 Evidence Summary",
        "",
        json.dumps(prior, indent=2, sort_keys=True),
        "",
        "## Support-Permutation Registry",
        "",
        json.dumps(permutation, indent=2, sort_keys=True),
        "",
        "## Task-Action Certificate Table",
        "",
        json.dumps(task_action, indent=2, sort_keys=True),
        "",
        "## State/Alias Certificate Table",
        "",
        json.dumps({"state_action": state_action, "alias": alias}, indent=2, sort_keys=True),
        "",
        "## Optimizer Parameter Partition",
        "",
        json.dumps(optimizer, indent=2, sort_keys=True),
        "",
        "## Posterior-Leaf Schedule and NoiseTape Trace",
        "",
        json.dumps({"posterior_leaf_schedule": schedule, "noise_tape_runtime_trace": noise}, indent=2, sort_keys=True),
        "",
        "## A0 Transparency Table",
        "",
        json.dumps({row["case"]["case_id"]: row.get("a0") for row in first_cases["cases"]}, indent=2, sort_keys=True),
        "",
        "## Tier 1 Primitive Boundary Trace",
        "",
        json.dumps(primitive, indent=2, sort_keys=True),
        "",
        "## Tier 2 Line-Local Trace If Reached",
        "",
        "Line-local trace artifacts are under `artifacts/r6a_phase2_support_permutation_first_divergence_v8/first_divergence_detail/`.",
        "",
        "## First-Divergence Table",
        "",
        json.dumps(first_divergences, indent=2, sort_keys=True),
        "",
        "## Synthetic Oracle Result",
        "",
        json.dumps(synthetic, indent=2, sort_keys=True),
        "",
        "## Loss/Backward/Adam Result",
        "",
        json.dumps({"loss": loss, "backward": backward, "adam": adam}, indent=2, sort_keys=True),
        "",
        "## Numerical/ULP Evidence",
        "",
        "The fixed comparator was `|a-b| <= 1e-12 + 1e-10*max(|a|,|b|)`. Per-event maximum absolute, relative, ULP, and over-tolerance counts are recorded in `first_divergence_trace.csv` and nested JSON certificates.",
        "",
        "## Final Labels",
        "",
        json.dumps(final_labels, indent=2, sort_keys=True),
        "",
        "## Claim Discipline",
        "",
        "A deterministic mismatch may be attributed to support-order dependence of the evaluated implementation only when Task action, initial state action, import provenance, instrumentation transparency, trace schedule, and synthetic-oracle interpretation all pass, and a first divergence is localized to a production source boundary or verified line-local intermediate.",
        "",
        "Passing the forward certificate establishes role-preserving support-permutation equivariance only for the audited tasks, permutations, initialization/state action, CPU float64 execution mode, and zero-noise diagnostic path. It is not a global theorem for all tasks or ordinary stochastic training.",
        "",
        "Phase 1 remains `PAIRING_SENSITIVE_QUERY_PILOT`.",
        "No GPU, retraining, P-slot, attribution, C-internal, J, or G was performed.",
        "",
        "## What Is And Is Not Established",
        "",
        "This audit localizes support-permutation failures for the audited cases only. It does not issue a global architecture theorem and does not alter the frozen Phase 1 or pre-existing Phase 2 results.",
    ]
    write_text(REPORT, "\n".join(lines) + "\n")


def main() -> int:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase2_before = collect_preexisting_phase2_paths()
    phase1_before = full_phase1_integrity()
    integrity_before = {
        "created_utc": utc_now(),
        "phase1_original_701": phase1_before,
        "preexisting_phase2_paths_before": phase2_before,
        "protected_file_integrity_pass": phase1_before["all_match_before"],
    }
    write_json(MANIFEST / "integrity_before.json", integrity_before)
    if not phase1_before["all_match_before"]:
        raise RuntimeError("Phase 1 protected-file integrity failed")
    prior = prior_v4_summary()
    child = run_child()
    phase1_after = full_phase1_integrity()
    phase2_after = compare_preexisting_phase2(phase2_before)
    integrity_after = {
        "created_utc": utc_now(),
        "phase1_original_701": phase1_after,
        "preexisting_phase2_no_write_guard": phase2_after,
        "protected_file_integrity_pass": phase1_after["all_match_before"] and phase2_after["pass"],
    }
    write_json(MANIFEST / "integrity_after.json", integrity_after)
    if not integrity_after["protected_file_integrity_pass"]:
        raise RuntimeError("Protected prior integrity failed after audit")
    write_report(integrity_before, integrity_after, prior, child)
    ledger = update_write_ledger()
    first_cases = read_json(ARTIFACT / "first_divergence_cases.json")
    task_rows = read_json(ARTIFACT / "task_action_certificate.json")["cases"]
    state_rows = read_json(ARTIFACT / "state_action_certificate.json")["cases"]
    a0_pass = sum(1 for row in first_cases["cases"] if row.get("a0", {}).get("source_record_replay_observed_pass") is True)
    summary = {
        "status": "R6A_PHASE2_SUPPORT_PERMUTATION_FIRST_DIVERGENCE_V8_COMPLETE",
        "protected_file_integrity_before_after": [integrity_before["protected_file_integrity_pass"], integrity_after["protected_file_integrity_pass"]],
        "import_provenance_status": read_json(MANIFEST / "imported_module_provenance.json")["import_provenance_status"],
        "real_task_permutation_cases_tested": child["real_case_count"],
        "task_action_counts": count_status(task_rows, "status", "TASK_ACTION_CERTIFIED"),
        "state_action_counts": count_status(state_rows, "status", "STATE_ACTION_CERTIFIED"),
        "a0_counts": {"pass": a0_pass, "fail": len(first_cases["cases"]) - a0_pass},
        "first_divergence_per_case": {row["case"]["case_id"]: row["final_label"] for row in first_cases["cases"]},
        "synthetic_oracle_result": {row["case"]["case_id"]: row["final_label"] for row in first_cases["cases"] if row["case"]["case_kind"] == "synthetic_oracle"},
        "loss_backward_adam_reached": child.get("loss_backward_adam_reached", False),
        "cpu_only_no_gpu": read_json(ARTIFACT / "cpu_execution_proof.json")["cuda_memory_allocated"] == 0,
        "final_report_path": str(REPORT),
        "write_ledger_count": ledger["new_file_count"],
    }
    write_json(ARTIFACT / "summary.json", summary)
    update_write_ledger()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
