from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, file_sha256, read_json, write_json, write_text


PREFIX = "r6a_phase2_semantic_state_action_decoder_boundary_v10"
V8_PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT_DIR = R6 / "reports" / PREFIX
REPORT = REPORT_DIR / "R6A_PHASE2_SEMANTIC_STATE_ACTION_DECODER_BOUNDARY_V10_REPORT.md"


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


def phase1_integrity() -> dict[str, Any]:
    path = R6 / "manifests" / "protected_paths_before_phase1.json"
    original = read_json(path)
    result = compare_manifest(original)
    result["original_manifest_path"] = str(path)
    result["original_protected_file_count"] = original["file_count"]
    return result


def collect_preexisting_phase2_paths() -> dict[str, Any]:
    entries = []
    for path in R6.rglob("*"):
        rel_parts = path.relative_to(R6).parts
        if not any(part.startswith("r6a_phase2_") for part in rel_parts):
            continue
        if PREFIX in rel_parts or path.name.startswith(PREFIX):
            continue
        stat = path.stat()
        entries.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": stat.st_size if path.is_file() else None,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    entries.sort(key=lambda row: row["path"])
    return {"created_utc": utc_now(), "entry_count": len(entries), "file_count": sum(1 for row in entries if row["is_file"]), "entries": entries}


def compare_entry_set(before: dict[str, Any]) -> dict[str, Any]:
    after = collect_preexisting_phase2_paths()
    before_by_path = {row["path"]: row for row in before["entries"]}
    after_by_path = {row["path"]: row for row in after["entries"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = []
    for path in sorted(set(before_by_path) & set(after_by_path)):
        if before_by_path[path] != after_by_path[path]:
            changed.append({"path": path, "before": before_by_path[path], "after": after_by_path[path]})
    return {"created_utc": utc_now(), "pass": not added and not removed and not changed, "added": added, "removed": removed, "changed": changed}


def collect_v8_artifacts() -> dict[str, Any]:
    roots = [R6 / "artifacts" / V8_PREFIX, R6 / "manifests" / V8_PREFIX, R6 / "reports" / V8_PREFIX, R6 / "scripts", R6 / "tests"]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if V8_PREFIX not in path.as_posix():
                continue
            stat = path.stat()
            files.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    files.sort(key=lambda row: row["path"])
    return {"created_utc": utc_now(), "file_count": len(files), "files": files}


def compare_v8_artifacts(before: dict[str, Any]) -> dict[str, Any]:
    after = collect_v8_artifacts()
    before_by_path = {row["path"]: row for row in before["files"]}
    after_by_path = {row["path"]: row for row in after["files"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = []
    for path in sorted(set(before_by_path) & set(after_by_path)):
        if before_by_path[path] != after_by_path[path]:
            changed.append({"path": path, "before": before_by_path[path], "after": after_by_path[path]})
    return {"created_utc": utc_now(), "pass": not added and not removed and not changed, "added": added, "removed": removed, "changed": changed}


def v8_readonly_summary(v8_before: dict[str, Any]) -> dict[str, Any]:
    base = R6 / "artifacts" / V8_PREFIX
    report = R6 / "reports" / V8_PREFIX / "R6A_PHASE2_SUPPORT_PERMUTATION_FIRST_DIVERGENCE_CERTIFICATE_V8_REPORT.md"
    summary = read_json(base / "summary.json")
    payload = {
        "created_utc": utc_now(),
        "v8_file_count_hashed": v8_before["file_count"],
        "v8_report_path": str(report),
        "v8_report_sha256": file_sha256(report) if report.exists() else None,
        "v8_first_divergence_per_case": summary.get("first_divergence_per_case"),
        "v8_state_action_counts": summary.get("state_action_counts"),
        "v8_read_only": True,
    }
    write_json(ARTIFACT / "v8_readonly_evidence_summary.json", payload)
    return payload


def run_child() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(ARTIFACT / f"{PREFIX}_tmp")
    (ARTIFACT / f"{PREFIX}_tmp").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(R6 / "scripts" / f"{PREFIX}_child.py")]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    (ARTIFACT / f"{PREFIX}_child_stdout.log").write_text(proc.stdout, encoding="utf-8")
    summary = read_json(ARTIFACT / "child_summary.json") if (ARTIFACT / "child_summary.json").exists() else {}
    summary["child_returncode"] = proc.returncode
    summary["child_stdout_tail"] = proc.stdout[-4000:]
    write_json(ARTIFACT / "child_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError("child failed")
    return summary


def update_write_ledger() -> dict[str, Any]:
    files = []
    for root in [ARTIFACT, MANIFEST, REPORT_DIR]:
        if not root.exists():
            continue
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


def write_report(integrity_before: dict[str, Any], integrity_after: dict[str, Any], v8_summary: dict[str, Any], child: dict[str, Any]) -> None:
    pytest_path = ARTIFACT / f"{PREFIX}_pytest.txt"
    pytest_result = pytest_path.read_text(encoding="utf-8").strip() if pytest_path.exists() else "Focused pytest not yet run when this report was generated."
    data = {
        "integrity_before": integrity_before,
        "integrity_after": integrity_after,
        "imported_source_provenance": read_json(MANIFEST / "imported_module_provenance.json"),
        "v8_action_behavior": read_json(ARTIFACT / "v8_action_behavior_audit.json"),
        "v8_readonly_summary": v8_summary,
        "cpu_execution_proof": read_json(ARTIFACT / "cpu_execution_proof.json"),
        "collision_witnesses": read_json(ARTIFACT / "parameter_collision_witnesses.json"),
        "affine_commutation": read_json(ARTIFACT / "affine_commutation_certificate.json"),
        "task_action": read_json(ARTIFACT / "task_action_certificate.json"),
        "semantic_state_action": read_json(ARTIFACT / "semantic_state_action_certificate.json"),
        "a0_reuse": read_json(ARTIFACT / "instrumentation_reuse_certificate.json"),
        "decoder_boundary_trace": read_json(ARTIFACT / "decoder_boundary_trace.json"),
        "forward_trace_summary": read_json(ARTIFACT / "forward_trace_summary.json"),
        "final_outcome_classification": read_json(ARTIFACT / "final_outcome_classification.json"),
        "child_summary": child,
    }
    lines = [
        "# R6A Phase 2 Semantic State-Action Repair and Decoder-Boundary Discrimination v10",
        "",
        "## Integrity Before/After",
        json.dumps({"before": integrity_before, "after": integrity_after}, indent=2, sort_keys=True),
        "",
        "## Imported-Source Provenance",
        json.dumps(data["imported_source_provenance"], indent=2, sort_keys=True),
        "",
        "## Read-Only v8 Action-Behavior Evidence",
        json.dumps({"v8_readonly_summary": v8_summary, "v8_action_behavior": data["v8_action_behavior"]}, indent=2, sort_keys=True),
        "",
        "## Source-Derived Semantic Action Table",
        "See `semantic_action_table.csv`. The action table uses only closed constructor roles and posterior leaf `dims[0]`; it does not use leading-dimension equality as an action criterion.",
        "",
        "## Collision Witnesses",
        json.dumps(data["collision_witnesses"], indent=2, sort_keys=True),
        "",
        "## Unit-Test Outcomes",
        pytest_result,
        "",
        "## Direct Affine Positive and Negative-Control Certificates",
        json.dumps(data["affine_commutation"], indent=2, sort_keys=True),
        "",
        "## Task-Action Result",
        json.dumps(data["task_action"], indent=2, sort_keys=True),
        "",
        "## Semantic State-Action Result",
        json.dumps(data["semantic_state_action"], indent=2, sort_keys=True),
        "",
        "## A0 Instrumentation Reuse Result",
        json.dumps(data["a0_reuse"], indent=2, sort_keys=True),
        "",
        "## Decoder-Boundary Trace",
        json.dumps(data["decoder_boundary_trace"], indent=2, sort_keys=True),
        "",
        "## First-Divergence Table",
        json.dumps(data["forward_trace_summary"], indent=2, sort_keys=True),
        "",
        "## Final Outcome Classification",
        json.dumps(data["final_outcome_classification"], indent=2, sort_keys=True),
        "",
        "## Prohibited-Claim Compliance",
        "No support-order dependence, affine shortcut route, layer-removal recommendation, VRM addition, reasoning improvement/degradation, or Phase 1 reinterpretation is claimed.",
        "",
        "No GPU, retraining, loss, backward, Adam, VRM, P-slot, C-internal, J, G, or architecture edit was run.",
        "",
        "The frozen Phase 1 verdict remains `PAIRING_SENSITIVE_QUERY_PILOT`.",
    ]
    write_text(REPORT, "\n".join(lines) + "\n")


def main() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase1_before = phase1_integrity()
    phase2_before = collect_preexisting_phase2_paths()
    v8_before = collect_v8_artifacts()
    integrity_before = {"created_utc": utc_now(), "phase1": phase1_before, "preexisting_phase2": phase2_before, "v8_artifacts": v8_before, "pass": phase1_before["all_match_before"]}
    write_json(MANIFEST / "integrity_before.json", integrity_before)
    if not integrity_before["pass"]:
        raise RuntimeError("integrity before failed")
    v8_summary = v8_readonly_summary(v8_before)
    child = run_child()
    phase1_after = phase1_integrity()
    phase2_after = compare_entry_set(phase2_before)
    v8_after = compare_v8_artifacts(v8_before)
    integrity_after = {"created_utc": utc_now(), "phase1": phase1_after, "preexisting_phase2": phase2_after, "v8_artifacts": v8_after, "pass": phase1_after["all_match_before"] and phase2_after["pass"] and v8_after["pass"]}
    write_json(MANIFEST / "integrity_after.json", integrity_after)
    if not integrity_after["pass"]:
        raise RuntimeError("integrity after failed")
    write_report(integrity_before, integrity_after, v8_summary, child)
    ledger = update_write_ledger()
    summary = {
        "status": "R6A_PHASE2_SEMANTIC_STATE_ACTION_DECODER_BOUNDARY_V10_COMPLETE",
        "protected_file_integrity_before_after": [integrity_before["pass"], integrity_after["pass"]],
        "import_provenance_status": read_json(MANIFEST / "imported_module_provenance.json")["import_provenance_status"],
        "v8_action_behavior_status": child["v8_action_behavior_status"],
        "collision_witness_pass_fail": child["collision_witness_pass_fail"],
        "affine_certificate_pass_fail": child["affine_certificate_pass_fail"],
        "task_action_pass_fail": child["task_action_pass_fail"],
        "semantic_state_action_pass_fail": child["semantic_state_action_pass_fail"],
        "a0_reuse_pass_fail": child["a0_reuse_pass_fail"],
        "decoder_boundary_result_per_case": child["decoder_boundary_result_per_case"],
        "cpu_only_no_gpu": child["cpu_only_no_gpu"],
        "final_report_path": str(REPORT),
        "write_ledger_count": ledger["new_file_count"],
    }
    write_json(ARTIFACT / "summary.json", summary)
    update_write_ledger()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
