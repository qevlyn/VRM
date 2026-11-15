from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, file_sha256, read_json, write_json, write_text


PREFIX = "r6a_phase2_two_pass_semantic_boundary_provenance_v11_2"
V8_PREFIX = "r6a_phase2_support_permutation_first_divergence_v8"
V10_PREFIX = "r6a_phase2_semantic_state_action_decoder_boundary_v10"
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT_DIR = R6 / "reports" / PREFIX
REPORT = REPORT_DIR / "R6A_PHASE2_TWO_PASS_SEMANTIC_BOUNDARY_CAUSAL_PROVENANCE_V11_2_REPORT.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compare_manifest(before: dict[str, Any]) -> dict[str, Any]:
    rows = []
    ok = True
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
        ok = ok and matches
        rows.append({**row, **current, "matches_before": matches})
    return {"created_utc": utc_now(), "file_count": len(rows), "all_match_before": ok, "files": rows}


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
        rel = path.relative_to(R6)
        rel_parts = rel.parts
        rel_text = rel.as_posix()
        if not any(part.startswith("r6a_phase2_") for part in rel_parts) and "r6a_phase2_" not in path.name:
            continue
        if PREFIX in rel_parts or path.name.startswith(PREFIX) or PREFIX in path.name or f"/{PREFIX}/" in rel_text:
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


def collect_versioned_artifacts(prefix: str) -> dict[str, Any]:
    roots = [R6 / "artifacts" / prefix, R6 / "manifests" / prefix, R6 / "reports" / prefix, R6 / "scripts", R6 / "tests"]
    files = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or prefix not in path.as_posix():
                continue
            stat = path.stat()
            files.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    files.sort(key=lambda row: row["path"])
    return {"created_utc": utc_now(), "prefix": prefix, "file_count": len(files), "files": files}


def compare_versioned_artifacts(before: dict[str, Any]) -> dict[str, Any]:
    after = collect_versioned_artifacts(before["prefix"])
    before_by_path = {row["path"]: row for row in before["files"]}
    after_by_path = {row["path"]: row for row in after["files"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = []
    for path in sorted(set(before_by_path) & set(after_by_path)):
        if before_by_path[path] != after_by_path[path]:
            changed.append({"path": path, "before": before_by_path[path], "after": after_by_path[path]})
    return {"created_utc": utc_now(), "prefix": before["prefix"], "pass": not added and not removed and not changed, "added": added, "removed": removed, "changed": changed}


def prior_summary(v8_before: dict[str, Any], v10_before: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "created_utc": utc_now(),
        "v8_file_count_hashed": v8_before["file_count"],
        "v10_file_count_hashed": v10_before["file_count"],
        "v8_report_path": str(R6 / "reports" / V8_PREFIX / "R6A_PHASE2_SUPPORT_PERMUTATION_FIRST_DIVERGENCE_CERTIFICATE_V8_REPORT.md"),
        "v10_report_path": str(R6 / "reports" / V10_PREFIX / "R6A_PHASE2_SEMANTIC_STATE_ACTION_DECODER_BOUNDARY_V10_REPORT.md"),
        "v8_summary": read_json(R6 / "artifacts" / V8_PREFIX / "summary.json"),
        "v10_summary": read_json(R6 / "artifacts" / V10_PREFIX / "summary.json"),
        "v10_final_outcome": read_json(R6 / "artifacts" / V10_PREFIX / "final_outcome_classification.json"),
        "v10_semantic_action_table_path": str(R6 / "artifacts" / V10_PREFIX / "semantic_action_table.csv"),
        "v10_collision_witness_path": str(R6 / "artifacts" / V10_PREFIX / "parameter_collision_witnesses.json"),
        "v10_affine_certificate_path": str(R6 / "artifacts" / V10_PREFIX / "affine_commutation_certificate.json"),
        "v10_A0_certificate_path": str(R6 / "artifacts" / V10_PREFIX / "instrumentation_reuse_certificate.json"),
        "v10_decoder_boundary_trace_path": str(R6 / "artifacts" / V10_PREFIX / "decoder_boundary_trace.json"),
        "v10_test_result_path": str(R6 / "artifacts" / V10_PREFIX / f"{V10_PREFIX}_pytest.txt"),
        "read_only": True,
    }
    write_json(ARTIFACT / "prior_v8_v10_readonly_summary.json", payload)
    return payload


def run_child() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(ARTIFACT / f"{PREFIX}_tmp")
    (ARTIFACT / f"{PREFIX}_tmp").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(R6 / "scripts" / f"{PREFIX}_child.py")]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=4500)
    (ARTIFACT / f"{PREFIX}_child_stdout.log").write_text(proc.stdout, encoding="utf-8")
    summary = read_json(ARTIFACT / "child_summary.json") if (ARTIFACT / "child_summary.json").exists() else {}
    summary["child_returncode"] = proc.returncode
    summary["child_stdout_tail"] = proc.stdout[-4000:]
    write_json(ARTIFACT / "child_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError("child failed")
    return summary


def run_pytest() -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(R6 / "tests" / f"test_{PREFIX}.py")]
    proc = subprocess.run(cmd, cwd=str(R6), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    output = proc.stdout
    (ARTIFACT / f"{PREFIX}_pytest.txt").write_text(output, encoding="utf-8")
    passed = 0
    failed = 0
    for token in output.replace(",", " ").split():
        if token.isdigit():
            continue
    if proc.returncode == 0:
        if " passed" in output:
            passed = int(output.rsplit(" passed", 1)[0].split()[-1])
        elif output.strip().endswith("passed"):
            passed = 1
    else:
        if " failed" in output:
            failed = int(output.rsplit(" failed", 1)[0].split()[-1])
        else:
            failed = 1
        if " passed" in output:
            passed = int(output.rsplit(" passed", 1)[0].split()[-1])
    return {"returncode": proc.returncode, "passed": passed, "failed": failed, "output_tail": output[-4000:]}


def update_write_ledger() -> dict[str, Any]:
    files = []
    for root in [ARTIFACT, MANIFEST, REPORT_DIR]:
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


def write_report(integrity_before: dict[str, Any], integrity_after: dict[str, Any], child: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    def load(name: str) -> Any:
        return read_json(ARTIFACT / name)

    final_cases = load("final_case_classifications.json")
    lines = [
        "# R6A Phase 2 Two-Pass Semantic-Boundary Causal Provenance Scan v11.2",
        "",
        "This report is a complete semantic-boundary provenance graph with targeted source-local microtraces inside replay cones. It is not a full PyTorch execution graph.",
        "",
        "## Integrity Before/After",
        json.dumps({"before": integrity_before, "after": integrity_after}, indent=2, sort_keys=True),
        "",
        "## Imported-Source Provenance",
        json.dumps(read_json(MANIFEST / "imported_module_provenance.json"), indent=2, sort_keys=True),
        "",
        "## CPU-Only Proof",
        json.dumps(load("cpu_execution_proof.json"), indent=2, sort_keys=True),
        "",
        "## v10 Semantic-Action Reuse Result",
        json.dumps({"task_action": load("task_action_reuse_certificate.json"), "semantic_state_action": load("semantic_state_action_reuse_certificate.json"), "collision_witnesses": load("collision_witness_reuse_certificate.json"), "affine": load("affine_reuse_certificate.json")}, indent=2, sort_keys=True),
        "",
        "## Static Semantic-Boundary Graph",
        json.dumps(load("static_semantic_boundary_graph.json"), indent=2, sort_keys=True),
        "",
        "## Verified Actual add_residual Dataflow",
        "The source check is `ADD_RESIDUAL_PRENORM_DATAFLOW_VERIFIED`: `pre_norm=True` computes an observational normalized temporary, while the down-affine consumes raw `x`.",
        "",
        "## Scanner Transparency Results",
        json.dumps(load("scanner_transparency_certificate.json"), indent=2, sort_keys=True),
        "",
        "## Scanner Mutation-Calibration Results",
        json.dumps(load("scanner_mutation_calibration.json"), indent=2, sort_keys=True),
        "",
        "## Pass-1 Summary Scan for All Seven Cases",
        json.dumps(load("pass1_candidate_markers.json"), indent=2, sort_keys=True),
        "",
        "## Pass-2 Targeted Replay Evidence",
        json.dumps(load("targeted_replay_exact_comparison.json"), indent=2, sort_keys=True),
        "",
        "## Per-Case Table",
        json.dumps(final_cases, indent=2, sort_keys=True),
        "",
        "## Cummax Microtrace",
        json.dumps(load("cummax_microtrace.json"), indent=2, sort_keys=True),
        "",
        "## Numerical Oracle Evidence",
        json.dumps(load("selective_numerical_oracle.json"), indent=2, sort_keys=True),
        "",
        "## Decoder-Scope Certificate",
        json.dumps(load("decoder_scope_certificate.json"), indent=2, sort_keys=True),
        "",
        "## What Is and Is Not Established",
        "Established: the seven fixed cases were scanned with the v10 semantic state action, Pass 1 distinguished `E_hash` from the v10 threshold marker, and Pass 2 replay-confirmed byte markers where present.",
        "",
        "Not established: this does not prove support-order semantic dependence, a learned shortcut carrier, a reasoning mechanism, or any architecture intervention.",
        "",
        "The frozen Phase 1 verdict remains `PAIRING_SENSITIVE_QUERY_PILOT`.",
        "",
        "No GPU, retraining, loss, backward, Adam, P-slot, C-internal, J, G, attribution, VRM, or architecture intervention was performed.",
        "",
        "## Focused Tests",
        json.dumps(pytest_result, indent=2, sort_keys=True),
    ]
    write_text(REPORT, "\n".join(lines) + "\n")


def main() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    phase1_before = phase1_integrity()
    phase2_before = collect_preexisting_phase2_paths()
    v8_before = collect_versioned_artifacts(V8_PREFIX)
    v10_before = collect_versioned_artifacts(V10_PREFIX)
    integrity_before = {"created_utc": utc_now(), "phase1": phase1_before, "preexisting_phase2": phase2_before, "v8_artifacts": v8_before, "v10_artifacts": v10_before, "pass": phase1_before["all_match_before"]}
    write_json(MANIFEST / "integrity_before.json", integrity_before)
    if not integrity_before["pass"]:
        raise RuntimeError("integrity before failed")
    prior_summary(v8_before, v10_before)
    child = run_child()
    phase1_after = phase1_integrity()
    phase2_after = compare_entry_set(phase2_before)
    v8_after = compare_versioned_artifacts(v8_before)
    v10_after = compare_versioned_artifacts(v10_before)
    integrity_after = {"created_utc": utc_now(), "phase1": phase1_after, "preexisting_phase2": phase2_after, "v8_artifacts": v8_after, "v10_artifacts": v10_after, "pass": phase1_after["all_match_before"] and phase2_after["pass"] and v8_after["pass"] and v10_after["pass"]}
    write_json(MANIFEST / "integrity_after.json", integrity_after)
    if not integrity_after["pass"]:
        raise RuntimeError("integrity after failed")
    pytest_result = run_pytest()
    write_report(integrity_before, integrity_after, child, pytest_result)
    ledger = update_write_ledger()
    summary = {
        "protected_file_integrity_before_after": [integrity_before["pass"], integrity_after["pass"]],
        "import_provenance_status": child["import_provenance_status"],
        "scanner_transparency_pass_fail": child["scanner_transparency_pass_fail"],
        "mutation_calibration_pass_fail": child["mutation_calibration_pass_fail"],
        "eligible_cases_scanned": child["eligible_cases_scanned"],
        "E_hash_per_case": child["E_hash_per_case"],
        "E_bit_per_case": child["E_bit_per_case"],
        "E_threshold_per_case": child["E_threshold_per_case"],
        "decoder_scope_status": child["decoder_scope_status"],
        "audited_discrete_behavior_changed": child["audited_discrete_behavior_changed"],
        "final_classification_per_case": child["final_classification_per_case"],
        "focused_tests_passed_failed": [pytest_result["passed"], pytest_result["failed"]],
        "cpu_only_no_gpu": child["cpu_only_no_gpu"],
        "final_report_path": str(REPORT),
        "write_ledger_count": ledger["new_file_count"],
    }
    write_json(ARTIFACT / "summary.json", summary)
    update_write_ledger()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pytest_result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
