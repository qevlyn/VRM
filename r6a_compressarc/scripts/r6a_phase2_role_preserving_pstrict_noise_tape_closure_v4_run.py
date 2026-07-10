from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, file_sha256, read_json, write_json, write_text


PREFIX = "r6a_phase2_role_preserving_pstrict_noise_tape_closure_v4"
PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
R6 = PROJECT_ROOT / "r6a_compressarc"
ARTIFACT = R6 / "artifacts" / PREFIX
MANIFEST = R6 / "manifests" / PREFIX
REPORT_DIR = R6 / "reports" / PREFIX
REPORT = REPORT_DIR / "R6A_PHASE2_ROLE_PRESERVING_PSTRICT_CLOSURE_V4_REPORT.md"
ARCHIVE_DIR = R6 / "archives" / PREFIX


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


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
        current["matches_before"] = matches
        rows.append({**row, **current})
        all_match = all_match and matches
    return {"created_utc": utc_now(), "file_count": len(rows), "all_match_before": all_match, "files": rows}


def full_701_integrity() -> dict[str, Any]:
    original = read_json(R6 / "manifests" / "protected_paths_before_phase1.json")
    result = compare_manifest(original)
    result["original_manifest_path"] = str(R6 / "manifests" / "protected_paths_before_phase1.json")
    result["original_protected_file_count"] = original["file_count"]
    return result


def collect_existing_guard_paths() -> dict[str, Any]:
    files = []
    for path in R6.rglob("*"):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        name_hits = any(part.startswith("r6a_phase1_") or part.startswith("r6a_phase2_") for part in path.parts)
        if not name_hits:
            continue
        if PREFIX in path.parts or path.name.startswith(PREFIX):
            continue
        stat = path.stat()
        files.append(
            {
                "path": rel,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size_bytes": stat.st_size if path.is_file() else None,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(path) if path.is_file() else None,
            }
        )
    files.sort(key=lambda row: row["path"])
    return {"created_utc": utc_now(), "file_count": sum(1 for row in files if row["is_file"]), "entry_count": len(files), "entries": files}


def compare_guard(before: dict[str, Any]) -> dict[str, Any]:
    before_by_path = {row["path"]: row for row in before["entries"]}
    after = collect_existing_guard_paths()
    after_by_path = {row["path"]: row for row in after["entries"]}
    added = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    changed = []
    for path in sorted(set(after_by_path) & set(before_by_path)):
        old, new = before_by_path[path], after_by_path[path]
        if old != new:
            changed.append({"path": path, "before": old, "after": new})
    return {"created_utc": utc_now(), "added": added, "removed": removed, "changed": changed, "pass": not added and not removed and not changed}


def create_phase1_archive(files: list[dict[str, Any]]) -> dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / f"{PREFIX}_phase1_archive_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for row in files:
            path = PROJECT_ROOT / row["path"]
            zf.write(path, f"{PREFIX}_phase1_archive/{row['path']}")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
    return {"archive_path": str(archive), "zip_size_bytes": archive.stat().st_size, "zip_sha256": file_sha256(archive), "integrity_test_passed": bad is None, "file_count": len(files)}


def categorize_701_path(path: str) -> str:
    if path.startswith("vrm/"):
        return "protected_vrm_source"
    if path.startswith("tests/"):
        return "protected_repo_tests"
    if path.startswith("docs/R5A"):
        return "protected_r5a_docs"
    if path.startswith("artifacts/r5a") or path.startswith("artifacts/R5A"):
        return "protected_r5a_artifacts"
    if path.startswith("checkpoints/"):
        return "protected_checkpoints"
    if path.startswith("third_party/compressarc_reference/"):
        return "pinned_compressarc_reference_source"
    return "other_original_protected_scope"


def reconcile_scopes(current_701: dict[str, Any]) -> dict[str, Any]:
    old_295_path = R6 / "manifests" / "r6a_phase2_symmetry_contract" / "phase1_immutability_before.json"
    old_295 = read_json(old_295_path)
    paths_701 = {row["path"] for row in current_701["files"]}
    paths_295 = {row["path"] for row in old_295["files"]}
    diff_701_295 = sorted(paths_701 - paths_295)
    diff_295_701 = sorted(paths_295 - paths_701)
    excluded = [{"path": path, "category": categorize_701_path(path), "reason": "part of original 701 protected scope but not part of earlier Phase2 295 Phase1 archive/check subset"} for path in diff_701_295]
    archive_manifest_path = R6 / "manifests" / "r6a_phase2_symmetry_contract" / "phase1_archive_manifest.json"
    archive_manifest = read_json(archive_manifest_path)
    archive_status = []
    with zipfile.ZipFile(archive_manifest["archive_path"]) as zf:
        names = set(zf.namelist())
        for row in old_295["files"]:
            member = f"R6A_PHASE1_FROZEN_FOR_PHASE2/{row['path']}"
            present = member in names
            digest = None
            if present:
                digest = hashlib.sha256(zf.read(member)).hexdigest()
            archive_status.append({"path": row["path"], "member": member, "present": present, "sha256": digest, "matches_295_manifest": present and digest == row["sha256"]})
    payload = {
        "created_utc": utc_now(),
        "original_701_manifest_path": str(R6 / "manifests" / "protected_paths_before_phase1.json"),
        "earlier_295_scope_manifest_path": str(old_295_path),
        "archive_manifest_path": str(archive_manifest_path),
        "is_295_subset_of_701": paths_295 <= paths_701,
        "is_701_subset_of_295": paths_701 <= paths_295,
        "count_701": len(paths_701),
        "count_295": len(paths_295),
        "set_difference_701_minus_295": excluded,
        "set_difference_295_minus_701": diff_295_701,
        "current_integrity_result_for_701": {"all_match_before": current_701["all_match_before"], "file_count": current_701["file_count"]},
        "archive_checksum_status_by_archived_member": archive_status,
        "all_archived_members_match_295_manifest": all(row["matches_295_manifest"] for row in archive_status),
        "phase1_fully_reverified_for_701": current_701["all_match_before"],
    }
    write_json(REPORT_DIR / "integrity_scope_reconciliation.json", payload)
    return payload


def frozen_ledger_cycle_invariant() -> dict[str, Any]:
    structural = read_json(R6 / "artifacts" / "r6a_phase2_symmetry_preflight" / "s_cycle_invariance_structural_reanalysis.json")
    selected = {row["task_id"]: row for row in structural["tasks"] if row["lemma_assumption_status"] == "LEMMA_VERIFIED_FOR_TASK"}
    per_run = read_json(R6 / "artifacts" / "r6a_phase1_per_run_results.json")["rows"]
    aggregate = read_json(R6 / "artifacts" / "r6a_phase1_aggregate_results.json")
    phase1_cells = {cell["task_id"]: [] for cell in aggregate["cells"]}
    for cell in aggregate["cells"]:
        phase1_cells.setdefault(cell["task_id"], []).append(cell)
    by_key = {(row["task_id"], row["seed"], row["condition"]): row for row in per_run if row.get("status") == "RUN_COMPLETED"}
    rows = []
    for task_id, srow in selected.items():
        cells = []
        for seed in sorted({key[1] for key in by_key if key[0] == task_id}):
            f1 = by_key[(task_id, seed, "F1")]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            s = by_key[(task_id, seed, "S")]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            f2 = by_key[(task_id, seed, "F2")]["evaluator_metrics"]["target_cell_nll_from_final100_mean_probability"]
            pairing_delta = s - ((f1 + f2) / 2)
            noise_delta = abs(f1 - f2)
            cells.append({"seed": seed, "F1": f1, "S": s, "F2": f2, "pairing_delta": pairing_delta, "noise_delta": noise_delta, "pairing_minus_noise": pairing_delta - noise_delta})
        pairing_median = statistics.median([cell["pairing_delta"] for cell in cells])
        noise_median = statistics.median([cell["noise_delta"] for cell in cells])
        stratum = next(cell["stratum"] for cell in aggregate["cells"] if cell["task_id"] == task_id)
        loto = None
        for row in aggregate["stratum_a" if stratum == "A" else "stratum_b"]["leave_one_task_out"]:
            if row["removed_task_id"] == task_id:
                loto = row
        rows.append(
            {
                "analysis_label": "EXPLORATORY_FROZEN_LEDGER_REANALYSIS",
                "task_id": task_id,
                "stratum": stratum,
                "seed_rows": cells,
                "task_level_median_pairing_delta": pairing_median,
                "task_level_median_noise_delta": noise_median,
                "sign_of_task_level_median_pairing_delta": "positive" if pairing_median > 0 else ("negative" if pairing_median < 0 else "zero"),
                "median_pairing_delta_exceeds_median_noise_delta": pairing_median > noise_median,
                "directly_reusable_existing_loto": loto,
                "structural_lemma_assumptions_and_status": {
                    "lemma_assumption_status": srow["lemma_assumption_status"],
                    "s_cycle_invariant": srow["s_cycle_invariant"],
                    "task_masks_equal_under_s": srow["task_masks_equal_under_s"],
                    "predicted_query_shape_equal_under_s": srow["predicted_query_shape_equal_under_s"],
                    "grid_size_uncertain_branch_equal_under_s": srow["grid_size_uncertain_branch_equal_under_s"],
                },
            }
        )
    all_task_medians = []
    for task_id, cells in phase1_cells.items():
        all_task_medians.append(statistics.median([cell["delta"] for cell in cells]))
    subset_medians = [row["task_level_median_pairing_delta"] for row in rows]
    payload = {
        "analysis_label": "EXPLORATORY_FROZEN_LEDGER_REANALYSIS",
        "created_utc": utc_now(),
        "selected_task_count": len(rows),
        "selected_tasks": rows,
        "count_with_median_pairing_delta_gt_0": sum(row["task_level_median_pairing_delta"] > 0 for row in rows),
        "count_with_median_pairing_delta_gt_noise_delta": sum(row["median_pairing_delta_exceeds_median_noise_delta"] for row in rows),
        "all_task_vs_cycle_invariant_subset": {
            "all_task_count": len(all_task_medians),
            "all_task_median_of_task_medians": statistics.median(all_task_medians),
            "cycle_invariant_task_count": len(subset_medians),
            "cycle_invariant_median_of_task_medians": statistics.median(subset_medians),
        },
    }
    write_json(ARTIFACT / "cycle_invariant_effects.json", payload)
    csv_path = ARTIFACT / "cycle_invariant_effects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "stratum", "seed", "F1", "S", "F2", "pairing_delta", "noise_delta", "pairing_minus_noise", "task_median_pairing_delta", "task_median_noise_delta"])
        writer.writeheader()
        for row in rows:
            for seed in row["seed_rows"]:
                writer.writerow({"task_id": row["task_id"], "stratum": row["stratum"], **seed, "task_median_pairing_delta": row["task_level_median_pairing_delta"], "task_median_noise_delta": row["task_level_median_noise_delta"]})
    md = ["# Cycle-Invariant Frozen-Ledger Effects", "", "`EXPLORATORY_FROZEN_LEDGER_REANALYSIS`", "", f"- selected tasks: {len(rows)}", f"- median pairing delta > 0: {payload['count_with_median_pairing_delta_gt_0']}", f"- median pairing delta > noise delta: {payload['count_with_median_pairing_delta_gt_noise_delta']}"]
    write_text(ARTIFACT / "cycle_invariant_effects.md", "\n".join(md) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [row["task_level_median_noise_delta"] for row in rows]
        ys = [row["task_level_median_pairing_delta"] for row in rows]
        labels = [row["task_id"] for row in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=7)
        lim = max(xs + ys + [1e-6])
        ax.plot([0, lim], [0, lim], "k--", linewidth=1)
        ax.set_xlabel("median noise delta |F1-F2|")
        ax.set_ylabel("median pairing delta S - mean(F1,F2)")
        ax.set_title("Cycle-invariant subset (exploratory frozen ledger)")
        fig.tight_layout()
        fig.savefig(ARTIFACT / "figure_cycle_invariant_pairing_vs_noise.png")
        fig.savefig(ARTIFACT / "figure_cycle_invariant_pairing_vs_noise.pdf")
        plt.close(fig)
    except Exception as exc:
        payload["figure_error"] = f"{type(exc).__name__}: {exc}"
        write_json(ARTIFACT / "cycle_invariant_effects.json", payload)
    return payload


def prior_pstrict_classification() -> dict[str, Any]:
    prior_source = (R6 / "scripts" / "phase2_cpu_preflight_child.py").read_text(encoding="utf-8")
    prior_p = read_json(R6 / "artifacts" / "r6a_phase2_symmetry_preflight" / "p_strict_preflight.json")
    role_preserving = "permute_train_problem(problem, train_perm)" in prior_source and "range(n_train, n_train + len(problem[\"test\"]))" in prior_source
    classification = "P_ROLE_PRESERVING_UNCOUPLED_STOCHASTICITY" if role_preserving else "P_SCOPE_UNKNOWN_REQUIRES_MANUAL_REVIEW"
    payload = {
        "created_utc": utc_now(),
        "prior_implementation_path": str(R6 / "scripts" / "phase2_cpu_preflight_child.py"),
        "prior_manifest_path": str(R6 / "artifacts" / "r6a_phase2_symmetry_preflight" / "p_strict_preflight.json"),
        "prior_actual_labels": [{"task_id": row["task_id"], "classification": row["classification"]} for row in prior_p["actual_p_strict_preflights"]],
        "train_test_membership_moved": False if role_preserving else None,
        "prior_scope_classification": classification,
        "statement": "The old result remains historical and unmodified; it used role-preserving support permutation but did not explicitly couple stochastic latent noise under the support-permutation group action.",
    }
    write_json(ARTIFACT / "prior_pstrict_scope_classification.json", payload)
    return payload


def source_pin() -> dict[str, Any]:
    proc = subprocess.run(["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    payload = {"pinned_commit": PINNED_COMMIT, "git_head": proc.stdout.strip(), "verified": proc.returncode == 0 and proc.stdout.strip() == PINNED_COMMIT}
    write_json(MANIFEST / "upstream_source_pin_verification.json", payload)
    return payload


def run_child() -> dict[str, Any]:
    cmd = [sys.executable, str(R6 / "scripts" / f"{PREFIX}_child.py")]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200, env={**dict(), **__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    (ARTIFACT / "child_stdout.log").write_text(proc.stdout, encoding="utf-8")
    summary = read_json(ARTIFACT / "child_summary.json") if (ARTIFACT / "child_summary.json").exists() else {}
    summary["child_returncode"] = proc.returncode
    summary["child_stdout_tail"] = proc.stdout[-4000:]
    write_json(ARTIFACT / "child_summary.json", summary)
    if proc.returncode != 0:
        raise RuntimeError("child failed")
    return summary


def write_report(pin: dict[str, Any], integrity_before: dict[str, Any], integrity_after: dict[str, Any], reconciliation: dict[str, Any], cycle: dict[str, Any], prior: dict[str, Any], child: dict[str, Any]) -> None:
    closure = read_json(ARTIFACT / "closure_results.json")
    cpu = read_json(ARTIFACT / "cpu_execution_proof.json")
    alias_graph = read_json(ARTIFACT / "alias_graph.json")
    snapshot_restore = read_json(ARTIFACT / "snapshot_restore_audit.json")
    noise_manifest = read_json(ARTIFACT / "noise_tape_manifest.json")
    noise_runtime = read_json(ARTIFACT / "noise_tape_runtime_trace.json")
    noise_unhandled = read_json(ARTIFACT / "noise_tape_unhandled_call_report.json")
    final_labels = {row["task_id"]: row["final_label"] for row in closure["tasks"]}
    first_divergences = [
        {"task_id": row["task_id"], "final_label": row["final_label"], "first_divergence": row.get("first_divergence")}
        for row in closure["tasks"]
    ]
    stage_summary = [
        {
            "task_id": row["task_id"],
            "final_label": row["final_label"],
            "stages": {stage: value.get("pass") for stage, value in row.get("stages", {}).items()},
            "noise_runtime_trace_count": row.get("noise_runtime_trace_count"),
            "unhandled_noise_call_count": row.get("unhandled_noise_call_count"),
        }
        for row in closure["tasks"]
    ]
    lines = [
        "# R6A Phase 2 Role-Preserving P-Strict NoiseTape Closure v4",
        "",
        f"- upstream source pin verified: {pin['verified']} ({pin['git_head']})",
        f"- full 701 protected-file integrity before: {integrity_before['all_match_before']}",
        f"- full 701 protected-file integrity after: {integrity_after['all_match_before']}",
        f"- 295 subset of 701: {reconciliation['is_295_subset_of_701']}",
        f"- cycle-invariant selected tasks: {cycle['selected_task_count']}",
        f"- prior P-strict scope classification: {prior['prior_scope_classification']}",
        f"- CPU-only proof: {cpu['all_runtime_tensors_under_test_cpu']} / no CUDA allocation {cpu['cuda_memory_allocated'] == 0}",
        "",
        "Phase 1 verdict remains `PAIRING_SENSITIVE_QUERY_PILOT`.",
        "The old uncoupled P result remains historical and unmodified.",
        "No Phase 1 artifact changed.",
        "P-slot, attribution, C-internal, J, G, retraining, and GPU runs were not performed.",
        "",
        "## Role-Preserving Permutation Definition",
        "",
        "The only group action used here is support-order permutation preserving train/test roles: `G_role = S_n_train x I_n_test`.",
        "For support slots `0 <= i < n_train`, `perm_new_to_old[i] = (i + 1) mod n_train`; for query/test slots, `perm_new_to_old[i] = i`.",
        "The transformed raw challenge is rebuilt by reordering complete support pairs and leaving every query/test entry unchanged, then constructing a fresh production `Task`.",
        "",
        "## 701-versus-295 Reconciliation",
        "",
        json.dumps(reconciliation, indent=2, sort_keys=True),
        "",
        "## Frozen-Ledger Cycle-Invariant Descriptive Result",
        "",
        json.dumps(cycle, indent=2, sort_keys=True),
        "",
        "## Prior P-Strict Scope Classification",
        "",
        json.dumps(prior, indent=2, sort_keys=True),
        "",
        "## Role-Preserving P-Strict Closure Results",
        "",
        "### Stage Summary",
        "",
        json.dumps(stage_summary, indent=2, sort_keys=True),
        "",
        "### First-Divergence Table",
        "",
        json.dumps(first_divergences, indent=2, sort_keys=True),
        "",
        "### Final Closure Labels",
        "",
        json.dumps(final_labels, indent=2, sort_keys=True),
        "",
        json.dumps(closure, indent=2, sort_keys=True),
        "",
        "## Alias Graph and Snapshot-Restore Evidence",
        "",
        json.dumps({"alias_graph": alias_graph, "snapshot_restore_audit": snapshot_restore}, indent=2, sort_keys=True),
        "",
        "## NoiseTape Runtime Trace",
        "",
        json.dumps({"manifest": noise_manifest, "runtime_trace": noise_runtime, "unhandled_call_report": noise_unhandled}, indent=2, sort_keys=True),
        "",
        "## CPU Isolation Proof",
        "",
        json.dumps(cpu, indent=2, sort_keys=True),
    ]
    write_text(REPORT, "\n".join(lines) + "\n")


def update_write_ledger() -> dict[str, Any]:
    created = []
    for root in [MANIFEST, ARTIFACT, REPORT_DIR, ARCHIVE_DIR]:
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    created.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    for path in [R6 / "scripts" / f"{PREFIX}_run.py", R6 / "scripts" / f"{PREFIX}_child.py", R6 / "tests" / f"test_{PREFIX}.py"]:
        if path.exists():
            stat = path.stat()
            created.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    created.sort(key=lambda row: row["path"])
    payload = {"created_utc": utc_now(), "new_file_count": len(created), "new_files": created}
    write_json(MANIFEST / "write_ledger.json", payload)
    return payload


def main() -> int:
    MANIFEST.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before_guard = collect_existing_guard_paths()
    integrity_before = full_701_integrity()
    if not integrity_before["all_match_before"]:
        write_json(MANIFEST / "preflight_integrity_before.json", integrity_before)
        raise RuntimeError("original 701 protected-file integrity failed")
    phase1_archive = create_phase1_archive(integrity_before["files"])
    integrity_before["no_write_guard_before"] = before_guard
    integrity_before["phase1_701_archive"] = phase1_archive
    write_json(MANIFEST / "preflight_integrity_before.json", integrity_before)
    pin = source_pin()
    reconciliation = reconcile_scopes(integrity_before)
    cycle = frozen_ledger_cycle_invariant()
    prior = prior_pstrict_classification()
    child = run_child()
    integrity_after = full_701_integrity()
    after_guard = compare_guard(before_guard)
    integrity_after["no_write_guard_after"] = after_guard
    write_json(MANIFEST / "preflight_integrity_after.json", integrity_after)
    write_report(pin, integrity_before, integrity_after, reconciliation, cycle, prior, child)
    ledger = update_write_ledger()
    summary = {
        "status": "R6A_PHASE2_ROLE_PRESERVING_PSTRICT_CLOSURE_V4_COMPLETE",
        "protected_integrity_before": integrity_before["all_match_before"],
        "protected_integrity_after": integrity_after["all_match_before"],
        "reconciliation_295_subset_701": reconciliation["is_295_subset_of_701"],
        "cycle_invariant_task_count": cycle["selected_task_count"],
        "prior_pstrict_scope_classification": prior["prior_scope_classification"],
        "task_labels": {row["task_id"]: row["final_label"] for row in read_json(ARTIFACT / "closure_results.json")["tasks"]},
        "cpu_only_no_gpu": read_json(ARTIFACT / "cpu_execution_proof.json")["cuda_memory_allocated"] == 0,
        "report_path": str(REPORT),
        "write_ledger_count": ledger["new_file_count"],
    }
    write_json(ARTIFACT / "summary.json", summary)
    update_write_ledger()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
