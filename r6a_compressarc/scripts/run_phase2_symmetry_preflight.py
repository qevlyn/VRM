from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from r6a_common import PROJECT_ROOT, REFERENCE_ROOT, file_sha256, read_json, write_json, write_text


PINNED_COMMIT = "83a22218024d46273eb32b769a906340202ffb4d"
R6 = PROJECT_ROOT / "r6a_compressarc"
CONTRACT_DIR = R6 / "manifests" / "r6a_phase2_symmetry_contract"
PREFLIGHT_DIR = R6 / "artifacts" / "r6a_phase2_symmetry_preflight"
ARCHIVE_DIR = R6 / "archives"
REPORT_PATH = R6 / "reports" / "r6a_phase2_symmetry_preflight_report.md"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def phase1_paths() -> list[Path]:
    roots = [
        R6 / "manifests" / "r6a_phase1_multirepeat_frozen_manifest.json",
        R6 / "manifests" / "r6a_phase1_initialization_artifacts.json",
        R6 / "manifests" / "protected_paths_before_phase1.json",
        R6 / "manifests" / "protected_paths_after_phase1.json",
        R6 / "artifacts" / "r6a_phase1_per_run_results.json",
        R6 / "artifacts" / "r6a_phase1_per_task_results.json",
        R6 / "artifacts" / "r6a_phase1_aggregate_results.json",
        R6 / "artifacts" / "r6a_phase1_firewall_confirmation.json",
        R6 / "artifacts" / "r6a_phase1_pytest_pre.txt",
        R6 / "artifacts" / "r6a_phase1_pytest_post.txt",
        R6 / "artifacts" / "r6a_phase1_run_stdout.log",
        R6 / "artifacts" / "phase1_runs",
        R6 / "artifacts" / "phase1_initializations",
        R6 / "artifacts" / "phase1_firewall_cpu_negative",
        R6 / "reports" / "r6a_phase1_firewall_confirmation.md",
        R6 / "reports" / "r6a_phase1_multirepeat_pairing_report.md",
        R6 / "reports" / "r6a_phase1_decision_packet.md",
        R6 / "scripts" / "phase1_condition_solver.py",
        R6 / "scripts" / "run_phase1_multirepeat_pairing_audit.py",
        R6 / "scripts" / "finalize_phase1_reports.py",
        R6 / "tests" / "test_r6a_phase1_multirepeat.py",
    ]
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            files.add(root)
        else:
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    files.add(path)
    return sorted(files)


def manifest_for_files(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({"path": rel(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)})
    return {"created_utc": utc_now(), "file_count": len(rows), "files": rows}


def archive_phase1(before_manifest: dict[str, Any]) -> dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive_path = ARCHIVE_DIR / f"R6A_PHASE1_FROZEN_FOR_PHASE2_{stamp}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for row in before_manifest["files"]:
            path = PROJECT_ROOT / row["path"]
            zf.write(path, f"R6A_PHASE1_FROZEN_FOR_PHASE2/{row['path']}")
        zf.writestr("R6A_PHASE1_FROZEN_FOR_PHASE2/_archive_manifest/phase1_file_manifest.json", json.dumps(before_manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(archive_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    payload = {
        "created_utc": utc_now(),
        "archive_path": str(archive_path),
        "zip_size_bytes": archive_path.stat().st_size,
        "zip_sha256": file_sha256(archive_path),
        "integrity_test_passed": bad is None,
        "file_count": len(before_manifest["files"]),
        "archive_member_count": len(names),
        "note": "Phase 1 files were read and archived only; no Phase 1 artifact path was modified.",
    }
    write_json(CONTRACT_DIR / "phase1_archive_manifest.json", payload)
    return payload


def verify_source_pin() -> dict[str, Any]:
    proc = subprocess.run(["git", "-C", str(REFERENCE_ROOT), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    status = subprocess.run(["git", "-C", str(REFERENCE_ROOT), "status", "--short"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    source_files = [
        "preprocessing.py",
        "arc_compressor.py",
        "train.py",
        "solution_selection.py",
        "layers.py",
        "initializers.py",
        "multitensor_systems.py",
    ]
    rows = [{"path": name, "sha256": file_sha256(REFERENCE_ROOT / name), "size_bytes": (REFERENCE_ROOT / name).stat().st_size} for name in source_files]
    payload = {
        "created_utc": utc_now(),
        "pinned_commit": PINNED_COMMIT,
        "git_rev_parse_head": proc.stdout.strip(),
        "git_rev_parse_returncode": proc.returncode,
        "git_status_short": status.stdout.strip().splitlines(),
        "source_pin_verified": proc.returncode == 0 and proc.stdout.strip() == PINNED_COMMIT,
        "source_files": rows,
        "audit_source_policy": "Audit only local source at the pinned commit; do not audit current upstream master.",
    }
    write_json(CONTRACT_DIR / "upstream_source_pin_verification.json", payload)
    return payload


def write_source_contract(pin: dict[str, Any]) -> dict[str, Any]:
    claims = [
        {
            "claim_id": "IMPORT_DEFAULT_DEVICE",
            "source_file": "arc_compressor.py",
            "line_range": "8-11",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "The module resets NumPy/Torch seeds, sets float32, and requests CUDA as the default device at import.",
        },
        {
            "claim_id": "TASK_RECONSTRUCTS_FROM_RAW",
            "source_file": "preprocessing.py",
            "line_range": "15-28",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "Task construction collects shapes, predicts solution shapes, constructs multitensor system, masks, problem tensor, and optional solution tensor from the supplied problem.",
        },
        {
            "claim_id": "PREDICTED_QUERY_SHAPE_FLAGS",
            "source_file": "preprocessing.py",
            "line_range": "44-62",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "Query output shape prediction is controlled by in_out_same_size and all_out_same_size, otherwise maximum dimensions are used.",
        },
        {
            "claim_id": "COLOR_CANONICALIZATION",
            "source_file": "preprocessing.py",
            "line_range": "80-89",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "Known raw colors from support inputs/outputs and query inputs are sorted into the task color list, with black added.",
        },
        {
            "claim_id": "PROBLEM_AND_MASK_DEVICE",
            "source_file": "preprocessing.py",
            "line_range": "95-156",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "Problem and mask tensors are moved to the current Torch default device; masks are converted to the default dtype.",
        },
        {
            "claim_id": "LOSS_BRANCH_FLAGS",
            "source_file": "train.py",
            "line_range": "65-87",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "The reconstruction loss grid-size branch uses in_out_same_size, all_out_same_size, and all_in_same_size.",
        },
        {
            "claim_id": "POSTPROCESS_MASK_SHAPE_DEPENDENCE",
            "source_file": "layers.py",
            "line_range": "542-570",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "Mask postprocessing uses task shapes to add large negative modifiers outside per-example max dimensions.",
        },
        {
            "claim_id": "STOCHASTIC_LATENT_SAMPLE",
            "source_file": "layers.py",
            "line_range": "123-125",
            "pinned_commit": PINNED_COMMIT,
            "classification": "DIRECT_SOURCE_FACT",
            "claim": "The forward path samples latent noise with torch.randn; an exact group action on RNG streams is not defined in source.",
        },
        {
            "claim_id": "EXAMPLE_AXIS_STATE",
            "source_file": "multitensor_systems.py",
            "line_range": "17-33,53-68",
            "pinned_commit": PINNED_COMMIT,
            "classification": "SOURCE_DERIVED_INFERENCE",
            "claim": "When dims[0] is set, multitensor tensors have an example axis whose length is task.n_examples.",
        },
        {
            "claim_id": "ADAM_EMPTY_STEP_ZERO",
            "source_file": "train.py",
            "line_range": "49-109",
            "pinned_commit": PINNED_COMMIT,
            "classification": "SOURCE_DERIVED_INFERENCE",
            "claim": "Adam optimizer state is empty before the first optimizer step and is created by normal optimizer.step execution.",
        },
        {
            "claim_id": "D4_FULL_PIPELINE",
            "source_file": "initializers.py,layers.py,arc_compressor.py",
            "line_range": "initializers.py:122-168; layers.py:310-524; arc_compressor.py:37-39,76-89",
            "pinned_commit": PINNED_COMMIT,
            "classification": "UNKNOWN_REQUIRES_RUNTIME_CHECK",
            "claim": "The source contains x/y and directional symmetry machinery, but a complete D4 whole-pipeline null requires explicit action on raw grid, shape, masks, directions, axes, trainable state, and inverse scoring.",
        },
    ]
    payload = {
        "created_utc": utc_now(),
        "pinned_commit": PINNED_COMMIT,
        "source_pin_verified": pin["source_pin_verified"],
        "claims": claims,
        "j_classification_policy": {
            "default": "J_UNKNOWN_REQUIRES_MANUAL_REVIEW",
            "reason": "No D4 transform is labelled J_NULL_CANDIDATE without a source-backed full-pipeline action and CPU evidence.",
        },
    }
    write_json(CONTRACT_DIR / "r6a_phase2_symmetry_contract.json", payload)
    return payload


def run_cpu_child() -> dict[str, Any]:
    cmd = [sys.executable, str(R6 / "scripts" / "phase2_cpu_preflight_child.py")]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=7200)
    (PREFLIGHT_DIR / "cpu_child_stdout.log").write_text(proc.stdout, encoding="utf-8")
    summary_path = PREFLIGHT_DIR / "cpu_child_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary["child_command"] = cmd
    summary["child_returncode"] = proc.returncode
    summary["child_stdout_tail"] = proc.stdout[-4000:]
    write_json(summary_path, summary)
    if proc.returncode != 0:
        raise RuntimeError(f"CPU child failed with code {proc.returncode}")
    return summary


def frozen_ledger_reanalysis() -> dict[str, Any]:
    structural_path = PREFLIGHT_DIR / "s_cycle_invariance_structural_reanalysis.json"
    structural = read_json(structural_path)
    structural_sha = stable_hash(structural)
    per_task = read_json(R6 / "artifacts" / "r6a_phase1_per_task_results.json")["tasks"]
    aggregate = read_json(R6 / "artifacts" / "r6a_phase1_aggregate_results.json")
    status_by_task = {row["task_id"]: row for row in structural["tasks"]}
    rows = []
    for task in per_task:
        srow = status_by_task[task["task_id"]]
        rows.append(
            {
                "task_id": task["task_id"],
                "stratum": task["stratum"],
                "s_cycle_invariant": srow["s_cycle_invariant"],
                "lemma_assumption_status": srow["lemma_assumption_status"],
                "mean_delta": task["mean_delta"],
                "positive_seed_count": task["positive_seed_count"],
                "delta_gt_nu_seed_count": task["delta_gt_nu_seed_count"],
            }
        )
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["s_cycle_invariant"]), []).append(row["mean_delta"])
    payload = {
        "created_utc": utc_now(),
        "outcome_files_read_after_structural_sha_written": True,
        "structural_sha256_used": structural_sha,
        "phase1_verdict_unchanged": aggregate["final_verdict"],
        "rows": rows,
        "grouped_mean_delta": {key: sum(vals) / len(vals) for key, vals in grouped.items()},
        "interpretation": "Exploratory frozen-ledger reanalysis only; it does not alter the Phase 1 verdict.",
    }
    write_json(PREFLIGHT_DIR / "frozen_ledger_reanalysis.json", payload)
    return payload


def write_report(pin: dict[str, Any], archive: dict[str, Any], contract: dict[str, Any], child: dict[str, Any], ledger: dict[str, Any], immutability: dict[str, Any]) -> None:
    cpu = read_json(PREFLIGHT_DIR / "cpu_execution_proof.json")
    c_raw = read_json(PREFLIGHT_DIR / "c_raw_color_preflight.json")
    p_strict = read_json(PREFLIGHT_DIR / "p_strict_preflight.json")
    structural = read_json(PREFLIGHT_DIR / "s_cycle_invariance_structural_reanalysis.json")
    lines = [
        "# R6A Phase2 Source-Grounded Symmetry Preflight",
        "",
        f"- pinned upstream commit: {PINNED_COMMIT}",
        f"- source pin verified: {pin['source_pin_verified']}",
        f"- Phase 1 archive: {archive['archive_path']}",
        f"- Phase 1 archive SHA256: {archive['zip_sha256']}",
        f"- Phase 1 verdict unchanged: {ledger['phase1_verdict_unchanged']}",
        f"- Phase 1 immutability after Phase2: {immutability['all_match_before']}",
        f"- CPU isolation: {cpu['classification']}",
        f"- final seed proof: {cpu['fixed_seed']}",
        f"- tau_forward_abs: {p_strict['tau_forward']['tau_forward_abs']}",
        f"- tau_trajectory_abs: {p_strict['tau_trajectory']['tau_trajectory_abs']}",
        "",
        "## S Cycle-Invariance",
        "",
        json.dumps(
            [
                {
                    "task_id": row["task_id"],
                    "s_cycle_invariant": row["s_cycle_invariant"],
                    "lemma_assumption_status": row["lemma_assumption_status"],
                    "task_shapes_equal_under_s": row["task_shapes_equal_under_s"],
                    "task_masks_equal_under_s": row["task_masks_equal_under_s"],
                    "predicted_query_shape_equal_under_s": row["predicted_query_shape_equal_under_s"],
                    "task_flag_flip_under_s": row["task_flag_flip_under_s"],
                }
                for row in structural["tasks"]
            ],
            indent=2,
            sort_keys=True,
        ),
        "",
        "## C-Raw",
        "",
        json.dumps(c_raw, indent=2, sort_keys=True),
        "",
        "## P-Strict",
        "",
        json.dumps(
            {
                "tau_forward": p_strict["tau_forward"],
                "tau_trajectory": {
                    "tau_trajectory_abs": p_strict["tau_trajectory"]["tau_trajectory_abs"],
                    "tau_trajectory_rel": p_strict["tau_trajectory"]["tau_trajectory_rel"],
                    "derivation": p_strict["tau_trajectory"]["derivation"],
                },
                "actuals": [
                    {
                        "task_id": row["task_id"],
                        "classification": row["classification"],
                        "step0": row["step0"],
                        "step0_within_tau_forward": row["step0_within_tau_forward"],
                        "trajectory_within_tau_trajectory": row["trajectory_within_tau_trajectory"],
                        "trace": row["trace"],
                    }
                    for row in p_strict["actual_p_strict_preflights"]
                ],
                "optimizer_state_rule": p_strict["optimizer_state_rule"],
            },
            indent=2,
            sort_keys=True,
        ),
        "",
        "## CPU Proof",
        "",
        json.dumps(cpu, indent=2, sort_keys=True),
        "",
        "## Source Contract",
        "",
        json.dumps(contract, indent=2, sort_keys=True),
        "",
        "## Frozen-Ledger Reanalysis",
        "",
        json.dumps(ledger, indent=2, sort_keys=True),
        "",
    ]
    write_text(REPORT_PATH, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run R6A Phase2 source-grounded symmetry preflight.")
    parser.parse_args()
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    before = manifest_for_files(phase1_paths())
    write_json(CONTRACT_DIR / "phase1_immutability_before.json", before)
    archive = archive_phase1(before)
    pin = verify_source_pin()
    if not pin["source_pin_verified"]:
        raise RuntimeError("Pinned upstream source verification failed")
    contract = write_source_contract(pin)
    child = run_cpu_child()
    ledger = frozen_ledger_reanalysis()
    after = manifest_for_files([PROJECT_ROOT / row["path"] for row in before["files"]])
    before_by_path = {row["path"]: row for row in before["files"]}
    after_rows = []
    all_match = True
    for row in after["files"]:
        old = before_by_path[row["path"]]
        matches = old["sha256"] == row["sha256"] and old["size_bytes"] == row["size_bytes"]
        all_match = all_match and matches
        after_rows.append({**row, "matches_before": matches})
    immutability = {
        "created_utc": utc_now(),
        "file_count": len(after_rows),
        "all_match_before": all_match,
        "files": after_rows,
    }
    write_json(CONTRACT_DIR / "phase1_immutability_after.json", immutability)
    write_report(pin, archive, contract, child, ledger, immutability)
    result = {
        "status": "R6A_PHASE2_PREFLIGHT_COMPLETE",
        "created_utc": utc_now(),
        "archive": archive,
        "source_pin_verified": pin["source_pin_verified"],
        "cpu_child": child,
        "phase1_immutability": {"file_count": immutability["file_count"], "all_match_before": immutability["all_match_before"]},
        "report": str(REPORT_PATH),
    }
    write_json(PREFLIGHT_DIR / "r6a_phase2_preflight_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
