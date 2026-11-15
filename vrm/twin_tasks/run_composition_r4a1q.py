from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .composition_r4a1q_reconcile import (
    APPROVED_CHECKPOINTS,
    DEFAULT_SEEDS,
    R4A1Q_DATA_OUT,
    balanced_r4a0r_sealed_sample,
    build_metric_tensor_comparison,
    checkpoint_hashes,
    cuda_environment_report,
    hash_status,
    independent_pointer_coordinate_recomputation,
    load_frozen_model,
    parameter_hashes,
    protected_hashes,
    r4a1p_subset_sample,
    raw_pointer_metrics,
    reproduce_old_metrics,
    run_pytest,
    sample_manifest,
    save_json,
    save_text,
)

VERDICT_SCOPE_MISMATCH = "R4A1Q_METRICS_RECONCILED_SCOPE_MISMATCH"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


def write_topk_artifacts(data_out: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path = data_out / "per_cell_topk_forensics.jsonl"
    txt_path = data_out / "per_cell_topk_forensics.txt"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    lines: list[str] = []
    for row in rows:
        top = ", ".join(f"{item['source_coordinate']}={item['probability']:.12g}/c{item['source_color']}" for item in row["top5_source_coordinates_and_probabilities"])
        lines.append(
            f"seed={row['seed']} episode={row['episode_id']} seq={row['sequence_class']} "
            f"u={row['target_coordinate_u']} v*={row['oracle_source_coordinate_v_star']} "
            f"argmax={row['raw_argmax_source_coordinate']} mass={row['raw_exact_mass_A_u_v_star']:.12g} "
            f"max={row['raw_maximum_probability']:.12g} entropy={row['raw_entropy']:.12g} "
            f"target_color={row['target_color']} top5=[{top}]"
        )
    save_text(txt_path, "\n".join(lines) + "\n")


def write_summary(data_out: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# R4A-1Q Raw Pointer Metric Reconciliation", ""]
    lines.append(f"Outcome: `{aggregate['verdict']}`")
    lines.append("")
    lines.append("The same raw pointer tensor is used, but the historical summaries reduce different masks:")
    lines.append("")
    lines.append("- R4A-1P raw oracle-coordinate mass is foreground-only.")
    lines.append("- R4A-0R pointer entropy and max probability are all-cell means over all 49 target rows, including background rows.")
    lines.append("")
    lines.append("|seed|fg argmax|fg mass|fg max|fg entropy|all-cell max|all-cell entropy|same-color wrong|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for seed, row in aggregate["raw_pointer_metrics_by_seed"].items():
        lines.append(
            f"|{seed}|{row['raw_argmax_coordinate_accuracy']:.6f}|"
            f"{row['mean_raw_oracle_coordinate_mass']:.6f}|"
            f"{row['mean_raw_max_probability_on_foreground_rows']:.6f}|"
            f"{row['mean_raw_entropy_on_foreground_rows']:.6f}|"
            f"{row['old_r4a0r_scope_all_cell_mean_raw_max_probability']:.6f}|"
            f"{row['old_r4a0r_scope_all_cell_mean_raw_entropy']:.6f}|"
            f"{row['same_color_wrong_coordinate_rate']:.6f}|"
        )
    save_text(data_out / "summary.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), required=True)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-out", type=Path, default=R4A1Q_DATA_OUT)
    args = parser.parse_args(argv)

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    data_out = Path(args.data_out)
    data_out.mkdir(parents=True, exist_ok=True)

    save_json(data_out / "environment_report.json", cuda_environment_report())
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    protected_before = protected_hashes(data_out)
    checkpoint_before = checkpoint_hashes()
    save_json(data_out / "protected_hashes_before.json", protected_before)
    save_json(data_out / "checkpoint_hashes_before.json", checkpoint_before)
    save_json(
        data_out / "evaluation_sample_manifest.json",
        sample_manifest(seeds),
    )

    pytest_result = run_pytest(data_out)
    if not pytest_result["passed"]:
        protected_after = protected_hashes(data_out)
        checkpoint_after = checkpoint_hashes()
        save_json(data_out / "protected_hashes_after.json", protected_after)
        save_json(data_out / "checkpoint_hashes_after.json", checkpoint_after)
        aggregate = {
            "verdict": VERDICT_INCONCLUSIVE,
            "pytest_result": pytest_result,
            "protected_hash_status": hash_status(protected_before, protected_after),
            "checkpoint_hash_status": hash_status({key: row["sha256"] for key, row in checkpoint_before.items()}, {key: row["sha256"] for key, row in checkpoint_after.items()}),
        }
        save_json(data_out / "aggregate_seed_summary.json", aggregate)
        print(json.dumps({"verdict": VERDICT_INCONCLUSIVE, "pytest_passed": False}, indent=2, sort_keys=True), flush=True)
        return 4

    models_by_seed: dict[int, torch.nn.Module] = {}
    model_provenance: dict[str, Any] = {}
    param_before: dict[str, Any] = {}
    for seed in seeds:
        if seed not in APPROVED_CHECKPOINTS:
            raise RuntimeError(f"unapproved seed {seed}")
        model, provenance = load_frozen_model(seed, device)
        models_by_seed[seed] = model
        model_provenance[str(seed)] = provenance
        param_before[str(seed)] = parameter_hashes(model)

    raw_by_seed: dict[str, Any] = {}
    normalization_by_seed: dict[str, Any] = {}
    old_r4a0r_by_seed: dict[str, Any] = {}
    tensor_profiles: dict[str, Any] = {}
    topk_rows: list[dict[str, Any]] = []
    r4a1p_subset_metrics: dict[str, Any] = {}
    r4a1p_subset_normalization: dict[str, Any] = {}
    subset = r4a1p_subset_sample()

    for seed in seeds:
        sample = balanced_r4a0r_sealed_sample(seed)
        raw, normalization, topk, profile = raw_pointer_metrics(
            models_by_seed[seed],
            sample,
            device,
            args.batch_size,
            topk_limit=100,
            seed=seed,
            dataset_label="r4a0r_v2_sealed_balanced_2007",
        )
        raw_by_seed[str(seed)] = raw
        normalization_by_seed[str(seed)] = normalization
        tensor_profiles[str(seed)] = profile
        topk_rows.extend(topk)
        old_r4a0r_by_seed[str(seed)] = reproduce_old_metrics(models_by_seed[seed], sample, device, args.batch_size)
        if subset:
            subset_raw, subset_norm, _, _ = raw_pointer_metrics(
                models_by_seed[seed],
                subset,
                device,
                args.batch_size,
                topk_limit=0,
                seed=seed,
                dataset_label="r4a1p_length3_probe_subset_256",
            )
            r4a1p_subset_metrics[str(seed)] = subset_raw
            r4a1p_subset_normalization[str(seed)] = subset_norm

    old_r4a1p = independent_pointer_coordinate_recomputation(models_by_seed, device, args.batch_size)
    comparison = build_metric_tensor_comparison(raw_by_seed, old_r4a0r_by_seed, old_r4a1p, tensor_profiles)
    param_after = {str(seed): parameter_hashes(model) for seed, model in models_by_seed.items()}
    checkpoint_after = checkpoint_hashes()
    protected_after = protected_hashes(data_out)

    raw_norm_report = {
        "r4a0r_v2_sealed_balanced_2007": normalization_by_seed,
        "r4a1p_length3_probe_subset_256": r4a1p_subset_normalization,
        "all_rows_sum_to_one_within_1e_6": all(row["passed"] for row in normalization_by_seed.values()) and all(row["passed"] for row in r4a1p_subset_normalization.values()),
    }
    old_metric_report = {
        "r4a0r_evaluate_model_by_seed": old_r4a0r_by_seed,
        "r4a1p_independent_pointer_coordinate_recomputation": old_r4a1p,
    }
    aggregate = {
        "verdict": VERDICT_SCOPE_MISMATCH,
        "mode": args.mode,
        "cuda_required": True,
        "model_provenance": model_provenance,
        "raw_pointer_metrics_by_seed": raw_by_seed,
        "r4a1p_length3_subset_metrics_by_seed": r4a1p_subset_metrics,
        "raw_pointer_normalization_report": raw_norm_report,
        "old_metric_reproduction_by_seed": old_metric_report,
        "metric_tensor_comparison": comparison,
        "pytest_result": pytest_result,
        "parameter_hashes_unchanged": param_before == param_after,
        "checkpoint_hash_status": hash_status({key: row["sha256"] for key, row in checkpoint_before.items()}, {key: row["sha256"] for key, row in checkpoint_after.items()}),
        "protected_hash_status": hash_status(protected_before, protected_after),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "claim_boundary": {
            "r4a0r_claim_remaining": "R4A-0R pointer argmax source-coordinate accuracy remains exact-coordinate foreground argmax routing; its entropy/max summaries are all-cell raw pointer summaries.",
            "r4a1p_claim_remaining": "R4A-1P foreground raw mass on the unique oracle source coordinate remains foreground-only and raw-attention based; it should not be compared to all-cell entropy/max without matching masks.",
        },
    }

    save_json(data_out / "raw_pointer_metrics_by_seed.json", raw_by_seed)
    save_json(data_out / "raw_pointer_normalization_report.json", raw_norm_report)
    save_json(data_out / "old_metric_reproduction_by_seed.json", old_metric_report)
    save_json(data_out / "metric_tensor_comparison.json", comparison)
    save_json(data_out / "checkpoint_hashes_after.json", checkpoint_after)
    save_json(data_out / "protected_hashes_after.json", protected_after)
    write_topk_artifacts(data_out, topk_rows)
    write_summary(data_out, aggregate)
    save_json(data_out / "aggregate_seed_summary.json", aggregate)

    print(json.dumps({"verdict": VERDICT_SCOPE_MISMATCH, "pytest_passed": True}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
