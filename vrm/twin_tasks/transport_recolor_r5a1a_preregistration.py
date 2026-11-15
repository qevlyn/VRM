from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/mnt/d/VRM")
V2_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/transport_recolor_r5a0_v2"
DEFAULT_OUT = V2_ARTIFACT_ROOT / "r5a1a_preregistration_report.json"
PREREG_VERSION = "R5A1A_SHARED_RENDERER_CONTROL_PREFLIGHT_V1"

OUTCOME_CATEGORIES = (
    "CONTROL_UNDERFIT_OR_INVALID",
    "PRACTICAL_FACTORIZATION_ADVANTAGE_SUPPORTED",
    "EXPLICIT_FACTORIZATION_NOT_NECESSARY_ON_CURRENT_GRAMMAR",
    "CAPACITY_SENSITIVE_OR_INCONCLUSIVE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_granularity(count: int) -> float:
    return 1.0 / count


def build_report() -> dict[str, Any]:
    dataset_manifest = load_json(V2_ARTIFACT_ROOT / "dataset_manifest.json")
    initialization = load_json(V2_ARTIFACT_ROOT / "initialization_report.json")
    aggregate = load_json(V2_ARTIFACT_ROOT / "aggregate_seed_summary.json")

    train_count = int(dataset_manifest["split_episode_counts"]["train"])
    dev_count = int(dataset_manifest["split_episode_counts"]["dev"])
    sealed_count = int(dataset_manifest["split_episode_counts"]["sealed"])
    train_visible_class_count = 80 * 60
    dev_visible_class_count = 24 * 60
    sealed_class_count = 24 * 60
    sealed_heldout_mixed_count = 4 * sealed_class_count
    v2_params = int(initialization["trainable_parameters"])

    return {
        "preregistration_version": PREREG_VERSION,
        "status": "CONTROL_DESIGN_AND_PREFLIGHT_ONLY",
        "neutrality_statement": (
            "This preregistration does not state or assume that a generic shared renderer is expected to pass or fail. "
            "It fixes allowed interfaces, forbidden interfaces, thresholds, seed consistency rules, and outcome categories before any R5A-1A shared-renderer training."
        ),
        "prohibited_actions_in_this_task": {
            "training_started": False,
            "model_architecture_changed": False,
            "v2_grammar_or_split_changed": False,
            "v2_checkpoints_changed": False,
            "historical_v2_verdict_changed": False,
        },
        "source_artifact_anchors": {
            "dataset_manifest": {
                "path": str(V2_ARTIFACT_ROOT / "dataset_manifest.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "dataset_manifest.json"),
            },
            "initialization_report": {
                "path": str(V2_ARTIFACT_ROOT / "initialization_report.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "initialization_report.json"),
            },
            "aggregate_seed_summary": {
                "path": str(V2_ARTIFACT_ROOT / "aggregate_seed_summary.json"),
                "sha256": sha256_file(V2_ARTIFACT_ROOT / "aggregate_seed_summary.json"),
            },
        },
        "fixed_benchmark_anchor": {
            "benchmark": "transport_recolor_r5a0_v2",
            "split_episode_counts": dataset_manifest["split_episode_counts"],
            "train_visible_classes": ["C", "H", "V", "L", "K", "C->L", "L->H", "H->K", "K->V"],
            "heldout_mixed_classes": ["L->C", "H->L", "C->K", "K->C"],
            "v2_trainable_parameter_count": v2_params,
            "v2_reference_heldout_mixed_exact_by_seed": {
                seed: row["final_sealed_metrics"]["macro_exact_by_group"]["held_out_mixed"]
                for seed, row in aggregate["final_metrics_by_seed"].items()
            },
            "metric_granularity": {
                "train_overall_exact": exact_granularity(train_count),
                "dev_overall_exact": exact_granularity(dev_count),
                "sealed_overall_exact": exact_granularity(sealed_count),
                "train_visible_class_exact": exact_granularity(train_visible_class_count),
                "dev_visible_class_exact": exact_granularity(dev_visible_class_count),
                "sealed_per_class_exact": exact_granularity(sealed_class_count),
                "sealed_heldout_mixed_macro_episode_granularity": exact_granularity(sealed_heldout_mixed_count),
            },
        },
        "shared_renderer_allowed_interfaces": [
            "same raw grid tokenization and positional encoding as V2",
            "generic support/query encoding",
            "generic learned output-cell query tokens",
            "generic cross-attention from output cells to support/query tokens",
            "generic MLPs and final per-cell color logits",
            "generic output-size prediction based only on raw support/query information allowed by the benchmark",
            "generic attention, including attention that may internally learn correspondence without a hand-specified semantic transport/color interface",
        ],
        "shared_renderer_forbidden_interfaces": [
            "explicit source-coordinate prediction",
            "explicit pointer distribution or pointer supervision",
            "copy gate",
            "explicit transport representation",
            "explicit color-transition representation",
            "factor-labelled latent channels",
            "auxiliary geometry/recolor losses",
            "oracle/program/factor metadata or derived equivalents",
            "sealed-outcome-driven resizing, decoder repair, optimizer changes, threshold changes, margin changes, or variant selection",
        ],
        "generic_attention_policy": (
            "Generic attention is allowed and must not be classified as forbidden merely because it could internally learn correspondence. "
            "The prohibition is only against a hand-specified semantic transport/color interface or oracle-equivalent factor channel."
        ),
        "planned_shared_variants": {
            "shared_matched": {
                "purpose": "parameter-matched generic shared-renderer control",
                "parameter_budget_rule": "predeclare before training; target V2 parameter count with at most +/-10% deviation unless implementation arithmetic makes that impossible, in which case record the deviation before training",
                "v2_parameter_anchor": v2_params,
                "allowed_capacity_range_inclusive": [int(round(v2_params * 0.90)), int(round(v2_params * 1.10))],
                "semantic_factor_interface_allowed": False,
            },
            "shared_large": {
                "purpose": "larger generic shared-renderer capacity control",
                "parameter_budget_rule": "predeclare before training; target approximately 4x V2 parameters with at most +/-10% deviation unless implementation arithmetic makes that impossible, in which case record the deviation before training",
                "v2_parameter_anchor": v2_params,
                "allowed_capacity_range_inclusive": [int(round(v2_params * 4 * 0.90)), int(round(v2_params * 4 * 1.10))],
                "semantic_factor_interface_allowed": False,
            },
        },
        "pre_registered_thresholds": {
            "train_fit_adequacy_gate": {
                "applies_to": "each shared variant and each seed before interpreting held-out mixed results",
                "requirements": {
                    "train_visible_overall_exact_min": 0.995,
                    "train_visible_seen_mixed_macro_exact_min": 0.995,
                    "each_train_visible_operation_exact_min": 0.99,
                },
                "failure_outcome": "CONTROL_UNDERFIT_OR_INVALID",
                "justification": (
                    "The training split has 43,200 visible examples, so overall exact granularity is about 2.31e-5. "
                    "Each visible operation has 4,800 train examples, granularity about 2.08e-4. "
                    "The gate allows small optimization imperfections while excluding controls that have not learned the visible grammar well enough for held-out failure to be interpretable."
                ),
            },
            "practically_meaningful_heldout_mixed_effect_floor": {
                "absolute_exact_gap_vs_v2_min": 0.03,
                "comparison_metric": "sealed held-out mixed macro exact",
                "must_hold_for": "both shared_matched and shared_large, all three shared seeds, after train-fit adequacy passes",
                "justification": (
                    "Held-out mixed has 5,760 sealed examples across four classes, granularity about 1.74e-4. "
                    "A 0.03 floor is roughly 173 held-out examples, far above metric granularity and deterministic evaluation noise."
                ),
            },
            "equivalence_margin": {
                "absolute_exact_gap_vs_v2_max": 0.01,
                "comparison_metric": "sealed held-out mixed macro exact",
                "must_hold_for": "at least one train-adequate shared variant across all three shared seeds",
                "justification": (
                    "A 0.01 margin is roughly 58 held-out mixed examples, much larger than exact-metric granularity while still strict for this synthetic deterministic benchmark."
                ),
            },
            "seed_consistency_criterion": {
                "within_variant_max_minus_min_heldout_mixed_exact_max": 0.02,
                "condition_must_hold": "for any practical-advantage or equivalence interpretation",
                "failure_outcome": "CAPACITY_SENSITIVE_OR_INCONCLUSIVE",
                "justification": (
                    "A 0.02 seed spread is roughly 115 held-out mixed examples, larger than deterministic metric granularity. "
                    "Larger seed sensitivity prevents collapsing the result into either a modularity or monolithic conclusion."
                ),
            },
            "material_capacity_difference": {
                "shared_large_minus_shared_matched_mean_heldout_mixed_exact_abs_min": 0.02,
                "failure_outcome": "CAPACITY_SENSITIVE_OR_INCONCLUSIVE",
                "justification": "A material matched-vs-large gap indicates capacity sensitivity rather than a clean factorization or non-factorization result.",
            },
        },
        "pre_registered_outcome_categories": list(OUTCOME_CATEGORIES),
        "future_outcome_classification_rules": {
            "evaluation_order": [
                {
                    "outcome": "CONTROL_UNDERFIT_OR_INVALID",
                    "rule": "Use if any shared variant in any seed fails the train-fit adequacy gate, violates allowed-interface constraints, uses a forbidden interface, or breaches sealed-data discipline.",
                    "interpretation": "Do not interpret held-out mixed failure as evidence for factorization.",
                },
                {
                    "outcome": "CAPACITY_SENSITIVE_OR_INCONCLUSIVE",
                    "rule": "Use if train-fit passes but shared_matched and shared_large differ materially by the predeclared capacity criterion, seed spread exceeds the predeclared seed-consistency criterion, or neither practical-effect nor equivalence criteria hold.",
                    "interpretation": "Do not collapse into either a modularity win or a monolithic win.",
                },
                {
                    "outcome": "PRACTICAL_FACTORIZATION_ADVANTAGE_SUPPORTED",
                    "rule": "Use only if shared_matched and shared_large both pass train-fit adequacy and both underperform V2 held-out mixed exact by at least 0.03 in all three seeds without violating seed/capacity consistency rules.",
                    "interpretation": "Supports a practical generalization advantage for explicit V2 factorization on this fixed grammar and training regime only.",
                },
                {
                    "outcome": "EXPLICIT_FACTORIZATION_NOT_NECESSARY_ON_CURRENT_GRAMMAR",
                    "rule": "Use if a train-adequate shared variant matches V2 within the 0.01 equivalence margin on held-out mixed exact across all three seeds without violating seed consistency.",
                    "interpretation": "The current R5A-0 V2 grammar can be solved without the explicit transport/color factorization at the tested scale; this does not imply modularity is generally unhelpful.",
                },
            ],
            "notes": [
                "Thresholds must not be tuned after seeing R5A-1A results.",
                "V2 is a fixed reference for this preregistered comparison; sealed shared outcomes may be used only for final comparison.",
                "Any post-sealed modification creates a new experiment version and a new sealed evaluation surface.",
            ],
        },
        "sealed_data_discipline": {
            "allowed": "evaluate the fixed sealed split once for the pre-registered final comparison",
            "forbidden_uses": [
                "resize a shared variant",
                "add a missing decoder component",
                "alter optimizer or schedule",
                "change train-fit threshold",
                "change effect floor",
                "change equivalence margin",
                "select between architecture variants",
            ],
            "post_sealed_modification_rule": "Any post-sealed modification must create a new experiment version and a new sealed evaluation surface.",
        },
        "claim_boundaries": {
            "practical_factorization_advantage_supported_does_not_mean": [
                "all monolithic architectures fail",
                "hidden tools emerged",
                "the result transfers to ARC",
            ],
            "explicit_factorization_not_necessary_does_not_mean": [
                "modularity is generally unhelpful",
                "explicit factorization can never help",
                "the result transfers beyond this grammar and scale",
            ],
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = build_report()
    write_json(args.out, report)
    print(json.dumps({"path": str(args.out), "sha256": sha256_file(args.out), "version": PREREG_VERSION}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
