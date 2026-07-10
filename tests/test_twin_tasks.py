from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrm.twin_tasks.dataset import (
    candidate_order_label_report,
    duplicate_unpaired_report,
    group_twins,
    label_balance,
    paired_payload_bytes,
    paired_payload_hash,
    partition_integrity,
    support_input_bag,
    support_output_bag,
    unpaired_payload_bytes,
)
from vrm.twin_tasks.generator import generate_records, rejection_summary
from vrm.twin_tasks.invariants import assignment_from_record, is_deranged_relative, validate_twin_pair
from vrm.twin_tasks.probes import feature_scores, train_probe
from vrm.twin_tasks.run_pilot import protected_hashes

SEED = 20260628


def _small_suite():
    return generate_records(
        seed=SEED,
        split_pair_counts={"train": 12, "dev": 4, "sealed": 4},
        grid_size=7,
    )


def _first_twins():
    result = _small_suite()
    grouped = group_twins(result.records)
    return grouped[sorted(grouped)[0]]


def test_twin_query_candidates_labels_and_targets_have_required_relation() -> None:
    twin_a, twin_b = _first_twins()
    assert twin_a["query"] == twin_b["query"]
    assert twin_a["candidates"] == twin_b["candidates"]
    assert twin_a["label"] + twin_b["label"] == 1
    assert twin_a["target"] != twin_b["target"]


def test_support_multisets_and_pair_correspondences_have_required_relation() -> None:
    twin_a, twin_b = _first_twins()
    assert support_input_bag(twin_a) == support_input_bag(twin_b)
    assert support_output_bag(twin_a) == support_output_bag(twin_b)
    mapping_a = assignment_from_record(twin_a)
    mapping_b = assignment_from_record(twin_b)
    assert mapping_a != mapping_b
    assert is_deranged_relative(mapping_a, mapping_b)


def test_unpaired_payloads_match_and_paired_payloads_differ() -> None:
    twin_a, twin_b = _first_twins()
    assert unpaired_payload_bytes(twin_a) == unpaired_payload_bytes(twin_b)
    assert paired_payload_bytes(twin_a) != paired_payload_bytes(twin_b)


def test_random_pair_reordering_does_not_alter_unpaired_payload() -> None:
    twin_a, _ = _first_twins()
    reordered = {**twin_a, "supports": list(reversed(twin_a["supports"]))}
    assert unpaired_payload_bytes(twin_a) == unpaired_payload_bytes(reordered)


def test_pair_reordering_does_not_alter_relation_aware_probe_prediction() -> None:
    twin_a, _ = _first_twins()
    reordered = {**twin_a, "supports": list(reversed(twin_a["supports"]))}
    assert feature_scores(twin_a, "P5_RELATION_AWARE_PAIRED", SEED) == feature_scores(
        reordered,
        "P5_RELATION_AWARE_PAIRED",
        SEED,
    )


def test_no_rule_template_twin_or_base_ids_enter_probe_scores() -> None:
    twin_a, _ = _first_twins()
    redacted = {
        **twin_a,
        "base_id": "different",
        "twin": "B",
        "metadata": {"template_id": "forbidden", "rule_id": "forbidden"},
    }
    for probe in ("P1_QUERY_ONLY", "P2_SUPPORT_INPUT_BAG_ONLY", "P3_SUPPORT_OUTPUT_BAG_ONLY", "P4_UNPAIRED_SUPPORT_BAG", "P5_RELATION_AWARE_PAIRED"):
        assert feature_scores(twin_a, probe, SEED) == feature_scores(redacted, probe, SEED)


def test_split_grouping_and_no_cross_split_base_overlap() -> None:
    result = _small_suite()
    base_to_split = {}
    for record in result.records:
        previous = base_to_split.setdefault(record["base_id"], record["split"])
        assert previous == record["split"]
    split_sets = {split: set(ids) for split, ids in result.split_base_ids.items()}
    assert split_sets["train"].isdisjoint(split_sets["dev"])
    assert split_sets["train"].isdisjoint(split_sets["sealed"])
    assert split_sets["dev"].isdisjoint(split_sets["sealed"])


def test_labels_are_balanced_and_candidate_order_not_correlated() -> None:
    result = _small_suite()
    for split in ("train", "dev", "sealed"):
        records = [record for record in result.records if record["split"] == split]
        assert label_balance(records)["balanced"]
        assert candidate_order_label_report(records)["passed"]


def test_deranged_control_preserves_bags_but_destroys_correspondence() -> None:
    twin_a, _ = _first_twins()
    deranged = {**twin_a, "supports": []}
    outputs = [
        (support["output"], support["output_symbol_index"])
        for support in twin_a["supports"]
    ]
    for idx, support in enumerate(twin_a["supports"]):
        deranged["supports"].append(
            {
                **support,
                "output": outputs[(idx + 1) % len(outputs)][0],
                "output_symbol_index": outputs[(idx + 1) % len(outputs)][1],
            }
        )
    assert support_input_bag(twin_a) == support_input_bag(deranged)
    assert support_output_bag(twin_a) == support_output_bag(deranged)
    assert assignment_from_record(twin_a) != assignment_from_record(deranged)


def test_generation_is_deterministic_for_same_seed() -> None:
    first = _small_suite()
    second = _small_suite()
    payload_a = json.dumps(first.records, sort_keys=True)
    payload_b = json.dumps(second.records, sort_keys=True)
    assert hashlib.sha256(payload_a.encode()).hexdigest() == hashlib.sha256(payload_b.encode()).hexdigest()


def test_rejection_reasons_are_recorded_for_rejected_base_instances() -> None:
    summary = rejection_summary([{"base_id": "bad", "split": "train", "reason": "forced_failure"}])
    assert summary["rejected_count"] == 1
    assert summary["rejection_reasons"] == {"forced_failure": 1}


def test_exact_invariant_validator_accepts_generated_twins() -> None:
    twin_a, twin_b = _first_twins()
    report = validate_twin_pair(twin_a, twin_b)
    assert report["all_passed"]


def test_duplicate_and_partition_reports_pass_for_small_suite() -> None:
    result = _small_suite()
    by_split = {split: [record for record in result.records if record["split"] == split] for split in ("train", "dev", "sealed")}
    assert duplicate_unpaired_report(result.records)["passed"]
    assert partition_integrity(by_split)["passed"]


def test_relation_probe_is_order_invariant_after_training() -> None:
    result = _small_suite()
    train = [record for record in result.records if record["split"] == "train"]
    dev = [record for record in result.records if record["split"] == "dev"]
    sealed = [record for record in result.records if record["split"] == "sealed"]
    run = train_probe(
        probe_name="P5_RELATION_AWARE_PAIRED",
        train_records=train,
        dev_records=dev,
        sealed_records=sealed,
        seed=SEED,
        epochs=2,
        batch_size=8,
        learning_rate=3e-4,
        patience=2,
    )
    assert run["best_dev_accuracy"] >= 0.9
    sample = train[0]
    reordered = {**sample, "supports": list(reversed(sample["supports"]))}
    assert feature_scores(sample, "P5_RELATION_AWARE_PAIRED", SEED) == feature_scores(
        reordered,
        "P5_RELATION_AWARE_PAIRED",
        SEED,
    )


def test_existing_v1_1_protected_paths_are_not_touched_by_generation() -> None:
    before = protected_hashes()
    _small_suite()
    after = protected_hashes()
    assert before == after
