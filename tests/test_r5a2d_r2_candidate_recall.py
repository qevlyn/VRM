from __future__ import annotations

import inspect
from pathlib import Path

from vrm.twin_tasks import compiled_topology_r5a2c_r2_data as topology_data
from vrm.twin_tasks import compiled_topology_r5a2c_r2_executor as topology_executor
from vrm.twin_tasks import r5a2d_r2_candidate_recall_audit as audit
from vrm.twin_tasks import r5a2d_r2_candidate_surface_audit as surface
from vrm.twin_tasks import transport_recolor_r5a0v2_model as v2_model
from vrm.twin_tasks import transport_recolor_r5a0v2_oracles as v2_oracles
from vrm.twin_tasks.r5a2d_r1_raw_grid_fidelity import compare_raw_grid_fidelity, freeze_grid
from vrm.twin_tasks.r5a2d_r2_candidate_enumeration import FoldVisibleInputs, enumerate_topology_candidates, enumerate_v2_candidates
from vrm.twin_tasks.r5a2d_r2_candidate_ranking import Candidate, K_VALUES, deduplicate_and_rank, recall_at_k, stable_hash


def _grid(value: int = 0):
    return freeze_grid([[value for _ in range(7)] for _ in range(7)])


def _candidate(candidate_id: str, output, loss: int = 0):
    return Candidate(
        source_expert="unit",
        frozen_candidate_id=candidate_id,
        frozen_catalog_hash="h",
        query_index=0,
        output_grid=output,
        output_shape=(7, 7),
        visible_support_count=1,
        support_shape_mismatch_count=0,
        support_cell_hamming_loss=loss,
        support_failed_pair_count=int(loss > 0),
        raw_candidate_trace_ids=(candidate_id,),
        canonical_output_hash=compare_raw_grid_fidelity(output, output, context="h").canonical_source_grid_sha256,
        raw_grid_fidelity_passed=True,
        execution_status="OK",
    )


def test_exact_corpus_manifest_equality_with_r5a2d_r1():
    verification = audit.verify_r1_corpus_manifest()
    assert verification["exact_match"] is True
    assert verification["reconstructed_corpus_manifest_sha256"] == audit.REQUIRED_R1_MANIFEST_SHA256
    assert verification["task_count"] == 1360


def test_no_corpus_fallback_or_reselection():
    verification = audit.verify_r1_corpus_manifest()
    assert verification["required_corpus_root"] == "/mnt/d/VRM/arc-prize-2026-arc-agi-2"
    assert verification["no_corpus_fallback_or_reselection"] is True


def test_candidate_surface_map_does_not_modify_frozen_sources():
    before = audit.protected_hashes()
    report = surface.candidate_surface_report()
    after = audit.protected_hashes()
    assert before == after
    assert report["all_available"] is True


def test_unavailable_candidate_enumeration_has_honest_label():
    row = surface.unavailable_surface_example()
    assert row["candidate_enumeration_available"] is False
    assert row["availability_verdict"] == surface.UNAVAILABLE


def test_no_new_v2_forward_argument_exists():
    parameters = list(inspect.signature(v2_model.SupportConditionedTransportRecolorR5A0V2.forward).parameters)
    assert "force_class" not in parameters
    assert "class_id" not in parameters


def test_no_new_topology_runtime_family_or_tuple_argument_exists():
    raw_parameters = list(inspect.signature(topology_executor.execute_from_raw_grids).parameters)
    assert "family_id" not in raw_parameters
    assert "class_id" not in raw_parameters
    assert "hypothesis" not in raw_parameters


def test_candidate_ids_originate_only_from_frozen_catalogs():
    report = surface.candidate_surface_report()
    v2_row, topo_row = report["surfaces"]
    assert v2_row["raw_candidate_count"] == len(v2_oracles.SEALED_CLASSES)
    assert set(surface.v2_catalog_payload()["candidate_ids"]) == set(v2_oracles.SEALED_CLASSES)
    topo_ids = surface.topology_catalog_payload()["candidate_ids"]
    assert len(topo_ids) == len(topology_data.candidate_hypotheses())
    assert all("seed_reachable_recolor" in item or "border_reachable_recolor" in item or "enclosed_region_recolor" in item for item in topo_ids)


def test_raw_grid_fidelity_passes_for_accepted_invocation_and_rejects_mutations():
    fold = FoldVisibleInputs("unit", 0, _grid(0), (_grid(0),), (_grid(0),))
    result = enumerate_v2_candidates(fold)
    assert result["status"] == "OK"
    assert all(row["passed"] for row in result["fidelity_records"])
    source = freeze_grid([[((x + y) % 9) for x in range(7)] for y in range(7)])
    mutations = [
        source[:-1],
        source + (tuple(0 for _ in range(7)),),
        tuple(tuple((value + 1) % 9 for value in row) for row in source),
        tuple(tuple(reversed(row)) for row in source),
        ((0,),),
        tuple(tuple(0 if (x, y) == (1, 1) else value for x, value in enumerate(row)) for y, row in enumerate(source)),
    ]
    assert all(compare_raw_grid_fidelity(source, mutated, context="mutated").passed is False for mutated in mutations)


def test_hidden_loo_and_sealed_outputs_are_inaccessible_before_scoring():
    task = audit.r1_audit.ArcTaskOpen("unit", "unit.json", "0", (_grid(0), _grid(0)), (_grid(0), _grid(0)), (_grid(0),), "unit", None)
    ledger = audit.build_loo_candidate_ledger([task])
    assert all(row["hidden_LOO_train_output_accessible_to_candidate_generation"] is False for row in ledger)
    assert all(row["sealed_test_outputs_accessible_before_posthoc_commitment"] is False for row in ledger)
    commitment = audit.build_test_candidate_commitment([task])
    assert commitment["sealed_test_outputs_read"] is False


def test_candidate_rank_tuple_is_deterministic_and_visible_support_only():
    target = _grid(0)
    worse = _candidate("b", target, loss=2)
    better = _candidate("a", target, loss=0)
    first = deduplicate_and_rank([worse, better])
    second = deduplicate_and_rank([worse, better])
    assert [row.rank_tuple() for row in first] == [row.rank_tuple() for row in second]
    assert first[0].support_cell_hamming_loss == 0


def test_candidate_dedup_groups_identical_outputs_and_budget_slots():
    target = _grid(0)
    candidates = [_candidate("b", target, loss=1), _candidate("a", target, loss=0)]
    ranked = deduplicate_and_rank(candidates)
    assert len(ranked) == 1
    assert sorted(ranked[0].raw_candidate_trace_ids) == ["a", "b"]
    assert recall_at_k(ranked, target, 128) is True


def test_k_values_effective_k_and_monotonic_recall():
    assert K_VALUES == [1, 4, 16, 64, 128]
    target = _grid(1)
    ranked = [_candidate("a", _grid(0)), _candidate("b", target)]
    recalls = [recall_at_k(ranked, target, k) for k in K_VALUES]
    assert recalls == sorted(recalls)
    assert min(128, len(ranked)) == 2


def test_exact_support_and_all_catalog_are_reported_separately():
    task = audit.r1_audit.ArcTaskOpen("unit", "unit.json", "0", (_grid(0), _grid(0)), (_grid(0), _grid(0)), (_grid(0),), "unit", None)
    ledger = audit.build_loo_candidate_ledger([task])
    scored = audit.score_loo_candidate_recall([task], ledger)
    metrics = scored["metrics"]["per_expert"]["V2_transport_recolor"]
    assert "EXACT_SUPPORT_CANDIDATE_RECALL" in metrics
    assert "RANKED_ALL_CATALOG_CANDIDATE_RECALL" in metrics


def test_original_top1_is_not_conflated_with_catalog_ranked_top1():
    task = audit.r1_audit.ArcTaskOpen("unit", "unit.json", "0", (_grid(0), _grid(0)), (_grid(0), _grid(0)), (_grid(0),), "unit", None)
    ledger = audit.build_loo_candidate_ledger([task])
    scored = audit.score_loo_candidate_recall([task], ledger)
    metrics = scored["metrics"]["per_expert"]["V2_transport_recolor"]["RANKED_ALL_CATALOG_CANDIDATE_RECALL"]
    assert "original_frozen_executor_top1" in metrics
    assert "catalog_ranked_top1" in metrics


def test_oracle_union_is_not_conflated_with_merged_budget_union():
    task = audit.r1_audit.ArcTaskOpen("unit", "unit.json", "0", (_grid(0), _grid(0)), (_grid(0), _grid(0)), (_grid(0),), "unit", None)
    ledger = audit.build_loo_candidate_ledger([task])
    scored = audit.score_loo_candidate_recall([task], ledger)
    union = scored["metrics"]["union_metrics"]["RANKED_ALL_CATALOG_CANDIDATE_RECALL"]
    assert "ORACLE_PER_EXPERT_CANDIDATE_UNION_AT_K" in union
    assert "UNROUTED_MERGED_CANDIDATE_POOL_RECALL_AT_K" in union


def test_topology_loo_support_infeasibility_is_not_output_failure():
    episode = topology_data.make_episode("golden_suite", "border_reachable_recolor", 1, 7, None, "diagonal_bridge_a", 0)
    fold = FoldVisibleInputs("topo", 0, episode.supports[0].input, (episode.supports[1].input, episode.supports[2].input), (episode.supports[1].output, episode.supports[2].output))
    result = enumerate_topology_candidates(fold)
    assert result["status"] == "TOPOLOGY_LOO_INFEASIBLE_SUPPORT_CONTRACT"
    assert result["raw_executable_candidate_count"] == 0


def test_topology_seed_only_is_reported_separately():
    scored = {"fold_scores": [], "metrics": {"union_metrics": {"RANKED_ALL_CATALOG_CANDIDATE_RECALL": {"UNROUTED_MERGED_CANDIDATE_POOL_RECALL_AT_K": {f"recall_at_{k}": {"numerator": 0} for k in K_VALUES}}}}}
    report = audit.aggregate_recall_metrics([], [])
    assert "topology_seed_only_report" in report


def test_no_parameter_optimizer_backward_or_checkpoint_io_exists():
    report = audit.parameter_free_integrity()
    assert report["passed"] is True
    assert report["forbidden_code_terms_found"] == []


def test_protected_hashes_remain_unchanged_by_surface_and_ranking_spec():
    before = audit.protected_hashes()
    _ = surface.candidate_surface_report()
    _ = audit.candidate_ranking_spec()
    after = audit.protected_hashes()
    assert before == after


def test_stable_hashes_ignore_timestamp_fields_for_two_run_comparison():
    left = {"created_utc": "a", "value": 1, "nested": {"observed_runtime_seconds": 1.0, "x": 2}}
    right = {"created_utc": "b", "value": 1, "nested": {"observed_runtime_seconds": 2.0, "x": 2}}
    assert stable_hash(audit.strip_unstable(left)) == stable_hash(audit.strip_unstable(right))
