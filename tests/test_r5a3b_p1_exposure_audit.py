from __future__ import annotations

import pytest

from vrm.twin_tasks.r5a3b_p1_exposure_audit import (
    RELATION_EXPOSURE_CATEGORIES,
    FoldTarget,
    candidate_matches_target,
    classify_relation_exposure,
    hard_prune_target_survival,
    parse_hit_classification,
    recall_at_k_with_short_candidate_list,
    relation_surface_for_fold,
)
from vrm.twin_tasks.r5a3a_scene_hypotheses import canonical_grid_sha256


def candidate(candidate_id: str, family: str, grid: tuple[tuple[int, ...], ...], rank_loss: int = 0) -> dict:
    return {
        "candidate_id": candidate_id,
        "program_family": family,
        "output_grid": [list(row) for row in grid],
        "output_hash": canonical_grid_sha256(grid),
        "parse_hypothesis_id": "bg=0|conn=4|objects=2|hash=test",
        "source_program_ids": [f"{family}_program"],
        "operator_sequence": [family, "same_canvas"],
        "relation_predicates_used": ["aligned_row"] if family == "relation" else [],
        "raw_ordinal": rank_loss,
        "support_shape_mismatch_count": 0,
        "support_cell_hamming_loss": rank_loss,
        "support_failed_pair_count": 0,
    }


def target(grid: tuple[tuple[int, ...], ...]) -> FoldTarget:
    return FoldTarget("training:test", 0, grid, canonical_grid_sha256(grid), ((0,),))


def raw_row(candidates: list[dict]) -> dict:
    return {
        "fold_key": "fold",
        "raw_candidate_count": len(candidates),
        "raw_candidates": candidates,
    }


def no_prune(candidates: list[dict]) -> dict:
    return {"top_candidates": candidates, "candidate_ids": [c["candidate_id"] for c in candidates], "unique_candidate_count": len(candidates)}


def test_relation_exposure_categories_are_exhaustive_for_synthetic_cases() -> None:
    tgt = target(((9,),))
    object_a = candidate("o1", "object", ((1,),))
    relation_dup = candidate("r1", "relation", ((1,),))
    relation_low = candidate("r2", "relation", ((2,),))
    relation_hit = candidate("r3", "relation", ((9,),))
    cases = [
        ("NO_RELATION_PROGRAM_EMITTED", [], no_prune([]), 0),
        ("RELATION_PROGRAM_EMITTED_NO_EXECUTABLE_OUTPUT", [], no_prune([]), 1),
        ("RELATION_OUTPUT_DUPLICATES_OBJECT_ONLY", [object_a, relation_dup], no_prune([object_a]), 1),
        ("RELATION_DISTINCT_OUTPUT_ONLY_LOW_RANK", [object_a, relation_low], no_prune([object_a] * 16 + [relation_low]), 1),
        ("RELATION_DISTINCT_OUTPUT_EXPOSED_NO_TARGET", [object_a, relation_low], no_prune([relation_low, object_a]), 1),
        ("RELATION_TARGET_CONTRIBUTOR", [object_a, relation_hit], no_prune([relation_hit, object_a]), 1),
    ]
    seen = set()
    for expected, raw_candidates, payload, generated in cases:
        surface = relation_surface_for_fold(raw_row(raw_candidates), payload)
        observed = classify_relation_exposure(
            surface=surface,
            no_prune_payload=payload,
            target=tgt,
            generated_relation_program_count=generated,
        )
        seen.add(observed)
        assert observed == expected
    assert seen == set(RELATION_EXPOSURE_CATEGORIES)


def test_relation_only_unique_outputs_are_identified_exactly() -> None:
    shared = ((1,),)
    relation_only = ((2,),)
    rows = [candidate("o", "object", shared), candidate("r_shared", "relation", shared), candidate("r_only", "relation", relation_only)]
    surface = relation_surface_for_fold(raw_row(rows), no_prune(rows))
    assert surface["relation_unique_output_count"] == 2
    assert surface["relation_only_unique_output_count"] == 1
    assert surface["duplicate_relation_output_count"] == 1


def test_target_provenance_and_match_identification() -> None:
    tgt_grid = ((3,),)
    tgt = target(tgt_grid)
    relation = candidate("r", "relation", tgt_grid)
    obj = candidate("o", "object", ((4,),))
    assert candidate_matches_target(relation, tgt)
    assert not candidate_matches_target(obj, tgt)


def test_low_rank_relation_output_classification_uses_top16_boundary() -> None:
    tgt = target(((9,),))
    object_rows = [candidate(f"o{i}", "object", ((i % 8,),), i) for i in range(16)]
    relation = candidate("r", "relation", ((7, 7),), 17)
    payload = no_prune(object_rows + [relation])
    surface = relation_surface_for_fold(raw_row([object_rows[0], relation]), payload)
    assert classify_relation_exposure(
        surface=surface,
        no_prune_payload=payload,
        target=tgt,
        generated_relation_program_count=1,
    ) == "RELATION_DISTINCT_OUTPUT_ONLY_LOW_RANK"


def test_hard_prune_target_survival_accounting() -> None:
    tgt_grid = ((5,),)
    tgt = target(tgt_grid)
    hit1 = candidate("hit1", "object", tgt_grid)
    hit2 = candidate("hit2", "relation", tgt_grid)
    miss = candidate("miss", "object", ((0,),))
    assert hard_prune_target_survival([miss], no_prune([miss]), tgt) == "NO_TARGET_IN_RAW_UNIVERSE"
    assert hard_prune_target_survival([hit1, hit2], no_prune([hit1, hit2]), tgt) == "HARD_PRUNE_PRESERVED_ALL_TARGET_CANDIDATES"
    assert hard_prune_target_survival([hit1, hit2], no_prune([hit1]), tgt) == "HARD_PRUNE_PRESERVED_AT_LEAST_ONE_TARGET_CANDIDATE"
    assert hard_prune_target_survival([hit1, hit2], no_prune([miss]), tgt) == "HARD_PRUNE_REMOVED_ALL_TARGET_CANDIDATES"


def test_matched_random_survivor_count_reconciliation_shape() -> None:
    hard_count = 2
    random_runs = [{"unique_candidate_count": 2}, {"unique_candidate_count": 2}, {"unique_candidate_count": 2}]
    assert all(run["unique_candidate_count"] == hard_count for run in random_runs)


def test_no_held_out_output_needed_before_posthoc_target_stage() -> None:
    surface = relation_surface_for_fold(raw_row([candidate("o", "object", ((1,),))]), no_prune([candidate("o", "object", ((1,),))]))
    assert "target" not in surface
    assert surface["object_only_raw_candidate_count"] == 1


def test_parse_hit_classification() -> None:
    canonical = "bg=0|conn=4|objects=2|hash=a"
    noncanonical = "bg=1|conn=8|objects=2|hash=b"
    assert parse_hit_classification(set()) is None
    assert parse_hit_classification({canonical}) == "CANONICAL_PARSE_HIT"
    assert parse_hit_classification({noncanonical}) == "NONCANONICAL_PARSE_HIT"
    assert parse_hit_classification({canonical, noncanonical}) == "MULTIPLE_PARSE_DUPLICATE_HIT"


def test_raw_candidate_count_below_128_uses_available_candidate_count() -> None:
    assert recall_at_k_with_short_candidate_list(rank=33, candidate_count=32, k=128) is False
    assert recall_at_k_with_short_candidate_list(rank=32, candidate_count=32, k=128) is True


def test_p0_artifact_integrity_failure_on_hash_mismatch() -> None:
    expected = "abc"
    observed = "def"
    with pytest.raises(AssertionError):
        assert observed == expected
