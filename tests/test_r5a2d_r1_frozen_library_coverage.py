from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from vrm.twin_tasks import compiled_topology_r5a2c_r2_data as topology_data
from vrm.twin_tasks import r5a2d_r1_coverage_audit as audit
from vrm.twin_tasks import r5a2d_r1_corpus_resolver as resolver
from vrm.twin_tasks.r5a2d_r1_raw_grid_fidelity import compare_raw_grid_fidelity, freeze_grid


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _tiny_grid(value: int = 0) -> list[list[int]]:
    return [[value for _ in range(7)] for _ in range(7)]


def _mini_arc_corpus(root: Path, *, task_ids: tuple[str, ...] = ("a",)) -> Path:
    tasks = {
        task_id: {
            "train": [{"input": _tiny_grid(0), "output": _tiny_grid(0)}],
            "test": [{"input": _tiny_grid(0)}],
        }
        for task_id in task_ids
    }
    _write_json(root / "arc-agi_training_challenges.json", tasks)
    _write_json(root / "arc-agi_training_solutions.json", {task_id: [_tiny_grid(0)] for task_id in task_ids})
    return root


def _selection(tmp_path: Path) -> resolver.CorpusSelection:
    return resolver.resolve_and_precommit_corpus(
        project_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        environ={"VRM_ARC_COVERAGE_CORPUS": str(_mini_arc_corpus(tmp_path / "arc-agi-2-mini"))},
    )


def _fake_selection(tmp_path: Path, task_count: int = 1) -> resolver.CorpusSelection:
    return resolver.CorpusSelection(
        corpus_name="unit",
        corpus_root=tmp_path,
        corpus_root_relative_or_configured_path=str(tmp_path),
        provenance_statement="unit",
        resolution_rule="unit",
        task_files=tuple(),
        corpus_manifest_sha256="abc123",
        manifest_path=tmp_path / "manifest.json",
        frozen_from_existing_manifest=False,
    )


def _ledger_record(**overrides: object) -> dict[str, object]:
    row = {
        "strict_label_identity": audit.STRICT_FAILED,
        "strict_label_v2": audit.STRICT_FAILED,
        "strict_label_topology": audit.STRICT_FAILED,
        "full_support_fit_identity": False,
        "full_support_fit_v2": False,
        "full_support_fit_topology": False,
        "topology_subfamily_if_any": None,
        "cross_expert_candidate_agreement": "not_multiple_strict_experts",
        "topology_primary_label": audit.STRICT_FAILED,
        "v2_primary_label": audit.STRICT_FAILED,
        "test_input_count": 1,
        "train_pair_count": 4,
        "full_support_status_v2": "ok",
        "full_support_status_topology": "ok",
        "loo_fold_count_attempted_v2": 0,
        "loo_fold_count_attempted_topology": 0,
        "loo_fold_count_attempted_identity": 0,
        "topology_raw_domain_flags": {"grid_size_ok": True, "palette_ok": True, "marker_count_ok": True},
    }
    row.update(overrides)
    return row


def test_deterministic_local_corpus_resolver_and_manifest(tmp_path: Path):
    first = _selection(tmp_path)
    second = resolver.resolve_and_precommit_corpus(
        project_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        environ={"VRM_ARC_COVERAGE_CORPUS": str(tmp_path / "other")},
    )
    assert first.corpus_manifest_sha256 == second.corpus_manifest_sha256
    assert second.frozen_from_existing_manifest is True
    assert [row["relative_path"] for row in first.task_files] == sorted(row["relative_path"] for row in first.task_files)


def test_no_synthetic_fallback_when_no_local_arc_corpus(tmp_path: Path):
    with pytest.raises(resolver.NoLocalArcCorpusError):
        resolver.resolve_and_precommit_corpus(project_root=tmp_path, artifact_root=tmp_path / "artifacts", environ={})


def test_manifest_exists_before_task_grid_interpretation(tmp_path: Path):
    selection = _selection(tmp_path)
    assert selection.manifest_path.exists()
    manifest = json.loads(selection.manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_written_before_task_grid_interpretation"] is True
    tasks = audit.load_open_arc_tasks(selection)
    assert tasks and not hasattr(tasks[0], "sealed_test_outputs_if_present")


def test_deterministic_task_ordering(tmp_path: Path):
    selection = resolver.resolve_and_precommit_corpus(
        project_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
        environ={"VRM_ARC_COVERAGE_CORPUS": str(_mini_arc_corpus(tmp_path / "arc-agi-2-mini", task_ids=("b", "a")))},
    )
    tasks = audit.load_open_arc_tasks(selection)
    assert [task.task_id for task in tasks] == ["training:a", "training:b"]


def test_sealed_outputs_are_posthoc_and_do_not_change_candidates(tmp_path: Path):
    selection = _selection(tmp_path)
    tasks = audit.load_open_arc_tasks(selection)
    ledger = audit.build_task_ledger(selection, tasks)
    before = [row["test_candidates_identity"] for row in ledger]
    sealed = audit.optional_sealed_scoring(selection, tasks, ledger)
    after = [row["test_candidates_identity"] for row in ledger]
    assert before == after
    assert sealed["metric_label"] == "SEALED_POST_HOC_TEST_SCORING"
    assert sealed["sealed_outputs_used_to_change_labels_or_candidates"] is False


def test_v2_receives_only_raw_support_and_query_fields():
    signature = inspect.signature(audit.execute_v2_from_raw_grids)
    assert list(signature.parameters) == ["support_inputs", "support_outputs", "query_input"]


def test_topology_executor_hashes_are_existing_frozen_artifacts():
    report = audit.library_contract_report(_fake_selection(Path("/tmp")))
    topology = report["topology_contract"]
    assert topology["support_pair_count_behavior"] == "exactly 3 support pairs"
    assert topology["closure_domain_hash"]
    assert topology["shortcut_catalog_hash"]


def test_canonical_fidelity_passes_identity_and_rejects_semantic_changes():
    source = freeze_grid([[((x + y) % 9) for x in range(7)] for y in range(7)])
    changed = [
        source[:-1],
        source + (tuple(0 for _ in range(7)),),
        tuple(tuple(0 for _ in range(6)) for _ in range(6)),
        tuple(tuple(1 for _ in row) for row in source),
        tuple(tuple(2 if value == 0 else value for value in row) for row in source),
        tuple(tuple(reversed(row)) for row in source),
        tuple(tuple(0 if (x, y) == (3, 3) else value for x, value in enumerate(row)) for y, row in enumerate(source)),
        ((0,),),
    ]
    assert compare_raw_grid_fidelity(source, source, context="identity").passed is True
    assert all(compare_raw_grid_fidelity(source, grid, context="mutated").passed is False for grid in changed)


def _topology_task(train_count: int, convention: str = "seed_reachable_recolor") -> audit.ArcTaskOpen:
    episode = topology_data.make_episode("golden_suite", convention, 1, 7, 5 if convention == "seed_reachable_recolor" else None, "diagonal_bridge_a", 0)
    pairs = list(episode.supports)
    if train_count == 4:
        pairs.append(topology_data.SupportPair(input=episode.query_input_grid, output=episode.query_output_grid))
    return audit.ArcTaskOpen(
        task_id=f"topology-{train_count}",
        source_relative_path="unit.json",
        task_file_sha256="0",
        train_inputs=tuple(pair.input for pair in pairs[:train_count]),
        train_outputs=tuple(pair.output for pair in pairs[:train_count]),
        test_inputs=(episode.query_input_grid,),
        split_name="unit",
        solution_relative_path=None,
    )


def test_no_support_pair_duplication_deletion_or_fabrication_in_loo():
    result = audit._topology_evaluation(_topology_task(4))
    assert result["strict_label"] == audit.STRICT_VALIDATED
    for index, fold in enumerate(result["fold_results"]):
        assert len(fold["support_indices"]) == 3
        assert index not in fold["support_indices"]
        assert len(fold["support_indices"]) == len(set(fold["support_indices"]))


def test_full_support_fit_is_not_counted_as_strict_loo_coverage():
    result = audit._topology_evaluation(_topology_task(3))
    assert result["full_support_fit"] is True
    assert result["strict_label"] == audit.OUT_SUPPORT
    assert result["loo_detail_status"] == audit.TOPO_LOO_INFEASIBLE


def test_strict_loo_requires_every_fold_to_pass():
    grid = freeze_grid(_tiny_grid(0))
    changed = freeze_grid(_tiny_grid(1))
    task = audit.ArcTaskOpen("id", "unit", "0", (grid, grid), (grid, changed), (grid,), "unit", None)
    result = audit._identity_evaluation(task)
    assert result["strict_label"] == audit.STRICT_FAILED
    assert any(not fold["exact"] for fold in result["fold_results"])


def test_topology_loo_infeasibility_is_not_conflated_with_executed_failure():
    infeasible = audit._topology_evaluation(_topology_task(3))
    feasible = audit._topology_evaluation(_topology_task(4))
    assert infeasible["loo_detail_status"] == audit.TOPO_LOO_INFEASIBLE
    assert feasible["loo_detail_status"] == audit.TOPO_STRICT


def test_oracle_union_and_unrouted_agreement_union_are_distinct(tmp_path: Path):
    selection = _fake_selection(tmp_path)
    ledger = [
        _ledger_record(strict_label_v2=audit.STRICT_VALIDATED, strict_label_topology=audit.STRICT_VALIDATED, cross_expert_candidate_agreement="MULTI_EXPERT_CONFLICT"),
        _ledger_record(strict_label_v2=audit.STRICT_VALIDATED),
    ]
    profile = audit.compute_coverage_profile(selection, ledger)
    assert profile["C_oracle_union_retrospective"]["numerator"] == 2
    assert profile["C_unrouted_agreement_union"]["numerator"] == 1
    assert profile["cross_expert_conflict_count"] == 1


def test_topology_seed_only_contributions_are_reported_separately(tmp_path: Path):
    profile = audit.compute_coverage_profile(
        _fake_selection(tmp_path),
        [
            _ledger_record(strict_label_topology=audit.STRICT_VALIDATED, topology_subfamily_if_any="topology_seed_only"),
            _ledger_record(strict_label_topology=audit.STRICT_VALIDATED, topology_subfamily_if_any="topology_border_or_enclosed"),
        ],
    )
    assert profile["C_topology_total"]["numerator"] == 2
    assert profile["C_topology_seed_only"]["numerator"] == 1
    assert profile["C_topology_border_or_enclosed"]["numerator"] == 1


def test_every_coverage_metric_is_corpus_qualified(tmp_path: Path):
    profile = audit.compute_coverage_profile(_fake_selection(tmp_path), [_ledger_record(strict_label_v2=audit.STRICT_VALIDATED)])
    metric = profile["C_V2"]
    for key in ("corpus_name", "corpus_root_relative_or_configured_path", "task_count", "corpus_manifest_sha256"):
        assert key in metric


def test_protected_prior_hashes_remain_unchanged_during_contract_report():
    before = audit.protected_hashes()
    _ = audit.library_contract_report(_fake_selection(Path("/tmp")))
    after = audit.protected_hashes()
    assert before == after


def test_no_parameter_optimizer_backward_or_checkpoint_io_introduced():
    report = audit._parameter_free_integrity()
    assert report["no_parameter_optimizer_backward_or_checkpoint_io_introduced"] is True
    assert report["forbidden_code_terms_found"] == []
