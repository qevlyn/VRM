from __future__ import annotations

from vrm.twin_tasks import r5a3a_arc_audit as audit
from vrm.twin_tasks.r5a3a_arc_native_candidate_machine import (
    K_VALUES,
    SupportPair,
    candidate_machine_spec,
    generate_candidates,
    target_rank,
    validate_machine_inputs,
)
from vrm.twin_tasks.r5a3a_scene_hypotheses import (
    canonical_grid_sha256,
    extract_objects,
    freeze_grid,
    scene_hypotheses_for_grid,
    valid_raw_grid,
)
from vrm.twin_tasks.r5a3a_structural_programs import (
    MAX_PROGRAM_DEPTH,
    MAX_PROGRAM_ENUMERATION_COUNT,
    enumerate_programs,
    program_surface_spec,
)


def _grid(rows):
    return freeze_grid(rows)


def test_machine_contract_accepts_variable_arc_native_inputs():
    supports = tuple(SupportPair(_grid([[i % 10]]), _grid([[i % 10]])) for i in range(1, 11))
    queries = tuple(_grid([[0 for _ in range(30)] for _ in range(30)]) for _ in range(1))
    ok, reasons = validate_machine_inputs(supports, queries)
    assert ok, reasons
    assert valid_raw_grid(_grid([[9 for _ in range(30)] for _ in range(1)]))[0]
    assert not valid_raw_grid(_grid([[10]]))[0]


def test_named_stages_and_frozen_budget_are_declared():
    spec = candidate_machine_spec()
    assert spec["K"] == [1, 4, 16, 64, 128]
    assert K_VALUES == [1, 4, 16, 64, 128]
    assert MAX_PROGRAM_DEPTH == 2
    assert MAX_PROGRAM_ENUMERATION_COUNT == 8
    assert spec["named_stages"] == [
        "A. scene hypotheses",
        "B. object/region extraction",
        "C. support-pair delta extraction",
        "D. structural program enumeration",
        "E. program execution",
        "F. visible-support consistency scoring",
        "G. top-K deduplicated candidate output grids",
    ]


def test_scene_hypotheses_and_object_extraction_are_inspectable():
    grid = _grid([[0, 1, 1], [0, 0, 2], [2, 0, 0]])
    hypotheses = scene_hypotheses_for_grid(grid)
    assert 1 <= len(hypotheses) <= 4
    objects = extract_objects(grid, hypotheses[0])
    assert objects
    assert all(obj.object_id and obj.shape_hash for obj in objects)


def test_recolor_translate_and_changed_shape_candidates_hit_golden_targets():
    for case in audit.mechanism_golden_cases():
        supports = tuple(SupportPair(inp, out) for inp, out in case["supports"])
        result = generate_candidates(supports, (case["query"],))
        candidates = result["query_candidates"][0]
        assert result["accepted"]
        assert target_rank(candidates, case["target"]) is not None, case["name"]


def test_rotation_reflection_copy_and_two_step_programs_are_enumerated():
    inp = _grid([[1, 0, 0], [0, 0, 0]])
    out = _grid([[0, 1, 0], [0, 0, 0]])
    hypothesis = scene_hypotheses_for_grid(inp)[0]
    programs = enumerate_programs((inp,), (out,), (inp,), hypothesis, max_depth=2)
    ids = [program.program_id for program in programs]
    assert any("rotate" in item for item in ids)
    assert any("reflect" in item for item in ids)
    assert any("copy" in item for item in ids)
    assert any(program.depth == 2 for program in programs)


def test_distinct_executable_programs_emit_distinct_deduplicated_candidates():
    inp = _grid([[1, 0], [0, 0]])
    out = _grid([[2, 0], [0, 0]])
    result = generate_candidates((SupportPair(inp, out),), (inp,))
    candidates = result["query_candidates"][0]
    hashes = [candidate["output_hash"] for candidate in candidates]
    assert len(hashes) == len(set(hashes))
    assert len(candidates) >= 2


def test_property_report_covers_required_mechanism_families():
    report = audit.property_test_report()
    assert report["accepted_count"] == len(report["rows"])
    required = {
        "arbitrary grid sizes 1x1 to 30x30",
        "palette 0..9",
        "one through ten support pairs",
        "same-shape and changed-shape outputs",
        "translation, copy, recolor, rotation, reflection",
        "two-step compositions",
    }
    assert required.issubset(set(report["families_covered"]))


def test_raw_grid_hash_is_stable_and_fidelity_is_identity_inside_machine():
    grid = _grid([[1, 2], [3, 4]])
    result = generate_candidates((SupportPair(grid, grid),), (grid,), identity_only=True)
    assert result["raw_grid_fidelity_passed"]
    assert canonical_grid_sha256(grid) == canonical_grid_sha256(grid)


def test_pre_arc_commitment_contains_interpretation_rules():
    commitment = audit.pre_arc_audit_commitment()
    assert commitment["depth_limit"] == 2
    assert "refinement_or_ttt_controller_justification_rule" in commitment
    assert "rule_out_expanding_v0_operator_family_rule" in commitment


def test_parameter_free_and_protected_hashes_unchanged_by_reports():
    before = audit.protected_hashes()
    _ = audit.mechanism_preflight_report()
    _ = audit.property_test_report()
    _ = program_surface_spec()
    after = audit.protected_hashes()
    assert before == after
    assert audit.parameter_free_report()["passed"]


def test_no_arc_test_outputs_are_used_by_loader():
    tasks = audit.load_arc_tasks()
    assert tasks
    assert all(hasattr(task, "test_inputs") for task in tasks)
    assert all(not hasattr(task, "sealed_test_outputs_if_present") for task in tasks)
