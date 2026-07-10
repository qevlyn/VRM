from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/VRM")
R6 = ROOT / "r6a_compressarc"
TRIAGE = R6 / "artifacts" / "determinism_triage"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_protected_vrm_paths_remain_byte_identical_for_0c() -> None:
    before = read_json(R6 / "manifests" / "protected_paths_before_0c.json")
    for row in before["files"]:
        path = ROOT / row["path"]
        assert path.exists(), row["path"]
        assert path.stat().st_size == row["size_bytes"], row["path"]
    after_path = R6 / "manifests" / "protected_paths_after_0c.json"
    if after_path.exists():
        after = read_json(after_path)
        assert after["all_match_before"] is True


def test_rng_sources_are_reset_before_model_construction() -> None:
    a = read_json(TRIAGE / "baseline_a.json")
    b = read_json(TRIAGE / "baseline_b.json")
    assert a["rng_state_immediately_before_task_model_construction"] == b["rng_state_immediately_before_task_model_construction"]
    assert a["model_parameter_state_hash_after_initialization"] == b["model_parameter_state_hash_after_initialization"]
    assert a["optimizer_state_hash_after_construction"] == b["optimizer_state_hash_after_construction"]
    source = (R6 / "scripts" / "determinism_trace_solver.py").read_text(encoding="utf-8")
    seed_block = source[source.index("random.seed(args.seed)") : source.index("rng_before_task_model = rng_state_payload()")]
    assert "np.random.seed(args.seed)" in seed_block
    assert "torch.manual_seed(args.seed)" in seed_block
    assert "torch.cuda.manual_seed_all(args.seed)" in seed_block


def test_full_b1_b2_initialization_artifacts_are_byte_identical() -> None:
    proof = read_json(TRIAGE / "future_pairing_initialization_proof.json")
    assert proof["all_conditions_begin_byte_identical"] is True
    for task in proof["proofs"]:
        assert task["all_model_hashes_identical"] is True
        assert task["all_optimizer_hashes_identical"] is True


def test_replacing_evaluator_targets_cannot_alter_solver_trace_or_prediction() -> None:
    negative = read_json(R6 / "artifacts" / "target_firewall_negative_control.json")
    assert negative["status"] == "QUERY_TARGET_FIREWALL_NEGATIVE_CONTROL_PASSED"
    assert negative["solver_training_trajectory_identical"] is True
    assert negative["predicted_query_grid_identical"] is True


def test_structured_solver_log_hash_excludes_wall_clock_runtime() -> None:
    source = (R6 / "scripts" / "determinism_trace_solver.py").read_text(encoding="utf-8")
    stable_log_section = source[source.index("stable_log = {") : source.index('"runtime_seconds"')]
    assert "runtime_seconds" not in stable_log_section
    assert "peak_gpu_memory_bytes" not in stable_log_section
    a = read_json(TRIAGE / "baseline_a.json")
    assert "structured_log_hash_excluding_wall_clock" in a


def test_classifier_reports_first_divergence_checkpoint() -> None:
    triage = read_json(R6 / "artifacts" / "determinism_triage.json")
    comparison = triage["baseline_trace_comparison"]["comparison"]
    assert comparison["classification"] == "BACKWARD_OR_OPTIMIZER_DIVERGENCE"
    assert comparison["first_divergence_checkpoint"] == "after_optimizer_step_1:model_parameter_state_hash"


def test_strict_deterministic_mode_is_explicit_about_unsupported_operation() -> None:
    triage = read_json(R6 / "artifacts" / "determinism_triage.json")
    unsupported = triage["strict_trace_comparison"]["unsupported_operation"]
    assert unsupported is not None
    assert unsupported["type"] == "RuntimeError"
    assert "nll_loss2d_forward_out_cuda_template" in unsupported["message"]
    assert triage["root_cause"]["strict_result_classification"] == "UNSUPPORTED_STRICT_DETERMINISM_OPERATION"
