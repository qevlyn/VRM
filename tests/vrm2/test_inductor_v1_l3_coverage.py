from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch

from vrm2.inductor.evaluate_v1 import execute_programs_v1
from vrm2.inductor.execution_v1 import evaluate_induced_program_file_v1


ROOT = Path(__file__).resolve().parents[2]
COVERAGE = ROOT / "artifacts/vrm2/inductor/runtime/v1/l3_coverage"


def test_coverage_corpus_is_exactly_512_and_deterministically_regenerated():
    manifest = json.loads((COVERAGE / "manifest.json").read_text())
    assert manifest["seed"] == 830000
    assert manifest["episode_count"] == 512
    assert manifest["deterministic_regeneration_identical"] is True
    assert len((COVERAGE / "episodes.jsonl").read_bytes().splitlines()) == 512


def test_coverage_has_zero_prior_overlap_and_absent_complete_training_signatures():
    manifest = json.loads((COVERAGE / "manifest.json").read_text())
    assert all(value == 0 for value in manifest["fingerprint_overlap_counts"].values())
    assert manifest["complete_l3_signatures_absent_from_training"] is True
    assert manifest["balanced_composition_usage"] is True


def test_eligibility_audit_confirms_instruction_and_subsequence_coverage():
    audit = json.loads((COVERAGE / "eligibility_audit.json").read_text())
    assert audit["eligible_signature_count"] == 113
    eligible = [item for item in audit["records"] if item["eligible"]]
    assert len(eligible) == 113
    assert all(item["instruction_categories_seen_in_training"] for item in eligible)
    assert all(item["instruction_values_seen_in_training"] for item in eligible)
    assert all(item["training_length2_subsequence_hits"] for item in eligible)


def test_coverage_access_audit_is_sealed_before_phase_a():
    audit = json.loads((COVERAGE / "access_audit.json").read_text())
    assert audit["opened_for_training"] is False
    assert audit["opened_for_checkpoint_selection"] is False
    assert audit["phase_b_evaluation_count"] == 0


def test_stage_b_api_and_import_graph_have_no_support_loader():
    assert tuple(inspect.signature(evaluate_induced_program_file_v1).parameters) == (
        "queries_path", "programs_path", "interpreter_checkpoint", "batch_size", "device"
    )
    code = (
        "import sys; import vrm2.inductor.execution_v1; "
        "assert 'vrm2.inductor.episodes' not in sys.modules"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_hard_execution_boundary_uses_integer_ids():
    class RecordingInterpreter(torch.nn.Module):
        def forward(self, grids, typed_program_ids):
            assert typed_program_ids.dtype == torch.long
            return torch.zeros(len(grids), typed_program_ids.shape[1], 10, 8, 8)

    execute_programs_v1(
        RecordingInterpreter(),
        torch.zeros(2, 8, 8, dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
        torch.zeros(2, 3, 4, dtype=torch.long),
        2,
        torch.device("cpu"),
    )
