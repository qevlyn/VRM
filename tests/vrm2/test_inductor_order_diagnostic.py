from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

import vrm2.inductor.order_diagnostic_cli as diagnostic
from vrm2.data.serialization import canonical_json_bytes
from vrm2.inductor.episodes import load_support_episodes
from vrm2.inductor.evaluate_v1 import execute_programs_v1, predict_programs_v1
from vrm2.inductor.hashing import file_sha256
from vrm2.inductor.order_stage_b import execute_stage_b
from vrm2.inductor.train_v1 import load_checkpoint_v1, shuffled_instruction_targets


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/vrm2/inductor_order_causality.yaml").read_text(encoding="utf-8")
)


def test_fresh_stage_b_has_no_support_file_api_or_loader():
    parameters = inspect.signature(execute_stage_b).parameters
    source = inspect.getsource(execute_stage_b)
    assert not any("support" in name for name in parameters)
    assert "load_support_episodes" not in source
    assert '"support_file_loaded": False' in source


def test_execution_boundary_rejects_non_int64_programs():
    with pytest.raises(TypeError, match="hard torch.long IDs"):
        execute_programs_v1(
            torch.nn.Identity(),
            torch.zeros((1, 8, 8), dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
            torch.zeros((1, 3, 4), dtype=torch.float32),
            1,
            torch.device("cpu"),
        )


@pytest.mark.parametrize("role", ["main", "shuffled"])
def test_support_permutation_preserves_hard_program_for_frozen_checkpoint(role):
    path, _, _ = diagnostic.resolve_selected_checkpoint(role, CONFIG["checkpoints"][role])
    model, _ = load_checkpoint_v1(path, torch.device("cpu"))
    supports = load_support_episodes(
        ROOT / CONFIG["sealed_l2"]["support_path"]
    )[:8]
    lengths, programs, _, _ = predict_programs_v1(
        model, supports, 8, torch.device("cpu")
    )
    permuted_lengths, permuted_programs, _, _ = predict_programs_v1(
        model,
        supports,
        8,
        torch.device("cpu"),
        support_permutation=(2, 0, 1),
    )
    assert torch.equal(lengths, permuted_lengths)
    assert torch.equal(programs, permuted_programs)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "artifacts/vrm2/inductor/runtime/v1/pilot_seed_20260713/inductor_v1_seed_20260713.pt",
            "4f5a42621a7ad72cb4f5cc698e6b1dd1db8392053b66a074448c52c3c49ce9ae",
        ),
        (
            "artifacts/vrm2/inductor/runtime/v1/ablation_shuffled_order/inductor_v1_seed_20260713.pt",
            "dff4c4c2bb2c6efa43bf5ae53dffdf09062aba99d3d628585cdd3004e52b9d56",
        ),
        (
            "artifacts/vrm2/oracle/runtime/v1/certification/oracle_interpreter_seed_20260713.pt",
            "52c05ee52badf8bf3244d15d146b81a0f7a372a049473756771d05a90ee02e48",
        ),
    ],
)
def test_frozen_checkpoint_hash_preservation(path, expected):
    assert file_sha256(ROOT / path) == expected


def test_shuffled_target_intervention_is_nontrivial_and_preserves_null():
    programs = torch.tensor([[[1, 1, 1, 4], [2, 2, 2, 7], [0, 0, 0, 0]]])
    shuffled = shuffled_instruction_targets(programs, torch.tensor([2]))
    assert torch.equal(shuffled[0, 0], programs[0, 1])
    assert torch.equal(shuffled[0, 1], programs[0, 0])
    assert torch.equal(shuffled[0, 2], programs[0, 2])
    assert not torch.equal(shuffled, programs)


def test_coverage_l3_payload_is_denied_before_open(monkeypatch, tmp_path):
    prohibited = ROOT / CONFIG["prohibited_coverage_payloads"][0]
    local_config = {
        "runtime_dir": "unused",
        "prohibited_coverage_payloads": [str(prohibited)],
    }
    monkeypatch.setattr(diagnostic, "runtime_dir", lambda config: tmp_path)
    with pytest.raises(RuntimeError, match="prohibited L3 coverage access"):
        diagnostic._record_access(local_config, [prohibited], "unit-test")
    assert not (tmp_path / "access_audit.json").exists()


def test_repeated_source_manifest_artifact_is_deterministic():
    first = diagnostic.build_source_manifest()
    second = diagnostic.build_source_manifest()
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
