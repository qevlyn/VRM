from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from vrm.twin_tasks import transport_recolor_r5a1_preflight as preflight
from vrm.twin_tasks.transport_recolor_r5a1_shared_renderer import (
    ALLOWED_INPUT_FIELDS,
    COLOR_COUNT,
    GRID_SIZE,
    MEMORY_TOKEN_COUNT,
    OUTPUT_CELL_COUNT,
    GenericSharedRenderer,
    SharedRendererConfig,
    SharedRendererInput,
    pack_shared_input_from_records,
    tensor_digest,
)


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory):
    pytest_result = tmp_path_factory.mktemp("r5a1a") / "pytest_result.txt"
    pytest_result.write_text("300 passed\n", encoding="utf-8")
    return preflight.build_report(pytest_result)


def test_preflight_status_is_pass_or_declared_inconclusive_not_fail(report):
    assert report["status"] in {"R5A1A_PREFLIGHT_PASS", "R5A1A_PREFLIGHT_INCONCLUSIVE"}
    hard_gates = (
        "v2_source_hashes_unchanged_during_preflight",
        "all_three_variants_instantiate_and_forward",
        "no_explicit_factor_copy_or_oracle_interface",
        "information_firewall_passed",
        "canonical_batch_deterministic",
        "flop_counter_deterministic",
        "no_training_checkpoint_or_sealed_eval",
    )
    for gate in hard_gates:
        assert report["preflight_gates"][gate] is True, gate
    if report["status"] == "R5A1A_PREFLIGHT_INCONCLUSIVE":
        assert not report["preflight_gates"]["matched_budget_tolerances_satisfied"]
        assert report["capacity_search"]["infeasibility_reasons"]


def test_three_variants_instantiate_forward_and_emit_expected_shapes(report):
    assert set(report["variants"]) == {"shared_small", "shared_matched", "shared_large"}
    records = preflight.canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    for name, selected in report["capacity_search"]["selected"].items():
        config = preflight.config_from_candidate(name, selected)
        torch.manual_seed(preflight.DETERMINISTIC_SEED)
        model = GenericSharedRenderer(config).eval()
        with torch.no_grad():
            output = model(batch)
        assert list(output.color_logits.shape) == [4, GRID_SIZE, GRID_SIZE, COLOR_COUNT]
        assert list(output.output_size_logits.shape) == [4, 2, GRID_SIZE]
        assert list(output.memory_token_mask.shape) == [4, MEMORY_TOKEN_COUNT]
        assert list(output.output_cell_mask.shape) == [4, OUTPUT_CELL_COUNT]
        assert output.output_cell_mask.all()


def test_shared_forward_accepts_only_allowlisted_input_fields():
    records = preflight.canonical_records(batch_size=4)
    batch = pack_shared_input_from_records(records)
    assert set(ALLOWED_INPUT_FIELDS) == {
        "query_grids",
        "support_input_grids",
        "support_output_grids",
        "memory_token_mask",
        "output_cell_mask",
        "canvas_hw",
    }
    with pytest.raises(TypeError):
        SharedRendererInput(
            query_grids=batch.query_grids,
            support_input_grids=batch.support_input_grids,
            support_output_grids=batch.support_output_grids,
            memory_token_mask=batch.memory_token_mask,
            output_cell_mask=batch.output_cell_mask,
            canvas_hw=batch.canvas_hw,
            operation_class=torch.zeros((4,), dtype=torch.long),  # type: ignore[call-arg]
        )


def test_metadata_perturbation_does_not_change_packed_tensors_or_logits(report):
    firewall = report["information_firewall_audit"]
    assert firewall["metadata_perturbation"]["constructed_input_tensors_unchanged"] is True
    assert firewall["metadata_perturbation"]["eval_mode_logits_unchanged"] is True
    assert firewall["source_scan"]["passed"] is True
    assert firewall["no_explicit_factor_copy_or_oracle_interface"] is True
    inventory = {row["field_name"]: row["status"] for row in firewall["field_inventory"]}
    assert inventory["query"] == "allowed"
    assert inventory["supports[].input"] == "allowed"
    assert inventory["supports[].output"] == "allowed"
    assert inventory["target"] == "forbidden"
    assert inventory["operation_class"] == "forbidden"
    assert inventory["factor_metadata"] == "forbidden"


def test_canonical_batch_is_deterministic_training_side_only(report):
    records_a = preflight.canonical_records(batch_size=4)
    records_b = preflight.canonical_records(batch_size=4)
    batch_a = pack_shared_input_from_records(records_a)
    batch_b = pack_shared_input_from_records(records_b)
    assert [row["episode_id"] for row in records_a] == [row["episode_id"] for row in records_b]
    assert tensor_digest(batch_a) == tensor_digest(batch_b)
    assert tensor_digest(batch_a) == report["canonical_profiling_batch"]["tensor_sha256"]
    assert report["canonical_profiling_batch"]["training_side_only"] is True
    assert report["canonical_profiling_batch"]["sealed_rows_read"] is False
    assert report["canonical_profiling_batch"]["padding"]["memory_token_padding_false_count"] == 0
    assert report["canonical_profiling_batch"]["padding"]["output_cell_padding_false_count"] == 0


def test_flop_counter_and_capacity_search_are_deterministic(report):
    batch_size = report["canonical_profiling_batch"]["batch_size"]
    v2_params = report["v2_reference_budget"]["trainable_parameters_from_initialization_report"]
    v2_flops = report["v2_reference_budget"]["flops"]
    first = preflight.capacity_search(batch_size=batch_size, v2_params=v2_params, v2_flops=v2_flops)
    second = preflight.capacity_search(batch_size=batch_size, v2_params=v2_params, v2_flops=v2_flops)
    assert first == second
    for name, selected in first["selected"].items():
        config = preflight.config_from_candidate(name, selected)
        assert preflight.shared_flop_breakdown(config, batch_size=batch_size) == report["variants"][name]["flops"]
    assert report["flop_counting_convention"]["linear_and_matmul"] == "2*M*N*K"
    assert report["flop_counting_convention"]["counter_type"] == "analytic formula, not runtime profiler"


def test_candidate_variants_are_same_family_with_capacity_knobs_only(report):
    configs = {name: row["config"] for name, row in report["variants"].items()}
    invariant_keys = {
        "dropout",
        "grid_size",
        "color_count",
        "support_pairs",
        "output_cell_count",
    }
    invariant_values = {key: {configs[name][key] for name in configs} for key in invariant_keys}
    assert all(len(values) == 1 for values in invariant_values.values())
    assert configs["shared_small"]["hidden_size"] <= configs["shared_matched"]["hidden_size"] <= configs["shared_large"]["hidden_size"]
    assert report["capacity_search"]["search_uses_accuracy_or_runtime"] is False


def test_no_training_checkpoint_or_sealed_eval_and_v2_sources_unchanged(report):
    audit = report["no_training_checkpoint_or_sealed_eval_audit"]
    assert audit["passed"] is True
    assert audit["checkpoint_files_created"] == []
    assert audit["training_loop_terms_found_in_new_sources"] == []
    assert audit["sealed_evaluation_terms_found_in_new_sources"] == []
    assert report["protected_v2_source_hashes_before"] == report["protected_v2_source_hashes_after"]
    preflight_dir = Path("/mnt/d/VRM/artifacts/transport_recolor_r5a1a_preflight")
    if preflight_dir.exists():
        assert not list(preflight_dir.rglob("*.pt"))
        assert not list(preflight_dir.rglob("*.pth"))
        assert not list(preflight_dir.rglob("*.ckpt"))


def test_report_hash_ignores_timestamp_only(tmp_path: Path):
    pytest_result = tmp_path / "pytest.txt"
    pytest_result.write_text("300 passed\n", encoding="utf-8")
    first = preflight.build_report(pytest_result)
    second = preflight.build_report(pytest_result)
    assert preflight.stable_report_hash(first) == preflight.stable_report_hash(second)
    assert first["created_utc"] <= second["created_utc"]


def test_config_dataclass_serializes_without_factor_interfaces():
    config = SharedRendererConfig("unit", hidden_size=8, num_heads=1, encoder_layers=0, decoder_layers=1)
    data = asdict(config)
    assert data["variant_name"] == "unit"
    assert "transport_head" not in data
    assert "color_transition_head" not in data
    assert "pointer" not in data
    assert "copy" not in data
