from __future__ import annotations

import json
from pathlib import Path

import torch

from vrm2.data.composition_split import build_split_manifest
from vrm2.data.microworld import generate_episode
from vrm2.data.serialization import canonical_json_bytes, model_record
from vrm2.oracle.dataset import (
    ORACLE_MODEL_INPUT_FIELDS,
    assert_disjoint_examples,
    build_oracle_examples,
    make_oracle_batch,
    oracle_record,
)
from vrm2.oracle.evaluate import evaluate_model
from vrm2.oracle.model import OracleInterpreterV0, OracleModelConfig
from vrm2.oracle.train import load_oracle_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_normal_stage0_serialization_remains_typed_program_free(stage0_config):
    manifest = build_split_manifest(
        stage0_config["seeds"]["composition_split"], stage0_config["split_version"]
    )
    signature = manifest["splits"]["train"]["signatures"][0]
    episode = generate_episode(signature, 808, "train", stage0_config)
    payload = canonical_json_bytes(model_record(episode)).decode("ascii")
    assert "typed_program" not in payload
    assert "hidden_program" not in payload
    assert "composition_signature" not in payload


def test_only_oracle_boundary_exposes_numeric_typed_ids(small_train_oracle_examples):
    example = small_train_oracle_examples[0]
    record = oracle_record(example)
    assert all(isinstance(value, int) for instruction in record["typed_program_ids"] for value in instruction)
    batch = make_oracle_batch([item for item in small_train_oracle_examples if item.program_length == example.program_length][:2])
    inputs = batch.model_inputs()
    assert ORACLE_MODEL_INPUT_FIELDS == ("input_grid", "typed_program_ids")
    assert len(inputs) == 2
    assert all(isinstance(value, torch.Tensor) and value.dtype == torch.long for value in inputs)


def test_intermediate_targets_split_and_signature_are_not_model_inputs(small_train_oracle_examples):
    length = small_train_oracle_examples[0].program_length
    examples = [item for item in small_train_oracle_examples if item.program_length == length][:2]
    batch = make_oracle_batch(examples)
    inputs = batch.model_inputs()
    assert batch.intermediate_target_grids.data_ptr() not in {tensor.data_ptr() for tensor in inputs}
    assert not hasattr(batch, "split")
    assert not hasattr(batch, "composition_signature")


def test_evaluation_reports_length_two_and_three_separately(small_sealed_oracle_examples):
    model = OracleInterpreterV0(
        OracleModelConfig(hidden_dim=32, color_embedding_dim=16, workspace_tokens=2, shared_execution_layers=1)
    )
    metrics = evaluate_model(model, small_sealed_oracle_examples, 16, torch.device("cpu"))
    assert set(metrics["by_program_length"]) == {"2", "3"}


def test_declared_2000_scene_sealed_corpus_is_disjoint_from_training(stage0_config):
    config = load_oracle_config(REPOSITORY_ROOT / "configs" / "vrm2" / "oracle_interpreter_cert.yaml")
    sealed = build_oracle_examples(
        stage0_config, "sealed", config["data"]["sealed_examples"], config["data"]["sealed_seed"]
    )
    train = build_oracle_examples(stage0_config, "train", 128, config["data"]["train_seed"])
    assert len(sealed) == 2000
    assert sum(example.program_length == 2 for example in sealed) == 1000
    assert sum(example.program_length == 3 for example in sealed) == 1000
    assert_disjoint_examples(train, sealed)
