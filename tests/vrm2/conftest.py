from __future__ import annotations

from pathlib import Path

import pytest

from vrm2.data.microworld import load_config as load_stage0_config
from vrm2.oracle.dataset import build_oracle_examples
from vrm2.oracle.train import load_oracle_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def stage0_config():
    return load_stage0_config(REPOSITORY_ROOT / "configs" / "vrm2" / "stage0.yaml")


@pytest.fixture(scope="session")
def oracle_smoke_config():
    return load_oracle_config(REPOSITORY_ROOT / "configs" / "vrm2" / "oracle_interpreter_smoke.yaml")


@pytest.fixture(scope="session")
def small_train_oracle_examples(stage0_config):
    return build_oracle_examples(stage0_config, "train", 24, 710000)


@pytest.fixture(scope="session")
def small_sealed_oracle_examples(stage0_config):
    return build_oracle_examples(stage0_config, "sealed", 24, 720000)
