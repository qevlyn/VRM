from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from vrm2.data.composition_split import (
    build_split_manifest,
    enumerate_compositions,
    parse_signature,
    split_signature_sets,
)
from vrm2.data.microworld import (
    generate_dataset,
    generate_episode,
    generate_episode_with_diagnostics,
    load_config,
    main,
    sample_hidden_program,
)
from vrm2.data.operations import OPERATION_NAMES, SceneValidationError, execute_operation, execute_program
from vrm2.data.serialization import (
    FORBIDDEN_MODEL_TERMS,
    assert_model_record_safe,
    canonical_json_bytes,
    episode_bytes,
    model_record,
)
from vrm2.data.types import OperationStep, make_grid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "vrm2" / "stage0.yaml"


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def split_manifest(config: dict[str, object]) -> dict[str, object]:
    return build_split_manifest(config["seeds"]["composition_split"], config["split_version"])


def _first_signature(split_manifest: dict[str, object], split: str = "train") -> str:
    return split_manifest["splits"][split]["signatures"][0]


def test_same_seed_produces_byte_identical_episode_data(config, split_manifest):
    signature = _first_signature(split_manifest)
    first = generate_episode(signature, 401, "train", config)
    second = generate_episode(signature, 401, "train", config)
    assert episode_bytes(first) == episode_bytes(second)


def test_different_seeds_produce_variation(config, split_manifest):
    signature = _first_signature(split_manifest)
    first = generate_episode(signature, 401, "train", config)
    second = generate_episode(signature, 402, "train", config)
    assert episode_bytes(first) != episode_bytes(second)
    assert first.query_input != second.query_input or first.hidden_program != second.hidden_program


def test_all_grids_have_valid_colors_and_fixed_shape(config, split_manifest):
    episodes, _, _ = generate_dataset(config, "train", 6, 9100, split_manifest)
    for episode in episodes:
        grids = [
            *(pair.input_grid for pair in episode.support),
            *(pair.output_grid for pair in episode.support),
            episode.query_input,
            episode.query_output,
        ]
        assert all(len(grid) == 8 and all(len(row) == 8 for row in grid) for grid in grids)
        assert all(0 <= value <= 9 for grid in grids for row in grid for value in row)


def test_ground_truth_program_replay_is_exact(config, split_manifest):
    for split in ("train", "dev", "sealed"):
        episodes, _, _ = generate_dataset(config, split, 4, config["seeds"][split], split_manifest)
        for episode in episodes:
            for pair in episode.support:
                assert execute_program(pair.input_grid, episode.hidden_program.steps) == pair.output_grid
            assert execute_program(episode.query_input, episode.hidden_program.steps) == episode.query_output


def test_composition_signatures_are_disjoint_and_cover_universe(config):
    manifest = build_split_manifest(config["seeds"]["composition_split"], config["split_version"])
    sets = split_signature_sets(manifest)
    assert sets["train"].isdisjoint(sets["dev"])
    assert sets["train"].isdisjoint(sets["sealed"])
    assert sets["dev"].isdisjoint(sets["sealed"])
    universe = enumerate_compositions()
    assert set().union(*sets.values()) == set().union(*(set(values) for values in universe.values()))
    assert {signature for signature in sets["train"] if len(parse_signature(signature)) == 1} == set(universe[1])


def test_model_facing_serialization_has_no_generator_metadata(config, split_manifest):
    episode = generate_episode(_first_signature(split_manifest), 990, "train", config)
    record = model_record(episode)
    assert_model_record_safe(record)
    assert set(record) == {"support", "query_input", "query_output"}
    payload = canonical_json_bytes(record).decode("ascii").lower()
    for forbidden in FORBIDDEN_MODEL_TERMS:
        assert forbidden not in payload


def test_operations_never_write_outside_grid():
    rows = [[0] * 8 for _ in range(8)]
    rows[0][0] = 4
    grid = make_grid(rows, expected_size=8)
    before = canonical_json_bytes(grid)
    with pytest.raises(SceneValidationError, match="out_of_bounds"):
        execute_operation(grid, OperationStep("translate_leftmost", {"direction": "up"}))
    assert canonical_json_bytes(grid) == before


def test_ambiguous_scene_is_rejected_and_resampling_is_deterministic(config, split_manifest):
    rows = [[0] * 8 for _ in range(8)]
    rows[1][2] = 3
    rows[4][2] = 6
    tied_grid = make_grid(rows, expected_size=8)
    with pytest.raises(SceneValidationError, match="selector_tie"):
        execute_operation(tied_grid, OperationStep("translate_leftmost", {"direction": "right"}))

    signature = _first_signature(split_manifest)
    first_episode, first_diagnostics = generate_episode_with_diagnostics(signature, 401, "train", config)
    second_episode, second_diagnostics = generate_episode_with_diagnostics(signature, 401, "train", config)
    assert episode_bytes(first_episode) == episode_bytes(second_episode)
    assert first_diagnostics == second_diagnostics
    assert sum(first_diagnostics.rejection_counts.values()) > 0


def test_dataset_generation_terminates_within_configured_limits(config, split_manifest):
    for split in ("train", "dev", "sealed"):
        count = config["task_counts"][split]
        episodes, diagnostics, _ = generate_dataset(config, split, count, config["seeds"][split], split_manifest)
        assert len(episodes) == count
        assert all(item.episode_attempt <= config["validation_limits"]["max_episode_attempts"] for item in diagnostics)
        assert all(
            attempt <= config["validation_limits"]["max_scene_attempts"]
            for item in diagnostics
            for attempt in item.scene_attempts
        )


def test_split_manifest_is_stable_for_version_and_seed(config):
    first = build_split_manifest(config["seeds"]["composition_split"], config["split_version"])
    second = build_split_manifest(config["seeds"]["composition_split"], config["split_version"])
    changed_seed = build_split_manifest(config["seeds"]["composition_split"] + 1, config["split_version"])
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) != canonical_json_bytes(changed_seed)


def test_operation_parameters_do_not_depend_on_split_label(config, split_manifest):
    signature = _first_signature(split_manifest)
    function_parameters = inspect.signature(sample_hidden_program).parameters
    assert "split" not in function_parameters
    assert sample_hidden_program(signature, 17) == sample_hidden_program(signature, 17)


def test_operation_order_can_change_output():
    rows = [[0] * 8 for _ in range(8)]
    rows[3][3] = 1
    grid = make_grid(rows, expected_size=8)
    left_then_top = (
        OperationStep("recolor_leftmost", {"color": 2}),
        OperationStep("recolor_topmost", {"color": 3}),
    )
    top_then_left = tuple(reversed(left_then_top))
    assert execute_program(grid, left_then_top) != execute_program(grid, top_then_left)


def test_retry_limit_failure_is_bounded(config, split_manifest):
    bounded = copy.deepcopy(config)
    bounded["validation_limits"]["max_scene_attempts"] = 1
    bounded["validation_limits"]["max_episode_attempts"] = 1
    signature = "reflect_rows"
    # This seed is frozen by the generator version and exercises the explicit limit path.
    with pytest.raises(RuntimeError, match="exceeded"):
        generate_episode(signature, 4, "train", bounded)


def test_smoke_cli_writes_separate_leak_safe_files(tmp_path, capsys):
    output = tmp_path / "train.jsonl"
    exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--split",
            "train",
            "--num-tasks",
            "3",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    generator_manifest = tmp_path / "train.generator-manifest.json"
    split_manifest_path = tmp_path / "train.split-manifest.json"
    assert output.is_file() and generator_manifest.is_file() and split_manifest_path.is_file()
    dataset_text = output.read_text(encoding="utf-8").lower()
    assert len(dataset_text.splitlines()) == 3
    assert "hidden_program" not in dataset_text
    assert "composition_signature" not in dataset_text
    assert "episode_seed" not in dataset_text
    assert any(name in generator_manifest.read_text(encoding="utf-8") for name in OPERATION_NAMES)
    printed = json.loads(capsys.readouterr().out)
    assert printed["num_tasks"] == 3
    assert printed["split"] == "train"
