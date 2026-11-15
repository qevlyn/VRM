from __future__ import annotations

import inspect
import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from vrm.twin_tasks.composition_r4a0r_data import (
    all_query_types,
    apply_sequence_to_grid,
    partition_query_types,
    query_type_key,
    tensors_from_episodes,
)
from vrm.twin_tasks.composition_r4a1p_data import (
    REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
    SEALED_STRATUM,
    TRAIN_VISIBLE_STRATUM,
    build_length3_counterfactual_suite,
    build_length3_function_catalog,
    default_probe_query_types,
    derange_episode,
    frozen_strata_match_partition,
    make_probe_episode,
    reorder_episode,
    support_input_bag,
    support_output_bag,
    support_signature,
)
from vrm.twin_tasks.composition_r4a1p_forensics import (
    APPROVED_CHECKPOINTS,
    pointer_metric_semantics_audit,
    protected_hashes,
    load_frozen_model,
    mutate_forbidden_metadata,
    parameter_hashes,
)
from vrm.twin_tasks.composition_r4a1p_oracles import (
    affine_for_word,
    affine_key,
    enumerate_length3_words,
    is_scene_safe_for_word,
    source_index_for_output_exact,
)
from vrm.twin_tasks.run_composition_r4a1p import evaluate_probe_model

DATA_OUT = Path("/mnt/d/VRM/artifacts/composition_r4a1p_v1")


def _sample_episode(word: str = "CCC"):
    qtype = default_probe_query_types()[TRAIN_VISIBLE_STRATUM][0]
    return make_probe_episode(TRAIN_VISIBLE_STRATUM, qtype, word, 0, set(), defaultdict(int))


def test_protected_old_files_hash_snapshot_is_stable_for_read_only_call():
    before = protected_hashes(DATA_OUT)
    after = protected_hashes(DATA_OUT)
    assert before == after
    assert any("composition_r4a0r_v2" in path for path in before)
    assert any("composition_r4a0r_seed_20260628_best_dev.pt" in path for path in before)


def test_only_three_approved_frozen_checkpoints_are_available_to_loader():
    assert set(APPROVED_CHECKPOINTS) == {20260628, 20260629, 20260630}
    assert all(path.name == f"composition_r4a0r_seed_{seed}_best_dev.pt" for seed, path in APPROVED_CHECKPOINTS.items())
    assert all(path.exists() for path in APPROVED_CHECKPOINTS.values())


def test_runner_source_has_no_training_or_checkpoint_write_api():
    import vrm.twin_tasks.run_composition_r4a1p as runner

    source = inspect.getsource(runner)
    forbidden = ["torch.optim", ".backward(", "loss.backward", ".step(", "torch.save(", "save_checkpoint("]
    assert all(item not in source for item in forbidden)


def test_parameter_hashes_identical_before_and_after_evaluation():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    device = torch.device("cuda")
    model, _, _ = load_frozen_model(20260628, device)
    episode = _sample_episode("CCC")
    before = parameter_hashes(model)
    evaluate_probe_model(model, [episode], device, batch_size=1)
    after = parameter_hashes(model)
    assert before == after
    assert all(not row["requires_grad"] for row in after.values())


def test_pointer_forensic_metric_is_exact_coordinate_not_color_only():
    audit = pointer_metric_semantics_audit()
    assert audit["semantics_conclusion"] == "exact-coordinate metric"
    assert audit["background_target_cells_scored"] is False
    assert audit["multiple_same_color_source_cells_handled_as"].startswith("not credited")


def test_background_cells_are_excluded_from_unique_source_coordinate_accuracy():
    episode = _sample_episode("CCC")
    fg_total = sum(1 for row in episode.target for value in row if value != 0)
    bg_total = sum(1 for row in episode.target for value in row if value == 0)
    oracle_sources = []
    for y, row in enumerate(episode.target):
        for x, value in enumerate(row):
            if value != 0:
                oracle_sources.append(source_index_for_output_exact(episode.sequence_class, x, y)[0])
    assert fg_total == len(oracle_sources)
    assert bg_total > 0
    assert all(index >= 0 for index in oracle_sources)


def test_all_27_length_three_words_are_enumerated():
    words = enumerate_length3_words()
    assert len(words) == 27
    assert words[0] == "CCC"
    assert words[-1] == "TTT"


def test_affine_signatures_are_exact_integer_maps():
    assert affine_for_word("TTC") == (((0, -1), (1, 0)), (6, 2))
    assert affine_for_word("CFC") == affine_for_word("F")


def test_selected_words_are_distinct_from_every_prior_function():
    catalog_path = DATA_OUT / "length3_function_catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else build_length3_function_catalog(default_probe_query_types())
    prior_keys = {affine_key(affine_for_word(""))}
    for length in (1, 2):
        for word in __import__("itertools").product("CFT", repeat=length):
            prior_keys.add(affine_key(affine_for_word("".join(word))))
    for word in catalog["eligible_canonical_words"]:
        assert affine_key(affine_for_word(word)) not in prior_keys


def test_equivalent_length_three_words_are_grouped_correctly():
    catalog = build_length3_function_catalog(default_probe_query_types())
    groups = {entry["canonical_representative"]: entry["equivalent_words"] for entry in catalog["canonical_groups"]}
    assert groups["CCC"] == ["CCC", "FCF"]
    assert sorted(groups["CFC"]) == ["CFC", "FFF", "TFT"]


def test_selected_scene_is_safe_for_every_intermediate_operation():
    episode = _sample_episode("TFC")
    ok, reasons = is_scene_safe_for_word(episode.query, "TFC")
    assert ok, reasons
    for pair in episode.supports:
        ok, reasons = is_scene_safe_for_word(pair.input, "TFC")
        assert ok, reasons


def test_counterfactual_group_outputs_are_distinct():
    rows, manifest, _ = build_length3_counterfactual_suite(["CCC", "CCF", "CCT", "CTC", "CTF", "FCT", "FTC", "FTF", "TCC", "TCF", "TCT", "TFC"], default_probe_query_types(), group_count=1)
    assert manifest["passed"] is True
    assert len({str(row.target) for row in rows}) == len(rows)


def test_all_104_frozen_query_types_meet_probe_feasibility_when_artifact_exists():
    path = DATA_OUT / "length3_query_type_feasibility.json"
    if not path.exists():
        pytest.skip("R4A-1P dry run generates this full feasibility artifact")
    report = json.loads(path.read_text())
    assert report["visual_query_type_count"] == 104
    assert report["passed"]
    for by_word in report["details"].values():
        for by_qtype in by_word.values():
            for row in by_qtype.values():
                assert row["generated"] == REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION


def test_train_visible_and_sealed_strata_match_frozen_v2_manifest():
    assert frozen_strata_match_partition()
    strata = default_probe_query_types()
    split = partition_query_types()
    assert sorted(query_type_key(item) for item in strata[TRAIN_VISIBLE_STRATUM]) == sorted(query_type_key(item) for item in split["train"])
    assert sorted(query_type_key(item) for item in strata[SEALED_STRATUM]) == sorted(query_type_key(item) for item in split["sealed"])


def test_forbidden_metadata_mutation_cannot_change_model_inputs_or_output():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    device = torch.device("cuda")
    model, _, _ = load_frozen_model(20260628, device)
    episode = _sample_episode("CCC")
    mutated = mutate_forbidden_metadata(episode, 7)
    left = tensors_from_episodes([episode], device)
    right = tensors_from_episodes([mutated], device)
    assert torch.equal(left["query_grids"], right["query_grids"])
    assert torch.equal(left["support_input_grids"], right["support_input_grids"])
    assert torch.equal(left["support_output_grids"], right["support_output_grids"])
    with torch.no_grad():
        out_a = model(left["query_grids"], left["support_input_grids"], left["support_output_grids"])
        out_b = model(right["query_grids"], right["support_input_grids"], right["support_output_grids"])
    assert torch.allclose(out_a["output_probs"], out_b["output_probs"], atol=1e-7)
    assert torch.allclose(out_a["task_context"], out_b["task_context"], atol=1e-7)
    assert torch.allclose(out_a["pointer_attention"], out_b["pointer_attention"], atol=1e-7)


def test_pair_permutation_remains_coherent():
    episode = _sample_episode("CCC")
    original = support_signature(episode)
    viewed = support_signature(reorder_episode(episode, (2, 0, 1)))
    assert viewed == [original[2], original[0], original[1]]
    assert sorted(viewed) == sorted(original)


def test_zero_support_changes_only_support_evidence():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    device = torch.device("cuda")
    model, _, _ = load_frozen_model(20260628, device)
    episode = _sample_episode("CCC")
    batch = tensors_from_episodes([episode], device)
    with torch.no_grad():
        normal = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"])
        zero = model(batch["query_grids"], batch["support_input_grids"], batch["support_output_grids"], zero_support=True)
    assert torch.equal(batch["query_grids"], batch["query_grids"])
    assert not torch.equal(normal["decoded_grid"], zero["decoded_grid"])


def test_deranged_pairing_preserves_bags_and_breaks_only_correspondence():
    episode = _sample_episode("CCC")
    deranged = derange_episode(episode, 1)
    assert support_input_bag(episode) == support_input_bag(deranged)
    assert support_output_bag(episode) == support_output_bag(deranged)
    assert support_signature(episode) != support_signature(deranged)


def test_no_test_result_can_influence_checkpoint_choice():
    import vrm.twin_tasks.run_composition_r4a1p as runner
    import vrm.twin_tasks.composition_r4a1p_forensics as forensics

    source = inspect.getsource(runner) + inspect.getsource(forensics)
    assert "best_dev.pt" in source
    assert "selected_epoch = max" not in source
    assert "checkpoint_selection" in source


def test_reported_metrics_match_hand_built_single_episode_fixture():
    assert torch.cuda.is_available(), "CUDA_REQUIRED_NOT_AVAILABLE"
    device = torch.device("cuda")
    model, _, _ = load_frozen_model(20260628, device)
    episode = _sample_episode("CCC")
    metrics, details = evaluate_probe_model(model, [episode], device, batch_size=1)
    decoded_exact = details[0]["exact"]
    row = metrics["by_visual_stratum_and_canonical_function"][TRAIN_VISIBLE_STRATUM]["CCC"]
    assert row["episode_count"] == 1
    assert row["direct_grid_exact_accuracy"] == float(decoded_exact)


def test_all_query_types_catalog_still_has_128_entries():
    assert len(all_query_types()) == 128
