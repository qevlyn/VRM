from __future__ import annotations

import argparse
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.inductor.hashing import file_sha256, module_state_hashes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST_VERSION = "vrm2-order-causality-source-v1"
SOURCE_PATHS = (
    "vrm2/inductor/order_causality.py",
    "vrm2/inductor/order_selector.py",
    "vrm2/inductor/order_stage_b.py",
    "vrm2/inductor/order_diagnostic_cli.py",
    "vrm2/inductor/episodes.py",
    "vrm2/inductor/evaluate_v1.py",
    "vrm2/inductor/model_v1.py",
    "vrm2/inductor/sequential_decoder.py",
    "vrm2/inductor/schema_v1.py",
    "vrm2/inductor/train_v1.py",
    "vrm2/data/operations.py",
    "vrm2/data/program_schema.py",
    "configs/vrm2/inductor_order_causality.yaml",
    "configs/vrm2/inductor_v1_pilot.yaml",
    "configs/vrm2/inductor_v1_ablation_shuffled_order.yaml",
    "docs/vrm2/INDUCTOR_ORDER_CAUSALITY_PROTOCOL.md",
    "tests/vrm2/test_inductor_order_subset.py",
    "tests/vrm2/test_inductor_order_selector.py",
    "tests/vrm2/test_inductor_order_diagnostic.py",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\n".join(canonical_json_bytes(item) for item in records) + b"\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "vrm2-frozen-checkpoint-l2-order-causality-v1":
        raise ValueError("unexpected order-causality protocol version")
    return config


def runtime_dir(config: Mapping[str, Any]) -> Path:
    return REPOSITORY_ROOT / config["runtime_dir"]


def build_source_manifest() -> dict[str, Any]:
    files = {}
    for relative in SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"diagnostic source is missing: {relative}")
        files[relative] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return {"manifest_version": SOURCE_MANIFEST_VERSION, "files": files}


def verify_source_manifest(manifest_path: Path) -> dict[str, Any]:
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = build_source_manifest()
    changed = {
        name: {"expected": expected["files"].get(name), "actual": details}
        for name, details in actual["files"].items()
        if expected["files"].get(name) != details
    }
    missing = sorted(set(expected["files"]) - set(actual["files"]))
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "source_unchanged": not changed and not missing,
        "changed": changed,
        "missing": missing,
        "actual": actual,
    }


def _require_frozen_source(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = runtime_dir(config) / "source_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("diagnostic source has not been frozen")
    result = verify_source_manifest(manifest)
    if not result["source_unchanged"]:
        raise RuntimeError("ORDER_CAUSALITY_INVALID_SOURCE_CHANGED")
    return result


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _record_access(config: Mapping[str, Any], paths: Sequence[Path], label: str) -> None:
    output = runtime_dir(config) / "access_audit.json"
    if output.is_file():
        audit = json.loads(output.read_text(encoding="utf-8"))
    else:
        audit = {
            "opened_paths": [],
            "commands": [],
            "prohibited_coverage_payloads": [
                str(_resolved(path).resolve())
                for path in config["prohibited_coverage_payloads"]
            ],
            "coverage_l3_payload_access_count": 0,
        }
    opened = set(audit["opened_paths"])
    prohibited = set(audit["prohibited_coverage_payloads"])
    for path in paths:
        resolved = str(path.resolve())
        if resolved in prohibited:
            audit["coverage_l3_payload_access_count"] += 1
            raise RuntimeError("ORDER_CAUSALITY_INVALID: prohibited L3 coverage access")
        opened.add(resolved)
    audit["opened_paths"] = sorted(opened)
    audit["commands"].append(label)
    _write_json(output, audit)


def resolve_selected_checkpoint(
    role: str, specification: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from vrm2.inductor.model_v1 import component_parameter_counts
    from vrm2.inductor.train_v1 import load_checkpoint_v1

    report_path = _resolved(specification["selected_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = []
    for path in sorted(report_path.parent.glob("*.pt")):
        model, metadata = load_checkpoint_v1(path, torch.device("cpu"))
        embedded = metadata["inductor_v1_config"]
        if (
            int(metadata["seed"]) == int(report["seed"])
            and int(metadata["step"]) == int(report["checkpoint_step"])
            and embedded["ablation"] == report["ablation"]
            and component_parameter_counts(model)["total"]
            == int(report["component_parameter_counts"]["total"])
        ):
            candidates.append((path, model, metadata))
    if len(candidates) != 1:
        raise RuntimeError(
            f"selected report for {role} resolves to {len(candidates)} checkpoints"
        )
    path, model, metadata = candidates[0]
    archive = file_sha256(path)
    if archive != specification["expected_archive_sha256"]:
        raise RuntimeError(f"{role} checkpoint archive hash mismatch")
    if report["ablation"] != specification["expected_ablation"]:
        raise RuntimeError(f"{role} selected report has unexpected ablation")
    training_config_path = _resolved(specification["training_config"])
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    audit = {
        "role": role,
        "checkpoint_path": str(path.resolve()),
        "selected_report_path": str(report_path.resolve()),
        "selected_report_sha256": file_sha256(report_path),
        "archive_sha256": archive,
        **module_state_hashes(model),
        "model_version": model.model_version,
        "best_step": int(report["best_step"]),
        "checkpoint_step": int(metadata["step"]),
        "seed": int(metadata["seed"]),
        "ablation": report["ablation"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "component_parameter_counts": component_parameter_counts(model),
        "training_config_path": str(training_config_path.resolve()),
        "training_config_sha256": file_sha256(training_config_path),
        "embedded_training_config_sha256": sha256_hex(
            canonical_json_bytes(metadata["inductor_v1_config"])
        ),
        "external_config_matches_embedded": training_config
        == metadata["inductor_v1_config"],
    }
    return path, audit, metadata


def _interpreter_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    from vrm2.oracle.train import load_checkpoint

    specification = config["interpreter"]
    path = _resolved(specification["checkpoint"])
    model, metadata = load_checkpoint(path, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    result = {
        "checkpoint_path": str(path.resolve()),
        "archive_sha256": file_sha256(path),
        **module_state_hashes(model),
        "checkpoint_step": int(metadata["step"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "eval_mode": not model.training,
    }
    if (
        result["archive_sha256"] != specification["expected_archive_sha256"]
        or result["model_state_sha256"]
        != specification["expected_model_state_sha256"]
        or result["parameter_count"] != specification["expected_parameter_count"]
    ):
        raise RuntimeError("certified interpreter audit failed")
    return result


def shuffled_intervention_audit(
    config: Mapping[str, Any], checkpoint_audits: Mapping[str, Any]
) -> dict[str, Any]:
    from vrm2.inductor.train_v1 import shuffled_instruction_targets

    main_spec = config["checkpoints"]["main"]
    shuffled_spec = config["checkpoints"]["shuffled"]
    main_config = json.loads(_resolved(main_spec["training_config"]).read_text(encoding="utf-8"))
    shuffled_config = json.loads(
        _resolved(shuffled_spec["training_config"]).read_text(encoding="utf-8")
    )
    main_report = json.loads(_resolved(main_spec["selected_report"]).read_text(encoding="utf-8"))
    shuffled_report = json.loads(
        _resolved(shuffled_spec["selected_report"]).read_text(encoding="utf-8")
    )
    sample = torch.tensor(
        [[[1, 1, 1, 3], [2, 2, 2, 7], [0, 0, 0, 0]]], dtype=torch.long
    )
    shuffled_sample = shuffled_instruction_targets(sample, torch.tensor([2]))
    expected = torch.tensor(
        [[[2, 2, 2, 7], [1, 1, 1, 3], [0, 0, 0, 0]]], dtype=torch.long
    )
    training_main = dict(main_config["training"])
    training_shuffled = dict(shuffled_config["training"])
    main_flag = training_main.pop("shuffle_instruction_order")
    shuffled_flag = training_shuffled.pop("shuffle_instruction_order")
    equal_sections = {
        key: main_config[key] == shuffled_config[key]
        for key in (
            "protocol_version",
            "model_version",
            "model_seed",
            "typed_program_schema_version",
            "stage0_config",
            "stage0_dataset_sha256",
            "model",
            "data",
            "episodes",
            "l3_coverage",
            "evaluation",
            "gates",
        )
    }
    opened_main = main_report["opened_training_and_selection_paths"]
    opened_shuffled = shuffled_report["opened_training_and_selection_paths"]
    checks = {
        "main_shuffle_disabled": main_flag is False,
        "shuffled_order_enabled": shuffled_flag is True,
        "active_l2_instruction_order_changed": torch.equal(shuffled_sample, expected),
        "intervention_not_serialization_only": not torch.equal(shuffled_sample[:, :2], sample[:, :2]),
        "null_suffix_preserved": torch.equal(shuffled_sample[:, 2], sample[:, 2]),
        "support_and_query_data_unchanged": main_config["data"] == shuffled_config["data"]
        and main_config["episodes"] == shuffled_config["episodes"]
        and opened_main == opened_shuffled,
        "scientific_sections_equal": all(equal_sections.values()),
        "training_protocol_equal_except_intervention": training_main == training_shuffled,
        "same_seed": checkpoint_audits["main"]["seed"]
        == checkpoint_audits["shuffled"]["seed"]
        == 20260713,
        "same_architecture": main_config["model"] == shuffled_config["model"],
        "same_parameter_count": checkpoint_audits["main"]["parameter_count"]
        == checkpoint_audits["shuffled"]["parameter_count"]
        == 1349558,
        "same_actual_steps": main_report["steps_executed"]
        == shuffled_report["steps_executed"]
        == 10000,
        "same_checkpoint_selection_policy": main_config["training"]["checkpoint_selection"]
        == shuffled_config["training"]["checkpoint_selection"],
        "no_sealed_checkpoint_selection": all(
            "sealed" not in path.lower() and "coverage" not in path.lower()
            for path in (*opened_main, *opened_shuffled)
        ),
        "external_configs_match_checkpoints": checkpoint_audits["main"][
            "external_config_matches_embedded"
        ]
        and checkpoint_audits["shuffled"]["external_config_matches_embedded"],
    }
    return {
        "checks": checks,
        "intervention_valid": all(checks.values()),
        "equal_scientific_sections": equal_sections,
        "sample_before": sample.tolist(),
        "sample_after": shuffled_sample.tolist(),
        "implementation_source": inspect.getsource(shuffled_instruction_targets),
    }


def command_audit_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    checkpoint_audits = {}
    opened = [args.config.resolve()]
    for role in ("main", "shuffled"):
        path, audit, _ = resolve_selected_checkpoint(role, config["checkpoints"][role])
        checkpoint_audits[role] = audit
        opened.extend(
            [
                path,
                Path(audit["selected_report_path"]),
                Path(audit["training_config_path"]),
            ]
        )
    interpreter = _interpreter_audit(config)
    opened.append(Path(interpreter["checkpoint_path"]))
    support_path = _resolved(config["sealed_l2"]["support_path"])
    query_path = _resolved(config["sealed_l2"]["query_path"])
    from vrm2.inductor.order_causality import combined_episode_corpus_sha256

    sealed = {
        "support_path": str(support_path.resolve()),
        "query_path": str(query_path.resolve()),
        "support_sha256": file_sha256(support_path),
        "query_sha256": file_sha256(query_path),
        "episode_corpus_sha256": combined_episode_corpus_sha256(support_path, query_path),
        "episode_count": len(support_path.read_bytes().splitlines()),
    }
    expected = config["sealed_l2"]
    sealed["verified"] = (
        sealed["support_sha256"] == expected["support_sha256"]
        and sealed["query_sha256"] == expected["query_sha256"]
        and sealed["episode_corpus_sha256"] == expected["episode_corpus_sha256"]
        and sealed["episode_count"] == expected["episode_count"]
    )
    intervention = shuffled_intervention_audit(config, checkpoint_audits)
    result = {
        "manifest_version": "vrm2-order-causality-checkpoints-v1",
        "checkpoints": checkpoint_audits,
        "interpreter": interpreter,
        "sealed_l2": sealed,
        "shuffled_intervention": intervention,
        "all_preflight_checks_pass": sealed["verified"] and intervention["intervention_valid"],
    }
    output = runtime_dir(config) / "checkpoint_manifest.json"
    _write_json(output, result)
    _write_json(runtime_dir(config) / "shuffled_intervention_audit.json", intervention)
    _record_access(config, [*opened, support_path, query_path], "audit-artifacts")
    return result


def _deterministic_subset_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "compute"}


def command_build_subset(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    _require_frozen_source(config)
    from vrm2.inductor.episodes import load_query_episodes, load_support_episodes
    from vrm2.inductor.order_causality import build_order_causal_subset
    from vrm2.oracle.train import load_checkpoint

    support_path = _resolved(config["sealed_l2"]["support_path"])
    query_path = _resolved(config["sealed_l2"]["query_path"])
    interpreter_path = _resolved(config["interpreter"]["checkpoint"])
    supports = load_support_episodes(support_path)
    queries = load_query_episodes(query_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    interpreter, _ = load_checkpoint(interpreter_path, device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    hashes_before = module_state_hashes(interpreter)
    first = build_order_causal_subset(
        supports, queries, interpreter, int(config["evaluation_batch_size"]), device
    )
    second = build_order_causal_subset(
        supports, queries, interpreter, int(config["evaluation_batch_size"]), device
    )
    first_bytes = canonical_json_bytes(_deterministic_subset_view(first))
    second_bytes = canonical_json_bytes(_deterministic_subset_view(second))
    if first_bytes != second_bytes:
        raise RuntimeError("order subset construction is nondeterministic")
    first["deterministic_repeated_construction_identical"] = True
    first["deterministic_content_sha256"] = sha256_hex(first_bytes)
    first["interpreter_hashes_before"] = hashes_before
    first["interpreter_hashes_after"] = module_state_hashes(interpreter)
    first["interpreter_unchanged"] = (
        first["interpreter_hashes_before"] == first["interpreter_hashes_after"]
    )
    output = runtime_dir(config) / "subset_manifest.json"
    _write_json(output, first)
    _write_json(
        runtime_dir(config) / "subset_summary.json",
        {
            "funnel": first["funnel"],
            "reverse_validity_audit": first["reverse_validity_audit"],
            "disambiguating_support_distribution": first[
                "disambiguating_support_distribution"
            ],
            "deterministic_content_sha256": first["deterministic_content_sha256"],
            "compute": first["compute"],
        },
    )
    _record_access(
        config,
        [args.config.resolve(), support_path, query_path, interpreter_path],
        "build-subset",
    )
    return {
        "funnel": first["funnel"],
        "reverse_validity_audit": first["reverse_validity_audit"],
        "deterministic_content_sha256": first["deterministic_content_sha256"],
    }


def _prediction_records(
    role: str,
    checkpoint_sha256: str,
    supports: Sequence[Any],
    lengths: torch.Tensor,
    programs: torch.Tensor,
    confidences: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": support.episode_id,
            "checkpoint_role": role,
            "checkpoint_id": checkpoint_sha256,
            "predicted_program_length": int(lengths[index].item()),
            "predicted_typed_program_ids": programs[index].tolist(),
            "per_field_confidence": confidences[index].tolist(),
        }
        for index, support in enumerate(supports)
    ]


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    _require_frozen_source(config)
    from vrm2.inductor.episodes import load_support_episodes
    from vrm2.inductor.evaluate_v1 import predict_programs_v1
    from vrm2.inductor.order_selector import build_checkpoint_analysis_record
    from vrm2.inductor.train_v1 import load_checkpoint_v1

    role = args.checkpoint_role
    checkpoint_path, checkpoint_audit, _ = resolve_selected_checkpoint(
        role, config["checkpoints"][role]
    )
    support_path = _resolved(config["sealed_l2"]["support_path"])
    supports = load_support_episodes(support_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, _ = load_checkpoint_v1(checkpoint_path, device)
    state_before = module_state_hashes(model)
    lengths, programs, confidences, induction_seconds = predict_programs_v1(
        model, supports, int(config["evaluation_batch_size"]), device
    )
    repeated_lengths, repeated_programs, repeated_confidences, repeated_seconds = (
        predict_programs_v1(
            model, supports, int(config["evaluation_batch_size"]), device
        )
    )
    perm_lengths, perm_programs, _, permutation_seconds = predict_programs_v1(
        model,
        supports,
        int(config["evaluation_batch_size"]),
        device,
        support_permutation=(2, 0, 1),
    )
    records = _prediction_records(
        role,
        checkpoint_audit["archive_sha256"],
        supports,
        lengths,
        programs,
        confidences,
    )
    repeated = _prediction_records(
        role,
        checkpoint_audit["archive_sha256"],
        supports,
        repeated_lengths,
        repeated_programs,
        repeated_confidences,
    )
    if canonical_json_bytes(records) != canonical_json_bytes(repeated):
        raise RuntimeError("repeated frozen induction artifacts differ")
    prediction_path = runtime_dir(config) / f"{role}_predictions.jsonl"
    _write_jsonl(prediction_path, records)
    execution_path = runtime_dir(config) / f"{role}_execution.jsonl"
    execution_runtime = runtime_dir(config) / f"{role}_execution_runtime.json"
    query_path = _resolved(config["sealed_l2"]["query_path"])
    interpreter_path = _resolved(config["interpreter"]["checkpoint"])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "vrm2.inductor.order_stage_b",
            "--queries",
            str(query_path),
            "--predictions",
            str(prediction_path),
            "--interpreter-checkpoint",
            str(interpreter_path),
            "--output",
            str(execution_path),
            "--runtime",
            str(execution_runtime),
            "--batch-size",
            str(config["evaluation_batch_size"]),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    executions = _load_jsonl(execution_path)
    subset = json.loads(
        (runtime_dir(config) / "subset_manifest.json").read_text(encoding="utf-8")
    )
    episodes = {item["episode_id"]: item for item in subset["episode_index"]}
    execution_by_id = {item["episode_id"]: item for item in executions}
    analysis = [
        build_checkpoint_analysis_record(
            episodes[record["episode_id"]],
            record,
            execution_by_id[record["episode_id"]],
        )
        for record in records
    ]
    _write_jsonl(runtime_dir(config) / f"{role}_analysis.jsonl", analysis)
    hard_identity = float((lengths.eq(perm_lengths) & programs.eq(perm_programs).all(dim=(1, 2))).float().mean().item())
    stage_b = json.loads(execution_runtime.read_text(encoding="utf-8"))
    result = {
        "checkpoint_role": role,
        "checkpoint_audit_before": checkpoint_audit,
        "checkpoint_archive_sha256_after": file_sha256(checkpoint_path),
        "checkpoint_state_before": state_before,
        "checkpoint_state_after": module_state_hashes(model),
        "checkpoint_unchanged": state_before == module_state_hashes(model)
        and checkpoint_audit["archive_sha256"] == file_sha256(checkpoint_path),
        "num_episodes": len(records),
        "induction_seconds": induction_seconds,
        "repeated_induction_seconds": repeated_seconds,
        "support_permutation_induction_seconds": permutation_seconds,
        "support_permutation_hard_program_identity": hard_identity,
        "support_permutation_query_prediction_identity": 1.0 if hard_identity == 1.0 else 0.0,
        "deterministic_repeated_induction_identical": True,
        "prediction_sha256": file_sha256(prediction_path),
        "stage_b": stage_b,
        "stage_a_query_argument_present": False,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    _write_json(runtime_dir(config) / f"{role}_evaluation.json", result)
    _record_access(
        config,
        [
            args.config.resolve(),
            checkpoint_path,
            support_path,
            query_path,
            interpreter_path,
            runtime_dir(config) / "subset_manifest.json",
        ],
        f"evaluate:{role}",
    )
    return {
        "checkpoint_role": role,
        "num_episodes": len(records),
        "induction_seconds": induction_seconds,
        "support_permutation_hard_program_identity": hard_identity,
        "deterministic_repeated_induction_identical": True,
        "stage_b_support_file_loaded": stage_b["support_file_loaded"],
    }


def command_orientation_selector(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    _require_frozen_source(config)
    from vrm2.inductor.episodes import load_support_episodes
    from vrm2.inductor.order_selector import (
        ORIENTATION_AMBIGUOUS,
        support_only_orientation_selector,
    )
    from vrm2.oracle.train import load_checkpoint

    support_path = _resolved(config["sealed_l2"]["support_path"])
    supports = load_support_episodes(support_path)
    support_by_id = {item.episode_id: item for item in supports}
    subset = json.loads(
        (runtime_dir(config) / "subset_manifest.json").read_text(encoding="utf-8")
    )
    episode_by_id = {item["episode_id"]: item for item in subset["episode_index"]}
    final_ids = subset["stage_episode_ids"]["final_causal_subset"]
    predictions = {
        role: {
            item["episode_id"]: item
            for item in _load_jsonl(runtime_dir(config) / f"{role}_predictions.jsonl")
        }
        for role in ("main", "shuffled")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    interpreter_path = _resolved(config["interpreter"]["checkpoint"])
    interpreter, _ = load_checkpoint(interpreter_path, device)
    for parameter in interpreter.parameters():
        parameter.requires_grad_(False)
    interpreter.eval()
    hashes_before = module_state_hashes(interpreter)
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    records = []
    execution_seconds = 0.0
    for episode_id in final_ids:
        support = support_by_id[episode_id]
        episode = episode_by_id[episode_id]
        record: dict[str, Any] = {"episode_id": episode_id}
        for role in ("main", "shuffled"):
            prediction = predictions[role][episode_id]
            if int(prediction["predicted_program_length"]) != 2:
                record[role] = {
                    "status": "INELIGIBLE",
                    "choice": None,
                    "selected_is_ground_truth_forward": False,
                }
                continue
            selected = support_only_orientation_selector(
                interpreter,
                support.support_input_grids,
                support.support_output_grids,
                prediction["predicted_typed_program_ids"],
                int(config["evaluation_batch_size"]),
                device,
            )
            execution_seconds += float(selected["execution_seconds"])
            selected["status"] = (
                ORIENTATION_AMBIGUOUS
                if selected["choice"] == ORIENTATION_AMBIGUOUS
                else "SELECTED"
            )
            selected["selected_is_ground_truth_forward"] = (
                selected["selected_program"] == episode["forward_program"]
            )
            record[role] = selected
        ground_truth = support_only_orientation_selector(
            interpreter,
            support.support_input_grids,
            support.support_output_grids,
            episode["forward_program"],
            int(config["evaluation_batch_size"]),
            device,
        )
        execution_seconds += float(ground_truth["execution_seconds"])
        ground_truth["status"] = (
            ORIENTATION_AMBIGUOUS
            if ground_truth["choice"] == ORIENTATION_AMBIGUOUS
            else "SELECTED"
        )
        ground_truth["selected_is_ground_truth_forward"] = (
            ground_truth["selected_program"] == episode["forward_program"]
        )
        record["ground_truth"] = ground_truth
        records.append(record)
    _write_jsonl(runtime_dir(config) / "selector_analysis.jsonl", records)
    result = {
        "final_subset_episodes": len(final_ids),
        "selector_execution_seconds": execution_seconds,
        "interpreter_hashes_before": hashes_before,
        "interpreter_hashes_after": module_state_hashes(interpreter),
        "interpreter_unchanged": hashes_before == module_state_hashes(interpreter),
        "query_data_accessed": False,
        "hard_integer_program_dtype": "torch.int64",
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    _write_json(runtime_dir(config) / "selector_runtime.json", result)
    _record_access(
        config,
        [
            args.config.resolve(),
            support_path,
            interpreter_path,
            runtime_dir(config) / "subset_manifest.json",
            runtime_dir(config) / "main_predictions.jsonl",
            runtime_dir(config) / "shuffled_predictions.jsonl",
        ],
        "orientation-selector",
    )
    return result


def _mcnemar_exact_paired(counts: Mapping[str, int]) -> float:
    first = int(counts["main_correct_shuffled_wrong"])
    second = int(counts["main_wrong_shuffled_correct"])
    discordant = first + second
    if discordant == 0:
        return 1.0
    lower = min(first, second)
    probability = sum(math.comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def _render_report(summary: Mapping[str, Any]) -> str:
    funnel = summary["subset"]["funnel"]
    final = summary["metrics"]["final_causal_subset"]
    return "\n".join(
        (
            "# Frozen-checkpoint L2 order-causality diagnostic",
            "",
            f"Final causal subset: {funnel['final_causal_subset']} / 193 episodes.",
            f"Main native exact: {final['main']['native_query_exact']:.10f}.",
            f"Shuffled native exact: {final['shuffled']['native_query_exact']:.10f}.",
            f"Main unordered multiset exact: {final['main']['unordered_instruction_multiset_exact']:.10f}.",
            f"Shuffled unordered multiset exact: {final['shuffled']['unordered_instruction_multiset_exact']:.10f}.",
            "",
            f"Verdict: `{summary['verdict']}`",
            "",
            "The complete machine-readable metrics, decision table, integrity audit, and paired counts are in `summary.json`.",
        )
    ) + "\n"


def command_summarize(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    source = _require_frozen_source(config)
    from vrm2.inductor.model_v1 import SequentialResidualInductorV1
    from vrm2.inductor.order_selector import (
        aggregate_checkpoint_metrics,
        paired_query_counts,
        summarize_selector,
    )

    runtime = runtime_dir(config)
    checkpoint_manifest = json.loads(
        (runtime / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    subset = json.loads((runtime / "subset_manifest.json").read_text(encoding="utf-8"))
    analyses = {
        role: _load_jsonl(runtime / f"{role}_analysis.jsonl")
        for role in ("main", "shuffled")
    }
    evaluations = {
        role: json.loads((runtime / f"{role}_evaluation.json").read_text(encoding="utf-8"))
        for role in ("main", "shuffled")
    }
    selectors_nested = _load_jsonl(runtime / "selector_analysis.jsonl")
    selector_records = {
        role: [
            {"episode_id": item["episode_id"], **item[role]}
            for item in selectors_nested
        ]
        for role in ("main", "shuffled", "ground_truth")
    }
    subset_ids = {
        "all_sealed_l2": [item["episode_id"] for item in subset["episode_index"]],
        "query_order_sensitive": subset["stage_episode_ids"]["query_order_sensitive"],
        "support_identifiable": subset["stage_episode_ids"]["support_identifiable"],
        "final_causal_subset": subset["stage_episode_ids"]["final_causal_subset"],
    }
    metrics = {
        name: {
            role: aggregate_checkpoint_metrics(analyses[role], episode_ids)
            for role in ("main", "shuffled")
        }
        for name, episode_ids in subset_ids.items()
    }
    final_ids = subset_ids["final_causal_subset"]
    selector_metrics = {
        role: summarize_selector(
            selector_records[role], analyses[role], final_ids
        )
        for role in ("main", "shuffled")
    }
    selector_metrics["ground_truth"] = summarize_selector(
        selector_records["ground_truth"], [], final_ids, ground_truth=True
    )
    final = metrics["final_causal_subset"]
    deltas = {
        "unordered_instruction_multiset_exact": final["main"][
            "unordered_instruction_multiset_exact"
        ]
        - final["shuffled"]["unordered_instruction_multiset_exact"],
        "ordered_program_exact": final["main"]["ordered_program_exact"]
        - final["shuffled"]["ordered_program_exact"],
        "correct_orientation_conditional_rate": final["main"][
            "correct_orientation_conditional_rate"
        ]
        - final["shuffled"]["correct_orientation_conditional_rate"],
        "native_query_exact": final["main"]["native_query_exact"]
        - final["shuffled"]["native_query_exact"],
        "reversed_query_exact": final["main"]["reversed_query_exact"]
        - final["shuffled"]["reversed_query_exact"],
        "reverse_rescue_rate": final["main"]["reverse_rescue_rate"]
        - final["shuffled"]["reverse_rescue_rate"],
        "reverse_destruction_rate": final["main"]["reverse_destruction_rate"]
        - final["shuffled"]["reverse_destruction_rate"],
        "support_selected_query_exact": selector_metrics["main"][
            "selected_orientation_query_exact"
        ]
        - selector_metrics["shuffled"]["selected_orientation_query_exact"],
    }
    paired = paired_query_counts(analyses["main"], analyses["shuffled"], final_ids)
    paired["exact_mcnemar_two_sided_p"] = _mcnemar_exact_paired(paired)

    current_checkpoints = {}
    checkpoint_unchanged = True
    for role in ("main", "shuffled"):
        _, audit, _ = resolve_selected_checkpoint(role, config["checkpoints"][role])
        current_checkpoints[role] = audit
        checkpoint_unchanged &= all(
            audit[key] == checkpoint_manifest["checkpoints"][role][key]
            for key in (
                "archive_sha256",
                "model_state_sha256",
                "parameter_state_sha256",
                "buffer_state_sha256",
            )
        )
    current_interpreter = _interpreter_audit(config)
    interpreter_unchanged = all(
        current_interpreter[key] == checkpoint_manifest["interpreter"][key]
        for key in (
            "archive_sha256",
            "model_state_sha256",
            "parameter_state_sha256",
            "buffer_state_sha256",
        )
    )
    access = json.loads((runtime / "access_audit.json").read_text(encoding="utf-8"))
    stage_b_source = (REPOSITORY_ROOT / "vrm2/inductor/order_stage_b.py").read_text(
        encoding="utf-8"
    )
    induction_parameters = inspect.signature(
        SequentialResidualInductorV1.forward_autoregressive
    ).parameters
    integrity_checks = {
        "source_unchanged": source["source_unchanged"],
        "shuffled_intervention_valid": checkpoint_manifest["shuffled_intervention"][
            "intervention_valid"
        ],
        "sealed_l2_verified": checkpoint_manifest["sealed_l2"]["verified"],
        "subset_model_independent": subset["model_independent_selection"]
        and not subset.get("selection_uses_model_predictions", False),
        "subset_deterministic": subset["deterministic_repeated_construction_identical"],
        "main_support_permutation_program_identity": evaluations["main"][
            "support_permutation_hard_program_identity"
        ]
        == 1.0,
        "main_support_permutation_query_identity": evaluations["main"][
            "support_permutation_query_prediction_identity"
        ]
        == 1.0,
        "shuffled_support_permutation_program_identity": evaluations["shuffled"][
            "support_permutation_hard_program_identity"
        ]
        == 1.0,
        "shuffled_support_permutation_query_identity": evaluations["shuffled"][
            "support_permutation_query_prediction_identity"
        ]
        == 1.0,
        "hard_integer_boundary": all(
            evaluations[role]["stage_b"]["hard_integer_program_dtype"] == "torch.int64"
            for role in ("main", "shuffled")
        ),
        "fresh_stage_b_support_exclusion": all(
            not evaluations[role]["stage_b"]["support_file_loaded"]
            for role in ("main", "shuffled")
        )
        and "load_support_episodes" not in stage_b_source,
        "no_query_to_inductor_api": not any(
            "query" in name.lower() for name in induction_parameters
        ),
        "checkpoints_unchanged": checkpoint_unchanged,
        "interpreter_unchanged": interpreter_unchanged,
        "coverage_l3_payload_access_count_zero": access[
            "coverage_l3_payload_access_count"
        ]
        == 0
        and not set(access["opened_paths"])
        & set(access["prohibited_coverage_payloads"]),
        "deterministic_repeated_induction": all(
            evaluations[role]["deterministic_repeated_induction_identical"]
            for role in ("main", "shuffled")
        ),
    }
    structural_valid = all(integrity_checks.values())
    thresholds = config["decision_thresholds"]
    final_count = len(final_ids)
    contradiction_query = final["shuffled"]["native_query_exact"] >= final["main"][
        "native_query_exact"
    ] + thresholds["contradiction_query_delta"]
    contradiction_orientation = final["shuffled"][
        "correct_orientation_conditional_rate"
    ] >= final["main"]["correct_orientation_conditional_rate"] + thresholds[
        "contradiction_orientation_delta"
    ]
    order_bottleneck_by_role = {
        role: (
            final[role]["order_loss"] >= thresholds["order_loss"]
            and selector_metrics[role]["query_exact_gain_over_native"]
            >= thresholds["selector_gain"]
        )
        for role in ("main", "shuffled")
    }
    identity_bottleneck_by_role = {
        role: (
            final[role]["unordered_instruction_multiset_exact"]
            < thresholds["identity_ceiling"]
            and selector_metrics[role]["query_exact_gain_over_native"]
            < thresholds["selector_gain"]
        )
        for role in ("main", "shuffled")
    }
    if not source["source_unchanged"]:
        verdict = "ORDER_CAUSALITY_INVALID_SOURCE_CHANGED"
    elif not checkpoint_manifest["shuffled_intervention"]["intervention_valid"]:
        verdict = "ORDER_CAUSALITY_INVALID_SHUFFLE_INTERVENTION"
    elif not structural_valid:
        verdict = "ORDER_CAUSALITY_INVALID"
    elif final_count < int(config["minimum_final_causal_episodes"]):
        verdict = "ORDER_CAUSALITY_INSUFFICIENT_COVERAGE"
    elif contradiction_query and contradiction_orientation:
        verdict = "ORDER_MECHANISM_CONTRADICTED"
    elif any(order_bottleneck_by_role.values()):
        verdict = "ORDER_DECODING_BOTTLENECK_SUPPORTED"
    elif all(identity_bottleneck_by_role.values()):
        verdict = "INSTRUCTION_IDENTITY_BOTTLENECK_SUPPORTED"
    else:
        verdict = "ORDER_CAUSALITY_MIXED"
    decision_table = {
        "structural_validity": {
            "observed": structural_valid,
            "operator": "==",
            "threshold": True,
            "result": structural_valid,
        },
        "minimum_coverage": {
            "observed": final_count,
            "operator": ">=",
            "threshold": int(config["minimum_final_causal_episodes"]),
            "result": final_count >= int(config["minimum_final_causal_episodes"]),
        },
        "contradiction_query": {
            "observed": final["shuffled"]["native_query_exact"]
            - final["main"]["native_query_exact"],
            "operator": ">=",
            "threshold": thresholds["contradiction_query_delta"],
            "result": contradiction_query,
        },
        "contradiction_orientation": {
            "observed": final["shuffled"]["correct_orientation_conditional_rate"]
            - final["main"]["correct_orientation_conditional_rate"],
            "operator": ">=",
            "threshold": thresholds["contradiction_orientation_delta"],
            "result": contradiction_orientation,
        },
        "order_bottleneck_by_role": order_bottleneck_by_role,
        "identity_bottleneck_by_role": identity_bottleneck_by_role,
    }
    selector_runtime = json.loads(
        (runtime / "selector_runtime.json").read_text(encoding="utf-8")
    )
    summary = {
        "protocol_version": config["protocol_version"],
        "subset": {
            "funnel": subset["funnel"],
            "reverse_validity_audit": subset["reverse_validity_audit"],
            "disambiguating_support_distribution": subset[
                "disambiguating_support_distribution"
            ],
        },
        "metrics": metrics,
        "selectors": selector_metrics,
        "behavioral_decomposition": {
            role: {
                "native_query_exact": final[role]["native_query_exact"],
                "support_selected_predicted_multiset_query_exact": selector_metrics[role][
                    "selected_orientation_query_exact"
                ],
                "oracle_multiset_support_selected_query_exact": selector_metrics[
                    "ground_truth"
                ]["selected_orientation_query_exact"],
                "oracle_program_query_exact": 1.0,
            }
            for role in ("main", "shuffled")
        },
        "main_minus_shuffled_deltas": deltas,
        "paired_query_comparison": paired,
        "integrity": {
            "checks": integrity_checks,
            "all_pass": structural_valid,
            "checkpoint_postrun": current_checkpoints,
            "interpreter_postrun": current_interpreter,
            "coverage_l3_access_count": access["coverage_l3_payload_access_count"],
        },
        "compute": {
            "subset_construction_seconds": subset["compute"][
                "subset_construction_seconds"
            ],
            "main_induction_seconds": evaluations["main"]["induction_seconds"],
            "shuffled_induction_seconds": evaluations["shuffled"]["induction_seconds"],
            "main_execution_seconds": evaluations["main"]["stage_b"][
                "total_execution_seconds"
            ],
            "shuffled_execution_seconds": evaluations["shuffled"]["stage_b"][
                "total_execution_seconds"
            ],
            "selector_seconds": selector_runtime["selector_execution_seconds"],
            "peak_vram_bytes": max(
                subset["compute"]["peak_vram_bytes"],
                evaluations["main"]["peak_vram_bytes"],
                evaluations["shuffled"]["peak_vram_bytes"],
                selector_runtime["peak_vram_bytes"],
            ),
        },
        "decision_table": decision_table,
        "verdict": verdict,
    }
    _write_json(runtime / "summary.json", summary)
    (runtime / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def command_freeze_source(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    result = build_source_manifest()
    _write_json(runtime_dir(config) / "source_manifest.json", result)
    return result


def command_verify_source(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    result = verify_source_manifest(runtime_dir(config) / "source_manifest.json")
    output = args.output or runtime_dir(config) / "source_verification.json"
    _write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen L2 order-causality diagnostic")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("audit-artifacts", command_audit_artifacts),
        ("build-subset", command_build_subset),
        ("orientation-selector", command_orientation_selector),
        ("summarize", command_summarize),
        ("freeze-source", command_freeze_source),
        ("verify-source", command_verify_source),
    ):
        item = sub.add_parser(name)
        item.add_argument("--config", type=Path, required=True)
        if name == "verify-source":
            item.add_argument("--output", type=Path)
        item.set_defaults(function=function)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument(
        "--checkpoint-role", choices=("main", "shuffled"), required=True
    )
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.function(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    valid = result.get("source_unchanged", True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
