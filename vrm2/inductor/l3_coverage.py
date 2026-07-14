from __future__ import annotations

import collections
import itertools
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from vrm2.data.composition_split import build_split_manifest, parse_signature, stable_seed
from vrm2.data.microworld import GenerationLimitError, generate_episode_with_diagnostics
from vrm2.data.serialization import canonical_json_bytes, sha256_hex
from vrm2.data.types import TaskEpisode, grid_as_lists
from vrm2.inductor.episodes import write_episode_corpus
from vrm2.inductor.types import QueryEpisode, SupportEpisode
from vrm2.oracle.dataset import example_fingerprint
from vrm2.oracle.types import OracleExample


COVERAGE_CORPUS_VERSION = "vrm2-inductor-v1-coverage-l3-v1"
NULL_INSTRUCTION = (0, 0, 0, 0)


def _typed_program(episode: TaskEpisode) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(item.as_ids() for item in episode.hidden_program.typed_instructions)


def _scene_fingerprint(
    input_grid: Sequence[Sequence[int]],
    program: Sequence[Sequence[int]],
    output_grid: Sequence[Sequence[int]],
) -> str:
    return sha256_hex(
        canonical_json_bytes(
            {
                "input_grid": input_grid,
                "typed_program_ids": program,
                "target_output_grid": output_grid,
            }
        )
    )


def _prior_fingerprints(
    datasets: Mapping[str, Sequence[OracleExample]]
) -> dict[str, set[str]]:
    return {
        name: {example_fingerprint(item) for item in examples}
        for name, examples in datasets.items()
    }


def eligibility_audit(
    stage0_config: Mapping[str, Any],
    training_examples: Sequence[OracleExample],
    seed: int,
) -> dict[str, Any]:
    split_manifest = build_split_manifest(
        int(stage0_config["seeds"]["composition_split"]),
        str(stage0_config["split_version"]),
    )
    train_compositions = {item.composition_signature for item in training_examples}
    training_length2 = {
        signature for signature in train_compositions if len(parse_signature(signature)) == 2
    }
    training_categories = {
        tuple(instruction[:3])
        for example in training_examples
        for instruction in example.typed_program_ids
    }
    training_values_by_category: dict[tuple[int, int, int], set[int]] = collections.defaultdict(set)
    for example in training_examples:
        for instruction in example.typed_program_ids:
            training_values_by_category[tuple(instruction[:3])].add(int(instruction[3]))
    candidates = sorted(
        signature
        for signature in split_manifest["splits"]["sealed"]["signatures"]
        if len(parse_signature(signature)) == 3
    )
    records = []
    eligible = []
    for index, signature in enumerate(candidates):
        names = parse_signature(signature)
        subsequences = ["->".join(pair) for pair in itertools.combinations(names, 2)]
        subsequence_hits = sorted(set(subsequences) & training_length2)
        generated = None
        generation_error = None
        try:
            generated, _ = generate_episode_with_diagnostics(
                signature,
                seed + 10_000_000 + index,
                "sealed",
                stage0_config,
            )
        except GenerationLimitError as exc:
            generation_error = str(exc)
        typed = _typed_program(generated) if generated is not None else ()
        categories = [tuple(item[:3]) for item in typed]
        category_coverage = all(item in training_categories for item in categories)
        value_coverage = all(
            int(item[3]) in training_values_by_category[tuple(item[:3])] for item in typed
        )
        complete_absent = signature not in train_compositions
        valid_four_scenes = generated is not None
        is_eligible = (
            category_coverage
            and value_coverage
            and bool(subsequence_hits)
            and complete_absent
            and valid_four_scenes
        )
        if is_eligible:
            eligible.append(signature)
        records.append(
            {
                "composition_signature": signature,
                "instruction_categories": [list(item) for item in categories],
                "instruction_values_seen_in_training": value_coverage,
                "instruction_categories_seen_in_training": category_coverage,
                "training_length2_subsequence_hits": subsequence_hits,
                "complete_l3_signature_absent_from_training": complete_absent,
                "deterministically_generates_four_valid_scenes": valid_four_scenes,
                "generation_error": generation_error,
                "eligible": is_eligible,
            }
        )
    return {
        "audit_version": COVERAGE_CORPUS_VERSION,
        "candidate_count": len(candidates),
        "eligible_signature_count": len(eligible),
        "eligible_composition_signatures": eligible,
        "training_instruction_categories": [list(item) for item in sorted(training_categories)],
        "training_values_by_category": {
            canonical_json_bytes(list(key)).decode("utf-8"): sorted(values)
            for key, values in sorted(training_values_by_category.items())
        },
        "records": records,
        "all_eligibility_rules_pass": bool(eligible),
    }


def build_coverage_corpus(
    stage0_config: Mapping[str, Any],
    datasets: Mapping[str, Sequence[OracleExample]],
    *,
    seed: int = 830000,
    episode_count: int = 512,
) -> tuple[tuple[SupportEpisode, ...], tuple[QueryEpisode, ...], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    audit = eligibility_audit(stage0_config, datasets["train"], seed)
    eligible = list(audit["eligible_composition_signatures"])
    if not eligible:
        raise RuntimeError("INDUCTOR_V1_INVALID_L3_CORPUS")
    random.Random(stable_seed(COVERAGE_CORPUS_VERSION, seed, "eligible-order")).shuffle(eligible)
    prior = _prior_fingerprints(datasets)
    prior_union = set().union(*prior.values())
    seen_examples: set[str] = set()
    supports = []
    queries = []
    combined_records = []
    composition_counts: collections.Counter[str] = collections.Counter()
    typed_counts: collections.Counter[str] = collections.Counter()
    overlap_counts: collections.Counter[str] = collections.Counter()
    rejection_counts: collections.Counter[str] = collections.Counter()
    generation_attempts = 0
    episode_attempts_total = 0
    scene_attempts_total = 0
    for index in range(episode_count):
        signature = eligible[index % len(eligible)]
        accepted = None
        accepted_diagnostics = None
        accepted_fingerprints = None
        for retry in range(32):
            generation_attempts += 1
            episode_seed = seed + index + retry * 1_000_000
            try:
                episode, diagnostics = generate_episode_with_diagnostics(
                    signature, episode_seed, "sealed", stage0_config
                )
            except GenerationLimitError:
                rejection_counts["generation_limit"] += 1
                continue
            program = _typed_program(episode)
            pairs = [*episode.support]
            scene_pairs = [(item.input_grid, item.output_grid) for item in pairs]
            scene_pairs.append((episode.query_input, episode.query_output))
            fingerprints = [
                _scene_fingerprint(input_grid, program, output_grid)
                for input_grid, output_grid in scene_pairs
            ]
            input_grids = [input_grid for input_grid, _ in scene_pairs]
            if len(set(fingerprints)) != 4 or len(set(input_grids)) != 4:
                rejection_counts["episode_duplicate"] += 1
                continue
            if set(fingerprints) & (prior_union | seen_examples):
                rejection_counts["fingerprint_overlap"] += 1
                continue
            accepted = episode
            accepted_diagnostics = diagnostics
            accepted_fingerprints = fingerprints
            break
        if accepted is None or accepted_diagnostics is None or accepted_fingerprints is None:
            raise RuntimeError("INDUCTOR_V1_INVALID_L3_CORPUS")
        program = _typed_program(accepted)
        padded = tuple(tuple(item) for item in program)
        typed_key = canonical_json_bytes([list(item) for item in program]).decode("utf-8")
        episode_id = sha256_hex(
            canonical_json_bytes(
                [COVERAGE_CORPUS_VERSION, seed, index, signature, accepted_fingerprints]
            )
        )[:24]
        support = SupportEpisode(
            episode_id=episode_id,
            support_input_grids=tuple(item.input_grid for item in accepted.support),  # type: ignore[arg-type]
            support_output_grids=tuple(item.output_grid for item in accepted.support),  # type: ignore[arg-type]
            target_program_length=3,
            target_typed_program_ids=padded,  # type: ignore[arg-type]
        )
        query = QueryEpisode(
            episode_id=episode_id,
            query_input_grid=accepted.query_input,
            query_target_grid=accepted.query_output,
            evaluator_program_length=3,
            evaluator_typed_program_ids=padded,  # type: ignore[arg-type]
        )
        supports.append(support)
        queries.append(query)
        seen_examples.update(accepted_fingerprints)
        for corpus_name, corpus_fingerprints in prior.items():
            overlap_counts[corpus_name] += len(set(accepted_fingerprints) & corpus_fingerprints)
        composition_counts[signature] += 1
        typed_counts[typed_key] += 1
        rejection_counts.update(accepted_diagnostics.rejection_counts)
        episode_attempts_total += accepted_diagnostics.episode_attempt
        scene_attempts_total += sum(accepted_diagnostics.scene_attempts)
        combined_records.append(
            {
                "episode_id": episode_id,
                "supports": [
                    {
                        "input_grid": grid_as_lists(item.input_grid),
                        "output_grid": grid_as_lists(item.output_grid),
                    }
                    for item in accepted.support
                ],
                "query_input_grid": grid_as_lists(accepted.query_input),
                "query_target_grid": grid_as_lists(accepted.query_output),
                "evaluator_private": {
                    "program_length": 3,
                    "typed_program_ids": [list(item) for item in program],
                },
                "audit_private": {
                    "composition_signature": signature,
                    "example_fingerprints": accepted_fingerprints,
                },
            }
        )
    if len(supports) != 512:
        raise RuntimeError("INDUCTOR_V1_INVALID_L3_CORPUS")
    if any(overlap_counts.values()):
        raise RuntimeError("INDUCTOR_V1_INVALID_L3_CORPUS")
    count_values = list(composition_counts.values())
    if max(count_values) - min(count_values) > 1:
        raise AssertionError("coverage L3 composition usage is not balanced")
    manifest = {
        "corpus_version": COVERAGE_CORPUS_VERSION,
        "seed": seed,
        "episode_count": len(supports),
        "scene_example_count": 4 * len(supports),
        "eligible_signature_count": len(eligible),
        "composition_signature_count": len(composition_counts),
        "exact_typed_signature_count": len(typed_counts),
        "episodes_per_composition_signature": dict(sorted(composition_counts.items())),
        "episodes_per_exact_typed_signature": dict(sorted(typed_counts.items())),
        "fingerprint_overlap_counts": {
            name: overlap_counts[name] for name in sorted(prior)
        },
        "complete_l3_signatures_absent_from_training": True,
        "generation_attempts": generation_attempts,
        "episode_attempts_total": episode_attempts_total,
        "scene_attempts_total": scene_attempts_total,
        "rejected_scenes": dict(sorted(rejection_counts.items())),
        "balanced_composition_usage": True,
    }
    return tuple(supports), tuple(queries), combined_records, manifest, audit


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"\n".join(canonical_json_bytes(item) for item in records) + b"\n"


def write_coverage_corpus(
    output_dir: Path,
    supports: Sequence[SupportEpisode],
    queries: Sequence[QueryEpisode],
    combined_records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    deterministic_regeneration_identical: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_manifest = write_episode_corpus(
        output_dir, "coverage_l3", supports, queries, manifest
    )
    episode_bytes = _jsonl_bytes(combined_records)
    episode_path = output_dir / "episodes.jsonl"
    episode_path.write_bytes(episode_bytes)
    final_manifest = {
        **dict(manifest),
        "corpus_sha256": sha256_hex(episode_bytes),
        "corpus_bytes": len(episode_bytes),
        "support_sha256": episode_manifest["support_sha256"],
        "query_sha256": episode_manifest["query_sha256"],
        "deterministic_regeneration_identical": deterministic_regeneration_identical,
    }
    (output_dir / "manifest.json").write_bytes(
        canonical_json_bytes(final_manifest) + b"\n"
    )
    (output_dir / "eligibility_audit.json").write_bytes(
        canonical_json_bytes(audit) + b"\n"
    )
    access = {
        "corpus_generated_before_training": True,
        "deterministic_regeneration_performed": True,
        "opened_for_training": False,
        "opened_for_checkpoint_selection": False,
        "phase_b_evaluation_count": 0,
    }
    (output_dir / "access_audit.json").write_bytes(canonical_json_bytes(access) + b"\n")
    return final_manifest


def deterministic_corpus_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return _jsonl_bytes(records)
