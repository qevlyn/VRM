from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .composition_r4a0r_data import (
    R4A0REpisode,
    R4A0RSupportPair,
    all_query_types,
    apply_sequence_to_grid,
    grid_to_list,
    parse_grid,
    partition_query_types,
    query_type_from_dict,
    query_type_key,
    query_type_to_dict,
    render_scene,
    render_support_scene,
    stable_hash,
    stable_int,
    support_query_type,
)
from .composition_r4a1p_oracles import (
    affine_for_word,
    affine_key,
    affine_payload,
    arrow_word,
    compact_word,
    enumerate_length3_words,
    is_scene_safe_for_word,
    output_hash_for_word,
)

PROJECT_ROOT = Path("/mnt/d/VRM")
R4A0R_ARTIFACTS = PROJECT_ROOT / "artifacts/composition_r4a0r_v2"
TRAIN_VISIBLE_STRATUM = "TRAIN_VISIBLE_VISUAL_TYPES"
SEALED_STRATUM = "SEALED_VISUAL_TYPES"
VISUAL_STRATA = (TRAIN_VISIBLE_STRATUM, SEALED_STRATUM)
REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION = 24
REQUIRED_COUNTERFACTUAL_GROUPS = 1000


def load_frozen_query_type_manifest(path: Path = R4A0R_ARTIFACTS) -> dict[str, list[tuple[str, int, str, int]]]:
    manifest_path = Path(path) / "heldout_query_type_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "train": [query_type_from_dict(item) for item in raw["train"]],
        "dev": [query_type_from_dict(item) for item in raw["dev"]],
        "sealed": [query_type_from_dict(item) for item in raw["sealed"]],
    }


def default_probe_query_types() -> dict[str, list[tuple[str, int, str, int]]]:
    split = load_frozen_query_type_manifest()
    return {
        TRAIN_VISIBLE_STRATUM: split["train"],
        SEALED_STRATUM: split["sealed"],
    }


def prior_function_signatures() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = defaultdict(list)
    rows[affine_key(affine_for_word(""))].append("I")
    for length in (1, 2):
        for word in ("".join(parts) for parts in itertools.product(("C", "F", "T"), repeat=length)):
            rows[affine_key(affine_for_word(word))].append(arrow_word(word))
    return {key: sorted(values) for key, values in rows.items()}


def _variant(word: str, qtype: tuple[str, int, str, int], attempt: int, salt: str) -> int:
    return stable_int(f"r4a1p|{word}|{query_type_key(qtype)}|{attempt}|{salt}") % 10_000_000


def safe_query_scene_count(
    word: str,
    qtype: tuple[str, int, str, int],
    *,
    required: int = REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
    max_attempts: int = 512,
) -> tuple[int, list[str]]:
    count = 0
    seen: set[str] = set()
    reasons: list[str] = []
    for attempt in range(max_attempts):
        try:
            scene = render_scene(qtype, _variant(word, qtype, attempt, "catalog"), f"r4a1p|catalog|{word}|{attempt}")
        except RuntimeError as exc:
            reasons.append(str(exc))
            continue
        scene_hash = stable_hash(grid_to_list(scene))
        if scene_hash in seen:
            continue
        ok, scene_reasons = is_scene_safe_for_word(scene, word)
        if ok:
            seen.add(scene_hash)
            count += 1
            if count >= required:
                break
        else:
            reasons.extend(scene_reasons)
    return count, sorted(set(reasons))[:20]


def build_length3_function_catalog(
    probe_query_types: dict[str, list[tuple[str, int, str, int]]] | None = None,
) -> dict[str, Any]:
    probe_query_types = probe_query_types or default_probe_query_types()
    universal_query_types = probe_query_types[TRAIN_VISIBLE_STRATUM] + probe_query_types[SEALED_STRATUM]
    prior = prior_function_signatures()
    words = enumerate_length3_words()
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        groups[affine_key(affine_for_word(word))].append(word)

    group_entries: list[dict[str, Any]] = []
    selected: list[str] = []
    for key in sorted(groups, key=lambda item: min(groups[item])):
        equivalents = sorted(groups[key])
        canonical = equivalents[0]
        equals_prior = key in prior
        bad_qtypes: list[dict[str, Any]] = []
        min_safe_count = REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION
        if not equals_prior:
            for qtype in universal_query_types:
                count, reasons = safe_query_scene_count(canonical, qtype)
                min_safe_count = min(min_safe_count, count)
                if count < REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION:
                    bad_qtypes.append(
                        {
                            "query_type_key": query_type_key(qtype),
                            "safe_scene_count": count,
                            "required": REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
                            "reasons": reasons,
                        }
                    )
                    if len(bad_qtypes) >= 8:
                        break
        eligible = (not equals_prior) and not bad_qtypes
        if eligible:
            selected.append(canonical)
        reason = ""
        if equals_prior:
            reason = f"equals prior function(s): {', '.join(prior[key])}"
        elif bad_qtypes:
            reason = "not universally renderable across frozen train/sealed query types"
        group_entries.append(
            {
                "canonical_representative": canonical,
                "sequence_class": arrow_word(canonical),
                "equivalent_words": equivalents,
                "affine": affine_payload(canonical),
                "equals_prior_function": equals_prior,
                "matching_prior_functions": prior.get(key, []),
                "eligibility": eligible,
                "exclusion_reason": reason,
                "minimum_safe_scenes_per_frozen_query_type": min_safe_count,
                "bad_query_type_examples": bad_qtypes,
            }
        )

    word_entries = []
    for word in words:
        key = affine_key(affine_for_word(word))
        canonical = sorted(groups[key])[0]
        group = next(entry for entry in group_entries if entry["canonical_representative"] == canonical)
        word_entries.append(
            {
                "word": word,
                "sequence_class": arrow_word(word),
                "affine": affine_payload(word),
                "canonical_representative": canonical,
                "equivalent_words": sorted(groups[key]),
                "equals_prior_function": key in prior,
                "matching_prior_functions": prior.get(key, []),
                "eligibility": bool(group["eligibility"]),
                "exclusion_reason": group["exclusion_reason"],
            }
        )

    return {
        "experiment": "R4A-1P frozen length-three composition probe",
        "all_length3_words": words,
        "word_count": len(words),
        "prior_function_set": {
            "identity": ["I"],
            "single_step": [arrow_word(word) for word in ("C", "F", "T")],
            "all_two_step": [arrow_word("".join(parts)) for parts in itertools.product(("C", "F", "T"), repeat=2)],
        },
        "affine_equivalence_class_count": len(groups),
        "canonical_groups": group_entries,
        "word_entries": word_entries,
        "eligible_canonical_words": selected,
        "eligible_canonical_sequences": [arrow_word(word) for word in selected],
        "eligible_count": len(selected),
        "required_minimum_eligible_count": 12,
        "passed": len(selected) >= 12,
        "query_type_scope_for_universal_feasibility": {
            TRAIN_VISIBLE_STRATUM: len(probe_query_types[TRAIN_VISIBLE_STRATUM]),
            SEALED_STRATUM: len(probe_query_types[SEALED_STRATUM]),
        },
    }


def _find_query_scene(
    word: str,
    qtype: tuple[str, int, str, int],
    local_index: int,
    salt: str,
    used_hashes: set[str],
    rejections: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    for attempt in range(2048):
        variant = _variant(word, qtype, local_index * 4099 + attempt, salt)
        try:
            scene = render_scene(qtype, variant, f"r4a1p|{salt}|query|{word}|{local_index}|{attempt}")
        except RuntimeError as exc:
            rejections[f"render:{exc}"] += 1
            continue
        scene_hash = stable_hash(grid_to_list(scene))
        if scene_hash in used_hashes:
            rejections["duplicate_query_scene"] += 1
            continue
        ok, reasons = is_scene_safe_for_word(scene, word)
        if not ok:
            for reason in reasons:
                rejections[reason] += 1
            continue
        used_hashes.add(scene_hash)
        return scene
    raise RuntimeError(f"failed to find safe query scene for {word} {query_type_key(qtype)}")


def _find_support_input(
    word: str,
    qtype: tuple[str, int, str, int],
    query: tuple[tuple[int, ...], ...],
    target: tuple[tuple[int, ...], ...],
    pair_index: int,
    local_index: int,
    salt: str,
    rejections: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    for attempt in range(2048):
        sqtype = support_query_type(qtype, f"r4a1p|{word}|{salt}|{local_index}|{pair_index}|{attempt}")
        variant = stable_int(f"r4a1p|support|{word}|{query_type_key(qtype)}|{local_index}|{pair_index}|{attempt}") % 10_000_000
        try:
            support_input = render_support_scene(sqtype, variant, pair_index, f"r4a1p|{salt}|support|{word}")
        except RuntimeError as exc:
            rejections[f"support_render:{exc}"] += 1
            continue
        if support_input == query:
            rejections["support_input_equals_query"] += 1
            continue
        ok, reasons = is_scene_safe_for_word(support_input, word)
        if not ok:
            for reason in reasons:
                rejections[f"support_{reason}"] += 1
            continue
        support_output = apply_sequence_to_grid(support_input, arrow_word(word))
        if support_output == target:
            rejections["support_output_equals_target"] += 1
            continue
        return support_input
    raise RuntimeError(f"failed to find safe support input for {word} {query_type_key(qtype)}")


def make_probe_episode(
    stratum: str,
    qtype: tuple[str, int, str, int],
    word: str,
    local_index: int,
    used_hashes: set[str],
    rejections: dict[str, int],
) -> R4A0REpisode:
    query = _find_query_scene(word, qtype, local_index, stratum, used_hashes, rejections)
    target = apply_sequence_to_grid(query, arrow_word(word))
    supports: list[R4A0RSupportPair] = []
    for pair_index in range(3):
        support_input = _find_support_input(word, qtype, query, target, pair_index, local_index, stratum, rejections)
        supports.append(R4A0RSupportPair(input=support_input, output=apply_sequence_to_grid(support_input, arrow_word(word))))
    return R4A0REpisode(
        episode_id=f"r4a1p:{stratum}:{query_type_key(qtype)}:{word}:idx={local_index}",
        split=stratum,
        sequence_class=arrow_word(word),
        query_type=qtype,
        counterfactual_group_id=None,
        query=query,
        supports=tuple(supports),
        target=target,
    )


def episode_to_record(episode: R4A0REpisode, word: str | None = None) -> dict[str, Any]:
    canonical = word or compact_word(episode.sequence_class)
    return episode.to_record() | {
        "canonical_word": canonical,
        "canonical_sequence": arrow_word(canonical),
        "visual_stratum": episode.split,
        "affine_signature": affine_payload(canonical),
        "evaluation_only": True,
    }


def episode_from_record(record: dict[str, Any]) -> R4A0REpisode:
    return R4A0REpisode(
        episode_id=str(record["episode_id"]),
        split=str(record["split"]),
        sequence_class=str(record["sequence_class"]),
        query_type=query_type_from_dict(record["query_type"]),
        counterfactual_group_id=record.get("counterfactual_group_id"),
        query=parse_grid(record["query"]),
        supports=tuple(R4A0RSupportPair(input=parse_grid(pair["input"]), output=parse_grid(pair["output"])) for pair in record["supports"]),
        target=parse_grid(record["target"]),
    )


def write_jsonl(path: Path, episodes: Iterable[R4A0REpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode_to_record(episode), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[R4A0REpisode]:
    rows: list[R4A0REpisode] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(episode_from_record(json.loads(line)))
    return rows


def build_length3_probe_dataset(
    eligible_words: list[str],
    probe_query_types: dict[str, list[tuple[str, int, str, int]]] | None = None,
) -> tuple[list[R4A0REpisode], dict[str, Any], dict[str, Any], dict[str, Any]]:
    probe_query_types = probe_query_types or default_probe_query_types()
    episodes: list[R4A0REpisode] = []
    rejections: dict[str, int] = defaultdict(int)
    feasibility: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for stratum in VISUAL_STRATA:
        counts[stratum] = {}
        feasibility[stratum] = {}
        for word in eligible_words:
            counts[stratum][word] = 0
            feasibility[stratum][word] = {}
            for qtype in sorted(probe_query_types[stratum], key=query_type_key):
                used_hashes: set[str] = set()
                qkey = query_type_key(qtype)
                before_count = len(episodes)
                try:
                    for idx in range(REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION):
                        episodes.append(make_probe_episode(stratum, qtype, word, idx, used_hashes, rejections))
                except RuntimeError as exc:
                    feasibility[stratum][word][qkey] = {
                        "feasible": False,
                        "generated": len(episodes) - before_count,
                        "required": REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
                        "failure": str(exc),
                    }
                    continue
                generated = len(episodes) - before_count
                counts[stratum][word] += generated
                feasibility[stratum][word][qkey] = {
                    "feasible": generated == REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
                    "generated": generated,
                    "required": REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
                }
    all_feasible = all(row["feasible"] for by_word in feasibility.values() for by_qtype in by_word.values() for row in by_qtype.values())
    manifest = {
        "episode_count": len(episodes),
        "eligible_canonical_words": eligible_words,
        "visual_strata": {
            stratum: {
                "query_type_count": len(probe_query_types[stratum]),
                "query_type_keys": sorted(query_type_key(qtype) for qtype in probe_query_types[stratum]),
                "episode_count": sum(counts[stratum].values()),
                "per_function_counts": counts[stratum],
            }
            for stratum in VISUAL_STRATA
        },
        "expected_total": sum(len(probe_query_types[stratum]) * len(eligible_words) * REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION for stratum in VISUAL_STRATA),
        "episodes_per_query_type_function": REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION,
        "passed": all_feasible and len(episodes) == sum(len(probe_query_types[stratum]) * len(eligible_words) * REQUIRED_EPISODES_PER_QUERY_TYPE_FUNCTION for stratum in VISUAL_STRATA),
    }
    feasibility_report = {
        "all_104_frozen_query_types_feasible_for_every_selected_function": all_feasible,
        "visual_query_type_count": sum(len(probe_query_types[stratum]) for stratum in VISUAL_STRATA),
        "required_visual_query_type_count": 104,
        "details": feasibility,
        "passed": all_feasible,
    }
    rejection_report = {
        "scene_rejection_counts": dict(sorted(rejections.items())),
        "total_rejections": sum(rejections.values()),
        "generation_deterministic": True,
    }
    return episodes, manifest, feasibility_report, rejection_report


def _find_counterfactual_base(
    group_index: int,
    qtype: tuple[str, int, str, int],
    eligible_words: list[str],
    rejections: dict[str, int],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[tuple[int, ...], ...], ...]]:
    for attempt in range(4096):
        variant = stable_int(f"r4a1p|cf|{group_index}|{query_type_key(qtype)}|{attempt}") % 10_000_000
        try:
            query = render_scene(qtype, variant, f"r4a1p|counterfactual|query|{group_index}|{attempt}")
        except RuntimeError as exc:
            rejections[f"cf_render:{exc}"] += 1
            continue
        if not all(is_scene_safe_for_word(query, word)[0] for word in eligible_words):
            rejections["cf_query_not_safe_for_all_words"] += 1
            continue
        target_hashes = [output_hash_for_word(query, word) for word in eligible_words]
        if len(set(target_hashes)) != len(target_hashes):
            rejections["cf_targets_collide"] += 1
            continue
        support_inputs: list[tuple[tuple[int, ...], ...]] = []
        support_ok = True
        for pair_index in range(3):
            sqtype = support_query_type(qtype, f"r4a1p|cf|{group_index}|{pair_index}|{attempt}")
            svariant = stable_int(f"r4a1p|cf_support|{group_index}|{pair_index}|{attempt}") % 10_000_000
            try:
                support_input = render_support_scene(sqtype, svariant, pair_index, f"r4a1p|counterfactual|support|{group_index}")
            except RuntimeError as exc:
                rejections[f"cf_support_render:{exc}"] += 1
                support_ok = False
                break
            if support_input == query:
                rejections["cf_support_equals_query"] += 1
                support_ok = False
                break
            if not all(is_scene_safe_for_word(support_input, word)[0] for word in eligible_words):
                rejections["cf_support_not_safe_for_all_words"] += 1
                support_ok = False
                break
            support_inputs.append(support_input)
        if support_ok and len(support_inputs) == 3:
            return query, tuple(support_inputs)
    raise RuntimeError(f"failed to construct counterfactual base for {query_type_key(qtype)}")


def build_length3_counterfactual_suite(
    eligible_words: list[str],
    probe_query_types: dict[str, list[tuple[str, int, str, int]]] | None = None,
    group_count: int = REQUIRED_COUNTERFACTUAL_GROUPS,
) -> tuple[list[R4A0REpisode], dict[str, Any], dict[str, int]]:
    probe_query_types = probe_query_types or default_probe_query_types()
    qtypes = sorted(probe_query_types[TRAIN_VISIBLE_STRATUM] + probe_query_types[SEALED_STRATUM], key=query_type_key)
    rejections: dict[str, int] = defaultdict(int)
    rows: list[R4A0REpisode] = []
    for idx in range(group_count):
        qtype = qtypes[idx % len(qtypes)]
        group_seed = stable_int(f"r4a1p|cf_group|{idx}|{query_type_key(qtype)}") % 10_000_000
        query, support_inputs = _find_counterfactual_base(group_seed, qtype, eligible_words, rejections)
        group_id = f"r4a1p_length3_cf:{idx:06d}:{query_type_key(qtype)}"
        for word in eligible_words:
            supports = tuple(
                R4A0RSupportPair(input=support_input, output=apply_sequence_to_grid(support_input, arrow_word(word)))
                for support_input in support_inputs
            )
            rows.append(
                R4A0REpisode(
                    episode_id=f"{group_id}:{word}",
                    split="LENGTH3_ORDER_COUNTERFACTUAL",
                    sequence_class=arrow_word(word),
                    query_type=qtype,
                    counterfactual_group_id=group_id,
                    query=query,
                    supports=supports,
                    target=apply_sequence_to_grid(query, arrow_word(word)),
                )
            )
    manifest = counterfactual_manifest(rows, eligible_words, group_count)
    return rows, manifest, dict(sorted(rejections.items()))


def counterfactual_manifest(rows: list[R4A0REpisode], eligible_words: list[str], required_groups: int = REQUIRED_COUNTERFACTUAL_GROUPS) -> dict[str, Any]:
    groups: dict[str, list[R4A0REpisode]] = defaultdict(list)
    for episode in rows:
        groups[str(episode.counterfactual_group_id)].append(episode)
    bad: list[str] = []
    for group_id, episodes in groups.items():
        if len(episodes) != len(eligible_words):
            bad.append(group_id)
            continue
        if {compact_word(episode.sequence_class) for episode in episodes} != set(eligible_words):
            bad.append(group_id)
            continue
        if len({stable_hash(grid_to_list(episode.query)) for episode in episodes}) != 1:
            bad.append(group_id)
            continue
        support_inputs = {
            stable_hash([[grid_to_list(pair.input) for pair in episode.supports] for episode in episodes[:1]])
            for episode in episodes
        }
        if len(support_inputs) != 1:
            bad.append(group_id)
            continue
        if len({stable_hash(grid_to_list(episode.target)) for episode in episodes}) != len(eligible_words):
            bad.append(group_id)
    return {
        "group_count": len(groups),
        "episode_count": len(rows),
        "required_group_count": required_groups,
        "eligible_canonical_words": eligible_words,
        "members_per_group": len(eligible_words),
        "bad_group_count": len(bad),
        "bad_group_examples": bad[:20],
        "passed": len(groups) >= required_groups and not bad,
    }


def reorder_episode(episode: R4A0REpisode, order: tuple[int, ...] | list[int]) -> R4A0REpisode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid order {order}")
    return replace(episode, supports=tuple(episode.supports[index] for index in order))


def derange_episode(episode: R4A0REpisode, shift: int) -> R4A0REpisode:
    if shift not in {1, 2}:
        raise ValueError("there are only two non-identity three-pair derangements")
    inputs = [pair.input for pair in episode.supports]
    outputs = [pair.output for pair in episode.supports]
    return replace(episode, supports=tuple(R4A0RSupportPair(input=inputs[index], output=outputs[(index + shift) % 3]) for index in range(3)))


def support_input_bag(episode: R4A0REpisode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.input)) for pair in episode.supports)


def support_output_bag(episode: R4A0REpisode) -> list[str]:
    return sorted(stable_hash(grid_to_list(pair.output)) for pair in episode.supports)


def support_signature(episode: R4A0REpisode) -> list[tuple[str, str]]:
    return [(stable_hash(grid_to_list(pair.input)), stable_hash(grid_to_list(pair.output))) for pair in episode.supports]


def write_probe_artifacts(
    data_out: Path,
    episodes: list[R4A0REpisode],
    counterfactual_rows: list[R4A0REpisode],
    manifest: dict[str, Any],
    feasibility: dict[str, Any],
    rejection_report: dict[str, Any],
    counterfactual_suite_manifest: dict[str, Any],
) -> None:
    data_out.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_out / "length3_probe_dataset.jsonl", episodes)
    write_jsonl(data_out / "length3_order_counterfactual_suite.jsonl", counterfactual_rows)
    for path, payload in (
        (data_out / "length3_dataset_manifest.json", manifest),
        (data_out / "length3_query_type_feasibility.json", feasibility),
        (data_out / "length3_scene_rejection_report.json", rejection_report),
        (data_out / "length3_counterfactual_manifest.json", counterfactual_suite_manifest),
    ):
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_strata_match_partition() -> bool:
    manifest = load_frozen_query_type_manifest()
    partition = partition_query_types()
    return {
        split: sorted(query_type_key(item) for item in manifest[split]) == sorted(query_type_key(item) for item in partition[split])
        for split in ("train", "dev", "sealed")
    } == {"train": True, "dev": True, "sealed": True}


def records_by_group(rows: Iterable[R4A0REpisode]) -> dict[str, list[R4A0REpisode]]:
    grouped: dict[str, list[R4A0REpisode]] = defaultdict(list)
    for row in rows:
        grouped[str(row.counterfactual_group_id)].append(row)
    return dict(grouped)
