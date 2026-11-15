from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vrm.data import ArcEpisode, ArcPair, Grid, collate_episodes, model_inputs_from_batch

SPLITS = ("train", "dev", "sealed")
CONDITION_NORMAL = "NORMAL"
CONDITION_TWIN_SWAP_SUPPORTS = "TWIN_SWAP_SUPPORTS"
CONDITION_PAIR_ORDER_SHUFFLED = "PAIR_ORDER_SHUFFLED"
CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED = "SUPPORT_EVIDENCE_PATH_ZEROED"
EVAL_CONDITIONS = (
    CONDITION_NORMAL,
    CONDITION_TWIN_SWAP_SUPPORTS,
    CONDITION_PAIR_ORDER_SHUFFLED,
    CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED,
)
FORBIDDEN_MODEL_INPUT_KEYS = {
    "candidate_0",
    "candidate_1",
    "candidates",
    "label",
    "target",
    "target_output",
    "twin",
    "base_id",
    "rule_id",
    "template_id",
    "pair_position",
    "metadata",
    "query_targets",
    "height_targets",
    "width_targets",
}


@dataclass(frozen=True)
class TwinEpisode:
    record_index: int
    split: str
    base_id: str
    twin: str
    family: str
    label: int
    supports: tuple[ArcPair, ...]
    query_input: Grid
    target_output: Grid
    candidates: tuple[Grid, Grid]

    @property
    def episode_id(self) -> str:
        return f"{self.split}:{self.base_id}:{self.twin}"

    @property
    def alternate_label(self) -> int:
        return 1 - self.label

    def to_arc_episode(self) -> ArcEpisode:
        return ArcEpisode(
            task_id=self.episode_id,
            source_split=self.split,
            supports=self.supports,
            query=ArcPair(input=self.query_input, output=self.target_output, pair_index=0),
            query_pair_index=0,
            source_file="twin_task_pilot_v1",
        )


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_twin_records(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: read_jsonl(Path(data_dir) / f"{split}.jsonl") for split in SPLITS}


def _grid(raw: list[list[int]]) -> Grid:
    return Grid.from_raw(raw)


def _support_pair_from_record(support: dict[str, Any], pair_index: int) -> ArcPair:
    return ArcPair(
        input=_grid(support["input"]),
        output=_grid(support["output"]),
        pair_index=pair_index,
    )


def record_to_episode(record: dict[str, Any], record_index: int = 0) -> TwinEpisode:
    supports = tuple(
        _support_pair_from_record(support, pair_index=idx)
        for idx, support in enumerate(record["supports"])
    )
    candidates = (_grid(record["candidates"][0]), _grid(record["candidates"][1]))
    return TwinEpisode(
        record_index=record_index,
        split=str(record["split"]),
        base_id=str(record["base_id"]),
        twin=str(record["twin"]),
        family=str(record["family"]),
        label=int(record["label"]),
        supports=supports,
        query_input=_grid(record["query"]),
        target_output=_grid(record["target"]),
        candidates=candidates,
    )


def records_to_episodes(records: list[dict[str, Any]]) -> list[TwinEpisode]:
    return [record_to_episode(record, idx) for idx, record in enumerate(records)]


def load_twin_episodes(data_dir: Path) -> dict[str, list[TwinEpisode]]:
    return {split: records_to_episodes(rows) for split, rows in load_twin_records(data_dir).items()}


def group_by_base(episodes: list[TwinEpisode]) -> dict[str, dict[str, TwinEpisode]]:
    grouped: dict[str, dict[str, TwinEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.base_id, {})[episode.twin] = episode
    return grouped


def deterministic_permutation(
    length: int,
    *,
    seed: int,
    episode_id: str,
    step: int,
    salt: str,
    force_non_identity: bool = False,
) -> list[int]:
    if length < 1:
        raise ValueError("cannot permute an empty support list")
    order = list(range(length))
    rng = random.Random(stable_int(f"{seed}|{episode_id}|{step}|{salt}"))
    rng.shuffle(order)
    if force_non_identity and length > 1 and order == list(range(length)):
        rotate = 1 + stable_int(f"{seed}|{episode_id}|{step}|{salt}|rotate") % (length - 1)
        order = order[rotate:] + order[:rotate]
    return order


def reorder_supports(episode: TwinEpisode, order: list[int]) -> TwinEpisode:
    if sorted(order) != list(range(len(episode.supports))):
        raise ValueError(f"invalid support permutation {order}")
    reordered = tuple(
        ArcPair(
            input=episode.supports[source_idx].input,
            output=episode.supports[source_idx].output,
            pair_index=new_idx,
        )
        for new_idx, source_idx in enumerate(order)
    )
    return replace(episode, supports=reordered)


def training_view(episode: TwinEpisode, *, seed: int, step: int) -> TwinEpisode:
    order = deterministic_permutation(
        len(episode.supports),
        seed=seed,
        episode_id=episode.episode_id,
        step=step,
        salt="train_pair_order",
    )
    return reorder_supports(episode, order)


def pair_order_shuffled_view(episode: TwinEpisode, *, seed: int) -> TwinEpisode:
    order = deterministic_permutation(
        len(episode.supports),
        seed=seed,
        episode_id=episode.episode_id,
        step=0,
        salt="eval_pair_order_shuffled",
        force_non_identity=True,
    )
    return reorder_supports(episode, order)


def twin_swap_supports_view(episode: TwinEpisode, opposite: TwinEpisode) -> TwinEpisode:
    swapped_supports = tuple(
        ArcPair(input=pair.input, output=pair.output, pair_index=idx)
        for idx, pair in enumerate(opposite.supports)
    )
    return replace(episode, supports=swapped_supports)


def condition_view(
    episode: TwinEpisode,
    *,
    condition: str,
    opposite_by_episode: dict[str, TwinEpisode] | None = None,
    seed: int,
) -> TwinEpisode:
    if condition in {CONDITION_NORMAL, CONDITION_SUPPORT_EVIDENCE_PATH_ZEROED}:
        return episode
    if condition == CONDITION_PAIR_ORDER_SHUFFLED:
        return pair_order_shuffled_view(episode, seed=seed)
    if condition == CONDITION_TWIN_SWAP_SUPPORTS:
        key = f"{episode.base_id}:{episode.twin}"
        if opposite_by_episode is None or key not in opposite_by_episode:
            raise KeyError(f"missing opposite twin for {episode.episode_id}")
        return twin_swap_supports_view(episode, opposite_by_episode[key])
    raise ValueError(f"unknown condition {condition}")


def opposite_episode_lookup(episodes: list[TwinEpisode]) -> dict[str, TwinEpisode]:
    grouped = group_by_base(episodes)
    lookup: dict[str, TwinEpisode] = {}
    for base_id, twins in grouped.items():
        if set(twins) != {"A", "B"}:
            raise ValueError(f"base {base_id} does not have exactly A/B twins")
        lookup[f"{base_id}:A"] = twins["B"]
        lookup[f"{base_id}:B"] = twins["A"]
    return {f"{episode.base_id}:{episode.twin}": lookup[f"{episode.base_id}:{episode.twin}"] for episode in episodes}


def opposite_by_base_and_twin(episodes: list[TwinEpisode]) -> dict[str, TwinEpisode]:
    return {f"{episode.base_id}:{episode.twin}": opp for episode, opp in zip(episodes, opposite_episode_lookup(episodes).values())}


def build_opposite_lookup(episodes: list[TwinEpisode]) -> dict[str, TwinEpisode]:
    grouped = group_by_base(episodes)
    lookup: dict[str, TwinEpisode] = {}
    for base_id, twins in grouped.items():
        if set(twins) != {"A", "B"}:
            raise ValueError(f"base {base_id} does not have exactly A/B twins")
        lookup[f"{base_id}:A"] = twins["B"]
        lookup[f"{base_id}:B"] = twins["A"]
    return lookup


def collate_twin_episodes(episodes: list[TwinEpisode] | tuple[TwinEpisode, ...]) -> dict[str, Any]:
    return collate_episodes([episode.to_arc_episode() for episode in episodes])


def model_inputs_for_twin_batch(batch: dict[str, Any]) -> dict[str, Any]:
    inputs = model_inputs_from_batch(batch)
    forbidden = sorted(FORBIDDEN_MODEL_INPUT_KEYS.intersection(inputs))
    if forbidden:
        raise RuntimeError(f"forbidden fields entered model input: {forbidden}")
    return inputs


def support_signature(episode: TwinEpisode) -> list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    return [(pair.input.cells, pair.output.cells) for pair in episode.supports]


def support_input_bag_signature(episode: TwinEpisode) -> list[str]:
    return sorted(hashlib.sha256(json.dumps(pair.input.to_list(), sort_keys=True).encode()).hexdigest() for pair in episode.supports)


def support_output_bag_signature(episode: TwinEpisode) -> list[str]:
    return sorted(hashlib.sha256(json.dumps(pair.output.to_list(), sort_keys=True).encode()).hexdigest() for pair in episode.supports)


def data_manifest(data_dir: Path, episodes_by_split: dict[str, list[TwinEpisode]]) -> dict[str, Any]:
    return {
        "data_dir": str(data_dir),
        "source_files": {split: str(Path(data_dir) / f"{split}.jsonl") for split in SPLITS},
        "episode_counts": {split: len(rows) for split, rows in episodes_by_split.items()},
        "base_pair_counts": {split: len(group_by_base(rows)) for split, rows in episodes_by_split.items()},
        "model_input_contract": {
            "included": sorted(model_inputs_for_twin_batch(collate_twin_episodes(episodes_by_split["train"][:1])).keys()),
            "excluded": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
            "candidate_metadata_eval_only": True,
            "target_output_used_only_for_supervised_loss": True,
        },
    }
