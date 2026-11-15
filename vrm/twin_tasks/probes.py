from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .dataset import grid_hash

PROBE_NAMES = (
    "P1_QUERY_ONLY",
    "P2_SUPPORT_INPUT_BAG_ONLY",
    "P3_SUPPORT_OUTPUT_BAG_ONLY",
    "P4_UNPAIRED_SUPPORT_BAG",
    "P5_RELATION_AWARE_PAIRED",
    "P6_RELATION_AWARE_DERANGED",
)

PROBE_VIEW = {
    "P1_QUERY_ONLY": "query + candidates",
    "P2_SUPPORT_INPUT_BAG_ONLY": "query + candidates + canonical support input bag",
    "P3_SUPPORT_OUTPUT_BAG_ONLY": "query + candidates + canonical support output bag",
    "P4_UNPAIRED_SUPPORT_BAG": "query + candidates + canonical input/output bags",
    "P5_RELATION_AWARE_PAIRED": "query + candidates + coherent support pairs",
    "P6_RELATION_AWARE_DERANGED": "query + candidates + deranged support pairs",
}

EXPECTED = {
    "P1_QUERY_ONLY": "chance",
    "P2_SUPPORT_INPUT_BAG_ONLY": "chance",
    "P3_SUPPORT_OUTPUT_BAG_ONLY": "chance",
    "P4_UNPAIRED_SUPPORT_BAG": "chance",
    "P5_RELATION_AWARE_PAIRED": "above_chance",
    "P6_RELATION_AWARE_DERANGED": "deranged_control_le_55",
}


class PositiveScaleProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.raw_scale)
        return scores * scale


def grids_equal(left: list[list[int]], right: list[list[int]]) -> bool:
    return left == right


def deranged_supports(record: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    supports = list(record["supports"])
    if len(supports) < 2:
        return supports
    rng = random.Random(int(grid_hash(record["query"])[:12], 16) + seed)
    outputs = [support["output"] for support in supports]
    order = list(range(len(outputs)))
    for _ in range(100):
        rng.shuffle(order)
        if all(order[idx] != idx for idx in range(len(order))):
            break
    else:
        order = order[1:] + order[:1]
    return [
        {**support, "output": outputs[order[idx]], "deranged_output_from": order[idx]}
        for idx, support in enumerate(supports)
    ]


def relation_scores(record: dict[str, Any], *, deranged: bool, seed: int) -> list[float]:
    supports = deranged_supports(record, seed) if deranged else record["supports"]
    scores = []
    for candidate in record["candidates"]:
        score = 0.0
        for support in supports:
            if grids_equal(support["input"], record["query"]) and grids_equal(support["output"], candidate):
                score += 1.0
        scores.append(score)
    return scores


def output_bag_scores(record: dict[str, Any]) -> list[float]:
    output_hashes = {grid_hash(support["output"]) for support in record["supports"]}
    return [1.0 if grid_hash(candidate) in output_hashes else 0.0 for candidate in record["candidates"]]


def feature_scores(record: dict[str, Any], probe_name: str, seed: int) -> list[float]:
    if probe_name in {"P1_QUERY_ONLY", "P2_SUPPORT_INPUT_BAG_ONLY"}:
        return [0.0, 0.0]
    if probe_name in {"P3_SUPPORT_OUTPUT_BAG_ONLY", "P4_UNPAIRED_SUPPORT_BAG"}:
        return output_bag_scores(record)
    if probe_name == "P5_RELATION_AWARE_PAIRED":
        return relation_scores(record, deranged=False, seed=seed)
    if probe_name == "P6_RELATION_AWARE_DERANGED":
        return relation_scores(record, deranged=True, seed=seed)
    raise ValueError(f"unknown probe {probe_name}")


def tensors_for_probe(records: list[dict[str, Any]], probe_name: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.tensor([feature_scores(record, probe_name, seed) for record in records], dtype=torch.float32)
    labels = torch.tensor([record["label"] for record in records], dtype=torch.long)
    return scores, labels


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def train_probe(
    *,
    probe_name: str,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    sealed_records: list[dict[str, Any]],
    seed: int,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    patience: int = 5,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = PositiveScaleProbe()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_scores, train_labels = tensors_for_probe(train_records, probe_name, seed)
    dev_scores, dev_labels = tensors_for_probe(dev_records, probe_name, seed)
    sealed_scores, sealed_labels = tensors_for_probe(sealed_records, probe_name, seed)
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_dev = -1.0
    bad_epochs = 0
    logs = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(train_scores.shape[0], generator=generator)
        losses = []
        for start in range(0, order.numel(), batch_size):
            idx = order[start : start + batch_size]
            logits = model(train_scores[idx])
            loss = F.cross_entropy(logits, train_labels[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        with torch.no_grad():
            train_acc = accuracy_from_logits(model(train_scores), train_labels)
            dev_acc = accuracy_from_logits(model(dev_scores), dev_labels)
        logs.append(
            {
                "probe": probe_name,
                "seed": seed,
                "epoch": epoch,
                "mean_train_loss": statistics.fmean(losses),
                "train_accuracy": train_acc,
                "dev_accuracy": dev_acc,
            }
        )
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    with torch.no_grad():
        dev_logits = model(dev_scores)
        sealed_logits = model(sealed_scores)
    return {
        "probe": probe_name,
        "seed": seed,
        "parameter_count": model.parameter_count,
        "best_dev_accuracy": accuracy_from_logits(dev_logits, dev_labels),
        "sealed_accuracy": accuracy_from_logits(sealed_logits, sealed_labels),
        "dev_predictions": dev_logits.argmax(dim=-1).tolist(),
        "sealed_predictions": sealed_logits.argmax(dim=-1).tolist(),
        "logs": logs,
    }


def wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [center - spread, center + spread]


def summarize_probe_runs(runs: list[dict[str, Any]], *, split_name: str, example_count: int) -> dict[str, Any]:
    if split_name == "dev":
        accuracy_key = "best_dev_accuracy"
    elif split_name == "sealed":
        accuracy_key = "sealed_accuracy"
    else:
        accuracy_key = f"{split_name}_accuracy"
    accuracies = [run[accuracy_key] for run in runs]
    successes = round(sum(accuracies) * example_count)
    total = example_count * len(runs)
    ci = wilson_ci(successes, total)
    return {
        "probe": runs[0]["probe"],
        "input_view": PROBE_VIEW[runs[0]["probe"]],
        "expected_result": EXPECTED[runs[0]["probe"]],
        "parameter_count": runs[0]["parameter_count"],
        f"{split_name}_accuracies": accuracies,
        f"{split_name}_mean_accuracy": statistics.fmean(accuracies),
        f"{split_name}_std_accuracy": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        f"{split_name}_binomial_ci95": ci,
    }


def pass_fail_for_probe(probe_name: str, sealed_mean: float, ci95: list[float]) -> bool:
    if probe_name == "P5_RELATION_AWARE_PAIRED":
        return sealed_mean >= 0.90
    if probe_name == "P6_RELATION_AWARE_DERANGED":
        return sealed_mean <= 0.55
    if probe_name in {
        "P1_QUERY_ONLY",
        "P2_SUPPORT_INPUT_BAG_ONLY",
        "P3_SUPPORT_OUTPUT_BAG_ONLY",
        "P4_UNPAIRED_SUPPORT_BAG",
    }:
        return sealed_mean <= 0.55 and ci95[0] <= 0.50 <= ci95[1]
    raise ValueError(probe_name)
