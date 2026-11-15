from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .data import ARC_COLOR_COUNT, MAX_GRID_HEIGHT, MAX_GRID_TOKENS, MAX_GRID_WIDTH, PAD_ID, VOCAB_SIZE


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_weight: float = 0.2,
) -> torch.Tensor:
    weights = torch.ones(VOCAB_SIZE, dtype=logits.dtype, device=logits.device)
    weights[PAD_ID] = pad_weight
    return F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        weight=weights,
    )


def factorized_shape_color_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    shape_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    color_log_probs = outputs["color_log_probs"]
    targets = batch["query_targets"]
    target_mask = batch["query_target_mask"]
    if color_log_probs.shape[-1] != ARC_COLOR_COUNT:
        raise ValueError(f"V1.1 color output must have {ARC_COLOR_COUNT} classes")
    if target_mask.shape != targets.shape:
        raise ValueError("query_target_mask must match query_targets shape")
    if target_mask.dtype != torch.bool:
        raise TypeError("query_target_mask must be boolean")
    if not bool(target_mask.any(dim=1).all().item()):
        raise ValueError("each target must have at least one valid color cell")

    valid_targets = targets[target_mask]
    if bool((valid_targets >= ARC_COLOR_COUNT).any().item()):
        raise ValueError("target rectangle contains non-color PAD values")
    color_loss = F.nll_loss(
        color_log_probs[target_mask].reshape(-1, ARC_COLOR_COUNT),
        valid_targets.reshape(-1),
    )
    height_loss = F.cross_entropy(outputs["height_logits"], batch["height_targets"])
    width_loss = F.cross_entropy(outputs["width_logits"], batch["width_targets"])
    total_loss = color_loss + shape_loss_weight * 0.5 * (height_loss + width_loss)
    components = {
        "total_loss": total_loss,
        "color_loss": color_loss,
        "height_loss": height_loss,
        "width_loss": width_loss,
    }
    return total_loss, components


def decode_rectangular_padding(tokens: torch.Tensor) -> dict[str, Any]:
    flat = tokens.detach().cpu().to(torch.long)
    if flat.numel() != MAX_GRID_TOKENS:
        raise ValueError(f"expected {MAX_GRID_TOKENS} tokens")
    grid = flat.view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)
    mask = grid != PAD_ID
    if not bool(mask.any()):
        return {"valid": False, "height": 0, "width": 0, "grid": None}
    row_indices = torch.where(mask.any(dim=1))[0]
    col_indices = torch.where(mask.any(dim=0))[0]
    height = int(row_indices.max().item()) + 1
    width = int(col_indices.max().item()) + 1
    expected = torch.zeros_like(mask)
    expected[:height, :width] = True
    valid = bool(torch.equal(mask, expected))
    decoded = grid[:height, :width].clone() if valid else None
    return {"valid": valid, "height": height, "width": width, "grid": decoded}


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    query_output_heights: torch.Tensor,
    query_output_widths: torch.Tensor,
) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    all_matches = predictions == targets
    nonpad_mask = targets != PAD_ID
    token_accuracy_all = all_matches.float().mean().item()
    token_accuracy_nonpad = (
        all_matches[nonpad_mask].float().mean().item() if bool(nonpad_mask.any()) else 0.0
    )
    exact_padded = all_matches.all(dim=1).float().mean().item()

    invalid = 0
    shape_matches = 0
    exact_decoded = 0
    batch_size = targets.shape[0]
    for idx in range(batch_size):
        decoded = decode_rectangular_padding(predictions[idx])
        target_h = int(query_output_heights[idx].detach().cpu().item())
        target_w = int(query_output_widths[idx].detach().cpu().item())
        if not decoded["valid"]:
            invalid += 1
            continue
        if decoded["height"] == target_h and decoded["width"] == target_w:
            shape_matches += 1
            pred_grid = decoded["grid"]
            target_grid = targets[idx].detach().cpu().view(MAX_GRID_HEIGHT, MAX_GRID_WIDTH)[
                :target_h, :target_w
            ]
            if bool(torch.equal(pred_grid, target_grid)):
                exact_decoded += 1

    return {
        "token_accuracy_all": token_accuracy_all,
        "token_accuracy_nonpad": token_accuracy_nonpad,
        "exact_padded_grid_accuracy": exact_padded,
        "exact_decoded_grid_accuracy": exact_decoded / batch_size,
        "output_shape_match_rate": shape_matches / batch_size,
        "invalid_padding_pattern_rate": invalid / batch_size,
    }
