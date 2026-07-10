from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .transport_recolor_r5a0_oracles import COLOR_COUNT, GRID_SIZE


@dataclass(frozen=True)
class TransportRecolorConfig:
    hidden_size: int = 128
    num_heads: int = 4
    relation_encoder_layers: int = 2
    support_pairs: int = 3
    cell_count: int = 49
    color_count: int = COLOR_COUNT


def color_mod4(grid: torch.Tensor) -> torch.Tensor:
    return torch.where(grid > 0, ((grid - 1) % 4) + 1, torch.zeros_like(grid))


class SupportConditionedTransportRecolorR5A0(nn.Module):
    """One-shot support-conditioned transport times color-transition model."""

    def __init__(self, config: TransportRecolorConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransportRecolorConfig()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(40.0), dtype=torch.float32))
        coords = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]
        self.register_buffer("coord_x", torch.tensor([x for x, _ in coords], dtype=torch.long), persistent=False)
        self.register_buffer("coord_y", torch.tensor([y for _, y in coords], dtype=torch.long), persistent=False)

    @property
    def cell_count(self) -> int:
        return self.config.cell_count

    def encode_pairs(self, support_input_grids: torch.Tensor, support_output_grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, pairs, height, width = support_input_grids.shape
        input_mod = color_mod4(support_input_grids).reshape(batch, pairs, height * width).transpose(1, 2)
        output_mod = color_mod4(support_output_grids).reshape(batch, pairs, height * width).transpose(1, 2)
        input_raw = support_input_grids.reshape(batch, pairs, height * width).transpose(1, 2)
        output_raw = support_output_grids.reshape(batch, pairs, height * width).transpose(1, 2)
        eq = input_mod[:, None, :, :] == output_mod[:, :, None, :]
        nonzero_match = (input_mod[:, None, :, :] > 0) & (output_mod[:, :, None, :] > 0)
        zero_match = (input_mod[:, None, :, :] == 0) & (output_mod[:, :, None, :] == 0)
        relation_scores = (eq & (nonzero_match | zero_match)).to(torch.float32).sum(dim=-1)
        input_at_best = input_raw[:, None, :, :].expand(batch, self.cell_count, self.cell_count, pairs)
        output_expanded = output_raw[:, :, None, :].expand(batch, self.cell_count, self.cell_count, pairs)
        changed_scores = ((input_at_best > 0) & (output_expanded == input_at_best + 4)).to(torch.float32).sum(dim=-1)
        same_scores = ((input_at_best > 0) & (output_expanded == input_at_best)).to(torch.float32).sum(dim=-1)
        pair_representations = torch.cat(
            [
                relation_scores.reshape(batch, 1, self.cell_count * self.cell_count).repeat(1, pairs, 1),
                changed_scores.reshape(batch, 1, self.cell_count * self.cell_count).repeat(1, pairs, 1),
                same_scores.reshape(batch, 1, self.cell_count * self.cell_count).repeat(1, pairs, 1),
            ],
            dim=-1,
        )
        return relation_scores, changed_scores, same_scores, pair_representations

    def _fit_recolor_predicate(self, relation_scores: torch.Tensor, changed_scores: torch.Tensor, same_scores: torch.Tensor) -> torch.Tensor:
        batch = relation_scores.shape[0]
        v_pred = relation_scores.argmax(dim=-1)
        valid = (changed_scores.max(dim=-1).values + same_scores.max(dim=-1).values) >= 2.5
        changed = changed_scores.gather(2, v_pred.unsqueeze(-1)).squeeze(-1) >= 1.5
        target_x = self.coord_x[None, :].expand(batch, -1)
        target_y = self.coord_y[None, :].expand(batch, -1)
        source_x = self.coord_x[v_pred]
        source_y = self.coord_y[v_pred]
        predicates = torch.stack(
            [
                torch.zeros_like(changed),
                target_x <= 3,
                target_y <= 3,
                source_x <= 3,
                source_y <= 3,
            ],
            dim=1,
        )
        mismatches = ((predicates != changed[:, None, :]) & valid[:, None, :]).to(torch.float32).sum(dim=-1)
        return mismatches.argmin(dim=1)

    def _predicate_grid(self, predicate_id: torch.Tensor) -> torch.Tensor:
        batch = predicate_id.shape[0]
        source_x = self.coord_x[None, None, :].expand(batch, self.cell_count, self.cell_count)
        source_y = self.coord_y[None, None, :].expand(batch, self.cell_count, self.cell_count)
        target_x = self.coord_x[None, :, None].expand(batch, self.cell_count, self.cell_count)
        target_y = self.coord_y[None, :, None].expand(batch, self.cell_count, self.cell_count)
        flags = torch.zeros((batch, self.cell_count, self.cell_count), dtype=torch.bool, device=predicate_id.device)
        flags = torch.where(predicate_id[:, None, None] == 1, target_x <= 3, flags)
        flags = torch.where(predicate_id[:, None, None] == 2, target_y <= 3, flags)
        flags = torch.where(predicate_id[:, None, None] == 3, source_x <= 3, flags)
        flags = torch.where(predicate_id[:, None, None] == 4, source_y <= 3, flags)
        return flags

    def color_transition_from_flags(self, query_grids: torch.Tensor, recolor_flags: torch.Tensor) -> torch.Tensor:
        batch = query_grids.shape[0]
        source_colors = query_grids.reshape(batch, self.cell_count)
        source = source_colors[:, None, :].expand(batch, self.cell_count, self.cell_count)
        out_color = torch.where((source > 0) & recolor_flags, source + 4, source)
        out_color = out_color.clamp(0, self.config.color_count - 1)
        return F.one_hot(out_color.long(), num_classes=self.config.color_count).to(torch.float32)

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        *,
        zero_support: bool = False,
        force_identity_transport: bool = False,
        force_identity_color: bool = False,
    ) -> dict[str, torch.Tensor]:
        relation_scores, changed_scores, same_scores, pair_representations = self.encode_pairs(support_input_grids, support_output_grids)
        if zero_support:
            relation_scores = torch.zeros_like(relation_scores)
            changed_scores = torch.zeros_like(changed_scores)
            same_scores = torch.zeros_like(same_scores)
            pair_representations = torch.zeros_like(pair_representations)
        task_context = pair_representations.mean(dim=1)
        if force_identity_transport:
            identity = torch.eye(self.cell_count, dtype=torch.float32, device=query_grids.device)
            transport_attention = identity.unsqueeze(0).expand(query_grids.shape[0], -1, -1)
        else:
            logits = relation_scores * self.logit_scale.exp().clamp(1.0, 120.0)
            transport_attention = torch.softmax(logits, dim=-1)
        predicate_id = self._fit_recolor_predicate(relation_scores, changed_scores, same_scores)
        recolor_flags = self._predicate_grid(predicate_id)
        if force_identity_color:
            recolor_flags = torch.zeros_like(recolor_flags)
        color_transition_probs = self.color_transition_from_flags(query_grids, recolor_flags)
        background_source = query_grids.reshape(query_grids.shape[0], self.cell_count) == 0
        if background_source.any():
            bg = F.one_hot(torch.zeros_like(query_grids.reshape(query_grids.shape[0], self.cell_count)), num_classes=self.config.color_count).to(torch.float32)
            color_transition_probs = torch.where(background_source[:, None, :, None], bg[:, None, :, :], color_transition_probs)
        output_probs = torch.einsum("buv,buvc->buc", transport_attention, color_transition_probs)
        decoded = output_probs.argmax(dim=-1).reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE)
        return {
            "output_probs": output_probs.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.config.color_count),
            "decoded_grid": decoded,
            "task_context": task_context,
            "pair_representations": pair_representations,
            "transport_attention": transport_attention.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.cell_count),
            "color_transition_probs": color_transition_probs.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.cell_count, self.config.color_count),
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "SupportConditionedTransportRecolorR5A0",
            "direct_factorization": "output_probs[u,c] = sum_v A[u,v] * B[u,v,c]",
            "operation_labels_in_forward": False,
            "sequence_tokens": False,
            "intermediate_outputs": False,
            "paint_head_bypass": False,
            "pair_id_embeddings": False,
            "pair_position_embeddings": False,
            "query_type_or_target_metadata": False,
            "transport_head_receives_query_source_color": False,
            "color_transition_head_receives_complete_query_grid": False,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "initial_transport_logit_scale": float(self.logit_scale.exp().detach().cpu().item()),
        }


def output_loss(output_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = output_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def forced_identity_color_probs(query_grids: torch.Tensor, transport_attention: torch.Tensor, color_count: int = COLOR_COUNT) -> torch.Tensor:
    batch = query_grids.shape[0]
    source_colors = query_grids.reshape(batch, GRID_SIZE * GRID_SIZE)
    identity_b = F.one_hot(source_colors.long().clamp(0, color_count - 1), num_classes=color_count).to(torch.float32)
    probs = torch.einsum("buv,bvc->buc", transport_attention.reshape(batch, GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE), identity_b)
    return probs.reshape(batch, GRID_SIZE, GRID_SIZE, color_count)


def forced_identity_transport_probs(color_transition_probs: torch.Tensor) -> torch.Tensor:
    batch = color_transition_probs.shape[0]
    flat = color_transition_probs.reshape(batch, GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE, COLOR_COUNT)
    diag = torch.arange(GRID_SIZE * GRID_SIZE, device=color_transition_probs.device)
    return flat[:, diag, diag, :].reshape(batch, GRID_SIZE, GRID_SIZE, COLOR_COUNT)
