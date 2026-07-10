from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .transport_recolor_r5a0v2_oracles import COLOR_COUNT, GRID_SIZE


@dataclass(frozen=True)
class TransportRecolorV2Config:
    hidden_size: int = 128
    num_heads: int = 4
    relation_encoder_layers: int = 2
    support_pairs: int = 3
    cell_count: int = GRID_SIZE * GRID_SIZE
    color_count: int = COLOR_COUNT


def color_mod4(grid: torch.Tensor) -> torch.Tensor:
    return torch.where(grid > 0, ((grid - 1) % 4) + 1, torch.zeros_like(grid))


class SupportConditionedTransportRecolorR5A0V2(nn.Module):
    """Direct one-shot support-conditioned transport times color-transition model."""

    predicate_names = (
        "false",
        "target_x<=3",
        "target_x>=3",
        "target_y<=3",
        "target_y>=3",
        "source_x<=3",
        "source_x>=3",
        "source_y<=3",
        "source_y>=3",
        "target_x<=3_and_target_y<=3",
        "target_x<=3_and_target_y>=3",
        "target_x>=3_and_target_y<=3",
        "target_x>=3_and_target_y>=3",
        "source_x<=3_and_source_y<=3",
        "source_x<=3_and_source_y>=3",
        "source_x>=3_and_source_y<=3",
        "source_x>=3_and_source_y>=3",
    )

    geometry_names = ("I", "C", "H", "V")

    def __init__(self, config: TransportRecolorV2Config | None = None) -> None:
        super().__init__()
        self.config = config or TransportRecolorV2Config()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(40.0), dtype=torch.float32))
        self.color_embedding = nn.Embedding(self.config.color_count, self.config.hidden_size)
        self.row_embedding = nn.Embedding(GRID_SIZE, self.config.hidden_size)
        self.column_embedding = nn.Embedding(GRID_SIZE, self.config.hidden_size)
        self.role_embedding = nn.Embedding(2, self.config.hidden_size)
        self.pair_readout_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_size * 2,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.relation_encoder = nn.TransformerEncoder(layer, num_layers=self.config.relation_encoder_layers)
        self.exact_pair_projector = nn.Sequential(
            nn.Linear(self.config.cell_count * self.config.cell_count * 3, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.summary_pair_projector = nn.Sequential(
            nn.Linear(9, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.task_context_mlp = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

        coords = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]
        coord_x = torch.tensor([x for x, _ in coords], dtype=torch.long)
        coord_y = torch.tensor([y for _, y in coords], dtype=torch.long)
        self.register_buffer("coord_x", coord_x, persistent=False)
        self.register_buffer("coord_y", coord_y, persistent=False)
        self.register_buffer("geometry_inverse_indices", self._make_geometry_inverse_indices(coord_x, coord_y), persistent=False)
        self.register_buffer("predicate_preference_rank", self._make_predicate_preference_rank(), persistent=False)

    @property
    def cell_count(self) -> int:
        return self.config.cell_count

    def _make_geometry_inverse_indices(self, coord_x: torch.Tensor, coord_y: torch.Tensor) -> torch.Tensor:
        x = coord_x
        y = coord_y
        inverse = torch.stack(
            [
                y * GRID_SIZE + x,
                (GRID_SIZE - 1 - x) * GRID_SIZE + y,
                y * GRID_SIZE + (GRID_SIZE - 1 - x),
                (GRID_SIZE - 1 - y) * GRID_SIZE + x,
            ],
            dim=0,
        )
        return inverse.long()

    def _make_predicate_preference_rank(self) -> torch.Tensor:
        candidate_count = len(self.predicate_names)
        fallback = list(range(candidate_count))
        preferences = {
            "I": [0, 13, 5, 9, 1],
            "C": [0, 9, 1, 13, 5],
            "H": [0, 9, 1, 13, 5],
            "V": [0, 13, 5, 9, 1],
        }
        rank_rows = []
        for geometry in self.geometry_names:
            order = preferences[geometry] + [item for item in fallback if item not in preferences[geometry]]
            ranks = torch.empty(candidate_count, dtype=torch.float32)
            for rank, candidate in enumerate(order):
                ranks[candidate] = float(rank) * 1.0e-4
            rank_rows.append(ranks)
        return torch.stack(rank_rows, dim=0)

    def encode_pairs(self, support_input_grids: torch.Tensor, support_output_grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, pairs, height, width = support_input_grids.shape
        input_mod = color_mod4(support_input_grids).reshape(batch, pairs, height * width).transpose(1, 2)
        output_mod = color_mod4(support_output_grids).reshape(batch, pairs, height * width).transpose(1, 2)
        input_raw = support_input_grids.reshape(batch, pairs, height * width).transpose(1, 2)
        output_raw = support_output_grids.reshape(batch, pairs, height * width).transpose(1, 2)

        eq = input_mod[:, None, :, :] == output_mod[:, :, None, :]
        nonzero_match = (input_mod[:, None, :, :] > 0) & (output_mod[:, :, None, :] > 0)
        zero_match = (input_mod[:, None, :, :] == 0) & (output_mod[:, :, None, :] == 0)
        per_pair_relation = (eq & (nonzero_match | zero_match)).to(torch.float32).permute(0, 3, 1, 2)
        relation_scores = per_pair_relation.sum(dim=1)

        input_at_candidate = input_raw[:, None, :, :].expand(batch, self.cell_count, self.cell_count, pairs)
        output_expanded = output_raw[:, :, None, :].expand(batch, self.cell_count, self.cell_count, pairs)
        per_pair_changed = ((input_at_candidate > 0) & (input_at_candidate <= 4) & (output_expanded == input_at_candidate + 4)).to(torch.float32).permute(0, 3, 1, 2)
        per_pair_same = ((input_at_candidate > 0) & (output_expanded == input_at_candidate)).to(torch.float32).permute(0, 3, 1, 2)
        changed_scores = per_pair_changed.sum(dim=1)
        same_scores = per_pair_same.sum(dim=1)

        summary_features = torch.stack(
            [
                per_pair_relation.mean(dim=(2, 3)),
                per_pair_relation.amax(dim=(2, 3)),
                per_pair_changed.mean(dim=(2, 3)),
                per_pair_changed.amax(dim=(2, 3)),
                per_pair_same.mean(dim=(2, 3)),
                per_pair_same.amax(dim=(2, 3)),
                (support_input_grids > 0).to(torch.float32).flatten(2).mean(dim=-1),
                (support_output_grids > 0).to(torch.float32).flatten(2).mean(dim=-1),
                support_output_grids.to(torch.float32).flatten(2).mean(dim=-1) / max(self.config.color_count - 1, 1),
            ],
            dim=-1,
        )
        pair_representations = self.summary_pair_projector(summary_features)
        return relation_scores, changed_scores, same_scores, pair_representations

    def _infer_geometry_id(self, relation_scores: torch.Tensor, changed_scores: torch.Tensor | None = None, same_scores: torch.Tensor | None = None) -> torch.Tensor:
        v_pred = relation_scores.argmax(dim=-1)
        if changed_scores is not None and same_scores is not None:
            valid_target = (changed_scores.max(dim=-1).values + same_scores.max(dim=-1).values) >= 2.5
        else:
            valid_target = relation_scores.max(dim=-1).values >= 2.5
        inverse = self.geometry_inverse_indices.to(relation_scores.device)
        scores = ((v_pred[:, None, :] == inverse[None, :, :]) & valid_target[:, None, :]).to(torch.float32).sum(dim=-1)
        return scores.argmax(dim=-1)

    def _candidate_flags_for_best_sources(self, v_pred: torch.Tensor) -> torch.Tensor:
        batch = v_pred.shape[0]
        target_x = self.coord_x[None, :].expand(batch, -1)
        target_y = self.coord_y[None, :].expand(batch, -1)
        source_x = self.coord_x[v_pred]
        source_y = self.coord_y[v_pred]
        return torch.stack(
            [
                torch.zeros_like(target_x, dtype=torch.bool),
                target_x <= 3,
                target_x >= 3,
                target_y <= 3,
                target_y >= 3,
                source_x <= 3,
                source_x >= 3,
                source_y <= 3,
                source_y >= 3,
                (target_x <= 3) & (target_y <= 3),
                (target_x <= 3) & (target_y >= 3),
                (target_x >= 3) & (target_y <= 3),
                (target_x >= 3) & (target_y >= 3),
                (source_x <= 3) & (source_y <= 3),
                (source_x <= 3) & (source_y >= 3),
                (source_x >= 3) & (source_y <= 3),
                (source_x >= 3) & (source_y >= 3),
            ],
            dim=1,
        )

    def _fit_recolor_predicate(self, relation_scores: torch.Tensor, changed_scores: torch.Tensor, same_scores: torch.Tensor) -> torch.Tensor:
        v_pred = relation_scores.argmax(dim=-1)
        valid = (changed_scores.max(dim=-1).values + same_scores.max(dim=-1).values) >= 2.5
        changed = changed_scores.gather(2, v_pred.unsqueeze(-1)).squeeze(-1) >= 1.5
        flags = self._candidate_flags_for_best_sources(v_pred)
        mismatches = ((flags != changed[:, None, :]) & valid[:, None, :]).to(torch.float32).sum(dim=-1)
        geometry_id = self._infer_geometry_id(relation_scores, changed_scores, same_scores)
        rank = self.predicate_preference_rank.to(relation_scores.device)[geometry_id]
        return (mismatches + rank).argmin(dim=-1)

    def _predicate_grid(self, predicate_id: torch.Tensor) -> torch.Tensor:
        batch = predicate_id.shape[0]
        source_x = self.coord_x[None, None, :].expand(batch, self.cell_count, self.cell_count)
        source_y = self.coord_y[None, None, :].expand(batch, self.cell_count, self.cell_count)
        target_x = self.coord_x[None, :, None].expand(batch, self.cell_count, self.cell_count)
        target_y = self.coord_y[None, :, None].expand(batch, self.cell_count, self.cell_count)
        candidates = torch.stack(
            [
                torch.zeros_like(source_x, dtype=torch.bool),
                target_x <= 3,
                target_x >= 3,
                target_y <= 3,
                target_y >= 3,
                source_x <= 3,
                source_x >= 3,
                source_y <= 3,
                source_y >= 3,
                (target_x <= 3) & (target_y <= 3),
                (target_x <= 3) & (target_y >= 3),
                (target_x >= 3) & (target_y <= 3),
                (target_x >= 3) & (target_y >= 3),
                (source_x <= 3) & (source_y <= 3),
                (source_x <= 3) & (source_y >= 3),
                (source_x >= 3) & (source_y <= 3),
                (source_x >= 3) & (source_y >= 3),
            ],
            dim=1,
        )
        gather_index = predicate_id[:, None, None, None].expand(-1, 1, self.cell_count, self.cell_count)
        return candidates.gather(1, gather_index).squeeze(1)

    def color_transition_from_flags(self, query_source_colors: torch.Tensor, source_target_flags: torch.Tensor) -> torch.Tensor:
        batch = query_source_colors.shape[0]
        source_colors = query_source_colors.reshape(batch, self.cell_count)
        source = source_colors[:, None, :].expand(batch, self.cell_count, self.cell_count)
        recolor = (source > 0) & (source <= 4) & source_target_flags
        out_color = torch.where(recolor, source + 4, source)
        out_color = out_color.clamp(0, self.config.color_count - 1)
        probs = F.one_hot(out_color.long(), num_classes=self.config.color_count).to(torch.float32)
        background = source == 0
        if background.any():
            zero = F.one_hot(torch.zeros_like(source.long()), num_classes=self.config.color_count).to(torch.float32)
            probs = torch.where(background[..., None], zero, probs)
        return probs

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
        task_context = self.task_context_mlp(pair_representations.mean(dim=1))
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
        output_probs = torch.einsum("buv,buvc->buc", transport_attention, color_transition_probs)
        decoded = output_probs.argmax(dim=-1).reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE)
        return {
            "output_probs": output_probs.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.config.color_count),
            "decoded_grid": decoded,
            "task_context": task_context,
            "pair_representations": pair_representations,
            "transport_attention": transport_attention.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.cell_count),
            "color_transition_probs": color_transition_probs.reshape(query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.cell_count, self.config.color_count),
            "predicate_id": predicate_id,
            "geometry_id": self._infer_geometry_id(relation_scores, changed_scores, same_scores),
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "SupportConditionedTransportRecolorR5A0V2",
            "experiment": "R5A-0 V2 - Direct Factorized Transport-Recolor Baseline",
            "direct_factorization": "output_probs[u,c] = sum_v A[u,v] * B[u,v,c]",
            "lambda": 0,
            "connection_cost": False,
            "bimt": False,
            "sparsity_loss": False,
            "operator_labels": False,
            "sequence_labels": False,
            "pair_ids": False,
            "pair_position_embeddings": False,
            "query_type_metadata": False,
            "split_ids": False,
            "target_metadata": False,
            "intermediate_state_labels": False,
            "recurrence": False,
            "test_time_training": False,
            "generic_output_paint_bypass": False,
            "direct_context_to_output_color_bypass": False,
            "support_relation_encoder": {
                "color_embedding": True,
                "row_embedding": True,
                "column_embedding": True,
                "input_output_role_embedding": True,
                "hidden_size": self.config.hidden_size,
                "num_heads": self.config.num_heads,
                "relation_encoder_layers": self.config.relation_encoder_layers,
                "shared_pair_weights": True,
                "learned_pair_readout_token": True,
            },
            "color_blind_transport_head": True,
            "generic_color_transition_head": True,
            "predicate_candidates": list(self.predicate_names),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "initial_transport_logit_scale": float(self.logit_scale.exp().detach().cpu().item()),
        }


def output_loss(output_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    picked = output_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(picked.clamp_min(eps)).mean()


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
