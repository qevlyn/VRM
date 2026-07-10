from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT


@dataclass(frozen=True)
class PairReaderConfig:
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    max_height: int = 7
    max_width: int = 7
    color_count: int = ARC_COLOR_COUNT
    temperature: float = 0.1
    background_color: int = 0


class ForegroundSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(
            self.attn_norm(tokens),
            self.attn_norm(tokens),
            self.attn_norm(tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class GridMatcherEncoder(nn.Module):
    """Shared foreground-only encoder for query grids and support-input grids."""

    def __init__(self, config: PairReaderConfig) -> None:
        super().__init__()
        self.config = config
        self.color_embedding = nn.Embedding(config.color_count, config.hidden_size)
        self.row_embedding = nn.Embedding(config.max_height, config.hidden_size)
        self.col_embedding = nn.Embedding(config.max_width, config.hidden_size)
        self.readout = nn.Parameter(torch.zeros(config.hidden_size))
        self.blocks = nn.ModuleList(
            ForegroundSelfAttentionBlock(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        nn.init.normal_(self.readout, std=0.02)

    def _foreground_tokens(self, grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if grids.ndim != 3:
            raise ValueError(f"matcher expects [batch,height,width] grids, got {tuple(grids.shape)}")
        if grids.shape[1] > self.config.max_height or grids.shape[2] > self.config.max_width:
            raise ValueError(f"grid shape {tuple(grids.shape[1:])} exceeds configured matcher extent")
        if grids.min().item() < 0 or grids.max().item() >= self.config.color_count:
            raise ValueError("grid colors must be in ARC color range")
        foreground = grids != self.config.background_color
        counts = foreground.flatten(1).sum(dim=1)
        if bool((counts == 0).any().item()):
            raise ValueError("PairKeyValueReader requires at least one foreground cell per matcher grid")
        max_tokens = int(counts.max().item())
        batch = grids.shape[0]
        device = grids.device
        colors = torch.zeros((batch, max_tokens), dtype=torch.long, device=device)
        rows = torch.zeros((batch, max_tokens), dtype=torch.long, device=device)
        cols = torch.zeros((batch, max_tokens), dtype=torch.long, device=device)
        padding = torch.ones((batch, max_tokens), dtype=torch.bool, device=device)
        for item_idx in range(batch):
            positions = foreground[item_idx].nonzero(as_tuple=False)
            count = positions.shape[0]
            rows[item_idx, :count] = positions[:, 0]
            cols[item_idx, :count] = positions[:, 1]
            colors[item_idx, :count] = grids[item_idx, positions[:, 0], positions[:, 1]]
            padding[item_idx, :count] = False
        return colors, rows, cols, padding

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        colors, rows, cols, padding = self._foreground_tokens(grids)
        token_embeddings = self.color_embedding(colors) + self.row_embedding(rows) + self.col_embedding(cols)
        readout = self.readout.view(1, 1, -1).expand(grids.shape[0], 1, -1)
        tokens = torch.cat([readout, token_embeddings], dim=1)
        key_padding_mask = torch.cat(
            [torch.zeros((grids.shape[0], 1), dtype=torch.bool, device=grids.device), padding],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask)
        return F.normalize(self.output_norm(tokens[:, 0]), p=2, dim=-1)


class PairKeyValueReader(nn.Module):
    """Learned key-value reader with raw support-output grids as values."""

    def __init__(self, config: PairReaderConfig | None = None) -> None:
        super().__init__()
        self.config = config or PairReaderConfig()
        self.matcher = GridMatcherEncoder(self.config)
        if self.config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.register_buffer("temperature", torch.tensor(float(self.config.temperature)))

    @property
    def query_encoder(self) -> GridMatcherEncoder:
        return self.matcher

    @property
    def support_input_encoder(self) -> GridMatcherEncoder:
        return self.matcher

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_grids.ndim != 3:
            raise ValueError("query_grids must have shape [batch,height,width]")
        if support_input_grids.ndim != 4 or support_output_grids.ndim != 4:
            raise ValueError("support grids must have shape [batch,pairs,height,width]")
        if support_input_grids.shape != support_output_grids.shape:
            raise ValueError("support input and output tensors must share shape")
        if support_input_grids.shape[0] != query_grids.shape[0]:
            raise ValueError("query and support batch sizes differ")
        if support_input_grids.shape[2:] != query_grids.shape[1:]:
            raise ValueError("query and support grid shapes differ")

        batch, pair_count, height, width = support_input_grids.shape
        query_vec = self.matcher(query_grids)
        key_vecs = self.matcher(support_input_grids.reshape(batch * pair_count, height, width))
        key_vecs = key_vecs.reshape(batch, pair_count, -1)
        retrieval_logits = torch.einsum("bd,bpd->bp", query_vec, key_vecs) / self.temperature.clamp_min(1e-6)
        alpha = torch.softmax(retrieval_logits, dim=-1)
        values = F.one_hot(support_output_grids.long(), num_classes=self.config.color_count).to(torch.float32)
        retrieved = torch.sum(alpha[:, :, None, None, None] * values, dim=1)
        decoded = retrieved.argmax(dim=-1)
        return {
            "retrieval_logits": retrieval_logits,
            "retrieval_weights": alpha,
            "retrieved_color_probabilities": retrieved,
            "decoded_grid": decoded,
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "Pair Key-Value Reader R1",
            "config": asdict(self.config),
            "parameter_count": {
                "total": sum(parameter.numel() for parameter in self.parameters()),
                "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            },
            "query_and_support_input_encoder_share_weights": self.query_encoder is self.support_input_encoder,
            "uses_pair_position_embedding": False,
            "uses_pair_index_embedding": False,
            "value_route": "raw support output one-hot grids mixed only by retrieval alpha",
            "foreground_strategy": (
                "foreground-only tokenization of non-background cells with color, row, and column embeddings; "
                "empty foreground matcher grids are treated as dataset errors"
            ),
        }


def output_nll_from_retrieved_probs(
    retrieved_color_probabilities: torch.Tensor,
    target_grids: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if retrieved_color_probabilities.ndim != 4:
        raise ValueError("retrieved probabilities must have shape [batch,height,width,color]")
    gathered = retrieved_color_probabilities.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def pair_reader_loss(
    outputs: dict[str, torch.Tensor],
    target_grids: torch.Tensor,
    relevant_pair_indices: torch.Tensor,
    *,
    lambda_retrieval: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output_loss = output_nll_from_retrieved_probs(outputs["retrieved_color_probabilities"], target_grids)
    retrieval_loss = F.cross_entropy(outputs["retrieval_logits"], relevant_pair_indices.long())
    total = output_loss + float(lambda_retrieval) * retrieval_loss
    return total, {
        "total_loss": total.detach(),
        "output_loss": output_loss.detach(),
        "retrieval_loss": retrieval_loss.detach(),
    }


def entropy_from_weights(alpha: torch.Tensor) -> torch.Tensor:
    return -(alpha.clamp_min(1e-8) * torch.log(alpha.clamp_min(1e-8))).sum(dim=-1)


def convex_value_mix(support_output_grids: torch.Tensor, alpha: torch.Tensor, color_count: int = ARC_COLOR_COUNT) -> torch.Tensor:
    values = F.one_hot(support_output_grids.long(), num_classes=color_count).to(torch.float32)
    return torch.sum(alpha[:, :, None, None, None] * values, dim=1)

