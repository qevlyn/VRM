from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import (
    ARC_COLOR_COUNT,
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    MAX_GRID_WIDTH,
    MAX_SUPPORT_PAIRS,
    PAD_ID,
    VOCAB_SIZE,
)

ROLE_COUNT = 2
SEGMENT_QUERY_INPUT = 0
SEGMENT_OUTPUT_CANVAS = 1
NUM_SEGMENTS = 2


@dataclass
class VRMConfig:
    hidden_size: int = 128
    num_heads: int = 4
    pair_encoder_layers: int = 1
    rewrite_layers: int = 2
    rewrite_steps: int = 4
    rule_slots: int = 32
    dropout: float = 0.0
    max_grid_height: int = MAX_GRID_HEIGHT
    max_grid_width: int = MAX_GRID_WIDTH
    max_grid_tokens: int = MAX_GRID_TOKENS
    max_support_pairs: int = MAX_SUPPORT_PAIRS
    vocab_size: int = VOCAB_SIZE
    output_color_count: int = ARC_COLOR_COUNT
    shape_classes: int = MAX_GRID_HEIGHT


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = query if context is None else context
        batch_size, query_len, _ = query.shape
        key_len = context.shape[1]
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if key_mask is not None:
            validate_key_mask(key_mask, batch_size=batch_size, key_len=key_len, name="attention key_mask")
            attn_mask = torch.zeros(
                (batch_size, 1, 1, key_len),
                dtype=q.dtype,
                device=q.device,
            )
            attn_mask = attn_mask.masked_fill(~key_mask[:, None, None, :].to(torch.bool), float("-inf"))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_size)
        return self.out_proj(attended)


def validate_key_mask(key_mask: torch.Tensor, *, batch_size: int, key_len: int, name: str) -> None:
    if key_mask.shape != (batch_size, key_len):
        raise ValueError(f"{name} must have shape {(batch_size, key_len)}, got {tuple(key_mask.shape)}")
    if key_mask.dtype != torch.bool:
        raise TypeError(f"{name} must be boolean, got {key_mask.dtype}")
    if not bool(key_mask.any(dim=1).all().item()):
        raise ValueError(f"{name} has a batch item with zero valid keys")


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4) -> None:
        super().__init__()
        inner_size = hidden_size * expansion
        self.w12 = nn.Linear(hidden_size, inner_size * 2, bias=False)
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class PairEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class RewriteLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, use_rule_tokens: bool = True) -> None:
        super().__init__()
        self.canvas_norm = RMSNorm(hidden_size)
        self.canvas_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.query_norm = RMSNorm(hidden_size)
        self.query_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.support_norm = RMSNorm(hidden_size)
        self.support_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.use_rule_tokens = use_rule_tokens
        if use_rule_tokens:
            self.rule_norm = RMSNorm(hidden_size)
            self.rule_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        canvas: torch.Tensor,
        query_bank: torch.Tensor,
        query_mask: torch.Tensor,
        support_bank: torch.Tensor,
        support_mask: torch.Tensor,
        rule_bank: torch.Tensor | None,
        disable_support_evidence: bool = False,
    ) -> torch.Tensor:
        canvas = canvas + self.dropout(self.canvas_attn(self.canvas_norm(canvas)))
        canvas = canvas + self.dropout(self.query_attn(self.query_norm(canvas), query_bank, key_mask=query_mask))
        if not disable_support_evidence:
            canvas = canvas + self.dropout(
                self.support_attn(self.support_norm(canvas), support_bank, key_mask=support_mask)
            )
        if self.use_rule_tokens and rule_bank is not None:
            canvas = canvas + self.dropout(self.rule_attn(self.rule_norm(canvas), rule_bank))
        canvas = canvas + self.dropout(self.ffn(self.ffn_norm(canvas)))
        return canvas


class VRM(nn.Module):
    """VRM V1.1 - Masked Query Spatial Evidence Rewrite.

    Support evidence is packed into variable-length valid-cell sequences and remains
    directly cross-attended by the shape/output canvas state at every rewrite step.
    """

    def __init__(self, config: VRMConfig | None = None) -> None:
        super().__init__()
        self.config = config or VRMConfig()
        cfg = self.config
        if cfg.max_grid_tokens != cfg.max_grid_height * cfg.max_grid_width:
            raise ValueError("max_grid_tokens must equal max_grid_height * max_grid_width")
        if cfg.max_support_pairs > MAX_SUPPORT_PAIRS:
            raise ValueError(f"max_support_pairs cannot exceed data.MAX_SUPPORT_PAIRS={MAX_SUPPORT_PAIRS}")

        self.color_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.row_embedding = nn.Embedding(cfg.max_grid_height, cfg.hidden_size)
        self.col_embedding = nn.Embedding(cfg.max_grid_width, cfg.hidden_size)
        self.role_embedding = nn.Embedding(ROLE_COUNT, cfg.hidden_size)
        self.support_pair_embedding = nn.Embedding(cfg.max_support_pairs, cfg.hidden_size)
        self.segment_embedding = nn.Embedding(NUM_SEGMENTS, cfg.hidden_size)
        self.height_shape_token = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
        self.width_shape_token = nn.Parameter(torch.randn(cfg.hidden_size) * 0.02)
        self.output_slot_embedding = nn.Parameter(torch.randn(cfg.max_grid_tokens, cfg.hidden_size) * 0.02)
        self.rule_tokens = (
            nn.Parameter(torch.randn(cfg.rule_slots, cfg.hidden_size) * 0.02)
            if cfg.rule_slots > 0
            else None
        )

        self.pair_encoder = nn.ModuleList(
            [PairEncoderBlock(cfg.hidden_size, cfg.num_heads, cfg.dropout) for _ in range(cfg.pair_encoder_layers)]
        )
        if cfg.rule_slots > 0:
            self.rule_cross_norm = RMSNorm(cfg.hidden_size)
            self.rule_cross_attn = MultiHeadAttention(cfg.hidden_size, cfg.num_heads, cfg.dropout)
            self.rule_ffn_norm = RMSNorm(cfg.hidden_size)
            self.rule_ffn = SwiGLU(cfg.hidden_size)
            self.rule_dropout = nn.Dropout(cfg.dropout)
        self.rewrite_layers = nn.ModuleList(
            [
                RewriteLayer(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.dropout,
                    use_rule_tokens=cfg.rule_slots > 0,
                )
                for _ in range(cfg.rewrite_layers)
            ]
        )
        self.final_norm = RMSNorm(cfg.hidden_size)
        self.height_head = nn.Linear(cfg.hidden_size, cfg.shape_classes)
        self.width_head = nn.Linear(cfg.hidden_size, cfg.shape_classes)
        self.paint_head = nn.Linear(cfg.hidden_size, cfg.output_color_count)
        self.copy_gate_head = nn.Linear(cfg.hidden_size, 1)
        self.copy_q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.copy_k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        rows = torch.arange(cfg.max_grid_height).repeat_interleave(cfg.max_grid_width)
        cols = torch.arange(cfg.max_grid_width).repeat(cfg.max_grid_height)
        self.register_buffer("row_indices", rows, persistent=False)
        self.register_buffer("col_indices", cols, persistent=False)

    def encode_query_input(self, query_inputs: torch.Tensor) -> torch.Tensor:
        segment_ids = torch.full_like(query_inputs, SEGMENT_QUERY_INPUT)
        return (
            self.color_embedding(query_inputs)
            + self.row_embedding(self.row_indices.to(query_inputs.device))[None, :, :]
            + self.col_embedding(self.col_indices.to(query_inputs.device))[None, :, :]
            + self.segment_embedding(segment_ids)
        )

    def initialize_canvas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        segment_ids = torch.full(
            (batch_size, self.config.max_grid_tokens),
            SEGMENT_OUTPUT_CANVAS,
            dtype=torch.long,
            device=device,
        )
        return (
            self.output_slot_embedding.to(device)[None, :, :].expand(batch_size, -1, -1)
            + self.row_embedding(self.row_indices.to(device))[None, :, :]
            + self.col_embedding(self.col_indices.to(device))[None, :, :]
            + self.segment_embedding(segment_ids)
        )

    def embed_support_evidence(
        self,
        *,
        support_evidence_colors: torch.Tensor,
        support_evidence_rows: torch.Tensor,
        support_evidence_cols: torch.Tensor,
        support_evidence_roles: torch.Tensor,
        support_evidence_pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.color_embedding(support_evidence_colors)
            + self.row_embedding(support_evidence_rows)
            + self.col_embedding(support_evidence_cols)
            + self.role_embedding(support_evidence_roles)
            + self.support_pair_embedding(support_evidence_pair_indices)
        )

    def encode_support_evidence(
        self,
        *,
        support_evidence_colors: torch.Tensor,
        support_evidence_rows: torch.Tensor,
        support_evidence_cols: torch.Tensor,
        support_evidence_roles: torch.Tensor,
        support_evidence_pair_indices: torch.Tensor,
        support_evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed_support_evidence(
            support_evidence_colors=support_evidence_colors,
            support_evidence_rows=support_evidence_rows,
            support_evidence_cols=support_evidence_cols,
            support_evidence_roles=support_evidence_roles,
            support_evidence_pair_indices=support_evidence_pair_indices,
        )
        encoded_items: list[torch.Tensor] = []
        for batch_idx in range(embedded.shape[0]):
            valid = support_evidence_mask[batch_idx].to(torch.bool)
            if not bool(valid.any()):
                raise ValueError("each episode must contain at least one packed support evidence token")
            pair_ids = torch.unique(support_evidence_pair_indices[batch_idx, valid]).tolist()
            pair_outputs: list[torch.Tensor] = []
            for pair_id in sorted(int(item) for item in pair_ids):
                pair_mask = valid & (support_evidence_pair_indices[batch_idx] == pair_id)
                pair_tokens = embedded[batch_idx, pair_mask, :].unsqueeze(0)
                for layer in self.pair_encoder:
                    pair_tokens = layer(pair_tokens)
                pair_outputs.append(pair_tokens.squeeze(0))
            encoded_items.append(torch.cat(pair_outputs, dim=0))
        support_bank = pad_sequence(encoded_items, batch_first=True)
        lengths = torch.tensor([item.shape[0] for item in encoded_items], device=support_bank.device)
        positions = torch.arange(support_bank.shape[1], device=support_bank.device)[None, :]
        support_mask = positions < lengths[:, None]
        return support_bank, support_mask

    def prepare_rule_tokens(
        self,
        support_bank: torch.Tensor,
        support_mask: torch.Tensor,
        disable_support_evidence: bool = False,
    ) -> torch.Tensor | None:
        if self.rule_tokens is None:
            return None
        batch_size = support_bank.shape[0]
        rule_bank = self.rule_tokens[None, :, :].expand(batch_size, -1, -1)
        if disable_support_evidence:
            return rule_bank
        rule_bank = rule_bank + self.rule_dropout(
            self.rule_cross_attn(self.rule_cross_norm(rule_bank), support_bank, key_mask=support_mask)
        )
        rule_bank = rule_bank + self.rule_dropout(self.rule_ffn(self.rule_ffn_norm(rule_bank)))
        return rule_bank

    def copy_and_paint_log_probs(
        self,
        canvas: torch.Tensor,
        query_bank: torch.Tensor,
        query_inputs: torch.Tensor,
        query_input_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, canvas_len, _ = canvas.shape
        query_len = query_bank.shape[1]
        validate_key_mask(query_input_mask, batch_size=batch_size, key_len=query_len, name="query_input_mask")
        head_dim = self.config.hidden_size // self.config.num_heads
        q = self.copy_q(canvas).view(batch_size, canvas_len, self.config.num_heads, head_dim).transpose(1, 2)
        k = self.copy_k(query_bank).view(batch_size, query_len, self.config.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~query_input_mask[:, None, None, :], float("-inf"))
        copy_attention_by_head = torch.softmax(scores, dim=-1)
        if not torch.isfinite(copy_attention_by_head).all():
            raise RuntimeError("copy attention produced non-finite values")
        copy_attention = copy_attention_by_head.mean(dim=1)
        safe_query_inputs = query_inputs.clamp(min=0, max=self.config.output_color_count - 1)
        query_colors = F.one_hot(safe_query_inputs, num_classes=self.config.output_color_count).to(dtype=canvas.dtype)
        query_colors = query_colors * query_input_mask[:, :, None].to(dtype=canvas.dtype)
        copy_distribution = torch.matmul(copy_attention, query_colors)
        paint_distribution = torch.softmax(self.paint_head(canvas), dim=-1)
        copy_gate = torch.sigmoid(self.copy_gate_head(canvas))
        final_distribution = copy_gate * copy_distribution + (1.0 - copy_gate) * paint_distribution
        probability_sum = final_distribution.sum(dim=-1)
        if not torch.allclose(probability_sum, torch.ones_like(probability_sum), atol=1e-4, rtol=1e-4):
            raise RuntimeError("copy-and-paint probabilities do not sum to one")
        safe_floor = 1e-8
        color_log_probs = torch.log(final_distribution.clamp_min(safe_floor))
        copy_entropy = -(copy_distribution * torch.log(copy_distribution.clamp_min(safe_floor))).sum(dim=-1)
        paint_entropy = -(paint_distribution * torch.log(paint_distribution.clamp_min(safe_floor))).sum(dim=-1)
        return {
            "color_logits": color_log_probs,
            "color_log_probs": color_log_probs,
            "color_probabilities": final_distribution,
            "copy_attention": copy_attention,
            "copy_distribution": copy_distribution,
            "paint_distribution": paint_distribution,
            "copy_gate": copy_gate,
            "copy_entropy": copy_entropy,
            "paint_entropy": paint_entropy,
        }

    def forward(
        self,
        *,
        support_evidence_colors: torch.Tensor,
        support_evidence_rows: torch.Tensor,
        support_evidence_cols: torch.Tensor,
        support_evidence_roles: torch.Tensor,
        support_evidence_pair_indices: torch.Tensor,
        support_evidence_mask: torch.Tensor,
        query_inputs: torch.Tensor,
        query_input_mask: torch.Tensor,
        disable_support_evidence: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = query_inputs.shape[0]
        validate_key_mask(
            query_input_mask,
            batch_size=batch_size,
            key_len=self.config.max_grid_tokens,
            name="query_input_mask",
        )
        query_bank = self.encode_query_input(query_inputs)
        support_bank, support_mask = self.encode_support_evidence(
            support_evidence_colors=support_evidence_colors,
            support_evidence_rows=support_evidence_rows,
            support_evidence_cols=support_evidence_cols,
            support_evidence_roles=support_evidence_roles,
            support_evidence_pair_indices=support_evidence_pair_indices,
            support_evidence_mask=support_evidence_mask,
        )
        rule_bank = self.prepare_rule_tokens(
            support_bank,
            support_mask,
            disable_support_evidence=disable_support_evidence,
        )
        canvas = self.initialize_canvas(batch_size, query_inputs.device)
        height_token = self.height_shape_token.to(query_inputs.device)[None, None, :].expand(batch_size, 1, -1)
        width_token = self.width_shape_token.to(query_inputs.device)[None, None, :].expand(batch_size, 1, -1)
        rewrite_state = torch.cat([height_token, width_token, canvas], dim=1)
        for _ in range(self.config.rewrite_steps):
            for layer in self.rewrite_layers:
                rewrite_state = layer(
                    rewrite_state,
                    query_bank,
                    query_input_mask,
                    support_bank,
                    support_mask,
                    rule_bank,
                    disable_support_evidence=disable_support_evidence,
                )
        final_state = self.final_norm(rewrite_state)
        height_logits = self.height_head(final_state[:, 0, :])
        width_logits = self.width_head(final_state[:, 1, :])
        canvas = final_state[:, 2:, :]
        color_outputs = self.copy_and_paint_log_probs(canvas, query_bank, query_inputs, query_input_mask)
        return {
            "color_logits": color_outputs["color_logits"],
            "color_log_probs": color_outputs["color_log_probs"],
            "logits": color_outputs["color_logits"],
            "color_probabilities": color_outputs["color_probabilities"],
            "height_logits": height_logits,
            "width_logits": width_logits,
            "final_canvas": canvas,
            "final_rewrite_state": final_state,
            "support_evidence_bank": support_bank,
            "support_evidence_mask": support_mask,
            "rule_tokens": rule_bank if rule_bank is not None else canvas.new_zeros((batch_size, 0, self.config.hidden_size)),
            "copy_gate": color_outputs["copy_gate"],
            "copy_attention": color_outputs["copy_attention"],
            "copy_distribution": color_outputs["copy_distribution"],
            "paint_distribution": color_outputs["paint_distribution"],
            "copy_entropy": color_outputs["copy_entropy"],
            "paint_entropy": color_outputs["paint_entropy"],
        }
