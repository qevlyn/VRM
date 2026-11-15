from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

GRID_SIZE = 7
CELL_COUNT = GRID_SIZE * GRID_SIZE
COLOR_COUNT = 9
SUPPORT_PAIRS = 3
SUPPORT_ROLES = 2
SUPPORT_TOKEN_COUNT = SUPPORT_PAIRS * SUPPORT_ROLES * CELL_COUNT
QUERY_TOKEN_COUNT = CELL_COUNT
MEMORY_TOKEN_COUNT = SUPPORT_TOKEN_COUNT + QUERY_TOKEN_COUNT
OUTPUT_CELL_COUNT = CELL_COUNT

ALLOWED_INPUT_FIELDS = (
    "query_grids",
    "support_input_grids",
    "support_output_grids",
    "memory_token_mask",
    "output_cell_mask",
    "canvas_hw",
)

FORBIDDEN_FIELD_NAMES = (
    "operation",
    "program",
    "class",
    "split",
    "seed",
    "signature",
    "oracle",
    "target",
    "transport",
    "color_transition",
    "pointer",
    "copy",
)


@dataclass(frozen=True, slots=True)
class SharedRendererInput:
    query_grids: torch.Tensor
    support_input_grids: torch.Tensor
    support_output_grids: torch.Tensor
    memory_token_mask: torch.Tensor
    output_cell_mask: torch.Tensor
    canvas_hw: torch.Tensor

    def validate(self) -> None:
        if self.query_grids.ndim != 3 or tuple(self.query_grids.shape[-2:]) != (GRID_SIZE, GRID_SIZE):
            raise ValueError(f"query_grids must have shape [B,{GRID_SIZE},{GRID_SIZE}]")
        if self.support_input_grids.ndim != 4 or tuple(self.support_input_grids.shape[1:]) != (SUPPORT_PAIRS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"support_input_grids must have shape [B,{SUPPORT_PAIRS},{GRID_SIZE},{GRID_SIZE}]")
        if self.support_output_grids.shape != self.support_input_grids.shape:
            raise ValueError("support_output_grids must match support_input_grids shape")
        batch = self.query_grids.shape[0]
        if self.memory_token_mask.shape != (batch, MEMORY_TOKEN_COUNT):
            raise ValueError(f"memory_token_mask must have shape [B,{MEMORY_TOKEN_COUNT}]")
        if self.output_cell_mask.shape != (batch, OUTPUT_CELL_COUNT):
            raise ValueError(f"output_cell_mask must have shape [B,{OUTPUT_CELL_COUNT}]")
        if self.canvas_hw.shape != (batch, 2):
            raise ValueError("canvas_hw must have shape [B,2]")
        if self.query_grids.device != self.support_input_grids.device or self.query_grids.device != self.support_output_grids.device:
            raise ValueError("all grid tensors must be on the same device")


@dataclass(frozen=True)
class SharedRendererConfig:
    variant_name: str
    hidden_size: int
    num_heads: int
    encoder_layers: int
    decoder_layers: int
    mlp_ratio: int = 2
    dropout: float = 0.0
    grid_size: int = GRID_SIZE
    color_count: int = COLOR_COUNT
    support_pairs: int = SUPPORT_PAIRS
    output_cell_count: int = OUTPUT_CELL_COUNT

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_heads <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("num_heads must divide hidden_size")
        if self.encoder_layers < 0 or self.decoder_layers <= 0:
            raise ValueError("encoder_layers must be non-negative and decoder_layers must be positive")
        if self.grid_size != GRID_SIZE or self.support_pairs != SUPPORT_PAIRS or self.output_cell_count != OUTPUT_CELL_COUNT:
            raise ValueError("R5A-1A preflight fixes the canonical 7x7 V2 support/query canvas")


@dataclass(frozen=True)
class SharedRendererOutput:
    color_logits: torch.Tensor
    output_size_logits: torch.Tensor
    memory_token_mask: torch.Tensor
    output_cell_mask: torch.Tensor


def default_variant_configs() -> dict[str, SharedRendererConfig]:
    return {
        "shared_small": SharedRendererConfig("shared_small", hidden_size=8, num_heads=1, encoder_layers=0, decoder_layers=1, mlp_ratio=2),
        "shared_matched": SharedRendererConfig("shared_matched", hidden_size=96, num_heads=4, encoder_layers=0, decoder_layers=2, mlp_ratio=4),
        "shared_large": SharedRendererConfig("shared_large", hidden_size=160, num_heads=8, encoder_layers=0, decoder_layers=3, mlp_ratio=4),
    }


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int) -> None:
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GenericDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardBlock(hidden_size, mlp_ratio)

    def forward(self, output_tokens: torch.Tensor, memory_tokens: torch.Tensor, memory_token_mask: torch.Tensor) -> torch.Tensor:
        x = output_tokens
        self_in = self.self_norm(x)
        self_out, _ = self.self_attention(self_in, self_in, self_in, need_weights=False)
        x = x + self_out
        cross_in = self.cross_norm(x)
        padding_mask = ~memory_token_mask.bool()
        cross_out, _ = self.cross_attention(cross_in, memory_tokens, memory_tokens, key_padding_mask=padding_mask, need_weights=False)
        x = x + cross_out
        return self.ffn(x)


class GenericSharedRenderer(nn.Module):
    """Generic shared renderer with no explicit transport/color/pointer/copy interface."""

    def __init__(self, config: SharedRendererConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        self.color_embedding = nn.Embedding(config.color_count, h)
        self.row_embedding = nn.Embedding(config.grid_size, h)
        self.column_embedding = nn.Embedding(config.grid_size, h)
        self.role_embedding = nn.Embedding(4, h)
        self.pair_embedding = nn.Embedding(config.support_pairs + 1, h)
        self.output_cell_queries = nn.Parameter(torch.zeros(config.output_cell_count, h))
        self.input_ffn = FeedForwardBlock(h, config.mlp_ratio)
        self.encoder_layers = nn.ModuleList([FeedForwardBlock(h, config.mlp_ratio) for _ in range(config.encoder_layers)])
        self.decoder_layers = nn.ModuleList([GenericDecoderLayer(h, config.num_heads, config.mlp_ratio, config.dropout) for _ in range(config.decoder_layers)])
        self.color_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.color_count))
        self.size_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h * config.mlp_ratio),
            nn.GELU(),
            nn.Linear(h * config.mlp_ratio, 2 * config.grid_size),
        )
        coords = [(x, y) for y in range(config.grid_size) for x in range(config.grid_size)]
        self.register_buffer("coord_x", torch.tensor([x for x, _ in coords], dtype=torch.long), persistent=False)
        self.register_buffer("coord_y", torch.tensor([y for _, y in coords], dtype=torch.long), persistent=False)

    def _embed_cells(self, colors: torch.Tensor, role_id: int, pair_ids: torch.Tensor) -> torch.Tensor:
        batch = colors.shape[0]
        flat = colors.reshape(batch, -1)
        token_count = flat.shape[1]
        reps_per_canvas = token_count // CELL_COUNT
        x = self.coord_x.repeat(reps_per_canvas).to(flat.device)
        y = self.coord_y.repeat(reps_per_canvas).to(flat.device)
        role = torch.full((token_count,), role_id, dtype=torch.long, device=flat.device)
        return (
            self.color_embedding(flat.clamp(0, self.config.color_count - 1))
            + self.row_embedding(y)[None, :, :]
            + self.column_embedding(x)[None, :, :]
            + self.role_embedding(role)[None, :, :]
            + self.pair_embedding(pair_ids.to(flat.device))[None, :, :]
        )

    def encode_memory(self, batch: SharedRendererInput) -> torch.Tensor:
        batch.validate()
        bsz = batch.query_grids.shape[0]
        pair_ids = torch.arange(self.config.support_pairs, device=batch.query_grids.device).repeat_interleave(SUPPORT_ROLES * CELL_COUNT)
        support_inputs = batch.support_input_grids.reshape(bsz, self.config.support_pairs, CELL_COUNT)
        support_outputs = batch.support_output_grids.reshape(bsz, self.config.support_pairs, CELL_COUNT)
        support_interleaved = torch.stack([support_inputs, support_outputs], dim=2).reshape(bsz, SUPPORT_TOKEN_COUNT)
        support_roles = torch.tensor(
            ([0] * CELL_COUNT + [1] * CELL_COUNT) * self.config.support_pairs,
            dtype=torch.long,
            device=batch.query_grids.device,
        )
        support_x = self.coord_x.repeat(SUPPORT_ROLES * self.config.support_pairs).to(batch.query_grids.device)
        support_y = self.coord_y.repeat(SUPPORT_ROLES * self.config.support_pairs).to(batch.query_grids.device)
        support_tokens = (
            self.color_embedding(support_interleaved.clamp(0, self.config.color_count - 1))
            + self.row_embedding(support_y)[None, :, :]
            + self.column_embedding(support_x)[None, :, :]
            + self.role_embedding(support_roles)[None, :, :]
            + self.pair_embedding(pair_ids)[None, :, :]
        )
        query_pair_ids = torch.full((QUERY_TOKEN_COUNT,), self.config.support_pairs, dtype=torch.long, device=batch.query_grids.device)
        query_tokens = self._embed_cells(batch.query_grids, 2, query_pair_ids)
        memory = torch.cat([support_tokens, query_tokens], dim=1)
        memory = self.input_ffn(memory)
        for layer in self.encoder_layers:
            memory = layer(memory)
        return memory

    def output_queries(self, batch: SharedRendererInput) -> torch.Tensor:
        bsz = batch.query_grids.shape[0]
        x = self.coord_x.to(batch.query_grids.device)
        y = self.coord_y.to(batch.query_grids.device)
        role = torch.full((OUTPUT_CELL_COUNT,), 3, dtype=torch.long, device=batch.query_grids.device)
        pair = torch.full((OUTPUT_CELL_COUNT,), self.config.support_pairs, dtype=torch.long, device=batch.query_grids.device)
        tokens = (
            self.output_cell_queries[None, :, :].expand(bsz, -1, -1)
            + self.row_embedding(y)[None, :, :]
            + self.column_embedding(x)[None, :, :]
            + self.role_embedding(role)[None, :, :]
            + self.pair_embedding(pair)[None, :, :]
        )
        return tokens

    def forward(self, batch: SharedRendererInput) -> SharedRendererOutput:
        if not isinstance(batch, SharedRendererInput):
            raise TypeError("GenericSharedRenderer.forward accepts only SharedRendererInput")
        batch.validate()
        memory = self.encode_memory(batch)
        output_tokens = self.output_queries(batch)
        for layer in self.decoder_layers:
            output_tokens = layer(output_tokens, memory, batch.memory_token_mask)
        color_logits = self.color_head(output_tokens).reshape(batch.query_grids.shape[0], GRID_SIZE, GRID_SIZE, self.config.color_count)
        pooled_denominator = batch.memory_token_mask.to(memory.dtype).sum(dim=1).clamp_min(1.0)[:, None]
        pooled = (memory * batch.memory_token_mask[:, :, None].to(memory.dtype)).sum(dim=1) / pooled_denominator
        size_logits = self.size_head(pooled).reshape(batch.query_grids.shape[0], 2, GRID_SIZE)
        return SharedRendererOutput(
            color_logits=color_logits,
            output_size_logits=size_logits,
            memory_token_mask=batch.memory_token_mask,
            output_cell_mask=batch.output_cell_mask,
        )


def pack_shared_input_from_records(records: list[dict[str, Any]], device: torch.device | None = None) -> SharedRendererInput:
    query = torch.tensor([record["query"] for record in records], dtype=torch.long, device=device)
    support_in = torch.tensor([[pair["input"] for pair in record["supports"]] for record in records], dtype=torch.long, device=device)
    support_out = torch.tensor([[pair["output"] for pair in record["supports"]] for record in records], dtype=torch.long, device=device)
    batch = query.shape[0]
    memory_mask = torch.ones((batch, MEMORY_TOKEN_COUNT), dtype=torch.bool, device=device)
    output_mask = torch.ones((batch, OUTPUT_CELL_COUNT), dtype=torch.bool, device=device)
    canvas_hw = torch.full((batch, 2), GRID_SIZE, dtype=torch.long, device=device)
    return SharedRendererInput(
        query_grids=query,
        support_input_grids=support_in,
        support_output_grids=support_out,
        memory_token_mask=memory_mask,
        output_cell_mask=output_mask,
        canvas_hw=canvas_hw,
    )


def tensor_digest(batch: SharedRendererInput) -> str:
    import hashlib

    digest = hashlib.sha256()
    for tensor in (
        batch.query_grids,
        batch.support_input_grids,
        batch.support_output_grids,
        batch.memory_token_mask,
        batch.output_cell_mask,
        batch.canvas_hw,
    ):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(cpu.shape)).encode("utf-8"))
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_embedding_parameters(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            total += sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    if hasattr(model, "output_cell_queries"):
        total += int(model.output_cell_queries.numel())
    return total
