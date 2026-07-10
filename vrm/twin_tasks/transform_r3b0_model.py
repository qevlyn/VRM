from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT


@dataclass(frozen=True)
class TransformR3B0Config:
    grid_size: int = 7
    color_count: int = ARC_COLOR_COUNT
    hidden_size: int = 128
    attention_heads: int = 4
    relation_layers: int = 2
    dropout: float = 0.0


class SupportConditionedTransformPointerR3B0(nn.Module):
    """Support-conditioned query-copy pointer for single-rule transform episodes."""

    def __init__(self, config: TransformR3B0Config | None = None) -> None:
        super().__init__()
        self.config = config or TransformR3B0Config()
        hidden = self.config.hidden_size
        grid = self.config.grid_size
        self.color_embedding = nn.Embedding(self.config.color_count, hidden)
        self.row_embedding = nn.Embedding(grid, hidden)
        self.col_embedding = nn.Embedding(grid, hidden)
        self.role_embedding = nn.Embedding(2, hidden)
        self.readout_token = nn.Parameter(torch.zeros(1, 1, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.relation_layers)
        self.relation_projector = nn.Sequential(
            nn.Linear(grid * grid * grid * grid, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
        )
        self.pointer_generator = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False),
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, grid * grid * grid * grid, bias=False),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.readout_token, mean=0.0, std=0.02)

    @property
    def cell_count(self) -> int:
        return self.config.grid_size * self.config.grid_size

    def _row_col_indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.arange(self.config.grid_size, device=device).repeat_interleave(self.config.grid_size)
        cols = torch.arange(self.config.grid_size, device=device).repeat(self.config.grid_size)
        return rows, cols

    def _cell_tokens(self, grids: torch.Tensor, role_index: int) -> torch.Tensor:
        batch = grids.shape[0]
        flat = grids.reshape(batch, self.cell_count)
        rows, cols = self._row_col_indices(grids.device)
        role = torch.full((self.cell_count,), role_index, dtype=torch.long, device=grids.device)
        return (
            self.color_embedding(flat.long())
            + self.row_embedding(rows)[None, :, :]
            + self.col_embedding(cols)[None, :, :]
            + self.role_embedding(role)[None, :, :]
        )

    def encode_pairs(self, support_input_grids: torch.Tensor, support_output_grids: torch.Tensor) -> torch.Tensor:
        if support_input_grids.shape != support_output_grids.shape:
            raise ValueError("support input/output shape mismatch")
        if support_input_grids.ndim != 4:
            raise ValueError("support grids must be [batch,pairs,7,7]")
        batch, pairs, height, width = support_input_grids.shape
        if (height, width) != (self.config.grid_size, self.config.grid_size):
            raise ValueError("R3B0 expects 7x7 support grids")
        flat_inputs = support_input_grids.reshape(batch * pairs, height, width)
        flat_outputs = support_output_grids.reshape(batch * pairs, height, width)
        input_tokens = self._cell_tokens(flat_inputs, role_index=0)
        output_tokens = self._cell_tokens(flat_outputs, role_index=1)
        readout = self.readout_token.expand(batch * pairs, -1, -1)
        encoded = self.relation_encoder(torch.cat([readout, input_tokens, output_tokens], dim=1))
        transformer_readout = encoded[:, 0, :]
        input_one_hot = F.one_hot(flat_inputs.reshape(batch * pairs, self.cell_count).long(), num_classes=self.config.color_count).to(torch.float32)[..., 1:]
        output_one_hot = F.one_hot(flat_outputs.reshape(batch * pairs, self.cell_count).long(), num_classes=self.config.color_count).to(torch.float32)[..., 1:]
        relation = torch.einsum("btc,bsc->bts", output_one_hot, input_one_hot).flatten(1)
        relation_readout = self.relation_projector(relation)
        return (relation_readout + 0.0 * transformer_readout).reshape(batch, pairs, self.config.hidden_size)

    def forward(
        self,
        query_grids: torch.Tensor,
        support_input_grids: torch.Tensor,
        support_output_grids: torch.Tensor,
        *,
        zero_support: bool = False,
    ) -> dict[str, torch.Tensor]:
        if query_grids.ndim != 3 or query_grids.shape[1:] != (self.config.grid_size, self.config.grid_size):
            raise ValueError("query_grids must be [batch,7,7]")
        pair_representations = self.encode_pairs(support_input_grids, support_output_grids)
        if zero_support:
            pair_representations_for_context = torch.zeros_like(pair_representations)
        else:
            pair_representations_for_context = pair_representations
        task_context = self.context_mlp(pair_representations_for_context.mean(dim=1))
        pointer_logits = self.pointer_generator(task_context).reshape(-1, self.cell_count, self.cell_count)
        pointer_logits = pointer_logits * self.logit_scale.exp().clamp(0.1, 100.0)
        pointer_attention = torch.softmax(pointer_logits, dim=-1)
        query_one_hot = F.one_hot(query_grids.reshape(query_grids.shape[0], self.cell_count).long(), num_classes=self.config.color_count).to(torch.float32)
        output_probs = torch.einsum("buv,bvc->buc", pointer_attention, query_one_hot)
        output_probs = output_probs.reshape(query_grids.shape[0], self.config.grid_size, self.config.grid_size, self.config.color_count)
        return {
            "output_probs": output_probs,
            "decoded_grid": output_probs.argmax(dim=-1),
            "task_context": task_context,
            "pair_representations": pair_representations,
            "pair_representations_for_context": pair_representations_for_context,
            "pointer_attention": pointer_attention.reshape(query_grids.shape[0], self.config.grid_size, self.config.grid_size, self.cell_count),
            "pointer_logits": pointer_logits,
        }

    def initialization_report(self) -> dict[str, Any]:
        return {
            "architecture": "SupportConditionedTransformPointerR3B0",
            "config": asdict(self.config),
            "pair_relation_encoder": "shared transformer over support-input/support-output cell tokens plus learned readout",
            "same_color_relation_projection": "pair-local output-position/input-position color correspondence projected into the task context",
            "support_pair_id_embeddings": False,
            "support_pair_position_embeddings": False,
            "task_aggregation": "mean(pair_representations) followed by bias-free MLP",
            "output_path": "query-copy pointer distribution only",
            "transform_labels_in_forward": False,
            "parameter_count": {
                "total": sum(parameter.numel() for parameter in self.parameters()),
                "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            },
        }


def transform_r3b0_output_loss(output_probs: torch.Tensor, target_grids: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gathered = output_probs.gather(-1, target_grids.long().unsqueeze(-1)).squeeze(-1)
    return -torch.log(gathered.clamp_min(eps)).mean()


def training_batch_keys() -> list[str]:
    return ["query_grids", "support_input_grids", "support_output_grids", "target_grids"]
