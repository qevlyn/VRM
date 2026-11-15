from __future__ import annotations

import torch
from torch.nn import functional as F

from vrm.data import ARC_COLOR_COUNT

from .composition_r4a0r_data import GRID_SIZE, apply_sequence_to_coord


def apply_sequence_to_grid(grid: torch.Tensor, sequence_class: str) -> torch.Tensor:
    if grid.shape[-2:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError("R4A-0R oracle expects 7x7 grids")
    result = torch.zeros_like(grid)
    flat_prefix = grid.shape[:-2]
    source = grid.reshape(-1, GRID_SIZE, GRID_SIZE)
    target = result.reshape(-1, GRID_SIZE, GRID_SIZE)
    for batch_index in range(source.shape[0]):
        positions = (source[batch_index] != 0).nonzero(as_tuple=False)
        for row, col in positions.tolist():
            out_x, out_y = apply_sequence_to_coord(sequence_class, (int(col), int(row)))
            if target[batch_index, out_y, out_x] != 0:
                raise ValueError("sequence produced colliding foreground cells")
            target[batch_index, out_y, out_x] = source[batch_index, row, col]
    return result.reshape(*flat_prefix, GRID_SIZE, GRID_SIZE)


def oracle_probs(grid: torch.Tensor, sequence_class: str) -> torch.Tensor:
    output = apply_sequence_to_grid(grid, sequence_class)
    return F.one_hot(output.long(), num_classes=ARC_COLOR_COUNT).to(torch.float32)
