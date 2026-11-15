"""VRM: Variable-memory Recursive Model."""

from .data import (
    COLOR_IDS,
    DEFAULT_ARC_DATA_ROOT,
    MAX_GRID_HEIGHT,
    MAX_GRID_TOKENS,
    MAX_GRID_WIDTH,
    PAD_ID,
    VOCAB_SIZE,
)
from .model import VRM, VRMConfig

__all__ = [
    "COLOR_IDS",
    "DEFAULT_ARC_DATA_ROOT",
    "MAX_GRID_HEIGHT",
    "MAX_GRID_TOKENS",
    "MAX_GRID_WIDTH",
    "PAD_ID",
    "VOCAB_SIZE",
    "VRM",
    "VRMConfig",
]
