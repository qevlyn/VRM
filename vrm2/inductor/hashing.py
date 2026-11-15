from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_mapping_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    """Hash ordered tensor names, dtypes, shapes, and raw values."""

    digest = hashlib.sha256()
    for name, value in tensors.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_state_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {
        "model_state_sha256": tensor_mapping_sha256(module.state_dict()),
        "parameter_state_sha256": tensor_mapping_sha256(dict(module.named_parameters())),
        "buffer_state_sha256": tensor_mapping_sha256(dict(module.named_buffers())),
    }
