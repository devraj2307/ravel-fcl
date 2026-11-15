"""Device and RNG helpers shared by the training code.

PyTorch random generators are bound to a device: a generator created for the
CPU cannot be passed to ``torch.randn(..., device="cuda")`` (and vice versa).
These helpers keep the device selection and generator construction consistent
across the client, server, defense and attack modules.
"""

from __future__ import annotations

from typing import Union

import torch

DeviceLike = Union[str, torch.device]


def resolve_device(device: DeviceLike) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise return ``device``."""
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device requests CUDA but torch.cuda.is_available() is False"
        )
    return resolved


def make_generator(seed: int, device: DeviceLike = "cpu") -> torch.Generator:
    """Create a seeded generator on the same device as the tensors it feeds."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cannot create a CUDA generator without CUDA support")
    return torch.Generator(device=resolved).manual_seed(int(seed))
