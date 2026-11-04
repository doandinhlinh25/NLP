"""Sinusoidal positional encodings used by the LAB 07 Transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_positional_encoding(sequence_length: int, model_dimension: int, *, device: torch.device | None = None) -> Tensor:
    """Return a ``(1, sequence_length, model_dimension)`` sinusoidal encoding.

    Args:
        sequence_length: Number of token positions to represent.
        model_dimension: Number of features per token; must be positive.
        device: Optional target device for the returned tensor.

    Returns:
        Positional encodings with alternating sine and cosine features.
    """
    if sequence_length < 0 or model_dimension <= 0:
        raise ValueError("sequence_length must be non-negative and model_dimension must be positive")
    positions = torch.arange(sequence_length, dtype=torch.float32, device=device).unsqueeze(1)
    feature_indices = torch.arange(model_dimension, dtype=torch.float32, device=device)
    frequencies = torch.exp(-math.log(10000.0) * (2 * torch.floor(feature_indices / 2)) / model_dimension)
    angles = positions * frequencies.unsqueeze(0)
    encodings = torch.where((feature_indices.to(torch.long) % 2) == 0, torch.sin(angles), torch.cos(angles))
    return encodings.unsqueeze(0)
