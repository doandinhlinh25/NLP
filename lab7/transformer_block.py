"""A minimal, explicitly assembled Transformer encoder block for LAB 07."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TransformerBlock(nn.Module):
    """Apply multi-head self-attention, residual norms, and a position-wise FFN."""

    def __init__(self, model_dimension: int, number_of_heads: int, feedforward_dimension: int, dropout: float = 0.1) -> None:
        """Create learned Q/K/V projections and the two encoder sublayers."""
        super().__init__()
        if model_dimension <= 0 or number_of_heads <= 0 or feedforward_dimension <= 0:
            raise ValueError("model and feed-forward dimensions and head count must be positive")
        if model_dimension % number_of_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_heads")
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = model_dimension // number_of_heads
        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(model_dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dimension, feedforward_dimension),
            nn.GELU(),
            nn.Linear(feedforward_dimension, model_dimension),
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(model_dimension)

    def forward(self, inputs: Tensor, padding_mask: Tensor | None = None, *, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Transform ``(batch, time, features)`` input and optionally return weights.

        Args:
            inputs: Floating point tensor shaped ``(B, T, d_model)``.
            padding_mask: Optional boolean ``(B, T)`` mask; ``True`` marks padding.
            return_attention: Whether to return attention maps shaped ``(B, H, T, T)``.

        Returns:
            Encoded tokens, optionally paired with the attention probability tensor.
        """
        batch_size, sequence_length, feature_count = inputs.shape
        if feature_count != self.model_dimension:
            raise ValueError("last input dimension must equal model_dimension")
        if padding_mask is not None and padding_mask.shape != (batch_size, sequence_length):
            raise ValueError("padding_mask must have shape (batch, sequence_length)")
        query = self._split_heads(self.query_projection(inputs))
        key = self._split_heads(self.key_projection(inputs))
        value = self._split_heads(self.value_projection(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.attention_dropout(attention), value)
        merged_context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dimension)
        attention_output = self.attention_norm(inputs + self.output_projection(merged_context))
        output = self.feedforward_norm(attention_output + self.feedforward_dropout(self.feedforward(attention_output)))
        if return_attention:
            return output, attention
        return output

    def _split_heads(self, projected: Tensor) -> Tensor:
        """Reshape projected features to ``(B, H, T, d_k)``."""
        batch_size, sequence_length, _ = projected.shape
        return projected.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension).transpose(1, 2)
