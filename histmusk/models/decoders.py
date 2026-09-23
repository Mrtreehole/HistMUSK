"""Expression decoders."""

from __future__ import annotations

from torch import nn

from .layers import PortableLayerNorm


class ExpressionDecoder(nn.Sequential):
    def __init__(self, hidden_dim: int, num_genes: int, dropout: float = 0.1) -> None:
        super().__init__(
            PortableLayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_genes),
        )
