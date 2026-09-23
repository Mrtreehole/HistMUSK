"""Lightweight CellViT prototype pooling and spot-to-cell attention."""

from __future__ import annotations

import torch
from torch import nn

from .layers import PortableLayerNorm


class CellPrototypePool(nn.Module):
    def __init__(
        self,
        cell_dim: int = 1280,
        hidden_dim: int = 256,
        num_prototypes: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            PortableLayerNorm(cell_dim), nn.Linear(cell_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.prototype_queries = nn.Parameter(torch.empty(1, num_prototypes, hidden_dim))
        self.empty_cell_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        nn.init.trunc_normal_(self.prototype_queries, std=0.02)
        nn.init.trunc_normal_(self.empty_cell_token, std=0.02)

    def forward(self, cells: torch.Tensor, cell_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cells.ndim != 3 or cell_mask.shape != cells.shape[:2]:
            raise ValueError("Expected cells [B,N,D] and cell_mask [B,N]")
        projected = self.project(cells)
        valid_mask = cell_mask.clone()
        empty = ~valid_mask.any(dim=1)
        if empty.any():
            projected = projected.clone()
            valid_mask = valid_mask.clone()
            empty_token = self.empty_cell_token.to(projected)
            projected[empty, 0] = empty_token[0, 0]
            valid_mask[empty, 0] = True
        queries = self.prototype_queries.to(projected).expand(cells.shape[0], -1, -1)
        prototypes, weights = self.attention(
            queries, projected, projected, key_padding_mask=~valid_mask, need_weights=True
        )
        return prototypes, weights


class SpotToCellAttention(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, spot: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, weights = self.attention(spot[:, None, :], prototypes, prototypes, need_weights=True)
        return residual[:, 0, :], weights
