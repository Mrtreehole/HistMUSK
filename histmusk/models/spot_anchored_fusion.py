"""HistMUSK spot-anchored image, cell, and text fusion network."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .cell_prototype_pool import CellPrototypePool, SpotToCellAttention
from .decoders import ExpressionDecoder
from .layers import PortableLayerNorm


def _encoder(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        PortableLayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
        nn.Dropout(dropout), PortableLayerNorm(hidden_dim),
    )


class ScalarGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, max_value: float = 0.5) -> None:
        super().__init__()
        self.max_value = max_value
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.max_value * torch.sigmoid(self.network(inputs))


class SpotAnchoredCellTextFusion(nn.Module):
    def __init__(
        self,
        conch_dim: int,
        cell_dim: int,
        qwen_dim: int,
        num_genes: int,
        hidden_dim: int = 256,
        num_cell_prototypes: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        cell_dropout: float = 0.10,
        text_dropout: float = 0.30,
        enable_cell: bool = True,
        enable_text: bool = True,
        use_cell_gate: bool = True,
        use_text_gate: bool = True,
        gate_max: float = 0.5,
        cell_pooling: str = "prototype",
        cell_fusion: str = "gated_add",
        text_fusion: str = "gated_add",
    ) -> None:
        super().__init__()
        if cell_fusion not in {"gated_add", "concat"}:
            raise ValueError(f"Unknown cell_fusion: {cell_fusion}")
        if text_fusion not in {"gated_add", "concat", "catstraight"}:
            raise ValueError(f"Unknown text_fusion: {text_fusion}")
        self.hidden_dim = hidden_dim
        self.enable_cell = enable_cell
        self.enable_text = enable_text
        self.use_cell_gate = use_cell_gate
        self.use_text_gate = use_text_gate
        self.cell_dropout = cell_dropout
        self.text_dropout = text_dropout
        self.gate_max = gate_max
        self.cell_pooling = cell_pooling
        self.cell_fusion = cell_fusion
        self.text_fusion = text_fusion
        self.spot_encoder = _encoder(conch_dim, hidden_dim, dropout)
        self.qwen_dim = qwen_dim
        self.text_encoder = (
            None if text_fusion == "catstraight" else _encoder(qwen_dim, hidden_dim, dropout)
        )
        self.cell_pool = CellPrototypePool(cell_dim, hidden_dim, num_cell_prototypes, num_heads, dropout)
        self.cell_mean_project = _encoder(cell_dim, hidden_dim, dropout)
        self.spot_to_cell = SpotToCellAttention(hidden_dim, num_heads, dropout)
        self.cell_gate = ScalarGate(hidden_dim * 3 + 1, hidden_dim, dropout, gate_max)
        self.cell_concat_fusion = (
            _encoder(hidden_dim * 2, hidden_dim, dropout)
            if cell_fusion == "concat"
            else None
        )
        self.text_gate = (
            ScalarGate(hidden_dim * 4, hidden_dim, dropout, gate_max)
            if text_fusion == "gated_add"
            else None
        )
        self.fusion_norm = PortableLayerNorm(hidden_dim)
        self.text_concat_fusion = (
            _encoder(hidden_dim * 2, hidden_dim, dropout)
            if text_fusion == "concat"
            else None
        )
        self.text_straight_fusion = (
            _encoder(hidden_dim + qwen_dim, hidden_dim, dropout)
            if text_fusion == "catstraight"
            else None
        )
        self.ffn = nn.Sequential(
            PortableLayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.spot_decoder = ExpressionDecoder(hidden_dim, num_genes, dropout)
        self.fusion_decoder = ExpressionDecoder(hidden_dim, num_genes, dropout)

    def _cell_branch(
        self, spot: torch.Tensor, cells: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cell_pooling == "prototype":
            prototypes, _ = self.cell_pool(cells, mask)
        elif self.cell_pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(cells.dtype)
            pooled = (cells * mask[..., None]).sum(dim=1) / denom
            prototypes = self.cell_mean_project(pooled)[:, None, :]
        else:
            raise ValueError(f"Unknown cell_pooling: {self.cell_pooling}")
        return self.spot_to_cell(spot, prototypes)

    def forward(
        self,
        batch: dict[str, Any],
        modality_mode: str = "full",
        apply_modality_dropout: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if modality_mode not in {"full", "no_cell", "no_text", "spot_only"}:
            raise ValueError(f"Unknown modality mode: {modality_mode}")
        spot = self.spot_encoder(batch["conch_embedding"])
        batch_size = spot.shape[0]
        zeros = spot.new_zeros((batch_size, self.hidden_dim))
        raw_text_zeros = spot.new_zeros((batch_size, self.qwen_dim))
        empty_attention = spot.new_zeros((batch_size, 1, 1))
        cell_count = batch["cell_count"].to(spot.dtype)

        cell_active = self.enable_cell and modality_mode not in {"no_cell", "spot_only"}
        if cell_active:
            cell_residual, cell_attention = self._cell_branch(
                spot, batch["cell_embeddings"], batch["cell_mask"]
            )
            if self.cell_fusion == "gated_add":
                gate_input = torch.cat(
                    [
                        spot,
                        cell_residual,
                        (spot - cell_residual).abs(),
                        torch.log1p(cell_count)[:, None],
                    ],
                    dim=-1,
                )
                alpha = (
                    self.cell_gate(gate_input)
                    if self.use_cell_gate
                    else spot.new_full((batch_size, 1), self.gate_max)
                )
            else:
                # In concat mode alpha is an availability/dropout mask, not a learned gate.
                alpha = spot.new_ones((batch_size, 1))
            # DirectML autograd cannot multiply a differentiable float tensor by a bool tensor.
            alpha = alpha * cell_count.gt(0).to(dtype=alpha.dtype)[:, None]
        else:
            cell_residual, cell_attention = zeros, empty_attention
            alpha = spot.new_zeros((batch_size, 1))

        if self.enable_text and modality_mode not in {"no_text", "spot_only"}:
            if self.text_fusion == "catstraight":
                # Preserve the original binary/continuous embedding values until fusion.
                text = batch["qwen_embedding"].to(dtype=spot.dtype)
            else:
                assert self.text_encoder is not None
                text = self.text_encoder(batch["qwen_embedding"])
            if self.text_fusion in {"concat", "catstraight"}:
                # In concat mode beta is an availability/dropout mask, not a learned gate.
                beta = spot.new_ones((batch_size, 1))
            else:
                gate_input = torch.cat([spot, text, (spot - text).abs(), spot * text], dim=-1)
                if self.use_text_gate:
                    assert self.text_gate is not None
                    beta = self.text_gate(gate_input)
                else:
                    beta = spot.new_full((batch_size, 1), self.gate_max)
        else:
            text = raw_text_zeros if self.text_fusion == "catstraight" else zeros
            beta = spot.new_zeros((batch_size, 1))

        do_dropout = self.training if apply_modality_dropout is None else apply_modality_dropout
        if do_dropout:
            if self.cell_dropout > 0:
                keep = torch.rand((batch_size, 1), device=spot.device).ge(self.cell_dropout).to(alpha.dtype)
                alpha = alpha * keep
            if self.text_dropout > 0:
                keep = torch.rand((batch_size, 1), device=spot.device).ge(self.text_dropout).to(beta.dtype)
                beta = beta * keep
        if self.cell_fusion == "concat" and cell_active:
            assert self.cell_concat_fusion is not None
            spot_cell = self.cell_concat_fusion(
                torch.cat([spot, alpha * cell_residual], dim=-1)
            )
        else:
            spot_cell = spot + alpha * cell_residual

        if self.text_fusion == "catstraight":
            assert self.text_straight_fusion is not None
            fused = self.text_straight_fusion(
                torch.cat([self.fusion_norm(spot_cell), beta * text], dim=-1)
            )
        elif self.text_fusion == "concat":
            assert self.text_concat_fusion is not None
            fused = self.text_concat_fusion(
                torch.cat([self.fusion_norm(spot_cell), beta * text], dim=-1)
            )
        else:
            fused = self.fusion_norm(spot_cell + beta * text)
        fused = fused + self.ffn(fused)
        return {
            "prediction": self.fusion_decoder(fused),
            "spot_prediction": self.spot_decoder(spot),
            "spot_feature": spot,
            "fused_feature": fused,
            "cell_residual": cell_residual,
            "text_feature": text,
            "cell_gate": alpha,
            "text_gate": beta,
            "cell_attention": cell_attention,
        }
