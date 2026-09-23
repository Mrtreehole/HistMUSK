"""Finite-safe expression and correlation losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


LOSS_PROFILES: dict[str, dict[str, float]] = {
    "reduced_aux_consistency": {
        "spot_pcc_weight": 0.1,
        "gene_pcc_weight": 0.1,
        "spot_aux_weight": 0.05,
        "spot_aux_pcc_weight": 0.2,
        "residual_weight": 0.0,
        "consistency_weight": 0.005,
    }
}


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=dim)
    denominator_squared = (
        x_centered.square().sum(dim=dim) * y_centered.square().sum(dim=dim)
    )
    valid = denominator_squared > eps * eps
    denominator = torch.sqrt(denominator_squared.clamp_min(eps * eps))
    safe = numerator / denominator
    return torch.where(valid, safe.clamp(-1.0, 1.0), torch.zeros_like(safe))


def spot_pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - pearson_correlation(prediction, target, dim=1).mean()


def gene_pearson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - pearson_correlation(prediction, target, dim=0).mean()


class SACTFLoss(nn.Module):
    def __init__(
        self,
        spot_pcc_weight: float = 0.1,
        gene_pcc_weight: float = 0.1,
        spot_aux_weight: float = 0.05,
        spot_aux_pcc_weight: float = 0.2,
        residual_weight: float = 0.0,
        consistency_weight: float = 0.005,
        text_residual_weight: float = 0.0,
        text_aux_weight: float = 0.0,
        text_aux_gene_pcc_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.spot_pcc_weight = spot_pcc_weight
        self.gene_pcc_weight = gene_pcc_weight
        self.spot_aux_weight = spot_aux_weight
        self.spot_aux_pcc_weight = spot_aux_pcc_weight
        self.residual_weight = residual_weight
        self.consistency_weight = consistency_weight
        self.text_residual_weight = text_residual_weight
        self.text_aux_weight = text_aux_weight
        self.text_aux_gene_pcc_weight = text_aux_gene_pcc_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        dropped_prediction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = outputs["prediction"]
        mse = F.mse_loss(prediction, target)
        spot_pcc = spot_pearson_loss(prediction, target)
        gene_pcc = gene_pearson_loss(prediction, target)
        spot_aux = F.mse_loss(outputs["spot_prediction"], target)
        spot_aux = spot_aux + self.spot_aux_pcc_weight * spot_pearson_loss(outputs["spot_prediction"], target)
        if "residual_prediction" in outputs:
            residual = (
                outputs["text_gate"] * outputs["residual_prediction"]
            ).square().mean()
        else:
            cell_term = outputs["cell_gate"] * outputs["cell_residual"]
            text_term = outputs["text_gate"] * outputs["text_feature"]
            residual = cell_term.square().mean() + text_term.square().mean()
        consistency = prediction.new_zeros(())
        if dropped_prediction is not None:
            consistency = F.mse_loss(dropped_prediction, prediction.detach())
        text_residual = prediction.new_zeros(())
        text_aux = prediction.new_zeros(())
        if "residual_prediction" in outputs:
            residual_target = target - outputs["base_prediction"].detach()
            text_residual = F.mse_loss(outputs["residual_prediction"], residual_target)
        if "text_aux_prediction" in outputs:
            text_aux = F.mse_loss(outputs["text_aux_prediction"], target)
            if self.text_aux_gene_pcc_weight > 0:
                text_aux = text_aux + self.text_aux_gene_pcc_weight * gene_pearson_loss(
                    outputs["text_aux_prediction"], target
                )
        total = (
            mse + self.spot_pcc_weight * spot_pcc + self.gene_pcc_weight * gene_pcc
            + self.spot_aux_weight * spot_aux
            + self.residual_weight * residual + self.consistency_weight * consistency
            + self.text_residual_weight * text_residual
            + self.text_aux_weight * text_aux
        )
        values = {
            "loss": total, "mse": mse, "spot_pcc_loss": spot_pcc,
            "gene_pcc_loss": gene_pcc, "spot_aux_loss": spot_aux, "residual_loss": residual,
            "consistency_loss": consistency,
            "text_residual_loss": text_residual,
            "text_aux_loss": text_aux,
        }
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise FloatingPointError("Non-finite HistMUSK loss")
        return values


def build_loss(config: dict[str, Any]) -> SACTFLoss:
    profile_name = str(config.get("profile", "reduced_aux_consistency"))
    if profile_name not in LOSS_PROFILES:
        raise ValueError(
            f"Unknown loss profile {profile_name!r}; available={sorted(LOSS_PROFILES)}"
        )
    values = {
        **LOSS_PROFILES[profile_name],
        **{key: value for key, value in config.items() if key != "profile"},
    }
    return SACTFLoss(
        spot_pcc_weight=float(values["spot_pcc_weight"]),
        gene_pcc_weight=float(values["gene_pcc_weight"]),
        spot_aux_weight=float(values["spot_aux_weight"]),
        spot_aux_pcc_weight=float(values["spot_aux_pcc_weight"]),
        residual_weight=float(values["residual_weight"]),
        consistency_weight=float(values["consistency_weight"]),
        text_residual_weight=float(values.get("text_residual_weight", 0.0)),
        text_aux_weight=float(values.get("text_aux_weight", 0.0)),
        text_aux_gene_pcc_weight=float(values.get("text_aux_gene_pcc_weight", 0.0)),
    )
