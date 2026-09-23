"""HistMUSK model construction."""

from __future__ import annotations

from typing import Any

from torch import nn

from .spot_anchored_fusion import SpotAnchoredCellTextFusion

# Public project name while retaining the historical class name for checkpoint
# compatibility with models trained in the research workspace.
HistMUSKModel = SpotAnchoredCellTextFusion


def build_model(config: dict[str, Any], dimensions: dict[str, int]) -> nn.Module:
    """Build HistMUSK after validating configured input dimensions."""
    model_config = config["model"]
    model_type = str(model_config.get("type", "histmusk")).lower()
    if model_type not in {"histmusk", "sactf", "catstraight"}:
        raise ValueError(
            f"Unsupported model.type={model_type!r}; expected histmusk, sactf, or catstraight"
        )

    expected_qwen_dim = model_config.get("expected_qwen_dim")
    expected_qwen_dims = model_config.get("expected_qwen_dims")
    if expected_qwen_dim is not None and expected_qwen_dims is not None:
        raise ValueError("Configure only one of expected_qwen_dim or expected_qwen_dims")
    if expected_qwen_dim is not None and int(expected_qwen_dim) != dimensions["qwen_dim"]:
        raise ValueError(
            "Configured model.expected_qwen_dim does not match the prepared dataset: "
            f"expected {int(expected_qwen_dim)}, found {dimensions['qwen_dim']}"
        )
    if expected_qwen_dims is not None:
        allowed = tuple(int(value) for value in expected_qwen_dims)
        if dimensions["qwen_dim"] not in allowed:
            raise ValueError(
                "Configured model.expected_qwen_dims does not include the prepared "
                f"dataset width: allowed {list(allowed)}, found {dimensions['qwen_dim']}"
            )

    text_fusion = (
        "catstraight"
        if model_type == "catstraight"
        else str(model_config.get("text_fusion", "concat"))
    )
    return HistMUSKModel(
        conch_dim=dimensions["conch_dim"],
        cell_dim=dimensions["cell_dim"],
        qwen_dim=dimensions["qwen_dim"],
        num_genes=dimensions["num_genes"],
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        num_cell_prototypes=int(model_config.get("num_cell_prototypes", 4)),
        num_heads=int(model_config.get("num_heads", 8)),
        dropout=float(model_config.get("dropout", 0.1)),
        cell_dropout=float(model_config.get("cell_dropout", 0.05)),
        text_dropout=float(model_config.get("text_dropout", 0.05)),
        enable_cell=bool(model_config.get("enable_cell", True)),
        enable_text=bool(model_config.get("enable_text", True)),
        use_cell_gate=bool(model_config.get("use_cell_gate", True)),
        use_text_gate=bool(model_config.get("use_text_gate", False)),
        gate_max=float(model_config.get("gate_max", 0.5)),
        cell_pooling=str(model_config.get("cell_pooling", "prototype")),
        cell_fusion=str(model_config.get("cell_fusion", "gated_add")),
        text_fusion=text_fusion,
    )


__all__ = ["HistMUSKModel", "SpotAnchoredCellTextFusion", "build_model"]
