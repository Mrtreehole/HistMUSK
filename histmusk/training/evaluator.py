"""Prediction metrics, group summaries, and diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from ..models.losses import pearson_correlation
from ..utils.device import ResolvedDevice, autocast_context
from .trainer import move_batch


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: ResolvedDevice,
    mode: str = "full",
    mixed_precision: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[Any]] = {
        "prediction": [], "target": [], "cell_gate": [], "text_gate": [], "cell_count": [],
        "text_cosine": [], "cell_attention": [],
        "spot_index": [], "spot_id": [], "slide_id": [], "sample_id": [], "patient_id": [],
    }
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        metadata = {key: batch[key] for key in ("spot_id", "slide_id", "sample_id", "patient_id")}
        batch = move_batch(batch, device)
        with autocast_context(device, mixed_precision):
            outputs = model(batch, modality_mode=mode, apply_modality_dropout=False)
        if mode == "spot_only":
            outputs["prediction"] = outputs["spot_prediction"]
        collected["prediction"].append(outputs["prediction"].float().cpu())
        for key in ("cell_gate", "text_gate"):
            gate = outputs[key]
            if gate.ndim == 2 and gate.shape[1] > 1:
                gate = gate.mean(dim=1, keepdim=True)
            collected[key].append(gate.float().cpu())
        if outputs["spot_feature"].shape[-1] == outputs["text_feature"].shape[-1]:
            text_cosine = F.cosine_similarity(
                outputs["spot_feature"], outputs["text_feature"], dim=-1
            )
        else:
            # Raw catstraight text vectors intentionally remain in their source width.
            text_cosine = outputs["spot_feature"].new_full(
                (outputs["spot_feature"].shape[0],), float("nan")
            )
        collected["text_cosine"].append(text_cosine.float().cpu())
        collected["cell_attention"].append(outputs["cell_attention"].squeeze(1).float().cpu())
        collected["target"].append(batch["target"].float().cpu())
        collected["cell_count"].append(batch["cell_count"].cpu())
        collected["spot_index"].append(batch["spot_index"].cpu())
        for key, value in metadata.items():
            collected[key].extend(value)
    if not collected["prediction"]:
        raise RuntimeError("Prediction loader produced no batches")
    for key in ("prediction", "target", "cell_gate", "text_gate", "text_cosine", "cell_attention", "cell_count", "spot_index"):
        collected[key] = torch.cat(collected[key]).numpy()
    return collected


def compute_metrics(payload: dict[str, Any], gene_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    prediction = np.asarray(payload["prediction"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    spot_pcc = pearson_correlation(torch.from_numpy(prediction), torch.from_numpy(target), dim=1).numpy()
    gene_pcc = pearson_correlation(torch.from_numpy(prediction), torch.from_numpy(target), dim=0).numpy()
    valid_gene_pcc = (np.std(prediction, axis=0) > 0) & (np.std(target, axis=0) > 0)
    gene_spearman = np.array([
        spearmanr(prediction[:, index], target[:, index]).statistic
        if np.std(prediction[:, index]) > 0 and np.std(target[:, index]) > 0 else 0.0
        for index in range(prediction.shape[1])
    ])
    mse = float(np.mean(np.square(prediction - target)))
    cell_gate_values = np.asarray(payload["cell_gate"]).ravel()
    text_gate_values = np.asarray(payload["text_gate"]).ravel()
    cell_count_values = np.asarray(payload["cell_count"], dtype=np.float64)
    text_cosine = np.asarray(payload["text_cosine"], dtype=np.float64)
    alpha_count_corr = float(np.corrcoef(cell_gate_values, cell_count_values)[0, 1]) if np.std(cell_gate_values) > 0 else 0.0
    beta_cosine_corr = float(np.corrcoef(text_gate_values, text_cosine)[0, 1]) if np.std(text_gate_values) > 0 else 0.0
    metrics = {
        "mse": mse, "mae": float(np.mean(np.abs(prediction - target))),
        "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
        "spot_pcc_mean": float(np.mean(spot_pcc)), "spot_pcc_median": float(np.median(spot_pcc)),
        "spot_pcc_q25": float(np.quantile(spot_pcc, 0.25)), "spot_pcc_q75": float(np.quantile(spot_pcc, 0.75)),
        "gene_pcc_mean": float(np.mean(gene_pcc)), "gene_pcc_median": float(np.median(gene_pcc)),
        "gene_pcc_mean_valid": float(np.mean(gene_pcc[valid_gene_pcc])) if valid_gene_pcc.any() else float("nan"),
        "gene_pcc_median_valid": float(np.median(gene_pcc[valid_gene_pcc])) if valid_gene_pcc.any() else float("nan"),
        "gene_pcc_valid_genes": int(valid_gene_pcc.sum()),
        "gene_spearman_mean": float(np.nanmean(gene_spearman)),
        "cell_gate_mean": float(np.mean(payload["cell_gate"])),
        "text_gate_mean": float(np.mean(payload["text_gate"])),
        "cell_gate_cell_count_pearson": alpha_count_corr,
        "text_gate_text_cosine_pearson": beta_cosine_corr,
    }
    per_gene = pd.DataFrame({"gene": gene_names, "pearson": gene_pcc, "spearman": gene_spearman})
    per_spot = pd.DataFrame({
        "spot_index": payload["spot_index"], "spot_id": payload["spot_id"],
        "slide_id": payload["slide_id"], "patient_id": payload["patient_id"], "spot_pcc": spot_pcc,
        "mse": np.mean(np.square(prediction - target), axis=1),
        "cell_gate": payload["cell_gate"].ravel(), "text_gate": payload["text_gate"].ravel(),
        "cell_count": payload["cell_count"],
        "text_cosine": text_cosine,
    })
    return metrics, per_gene, per_spot


def compute_group_gene_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: list[Any] | np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    """Compute gene-wise PCC within each slide/patient without pooling groups."""
    group_values = np.asarray(groups, dtype=str)
    if len(group_values) != len(prediction):
        raise ValueError(f"{group_column} count does not match prediction rows")
    rows: list[dict[str, Any]] = []
    for group in sorted(value for value in np.unique(group_values) if value):
        selected = group_values == group
        group_prediction = prediction[selected]
        group_target = target[selected]
        valid = (
            (np.std(group_prediction, axis=0) > 0)
            & (np.std(group_target, axis=0) > 0)
        )
        correlations = pearson_correlation(
            torch.from_numpy(group_prediction), torch.from_numpy(group_target), dim=0
        ).numpy()
        valid_correlations = correlations[valid]
        rows.append(
            {
                group_column: group,
                "spots": int(selected.sum()),
                "genes": int(prediction.shape[1]),
                "valid_genes": int(valid.sum()),
                "invalid_genes": int((~valid).sum()),
                "valid_gene_fraction": float(valid.mean()),
                "gene_pcc_mean_valid": (
                    float(np.mean(valid_correlations)) if len(valid_correlations) else float("nan")
                ),
                "gene_pcc_median_valid": (
                    float(np.median(valid_correlations)) if len(valid_correlations) else float("nan")
                ),
                "gene_pcc_q25_valid": (
                    float(np.quantile(valid_correlations, 0.25))
                    if len(valid_correlations)
                    else float("nan")
                ),
                "gene_pcc_q75_valid": (
                    float(np.quantile(valid_correlations, 0.75))
                    if len(valid_correlations)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def save_evaluation(payload: dict[str, Any], gene_names: list[str], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(payload["prediction"])
    target = np.asarray(payload["target"])
    metrics, per_gene, per_spot = compute_metrics(payload, gene_names)
    per_slide_gene = compute_group_gene_metrics(
        prediction, target, payload["slide_id"], "slide_id"
    )
    if not per_slide_gene.empty:
        slide_values = per_slide_gene["gene_pcc_mean_valid"].dropna().to_numpy()
        metrics.update(
            {
                "slide_gene_pcc_macro_mean_valid": (
                    float(np.mean(slide_values)) if len(slide_values) else float("nan")
                ),
                "slide_gene_pcc_macro_median_valid": (
                    float(np.median(slide_values)) if len(slide_values) else float("nan")
                ),
                "slide_gene_pcc_valid_slides": int(len(slide_values)),
                "slide_gene_pcc_valid_gene_fraction_mean": float(
                    per_slide_gene["valid_gene_fraction"].mean()
                ),
            }
        )
    per_slide_gene.to_csv(output_dir / "per_slide_gene_metrics.csv", index=False)
    per_patient_gene = compute_group_gene_metrics(
        prediction, target, payload["patient_id"], "patient_id"
    )
    if not per_patient_gene.empty:
        patient_values = per_patient_gene["gene_pcc_mean_valid"].dropna().to_numpy()
        metrics.update(
            {
                "patient_gene_pcc_macro_mean_valid": (
                    float(np.mean(patient_values)) if len(patient_values) else float("nan")
                ),
                "patient_gene_pcc_macro_median_valid": (
                    float(np.median(patient_values)) if len(patient_values) else float("nan")
                ),
                "patient_gene_pcc_valid_patients": int(len(patient_values)),
            }
        )
        per_patient_gene.to_csv(output_dir / "per_patient_gene_metrics.csv", index=False)
    per_gene.to_csv(output_dir / "per_gene_metrics.csv", index=False)
    per_spot.to_parquet(output_dir / "per_spot_metrics.parquet", index=False)
    per_slide = per_spot.groupby("slide_id").agg(
        spots=("spot_index", "size"), mse=("mse", "mean"), spot_pcc=("spot_pcc", "mean"),
        cell_gate=("cell_gate", "mean"), text_gate=("text_gate", "mean"),
    ).reset_index()
    per_slide.to_csv(output_dir / "per_slide_metrics.csv", index=False)
    if per_spot["patient_id"].astype(str).str.len().gt(0).any():
        per_spot[per_spot["patient_id"].astype(str).str.len().gt(0)].groupby("patient_id").agg(
            spots=("spot_index", "size"), mse=("mse", "mean"), spot_pcc=("spot_pcc", "mean")
        ).reset_index().to_csv(output_dir / "per_patient_metrics.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    gate_stats = pd.DataFrame({
        "gate": ["cell", "text"],
        "mean": [np.mean(payload["cell_gate"]), np.mean(payload["text_gate"])],
        "std": [np.std(payload["cell_gate"]), np.std(payload["text_gate"])],
        "min": [np.min(payload["cell_gate"]), np.min(payload["text_gate"])],
        "max": [np.max(payload["cell_gate"]), np.max(payload["text_gate"])],
        "reliability_correlation": [metrics["cell_gate_cell_count_pearson"], metrics["text_gate_text_cosine_pearson"]],
    })
    gate_stats.to_csv(output_dir / "gate_statistics.csv", index=False)
    for values, title, name in [
        (per_gene["pearson"], "Gene-wise PCC", "gene_pcc_distribution.png"),
        (per_spot["spot_pcc"], "Spot-wise PCC", "spot_pcc_distribution.png"),
        (per_spot["cell_gate"], "Cell gate", "cell_gate_distribution.png"),
        (per_spot["text_gate"], "Text gate", "text_gate_distribution.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=40)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(per_spot["cell_count"], per_spot["cell_gate"], s=3, alpha=0.25)
    ax.set(xlabel="Cell count", ylabel="Cell gate")
    fig.tight_layout()
    fig.savefig(output_dir / "gate_vs_cell_count.png", dpi=160)
    plt.close(fig)
    attention = np.asarray(payload["cell_attention"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(attention.ravel(), bins=40)
    ax.set(xlabel="Attention weight", ylabel="Count", title="Spot-to-cell prototype attention")
    fig.tight_layout()
    fig.savefig(output_dir / "cell_prototype_attention_distribution.png", dpi=160)
    plt.close(fig)
    selected = np.argsort(per_gene["pearson"].to_numpy())[-6:]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, gene_index in zip(axes.ravel(), selected):
        axis.scatter(target[:, gene_index], prediction[:, gene_index], s=3, alpha=0.25)
        axis.set_title(gene_names[gene_index])
        axis.set(xlabel="Truth", ylabel="Prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_vs_truth_selected_genes.png", dpi=160)
    plt.close(fig)
    return metrics
