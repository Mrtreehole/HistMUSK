#!/usr/bin/env python3
"""Audit multimodal arrays, prepare aligned expression targets, and write QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from histmusk.data.target_processing import prepare_expression_targets


def array_statistics(array: Any, row_chunk: int = 4096) -> dict[str, Any]:
    count = 0
    total = 0.0
    total_square = 0.0
    finite = True
    zero_rows = 0
    norms: list[np.ndarray] = []
    for start in range(0, array.shape[0], row_chunk):
        block = np.asarray(array[start : start + row_chunk])
        finite &= bool(np.isfinite(block).all())
        values = block.astype(np.float64, copy=False)
        count += values.size
        total += values.sum(dtype=np.float64)
        total_square += np.square(values).sum(dtype=np.float64)
        if block.ndim == 2:
            block_norms = np.linalg.norm(values, axis=1)
            norms.append(block_norms.astype(np.float32))
            zero_rows += int(np.count_nonzero(block_norms == 0))
    mean = total / max(count, 1)
    std = np.sqrt(max(total_square / max(count, 1) - mean * mean, 0.0))
    norm_values = np.concatenate(norms) if norms else np.empty(0, dtype=np.float32)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype), "finite": finite,
        "mean": float(mean), "std": float(std), "zero_rows": zero_rows,
        "norm_mean": float(norm_values.mean()) if len(norm_values) else None,
        "norm_std": float(norm_values.std()) if len(norm_values) else None,
        "norm_quantiles": np.quantile(norm_values, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(norm_values) else None,
        "_norms": norm_values,
    }


def inspect(
    data_dir: Path,
    expression_h5ad: Path,
    target_dir: Path,
    overwrite: bool,
    qc_dir: Path | None = None,
) -> dict[str, Any]:
    qc_dir = (qc_dir or target_dir / "qc").resolve()
    target_manifest = prepare_expression_targets(data_dir, expression_h5ad, target_dir, 42, overwrite)
    qc_dir.mkdir(parents=True, exist_ok=True)
    spots = pd.read_parquet(data_dir / "spots.parquet").sort_values("spot_index").reset_index(drop=True)
    splits = pd.read_parquet(target_dir / "splits.parquet").sort_values("spot_index").reset_index(drop=True)
    target_index = pd.read_parquet(target_dir / "target_index.parquet").sort_values("spot_index").reset_index(drop=True)
    features = zarr.open_group(str(data_dir / "features.zarr"), mode="r")
    targets = zarr.open_group(str(target_dir / "gene_expression.zarr"), mode="r")
    arrays = {name: {"shape": list(features[name].shape), "dtype": str(features[name].dtype)} for name in features.array_keys()}
    n_spots = len(spots)
    indptr = features["cell_indptr"][:]
    assigned = int(indptr[-1])
    total_cells = int(features["cell_embeddings"].shape[0])
    cell_map = features["cell_to_spot_index"]
    prefix_map = cell_map[:assigned]
    tail_valid = True
    for start in range(assigned, total_cells, 65536):
        tail_valid &= bool(np.all(cell_map[start : start + 65536] == -1))
    source_checks = {
        "spot_ids_unique": bool(spots["spot_id"].is_unique),
        "spot_index_contiguous": bool(np.array_equal(spots["spot_index"], np.arange(n_spots))),
        "cell_indptr_length": len(indptr) == n_spots + 1,
        "cell_indptr_starts_zero": int(indptr[0]) == 0,
        "cell_indptr_monotonic": bool(np.all(np.diff(indptr) >= 0)),
        "cell_indptr_ends_assigned_prefix": assigned <= total_cells,
        "assigned_prefix_nonnegative": bool(np.all(prefix_map >= 0)),
        "unassigned_tail_is_minus_one": tail_valid,
        "spot_cell_counts_match_indptr": bool(np.array_equal(spots["cell_count"].to_numpy(), np.diff(indptr))),
        "conch_spot_count": features["conch_embeddings"].shape[0] == n_spots,
        "qwen_spot_count": features["qwen_embeddings"].shape[0] == n_spots,
        "target_spot_count": targets["log_normalized"].shape[0] == n_spots,
        "target_gene_count": targets["log_normalized"].shape[1] == target_manifest["genes"],
        "target_raw_log_shapes_match": targets["raw_counts"].shape == targets["log_normalized"].shape,
        "target_index_spot_ids_match": bool(spots["spot_id"].equals(target_index["spot_id"])),
        "target_source_rows_unique": bool(
            target_index.loc[
                target_index.get("target_available", pd.Series(True, index=target_index.index)).astype(bool),
                "h5ad_source_row",
            ].is_unique
        ),
        "target_unavailable_rows_are_minus_one": bool(
            (
                target_index.loc[
                    ~target_index.get(
                        "target_available", pd.Series(True, index=target_index.index)
                    ).astype(bool),
                    "h5ad_source_row",
                ]
                == -1
            ).all()
        ),
        "split_spot_ids_match": bool(spots["spot_id"].equals(splits["spot_id"])),
    }
    if not all(source_checks.values()):
        failed = [name for name, passed in source_checks.items() if not passed]
        raise RuntimeError(f"Critical training-data checks failed: {failed}")

    stats = {}
    for name in ("conch_embeddings", "qwen_embeddings", "cell_embeddings"):
        print(f"Computing full statistics for {name}...", flush=True)
        stats[name] = array_statistics(features[name])
    print("Computing full target statistics...", flush=True)
    stats["raw_counts"] = array_statistics(targets["raw_counts"])
    stats["log_normalized"] = array_statistics(targets["log_normalized"])
    if not all(item["finite"] for item in stats.values()):
        raise FloatingPointError("Non-finite value in training arrays")

    split_summary = splits.groupby("split").agg(
        spots=("spot_index", "size"), slides=("slide_id", "nunique"), patients=("patient_id", "nunique")
    ).to_dict(orient="index")
    active_splits = splits[splits["split"].isin(["train", "val", "test"])]
    leakage = int(active_splits.groupby("slide_id")["split"].nunique().gt(1).sum())
    if leakage:
        raise RuntimeError(f"Slide leakage found in {leakage} slides")
    cell_distribution = spots["cell_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spots["cell_count"], bins=60)
    ax.set(xlabel="Assigned cells per spot", ylabel="Spots")
    fig.tight_layout()
    fig.savefig(qc_dir / "cell_count_histogram.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name in zip(axes, ("conch_embeddings", "qwen_embeddings", "cell_embeddings")):
        norm_values = stats[name]["_norms"]
        axis.hist(norm_values, bins=60)
        axis.set_title(name.replace("_embeddings", ""))
        axis.set_xlabel("L2 norm")
        if np.std(norm_values) < 1e-5:
            center = float(np.mean(norm_values))
            axis.axvline(center, color="tab:red", linewidth=3)
            axis.set_xlim(center - 1e-5, center + 1e-5)
            axis.text(0.03, 0.92, f"all values ~ {center:.6f}", transform=axis.transAxes)
        elif name == "cell_embeddings":
            axis.set_xlim(float(np.quantile(norm_values, 0.001)), float(np.quantile(norm_values, 0.995)))
    fig.tight_layout()
    fig.savefig(qc_dir / "embedding_norms.png", dpi=170)
    plt.close(fig)
    raw_sample = targets["raw_counts"][:: max(1, n_spots // 5000)].ravel()
    log_sample = targets["log_normalized"][:: max(1, n_spots // 5000)].ravel()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(raw_sample, bins=60, log=True)
    axes[0].set_title("Selected-HVG raw counts")
    axes[1].hist(log_sample, bins=60)
    axes[1].set_title("Log-normalized expression")
    fig.tight_layout()
    fig.savefig(qc_dir / "expression_distribution.png", dpi=170)
    plt.close(fig)

    for item in stats.values():
        item.pop("_norms", None)
    report = {
        "status": "passed", "data_dir": str(data_dir.resolve()),
        "target_dir": str(target_dir.resolve()), "arrays": arrays,
        "n_spots": n_spots, "total_cells": total_cells, "assigned_cells": assigned,
        "unassigned_cells": total_cells - assigned,
        "source_checks": source_checks, "statistics": stats,
        "cell_count_distribution": cell_distribution,
        "expression": target_manifest, "split_summary": split_summary,
        "split_group": target_manifest["group_column"], "slide_leakage_count": leakage,
        "cell_csr_note": (
            "cell_embeddings stores all cells; cell_indptr addresses the assigned prefix and the remaining tail "
            "is retained with cell_to_spot_index=-1"
        ),
    }
    (qc_dir / "training_data_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Training data report", "", "Status: **PASSED**", "",
        f"- Spots: {n_spots}",
        f"- Slides / split group: {spots.slide_id.nunique()} / {target_manifest['group_column']}",
        f"- Cells: {total_cells} total, {assigned} assigned, {total_cells-assigned} retained unassigned",
        f"- Image embeddings: {arrays['conch_embeddings']['shape']}",
        f"- Qwen: {arrays['qwen_embeddings']['shape']}",
        f"- CellViT: {arrays['cell_embeddings']['shape']}",
        f"- Expression: {target_manifest['target_shapes']['log_normalized']}, supplied H5AD gene panel",
        f"- Expression target-available spots: {target_manifest['alignment']['target_available_spots']}",
        f"- Expression target-unavailable spots excluded: {target_manifest['alignment']['missing_spot_ids']}",
        f"- Expression source: {target_manifest['source_h5ad']['path']}",
        f"- Explicitly reordered H5AD rows: {target_manifest['alignment']['moved_rows']}",
        f"- All-zero target-panel rows retained: {target_manifest['zero_selected_gene_library_spots']}",
        "", "## Splits", "",
    ]
    lines.extend(f"- {name}: {values['slides']} slides, {values['spots']} spots" for name, values in split_summary.items())
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in source_checks.items())
    (qc_dir / "training_data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expression-h5ad", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--overwrite-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect(
        args.data_dir,
        args.expression_h5ad,
        args.target_dir,
        args.overwrite_targets,
        args.qc_dir,
    )
    print(json.dumps({key: report[key] for key in ("status", "n_spots", "total_cells", "split_summary")}, indent=2))


if __name__ == "__main__":
    main()
