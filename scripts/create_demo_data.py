#!/usr/bin/env python3
"""Create a small, fully aligned synthetic HistMUSK dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


def _array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    chunks: tuple[int, ...],
) -> Any:
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    return group.create_dataset(
        name,
        data=values,
        chunks=tuple(min(size, chunk) for size, chunk in zip(values.shape, chunks)),
        compressor=compressor,
    )


def create_demo_dataset(
    output_root: str | Path,
    *,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write feature data and an expression H5AD suitable for a smoke run."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Demo output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    data_dir = output_root / "multimodal"
    data_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    n_slides = 6
    spots_per_slide = 8
    n_spots = n_slides * spots_per_slide
    image_dim, text_dim, cell_dim, n_genes = 64, 128, 32, 16
    latent_dim = 8

    slide_index = np.repeat(np.arange(n_slides), spots_per_slide)
    local_index = np.tile(np.arange(spots_per_slide), n_slides)
    sample_ids = np.asarray([f"demo_slide_{value:02d}" for value in slide_index])
    patient_ids = np.asarray([f"demo_patient_{value:02d}" for value in slide_index])
    spot_ids = np.asarray(
        [f"{sample}_spot_{local:03d}" for sample, local in zip(sample_ids, local_index)]
    )
    x = (local_index % 4) * 100 + slide_index * 20
    y = (local_index // 4) * 100 + slide_index * 15
    coordinates = np.column_stack([x, y]).astype(np.float32)

    latent = rng.normal(size=(n_spots, latent_dim)).astype(np.float32)
    image_embeddings = (
        latent @ rng.normal(size=(latent_dim, image_dim)).astype(np.float32)
        + rng.normal(scale=0.15, size=(n_spots, image_dim)).astype(np.float32)
    )
    text_embeddings = (
        latent @ rng.normal(size=(latent_dim, text_dim)).astype(np.float32)
        + rng.normal(scale=0.20, size=(n_spots, text_dim)).astype(np.float32)
    )

    cell_counts = rng.integers(1, 6, size=n_spots, endpoint=False)
    indptr = np.concatenate([[0], np.cumsum(cell_counts)]).astype(np.int64)
    total_cells = int(indptr[-1])
    cell_embeddings = np.empty((total_cells, cell_dim), dtype=np.float32)
    cell_positions = np.empty((total_cells, 2), dtype=np.float32)
    cell_to_spot = np.empty(total_cells, dtype=np.int64)
    cell_projection = rng.normal(size=(latent_dim, cell_dim)).astype(np.float32)
    for spot_index in range(n_spots):
        start, stop = int(indptr[spot_index]), int(indptr[spot_index + 1])
        count = stop - start
        cell_embeddings[start:stop] = (
            latent[spot_index] @ cell_projection
            + rng.normal(scale=0.25, size=(count, cell_dim))
        )
        cell_positions[start:stop] = coordinates[spot_index] + rng.normal(
            scale=12.0, size=(count, 2)
        )
        cell_to_spot[start:stop] = spot_index

    spots = pd.DataFrame(
        {
            "spot_index": np.arange(n_spots, dtype=np.int64),
            "spot_id": spot_ids,
            "barcode": spot_ids,
            "sample_id": sample_ids,
            "patient_id": patient_ids,
            "slide_id": sample_ids,
            "array_row": local_index // 4,
            "array_col": local_index % 4,
            "x_wsi": coordinates[:, 0],
            "y_wsi": coordinates[:, 1],
            "in_tissue": True,
            "cell_count": cell_counts.astype(np.int64),
            "has_conch": True,
            "has_qwen": True,
            "has_cells": True,
            "qc_pass": True,
        }
    )
    spots.to_parquet(data_dir / "spots.parquet", index=False)

    features = zarr.open_group(str(data_dir / "features.zarr"), mode="w")
    _array(features, "cell_embeddings", cell_embeddings, (64, cell_dim))
    _array(features, "cell_positions_wsi", cell_positions, (64, 2))
    _array(features, "cell_to_spot_index", cell_to_spot, (128,))
    _array(features, "cell_indptr", indptr, (n_spots + 1,))
    # The historical storage key is retained for compatibility; these are
    # generic image embeddings and may be replaced by UNI features at runtime.
    _array(features, "conch_embeddings", image_embeddings, (16, image_dim))
    _array(features, "qwen_embeddings", text_embeddings, (16, text_dim))
    _array(features, "spatial_coordinates", coordinates, (16, 2))

    gene_weights = rng.normal(scale=0.35, size=(latent_dim, n_genes))
    log_rate = 1.4 + latent.astype(np.float64) @ gene_weights
    rates = np.exp(np.clip(log_rate, -1.0, 3.0))
    counts = rng.poisson(rates).astype(np.int32)
    zero_library = counts.sum(axis=1) == 0
    counts[zero_library, 0] = 1
    library = counts.sum(axis=1, keepdims=True, dtype=np.float64)
    log_normalized = np.log1p(counts.astype(np.float64) / library * 1e4).astype(np.float32)
    obs_names = [
        f"{sample}_count__{spot_id}" for sample, spot_id in zip(sample_ids, spot_ids)
    ]
    expression = ad.AnnData(
        X=log_normalized,
        obs=pd.DataFrame(index=pd.Index(obs_names, name="observation")),
        var=pd.DataFrame(index=pd.Index([f"GENE_{index:03d}" for index in range(n_genes)])),
    )
    expression.layers["counts"] = counts
    expression_path = output_root / "expression.h5ad"
    expression.write_h5ad(expression_path)

    manifest = {
        "dataset": "HistMUSK synthetic smoke dataset",
        "seed": seed,
        "spot_order": "spot_index ascending",
        "arrays": {
            name: {"shape": list(features[name].shape), "dtype": str(features[name].dtype)}
            for name in features.array_keys()
        },
        "expression_h5ad": str(expression_path),
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "data_dir": str(data_dir),
        "expression_h5ad": str(expression_path),
        "spots": n_spots,
        "cells": total_cells,
        "genes": n_genes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("demo"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = create_demo_dataset(args.output_root, seed=args.seed, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
