"""Build explicitly ID-aligned expression targets from the supplied HVG AnnData file."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from numcodecs import Blosc

from .splits import create_group_splits, load_or_create_splits


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_array(group: zarr.Group, name: str, shape: tuple[int, ...], dtype: str) -> Any:
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    chunks = (min(1024, shape[0]), shape[1])
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)


def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _spot_ids_from_obs_names(obs_names: pd.Index) -> tuple[pd.Index, list[str]]:
    spot_ids: list[str] = []
    section_ids: list[str] = []
    malformed: list[str] = []
    for value in obs_names.astype(str):
        if "__" not in value:
            malformed.append(value)
            continue
        section, spot_id = value.split("__", 1)
        if not section or not spot_id:
            malformed.append(value)
            continue
        section_ids.append(section.removesuffix("_count"))
        spot_ids.append(spot_id)
    if malformed:
        raise ValueError(
            "H5AD obs_names must be '<section>__<full_spot_id>'; "
            f"malformed examples={malformed[:10]}"
        )
    return pd.Index(spot_ids, name="spot_id"), section_ids


def _create_target_aware_splits(
    spots: pd.DataFrame,
    target_available: np.ndarray,
    path: Path,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Create normal group splits while forcing spots without targets to excluded."""
    if bool(np.all(target_available)):
        splits, group_column = load_or_create_splits(spots, path, seed=seed)
        splits["target_available"] = True
        splits.to_parquet(path, index=False)
        return splits, group_column

    available_spots = spots.loc[target_available].copy()
    splits_available, group_column = create_group_splits(available_spots, seed=seed)
    splits = spots[
        ["spot_index", "spot_id", "sample_id", "patient_id", "slide_id"]
    ].copy()
    splits["split"] = "excluded"
    splits["target_available"] = target_available
    split_by_id = splits_available.set_index("spot_id")["split"]
    splits.loc[target_available, "split"] = (
        splits.loc[target_available, "spot_id"].map(split_by_id).to_numpy()
    )
    if splits.loc[target_available, "split"].isna().any():
        raise RuntimeError("Some target-available spots were not assigned to a split")
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)
    return splits, group_column


def _validate_complete_cache(cache_dir: Path, expression_h5ad: Path) -> dict[str, Any] | None:
    manifest_path = cache_dir / "target_manifest.json"
    required = [
        manifest_path,
        cache_dir / "gene_expression.zarr",
        cache_dir / "gene_names.json",
        cache_dir / "splits.parquet",
        cache_dir / "target_index.parquet",
        cache_dir / "target_scaler.npz",
    ]
    if not all(path.exists() for path in required):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_h5ad", {})
    stat = expression_h5ad.stat()
    unchanged = (
        Path(source.get("path", "")) == expression_h5ad.resolve()
        and source.get("size") == stat.st_size
        and source.get("mtime_ns") == stat.st_mtime_ns
    )
    if not unchanged:
        raise RuntimeError(
            f"Target cache source differs from {expression_h5ad}; use --overwrite-targets "
            "or a new --target-dir"
        )
    return manifest


def prepare_expression_targets(
    data_dir: str | Path,
    expression_h5ad: str | Path,
    cache_dir: str | Path,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy supplied HVG targets into spot-index order without row-order assumptions."""
    data_dir = Path(data_dir).resolve()
    expression_h5ad = Path(expression_h5ad).resolve()
    cache_dir = Path(cache_dir).resolve()
    if not expression_h5ad.is_file():
        raise FileNotFoundError(f"Expression H5AD not found: {expression_h5ad}")
    if cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    if cache_dir.exists() and not overwrite:
        existing = _validate_complete_cache(cache_dir, expression_h5ad)
        if existing is not None:
            return existing
        if any(cache_dir.iterdir()):
            raise FileExistsError(f"Incomplete/non-empty target cache exists: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    spots = pd.read_parquet(data_dir / "spots.parquet").sort_values("spot_index").reset_index(drop=True)
    expected_indices = np.arange(len(spots), dtype=np.int64)
    if not np.array_equal(spots["spot_index"].to_numpy(), expected_indices):
        raise ValueError("spots.parquet spot_index is not contiguous")
    if not spots["spot_id"].is_unique:
        raise ValueError("spots.parquet spot_id is not unique")

    adata = ad.read_h5ad(expression_h5ad, backed="r")
    try:
        if "counts" not in adata.layers:
            raise ValueError("Expression H5AD must contain layers['counts']")
        if adata.n_obs <= 0:
            raise ValueError("Expression H5AD has no spots")
        if adata.n_vars <= 0:
            raise ValueError("Expression H5AD has no genes")
        if not adata.var_names.is_unique:
            duplicates = adata.var_names[adata.var_names.duplicated()].astype(str).tolist()
            raise ValueError(f"Expression H5AD has duplicate genes: {duplicates[:10]}")

        source_spot_ids, source_sections = _spot_ids_from_obs_names(adata.obs_names)
        if not source_spot_ids.is_unique:
            duplicates = source_spot_ids[source_spot_ids.duplicated()].tolist()
            raise ValueError(f"Derived H5AD spot IDs are duplicated: {duplicates[:10]}")
        expected_spot_ids = pd.Index(spots["spot_id"].astype(str), name="spot_id")
        missing = expected_spot_ids.difference(source_spot_ids)
        extra = source_spot_ids.difference(expected_spot_ids)
        if len(extra):
            raise ValueError(
                "H5AD contains spot IDs absent from the multimodal dataset: "
                f"{extra[:10].tolist()}"
            )

        target_available = expected_spot_ids.isin(source_spot_ids)
        available_indices = np.flatnonzero(target_available).astype(np.int64)
        available_spot_ids = expected_spot_ids[target_available]
        source_row_by_id = pd.Series(np.arange(adata.n_obs, dtype=np.int64), index=source_spot_ids)
        source_rows = source_row_by_id.reindex(available_spot_ids).to_numpy(dtype=np.int64)
        source_section_by_id = pd.Series(source_sections, index=source_spot_ids)
        aligned_sections = source_section_by_id.reindex(available_spot_ids).to_numpy(dtype=str)
        expected_sections = spots.loc[target_available, "sample_id"].astype(str).to_numpy()
        if not np.array_equal(aligned_sections, expected_sections):
            bad = np.flatnonzero(aligned_sections != expected_sections)[:10]
            raise ValueError(
                "H5AD section/sample mismatch after ID alignment: "
                + ", ".join(
                    f"{available_spot_ids[i]}: h5ad={aligned_sections[i]}, spots={expected_sections[i]}"
                    for i in bad
                )
            )

        x_source = adata.X[:]
        counts_source = adata.layers["counts"]
        if x_source.shape != adata.shape or counts_source.shape != adata.shape:
            raise ValueError("H5AD X/counts shapes do not match AnnData shape")
        x_data = x_source.data if sp.issparse(x_source) else np.asarray(x_source)
        count_data = counts_source.data if sp.issparse(counts_source) else np.asarray(counts_source)
        if not np.isfinite(x_data).all() or not np.isfinite(count_data).all():
            raise FloatingPointError("H5AD expression contains NaN or Inf")
        if np.any(x_data < 0):
            raise ValueError("H5AD X contains negative values")
        if np.any(count_data < 0) or np.any(count_data != np.floor(count_data)):
            raise ValueError("H5AD counts must be non-negative integer-valued data")
        if count_data.size and count_data.max() > np.iinfo(np.int32).max:
            raise OverflowError("H5AD count exceeds int32")

        genes = adata.var_names.astype(str).tolist()
        n_genes = len(genes)
        target_path = cache_dir / "gene_expression.zarr"
        root = zarr.open_group(str(target_path), mode="w")
        raw_array = _create_array(root, "raw_counts", (len(spots), n_genes), "i4")
        log_array = _create_array(root, "log_normalized", (len(spots), n_genes), "f4")
        zero_target_spot_ids: list[str] = []
        max_normalization_error = 0.0
        for start in range(0, len(available_indices), 1024):
            stop = min(start + 1024, len(available_indices))
            rows = source_rows[start:stop]
            destination_rows = available_indices[start:stop]
            raw_block = _to_dense(counts_source[rows, :])
            log_block = _to_dense(x_source[rows, :]).astype(np.float32, copy=False)
            libraries = raw_block.sum(axis=1, dtype=np.float64)
            zero_mask = libraries == 0
            zero_target_spot_ids.extend(available_spot_ids[start:stop][zero_mask].tolist())
            expected_log = np.log1p(
                raw_block.astype(np.float64)
                / np.where(libraries > 0, libraries, 1.0)[:, None]
                * 1e4
            )
            max_normalization_error = max(
                max_normalization_error, float(np.max(np.abs(log_block - expected_log), initial=0.0))
            )
            raw_array.oindex[destination_rows, :] = raw_block.astype(np.int32, copy=False)
            log_array.oindex[destination_rows, :] = log_block

        if max_normalization_error > 1e-5:
            raise ValueError(
                "H5AD X is not log1p(count / selected-gene library * 10000): "
                f"max_abs_error={max_normalization_error}"
            )
        root.attrs.update({
            "spot_order": "processed_multimodal_dataset/spots.parquet sorted by spot_index",
            "source": str(expression_h5ad),
            "source_alignment": "obs_names split once on '__'; suffix is full spot_id",
            "normalization": "source X; verified log1p(count / selected-gene library_size * 10000)",
            "target_available_spots": int(target_available.sum()),
            "target_unavailable_spots": int((~target_available).sum()),
        })

        h5ad_source_rows = np.full(len(spots), -1, dtype=np.int64)
        h5ad_source_rows[available_indices] = source_rows
        h5ad_obs_names = np.full(len(spots), "", dtype=object)
        h5ad_obs_names[available_indices] = np.asarray(adata.obs_names.astype(str))[source_rows]
        h5ad_sections = np.full(len(spots), "", dtype=object)
        h5ad_sections[available_indices] = aligned_sections
        index_frame = pd.DataFrame({
            "spot_index": expected_indices,
            "spot_id": expected_spot_ids,
            "target_available": target_available,
            "h5ad_source_row": h5ad_source_rows,
            "h5ad_obs_name": h5ad_obs_names,
            "h5ad_section_id": h5ad_sections,
        })
        index_frame.to_parquet(cache_dir / "target_index.parquet", index=False)
        (cache_dir / "gene_names.json").write_text(json.dumps(genes, indent=2) + "\n", encoding="utf-8")

        splits, group_column = _create_target_aware_splits(
            spots, target_available, cache_dir / "splits.parquet", seed
        )
        train_rows = splits.loc[splits["split"] == "train", "spot_index"].to_numpy(dtype=np.int64)
        if not len(train_rows):
            raise ValueError("No target-available training spots were created")
        train_sum = np.zeros(n_genes, dtype=np.float64)
        train_sumsq = np.zeros(n_genes, dtype=np.float64)
        for start in range(0, len(train_rows), 2048):
            block = log_array.oindex[train_rows[start : start + 2048], :].astype(np.float64)
            train_sum += block.sum(axis=0)
            train_sumsq += np.square(block).sum(axis=0)
        scaler_mean = train_sum / len(train_rows)
        scaler_std = np.sqrt(np.maximum(train_sumsq / len(train_rows) - np.square(scaler_mean), 0.0))
        scaler_std[scaler_std < 1e-6] = 1.0
        np.savez_compressed(
            cache_dir / "target_scaler.npz",
            gene_names=np.asarray(genes, dtype=str),
            train_mean=scaler_mean.astype(np.float32),
            train_std=scaler_std.astype(np.float32),
            target_mode=np.asarray("train_gene_zscore"),
        )

        source_stat = expression_h5ad.stat()
        split_counts = splits.groupby("split").agg(
            spots=("spot_index", "size"), slides=("slide_id", "nunique")
        )
        annotation_ids = adata.obs["cellvit_spot_id"].astype(str) if "cellvit_spot_id" in adata.obs else None
        missing_annotation_ids = int((annotation_ids == "").sum()) if annotation_ids is not None else None
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source_h5ad": {
                "path": str(expression_h5ad),
                "size": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
                "sha256": sha256_file(expression_h5ad),
                "shape": [adata.n_obs, adata.n_vars],
                "x_dtype": str(adata.X.dtype),
                "counts_dtype": str(adata.layers["counts"].dtype),
            },
            "alignment_rule": "obs_names split once on '__'; suffix is explicit full spot_id; never row order",
            "alignment": {
                "spot_id_sets_equal": len(missing) == 0,
                "target_available_spots": int(target_available.sum()),
                "missing_spot_ids": int(len(missing)),
                "missing_spot_id_examples": missing[:10].tolist(),
                "extra_spot_ids": 0,
                "missing_spot_policy": "retained in global index; target_available=false; split=excluded",
                "source_order_matches_spot_index": bool(
                    np.array_equal(source_rows, available_indices)
                ),
                "source_order_matches_available_target_order": bool(
                    np.array_equal(source_rows, np.arange(adata.n_obs, dtype=np.int64))
                ),
                "moved_rows": int(
                    np.count_nonzero(source_rows != np.arange(adata.n_obs, dtype=np.int64))
                ),
                "max_row_displacement": int(
                    np.max(
                        np.abs(source_rows - np.arange(adata.n_obs, dtype=np.int64)),
                        initial=0,
                    )
                ),
                "sample_ids_match_after_alignment": True,
                "cellvit_spot_id_missing_values": missing_annotation_ids,
                "mapping_file": str((cache_dir / "target_index.parquet").resolve()),
            },
            "genes": n_genes,
            "gene_order": "H5AD var_names order",
            "hvg_protocol": _jsonable(adata.uns.get("hvg_protocol", {})),
            "normalization": "source X; verified log1p(count / selected-gene library_size * 10000)",
            "max_normalization_abs_error": max_normalization_error,
            "target_shapes": {
                "raw_counts": [len(spots), n_genes],
                "log_normalized": [len(spots), n_genes],
            },
            "target_dtypes": {"raw_counts": "int32", "log_normalized": "float32"},
            "zero_selected_gene_library_spot_ids": sorted(zero_target_spot_ids),
            "zero_selected_gene_library_spots": len(zero_target_spot_ids),
            "zero_row_policy": (
                "target-available all-zero rows retained; target-unavailable rows are storage "
                "placeholders and always excluded"
            ),
            "group_column": group_column,
            "seed": seed,
            "split_counts": split_counts.to_dict(orient="index"),
            "scaler_fit": "training split only",
            "excluded_spots": int((~target_available).sum()),
        }
        (cache_dir / "target_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        adata.file.close()


def load_scaler(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    payload = np.load(path, allow_pickle=False)
    return payload["train_mean"], payload["train_std"], payload["gene_names"].astype(str).tolist()


def fit_target_scaler_from_splits(
    target_dir: str | Path,
    splits: pd.DataFrame | str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Fit gene standardization using only rows labelled as training data."""
    target_dir = Path(target_dir).resolve()
    output_path = Path(output_path).resolve()
    split_frame = (
        pd.read_parquet(splits)
        if isinstance(splits, (str, Path))
        else splits.copy()
    )
    required = {"spot_index", "spot_id", "split"}
    missing = required.difference(split_frame.columns)
    if missing:
        raise ValueError(f"Split table lacks required columns: {sorted(missing)}")
    if not split_frame["spot_id"].astype(str).is_unique:
        raise ValueError("Split table contains duplicate spot IDs")
    train = split_frame["split"].eq("train").to_numpy()
    if "target_available" in split_frame:
        train &= split_frame["target_available"].astype(bool).to_numpy()
    train_rows = np.sort(
        split_frame.loc[train, "spot_index"].to_numpy(dtype=np.int64)
    )
    if not len(train_rows):
        raise ValueError("Cannot fit target scaler without training spots")
    if len(np.unique(train_rows)) != len(train_rows):
        raise ValueError("Training split contains duplicate target rows")

    targets = zarr.open_group(str(target_dir / "gene_expression.zarr"), mode="r")
    log_array = targets["log_normalized"]
    if train_rows.min() < 0 or train_rows.max() >= log_array.shape[0]:
        raise IndexError("Training split contains target rows outside gene_expression.zarr")
    genes = [
        str(value)
        for value in json.loads((target_dir / "gene_names.json").read_text(encoding="utf-8"))
    ]
    if len(genes) != log_array.shape[1]:
        raise ValueError("Gene name count does not match target matrix width")

    train_sum = np.zeros(len(genes), dtype=np.float64)
    train_sumsq = np.zeros(len(genes), dtype=np.float64)
    for start in range(0, len(train_rows), 2048):
        block = log_array.oindex[train_rows[start : start + 2048], :].astype(np.float64)
        if not np.isfinite(block).all():
            raise FloatingPointError("Training targets contain NaN or Inf")
        train_sum += block.sum(axis=0)
        train_sumsq += np.square(block).sum(axis=0)
    mean = train_sum / len(train_rows)
    std = np.sqrt(np.maximum(train_sumsq / len(train_rows) - np.square(mean), 0.0))
    constant_genes = std < 1e-6
    std[constant_genes] = 1.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        gene_names=np.asarray(genes, dtype=str),
        train_mean=mean.astype(np.float32),
        train_std=std.astype(np.float32),
        target_mode=np.asarray("train_gene_zscore"),
        training_spots=np.asarray(len(train_rows), dtype=np.int64),
    )
    return {
        "path": str(output_path),
        "training_spots": int(len(train_rows)),
        "genes": int(len(genes)),
        "constant_genes": int(constant_genes.sum()),
    }
