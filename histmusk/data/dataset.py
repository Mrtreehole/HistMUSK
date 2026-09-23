"""Worker-safe lazy Zarr dataset for multimodal spots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import zarr
import h5py
from torch.utils.data import Dataset

from .target_processing import load_scaler

TargetMode = Literal["log_normalized", "train_gene_zscore", "raw_counts"]


def _load_base_predictions_npz(
    path: Path, feature_spot_ids: pd.Index
) -> tuple[np.ndarray, dict[int, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Base prediction NPZ not found: {path}")
    payload = np.load(path, allow_pickle=False)
    if "spot_id" not in payload or "prediction" not in payload:
        raise ValueError("Base prediction NPZ requires spot_id and prediction arrays")
    source_ids = pd.Index(payload["spot_id"].astype(str), name="spot_id")
    predictions = np.asarray(payload["prediction"], dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[0] != len(source_ids):
        raise ValueError("Base prediction rows do not match spot IDs")
    if not source_ids.is_unique:
        raise ValueError("Base prediction NPZ contains duplicate spot IDs")
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Base predictions contain NaN or Inf")
    unknown = source_ids.difference(feature_spot_ids, sort=False)
    if len(unknown):
        raise ValueError(f"Base predictions contain unknown spot IDs: {unknown[:10].tolist()}")
    source_row = pd.Series(np.arange(len(source_ids)), index=source_ids)
    feature_row = pd.Series(np.arange(len(feature_spot_ids)), index=feature_spot_ids)
    common = feature_spot_ids.intersection(source_ids, sort=False)
    row_map = {
        int(feature_row.loc[spot_id]): int(source_row.loc[spot_id]) for spot_id in common
    }
    return predictions, row_map, {
        "path": str(path),
        "rows": int(predictions.shape[0]),
        "dimension": int(predictions.shape[1]),
        "matched_spots": int(len(common)),
        "missing_spot_ids": feature_spot_ids.difference(source_ids, sort=False).tolist(),
    }


def _load_spot_whitelist(path: Path) -> tuple[pd.Index, dict[str, Any]]:
    """Load a unique explicit spot-ID whitelist from a table or text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Spot whitelist not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pd.read_parquet(path)
        if "spot_id" not in table:
            raise ValueError(f"Spot whitelist parquet has no spot_id column: {path}")
        values = table["spot_id"]
    elif suffix in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        if "spot_id" not in table:
            raise ValueError(f"Spot whitelist table has no spot_id column: {path}")
        values = table["spot_id"]
    elif suffix in {".txt", ".list"}:
        values = pd.Series(
            [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    else:
        raise ValueError(
            "Spot whitelist must be .parquet, .csv, .tsv, .txt, or .list: "
            f"{path}"
        )
    if values.isna().any():
        raise ValueError(f"Spot whitelist contains missing IDs: {path}")
    spot_ids = pd.Index(values.astype(str), name="spot_id")
    if not len(spot_ids):
        raise ValueError(f"Spot whitelist is empty: {path}")
    if not spot_ids.is_unique:
        duplicates = spot_ids[spot_ids.duplicated()].unique().tolist()
        raise ValueError(f"Spot whitelist has duplicate spot IDs: {duplicates[:10]}")
    return spot_ids, {"path": str(path), "spot_count": int(len(spot_ids))}


def _load_qwen_h5ad(
    path: Path,
    feature_spot_ids: pd.Index,
) -> tuple[np.ndarray, dict[int, int], dict[str, Any]]:
    """Load one explicit-ID Qwen matrix and map feature rows to source rows."""
    import anndata as ad

    if not path.is_file():
        raise FileNotFoundError(f"Qwen H5AD not found: {path}")
    adata = ad.read_h5ad(path, backed="r")
    try:
        embedding_keys = [
            str(key)
            for key in adata.obsm.keys()
            if "qwen" in str(key).lower() or "embedding" in str(key).lower()
        ]
        if len(embedding_keys) != 1:
            raise ValueError(
                "External Qwen H5AD must contain exactly one embedding in obsm; "
                f"found {embedding_keys}"
            )
        embedding_key = embedding_keys[0]
        embeddings = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != adata.n_obs:
            raise ValueError(
                f"Invalid external Qwen matrix shape: {embeddings.shape}, n_obs={adata.n_obs}"
            )
        if not np.isfinite(embeddings).all():
            raise FloatingPointError("External Qwen embeddings contain NaN or Inf")
        if "spot_id" in adata.obs:
            source_ids = pd.Index(adata.obs["spot_id"].astype(str), name="spot_id")
            id_source = "obs['spot_id']"
        else:
            source_ids = pd.Index(adata.obs_names.astype(str), name="spot_id")
            id_source = "obs_names"
        if not source_ids.is_unique:
            duplicated = source_ids[source_ids.duplicated()].tolist()
            raise ValueError(f"External Qwen H5AD has duplicate spot IDs: {duplicated[:10]}")
        if len(source_ids) != embeddings.shape[0]:
            raise ValueError("External Qwen spot ID count does not match embedding rows")

        extra = source_ids.difference(feature_spot_ids)
        if len(extra):
            raise ValueError(
                "External Qwen H5AD contains IDs absent from the multimodal dataset: "
                f"{extra[:10].tolist()}"
            )
        source_row_by_id = pd.Series(
            np.arange(len(source_ids), dtype=np.int64), index=source_ids
        )
        feature_rows = pd.Series(
            np.arange(len(feature_spot_ids), dtype=np.int64), index=feature_spot_ids
        )
        common_ids = feature_spot_ids.intersection(source_ids, sort=False)
        row_by_spot_index = {
            int(feature_rows.loc[spot_id]): int(source_row_by_id.loc[spot_id])
            for spot_id in common_ids
        }
        missing = feature_spot_ids.difference(source_ids, sort=False).tolist()
        metadata = {
            "path": str(path),
            "embedding_key": embedding_key,
            "embedding_dim": int(embeddings.shape[1]),
            "source_rows": int(embeddings.shape[0]),
            "id_source": id_source,
            "matched_spots": len(common_ids),
            "missing_spot_ids": missing,
            "extra_spot_ids": [],
        }
        return embeddings, row_by_spot_index, metadata
    finally:
        adata.file.close()


def inspect_image_features_h5(
    path: Path,
    feature_spot_ids: pd.Index,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate an explicit-ID image feature file and map spot rows to HDF5 rows."""
    if not path.is_file():
        raise FileNotFoundError(f"Image feature HDF5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if "features" not in handle or "paths" not in handle:
            raise ValueError(
                f"Image feature HDF5 must contain 'features' and 'paths': {path}"
            )
        features = handle["features"]
        paths = handle["paths"]
        if features.ndim != 2 or paths.ndim != 1 or features.shape[0] != paths.shape[0]:
            raise ValueError(
                "Invalid image feature arrays: "
                f"features={features.shape}, paths={paths.shape}, file={path}"
            )
        if features.dtype.kind != "f":
            raise TypeError(f"Image features must be floating point, found {features.dtype}")
        source_ids = pd.Index(
            [
                Path(value.decode() if isinstance(value, bytes) else str(value)).stem
                for value in paths[:]
            ],
            name="spot_id",
        )
        if not source_ids.is_unique:
            duplicates = source_ids[source_ids.duplicated()].unique().tolist()
            raise ValueError(f"Image feature HDF5 has duplicate spot IDs: {duplicates[:10]}")

        extra = source_ids.difference(feature_spot_ids, sort=False)
        if len(extra):
            raise ValueError(
                "Image feature HDF5 contains IDs absent from the multimodal dataset: "
                f"{extra[:10].tolist()}"
            )
        source_row_by_id = pd.Series(
            np.arange(len(source_ids), dtype=np.int64), index=source_ids
        )
        source_rows = source_row_by_id.reindex(feature_spot_ids).to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        row_by_spot_index = np.full(len(feature_spot_ids), -1, dtype=np.int64)
        available = np.isfinite(source_rows)
        row_by_spot_index[available] = source_rows[available].astype(np.int64)
        missing = feature_spot_ids[~available].tolist()
        metadata = {
            "path": str(path),
            "model": str(handle.attrs.get("model", path.stem)),
            "embedding_dim": int(features.shape[1]),
            "source_rows": int(features.shape[0]),
            "id_source": "stem(paths)",
            "matched_spots": int(available.sum()),
            "missing_spot_ids": missing,
            "extra_spot_ids": [],
            "dtype": str(features.dtype),
        }
        return row_by_spot_index, metadata


class MultimodalSpotDataset(Dataset[dict[str, Any]]):
    """Read only requested spot/cell slices; no feature matrix is copied at init."""

    def __init__(
        self,
        data_dir: str | Path,
        target_dir: str | Path,
        split: str | None = None,
        target_mode: TargetMode = "train_gene_zscore",
        load_cells: bool = True,
        load_qwen: bool = True,
        exclude_empty_spots: bool = True,
        qwen_h5ad: str | Path | None = None,
        exclude_missing_qwen: bool = True,
        image_features_h5: str | Path | None = None,
        exclude_missing_image: bool = True,
        spot_whitelist: str | Path | None = None,
        target_scaler: str | Path | None = None,
        base_predictions_npz: str | Path | None = None,
        splits_file: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.splits_path = (
            Path(splits_file).expanduser().resolve()
            if splits_file is not None
            else self.target_dir / "splits.parquet"
        )
        spots = pd.read_parquet(self.data_dir / "spots.parquet").sort_values("spot_index").reset_index(drop=True)
        splits = pd.read_parquet(self.splits_path).sort_values("spot_index").reset_index(drop=True)
        if not spots["spot_id"].is_unique or not splits["spot_id"].is_unique:
            raise ValueError("Multimodal spots and target splits must have unique spot IDs")
        feature_ids = pd.Index(spots["spot_id"].astype(str), name="spot_id")
        target_ids = pd.Index(splits["spot_id"].astype(str), name="spot_id")
        missing = feature_ids.difference(target_ids)
        extra = target_ids.difference(feature_ids)
        if len(missing) or len(extra):
            raise ValueError(
                "Target splits and multimodal spots have different spot ID sets: "
                f"missing={missing[:10].tolist()}, extra={extra[:10].tolist()}"
            )
        aligned_splits = splits.assign(spot_id=target_ids).set_index("spot_id").loc[feature_ids]
        target_indices = aligned_splits["spot_index"].to_numpy(dtype=np.int64)
        if len(np.unique(target_indices)) != len(target_indices):
            raise ValueError("Target split row indices are not unique")
        for column in ("sample_id", "patient_id", "slide_id"):
            if column in aligned_splits:
                target_values = aligned_splits[column].fillna("").astype(str).to_numpy()
                feature_values = spots[column].fillna("").astype(str).to_numpy()
                if not np.array_equal(target_values, feature_values):
                    raise ValueError(
                        f"Target split {column} values differ after explicit spot-ID alignment"
                    )
        spots["target_index"] = target_indices
        spots["target_split"] = aligned_splits["split"].to_numpy()
        if "target_available" in aligned_splits:
            spots["target_available"] = aligned_splits["target_available"].astype(bool).to_numpy()
        else:
            spots["target_available"] = True
        self.missing_target_spots_total = int((~spots["target_available"]).sum())
        self.missing_target_spots_excluded = 0
        self.spot_whitelist_source: dict[str, Any] | None = None
        self.spots_excluded_by_whitelist_total = 0
        self.spots_excluded_by_whitelist_in_split = 0
        whitelist_ids: pd.Index | None = None
        whitelist_keep = np.ones(len(spots), dtype=bool)
        if spot_whitelist is not None:
            whitelist_path = Path(spot_whitelist).expanduser().resolve()
            whitelist_ids, whitelist_source = _load_spot_whitelist(whitelist_path)
            unknown = whitelist_ids.difference(feature_ids, sort=False)
            if len(unknown):
                raise ValueError(
                    "Spot whitelist contains IDs absent from the multimodal dataset: "
                    f"{unknown[:10].tolist()}"
                )
            whitelist_keep = feature_ids.isin(whitelist_ids)
            matched = int(whitelist_keep.sum())
            if matched != len(whitelist_ids):
                raise ValueError(
                    f"Spot whitelist matched {matched} rows but contains {len(whitelist_ids)} IDs"
                )
            self.spots_excluded_by_whitelist_total = int((~whitelist_keep).sum())
            whitelist_source.update(
                {
                    "matched_spots": matched,
                    "excluded_multimodal_spots": self.spots_excluded_by_whitelist_total,
                }
            )
            self.spot_whitelist_source = whitelist_source
        self._external_qwen: np.ndarray | None = None
        self._external_qwen_row_by_spot_index: dict[int, int] | None = None
        self.qwen_source: dict[str, Any] | None = None
        self.missing_qwen_spots_total = 0
        self.missing_qwen_spots_excluded = 0
        if qwen_h5ad is not None:
            qwen_path = Path(qwen_h5ad).expanduser().resolve()
            external, row_map, source = _load_qwen_h5ad(qwen_path, feature_ids)
            self._external_qwen = external
            self._external_qwen_row_by_spot_index = row_map
            self.qwen_source = source
            available = spots["spot_index"].isin(row_map).to_numpy()
            missing_count = int((~available).sum())
            self.missing_qwen_spots_total = missing_count
            if missing_count and not exclude_missing_qwen:
                raise ValueError(
                    f"External Qwen H5AD is missing {missing_count} multimodal spots: "
                    f"{source['missing_spot_ids'][:10]}"
                )
            spots["external_qwen_available"] = available
        self._base_predictions: np.ndarray | None = None
        self._base_prediction_row_by_spot_index: dict[int, int] | None = None
        self.base_prediction_source: dict[str, Any] | None = None
        if base_predictions_npz is not None:
            base_path = Path(base_predictions_npz).expanduser().resolve()
            base, row_map, source = _load_base_predictions_npz(base_path, feature_ids)
            self._base_predictions = base
            self._base_prediction_row_by_spot_index = row_map
            self.base_prediction_source = source
            spots["base_prediction_available"] = spots["spot_index"].isin(row_map).to_numpy()
        self._image_features_path: Path | None = None
        self._image_row_by_spot_index: np.ndarray | None = None
        self._image_features: h5py.File | None = None
        self.image_source: dict[str, Any] | None = None
        self.missing_image_spots_total = 0
        self.missing_image_spots_excluded = 0
        if image_features_h5 is not None:
            image_path = Path(image_features_h5).expanduser().resolve()
            row_map, source = inspect_image_features_h5(image_path, feature_ids)
            self._image_features_path = image_path
            self._image_row_by_spot_index = row_map
            self.image_source = source
            available = row_map >= 0
            self.missing_image_spots_total = int((~available).sum())
            if self.missing_image_spots_total and not exclude_missing_image:
                raise ValueError(
                    f"External image features are missing {self.missing_image_spots_total} "
                    f"multimodal spots: {source['missing_spot_ids'][:10]}"
                )
            spots["external_image_available"] = available
        if split is not None:
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown split: {split}")
            keep = spots["target_split"].eq(split).to_numpy()
            spots = spots.loc[keep].reset_index(drop=True)
            whitelist_keep = whitelist_keep[keep]
        target_available = spots["target_available"].to_numpy(dtype=bool)
        self.missing_target_spots_excluded = int((~target_available).sum())
        spots = spots.loc[target_available].reset_index(drop=True)
        whitelist_keep = whitelist_keep[target_available]
        if whitelist_ids is not None:
            self.spots_excluded_by_whitelist_in_split = int((~whitelist_keep).sum())
            spots = spots.loc[whitelist_keep].reset_index(drop=True)
        if qwen_h5ad is not None and exclude_missing_qwen:
            available = spots["external_qwen_available"].to_numpy(dtype=bool)
            self.missing_qwen_spots_excluded = int((~available).sum())
            spots = spots.loc[available].reset_index(drop=True)
        if image_features_h5 is not None and exclude_missing_image:
            available = spots["external_image_available"].to_numpy(dtype=bool)
            self.missing_image_spots_excluded = int((~available).sum())
            spots = spots.loc[available].reset_index(drop=True)
        self.empty_spots_excluded = int(spots["cell_count"].eq(0).sum()) if exclude_empty_spots else 0
        if exclude_empty_spots:
            spots = spots.loc[spots["cell_count"].gt(0)].reset_index(drop=True)
        if base_predictions_npz is not None:
            available = spots["base_prediction_available"].to_numpy(dtype=bool)
            if not available.all():
                missing_ids = spots.loc[~available, "spot_id"].astype(str).tolist()
                raise ValueError(
                    f"Base prediction NPZ is missing {len(missing_ids)} eligible spots: "
                    f"{missing_ids[:10]}"
                )
        self.spots = spots
        self.split = split
        self.target_mode = target_mode
        self.load_cells = load_cells
        self.load_qwen = load_qwen
        self._features: zarr.Group | None = None
        self._targets: zarr.Group | None = None
        self._cell_index_map: dict[int, int] | None = None
        self._qwen_index_map: dict[int, int] | None = None
        self._row_caches: dict[str, tuple[int, np.ndarray]] = {}
        self.target_scaler_path = (
            Path(target_scaler).expanduser().resolve()
            if target_scaler is not None
            else self.target_dir / "target_scaler.npz"
        )
        self.train_mean, self.train_std, self.gene_names = load_scaler(
            self.target_scaler_path
        )
        metadata = zarr.open_group(str(self.data_dir / "features.zarr"), mode="r")
        target_metadata = zarr.open_group(str(self.target_dir / "gene_expression.zarr"), mode="r")
        self.cell_dim = int(metadata["cell_embeddings"].shape[1])
        self.image_dim = (
            int(self.image_source["embedding_dim"])
            if self.image_source is not None
            else int(metadata["conch_embeddings"].shape[1])
        )
        self.qwen_dim = (
            int(self._external_qwen.shape[1])
            if self._external_qwen is not None
            else int(metadata["qwen_embeddings"].shape[1])
        )
        self.num_genes = int(target_metadata["log_normalized"].shape[1])
        target_rows = int(target_metadata["log_normalized"].shape[0])
        if len(target_indices) != target_rows or set(target_indices.tolist()) != set(range(target_rows)):
            raise ValueError("Target split row mapping is not a complete permutation of target arrays")
        if target_metadata["raw_counts"].shape != target_metadata["log_normalized"].shape:
            raise ValueError("Raw and log-normalized target arrays have different shapes")
        if len(self.gene_names) != self.num_genes:
            raise ValueError("Target scaler gene count does not match target arrays")
        if target_mode not in {"log_normalized", "train_gene_zscore", "raw_counts"}:
            raise ValueError(f"Unsupported target_mode: {target_mode}")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_features"] = None
        state["_targets"] = None
        state["_image_features"] = None
        state["_row_caches"] = {}
        return state

    def _ensure_open(self) -> tuple[zarr.Group, zarr.Group]:
        if self._features is None:
            self._features = zarr.open_group(str(self.data_dir / "features.zarr"), mode="r")
        if self._targets is None:
            self._targets = zarr.open_group(str(self.target_dir / "gene_expression.zarr"), mode="r")
        return self._features, self._targets

    def _ensure_image_open(self) -> h5py.Dataset | None:
        if self._image_features_path is None:
            return None
        if self._image_features is None:
            self._image_features = h5py.File(self._image_features_path, "r")
        return self._image_features["features"]

    def _cached_row(self, cache_name: str, array: Any, row: int) -> np.ndarray:
        chunks = getattr(array, "chunks", None)
        block_size = int(chunks[0]) if chunks else min(1024, int(array.shape[0]))
        block_start = (row // block_size) * block_size
        cached = self._row_caches.get(cache_name)
        if cached is None or cached[0] != block_start:
            block = np.asarray(array[block_start : min(block_start + block_size, array.shape[0])])
            self._row_caches[cache_name] = (block_start, block)
        else:
            block = cached[1]
        return block[row - block_start]

    def __len__(self) -> int:
        return len(self.spots)

    def set_negative_control(self, control: str | None, seed: int = 42) -> None:
        """Shuffle an auxiliary modality within each slide while targets stay fixed."""
        self._cell_index_map = None
        self._qwen_index_map = None
        if control is None:
            return
        if control not in {"qwen_shuffled", "cells_shuffled"}:
            raise ValueError(f"Unknown negative control: {control}")
        rng = np.random.default_rng(seed)
        mapping: dict[int, int] = {}
        for _, group in self.spots.groupby("slide_id", sort=True):
            indices = group["spot_index"].to_numpy(dtype=np.int64)
            shuffled = indices.copy()
            rng.shuffle(shuffled)
            mapping.update(zip(indices.tolist(), shuffled.tolist()))
        if control == "qwen_shuffled":
            self._qwen_index_map = mapping
        else:
            self._cell_index_map = mapping

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        features, targets = self._ensure_open()
        image_features = self._ensure_image_open()
        row = self.spots.iloc[index]
        spot_index = int(row.spot_index)
        target_index = int(row.target_index)
        cell_source = self._cell_index_map.get(spot_index, spot_index) if self._cell_index_map else spot_index
        qwen_source = self._qwen_index_map.get(spot_index, spot_index) if self._qwen_index_map else spot_index
        if self.load_cells:
            indptr = features["cell_indptr"]
            start, stop = int(indptr[cell_source]), int(indptr[cell_source + 1])
            cells = features["cell_embeddings"][start:stop].astype(np.float32, copy=True)
            cell_count = stop - start
        else:
            start = stop = 0
            cells = np.empty((0, self.cell_dim), dtype=np.float32)
            cell_count = 0
        if self.target_mode == "raw_counts":
            target = self._cached_row("raw_counts", targets["raw_counts"], target_index).astype(np.float32)
        else:
            target = self._cached_row("log_normalized", targets["log_normalized"], target_index).astype(np.float32)
            if self.target_mode == "train_gene_zscore":
                target = (target - self.train_mean) / self.train_std
        patient = "" if pd.isna(row.patient_id) else str(row.patient_id)
        return {
            "spot_index": spot_index,
            "spot_id": str(row.spot_id),
            "sample_id": str(row.sample_id),
            "patient_id": patient,
            "slide_id": str(row.slide_id),
            "cell_embeddings": torch.from_numpy(cells),
            "conch_embedding": torch.from_numpy(
                self._cached_row(
                    "external_image",
                    image_features,
                    int(self._image_row_by_spot_index[spot_index]),
                ).astype(np.float32)
                if image_features is not None and self._image_row_by_spot_index is not None
                else self._cached_row(
                    "conch", features["conch_embeddings"], spot_index
                ).astype(np.float32)
            ),
            "qwen_embedding": torch.from_numpy(
                self._external_qwen[
                    self._external_qwen_row_by_spot_index[qwen_source]
                ].astype(np.float32, copy=True)
                if self.load_qwen and self._external_qwen is not None
                else self._cached_row(
                    "qwen", features["qwen_embeddings"], qwen_source
                ).astype(np.float32)
                if self.load_qwen
                else np.zeros(self.qwen_dim, dtype=np.float32)
            ),
            "coordinate": torch.from_numpy(
                self._cached_row("coordinate", features["spatial_coordinates"], spot_index).astype(np.float32)
            ),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            **(
                {
                    "base_prediction": torch.from_numpy(
                        self._base_predictions[
                            self._base_prediction_row_by_spot_index[spot_index]
                        ].astype(np.float32, copy=True)
                    )
                }
                if self._base_predictions is not None
                and self._base_prediction_row_by_spot_index is not None
                else {}
            ),
            "cell_count": cell_count,
            "cell_source_spot_index": cell_source,
            "cell_loaded": self.load_cells,
        }
