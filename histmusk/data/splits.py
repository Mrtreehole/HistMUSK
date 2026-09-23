"""Leakage-safe group-level dataset splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class ChunkShuffleSampler(Sampler[int]):
    """Shuffle storage chunks and their members while preserving chunk-local I/O."""

    def __init__(self, spot_indices: np.ndarray, chunk_size: int = 1024, seed: int = 42) -> None:
        self.seed = seed
        self.epoch = 0
        self.groups: dict[int, list[int]] = {}
        for dataset_index, spot_index in enumerate(spot_indices):
            self.groups.setdefault(int(spot_index) // chunk_size, []).append(dataset_index)
        self.length = len(spot_indices)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        group_keys = np.asarray(list(self.groups), dtype=np.int64)
        rng.shuffle(group_keys)
        for key in group_keys:
            members = np.asarray(self.groups[int(key)], dtype=np.int64)
            rng.shuffle(members)
            yield from members.tolist()

    def __len__(self) -> int:
        return self.length


def choose_group_column(spots: pd.DataFrame) -> str:
    """Use patient IDs when complete, otherwise fall back to slide IDs."""
    if "patient_id" in spots:
        patient = spots["patient_id"].fillna("").astype(str).str.strip()
        if patient.ne("").all():
            return "patient_id"
    if "slide_id" not in spots or spots["slide_id"].isna().any():
        raise ValueError("Neither complete patient_id nor complete slide_id is available")
    return "slide_id"


def create_group_splits(
    spots: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, str]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below one")
    group_column = choose_group_column(spots)
    groups = np.array(sorted(spots[group_column].astype(str).unique()), dtype=object)
    if len(groups) < 3:
        raise ValueError("At least three patient/slide groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_train = max(1, int(round(len(groups) * train_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    if n_train + n_val >= len(groups):
        n_val = 1
        n_train = len(groups) - 2
    group_split = {
        **{str(group): "train" for group in groups[:n_train]},
        **{str(group): "val" for group in groups[n_train : n_train + n_val]},
        **{str(group): "test" for group in groups[n_train + n_val :]},
    }
    result = spots[["spot_index", "spot_id", "sample_id", "patient_id", "slide_id"]].copy()
    result["split"] = spots[group_column].astype(str).map(group_split)
    if result["split"].isna().any():
        raise RuntimeError("Some spots were not assigned to a split")
    assert_no_group_leakage(result, group_column)
    return result, group_column


def assert_no_group_leakage(splits: pd.DataFrame, group_column: str) -> None:
    active = splits[splits["split"].isin(["train", "val", "test"])]
    overlap = active.groupby(group_column, dropna=False)["split"].nunique()
    leaked = overlap[overlap > 1]
    if len(leaked):
        raise ValueError(f"{group_column} leakage across splits: {leaked.index.tolist()[:10]}")


def create_leave_one_patient_out_splits(
    spots: pd.DataFrame,
    test_patient: str,
    val_patient: str,
    target_available: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Create one strict patient-level train/validation/test split."""
    required = {
        "spot_index", "spot_id", "sample_id", "patient_id", "slide_id"
    }
    missing = required.difference(spots.columns)
    if missing:
        raise ValueError(f"spots table lacks required columns: {sorted(missing)}")
    ordered = spots.sort_values("spot_index").reset_index(drop=True).copy()
    patient_ids = ordered["patient_id"].fillna("").astype(str).str.strip()
    if patient_ids.eq("").any():
        raise ValueError("Strict patient LOPO requires a non-empty patient_id for every spot")
    patients = sorted(patient_ids.unique().tolist())
    if len(patients) < 3:
        raise ValueError("Strict patient LOPO requires at least three patients")
    if test_patient == val_patient:
        raise ValueError("Test and validation patients must differ")
    unknown = {str(test_patient), str(val_patient)}.difference(patients)
    if unknown:
        raise ValueError(f"Unknown LOPO patient IDs: {sorted(unknown)}")

    result = ordered[
        ["spot_index", "spot_id", "sample_id", "patient_id", "slide_id"]
    ].copy()
    result["split"] = "train"
    result.loc[patient_ids.eq(str(val_patient)), "split"] = "val"
    result.loc[patient_ids.eq(str(test_patient)), "split"] = "test"
    if target_available is None:
        available = np.ones(len(result), dtype=bool)
    else:
        available = np.asarray(target_available, dtype=bool)
        if available.shape != (len(result),):
            raise ValueError(
                f"target_available must have shape {(len(result),)}, found {available.shape}"
            )
    result["target_available"] = available
    result.loc[~available, "split"] = "excluded"
    assert_no_group_leakage(result, "patient_id")
    active_counts = result.loc[result["target_available"], "split"].value_counts()
    empty = [name for name in ("train", "val", "test") if active_counts.get(name, 0) == 0]
    if empty:
        raise ValueError(f"LOPO split has no target-available rows for: {empty}")
    return result


def load_or_create_splits(
    spots: pd.DataFrame,
    path: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if path.is_file():
        splits = pd.read_parquet(path).sort_values("spot_index").reset_index(drop=True)
        if not splits["spot_id"].equals(spots.sort_values("spot_index")["spot_id"].reset_index(drop=True)):
            raise ValueError(f"Existing split file does not match spots: {path}")
        group_column = choose_group_column(spots)
        assert_no_group_leakage(splits, group_column)
        return splits, group_column
    splits, group_column = create_group_splits(spots, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(path, index=False)
    return splits, group_column
