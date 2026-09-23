from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from histmusk.data.dataset import MultimodalSpotDataset
from histmusk.data.splits import create_leave_one_patient_out_splits
from histmusk.data.target_processing import (
    fit_target_scaler_from_splits,
    prepare_expression_targets,
)
from scripts.create_demo_data import create_demo_dataset


def test_patient_lopo_has_no_leakage_and_train_only_scaler(tmp_path) -> None:
    create_demo_dataset(tmp_path / "demo")
    data_dir = tmp_path / "demo" / "multimodal"
    expression = tmp_path / "demo" / "expression.h5ad"
    target_dir = tmp_path / "demo" / "targets"
    prepare_expression_targets(data_dir, expression, target_dir, seed=42)

    spots = pd.read_parquet(data_dir / "spots.parquet").sort_values("spot_index")
    target_index = pd.read_parquet(target_dir / "target_index.parquet").sort_values(
        "spot_index"
    )
    splits = create_leave_one_patient_out_splits(
        spots,
        test_patient="demo_patient_00",
        val_patient="demo_patient_01",
        target_available=target_index["target_available"].to_numpy(),
    )
    split_path = tmp_path / "fold" / "splits.parquet"
    split_path.parent.mkdir()
    splits.to_parquet(split_path, index=False)
    scaler_path = tmp_path / "fold" / "target_scaler.npz"
    metadata = fit_target_scaler_from_splits(target_dir, splits, scaler_path)

    by_patient = splits.groupby("patient_id")["split"].nunique()
    assert by_patient.max() == 1
    assert set(splits.loc[splits["split"] == "test", "patient_id"]) == {
        "demo_patient_00"
    }
    assert set(splits.loc[splits["split"] == "val", "patient_id"]) == {
        "demo_patient_01"
    }
    assert metadata["training_spots"] == int(splits["split"].eq("train").sum())

    target = zarr.open_group(str(target_dir / "gene_expression.zarr"), mode="r")[
        "log_normalized"
    ]
    train_rows = splits.loc[splits["split"].eq("train"), "spot_index"].to_numpy()
    expected_mean = np.asarray(target.oindex[train_rows, :]).mean(axis=0)
    scaler = np.load(scaler_path, allow_pickle=False)
    np.testing.assert_allclose(scaler["train_mean"], expected_mean, rtol=1e-5, atol=1e-5)

    train = MultimodalSpotDataset(
        data_dir,
        target_dir,
        split="train",
        splits_file=split_path,
        target_scaler=scaler_path,
    )
    validation = MultimodalSpotDataset(
        data_dir,
        target_dir,
        split="val",
        splits_file=split_path,
        target_scaler=scaler_path,
    )
    test = MultimodalSpotDataset(
        data_dir,
        target_dir,
        split="test",
        splits_file=split_path,
        target_scaler=scaler_path,
    )
    assert set(train.spots["patient_id"]).isdisjoint(validation.spots["patient_id"])
    assert set(train.spots["patient_id"]).isdisjoint(test.spots["patient_id"])
    assert set(validation.spots["patient_id"]).isdisjoint(test.spots["patient_id"])
    assert set(test.spots["patient_id"]) == {"demo_patient_00"}
