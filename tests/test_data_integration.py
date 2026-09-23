from __future__ import annotations

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from histmusk.data.collate import multimodal_collate_fn
from histmusk.data.dataset import MultimodalSpotDataset
from histmusk.data.target_processing import prepare_expression_targets
from scripts.create_demo_data import create_demo_dataset


def test_demo_data_target_alignment_and_batch(tmp_path) -> None:
    created = create_demo_dataset(tmp_path / "demo")
    data_dir = tmp_path / "demo" / "multimodal"
    expression = tmp_path / "demo" / "expression.h5ad"
    target_dir = tmp_path / "demo" / "targets"
    manifest = prepare_expression_targets(data_dir, expression, target_dir, seed=42)

    assert created["spots"] == 48
    assert manifest["alignment"]["target_available_spots"] == 48
    train = MultimodalSpotDataset(data_dir, target_dir, split="train")
    assert len(train) > 0
    first = train[0]
    assert first["cell_embeddings"].shape[0] == first["cell_count"]
    assert first["qwen_embedding"].shape == (128,)
    assert first["coordinate"].shape == (2,)
    assert np.isfinite(first["target"].numpy()).all()

    batch = next(
        iter(
            DataLoader(
                train,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=multimodal_collate_fn,
            )
        )
    )
    assert batch["cell_embeddings"].shape[0] == 4
    assert batch["conch_embedding"].shape == (4, 64)
    assert batch["qwen_embedding"].shape == (4, 128)
    assert batch["target"].shape == (4, 16)


def test_external_features_are_joined_by_spot_id(tmp_path) -> None:
    create_demo_dataset(tmp_path / "demo")
    data_dir = tmp_path / "demo" / "multimodal"
    expression = tmp_path / "demo" / "expression.h5ad"
    target_dir = tmp_path / "demo" / "targets"
    prepare_expression_targets(data_dir, expression, target_dir, seed=42)
    spots = pd.read_parquet(data_dir / "spots.parquet").sort_values("spot_index")

    reversed_spots = spots.iloc[::-1].reset_index(drop=True)
    image_path = tmp_path / "uni_features.h5"
    image_values = np.repeat(
        reversed_spots["spot_index"].to_numpy(dtype=np.float32)[:, None], 10, axis=1
    )
    with h5py.File(image_path, "w") as handle:
        handle.create_dataset("features", data=image_values)
        handle.create_dataset(
            "paths",
            data=np.asarray(
                [f"tiles/{spot_id}.png" for spot_id in reversed_spots["spot_id"]],
                dtype=object,
            ),
            dtype=h5py.string_dtype("utf-8"),
        )
        handle.attrs["model"] = "synthetic-uni"

    text_values = np.repeat(
        reversed_spots["spot_index"].to_numpy(dtype=np.float32)[:, None], 128, axis=1
    )
    text_data = ad.AnnData(
        X=np.zeros((len(reversed_spots), 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"spot_id": reversed_spots["spot_id"].astype(str).to_numpy()},
            index=[f"row_{index}" for index in range(len(reversed_spots))],
        ),
    )
    text_data.obsm["qwen_embedding"] = text_values
    text_path = tmp_path / "text.h5ad"
    text_data.write_h5ad(text_path)

    dataset = MultimodalSpotDataset(
        data_dir,
        target_dir,
        split="train",
        qwen_h5ad=text_path,
        image_features_h5=image_path,
    )
    assert dataset.image_dim == 10
    assert dataset.qwen_dim == 128
    for index in range(min(5, len(dataset))):
        item = dataset[index]
        expected = float(item["spot_index"])
        assert item["conch_embedding"][0].item() == expected
        assert item["qwen_embedding"][0].item() == expected
