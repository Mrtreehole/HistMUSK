#!/usr/bin/env python3
"""Generate spot-level predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

from histmusk.data.collate import multimodal_collate_fn
from histmusk.data.dataset import MultimodalSpotDataset
from histmusk.models import build_model
from histmusk.training.checkpointing import load_model_weights
from histmusk.training.evaluator import collect_predictions
from histmusk.utils.device import print_device_report, resolve_device, resolve_mixed_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data_cache_hvg40spot")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-h5ad", type=Path, help="Override checkpoint external Qwen H5AD")
    parser.add_argument(
        "--image-features-h5",
        type=Path,
        help="Override checkpoint external image feature HDF5",
    )
    parser.add_argument(
        "--spot-whitelist", type=Path,
        help="Override the checkpoint's explicit spot-ID whitelist",
    )
    parser.add_argument(
        "--target-scaler", type=Path,
        help="Override the checkpoint's cohort-specific target scaler",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        help="Override the checkpoint's train/validation/test split table",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(args.device or config.get("training", {}).get("device", "auto"))
    mixed_precision = resolve_mixed_precision(
        args.mixed_precision or config.get("training", {}).get("mixed_precision", "auto"), device
    )
    print_device_report(device, mixed_precision)
    configured_qwen_h5ad = args.qwen_h5ad or config.get("data", {}).get("qwen_h5ad_path")
    configured_image_features_h5 = (
        args.image_features_h5 or config.get("data", {}).get("image_features_h5_path")
    )
    configured_spot_whitelist = (
        args.spot_whitelist or config.get("data", {}).get("spot_whitelist_path")
    )
    configured_target_scaler = (
        args.target_scaler or config.get("data", {}).get("target_scaler_path")
    )
    configured_splits_file = (
        args.splits_file or config.get("data", {}).get("splits_path")
    )
    resolved_inputs = []
    for value in (
        configured_qwen_h5ad,
        configured_image_features_h5,
        configured_spot_whitelist,
        configured_target_scaler,
        configured_splits_file,
    ):
        if value is None:
            resolved_inputs.append(None)
            continue
        path = Path(value).expanduser()
        resolved_inputs.append(path.resolve() if path.is_absolute() else (ROOT / path).resolve())
    (
        configured_qwen_h5ad,
        configured_image_features_h5,
        configured_spot_whitelist,
        configured_target_scaler,
        configured_splits_file,
    ) = resolved_inputs
    dataset_split = None if args.split == "all" else args.split
    dataset = MultimodalSpotDataset(
        args.data_dir, args.target_dir, dataset_split, config["target"]["mode"],
        bool(config["model"].get("enable_cell", True)),
        bool(config["model"].get("enable_text", True)),
        qwen_h5ad=configured_qwen_h5ad,
        image_features_h5=configured_image_features_h5,
        spot_whitelist=configured_spot_whitelist,
        target_scaler=configured_target_scaler,
        splits_file=configured_splits_file,
    )
    f = zarr.open_group(str(args.data_dir / "features.zarr"), mode="r")
    t = zarr.open_group(str(args.target_dir / "gene_expression.zarr"), mode="r")
    dims = {"cell_dim": f["cell_embeddings"].shape[1], "conch_dim": dataset.image_dim,
            "qwen_dim": dataset.qwen_dim, "num_genes": t["log_normalized"].shape[1]}
    model = build_model(config, {k: int(v) for k, v in dims.items()})
    load_model_weights(model, args.checkpoint, strict=True)
    model.to(device.device)
    print(
        f"prediction_cohort split={args.split} spots={len(dataset)}; "
        f"whitelist={configured_spot_whitelist or 'none'}",
        flush=True,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn, pin_memory=device.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    payload = collect_predictions(model, loader, device, mixed_precision=mixed_precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "predictions.npy", payload["prediction"].astype(np.float32))
    pd.DataFrame({"spot_index": payload["spot_index"], "spot_id": payload["spot_id"]}).to_parquet(
        args.output_dir / "prediction_index.parquet", index=False
    )


if __name__ == "__main__": main()
