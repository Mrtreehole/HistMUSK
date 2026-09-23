#!/usr/bin/env python3
"""Evaluate a checkpoint under full, missing-modality, and shuffled controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch
import zarr
from torch.utils.data import DataLoader

from histmusk.data.collate import multimodal_collate_fn
from histmusk.data.dataset import MultimodalSpotDataset
from histmusk.models import build_model
from histmusk.training.checkpointing import load_model_weights
from histmusk.training.evaluator import collect_predictions, save_evaluation
from histmusk.utils.device import print_device_report, resolve_device, resolve_mixed_precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data_cache_hvg40spot")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--qwen-h5ad", type=Path,
        help="Override the checkpoint's external Qwen H5AD path",
    )
    parser.add_argument(
        "--image-features-h5", type=Path,
        help="Override the checkpoint's external image feature HDF5 path",
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
    parser.add_argument("--base-predictions-npz", type=Path)
    parser.add_argument(
        "--save-predictions-npz", type=Path,
        help="Save explicit-ID predictions for the full scenario",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, help="Limit batches for a smoke test")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["full", "no_cell", "no_text", "spot_only", "qwen_shuffled", "cells_shuffled"],
        help="Evaluation scenarios to run (default: all compatible scenarios)",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(args.device or config.get("training", {}).get("device", "auto"))
    mixed_precision = resolve_mixed_precision(
        args.mixed_precision or config.get("training", {}).get("mixed_precision", "auto"), device
    )
    print_device_report(device, mixed_precision)
    configured_qwen_h5ad = args.qwen_h5ad or config.get("data", {}).get("qwen_h5ad_path")
    if configured_qwen_h5ad:
        configured_qwen_h5ad = Path(configured_qwen_h5ad).expanduser()
        configured_qwen_h5ad = (
            configured_qwen_h5ad.resolve()
            if configured_qwen_h5ad.is_absolute()
            else (ROOT / configured_qwen_h5ad).resolve()
        )
    configured_image_features_h5 = (
        args.image_features_h5 or config.get("data", {}).get("image_features_h5_path")
    )
    if configured_image_features_h5:
        configured_image_features_h5 = Path(configured_image_features_h5).expanduser()
        configured_image_features_h5 = (
            configured_image_features_h5.resolve()
            if configured_image_features_h5.is_absolute()
            else (ROOT / configured_image_features_h5).resolve()
        )
    configured_spot_whitelist = (
        args.spot_whitelist or config.get("data", {}).get("spot_whitelist_path")
    )
    if configured_spot_whitelist:
        configured_spot_whitelist = Path(configured_spot_whitelist).expanduser()
        configured_spot_whitelist = (
            configured_spot_whitelist.resolve()
            if configured_spot_whitelist.is_absolute()
            else (ROOT / configured_spot_whitelist).resolve()
        )
    configured_target_scaler = (
        args.target_scaler or config.get("data", {}).get("target_scaler_path")
    )
    if configured_target_scaler:
        configured_target_scaler = Path(configured_target_scaler).expanduser()
        configured_target_scaler = (
            configured_target_scaler.resolve()
            if configured_target_scaler.is_absolute()
            else (ROOT / configured_target_scaler).resolve()
        )
    configured_splits_file = (
        args.splits_file or config.get("data", {}).get("splits_path")
    )
    if configured_splits_file:
        configured_splits_file = Path(configured_splits_file).expanduser()
        configured_splits_file = (
            configured_splits_file.resolve()
            if configured_splits_file.is_absolute()
            else (ROOT / configured_splits_file).resolve()
        )
    configured_base_predictions = (
        args.base_predictions_npz
        or config.get("data", {}).get("base_predictions_npz_path")
    )
    if configured_base_predictions:
        configured_base_predictions = Path(configured_base_predictions).expanduser()
        configured_base_predictions = (
            configured_base_predictions.resolve()
            if configured_base_predictions.is_absolute()
            else (ROOT / configured_base_predictions).resolve()
        )
    dataset_split = None if args.split == "all" else args.split
    dataset = MultimodalSpotDataset(
        args.data_dir, args.target_dir, dataset_split, config["target"]["mode"],
        bool(config["model"].get("enable_cell", True)), bool(config["model"].get("enable_text", True)),
        qwen_h5ad=configured_qwen_h5ad,
        image_features_h5=configured_image_features_h5,
        spot_whitelist=configured_spot_whitelist,
        target_scaler=configured_target_scaler,
        base_predictions_npz=configured_base_predictions,
        splits_file=configured_splits_file,
    )
    print(
        f"evaluation_cohort split={args.split} spots={len(dataset)}; "
        f"whitelist={configured_spot_whitelist or 'none'}; "
        f"whitelist_excluded={dataset.spots_excluded_by_whitelist_in_split}; "
        f"empty_cell_excluded={dataset.empty_spots_excluded}",
        flush=True,
    )
    if configured_qwen_h5ad:
        print(
            f"external_qwen_h5ad={configured_qwen_h5ad}; qwen_dim={dataset.qwen_dim}; "
            f"missing_spots_total={dataset.missing_qwen_spots_total}; "
            f"missing_{args.split}_spots_excluded={dataset.missing_qwen_spots_excluded}",
            flush=True,
        )
    if configured_image_features_h5:
        print(
            f"external_image_features_h5={configured_image_features_h5}; "
            f"image_dim={dataset.image_dim}; "
            f"missing_spots_total={dataset.missing_image_spots_total}; "
            f"missing_{args.split}_spots_excluded={dataset.missing_image_spots_excluded}",
            flush=True,
        )
    features = zarr.open_group(str(args.data_dir / "features.zarr"), mode="r")
    targets = zarr.open_group(str(args.target_dir / "gene_expression.zarr"), mode="r")
    dims = {"cell_dim": features["cell_embeddings"].shape[1], "conch_dim": dataset.image_dim,
            "qwen_dim": dataset.qwen_dim, "num_genes": targets["log_normalized"].shape[1]}
    model = build_model(config, {key: int(value) for key, value in dims.items()})
    load_model_weights(model, args.checkpoint, strict=True)
    model.to(device.device)
    def loader() -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                          collate_fn=multimodal_collate_fn, shuffle=False,
                          pin_memory=device.pin_memory, persistent_workers=args.num_workers > 0)
    scenarios = [
        ("full", "full", None), ("no_cell", "no_cell", None), ("no_text", "no_text", None),
        ("spot_only", "spot_only", None), ("qwen_shuffled", "full", "qwen_shuffled"),
        ("cells_shuffled", "full", "cells_shuffled"),
    ]
    if config["training"].get("stage") == "spot_only":
        scenarios = [("spot_only", "spot_only", None)]
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = [scenario for scenario in scenarios if scenario[0] in requested]
        unavailable = requested.difference(name for name, _, _ in scenarios)
        if unavailable:
            raise ValueError(
                f"Requested scenarios are incompatible with this checkpoint: {sorted(unavailable)}"
            )
    results = {}
    for name, mode, control in scenarios:
        print(f"Evaluating scenario={name} mode={mode} control={control or 'none'}", flush=True)
        dataset.set_negative_control(control, seed=42)
        payload = collect_predictions(
            model, loader(), device, mode, mixed_precision, max_batches=args.max_batches
        )
        results[name] = save_evaluation(payload, dataset.gene_names, args.output_dir / name)
        if name == "full" and args.save_predictions_npz:
            args.save_predictions_npz.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np
            np.savez_compressed(
                args.save_predictions_npz,
                spot_id=np.asarray(payload["spot_id"], dtype=str),
                prediction=np.asarray(payload["prediction"], dtype=np.float32),
                target=np.asarray(payload["target"], dtype=np.float32),
            )
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    rows = [{"condition": name, **metrics} for name, metrics in results.items()]
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
