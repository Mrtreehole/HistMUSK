#!/usr/bin/env python3
"""Train HistMUSK from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "histmusk-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "histmusk-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from torch.utils.data import DataLoader

from histmusk.data.collate import multimodal_collate_fn
from histmusk.data.dataset import MultimodalSpotDataset
from histmusk.data.target_processing import prepare_expression_targets
from histmusk.data.splits import ChunkShuffleSampler
from histmusk.models import build_model
from histmusk.models.losses import build_loss
from histmusk.training.checkpointing import load_model_weights, load_training_checkpoint
from histmusk.training.evaluator import collect_predictions, save_evaluation
from histmusk.training.trainer import fit
from histmusk.utils.device import (
    ResolvedDevice,
    print_device_report,
    resolve_device,
    resolve_mixed_precision,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def dimensions(data_dir: Path, target_dir: Path) -> dict[str, int]:
    features = zarr.open_group(str(data_dir / "features.zarr"), mode="r")
    targets = zarr.open_group(str(target_dir / "gene_expression.zarr"), mode="r")
    return {
        "cell_dim": int(features["cell_embeddings"].shape[1]),
        "conch_dim": int(features["conch_embeddings"].shape[1]),
        "qwen_dim": int(features["qwen_embeddings"].shape[1]),
        "num_genes": int(targets["log_normalized"].shape[1]),
    }


def write_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    data_dir: Path,
    target_dir: Path,
    target_scaler: Path,
    splits_file: Path,
) -> None:
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout
    (output_dir / "environment.txt").write_text(environment, encoding="utf-8")
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    (output_dir / "git_commit.txt").write_text(git.stdout.strip() + "\n" if git.returncode == 0 else "unavailable\n")
    manifests = {
        "multimodal": json.loads((data_dir / "manifest.json").read_text(encoding="utf-8")),
        "targets": json.loads((target_dir / "target_manifest.json").read_text(encoding="utf-8")),
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(target_scaler, output_dir / "target_scaler.npz")
    shutil.copy2(splits_file, output_dir / "splits.parquet")


def make_loader(
    dataset: MultimodalSpotDataset,
    config: dict[str, Any],
    shuffle: bool,
    device: ResolvedDevice,
) -> DataLoader:
    workers = int(config["training"].get("num_workers", 4))
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    sampler = ChunkShuffleSampler(dataset.spots["spot_index"].to_numpy(), seed=int(config["training"]["seed"])) if shuffle else None
    return DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, sampler=sampler,
        num_workers=workers, collate_fn=multimodal_collate_fn, pin_memory=device.pin_memory,
        persistent_workers=workers > 0, generator=generator,
    )


def plot_training_curves(path: Path, output: Path) -> None:
    log = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(log["epoch"], log["train_loss"], label="train")
    axes[0].plot(log["epoch"], log["val_loss"], label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(log["epoch"], log["mean_gene_pcc"], label="gene PCC")
    axes[1].plot(log["epoch"], log["mean_spot_pcc"], label="spot PCC")
    axes[1].set(xlabel="Epoch", ylabel="PCC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data_cache_hvg40spot")
    parser.add_argument("--expression-h5ad", type=Path)
    parser.add_argument(
        "--qwen-h5ad", type=Path,
        help="Override features.zarr text embeddings with an explicitly ID-aligned Qwen H5AD",
    )
    parser.add_argument(
        "--image-features-h5", type=Path,
        help="Override stored image rows with explicitly ID-aligned image features",
    )
    parser.add_argument(
        "--spot-whitelist", type=Path,
        help="Parquet/CSV/TSV/text file of explicit spot IDs to retain",
    )
    parser.add_argument(
        "--target-scaler", type=Path,
        help="Override target_scaler.npz, for example with a cohort-fitted scaler",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        help="Override target-dir/splits.parquet with an explicit train/val/test split",
    )
    parser.add_argument(
        "--base-predictions-npz", type=Path,
        help="Explicit-ID frozen OOF/base predictions for residual text correction",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"], default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--cell-dropout", type=float)
    parser.add_argument("--text-dropout", type=float)
    parser.add_argument("--disable-cell", action="store_true")
    parser.add_argument("--disable-text", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--fixed-epochs-no-validation",
        action="store_true",
        help=(
            "Train for exactly --max-epochs using every row labelled train, without "
            "validation, early stopping, or a validation-driven scheduler. The selected "
            "checkpoint is checkpoints/last.pt."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_checkpoint and args.resume_checkpoint:
        raise ValueError("--init-checkpoint and --resume-checkpoint are mutually exclusive")
    if args.resume_checkpoint and args.overwrite:
        raise ValueError("Do not combine --resume-checkpoint with --overwrite")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resume_config: dict[str, Any] = {}
    if args.resume_checkpoint:
        resume_config = torch.load(
            args.resume_checkpoint, map_location="cpu", weights_only=False
        ).get("config", {})
    configured_qwen_h5ad = args.qwen_h5ad or config.get("data", {}).get("qwen_h5ad_path")
    if configured_qwen_h5ad is None and resume_config:
        configured_qwen_h5ad = resume_config.get("data", {}).get("qwen_h5ad_path")
    if configured_qwen_h5ad:
        configured_qwen_h5ad = Path(configured_qwen_h5ad).expanduser()
        if not configured_qwen_h5ad.is_absolute():
            configured_qwen_h5ad = (ROOT / configured_qwen_h5ad).resolve()
        else:
            configured_qwen_h5ad = configured_qwen_h5ad.resolve()
        config.setdefault("data", {})["qwen_h5ad_path"] = str(configured_qwen_h5ad)
    configured_image_features_h5 = (
        args.image_features_h5 or config.get("data", {}).get("image_features_h5_path")
    )
    if configured_image_features_h5 is None and resume_config:
        configured_image_features_h5 = resume_config.get("data", {}).get(
            "image_features_h5_path"
        )
    if configured_image_features_h5:
        configured_image_features_h5 = Path(configured_image_features_h5).expanduser()
        configured_image_features_h5 = (
            configured_image_features_h5.resolve()
            if configured_image_features_h5.is_absolute()
            else (ROOT / configured_image_features_h5).resolve()
        )
        config.setdefault("data", {})["image_features_h5_path"] = str(
            configured_image_features_h5
        )
    configured_spot_whitelist = (
        args.spot_whitelist or config.get("data", {}).get("spot_whitelist_path")
    )
    if configured_spot_whitelist is None and resume_config:
        configured_spot_whitelist = resume_config.get("data", {}).get("spot_whitelist_path")
    if configured_spot_whitelist:
        configured_spot_whitelist = Path(configured_spot_whitelist).expanduser()
        if not configured_spot_whitelist.is_absolute():
            configured_spot_whitelist = (ROOT / configured_spot_whitelist).resolve()
        else:
            configured_spot_whitelist = configured_spot_whitelist.resolve()
        config.setdefault("data", {})["spot_whitelist_path"] = str(configured_spot_whitelist)
    configured_target_scaler = (
        args.target_scaler or config.get("data", {}).get("target_scaler_path")
    )
    if configured_target_scaler is None and resume_config:
        configured_target_scaler = resume_config.get("data", {}).get("target_scaler_path")
    if configured_target_scaler:
        configured_target_scaler = Path(configured_target_scaler).expanduser()
        if not configured_target_scaler.is_absolute():
            configured_target_scaler = (ROOT / configured_target_scaler).resolve()
        else:
            configured_target_scaler = configured_target_scaler.resolve()
    else:
        configured_target_scaler = (args.target_dir / "target_scaler.npz").resolve()
    config.setdefault("data", {})["target_scaler_path"] = str(configured_target_scaler)
    configured_splits_file = (
        args.splits_file or config.get("data", {}).get("splits_path")
    )
    if configured_splits_file is None and resume_config:
        configured_splits_file = resume_config.get("data", {}).get("splits_path")
    if configured_splits_file:
        configured_splits_file = Path(configured_splits_file).expanduser()
        configured_splits_file = (
            configured_splits_file.resolve()
            if configured_splits_file.is_absolute()
            else (ROOT / configured_splits_file).resolve()
        )
    else:
        configured_splits_file = (args.target_dir / "splits.parquet").resolve()
    config.setdefault("data", {})["splits_path"] = str(configured_splits_file)
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
        config.setdefault("data", {})["base_predictions_npz_path"] = str(
            configured_base_predictions
        )
    for key, value in {
        "seed": args.seed, "max_epochs": args.max_epochs, "batch_size": args.batch_size,
        "num_workers": args.num_workers, "learning_rate": args.learning_rate,
    }.items():
        if value is not None:
            config["training"][key] = value
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.mixed_precision is not None:
        config["training"]["mixed_precision"] = args.mixed_precision
    if args.cell_dropout is not None: config["model"]["cell_dropout"] = args.cell_dropout
    if args.text_dropout is not None: config["model"]["text_dropout"] = args.text_dropout
    if args.disable_cell: config["model"]["enable_cell"] = False
    if args.disable_text: config["model"]["enable_text"] = False
    config["run_name"] = args.run_name or args.output_dir.name
    config.setdefault("training", {})["fixed_epochs_no_validation"] = bool(
        args.fixed_epochs_no_validation
    )
    device = resolve_device(config["training"].get("device", "auto"))
    mixed_precision = resolve_mixed_precision(config["training"].get("mixed_precision", "auto"), device)
    print_device_report(device, mixed_precision)
    if config["target"]["mode"] == "raw_counts":
        raise ValueError("raw_counts cannot be trained with MSE; a Negative Binomial decoder is required")
    configured_h5ad = args.expression_h5ad or Path(config["target"]["h5ad_path"])
    if not configured_h5ad.is_absolute():
        configured_h5ad = (ROOT / configured_h5ad).resolve()
    config["target"]["h5ad_path"] = str(configured_h5ad)
    prepare_expression_targets(
        args.data_dir, configured_h5ad, args.target_dir,
        int(config["training"]["seed"]), False,
    )
    if not configured_target_scaler.is_file():
        raise FileNotFoundError(f"Target scaler not found: {configured_target_scaler}")
    if not configured_splits_file.is_file():
        raise FileNotFoundError(f"Split file not found: {configured_splits_file}")
    output_was_nonempty = args.output_dir.exists() and any(args.output_dir.iterdir())
    if output_was_nonempty and not args.resume_checkpoint:
        if not args.overwrite:
            raise FileExistsError(f"Output is non-empty; pass --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    for directory in ("checkpoints", "metrics", "predictions", "plots"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint or not output_was_nonempty:
        write_run_metadata(
            args.output_dir,
            config,
            args.data_dir,
            args.target_dir,
            configured_target_scaler,
            configured_splits_file,
        )
    else:
        with (args.output_dir / "resume_commands.txt").open("a", encoding="utf-8") as handle:
            handle.write(" ".join(sys.argv) + "\n")
    seed = int(config["training"]["seed"])
    set_seed(seed)
    load_cells = bool(config["model"].get("enable_cell", True))
    load_qwen = bool(config["model"].get("enable_text", True))
    train_dataset = MultimodalSpotDataset(
        args.data_dir, args.target_dir, "train", config["target"]["mode"], load_cells, load_qwen,
        qwen_h5ad=configured_qwen_h5ad,
        image_features_h5=configured_image_features_h5,
        spot_whitelist=configured_spot_whitelist,
        target_scaler=configured_target_scaler,
        base_predictions_npz=configured_base_predictions,
        splits_file=configured_splits_file,
    )
    val_dataset = None
    if not args.fixed_epochs_no_validation:
        val_dataset = MultimodalSpotDataset(
            args.data_dir, args.target_dir, "val", config["target"]["mode"], load_cells, load_qwen,
            qwen_h5ad=configured_qwen_h5ad,
            image_features_h5=configured_image_features_h5,
            spot_whitelist=configured_spot_whitelist,
            target_scaler=configured_target_scaler,
            base_predictions_npz=configured_base_predictions,
            splits_file=configured_splits_file,
        )
        if train_dataset.qwen_dim != val_dataset.qwen_dim:
            raise ValueError("Train and validation Qwen dimensions differ")
        if train_dataset.image_dim != val_dataset.image_dim:
            raise ValueError("Train and validation image feature dimensions differ")
    if configured_qwen_h5ad:
        print(
            f"external_qwen_h5ad={configured_qwen_h5ad}; qwen_dim={train_dataset.qwen_dim}; "
            f"missing_spots_total={train_dataset.missing_qwen_spots_total}; "
            f"missing_train_spots_excluded={train_dataset.missing_qwen_spots_excluded}",
            flush=True,
        )
    if configured_image_features_h5:
        print(
            f"external_image_features_h5={configured_image_features_h5}; "
            f"image_dim={train_dataset.image_dim}; "
            f"missing_spots_total={train_dataset.missing_image_spots_total}; "
            f"missing_train_spots_excluded={train_dataset.missing_image_spots_excluded}",
            flush=True,
        )
    print(
        f"cohort train={len(train_dataset)} "
        f"val={len(val_dataset) if val_dataset is not None else 'disabled'}; "
        f"whitelist={configured_spot_whitelist or 'none'}; "
        f"whitelist_excluded_train={train_dataset.spots_excluded_by_whitelist_in_split}; "
        f"whitelist_excluded_val={val_dataset.spots_excluded_by_whitelist_in_split if val_dataset is not None else 0}; "
        f"missing_target_spots_total={train_dataset.missing_target_spots_total}; "
        f"empty_cell_excluded_train={train_dataset.empty_spots_excluded}; "
        f"empty_cell_excluded_val={val_dataset.empty_spots_excluded if val_dataset is not None else 0}",
        flush=True,
    )
    negative_control = config["training"].get("negative_control")
    if negative_control:
        train_dataset.set_negative_control(str(negative_control), seed=seed)
        if val_dataset is not None:
            val_dataset.set_negative_control(str(negative_control), seed=seed)
    train_loader = make_loader(train_dataset, config, True, device)
    val_loader = (
        make_loader(val_dataset, config, False, device) if val_dataset is not None else None
    )
    dims = dimensions(args.data_dir, args.target_dir)
    dims["conch_dim"] = train_dataset.image_dim
    dims["qwen_dim"] = train_dataset.qwen_dim
    model = build_model(config, dims)
    if args.init_checkpoint:
        missing, unexpected = load_model_weights(model, args.init_checkpoint, strict=False)
        print(f"Initialized checkpoint; missing={len(missing)}, unexpected={len(unexpected)}")
    model.to(device.device)
    print(
        f"device={device.device}; backend={device.backend}; dimensions={dims}; "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    criterion = build_loss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_name = str(config["training"].get("scheduler", "ReduceLROnPlateau")).lower()
    if args.fixed_epochs_no_validation or scheduler_name in {"none", "off", "disabled"}:
        scheduler = None
    elif scheduler_name == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['training'].get('scheduler')}")
    start_epoch = 1
    initial_best_metric = -float("inf")
    initial_amp_scaler_state = None
    if args.resume_checkpoint:
        resume_payload = load_training_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, device=device,
            restore_rng=True, strict=True,
        )
        checkpoint_genes = [str(value) for value in resume_payload.get("gene_names", [])]
        if checkpoint_genes != train_dataset.gene_names:
            raise ValueError("Resume checkpoint gene order does not match the current target cache")
        completed_epoch = int(resume_payload["epoch"])
        start_epoch = completed_epoch + 1
        initial_best_metric = float(resume_payload.get("best_metric", -float("inf")))
        initial_amp_scaler_state = resume_payload.get("amp_scaler_state_dict")
        best_checkpoint = args.output_dir / "checkpoints" / "best.pt"
        if best_checkpoint.is_file():
            best_payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            initial_best_metric = max(initial_best_metric, float(best_payload.get("best_metric", -float("inf"))))
        if hasattr(train_loader.sampler, "epoch"):
            train_loader.sampler.epoch = completed_epoch
        print(
            f"Resumed checkpoint={args.resume_checkpoint}; completed_epoch={completed_epoch}; "
            f"next_epoch={start_epoch}; best_metric={initial_best_metric:.6f}",
            flush=True,
        )
    scaler_payload = np.load(configured_target_scaler, allow_pickle=False)
    scaler = {key: scaler_payload[key] for key in scaler_payload.files}
    effective_max_epochs = int(config["training"]["max_epochs"])
    if args.max_train_batches is not None and args.max_epochs is None:
        effective_max_epochs = 1
        print("Smoke batch limit supplied without --max-epochs; running one epoch.", flush=True)
    if start_epoch > effective_max_epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}, but max_epochs={effective_max_epochs}"
        )
    best = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config, args.output_dir,
        train_dataset.gene_names, scaler, device, effective_max_epochs,
        args.max_train_batches, args.max_val_batches, start_epoch, initial_best_metric,
        initial_amp_scaler_state,
    )
    selected_checkpoint = args.output_dir / "checkpoints" / (
        "last.pt" if args.fixed_epochs_no_validation else "best.pt"
    )
    load_model_weights(model, selected_checkpoint, strict=True)
    metrics = None
    if val_loader is not None:
        mode = "spot_only" if config["training"].get("stage") == "spot_only" else "full"
        payload = collect_predictions(
            model, val_loader, device, mode=mode, mixed_precision=mixed_precision,
            max_batches=args.max_val_batches,
        )
        metrics = save_evaluation(
            payload, train_dataset.gene_names, args.output_dir / "metrics" / "validation"
        )
    plot_training_curves(args.output_dir / "train_log.csv", args.output_dir / "plots" / "training_curves.png")
    summary = {"best_validation": best, "final_validation": metrics, "dimensions": dims,
               "parameters": sum(p.numel() for p in model.parameters()),
               "device": str(device.device), "backend": device.backend,
               "gpu_name": device.name if device.backend != "cpu" else None,
               "mixed_precision": mixed_precision,
               "qwen_source": train_dataset.qwen_source,
               "image_source": train_dataset.image_source,
               "spot_whitelist_source": train_dataset.spot_whitelist_source,
               "target_scaler_path": str(configured_target_scaler),
               "splits_path": str(configured_splits_file),
               "base_prediction_source": train_dataset.base_prediction_source,
               "selected_checkpoint": str(selected_checkpoint),
               "fixed_epochs_no_validation": bool(args.fixed_epochs_no_validation),
               "train_spots": len(train_dataset),
               "val_spots": len(val_dataset) if val_dataset is not None else 0,
               "whitelist_excluded_train": train_dataset.spots_excluded_by_whitelist_in_split,
               "whitelist_excluded_val": val_dataset.spots_excluded_by_whitelist_in_split if val_dataset is not None else 0,
               "empty_cell_spots_excluded_train": train_dataset.empty_spots_excluded,
               "empty_cell_spots_excluded_val": val_dataset.empty_spots_excluded if val_dataset is not None else 0,
               "missing_qwen_spots_total": train_dataset.missing_qwen_spots_total,
               "missing_qwen_spots_excluded": train_dataset.missing_qwen_spots_excluded,
               "missing_image_spots_total": train_dataset.missing_image_spots_total,
               "missing_image_spots_excluded": train_dataset.missing_image_spots_excluded}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
