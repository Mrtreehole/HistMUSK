#!/usr/bin/env python3
"""Run strict leave-one-patient-out HistMUSK training and test evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from histmusk.data.splits import create_leave_one_patient_out_splits
from histmusk.data.target_processing import (
    fit_target_scaler_from_splits,
    prepare_expression_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expression-h5ad", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--qwen-h5ad", type=Path)
    parser.add_argument("--image-features-h5", type=Path)
    parser.add_argument("--spot-whitelist", type=Path)
    parser.add_argument(
        "--patients",
        nargs="+",
        help="Optional held-out patient IDs; default runs every target-available patient",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="Comma-separated devices, for example cuda:0,cuda:1,cuda:2,cuda:3 or cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--mixed-precision", choices=["auto", "on", "off"])
    parser.add_argument(
        "--evaluation-scenarios",
        nargs="+",
        default=["full"],
        choices=["full", "no_cell", "no_text", "spot_only", "qwen_shuffled", "cells_shuffled"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _device_names(value: str) -> list[str]:
    devices: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            item = f"cuda:{item}"
        devices.append(item)
    if not devices:
        raise ValueError("--devices did not contain a device")
    if "cpu" in devices and len(devices) != 1:
        raise ValueError("cpu cannot be combined with CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicates")
    return devices


def _slug(patient_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id).strip("_.-") or "patient"
    digest = hashlib.sha1(patient_id.encode("utf-8")).hexdigest()[:8]
    return f"{clean}_{digest}"


def _append_option(command: list[str], name: str, value: Any | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _validation_map(patients: list[str], seed: int) -> dict[str, str]:
    ordered = np.asarray(sorted(patients), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered)
    return {
        str(test): str(ordered[(index + 1) % len(ordered)])
        for index, test in enumerate(ordered)
    }


def _load_available_spots(data_dir: Path, target_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    spots = pd.read_parquet(data_dir / "spots.parquet").sort_values("spot_index").reset_index(drop=True)
    target_index = pd.read_parquet(target_dir / "target_index.parquet")
    if not spots["spot_id"].astype(str).is_unique:
        raise ValueError("spots.parquet contains duplicate spot IDs")
    if not target_index["spot_id"].astype(str).is_unique:
        raise ValueError("target_index.parquet contains duplicate spot IDs")
    available_by_id = target_index.assign(
        spot_id=target_index["spot_id"].astype(str)
    ).set_index("spot_id")["target_available"]
    available = available_by_id.reindex(spots["spot_id"].astype(str))
    if available.isna().any():
        missing = spots.loc[available.isna().to_numpy(), "spot_id"].astype(str).tolist()
        raise ValueError(f"Target index is missing multimodal spots: {missing[:10]}")
    patient = spots["patient_id"].fillna("").astype(str).str.strip()
    if patient.eq("").any():
        raise ValueError("Patient LOPO requires a non-empty patient_id for every spot")
    spots["patient_id"] = patient
    return spots, available.to_numpy(dtype=bool)


def _fold_paths(output_root: Path, patient_id: str) -> dict[str, Path]:
    fold = output_root / f"held_out_{_slug(patient_id)}"
    return {
        "fold": fold,
        "split_dir": fold / "split",
        "splits": fold / "split" / "splits.parquet",
        "scaler": fold / "split" / "target_scaler.npz",
        "manifest": fold / "split" / "fold_manifest.json",
        "training": fold / "training",
        "evaluation": fold / "test_evaluation",
        "predictions": fold / "test_evaluation" / "full_predictions.npz",
    }


def _make_fold(
    args: argparse.Namespace,
    spots: pd.DataFrame,
    available: np.ndarray,
    test_patient: str,
    val_patient: str,
) -> dict[str, Path]:
    paths = _fold_paths(args.output_root, test_patient)
    if paths["fold"].exists() and args.overwrite:
        shutil.rmtree(paths["fold"])
    paths["split_dir"].mkdir(parents=True, exist_ok=True)
    splits = create_leave_one_patient_out_splits(
        spots, test_patient=test_patient, val_patient=val_patient, target_available=available
    )
    splits.to_parquet(paths["splits"], index=False)
    scaler = fit_target_scaler_from_splits(args.target_dir, splits, paths["scaler"])
    counts = (
        splits.groupby("split", observed=True)
        .agg(spots=("spot_index", "size"), patients=("patient_id", "nunique"))
        .to_dict(orient="index")
    )
    manifest = {
        "protocol": "strict_leave_one_patient_out",
        "test_patient": test_patient,
        "validation_patient": val_patient,
        "training_patients": sorted(
            splits.loc[splits["split"].eq("train"), "patient_id"].astype(str).unique().tolist()
        ),
        "split_counts": counts,
        "target_scaler": scaler,
        "seed": args.seed,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def _fold_complete(paths: dict[str, Path]) -> bool:
    return (
        (paths["evaluation"] / "metrics.json").is_file()
        and (paths["evaluation"] / "full" / "metrics.json").is_file()
        and (paths["evaluation"] / "full" / "per_gene_metrics.csv").is_file()
    )


def _run_fold(
    args: argparse.Namespace,
    device: str,
    test_patient: str,
    val_patient: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    if _fold_complete(paths) and not args.overwrite:
        print(f"SKIP completed held_out={test_patient}", flush=True)
        return {"patient_id": test_patient, "status": "skipped", "device": device}
    if paths["training"].exists() and any(paths["training"].iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Incomplete fold output exists for {test_patient}; rerun with --overwrite: {paths['fold']}"
        )

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--config", str(args.config),
        "--data-dir", str(args.data_dir),
        "--expression-h5ad", str(args.expression_h5ad),
        "--target-dir", str(args.target_dir),
        "--target-scaler", str(paths["scaler"]),
        "--splits-file", str(paths["splits"]),
        "--output-dir", str(paths["training"]),
        "--run-name", f"patient_lopo_{_slug(test_patient)}",
        "--device", device,
        "--seed", str(args.seed),
    ]
    for option, value in (
        ("--qwen-h5ad", args.qwen_h5ad),
        ("--image-features-h5", args.image_features_h5),
        ("--spot-whitelist", args.spot_whitelist),
        ("--max-epochs", args.max_epochs),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--max-train-batches", args.max_train_batches),
        ("--max-val-batches", args.max_val_batches),
        ("--mixed-precision", args.mixed_precision),
    ):
        _append_option(train_command, option, value)
    if args.overwrite:
        train_command.append("--overwrite")

    print(
        f"START held_out={test_patient} validation={val_patient} device={device}", flush=True
    )
    _run(train_command, paths["fold"] / "train.log")
    checkpoint = paths["training"] / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce best checkpoint: {checkpoint}")

    evaluate_command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--data-dir", str(args.data_dir),
        "--target-dir", str(args.target_dir),
        "--checkpoint", str(checkpoint),
        "--target-scaler", str(paths["scaler"]),
        "--splits-file", str(paths["splits"]),
        "--split", "test",
        "--output-dir", str(paths["evaluation"]),
        "--save-predictions-npz", str(paths["predictions"]),
        "--device", device,
        "--scenarios", *args.evaluation_scenarios,
    ]
    for option, value in (
        ("--qwen-h5ad", args.qwen_h5ad),
        ("--image-features-h5", args.image_features_h5),
        ("--spot-whitelist", args.spot_whitelist),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
        ("--max-batches", args.max_eval_batches),
        ("--mixed-precision", args.mixed_precision),
    ):
        _append_option(evaluate_command, option, value)
    _run(evaluate_command, paths["fold"] / "evaluate.log")
    print(f"DONE held_out={test_patient} device={device}", flush=True)
    return {"patient_id": test_patient, "status": "completed", "device": device}


def _aggregate(output_root: Path, patients: list[str]) -> None:
    summary_rows: list[dict[str, Any]] = []
    per_gene_frames: list[pd.DataFrame] = []
    prediction_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    spot_id_blocks: list[np.ndarray] = []
    patient_blocks: list[np.ndarray] = []
    gene_names: np.ndarray | None = None

    for patient in patients:
        paths = _fold_paths(output_root, patient)
        metrics_path = paths["evaluation"] / "full" / "metrics.json"
        gene_path = paths["evaluation"] / "full" / "per_gene_metrics.csv"
        if not metrics_path.is_file() or not gene_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        summary_rows.append(
            {
                "held_out_patient": patient,
                "validation_patient": manifest["validation_patient"],
                "train_spots": manifest["split_counts"]["train"]["spots"],
                "validation_spots": manifest["split_counts"]["val"]["spots"],
                "test_spots": manifest["split_counts"]["test"]["spots"],
                **metrics,
            }
        )
        genes = pd.read_csv(gene_path).assign(held_out_patient=patient)
        per_gene_frames.append(genes)

        if paths["predictions"].is_file():
            payload = np.load(paths["predictions"], allow_pickle=False)
            scaler = np.load(paths["scaler"], allow_pickle=False)
            mean = scaler["train_mean"].astype(np.float64)
            std = scaler["train_std"].astype(np.float64)
            prediction_blocks.append(payload["prediction"].astype(np.float64) * std + mean)
            target_blocks.append(payload["target"].astype(np.float64) * std + mean)
            spot_id_blocks.append(payload["spot_id"].astype(str))
            patient_blocks.append(np.repeat(patient, len(payload["spot_id"])))
            current_genes = scaler["gene_names"].astype(str)
            if gene_names is None:
                gene_names = current_genes
            elif not np.array_equal(gene_names, current_genes):
                raise ValueError("Gene order differs between LOPO folds")

    if not summary_rows:
        raise RuntimeError("No completed full-scenario LOPO folds were found")
    summary = pd.DataFrame(summary_rows).sort_values("held_out_patient")
    summary.to_csv(output_root / "lopo_patient_metrics.csv", index=False)
    numeric = summary.select_dtypes(include=[np.number])
    aggregate_metrics = {
        "completed_patients": int(len(summary)),
        "macro_mean": numeric.mean().to_dict(),
        "macro_std": numeric.std(ddof=1).to_dict(),
    }

    per_gene = pd.concat(per_gene_frames, ignore_index=True)
    per_gene.to_csv(output_root / "lopo_per_gene_metrics_long.csv", index=False)
    per_gene.pivot(index="gene", columns="held_out_patient", values="pearson").to_csv(
        output_root / "lopo_gene_pcc_by_patient.csv"
    )

    if prediction_blocks and gene_names is not None:
        prediction = np.concatenate(prediction_blocks)
        target = np.concatenate(target_blocks)
        spot_ids = np.concatenate(spot_id_blocks)
        patient_ids = np.concatenate(patient_blocks)
        if len(np.unique(spot_ids)) != len(spot_ids):
            raise ValueError("A spot appears in more than one held-out fold")
        np.savez_compressed(
            output_root / "lopo_oof_predictions_log_normalized.npz",
            spot_id=spot_ids,
            patient_id=patient_ids,
            gene_names=gene_names,
            prediction=prediction.astype(np.float32),
            target=target.astype(np.float32),
        )
        pcc: list[float] = []
        rho: list[float] = []
        for index in range(prediction.shape[1]):
            pred = prediction[:, index]
            truth = target[:, index]
            if np.std(pred) == 0 or np.std(truth) == 0:
                pcc.append(0.0)
                rho.append(0.0)
            else:
                pcc.append(float(np.corrcoef(pred, truth)[0, 1]))
                rho.append(float(spearmanr(pred, truth).statistic))
        pooled = pd.DataFrame({"gene": gene_names, "pearson": pcc, "spearman": rho})
        pooled.to_csv(output_root / "lopo_pooled_per_gene_metrics.csv", index=False)
        aggregate_metrics["pooled_oof"] = {
            "spots": int(len(prediction)),
            "gene_pcc_mean": float(np.mean(pcc)),
            "gene_spearman_mean": float(np.mean(rho)),
            "mse_log_normalized": float(np.mean(np.square(prediction - target))),
        }
    (output_root / "lopo_summary.json").write_text(
        json.dumps(aggregate_metrics, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.expression_h5ad = args.expression_h5ad.expanduser().resolve()
    args.target_dir = args.target_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for name in ("qwen_h5ad", "image_features_h5", "spot_whitelist"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    devices = _device_names(args.devices)
    if "full" not in args.evaluation_scenarios:
        raise ValueError(
            "Patient LOPO aggregation requires the full evaluation scenario"
        )

    prepare_expression_targets(
        args.data_dir, args.expression_h5ad, args.target_dir, args.seed, False
    )
    spots, available = _load_available_spots(args.data_dir, args.target_dir)
    all_patients = sorted(spots.loc[available, "patient_id"].astype(str).unique().tolist())
    if len(all_patients) < 3:
        raise ValueError("Strict patient LOPO requires at least three target-available patients")
    selected = all_patients if args.patients is None else [str(value) for value in args.patients]
    unknown = sorted(set(selected).difference(all_patients))
    if unknown:
        raise ValueError(f"Unknown held-out patients: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("--patients contains duplicates")
    validation = _validation_map(all_patients, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    fold_paths = {
        patient: _make_fold(args, spots, available, patient, validation[patient])
        for patient in selected
    }
    protocol = {
        "protocol": "strict_leave_one_patient_out",
        "all_patients": all_patients,
        "held_out_patients": selected,
        "validation_patient_by_test_patient": {patient: validation[patient] for patient in selected},
        "devices": devices,
        "seed": args.seed,
        "fold_parallelism": len(devices),
    }
    (args.output_root / "lopo_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    if args.dry_run:
        return

    queues = [selected[index::len(devices)] for index in range(len(devices))]

    def run_queue(device: str, patients: list[str]) -> list[dict[str, Any]]:
        return [
            _run_fold(args, device, patient, validation[patient], fold_paths[patient])
            for patient in patients
        ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_queue, device, queue)
            for device, queue in zip(devices, queues)
            if queue
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    _aggregate(args.output_root, selected)
    print(f"LOPO complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
