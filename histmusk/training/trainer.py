"""Training loop for HistMUSK and its modality ablations."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.losses import SACTFLoss, pearson_correlation
from ..utils.device import (
    ResolvedDevice,
    autocast_context,
    create_grad_scaler,
    resolve_mixed_precision,
    synchronize_device,
)
from .checkpointing import save_checkpoint
from .early_stopping import EarlyStopping


def modality_mode_for_stage(stage: str) -> str:
    modes = {"spot_only": "spot_only", "spot_cell": "no_text", "full": "full"}
    try:
        return modes[stage]
    except KeyError as error:
        raise ValueError(f"Unknown training stage: {stage}") from error


def move_batch(batch: dict[str, Any], device: ResolvedDevice) -> dict[str, Any]:
    return {
        key: value.to(device.device, non_blocking=device.non_blocking) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _effective_outputs(outputs: dict[str, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
    if stage != "spot_only":
        return outputs
    result = dict(outputs)
    result["prediction"] = outputs["spot_prediction"]
    return result


def _skip_amp_step_and_reduce_scale(
    scaler: Any, optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """Recover from an AMP overflow without applying contaminated gradients."""
    old_scale = float(scaler.get_scale())
    new_scale = max(old_scale * 0.5, 1.0)
    try:
        scaler.update(new_scale=new_scale)
    except TypeError:  # compatibility with older torch.cuda.amp
        scaler.update(new_scale)
    optimizer.zero_grad(set_to_none=True)
    return old_scale, new_scale


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SACTFLoss,
    device: ResolvedDevice,
    stage: str,
    max_batches: int | None = None,
    mixed_precision: bool = False,
) -> dict[str, float]:
    model.eval()
    modality_mode = modality_mode_for_stage(stage)
    predictions, targets = [], []
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_context(device, mixed_precision):
            outputs = _effective_outputs(
                model(batch, modality_mode=modality_mode, apply_modality_dropout=False), stage
            )
            validation_loss = criterion(outputs, batch["target"])["loss"]
        losses.append(float(validation_loss.detach().cpu()))
        predictions.append(outputs["prediction"].float().cpu())
        targets.append(batch["target"].float().cpu())
    if not predictions:
        raise RuntimeError("Validation loader produced no batches")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    gene_pcc = pearson_correlation(prediction, target, dim=0).mean().item()
    spot_pcc = pearson_correlation(prediction, target, dim=1).mean().item()
    mse = torch.mean((prediction - target).square()).item()
    normalized_mse = mse / max(float(target.var(unbiased=False)), 1e-8)
    score = 0.5 * gene_pcc + 0.5 * spot_pcc - 0.1 * normalized_mse
    return {
        "loss": float(np.mean(losses)), "mse": mse, "mean_gene_pcc": gene_pcc,
        "mean_spot_pcc": spot_pcc, "selection_score": score,
    }


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    criterion: SACTFLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: dict[str, Any],
    output_dir: str | Path,
    gene_names: list[str],
    target_scaler: dict[str, Any],
    device: ResolvedDevice,
    max_epochs: int,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    start_epoch: int = 1,
    initial_best_metric: float = -math.inf,
    initial_amp_scaler_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    stage = str(training.get("stage", "full"))
    modality_mode = modality_mode_for_stage(stage)
    mixed_precision = resolve_mixed_precision(training.get("mixed_precision", "auto"), device)
    scaler = create_grad_scaler(device, mixed_precision)
    if initial_amp_scaler_state:
        scaler.load_state_dict(initial_amp_scaler_state)
        print(f"Restored AMP loss_scale={float(scaler.get_scale()):g}", flush=True)
    use_validation = val_loader is not None
    early = EarlyStopping(int(training.get("early_stopping_patience", 20)), mode="max")
    log_path = output_dir / "train_log.csv"
    fieldnames = [
        "epoch", "train_loss", "val_loss", "val_mse", "mean_gene_pcc", "mean_spot_pcc",
        "selection_score", "train_mse", "train_spot_pcc_loss", "train_gene_pcc_loss",
        "train_spot_aux_loss", "train_residual_loss", "train_consistency_loss",
        "train_text_residual_loss", "train_text_aux_loss",
        "learning_rate", "gradient_norm", "seconds",
    ]
    if start_epoch < 1:
        raise ValueError(f"start_epoch must be positive, got {start_epoch}")
    best_metric = float(initial_best_metric)
    best_result: dict[str, float] = {}
    append_log = start_epoch > 1 and log_path.is_file()
    log_fieldnames = fieldnames
    if append_log:
        with log_path.open("r", encoding="utf-8", newline="") as existing_handle:
            existing_fields = csv.DictReader(existing_handle).fieldnames
        if existing_fields and existing_fields != fieldnames:
            log_fieldnames = existing_fields
            print(
                "Resuming a legacy train_log.csv without per-component loss columns",
                flush=True,
            )
    if append_log and use_validation:
        historical_scores: list[float] = []
        with log_path.open("r", encoding="utf-8", newline="") as historical_handle:
            for row in csv.DictReader(historical_handle):
                if int(row["epoch"]) < start_epoch:
                    historical_scores.append(float(row["selection_score"]))
        if historical_scores:
            historical_best = -math.inf
            bad_epochs = 0
            for score in historical_scores:
                if score > historical_best:
                    historical_best = score
                    bad_epochs = 0
                else:
                    bad_epochs += 1
            early.best = max(historical_best, best_metric)
            early.bad_epochs = bad_epochs if historical_best >= best_metric else 0
            print(
                f"Restored early-stopping state: best={early.best:.6f}; bad_epochs={early.bad_epochs}",
                flush=True,
            )
    with log_path.open("a" if append_log else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=log_fieldnames)
        if not append_log:
            writer.writeheader()
        for epoch in range(start_epoch, max_epochs + 1):
            start_time = time.time()
            model.train()
            train_losses: list[float] = []
            train_components = {
                key: []
                for key in (
                    "mse", "spot_pcc_loss", "gene_pcc_loss", "spot_aux_loss",
                    "residual_loss", "consistency_loss",
                    "text_residual_loss", "text_aux_loss",
                )
            }
            gradient_norms: list[float] = []
            amp_overflow_steps = 0
            max_amp_overflows = int(training.get("max_amp_overflow_steps_per_epoch", 20))
            for batch_index, batch in enumerate(train_loader):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                parameter_probe = None
                if epoch == 1 and batch_index == 0:
                    parameter_probe = next(parameter for parameter in model.parameters() if parameter.requires_grad)
                    parameter_before = parameter_probe.detach().cpu().clone()
                with autocast_context(device, mixed_precision):
                    outputs = _effective_outputs(model(batch, modality_mode=modality_mode), stage)
                    dropped = None
                    if criterion.consistency_weight > 0 and stage == "full":
                        drop_mode = "no_text" if (epoch + batch_index) % 2 == 0 else "no_cell"
                        dropped = model(batch, modality_mode=drop_mode, apply_modality_dropout=False)["prediction"]
                    loss_values = criterion(outputs, batch["target"], dropped)
                if not torch.isfinite(loss_values["loss"]):
                    raise FloatingPointError(
                        f"Non-finite forward loss at epoch={epoch}, batch={batch_index}"
                    )
                scaler.scale(loss_values["loss"]).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("gradient_clip_norm", 1.0))
                )
                if not torch.isfinite(gradient_norm):
                    if not mixed_precision:
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch={epoch}, batch={batch_index}"
                        )
                    amp_overflow_steps += 1
                    old_scale, new_scale = _skip_amp_step_and_reduce_scale(scaler, optimizer)
                    print(
                        f"AMP overflow: epoch={epoch} batch={batch_index}; optimizer step skipped; "
                        f"loss_scale={old_scale:g}->{new_scale:g}; "
                        f"epoch_overflows={amp_overflow_steps}/{max_amp_overflows}",
                        flush=True,
                    )
                    if amp_overflow_steps > max_amp_overflows:
                        raise FloatingPointError(
                            f"Too many AMP overflows in epoch {epoch}: {amp_overflow_steps}"
                        )
                    continue
                scaler.step(optimizer)
                scaler.update()
                if parameter_probe is not None:
                    parameter_updated = not torch.equal(parameter_before, parameter_probe.detach().cpu())
                    print(f"parameter_updated={parameter_updated}", flush=True)
                    if not parameter_updated:
                        raise RuntimeError("Optimizer step did not update the parameter probe")
                train_losses.append(float(loss_values["loss"].detach().cpu()))
                for key in train_components:
                    train_components[key].append(float(loss_values[key].detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))
                if epoch == 1 and batch_index == 0:
                    shapes = {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
                    alpha, beta = outputs["cell_gate"], outputs["text_gate"]
                    print(f"Batch shapes: {shapes}", flush=True)
                    print(
                        f"alpha mean/min/max={alpha.mean().item():.6f}/{alpha.min().item():.6f}/{alpha.max().item():.6f}; "
                        f"beta mean/min/max={beta.mean().item():.6f}/{beta.min().item():.6f}/{beta.max().item():.6f}",
                        flush=True,
                    )
                    print(f"gradient_norm={float(gradient_norm.detach().cpu()):.6f}", flush=True)
                    if outputs["prediction"].shape != batch["target"].shape:
                        raise RuntimeError("Prediction/target shape mismatch")
            if not train_losses:
                raise RuntimeError("Training loader produced no batches")
            synchronize_device(device)
            if use_validation:
                validation = validate_epoch(
                    model, val_loader, criterion, device, stage, max_val_batches, mixed_precision
                )
                metric = validation["selection_score"]
                if scheduler is not None:
                    scheduler.step(metric)
            else:
                validation = {
                    "loss": float("nan"),
                    "mse": float("nan"),
                    "mean_gene_pcc": float("nan"),
                    "mean_spot_pcc": float("nan"),
                    "selection_score": -float(np.mean(train_losses)),
                }
                metric = validation["selection_score"]
            row = {
                "epoch": epoch, "train_loss": float(np.mean(train_losses)),
                "val_loss": validation["loss"], "val_mse": validation["mse"],
                "mean_gene_pcc": validation["mean_gene_pcc"],
                "mean_spot_pcc": validation["mean_spot_pcc"],
                "selection_score": metric,
                "train_mse": float(np.mean(train_components["mse"])),
                "train_spot_pcc_loss": float(np.mean(train_components["spot_pcc_loss"])),
                "train_gene_pcc_loss": float(np.mean(train_components["gene_pcc_loss"])),
                "train_spot_aux_loss": float(np.mean(train_components["spot_aux_loss"])),
                "train_residual_loss": float(np.mean(train_components["residual_loss"])),
                "train_consistency_loss": float(np.mean(train_components["consistency_loss"])),
                "train_text_residual_loss": float(np.mean(train_components["text_residual_loss"])),
                "train_text_aux_loss": float(np.mean(train_components["text_aux_loss"])),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(np.mean(gradient_norms)), "seconds": time.time() - start_time,
            }
            writer.writerow({key: row[key] for key in log_fieldnames})
            handle.flush()
            if use_validation:
                print(
                    f"epoch={epoch} train_loss={row['train_loss']:.6f} "
                    f"val_loss={row['val_loss']:.6f} gene_pcc={row['mean_gene_pcc']:.4f} "
                    f"spot_pcc={row['mean_spot_pcc']:.4f} score={metric:.4f}", flush=True,
                )
            else:
                print(
                    f"epoch={epoch} train_loss={row['train_loss']:.6f} "
                    "fixed_epoch_training=true",
                    flush=True,
                )
            epoch_best_metric = max(best_metric, metric)
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, epoch_best_metric,
                config, gene_names, target_scaler, device, scaler,
            )
            if use_validation and metric > best_metric:
                best_metric = metric
                best_result = validation
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric,
                    config, gene_names, target_scaler, device, scaler,
                )
            if use_validation and early.update(metric):
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
            if not use_validation:
                best_metric = metric
                best_result = {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(train_losses)),
                    "selection_score": metric,
                }
    if not best_result and math.isfinite(best_metric):
        best_result = {"selection_score": best_metric}
    return best_result
