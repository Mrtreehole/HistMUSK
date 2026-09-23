"""Reproducible checkpoint save/load helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.device import (
    ResolvedDevice,
    get_accelerator_rng_state,
    set_accelerator_rng_state,
)


def random_state(device: ResolvedDevice | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()
    }
    accelerator_state = get_accelerator_rng_state(device)
    if accelerator_state is not None:
        # Keep the historical key for backward compatibility with existing checkpoints.
        state["cuda"] = accelerator_state
    return state


def restore_random_state(state: dict[str, Any], device: ResolvedDevice | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    set_accelerator_rng_state(state.get("cuda"), device)


def _to_cpu(value: Any) -> Any:
    """Recursively make checkpoint tensor payloads portable across all backends."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return value.__class__((key, _to_cpu(item)) for key, item in value.items())
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    gene_names: list[str],
    target_scaler: dict[str, Any],
    device: ResolvedDevice | None = None,
    amp_scaler: Any | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": _to_cpu(model.state_dict()),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "epoch": epoch, "best_metric": best_metric, "config": config,
        "gene_names": gene_names, "target_scaler": target_scaler,
        "amp_scaler_state_dict": _to_cpu(amp_scaler.state_dict()) if amp_scaler is not None else None,
        "random_state": random_state(device), "checkpoint_format_version": 2,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model_weights(model: torch.nn.Module, path: str | Path, strict: bool = False) -> tuple[list[str], list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(payload["model_state_dict"], strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: ResolvedDevice | None = None,
    restore_rng: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """Load legacy or portable checkpoints through CPU, then populate current-device objects."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng and payload.get("random_state") is not None:
        restore_random_state(payload["random_state"], device)
    return payload
