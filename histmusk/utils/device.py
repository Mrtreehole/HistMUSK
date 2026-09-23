"""Capability-based PyTorch device selection for Linux NVIDIA CUDA and CPU."""

from __future__ import annotations

import platform
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

import torch

Backend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    device_count: int = 0
    device_names: tuple[str, ...] = ()
    version: str | None = None
    reason: str = ""
    amp_supported: bool = False
    amp_reason: str = ""


@dataclass(frozen=True)
class ResolvedDevice:
    requested: str
    device: torch.device
    backend: Backend
    name: str
    index: int | None
    amp_supported: bool
    amp_reason: str
    pin_memory: bool
    non_blocking: bool

    def __str__(self) -> str:
        return str(self.device)


def _cuda_status() -> BackendStatus:
    if torch.version.hip:
        return BackendStatus(False, reason="This Linux deployment targets NVIDIA CUDA, not ROCm/HIP")
    if not torch.cuda.is_available():
        return BackendStatus(False, reason="torch.cuda.is_available() is False")
    try:
        count = torch.cuda.device_count()
        names = tuple(torch.cuda.get_device_name(index) for index in range(count))
    except Exception as exc:  # pragma: no cover - depends on driver failures
        return BackendStatus(False, reason=f"CUDA device query failed: {type(exc).__name__}: {exc}")
    if count <= 0:
        return BackendStatus(False, reason="CUDA reported no devices")
    return BackendStatus(
        True,
        count,
        names,
        str(torch.version.cuda) if torch.version.cuda else None,
        amp_supported=True,
        amp_reason="native NVIDIA CUDA AMP is available",
    )


@lru_cache(maxsize=1)
def _backend_statuses() -> dict[Backend, BackendStatus]:
    return {
        "cpu": BackendStatus(
            True, 1, (platform.processor() or "CPU",), amp_reason="CPU mixed precision is disabled"
        ),
        "cuda": _cuda_status(),
    }


def detect_available_backends(refresh: bool = False) -> dict[str, dict[str, Any]]:
    if refresh:
        _backend_statuses.cache_clear()
    return {name: asdict(status) for name, status in _backend_statuses().items()}


def resolve_device(requested_device: str | None = "auto") -> ResolvedDevice:
    requested = str(requested_device or "auto").strip().lower()
    statuses = _backend_statuses()
    index: int | None = None
    if requested == "auto":
        selected: Backend = "cuda" if statuses["cuda"].available else "cpu"
        if selected == "cpu":
            message = "!!! WARNING: --device auto found no NVIDIA CUDA GPU; falling back to CPU !!!"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(message, file=sys.stderr, flush=True)
    elif requested.startswith("cuda:"):
        selected = "cuda"
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested_device}") from exc
    elif requested in {"cpu", "cuda"}:
        selected = requested  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown device {requested_device!r}; expected auto, cpu, cuda, or cuda:N")

    status = statuses[selected]
    if not status.available:
        raise RuntimeError(f"Requested device {requested_device!r} is unavailable: {status.reason}")
    if selected == "cuda":
        index = 0 if index is None else index
        if index < 0 or index >= status.device_count:
            raise RuntimeError(f"Requested CUDA index {index}, but only {status.device_count} device(s) are available")
        torch_device = torch.device(f"cuda:{index}")
        pin_memory = True
        non_blocking = True
    else:
        torch_device = torch.device("cpu")
        index = None
        pin_memory = False
        non_blocking = False
    name = status.device_names[index or 0] if status.device_names else selected.upper()
    return ResolvedDevice(
        requested=requested,
        device=torch_device,
        backend=selected,
        name=name,
        index=index,
        amp_supported=status.amp_supported,
        amp_reason=status.amp_reason,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
    )


def supports_amp(device: ResolvedDevice) -> bool:
    return device.amp_supported


def resolve_mixed_precision(setting: str | bool | None, device: ResolvedDevice) -> bool:
    normalized = "auto" if setting is None else str(setting).strip().lower()
    if isinstance(setting, bool):
        requested = setting
    elif normalized == "auto":
        requested = device.amp_supported
    elif normalized in {"true", "on", "yes", "1", "enabled"}:
        requested = True
    elif normalized in {"false", "off", "no", "0", "disabled"}:
        requested = False
    else:
        raise ValueError(f"Unknown mixed_precision setting: {setting!r}")
    if requested and not device.amp_supported:
        warnings.warn(device.amp_reason, RuntimeWarning, stacklevel=2)
        print(f"Mixed precision disabled: {device.amp_reason}", file=sys.stderr, flush=True)
        return False
    return requested


def create_grad_scaler(device: ResolvedDevice, enabled: bool | None = None) -> Any:
    """Create a CUDA scaler with both modern and legacy PyTorch AMP APIs."""
    use_amp = device.amp_supported if enabled is None else bool(enabled and device.amp_supported)
    scaler_enabled = use_amp and device.backend == "cuda"
    amp_module = getattr(torch, "amp", None)
    modern_scaler = getattr(amp_module, "GradScaler", None)
    if modern_scaler is not None:
        return modern_scaler("cuda", enabled=scaler_enabled)
    # PyTorch releases before torch.amp.GradScaler exposed the same scaler here.
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def autocast_context(device: ResolvedDevice, enabled: bool | None = None):
    """Create a CUDA autocast context across old and new PyTorch releases."""
    use_amp = device.amp_supported if enabled is None else bool(enabled and device.amp_supported)
    if use_amp and device.backend == "cuda":
        amp_module = getattr(torch, "amp", None)
        modern_autocast = getattr(amp_module, "autocast", None)
        if modern_autocast is not None:
            return modern_autocast(device_type="cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def synchronize_device(device: ResolvedDevice) -> None:
    if device.backend == "cuda":
        torch.cuda.synchronize(device.index)


def get_accelerator_rng_state(device: ResolvedDevice | None = None) -> Any | None:
    if device is not None and device.backend != "cuda":
        return None
    return torch.cuda.get_rng_state_all() if torch.cuda.is_available() and not torch.version.hip else None


def set_accelerator_rng_state(state: Any | None, device: ResolvedDevice | None = None) -> None:
    if state is None or (device is not None and device.backend != "cuda"):
        return
    if torch.cuda.is_available() and not torch.version.hip:
        torch.cuda.set_rng_state_all(state)


def describe_device(device: ResolvedDevice, mixed_precision: bool | None = None) -> dict[str, Any]:
    return {
        "requested_device": device.requested,
        "resolved_device": str(device.device),
        "backend": device.backend,
        "gpu_name": device.name if device.backend == "cuda" else None,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision": device.amp_supported if mixed_precision is None else bool(mixed_precision),
        "mixed_precision_reason": device.amp_reason,
        "pin_memory": device.pin_memory,
        "non_blocking": device.non_blocking,
    }


def print_device_report(device: ResolvedDevice, mixed_precision: bool) -> None:
    print("=== Accelerator configuration ===", flush=True)
    for key, value in describe_device(device, mixed_precision).items():
        print(f"{key}: {value}", flush=True)
