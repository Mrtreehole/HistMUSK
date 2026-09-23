"""Runtime utilities shared by training and inference entry points."""

from .device import (
    ResolvedDevice,
    autocast_context,
    create_grad_scaler,
    describe_device,
    detect_available_backends,
    get_accelerator_rng_state,
    print_device_report,
    resolve_device,
    resolve_mixed_precision,
    set_accelerator_rng_state,
    supports_amp,
    synchronize_device,
)

__all__ = [
    "ResolvedDevice",
    "autocast_context",
    "create_grad_scaler",
    "describe_device",
    "detect_available_backends",
    "get_accelerator_rng_state",
    "print_device_report",
    "resolve_device",
    "resolve_mixed_precision",
    "set_accelerator_rng_state",
    "supports_amp",
    "synchronize_device",
]
