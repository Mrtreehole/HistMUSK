#!/usr/bin/env python3
"""Print backend diagnostics and run a real matmul/backward/optimizer step."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from histmusk.utils.device import (
    describe_device,
    detect_available_backends,
    print_device_report,
    resolve_device,
    synchronize_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    resolved = resolve_device(args.device)
    print(f"python_version: {platform.python_version()}")
    print(f"python_executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"torch_cuda_is_available: {torch.cuda.is_available()}")
    print(f"torch_cuda_device_count: {torch.cuda.device_count()}")
    print(f"torch_cuda_device_names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    print(f"available_backends: {json.dumps(detect_available_backends(), indent=2)}")
    print_device_report(resolved, mixed_precision=False)

    left = torch.randn((256, 256), device=resolved.device, requires_grad=True)
    right = torch.randn((256, 256), device=resolved.device, requires_grad=True)
    optimizer = torch.optim.SGD([left, right], lr=1e-5)
    before = left.detach().cpu().clone()
    synchronize_device(resolved)
    started = time.perf_counter()
    loss = (left @ right).square().mean()
    loss.backward()
    optimizer.step()
    synchronize_device(resolved)
    elapsed = time.perf_counter() - started
    result = {
        **describe_device(resolved, mixed_precision=False),
        "matmul_loss": float(loss.detach().cpu()),
        "gradient_finite": bool(torch.isfinite(left.grad.detach().cpu()).all()),
        "parameter_changed": not torch.equal(before, left.detach().cpu()),
        "elapsed_seconds": elapsed,
    }
    print("=== Runtime acceptance probe ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
