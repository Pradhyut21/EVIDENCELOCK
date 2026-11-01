"""Readiness evidence; hardware flags are not model feasibility results."""

import platform
import shutil
import subprocess
import sys
from importlib import metadata


class ExecutionBlocked(RuntimeError):
    pass


def doctor():
    packages = {}
    for name in [
        "torch",
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
        "curatorkit",
        "aligntune",
        "auditkit",
        "pydantic",
        "pytest",
        "ruff",
    ]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        observed = {"exit_code": gpu.returncode, "output": gpu.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        observed = {"error_type": type(exc).__name__}
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "isolated": sys.prefix != sys.base_prefix,
        "packages": packages,
        "gpu": observed,
        "disk": shutil.disk_usage(".")._asdict(),
        "build1_complete": False,
    }


def require_gpu():
    try:
        import torch
    except ImportError as exc:
        raise ExecutionBlocked(
            "CUDA PyTorch is not installed in this environment. Use a verified isolated GPU runtime."
        ) from exc
    if not torch.cuda.is_available():
        raise ExecutionBlocked("CUDA is unavailable; no CPU full-model fallback is authorized.")
    name = torch.cuda.get_device_name(0)
    if "MX330" in name:
        raise ExecutionBlocked("The observed MX330 is excluded from full-model inference/training.")
    return torch
