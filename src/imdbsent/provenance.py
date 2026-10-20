"""Environment / provenance capture written next to every run."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import platform
import subprocess
import sys
from typing import Any

from .config import PROJECT_ROOT, config_hash

TRACKED_PACKAGES = [
    "datasets", "pandas", "pyarrow", "numpy", "scikit-learn", "vaderSentiment",
    "pyyaml", "joblib", "torch", "transformers", "accelerate", "huggingface-hub",
]


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch  # noqa: WPS433

        info["torch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["torch_cuda"] = None
    return info


def manifest(cfg: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_path": cfg.get("_config_path"),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "packages": package_versions(),
        "git": git_state(),
        "hardware": hardware(),
        **extra,
    }
