"""Config loading, hashing and project paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


class Paths:
    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root
        self.processed = root / "data" / "processed"
        self.models = root / "models"
        self.runs = root / "runs"
        self.results = root / "results"
        self.lock = root / "frozen.lock"

    def split_file(self, split: str) -> Path:
        return self.processed / f"{split}.parquet"

    @property
    def demos_file(self) -> Path:
        return self.processed / "demo_sets.json"

    @property
    def manifest_file(self) -> Path:
        return self.processed / "data_manifest.json"

    def run_dir(self, split: str, condition: str) -> Path:
        return self.runs / split / condition


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of the parsed config content (comments/whitespace do not matter)."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
