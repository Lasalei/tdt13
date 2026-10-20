"""Prediction files: one JSON object per review, append-only, resumable.

runs/<split>/<condition>/
  predictions.jsonl   one record per review (schema below)
  run_manifest.json   config hash, seeds, model revision, package versions, timings

Record schema (all methods):
  review_id, split, condition, method, true_label, pred_label, correct, valid,
  score, raw_output, demo_ids, n_words, truncated
pred_label is "positive" | "negative" | "invalid" (LLM) | "neutral" (VADER, no decision).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import config_hash

PRED_FILE = "predictions.jsonl"
MANIFEST_FILE = "run_manifest.json"
# Fields that must be identical when resuming a run.
# settings_hash covers only the config sections that affect this method's outputs.
RESUME_KEYS = ("settings_hash", "condition", "split", "method", "demo_ids", "model_revision_resolved")


def _repair_and_read_ids(path: Path) -> set[str]:
    """Read finished review IDs; drop a partially written trailing line after a crash."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    good_bytes = 0
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                ids.add(json.loads(line)["review_id"])
            except (json.JSONDecodeError, KeyError):
                break
            good_bytes += len(line)
    if good_bytes != path.stat().st_size:
        with open(path, "r+b") as f:
            f.truncate(good_bytes)
    return ids


class PredictionWriter:
    def __init__(self, run_dir: Path, run_manifest: dict[str, Any], overwrite: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pred_path = self.run_dir / PRED_FILE
        self.man_path = self.run_dir / MANIFEST_FILE
        if overwrite:
            self.pred_path.unlink(missing_ok=True)
            self.man_path.unlink(missing_ok=True)
        self.done = _repair_and_read_ids(self.pred_path)
        if self.man_path.exists() and self.done:
            old = json.loads(self.man_path.read_text(encoding="utf-8"))
            diff = [k for k in RESUME_KEYS if old.get(k) != run_manifest.get(k)]
            if diff:
                raise RuntimeError(
                    f"Cannot resume {self.run_dir}: settings changed {diff}. "
                    "Use --overwrite to start this run again.")
            run_manifest.setdefault("resumed_from", old.get("created_utc"))
            run_manifest["sessions"] = old.get("sessions", []) + [run_manifest["created_utc"]]
        else:
            run_manifest["sessions"] = [run_manifest["created_utc"]]
        self.manifest = run_manifest
        self.write_manifest()
        self._fh = open(self.pred_path, "a", encoding="utf-8")

    def write_manifest(self, **updates: Any) -> None:
        self.manifest.update(updates)
        tmp = self.man_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(tmp, self.man_path)

    def write(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self._fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            self.done.add(r["review_id"])
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def read_predictions(run_dir: Path) -> pd.DataFrame:
    return pd.read_json(Path(run_dir) / PRED_FILE, lines=True, dtype={"review_id": str})


METHOD_SECTIONS = {
    "vader": ("vader",),
    "tfidf_lr": ("tfidf",),
    "qwen": ("llm", "few_shot"),
}


def settings_hash(cfg: dict[str, Any], method: str) -> str:
    sub = {k: cfg[k] for k in ("seed", "data", "preprocessing", *METHOD_SECTIONS[method])}
    return config_hash(sub)


def base_run_manifest(cfg: dict[str, Any], split: str, condition: str, method: str, **extra: Any) -> dict[str, Any]:
    from .provenance import manifest

    return manifest(cfg, split=split, condition=condition, method=method,
                    settings_hash=settings_hash(cfg, method),
                    development_result=(split != "test"), **extra)


__all__ = ["PredictionWriter", "read_predictions", "base_run_manifest", "config_hash"]
