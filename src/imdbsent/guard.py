"""Test-set discipline: the test split can only be loaded after `freeze`.

`freeze` records the config hash, data/demonstration file hashes and git commit in
frozen.lock. Loading the test split checks that all of these still match, so the
configuration evaluated on test is exactly the one that was frozen after development.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pandas as pd

from .config import Paths, config_hash, file_sha256
from .provenance import git_state


class TestSplitLocked(RuntimeError):
    __test__ = False  # not a pytest test class


def _fingerprint(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    fp = {"config_hash": config_hash(cfg)}
    for name in ("train_fit", "validation", "test"):
        fp[f"{name}.parquet"] = file_sha256(paths.split_file(name))
    fp["demo_sets.json"] = file_sha256(paths.demos_file) if paths.demos_file.exists() else None
    # Trained TF-IDF models (selected on validation) are part of the frozen state.
    for m in sorted(paths.models.glob("*.joblib")) if paths.models.exists() else []:
        fp[f"models/{m.name}"] = file_sha256(m)
    return fp


def freeze(cfg: dict[str, Any], paths: Paths, allow_unpinned: bool = False) -> dict[str, Any]:
    if paths.manifest_file.exists():
        dman = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
        if dman.get("provenance", {}).get("synthetic"):
            raise RuntimeError("Refusing to freeze: data/processed was built from SYNTHETIC fixture data.")
    problems = []
    if not cfg["llm"].get("revision"):
        problems.append("llm.revision is not pinned to a commit SHA")
    if not cfg["data"].get("hf_revision"):
        problems.append("data.hf_revision is not pinned to a commit SHA")
    if not paths.demos_file.exists():
        problems.append("demo_sets.json missing (run select-demos)")
    git = git_state()
    if git.get("dirty"):
        problems.append("git working tree has uncommitted changes")
    if problems and not allow_unpinned:
        raise RuntimeError("Refusing to freeze:\n  - " + "\n  - ".join(problems))
    lock = {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "warnings": problems,
        **_fingerprint(cfg, paths),
    }
    paths.lock.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def check_lock(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    if not paths.lock.exists():
        raise TestSplitLocked(
            "The test split is locked. Finish development on validation, then run "
            "`imdbsent freeze`. See README, 'Test-set discipline'.")
    lock = json.loads(paths.lock.read_text(encoding="utf-8"))
    now = _fingerprint(cfg, paths)
    diff = [k for k, v in now.items() if lock.get(k) != v]
    if diff:
        raise TestSplitLocked(f"Frozen configuration does not match current state: {diff}. "
                              "Re-freezing after looking at test results is not allowed.")
    return lock


def load_split(cfg: dict[str, Any], paths: Paths, split: str) -> pd.DataFrame:
    if split not in ("train_fit", "validation", "test"):
        raise ValueError(split)
    if split == "test":
        check_lock(cfg, paths)
    return pd.read_parquet(paths.split_file(split))
