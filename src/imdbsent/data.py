"""Dataset loading, normalisation, truncation and the train_fit / validation split.

Outputs (data/processed/):
  train_fit.parquet   22,500 reviews: fit TF-IDF+LR, pool for few-shot demonstrations
  validation.parquet   2,500 reviews: development and configuration decisions
  test.parquet        25,000 reviews: official test split, guarded until freeze
  data_manifest.json  provenance, counts, truncation statistics, duplicate checks
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Paths, file_sha256
from .provenance import manifest

_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")
TRUNC_JOINER = " ... "


# --------------------------------------------------------------------------- text
def normalize_text(text: str, replace_br: bool = True) -> str:
    if replace_br:
        text = _BR.sub(" ", text)
    return _WS.sub(" ", text).strip()


def truncate_words(text: str, max_words: int, head_words: int) -> tuple[str, int, bool]:
    """Head+tail truncation on whitespace words. Returns (text, n_words_original, truncated)."""
    words = text.split()
    n = len(words)
    if n <= max_words:
        return text, n, False
    if not 0 < head_words < max_words:
        raise ValueError("head_words must be in (0, max_words)")
    tail = max_words - head_words
    return " ".join(words[:head_words]) + TRUNC_JOINER + " ".join(words[-tail:]), n, True


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- loading
def load_raw(cfg: dict[str, Any], source: str = "hub", parquet_dir: str | None = None
             ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return {'train': df, 'test': df} with columns text,label and a provenance dict.

    The 'unsupervised' split (50,000 unlabelled reviews) is never loaded.
    """
    dcfg = cfg["data"]
    prov: dict[str, Any] = {"source": source, "hf_dataset": dcfg["hf_dataset"],
                            "hf_config": dcfg["hf_config"], "hf_revision_requested": dcfg.get("hf_revision")}
    frames: dict[str, pd.DataFrame] = {}
    if source == "hub":
        from datasets import load_dataset

        for split in ("train", "test"):
            ds = load_dataset(dcfg["hf_dataset"], dcfg["hf_config"], split=split,
                              revision=dcfg.get("hf_revision"))
            frames[split] = ds.to_pandas()[["text", "label"]]
        try:
            from huggingface_hub import HfApi

            prov["hf_revision_resolved"] = HfApi().dataset_info(
                dcfg["hf_dataset"], revision=dcfg.get("hf_revision")).sha
        except Exception as exc:  # offline or API change
            prov["hf_revision_resolved"] = None
            prov["hf_revision_error"] = repr(exc)
    elif source == "parquet":
        if not parquet_dir:
            raise ValueError("--parquet-dir is required for source=parquet")
        pdir = Path(parquet_dir)
        prov["files"] = {}
        # A file named SYNTHETIC marks fixture data for pipeline tests; the flag propagates
        # into every manifest and results banner.
        prov["synthetic"] = (pdir / "SYNTHETIC").exists()
        for split in ("train", "test"):
            files = sorted(pdir.glob(f"{split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"No {split}-*.parquet in {pdir}")
            frames[split] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)[["text", "label"]]
            prov["files"].update({f.name: file_sha256(f) for f in files})
    else:
        raise ValueError(f"Unknown source {source!r}")

    for split, df in frames.items():
        bad = set(df["label"].unique()) - {0, 1}
        if bad:
            raise ValueError(f"Unexpected labels in {split}: {bad}")
    return frames, prov


# --------------------------------------------------------------------------- ids and split
def add_ids(df: pd.DataFrame, prefix: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Normalise text, add content-based review IDs, truncation fields."""
    pp = cfg["preprocessing"]
    tr = pp["truncation"]
    out = df.copy()
    out["source_row"] = np.arange(len(out))
    out["text_norm"] = [normalize_text(t, pp["replace_br"]) for t in out["text"]]
    out["sha1"] = [text_sha1(t) for t in out["text_norm"]]
    # Content-based ID that does not depend on row order; exact duplicates get a suffix.
    out["dup_index"] = out.groupby("sha1").cumcount()
    out["review_id"] = [f"{prefix}-{s[:12]}-{d}" for s, d in zip(out["sha1"], out["dup_index"])]
    trunc = [truncate_words(t, tr["max_words"], tr["head_words"]) for t in out["text_norm"]]
    out["text_eval"] = [t[0] for t in trunc]
    out["n_words"] = [t[1] for t in trunc]
    out["truncated"] = [t[2] for t in trunc]
    out = out.drop(columns=["text"])
    assert out["review_id"].is_unique
    return out


def stratified_pick(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Boolean mask selecting n rows stratified by label.

    Rows are first put in a canonical order (label, sha1, dup_index) so the
    selection depends only on content + seed, not on the source file order.
    """
    canon = df.sort_values(["label", "sha1", "dup_index"]).index
    rng = np.random.default_rng(seed)
    mask = pd.Series(False, index=df.index)
    counts = df["label"].value_counts().sort_index()
    # Largest-remainder allocation of n across labels.
    exact = counts / counts.sum() * n
    alloc = np.floor(exact).astype(int)
    for lab in (exact - alloc).sort_values(ascending=False).index[: n - alloc.sum()]:
        alloc[lab] += 1
    for lab, k in alloc.items():
        idx = [i for i in canon if df.at[i, "label"] == lab]
        chosen = rng.choice(len(idx), size=k, replace=False)
        mask.loc[[idx[c] for c in chosen]] = True
    return mask.to_numpy()


def prepare(cfg: dict[str, Any], paths: Paths, source: str = "hub",
            parquet_dir: str | None = None) -> dict[str, Any]:
    frames, prov = load_raw(cfg, source, parquet_dir)
    train = add_ids(frames["train"], "tr", cfg)
    test = add_ids(frames["test"], "te", cfg)

    seed = cfg["seed"]
    val_mask = stratified_pick(train, cfg["data"]["validation_size"], seed)
    validation = train[val_mask].reset_index(drop=True)
    train_fit = train[~val_mask].reset_index(drop=True)
    validation["in_dev_subset"] = stratified_pick(
        validation, cfg["data"]["dev_llm_subset_size"], seed + 1)

    # Demonstration eligibility (train_fit only). Exclude any text that also appears
    # in validation or test, or more than once in train_fit, so a demonstration can
    # never be (a copy of) an evaluated review. Comparison uses text only, no labels.
    fs = cfg["few_shot"]
    other_text = set(validation["sha1"]) | set(test["sha1"])
    dup_in_fit = train_fit["sha1"].duplicated(keep=False)
    train_fit["demo_eligible"] = (
        train_fit["n_words"].between(fs["demo_min_words"], fs["demo_max_words"])
        & ~train_fit["sha1"].isin(other_text) & ~dup_in_fit
    )

    paths.processed.mkdir(parents=True, exist_ok=True)
    cols = ["review_id", "source_row", "label", "text_norm", "text_eval", "n_words",
            "truncated", "sha1", "dup_index"]
    train_fit[cols + ["demo_eligible"]].to_parquet(paths.split_file("train_fit"), index=False)
    validation[cols + ["in_dev_subset"]].to_parquet(paths.split_file("validation"), index=False)
    test[cols].to_parquet(paths.split_file("test"), index=False)

    def stats(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "n": int(len(df)),
            "label_counts": {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "n_truncated": int(df["truncated"].sum()),
            "pct_truncated": round(100 * float(df["truncated"].mean()), 2),
            "words_median": float(df["n_words"].median()),
            "words_p90": float(df["n_words"].quantile(0.9)),
            "words_max": int(df["n_words"].max()),
        }

    s_fit, s_val, s_test = set(train_fit["sha1"]), set(validation["sha1"]), set(test["sha1"])
    man = manifest(
        cfg,
        provenance=prov,
        splits={"train_fit": stats(train_fit), "validation": stats(validation), "test": stats(test)},
        dev_llm_subset=int(validation["in_dev_subset"].sum()),
        demo_eligible=int(train_fit["demo_eligible"].sum()),
        id_overlap={"train_fit&validation": len(set(train_fit.review_id) & set(validation.review_id))},
        # Exact-duplicate texts across splits. Reported, NOT removed: the official
        # split is preserved. Duplicates are excluded from the demonstration pool.
        exact_text_overlap={
            "train_fit&validation": len(s_fit & s_val),
            "train_fit&test": len(s_fit & s_test),
            "validation&test": len(s_val & s_test),
        },
        split_files={n: file_sha256(paths.split_file(n)) for n in ("train_fit", "validation", "test")},
    )
    paths.manifest_file.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man
