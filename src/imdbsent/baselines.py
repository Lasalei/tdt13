"""VADER and TF-IDF + logistic regression baselines.

Conditions (the "_fulltext" variants exist only to quantify the effect of truncation;
the primary comparison with the LLM uses `text_eval`, identical for every method):
  vader            VADER on text_eval
  vader_fulltext   VADER on text_norm (untruncated)
  tfidf_lr         TF-IDF + LR, trained and applied on text_eval
  tfidf_lr_fulltext TF-IDF + LR, trained and applied on text_norm
"""
from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from . import LABEL_NAMES, NEUTRAL
from .config import Paths
from .guard import load_split
from .predio import PredictionWriter, base_run_manifest, settings_hash

BASELINE_CONDITIONS = {
    "vader": ("vader", "text_eval"),
    "vader_fulltext": ("vader", "text_norm"),
    "tfidf_lr": ("tfidf_lr", "text_eval"),
    "tfidf_lr_fulltext": ("tfidf_lr", "text_norm"),
}


# --------------------------------------------------------------------------- VADER
def vader_label(compound: float, threshold: float = 0.0) -> str:
    """Map VADER's compound score in [-1, 1] to a dataset label.

    compound >  threshold -> "positive"
    compound <  threshold -> "negative"
    compound == threshold -> "neutral": VADER made no decision. This is counted as an
                             incorrect prediction (like an invalid LLM output) and
                             reported separately. With threshold 0.0 this happens when
                             no lexicon word in the review carries sentiment.
    """
    if compound > threshold:
        return "positive"
    if compound < threshold:
        return "negative"
    return NEUTRAL


def run_vader(cfg: dict[str, Any], paths: Paths, split: str, condition: str, overwrite: bool = False) -> dict[str, Any]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _, field = BASELINE_CONDITIONS[condition]
    vcfg = cfg["vader"]
    df = load_split(cfg, paths, split)
    man = base_run_manifest(cfg, split, condition, "vader", text_field=field, demo_ids=None,
                            model_revision_resolved=None,
                            label_mapping=(f"compound > {vcfg['threshold']} -> positive; "
                                           f"< {vcfg['threshold']} -> negative; "
                                           f"== {vcfg['threshold']} -> neutral (counted incorrect)"))
    w = PredictionWriter(paths.run_dir(split, condition), man, overwrite=overwrite)
    todo = df[~df["review_id"].isin(w.done)]
    sia = SentimentIntensityAnalyzer()
    t0 = time.perf_counter()
    batch = []
    for row in todo.itertuples(index=False):
        compound = sia.polarity_scores(getattr(row, field))["compound"]
        pred = vader_label(compound, vcfg["threshold"])
        true = LABEL_NAMES[row.label]
        batch.append({
            "review_id": row.review_id, "split": split, "condition": condition, "method": "vader",
            "true_label": true, "pred_label": pred, "correct": pred == true,
            "valid": pred != NEUTRAL, "score": compound,
            "in_vader_neutral_band": abs(compound) < vcfg["neutral_band"],
            "raw_output": None, "demo_ids": None,
            "n_words": int(row.n_words), "truncated": bool(row.truncated),
        })
        if len(batch) >= 1000:
            w.write(batch)
            batch = []
    w.write(batch)
    w.write_manifest(n_predictions=len(w.done), seconds_this_session=round(time.perf_counter() - t0, 2))
    w.close()
    return w.manifest


# --------------------------------------------------------------------------- TF-IDF + LR
def _make_pipeline(tcfg: dict[str, Any], C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=tcfg["lowercase"], ngram_range=tuple(tcfg["ngram_range"]),
            min_df=tcfg["min_df"], max_df=tcfg["max_df"], sublinear_tf=tcfg["sublinear_tf"],
            max_features=tcfg["max_features"])),
        ("lr", LogisticRegression(C=C, solver="liblinear", max_iter=tcfg["max_iter"],
                                  random_state=seed)),
    ])


def model_path(paths: Paths, condition: str):
    return paths.models / f"{condition}.joblib"


def fit_tfidf(cfg: dict[str, Any], paths: Paths, condition: str) -> dict[str, Any]:
    """Fit on train_fit only; choose C by validation macro F1 (ties -> smaller C)."""
    _, field = BASELINE_CONDITIONS[condition]
    tcfg = cfg["tfidf"]
    train = load_split(cfg, paths, "train_fit")
    val = load_split(cfg, paths, "validation")
    search = []
    best = None
    for C in sorted(tcfg["C_grid"]):
        pipe = _make_pipeline(tcfg, C, cfg["seed"]).fit(train[field], train["label"])
        f1 = f1_score(val["label"], pipe.predict(val[field]), average="macro")
        search.append({"C": C, "validation_macro_f1": round(float(f1), 6)})
        if best is None or f1 > best[1]:
            best = (C, f1, pipe)
    C, _, pipe = best
    info = {
        "condition": condition, "text_field": field, "selected_C": C, "search": search,
        "fit_on": "train_fit", "n_train": int(len(train)),
        "vocab_size": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        "settings_hash": settings_hash(cfg, "tfidf_lr"),
    }
    paths.models.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "info": info}, model_path(paths, condition))
    return info


def run_tfidf(cfg: dict[str, Any], paths: Paths, split: str, condition: str, overwrite: bool = False) -> dict[str, Any]:
    _, field = BASELINE_CONDITIONS[condition]
    mp = model_path(paths, condition)
    if split == "validation" and (overwrite or not mp.exists()):
        fit_tfidf(cfg, paths, condition)
    if not mp.exists():
        raise FileNotFoundError(f"{mp} missing: run baselines on validation first.")
    bundle = joblib.load(mp)
    info, pipe = bundle["info"], bundle["pipeline"]
    if info["settings_hash"] != settings_hash(cfg, "tfidf_lr"):
        raise RuntimeError(f"{mp} was trained with different settings; rerun on validation.")
    df = load_split(cfg, paths, split)
    man = base_run_manifest(cfg, split, condition, "tfidf_lr", text_field=field, demo_ids=None,
                            model_revision_resolved=None, tfidf_model=info)
    w = PredictionWriter(paths.run_dir(split, condition), man, overwrite=True)  # cheap; always recompute
    t0 = time.perf_counter()
    proba = pipe.predict_proba(df[field])[:, list(pipe.classes_).index(1)]
    preds = np.where(proba >= 0.5, 1, 0)
    recs = []
    for row, p, pr in zip(df.itertuples(index=False), preds, proba):
        pred, true = LABEL_NAMES[int(p)], LABEL_NAMES[row.label]
        recs.append({
            "review_id": row.review_id, "split": split, "condition": condition, "method": "tfidf_lr",
            "true_label": true, "pred_label": pred, "correct": pred == true, "valid": True,
            "score": float(pr), "raw_output": None, "demo_ids": None,
            "n_words": int(row.n_words), "truncated": bool(row.truncated),
        })
    w.write(recs)
    w.write_manifest(n_predictions=len(w.done), seconds_this_session=round(time.perf_counter() - t0, 2))
    w.close()
    return w.manifest


def run_baseline(cfg, paths, split, condition, overwrite=False):
    method, _ = BASELINE_CONDITIONS[condition]
    fn = run_vader if method == "vader" else run_tfidf
    return fn(cfg, paths, split, condition, overwrite=overwrite)


__all__ = ["BASELINE_CONDITIONS", "vader_label", "run_baseline", "fit_tfidf"]
