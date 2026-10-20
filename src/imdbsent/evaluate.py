"""Metrics, confusion matrices and cross-method disagreement analysis.

Scoring rules
  * Accuracy: invalid LLM outputs and VADER "neutral" (no decision) count as incorrect.
  * Macro F1 / per-class precision & recall over the two dataset classes. A non-decision
    is a false negative for the true class and a false positive for neither class.
  * Confusion matrix: rows = true {negative, positive}; columns = predicted
    {negative, positive, no_decision}, where no_decision = invalid (LLM) or neutral (VADER).
  * Completeness: a run with fewer predictions than the split size is flagged
    incomplete and excluded from cross-method comparisons.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from . import INVALID, NEUTRAL
from .config import Paths
from .predio import MANIFEST_FILE, PRED_FILE

CLASSES = ["negative", "positive"]
NO_DECISION = "no_decision"


def _status_banner(paths: Paths, split_tag: str) -> str:
    man = json.loads(paths.manifest_file.read_text(encoding="utf-8")) if paths.manifest_file.exists() else {}
    if man.get("provenance", {}).get("synthetic"):
        return ("PIPELINE TEST ON SYNTHETIC DATA - these numbers are NOT experiment results "
                "and must not be reported.")
    if split_tag == "test":
        return "FINAL TEST RESULTS (frozen configuration)."
    return ("DEVELOPMENT RESULTS on validation data (train/validation only). "
            "Not final results; the test split was not used.")


def score_run(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].replace({INVALID: NO_DECISION, NEUTRAL: NO_DECISION}).to_numpy()
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    cm = pd.crosstab(pd.Categorical(y_true, CLASSES), pd.Categorical(y_pred, CLASSES + [NO_DECISION]),
                     dropna=False)
    n_nd = int((y_pred == NO_DECISION).sum())
    return {
        "n": int(len(df)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "n_no_decision": n_nd,
        "pct_no_decision": 100 * n_nd / len(df) if len(df) else 0.0,
        "accuracy_on_decided": float((y_true == y_pred)[y_pred != NO_DECISION].mean()) if n_nd < len(df) else None,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"rows_true": CLASSES, "cols_pred": CLASSES + [NO_DECISION],
                             "counts": cm.to_numpy().tolist()},
    }


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired correctness vectors."""
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only_correct": n01, "b_only_correct": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": min(1.0, 2 * p)}


def load_runs(paths: Paths, split_tag: str) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    base = paths.runs / split_tag
    runs = {}
    for d in sorted(base.glob("*/")) if base.exists() else []:
        if (d / PRED_FILE).exists():
            df = pd.read_json(d / PRED_FILE, lines=True, dtype={"review_id": str})
            man = json.loads((d / MANIFEST_FILE).read_text(encoding="utf-8"))
            runs[d.name] = (df, man)
    if split_tag == "validation-dev":
        # Score the baselines on exactly the same 500 dev reviews as the LLM.
        dev_ids = set(pd.read_parquet(paths.split_file("validation")).query("in_dev_subset")["review_id"])
        for name, (df, man) in load_runs(paths, "validation").items():
            if man.get("method") in ("vader", "tfidf_lr") and name not in runs:
                sub = df[df["review_id"].isin(dev_ids)].reset_index(drop=True)
                runs[name] = (sub, {**man, "n_target": len(dev_ids), "restricted_to": "validation-dev"})
    return runs


def _length_bin(n: int, cfg: dict[str, Any]) -> str:
    mx = cfg["preprocessing"]["truncation"]["max_words"]
    if n <= 100:
        return "001-100"
    if n <= 200:
        return "101-200"
    if n <= mx:
        return f"201-{mx}"
    return f">{mx} (truncated)"


def evaluate(cfg: dict[str, Any], paths: Paths, split_tag: str, out_dir: Path | None = None) -> dict[str, Any]:
    runs = load_runs(paths, split_tag)
    if not runs:
        raise FileNotFoundError(f"No runs under {paths.runs / split_tag}")
    out_dir = out_dir or paths.results / split_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = _status_banner(paths, split_tag)

    metrics, rows, cm_rows = {}, [], []
    for name, (df, man) in runs.items():
        if df["review_id"].duplicated().any():
            raise RuntimeError(f"{name}: duplicate review IDs in predictions")
        m = score_run(df)
        expected = man.get("n_target") or df.shape[0]
        m.update(method=man.get("method"), expected=int(expected), complete=len(df) >= expected,
                 backend=man.get("backend"), demo_ids=man.get("demo_ids"),
                 model_revision=man.get("model_revision_resolved"))
        metrics[name] = m
        rows.append({"condition": name, "method": m["method"], "n": m["n"], "complete": m["complete"],
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "precision_neg": m["per_class"]["negative"]["precision"],
                     "recall_neg": m["per_class"]["negative"]["recall"],
                     "precision_pos": m["per_class"]["positive"]["precision"],
                     "recall_pos": m["per_class"]["positive"]["recall"],
                     "n_no_decision": m["n_no_decision"]})
        for i, t in enumerate(CLASSES):
            for j, pcol in enumerate(CLASSES + [NO_DECISION]):
                cm_rows.append({"condition": name, "true": t, "pred": pcol,
                                "count": m["confusion_matrix"]["counts"][i][j]})
    summary = pd.DataFrame(rows).sort_values("condition")
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion_matrices.csv", index=False)

    # Few-shot: spread over demonstration sets for each k.
    fewshot = {}
    for k in cfg["few_shot"]["shots"]:
        names = [n for n in metrics if n.startswith(f"qwen_{k}shot_set") and metrics[n]["complete"]]
        if names:
            acc = np.array([metrics[n]["accuracy"] for n in names])
            f1 = np.array([metrics[n]["macro_f1"] for n in names])
            fewshot[f"{k}shot"] = {"sets": names, "accuracy_mean": acc.mean(), "accuracy_std": acc.std(ddof=1) if len(acc) > 1 else None,
                                   "accuracy_min": acc.min(), "accuracy_max": acc.max(),
                                   "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(ddof=1) if len(f1) > 1 else None,
                                   "invalid_total": int(sum(metrics[n]["n_no_decision"] for n in names))}

    analysis = disagreement_analysis(cfg, runs, metrics, out_dir)
    result = {"status": banner, "split_tag": split_tag, "metrics": metrics, "few_shot_spread": fewshot,
              "analysis": analysis}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (out_dir / "STATUS.txt").write_text(banner + "\n", encoding="utf-8")
    return result


def disagreement_analysis(cfg, runs, metrics, out_dir: Path) -> dict[str, Any]:
    complete = [n for n in runs if metrics[n]["complete"]]
    if len(complete) < 2:
        return {"note": "fewer than two complete runs; no cross-method analysis"}
    base = None
    for n in complete:
        df = runs[n][0][["review_id", "true_label", "n_words", "truncated", "pred_label", "correct"]]
        df = df.rename(columns={"pred_label": f"pred:{n}", "correct": f"ok:{n}"})
        base = df if base is None else base.merge(
            df.drop(columns=["true_label", "n_words", "truncated"]), on="review_id", how="inner")
    ok_cols = [f"ok:{n}" for n in complete]
    base["n_correct"] = base[ok_cols].sum(axis=1)
    base["pattern"] = np.select(
        [base["n_correct"] == len(complete), base["n_correct"] == 0], ["all_correct", "all_wrong"], "split")
    base["length_bin"] = [_length_bin(n, cfg) for n in base["n_words"]]
    if "vader" in runs and "vader" in complete:
        v = runs["vader"][0][["review_id", "score", "in_vader_neutral_band"]].rename(
            columns={"score": "vader_compound"})
        base = base.merge(v, on="review_id", how="left")
    base.to_csv(out_dir / "per_review_predictions.csv", index=False)

    by_len = base.groupby("length_bin")[ok_cols].mean().rename(columns=lambda c: c[3:])
    by_len["n"] = base.groupby("length_bin").size()
    by_len.to_csv(out_dir / "accuracy_by_length.csv")
    by_true = base.groupby("true_label")[ok_cols].mean().rename(columns=lambda c: c[3:])
    by_true.to_csv(out_dir / "accuracy_by_true_label.csv")
    out = {"n_reviews_compared": int(len(base)), "conditions": complete,
           "pattern_counts": base["pattern"].value_counts().to_dict()}
    if "in_vader_neutral_band" in base:
        out["accuracy_by_vader_neutral_band"] = (
            base.groupby("in_vader_neutral_band")[ok_cols].mean().rename(columns=lambda c: c[3:]).to_dict())

    pairs = []
    for a, b in combinations(complete, 2):
        pa, pb = base[f"pred:{a}"], base[f"pred:{b}"]
        pairs.append({"a": a, "b": b, "agreement": float((pa == pb).mean()),
                      **mcnemar_exact(base[f"ok:{a}"].to_numpy(bool), base[f"ok:{b}"].to_numpy(bool))})
    pd.DataFrame(pairs).to_csv(out_dir / "pairwise_agreement_mcnemar.csv", index=False)
    out["pairwise_file"] = "pairwise_agreement_mcnemar.csv"

    # Sample of split / all-wrong reviews for manual error analysis (text joined later
    # by review_id from data/processed to keep this file small).
    rng = np.random.default_rng(cfg["seed"])
    sample = []
    for pat in ("split", "all_wrong"):
        sub = base[base["pattern"] == pat]
        if len(sub):
            sample.append(sub.iloc[rng.choice(len(sub), size=min(50, len(sub)), replace=False)])
    if sample:
        pd.concat(sample).to_csv(out_dir / "error_analysis_sample.csv", index=False)
    return out
