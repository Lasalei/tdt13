"""Seeded sample of TEST reviews for manual error analysis. Runs no model and changes nothing.

    python scripts/make_error_analysis_sample.py [--root PROJECT_DIR] [--n 10] [--seed 13]

Reads existing predictions from runs/test/<condition>/predictions.jsonl and the review texts
from data/processed/test.parquet through `load_split("test")`, which only works after
`freeze` and checks frozen.lock. Writes, under <root>/results/error_analysis/ (not in git):

  error_analysis_test_sample.csv      one row per selected review (contains review texts)
  error_analysis_test_sample_METHOD.md  selection method, group sizes, seed

Groups (all on the test split, primary truncated-text conditions; an invalid LLM output or a
VADER no-decision counts as wrong, as in the evaluation):
  G1  qwen_zeroshot correct, tfidf_lr wrong
  G2  tfidf_lr correct, qwen_zeroshot wrong
  G3  qwen_zeroshot and tfidf_lr both wrong
  G4  vader wrong, qwen_zeroshot and tfidf_lr both correct
  G5  qwen_2shot_set1 and qwen_2shot_set1_flip predict different labels (incl. invalid)
Requested reviews (--include) are added first as group R0, whatever group they fall in.

Selection: candidates of each group are sorted by review_id; reviews already selected (R0 or
an earlier group) are removed; then up to n are drawn without replacement with
numpy.random.default_rng([seed, group_number]). Each group has its own generator, so a group's
sample does not depend on the size of other groups. G1-G4 are disjoint by definition; the
de-duplication matters for R0 and G5.

On Colab after freeze, run it from a separate clone so the frozen Drive checkout is unchanged:
    git clone -q https://github.com/Lasalei/tdt13.git /content/tdt13-tools
    python /content/tdt13-tools/scripts/make_error_analysis_sample.py --root /content/drive/MyDrive/tdt13 \\
        --include te-0a435a8a6c96-0
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ZS, TF, VD, S1, S1F = "qwen_zeroshot", "tfidf_lr", "vader", "qwen_2shot_set1", "qwen_2shot_set1_flip"
CORE = [ZS, TF, VD, S1, S1F]
GROUPS = [
    ("G1", "qwen_zeroshot correct, tfidf_lr wrong"),
    ("G2", "tfidf_lr correct, qwen_zeroshot wrong"),
    ("G3", "qwen_zeroshot and tfidf_lr both wrong"),
    ("G4", "vader wrong, qwen_zeroshot and tfidf_lr both correct"),
    ("G5", "qwen_2shot_set1 and qwen_2shot_set1_flip predict different labels"),
]


def wide_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per review: true label and pred/correct/raw/score per condition (inner join)."""
    base = None
    for c, df in preds.items():
        d = df[["review_id", "true_label", "pred_label", "correct", "raw_output", "score"]].rename(
            columns={"pred_label": f"pred:{c}", "correct": f"ok:{c}", "raw_output": f"raw:{c}",
                     "score": f"score:{c}"})
        base = d if base is None else base.merge(d.drop(columns="true_label"), on="review_id", how="inner")
    return base


def group_masks(w: pd.DataFrame) -> dict[str, pd.Series]:
    ok = {c: w[f"ok:{c}"].astype(bool) for c in (ZS, TF, VD)}
    return {
        "G1": ok[ZS] & ~ok[TF],
        "G2": ok[TF] & ~ok[ZS],
        "G3": ~ok[ZS] & ~ok[TF],
        "G4": ~ok[VD] & ok[ZS] & ok[TF],
        "G5": w[f"pred:{S1}"] != w[f"pred:{S1F}"],
    }


def select(w: pd.DataFrame, n: int, seed: int, include: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return selected rows (column sample_group) and per-group candidate/selected counts."""
    masks = group_masks(w)
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    stats = {}
    for rid in include:
        if rid not in set(w["review_id"]):
            raise KeyError(f"--include {rid}: not in the test predictions of all core conditions")
        chosen.append(("R0", rid))
        used.add(rid)
    for gi, (g, _) in enumerate(GROUPS, start=1):
        cands = sorted(w.loc[masks[g], "review_id"])
        pool = [r for r in cands if r not in used]
        rng = np.random.default_rng([seed, gi])
        pick = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()) if pool else []
        chosen += [(g, r) for r in pick]
        used.update(pick)
        stats[g] = {"candidates": len(cands), "after_dedup": len(pool), "selected": len(pick)}
    sel = pd.DataFrame(chosen, columns=["sample_group", "review_id"])
    member = {g: set(w.loc[masks[g], "review_id"]) for g, _ in GROUPS}
    sel["also_in_groups"] = [",".join(g for g, _ in GROUPS if r in member[g] and g != sg)
                             for sg, r in zip(sel["sample_group"], sel["review_id"])]
    return sel, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--include", nargs="*", default=["te-0a435a8a6c96-0"])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    from imdbsent.config import Paths, load_config
    from imdbsent.guard import load_split

    project = Path(a.root).resolve()
    cfg, paths = load_config(project / "configs" / "experiment.yaml"), Paths(project)
    test_dir = paths.runs / "test"
    qwen_all = sorted(d.name for d in test_dir.glob("qwen_*/") if (d / "predictions.jsonl").exists())
    missing = [c for c in CORE if not (test_dir / c / "predictions.jsonl").exists()]
    if missing:
        raise SystemExit(f"Missing test predictions for {missing} under {test_dir}")

    preds, manifests = {}, {}
    for c in CORE + [q for q in qwen_all if q not in CORE]:
        preds[c] = pd.read_json(test_dir / c / "predictions.jsonl", lines=True, dtype={"review_id": str})
        manifests[c] = json.loads((test_dir / c / "run_manifest.json").read_text(encoding="utf-8"))
    incomplete = [c for c in CORE if len(preds[c]) < 25_000 or preds[c]["review_id"].duplicated().any()]
    if incomplete:
        raise SystemExit(f"Core test runs incomplete or duplicated: {incomplete}")

    w = wide_table({c: preds[c] for c in CORE})
    sel, stats = select(w, a.n, a.seed, a.include)

    # Texts: only through the frozen-lock check.
    test = load_split(cfg, paths, "test")[["review_id", "text_norm", "text_eval", "n_words", "truncated"]]
    out = sel.merge(w, on="review_id", how="left").merge(test, on="review_id", how="left")
    if out["text_norm"].isna().any():
        raise SystemExit("Some selected review IDs are not in test.parquet")
    others = [q for q in qwen_all if q not in CORE]
    other_raw = {q: preds[q].set_index("review_id")["raw_output"] for q in others}
    other_pred = {q: preds[q].set_index("review_id")["pred_label"] for q in others}
    out["other_qwen_conditions"] = [json.dumps({q: {"pred": other_pred[q].get(r), "raw": other_raw[q].get(r)}
                                                for q in others}, ensure_ascii=False)
                                    for r in out["review_id"]]
    gdef = dict(GROUPS) | {"R0": "explicitly requested review"}
    out.insert(1, "group_definition", out["sample_group"].map(gdef))
    out = out.rename(columns={"text_norm": "full_text", f"score:{VD}": "vader_compound",
                              f"score:{TF}": "tfidf_lr_score"})
    cols = (["sample_group", "group_definition", "also_in_groups", "review_id", "true_label",
             "n_words", "truncated"]
            + [f"{p}:{c}" for c in CORE for p in ("pred", "ok")]
            + [f"raw:{c}" for c in (ZS, S1, S1F)] + ["vader_compound", "tfidf_lr_score",
                                                    "other_qwen_conditions", "full_text", "text_eval"])
    out = out[cols]

    out_dir = Path(a.out_dir) if a.out_dir else paths.results / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "error_analysis_test_sample.csv"
    out.to_csv(csv_path, index=False)

    gpus = sorted({str((m.get("hardware") or {}).get("gpu")) for c, m in manifests.items() if c.startswith("qwen")})
    group_rows = "\n".join(f"| {g} | {d} | {stats[g]['candidates']} | {stats[g]['after_dedup']} | {stats[g]['selected']} |"
                           for g, d in GROUPS)
    method = f"""# Manual error analysis sample (TEST split, final runs)

Created {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by
`scripts/make_error_analysis_sample.py` from existing test predictions. No model was run and no
model or prediction was changed. Texts were read with `load_split("test")`, which requires and
checks `frozen.lock`.

* Project folder: `{project}`
* Conditions (truncated text, primary): `{ZS}`, `{TF}`, `{VD}`, and `{S1}` vs. `{S1F}` (secondary order analysis)
* GPU of the Qwen test runs (from manifests): {', '.join(gpus)}
* Reviews compared (inner join of the five core runs): {len(w)}
* Seed {a.seed}, up to {a.n} reviews per group. Requested reviews (R0): {', '.join(a.include) or 'none'}

## Groups

An invalid LLM output (anything other than `positive`/`negative`, for example `neutral`) and a
VADER no-decision count as wrong, as in the evaluation.

| group | definition | candidates | after de-duplication | selected |
|---|---|---:|---:|---:|
{group_rows}

## Selection procedure

1. Requested reviews are added first as group R0.
2. For G1 to G5 in that order: candidates are sorted by `review_id`, reviews already selected are
   removed, and up to {a.n} are drawn without replacement with
   `numpy.random.default_rng([{a.seed}, group_number])` (group_number 1-5).
3. G1-G4 are disjoint by definition. `also_in_groups` lists the other groups a review belongs to.

## Columns

`full_text` is the whole review after `<br />` removal and whitespace normalisation (`text_norm`);
`text_eval` is the text the methods actually saw (head 250 + tail 150 words if longer than 400).
`pred:*` / `ok:*` are predictions and correctness, `raw:*` the raw Qwen outputs,
`vader_compound` the VADER score, `tfidf_lr_score` the logistic-regression probability (as stored
in the predictions), `other_qwen_conditions` the predictions and raw outputs of all other Qwen
test conditions (JSON).

The CSV contains test review texts. Keep it out of git.
"""
    (out_dir / "error_analysis_test_sample_METHOD.md").write_text(method, encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "rows": len(out), "groups": stats,
                      "requested": a.include}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
