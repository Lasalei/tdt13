"""Build a shareable report package (ZIP) from EXISTING results. Runs no experiments.

    python scripts/make_report_package.py [--root PROJECT_DIR] [--out-dir DIR]

--root points at the project folder holding configs/, data/, runs/ and results/ (default: this
repository). On Colab after freeze, run the script from a separate clone with
--root /content/drive/MyDrive/tdt13, so the frozen Drive checkout is not pulled or changed.

Reads only files that already exist under configs/, data/processed/ (manifest and demo sets),
runs/ and results/. It never loads a data split, never runs `freeze`, and never runs a model.
Works locally and on Colab (run it in the Drive repo folder to include the GPU runs).

Included: config, README, git state, data/truncation/prompt/demo documentation, metrics,
confusion matrices, comparisons, per-review predictions (IDs, labels, raw outputs; no review
text), run manifests, pilot timings, and an inventory of complete/incomplete/missing runs.
Excluded: model files (*.joblib, weights), dataset files (*.parquet, raw data), caches and
anything that looks like a secret. The package is scanned for secret-like strings before
zipping and the build fails if one is found.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT  # set from --root in main()
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from imdbsent.config import Paths, load_config  # noqa: E402
from imdbsent.demos import condition_names, is_flip, primary_condition_names  # noqa: E402
from imdbsent.predio import settings_hash  # noqa: E402
from imdbsent.prompts import build_messages  # noqa: E402

BASELINES = ["vader", "vader_fulltext", "tfidf_lr", "tfidf_lr_fulltext"]
EXCLUDE_SUFFIXES = {".parquet", ".arrow", ".joblib", ".safetensors", ".bin", ".pt", ".pth",
                    ".gguf", ".pkl", ".pem", ".key", ".tmp"}
SECRET_PATTERNS = [re.compile(p) for p in (
    r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9_-]{20,}", r"ghp_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}", r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----")]

# Category of each split tag folder under runs/ and results/.
def category(tag: str) -> str:
    if tag == "test":
        return "test_FINAL"
    if tag.endswith("-t4-autodtype"):
        return "archived_t4"
    if tag == "pilot":
        return "pilot"
    if tag in ("validation", "validation-dev"):
        return "development"
    return "other"


CATEGORY_NOTE = {
    "development": "DEVELOPMENT RESULTS (train/validation only). Not final results.",
    "archived_t4": ("ARCHIVED: earlier dev runs on a T4 with llm.dtype=auto. Kept as a record only; "
                    "superseded by the L4/bfloat16 dev runs and not part of any comparison."),
    "pilot": "PILOT: code check and runtime estimate only. Not an experiment result.",
    "test_FINAL": "FINAL TEST RESULTS (frozen configuration).",
    "other": "Unclassified run folder.",
}


def sh(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=PROJECT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # git missing etc.
        return f"<unavailable: {exc}>"


def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def copy_tree(src: Path, dst: Path) -> list[str]:
    skipped = []
    for f in sorted(src.rglob("*")):
        if f.is_dir():
            continue
        if f.suffix.lower() in EXCLUDE_SUFFIXES or f.name.startswith(".env"):
            skipped.append(str(f.relative_to(PROJECT)))
            continue
        t = dst / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t)
    return skipped


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def csv_rows(p: Path) -> list[dict]:
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ documentation
def doc_data(cfg, paths: Paths) -> str:
    if not paths.manifest_file.exists():
        return "# Data\n\ndata/processed/data_manifest.json not found.\n"
    m = jload(paths.manifest_file)
    tr = cfg["preprocessing"]["truncation"]
    rows = [{"split": k, "n": v["n"], "negative (0)": v["label_counts"].get("0"),
             "positive (1)": v["label_counts"].get("1"), "truncated": v["n_truncated"],
             "% truncated": v["pct_truncated"], "median words": v["words_median"],
             "p90 words": v["words_p90"], "max words": v["words_max"]}
            for k, v in m["splits"].items()]
    prov = m.get("provenance", {})
    return f"""# Data split, truncation and provenance

Source: `{prov.get('hf_dataset')}` (config `{prov.get('hf_config')}`), revision
`{prov.get('hf_revision_resolved')}` (requested `{prov.get('hf_revision_requested')}`), loaded from `{prov.get('source')}`.
Synthetic fixture data: {bool(prov.get('synthetic'))}.

## Splits

* The official IMDb `train` split (25,000) is split once into `train_fit` and a stratified
  `validation` subset of {cfg['data']['validation_size']} (seed {cfg['seed']}). The official `test` split
  (25,000) is kept intact. The `unsupervised` split is never loaded.
* A fixed stratified dev subset of {m.get('dev_llm_subset', cfg['data']['dev_llm_subset_size'])} validation reviews
  (`validation-dev`) is used for LLM development runs (seed {cfg['seed'] + 1}).
* Few-shot demonstrations come only from `train_fit` ({m.get('demo_eligible')} eligible reviews:
  {cfg['few_shot']['demo_min_words']}-{cfg['few_shot']['demo_max_words']} words, text not duplicated in train_fit/validation/test).

{md_table(rows, list(rows[0]))}

ID overlap between splits: `{json.dumps(m.get('id_overlap'))}`
Exact-text duplicates across splits (counted, not removed): `{json.dumps(m.get('exact_text_overlap'))}`

## Text preprocessing and truncation (identical for all methods in the primary comparison)

1. `<br />` replaced by a space; whitespace collapsed (`replace_br: {cfg['preprocessing']['replace_br']}`).
2. Reviews longer than {tr['max_words']} whitespace words keep the first {tr['head_words']} and the last
   {tr['max_words'] - tr['head_words']} words, joined by ` ... ` (field `text_eval`). The same text is used by VADER,
   TF-IDF+LR and Qwen. Full-text baseline variants (`*_fulltext`) are secondary conditions.
3. The LLM tokenizer is called with `truncation=False`; a prompt longer than the model context
   raises an error instead of being cut.

## File fingerprints (SHA-256)

{md_table([{'file': f'{k}.parquet', 'sha256': v} for k, v in m['split_files'].items()], ['file', 'sha256'])}

The full manifest is in `docs/data_manifest.json`.
"""


def doc_prompts(cfg, paths: Paths) -> str:
    if not paths.demos_file.exists():
        return "# Prompts\n\ndemo_sets.json not found.\n"
    demos = jload(paths.demos_file)
    lc = cfg["llm"]
    parts = [f"""# Prompts and demonstration sets

Model `{lc['model']}` at revision `{lc['revision']}`, dtype `{lc['dtype']}`, greedy decoding
(`do_sample={lc['do_sample']}`), `max_new_tokens={lc['max_new_tokens']}`, `repetition_penalty=1.0`, batch size {lc['batch_size']}.
The chat template of the model is applied to the messages below (`add_generation_prompt=True`).

Output parsing (strict): lowercase and strip, remove leading/trailing non-letters, take the first
word; valid only if it is `positive` or `negative`. Anything else is invalid and counts as incorrect.

Demonstration pool: {demos['pool_size']} reviews ({json.dumps(demos['pool_label_counts'])}), seed {demos['seed']}.
Sets (IDs): `{json.dumps(demos['sets'])}`

## Conditions

`_flip` conditions are a **secondary, post-hoc** order analysis (added before freeze after a
development observation): same demonstrations as the original condition, labels swapped inside
each pair. They are not part of the RQ2 spread over sets.
"""]
    rows = []
    for name, c in demos["conditions"].items():
        rows.append({"condition": name, "role": "secondary (order)" if is_flip(name) else "primary",
                     "shots": c["shots"], "set": c["set"],
                     "order": "-".join(l[0].upper() for l in c["demo_labels"]) or "(none)",
                     "demo_ids": ", ".join(c["demo_ids"])})
    parts.append(md_table(rows, ["condition", "role", "shots", "set", "order", "demo_ids"]))
    parts.append("\n## Rendered messages per condition\n\nThe target review is shown as `<TARGET REVIEW>`.\n")
    for name, c in demos["conditions"].items():
        parts.append(f"\n### {name}\n")
        for msg in build_messages(cfg, "<TARGET REVIEW>", c["demos"]):
            body = msg["content"].replace("\n", "\n    ")
            parts.append(f"**{msg['role']}:**\n\n    {body}\n")
    parts.append("\nThe demonstration texts and metadata are in `docs/demo_sets.json`.\n")
    return "\n".join(parts)


# ------------------------------------------------------------------ inventory
def manifest_row(tag: str, cond: str, mp: Path, expected_n: int | None, current: dict) -> dict:
    m = jload(mp)
    n = m.get("n_predictions")
    target = m.get("n_target") or expected_n
    complete = m.get("complete")
    if complete is None and n is not None and target:
        complete = n >= target
    hw, bi = m.get("hardware", {}), m.get("backend_info") or {}
    method = m.get("method")
    want = current.get("qwen" if method == "qwen" else method)
    return {"split_tag": tag, "category": category(tag), "condition": cond, "method": method,
            "status": "complete" if complete else "incomplete", "n_predictions": n, "n_target": target,
            "settings_match_current": (m.get("settings_hash") == want) if want else None,
            "device": bi.get("device") or ("cuda" if hw.get("torch_cuda") else "cpu"),
            "gpu": hw.get("gpu"), "dtype": bi.get("dtype"), "batch_size": m.get("batch_size"),
            "seconds_total": m.get("seconds_total"), "sessions": len(m.get("sessions") or []),
            "created_utc": m.get("created_utc"), "git_commit": (m.get("git") or {}).get("commit"),
            "model_revision": m.get("model_revision_resolved")}


def inventory(cfg, paths: Paths) -> list[dict]:
    current = {"qwen": settings_hash(cfg, "qwen"), "vader": settings_hash(cfg, "vader"),
               "tfidf_lr": settings_hash(cfg, "tfidf_lr")}
    n_val, n_dev = cfg["data"]["validation_size"], cfg["data"]["dev_llm_subset_size"]
    expected = {"validation": [(c, n_val) for c in BASELINES],
                "validation-dev": [(c, n_dev) for c in condition_names(cfg)],
                "test": [(c, 25_000) for c in BASELINES + condition_names(cfg)]}
    rows, seen = [], set()
    tags = sorted({d.name for d in paths.runs.iterdir() if d.is_dir()} if paths.runs.exists() else set())
    for tag in sorted(set(tags) | set(expected)):
        exp = dict(expected.get(tag, []))
        found = sorted(d.name for d in (paths.runs / tag).glob("*/") if (d / "run_manifest.json").exists()) \
            if (paths.runs / tag).exists() else []
        for cond in found:
            rows.append(manifest_row(tag, cond, paths.runs / tag / cond / "run_manifest.json", exp.get(cond), current))
            seen.add((tag, cond))
        for cond, n in expected.get(tag, []):
            if (tag, cond) not in seen:
                note = "not run yet (only after freeze)" if tag == "test" else "not found in this folder"
                rows.append({"split_tag": tag, "category": category(tag), "condition": cond,
                             "status": note, "n_target": n})
    return rows


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT), help="project folder (default: this repository)")
    ap.add_argument("--out-dir", default=None, help="default: <root>/results/report_packages")
    a = ap.parse_args()
    global PROJECT
    PROJECT = Path(a.root).resolve()
    config_file = PROJECT / "configs" / "experiment.yaml"
    cfg, paths = load_config(config_file), Paths(PROJECT)
    a.out_dir = a.out_dir or str(PROJECT / "results" / "report_packages")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"tdt13_report_package_{stamp}"
    out_root = Path(a.out_dir)
    pkg = out_root / name
    pkg.mkdir(parents=True)
    skipped: list[str] = []

    # repo state
    (pkg / "repo").mkdir()
    shutil.copy2(config_file, pkg / "repo" / "experiment.yaml")
    for f in ("README.md", "requirements.txt", "requirements-llm.txt", "requirements-lock.txt"):
        if (PROJECT / f).exists():
            shutil.copy2(PROJECT / f, pkg / "repo" / f)
    git = {"commit": sh("git", "rev-parse", "HEAD"), "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "status_porcelain": sh("git", "status", "--porcelain"), "log": sh("git", "log", "--oneline", "-25"),
           "remote": sh("git", "remote", "get-url", "origin")}
    (pkg / "repo" / "git_state.txt").write_text(
        f"commit: {git['commit']}\nbranch: {git['branch']}\nremote: {git['remote']}\n"
        f"working tree clean: {not git['status_porcelain']}\n\nstatus --porcelain:\n{git['status_porcelain']}\n"
        f"\nrecent commits:\n{git['log']}\n", encoding="utf-8")
    (pkg / "repo" / "config_hashes.json").write_text(json.dumps({
        "config_hash_note": "settings hashes of the current config; run manifests are compared with these",
        "settings_hash": {k: settings_hash(cfg, k) for k in ("qwen", "vader", "tfidf_lr")}}, indent=2))

    # documentation
    (pkg / "docs").mkdir()
    (pkg / "docs" / "data_split_and_truncation.md").write_text(doc_data(cfg, paths), encoding="utf-8")
    (pkg / "docs" / "prompts_and_demo_sets.md").write_text(doc_prompts(cfg, paths), encoding="utf-8")
    for f in (paths.manifest_file, paths.demos_file):
        if f.exists():
            shutil.copy2(f, pkg / "docs" / f.name)

    # results and runs, grouped by category
    present = {}
    for base, kind in ((paths.results, "results"), (paths.runs, "runs")):
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "report_packages"):
            cat = category(d.name)
            skipped += copy_tree(d, pkg / cat / d.name / kind)
            present.setdefault(cat, set()).add(d.name)
    for f in sorted(paths.results.glob("pilot_estimate_*.json")) if paths.results.exists() else []:
        (pkg / "pilot").mkdir(exist_ok=True)
        shutil.copy2(f, pkg / "pilot" / f.name)
        present.setdefault("pilot", set()).add(f.name)
    if paths.lock.exists():
        shutil.copy2(paths.lock, pkg / "repo" / "frozen.lock")
    for cat, note in CATEGORY_NOTE.items():
        if (pkg / cat).exists():
            (pkg / cat / "CATEGORY.txt").write_text(note + "\n", encoding="utf-8")

    # inventory
    inv = inventory(cfg, paths)
    cols = ["split_tag", "category", "condition", "status", "n_predictions", "n_target",
            "settings_match_current", "device", "gpu", "dtype", "batch_size", "seconds_total",
            "sessions", "created_utc", "git_commit", "model_revision"]
    import csv
    with open(pkg / "run_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(inv)

    # pilot timing summary
    pilot_md = []
    for f in sorted((pkg / "pilot").glob("pilot_estimate_*.json")) if (pkg / "pilot").exists() else []:
        p = jload(f)
        b = p.get("backend", {})
        pilot_md.append(f"* `{f.name}`: device `{b.get('device')}`, dtype `{b.get('dtype')}`, batch size "
                        f"{p.get('batch_size', cfg['llm']['batch_size'])}, {p.get('pilot_reviews_per_condition')} reviews per condition.\n"
                        f"  Seconds per review (first batch excluded): `{json.dumps(p.get('seconds_per_review'))}`\n"
                        f"  Diagnostics: `{json.dumps(p.get('diagnostics'))}`\n"
                        f"  Estimated test hours in total: {p.get('estimated_test_hours_total')}"
                        f" (for {len(p.get('estimated_test_hours_per_condition') or {})} conditions in the plan at that time)"
                        + ("" if b.get("device") == "cuda" else
                           f"\n  **Not a GPU pilot (device `{b.get('device')}`): code check only; this timing "
                           "is not used for planning.**"))
    pilot_rows = [r for r in inv if r["split_tag"] == "pilot"]

    # overview
    def summary_block(cat: str) -> str:
        blocks = []
        for tag in sorted(present.get(cat, [])):
            ms = paths.results / tag / "metrics_summary.csv"
            if not ms.exists():
                continue
            st = (paths.results / tag / "STATUS.txt")
            rows = csv_rows(ms)
            for r in rows:
                for k in ("accuracy", "macro_f1", "precision_neg", "recall_neg", "precision_pos", "recall_pos"):
                    if r.get(k):
                        r[k] = f"{float(r[k]):.4f}"
            blocks.append(f"#### `{tag}`\n\nSTATUS: {st.read_text().strip() if st.exists() else '(none)'}\n\n"
                          + md_table(rows, ["condition", "n", "complete", "accuracy", "macro_f1",
                                            "recall_neg", "recall_pos", "n_no_decision"]))
            oe = paths.results / tag / "order_effect.csv"
            if oe.exists():
                orows = csv_rows(oe)
                blocks.append("Order section (secondary, post-hoc; original vs. `_flip`):\n\n" + md_table(
                    orows, ["original", "flipped", "order_original", "order_flipped", "accuracy_original",
                            "accuracy_flipped", "agreement", "a_only_correct", "b_only_correct", "p_value"]))
            mj = paths.results / tag / "metrics.json"
            if mj.exists():
                fs = jload(mj).get("few_shot_spread")
                if fs:
                    blocks.append("Few-shot spread over the original sets (RQ2):\n\n```json\n"
                                  + json.dumps(fs, indent=1, default=str) + "\n```")
        return "\n\n".join(blocks) if blocks else "_No evaluated results in this category in this package._"

    counts = {}
    for r in inv:
        counts.setdefault(r["category"], {}).setdefault(r["status"], 0)
        counts[r["category"]][r["status"]] += 1
    inv_rows = [{"category": c, **{k: v for k, v in s.items()}} for c, s in sorted(counts.items())]
    status_cols = ["category"] + sorted({k for s in counts.values() for k in s})

    readme = f"""# TDT13 report package ({stamp})

Built by `scripts/make_report_package.py` from existing files only. No experiment, model run,
`freeze` or test-split loading was performed to build it.

* Git commit: `{git['commit']}` ({git['branch']}), working tree clean: {not git['status_porcelain']}
* Built from: `{PROJECT}` on `{platform.platform()}`. Runs that exist elsewhere (for example
  only on Google Drive) are listed as "not found in this folder".
* Project: IMDb sentiment, VADER vs. TF-IDF+LR vs. Qwen2.5-1.5B-Instruct (zero/few-shot).

## How to read this package

| folder | content | status |
|---|---|---|
| `development/` | `validation` (baselines, 2,500 reviews) and `validation-dev` (500 reviews, LLM + baselines on the same reviews) | {CATEGORY_NOTE['development']} |
| `archived_t4/` | `validation-dev-t4-autodtype` | {CATEGORY_NOTE['archived_t4']} |
| `pilot/` | pilot runs and `pilot_estimate_*.json` | {CATEGORY_NOTE['pilot']} |
| `test_FINAL/` | test split | {CATEGORY_NOTE['test_FINAL']} Present only after freeze. |
| `docs/` | data split, truncation, prompts, demonstration sets, data manifest | documentation |
| `repo/` | experiment.yaml, README, requirements, git state, current settings hashes | documentation |
| `run_inventory.csv` | every expected and found run: complete / incomplete / missing | overview |

Each results folder holds `metrics_summary.csv`, `metrics.json`, `confusion_matrices.csv`,
`pairwise_agreement_mcnemar.csv`, accuracy by length / true label, `per_review_predictions.csv`
and an error-analysis sample (where available). Each runs folder holds `predictions.jsonl`
(one line per review: review_id, true/predicted label, correctness, raw LLM output, prompt
tokens, length, truncation; **no review text**) and `run_manifest.json` (config and settings
hashes, git commit, packages, hardware, model revision, generation config, timings).

Categories present in this package: {', '.join(sorted(present)) or 'none'}.
Test results present: {'yes' if 'test_FINAL' in present else 'no (the test split has not been run)'}.

## Run inventory

{md_table(inv_rows, status_cols)}

{md_table(inv, ['split_tag', 'condition', 'status', 'n_predictions', 'n_target', 'settings_match_current', 'gpu', 'device', 'dtype'])}

`settings_match_current` compares the run's settings hash with the current config. The archived
T4 runs are expected to differ (dtype `auto`), pilot runs may differ.

## Pilot timings

{chr(10).join(pilot_md) or '_No pilot_estimate file in this package._'}

{md_table(pilot_rows, ['condition', 'status', 'n_predictions', 'device', 'gpu', 'dtype', 'batch_size', 'seconds_total', 'created_utc']) if pilot_rows else ''}

Timings are only valid for the hardware listed. Pilot outputs are code checks, not results.

## Development results

{summary_block('development')}

## Archived T4 runs (record only)

{summary_block('archived_t4')}

## Final test results

{summary_block('test_FINAL') if 'test_FINAL' in present else '_None. The test split has not been run (no freeze)._'}

## Excluded on purpose

Model files and weights (`*.joblib`, `*.safetensors`, ...), dataset files (`*.parquet`, raw IMDb
data), caches and secrets. The only dataset text included is the 12 few-shot demonstration
reviews in `docs/demo_sets.json` and the rendered prompts, which document the prompts used.
Files skipped while copying: {len(skipped)}.
"""
    (pkg / "README_PACKAGE.md").write_text(readme, encoding="utf-8")

    # secret scan and exclusion check
    hits = []
    for f in pkg.rglob("*"):
        if f.is_file():
            if f.suffix.lower() in EXCLUDE_SUFFIXES:
                hits.append(f"excluded type present: {f}")
            text = f.read_text(encoding="utf-8", errors="ignore")
            hits += [f"{f.relative_to(pkg)}: {p.pattern}" for p in SECRET_PATTERNS if p.search(text)]
    if hits:
        print("Refusing to zip, possible secrets or excluded files:\n  " + "\n  ".join(hits), file=sys.stderr)
        return 1

    zpath = out_root / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(pkg))
    shutil.rmtree(pkg)
    print(json.dumps({"zip": str(zpath), "size_mb": round(zpath.stat().st_size / 1e6, 2),
                      "categories": sorted(present), "skipped_files": skipped,
                      "inventory": {c: s for c, s in counts.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
