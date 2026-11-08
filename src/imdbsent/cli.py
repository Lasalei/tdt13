"""Command line entry point: `python -m imdbsent <command>` (or `imdbsent <command>`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Paths, load_config


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="imdbsent", description=__doc__)
    ap.add_argument("--config", default=None, help="path to experiment YAML")
    ap.add_argument("--root", default=None, help="project root for data/runs/results (default: repo)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download/load IMDb, split train -> train_fit/validation")
    p.add_argument("--source", choices=["hub", "parquet"], default="hub")
    p.add_argument("--parquet-dir", default=None)

    sub.add_parser("select-demos", help="choose fixed few-shot demonstration sets from train_fit")

    p = sub.add_parser("baselines", help="run VADER and TF-IDF+LR")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("llm", help="Qwen inference (checkpointed, resumable)")
    p.add_argument("--condition", required=True, help="e.g. qwen_zeroshot, qwen_2shot_set1, all")
    p.add_argument("--split-tag", default="validation-dev",
                   choices=["validation-dev", "validation", "test", "pipeline-test"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--backend", choices=["hf", "dummy"], default="hf")
    p.add_argument("--max-batches", type=int, default=None, help="stop after N batches (testing resume)")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("pilot", help="time a few LLM conditions and extrapolate the full run")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--conditions", nargs="*",
                   default=["qwen_zeroshot", "qwen_2shot_set1", "qwen_4shot_set1"])
    p.add_argument("--backend", choices=["hf", "dummy"], default="hf")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override llm.batch_size for the pilot only (recorded in the run manifest)")

    sub.add_parser("plan", help="print the planned number of predictions")

    p = sub.add_parser("evaluate", help="metrics + disagreement analysis for one split tag")
    p.add_argument("--split-tag", default="validation",
                   choices=["validation", "validation-dev", "test", "pipeline-test"])

    p = sub.add_parser("freeze", help="lock the configuration; required before any test run")
    p.add_argument("--allow-unpinned", action="store_true",
                   help="freeze even if model/dataset revisions are unpinned or git is dirty (recorded as warnings)")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    paths = Paths(Path(a.root).resolve()) if a.root else Paths()

    if a.cmd == "prepare":
        from .data import prepare

        _dump(prepare(cfg, paths, a.source, a.parquet_dir))
    elif a.cmd == "select-demos":
        from .demos import write_demo_sets

        d = write_demo_sets(cfg, paths)
        _dump({k: {"demo_ids": v["demo_ids"], "demo_labels": v["demo_labels"]} for k, v in d["conditions"].items()})
    elif a.cmd == "baselines":
        from .baselines import BASELINE_CONDITIONS, run_baseline

        for c in a.conditions or list(BASELINE_CONDITIONS):
            m = run_baseline(cfg, paths, a.split, c, overwrite=a.overwrite)
            print(f"{c}: {m['n_predictions']} predictions -> {paths.run_dir(a.split, c)}")
    elif a.cmd == "llm":
        from .demos import condition_names
        from .llm import DummyBackend, HFBackend, run_llm

        backend = DummyBackend() if a.backend == "dummy" else HFBackend(cfg)
        conds = condition_names(cfg) if a.condition == "all" else [a.condition]
        for c in conds:
            m = run_llm(cfg, paths, c, a.split_tag, a.limit, backend, a.batch_size, a.overwrite, a.max_batches)
            print(f"{c}: {m['n_predictions']}/{m['n_target']} complete={m['complete']}")
    elif a.cmd == "pilot":
        from .llm import DummyBackend, HFBackend, pilot_estimate

        backend = DummyBackend() if a.backend == "dummy" else HFBackend(cfg)
        res = pilot_estimate(cfg, paths, a.conditions, a.n, backend, a.batch_size)
        paths.results.mkdir(parents=True, exist_ok=True)
        (paths.results / f"pilot_estimate_{backend.name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        _dump(res)
    elif a.cmd == "plan":
        from .llm import planned_predictions

        _dump(planned_predictions(cfg))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate

        res = evaluate(cfg, paths, a.split_tag)
        print(res["status"])
        print(Path(paths.results / a.split_tag / "metrics_summary.csv").read_text())
    elif a.cmd == "freeze":
        from .guard import freeze

        _dump(freeze(cfg, paths, a.allow_unpinned))
    return 0


if __name__ == "__main__":
    sys.exit(main())
