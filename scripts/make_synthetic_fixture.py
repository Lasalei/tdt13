"""Generate SYNTHETIC IMDb-shaped parquet files for pipeline tests ONLY.

The texts are random word salads with a planted sentiment signal. They are not movie
reviews; any number computed on them says nothing about the methods. A SYNTHETIC marker
file is written next to the parquet files; `prepare` records it, `evaluate` prints a
warning banner, and `freeze` refuses to run on such data.

Usage: python scripts/make_synthetic_fixture.py OUT_DIR [--n-train 25000 --n-test 25000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS = "great wonderful excellent superb loved brilliant moving enjoyable fantastic charming".split()
NEG = "awful terrible boring dreadful hated stupid dull waste poor horrible".split()
NEU = ("the a movie film plot actor scene story director it was and of to in this that with "
       "character ending camera music script").split()


def make(n: int, rng: np.random.Generator, prefix: str) -> pd.DataFrame:
    labels = np.repeat([0, 1], n // 2)
    rng.shuffle(labels)
    texts = []
    for y in labels:
        length = int(np.clip(rng.lognormal(5.2, 0.7), 10, 1200))   # some > 400 words
        p_signal = rng.uniform(0.0, 0.12)                           # some reviews ~no signal
        words = []
        for _ in range(length):
            u = rng.random()
            if u < p_signal:
                words.append(rng.choice(POS if (y == 1) ^ (rng.random() < 0.3) else NEG))
            else:
                words.append(rng.choice(NEU))
            if rng.random() < 0.01:
                words.append("<br /><br />")
        texts.append(f"[{prefix}] " + " ".join(words))
    return pd.DataFrame({"text": texts, "label": labels.astype("int64")})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n-train", type=int, default=25_000)
    ap.add_argument("--n-test", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = make(a.n_train, rng, "SYNTHETIC")
    test = make(a.n_test, rng, "SYNTHETIC")
    # Plant a few exact duplicates (within train and train->test) to exercise the checks.
    train.loc[1, ["text", "label"]] = train.loc[0, ["text", "label"]].values
    test.loc[0, ["text", "label"]] = train.loc[2, ["text", "label"]].values
    train.to_parquet(out / "train-00000-of-00001.parquet", index=False)
    test.to_parquet(out / "test-00000-of-00001.parquet", index=False)
    (out / "SYNTHETIC").write_text("Synthetic fixture data for pipeline tests only. Not IMDb.\n")
    print(f"wrote synthetic fixture to {out}")


if __name__ == "__main__":
    main()
