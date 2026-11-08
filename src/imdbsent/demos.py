"""Fixed, balanced few-shot demonstration sets drawn from train_fit only.

Design (configs/experiment.yaml, section few_shot):
  * n_sets = 3 independent sets; no review appears in more than one set.
  * Each set holds 2 positive + 2 negative reviews sampled from the eligible pool
    (train_fit, 30-150 words, text not duplicated in train_fit/validation/test).
  * Nested: the 2-shot prompt of set k uses 1 positive + 1 negative from set k; the
    4-shot prompt of set k PREPENDS the other two reviews of set k. The two demos
    closest to the target review are therefore identical for 2- and 4-shot, so the
    2 -> 4 comparison within a set changes only the number of examples.
  * Order is deterministic and counterbalanced (not random), because the label of the
    last demonstration can bias the answer (recency):
        set1: 2-shot P N      4-shot P N P N
        set2: 2-shot N P      4-shot N P N P
        set3: 2-shot P N      4-shot P N P N
    Labels alternate in every prompt. With three sets the final label is negative in
    sets 1 and 3 and positive in set 2 (2:1, unavoidable with an odd number of sets);
    per-set metrics and confusion matrices are reported so any label bias is visible.

Secondary, post-hoc order analysis (added before freeze, after a development observation
that sets 1 and 3, both P-N ordered, gave identical predictions while set 2 (N-P) differed,
so demonstration selection and order were confounded):
  * qwen_{2,4}shot_set{1,2,3}_flip use exactly the same demonstrations as the original
    condition, with the two labels inside each pair swapped; the nested structure is kept:
        set1_flip: 2-shot N P      4-shot N P N P
        set2_flip: 2-shot P N      4-shot P N P N
        set3_flip: 2-shot N P      4-shot N P N P
  * They are appended after the original conditions; the original conditions, their
    demonstration IDs and their order are unchanged. They are not part of the primary
    analysis (RQ2 spread over sets); evaluate reports them in a separate order section.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import LABEL_NAMES
from .config import Paths, file_sha256
from .guard import load_split


FLIP_SUFFIX = "_flip"


def primary_condition_names(cfg: dict[str, Any]) -> list[str]:
    """Zero-shot and the original few-shot conditions (primary analysis, RQ1/RQ2)."""
    fs = cfg["few_shot"]
    names = ["qwen_zeroshot"]
    for k in fs["shots"]:
        names += [f"qwen_{k}shot_set{s}" for s in range(1, fs["n_sets"] + 1)]
    return names


def flip_condition_names(cfg: dict[str, Any]) -> list[str]:
    """Secondary order-analysis conditions: same demonstrations, label order swapped."""
    return [n + FLIP_SUFFIX for n in primary_condition_names(cfg) if n != "qwen_zeroshot"]


def is_flip(condition: str) -> bool:
    return condition.endswith(FLIP_SUFFIX)


def condition_names(cfg: dict[str, Any]) -> list[str]:
    """All LLM conditions that are run (`--condition all`): primary, then order flips."""
    return primary_condition_names(cfg) + flip_condition_names(cfg)


def select_demo_sets(cfg: dict[str, Any], train_fit: pd.DataFrame) -> dict[str, Any]:
    fs = cfg["few_shot"]
    if sorted(fs["shots"]) != [2, 4] or not fs["nested"]:
        raise NotImplementedError("This implementation covers the nested balanced 2/4-shot design.")
    per_label = max(fs["shots"]) // 2          # 2 per label per set
    pool = train_fit[train_fit["demo_eligible"]].sort_values("review_id").reset_index(drop=True)
    rng = np.random.default_rng(fs["seed"])
    picks: dict[int, list[int]] = {}
    for lab in (0, 1):
        idx = pool.index[pool["label"] == lab].to_numpy()
        need = per_label * fs["n_sets"]
        if len(idx) < need:
            raise ValueError(f"Demo pool too small for label {lab}: {len(idx)} < {need}")
        picks[lab] = list(rng.choice(idx, size=need, replace=False))

    def demo(i: int) -> dict[str, Any]:
        r = pool.loc[i]
        return {"review_id": r.review_id, "label": LABEL_NAMES[int(r.label)],
                "n_words": int(r.n_words), "text": r.text_eval}

    sets, conditions = {}, {"qwen_zeroshot": {"shots": 0, "set": None, "demos": []}}
    for s in range(fs["n_sets"]):
        neg = [demo(i) for i in picks[0][s * per_label:(s + 1) * per_label]]
        pos = [demo(i) for i in picks[1][s * per_label:(s + 1) * per_label]]
        sets[f"set{s + 1}"] = {"positive": [d["review_id"] for d in pos],
                               "negative": [d["review_id"] for d in neg]}
        pos_first = s % 2 == 0
        near = [pos[0], neg[0]] if pos_first else [neg[0], pos[0]]
        far = [pos[1], neg[1]] if pos_first else [neg[1], pos[1]]
        conditions[f"qwen_2shot_set{s + 1}"] = {"shots": 2, "set": s + 1, "demos": near}
        conditions[f"qwen_4shot_set{s + 1}"] = {"shots": 4, "set": s + 1, "demos": far + near}
    # Secondary order analysis: appended after all original conditions, which stay unchanged.
    for s in range(fs["n_sets"]):
        for k in (2, 4):
            orig = conditions[f"qwen_{k}shot_set{s + 1}"]
            d = orig["demos"]
            flipped = [x for i in range(0, len(d), 2) for x in (d[i + 1], d[i])]  # swap within pairs
            conditions[f"qwen_{k}shot_set{s + 1}{FLIP_SUFFIX}"] = {
                "shots": k, "set": s + 1, "demos": flipped,
                "order_flip_of": f"qwen_{k}shot_set{s + 1}", "secondary": True}
    for c in conditions.values():
        c["demo_ids"] = [d["review_id"] for d in c["demos"]]
        c["demo_labels"] = [d["label"] for d in c["demos"]]
    return {"seed": fs["seed"], "pool_size": int(len(pool)),
            "pool_label_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in pool["label"].value_counts().items()},
            "sets": sets, "conditions": conditions}


def write_demo_sets(cfg: dict[str, Any], paths: Paths) -> dict[str, Any]:
    out = select_demo_sets(cfg, load_split(cfg, paths, "train_fit"))
    out["train_fit_sha256"] = file_sha256(paths.split_file("train_fit"))
    paths.demos_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_demo_sets(paths: Paths) -> dict[str, Any]:
    if not paths.demos_file.exists():
        raise FileNotFoundError("demo_sets.json missing: run `imdbsent select-demos`.")
    return json.loads(paths.demos_file.read_text(encoding="utf-8"))
