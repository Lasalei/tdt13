"""Technical tests. All data here is SYNTHETIC; nothing tests model quality."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imdbsent.baselines import vader_label
from imdbsent.config import Paths, load_config
from imdbsent.data import normalize_text, prepare, stratified_pick, truncate_words
from imdbsent.demos import (condition_names, flip_condition_names, primary_condition_names,
                            select_demo_sets, write_demo_sets)
from imdbsent.evaluate import evaluate, mcnemar_exact, score_run
from imdbsent.guard import TestSplitLocked, freeze, load_split
from imdbsent.llm import DummyBackend, pilot_estimate, run_llm
from imdbsent.predio import PredictionWriter
from imdbsent.prompts import build_messages, parse_output

ROOT = Path(__file__).resolve().parents[1]


def _fixture_module():
    spec = importlib.util.spec_from_file_location("fx", ROOT / "scripts" / "make_synthetic_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """Small synthetic project: 2,000 train / 1,000 test, validation 200."""
    root = tmp_path_factory.mktemp("proj")
    raw = root / "raw"
    raw.mkdir()
    fx = _fixture_module()
    rng = np.random.default_rng(1)
    train, test = fx.make(2000, rng, "SYNTHETIC"), fx.make(1000, rng, "SYNTHETIC")
    test.loc[0, ["text", "label"]] = train.loc[5, ["text", "label"]].values  # cross-split duplicate
    train.to_parquet(raw / "train-00000-of-00001.parquet")
    test.to_parquet(raw / "test-00000-of-00001.parquet")
    (raw / "SYNTHETIC").write_text("x")
    cfg = load_config()
    cfg["data"]["validation_size"] = 200
    cfg["data"]["dev_llm_subset_size"] = 50
    paths = Paths(root)
    man = prepare(cfg, paths, "parquet", str(raw))
    write_demo_sets(cfg, paths)
    return cfg, paths, man, raw


# ---------------------------------------------------------------- text handling
def test_normalize_replaces_br_and_whitespace():
    assert normalize_text("Good.<br /><br />Bad  film") == "Good. Bad film"


def test_truncation_head_tail():
    text = " ".join(f"w{i}" for i in range(1000))
    out, n, trunc = truncate_words(text, 400, 250)
    words = out.split()
    assert n == 1000 and trunc
    assert words[:250] == [f"w{i}" for i in range(250)]
    assert words[-150:] == [f"w{i}" for i in range(850, 1000)]
    short, n2, t2 = truncate_words("a b c", 400, 250)
    assert (short, n2, t2) == ("a b c", 3, False)


# ---------------------------------------------------------------- split / leakage
def test_split_sizes_stratified_disjoint(small):
    cfg, paths, man, _ = small
    fit = pd.read_parquet(paths.split_file("train_fit"))
    val = pd.read_parquet(paths.split_file("validation"))
    assert len(fit) + len(val) == 2000 and len(val) == 200
    assert val["label"].value_counts().to_dict() == {0: 100, 1: 100}
    assert not set(fit.review_id) & set(val.review_id)
    assert val["in_dev_subset"].sum() == 50
    assert val[val.in_dev_subset]["label"].value_counts().to_dict() == {0: 25, 1: 25}
    assert man["exact_text_overlap"]["train_fit&test"] + man["exact_text_overlap"]["validation&test"] == 1
    assert man["provenance"]["synthetic"] is True


def test_split_independent_of_row_order(small):
    cfg, _, _, raw = small
    from imdbsent.data import add_ids

    df = add_ids(pd.read_parquet(raw / "train-00000-of-00001.parquet"), "tr", cfg)
    shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)
    a = set(df.loc[stratified_pick(df, 200, 13), "review_id"])
    b = set(shuffled.loc[stratified_pick(shuffled, 200, 13), "review_id"])
    assert a == b


def test_demos_only_from_train_fit_balanced_nested(small):
    cfg, paths, _, _ = small
    demos = json.loads(paths.demos_file.read_text())
    fit = pd.read_parquet(paths.split_file("train_fit")).set_index("review_id")
    val_ids = set(pd.read_parquet(paths.split_file("validation")).review_id)
    all_ids = []
    for name, c in demos["conditions"].items():
        for rid, lab in zip(c["demo_ids"], c["demo_labels"]):
            assert rid in fit.index and rid not in val_ids
            assert bool(fit.at[rid, "demo_eligible"])
            assert {0: "negative", 1: "positive"}[int(fit.at[rid, "label"])] == lab
        if c["shots"]:
            assert c["demo_labels"].count("positive") == c["demo_labels"].count("negative")
            # labels alternate in every prompt
            assert all(a != b for a, b in zip(c["demo_labels"], c["demo_labels"][1:]))
    for s in (1, 2, 3):
        two, four = demos["conditions"][f"qwen_2shot_set{s}"], demos["conditions"][f"qwen_4shot_set{s}"]
        assert four["demo_ids"][-2:] == two["demo_ids"]          # nested: same nearest demos
        all_ids += four["demo_ids"]
    assert len(all_ids) == len(set(all_ids)) == 12               # sets are disjoint
    last = [demos["conditions"][f"qwen_2shot_set{s}"]["demo_labels"][-1] for s in (1, 2, 3)]
    assert sorted(last) == ["negative", "negative", "positive"]


def test_demo_selection_deterministic(small):
    cfg, paths, _, _ = small
    fit = pd.read_parquet(paths.split_file("train_fit"))
    a = select_demo_sets(cfg, fit)
    b = select_demo_sets(cfg, fit.sample(frac=1, random_state=0))
    assert a["conditions"] == b["conditions"]


def test_test_split_locked_until_freeze(small):
    cfg, paths, _, _ = small
    with pytest.raises(TestSplitLocked):
        load_split(cfg, paths, "test")
    with pytest.raises(RuntimeError, match="SYNTHETIC"):
        freeze(cfg, paths, allow_unpinned=True)


# ---------------------------------------------------------------- methods
def test_vader_mapping():
    assert vader_label(0.3) == "positive"
    assert vader_label(-0.01) == "negative"
    assert vader_label(0.0) == "neutral"


@pytest.mark.parametrize("raw,exp", [
    ("positive", "positive"), (" Negative.", "negative"), ("POSITIVE\n", "positive"),
    ('"negative"', "negative"), ("positive review", "positive"),
    ("", "invalid"), ("neutral", "invalid"), ("The sentiment is positive", "invalid"),
    ("pos", "invalid"),
])
def test_parse_output(raw, exp):
    assert parse_output(raw) == exp


def test_prompts_consistent_across_conditions(small):
    cfg, paths, _, _ = small
    demos = json.loads(paths.demos_file.read_text())["conditions"]
    zero = build_messages(cfg, "REVIEW", demos["qwen_zeroshot"]["demos"])
    two = build_messages(cfg, "REVIEW", demos["qwen_2shot_set1"]["demos"])
    four = build_messages(cfg, "REVIEW", demos["qwen_4shot_set1"]["demos"])
    assert zero[0] == two[0] == four[0]                       # same system prompt
    assert zero[-1] == two[-1] == four[-1]                    # same target turn
    assert len(zero) == 2 and len(two) == 6 and len(four) == 10
    assert four[-5:] == two[-5:]                              # nested suffix
    assert all(m["content"].startswith(cfg["llm"]["instruction"].strip()) for m in four if m["role"] == "user")


# ---------------------------------------------------------------- evaluation / IO
def test_invalid_counts_as_incorrect():
    df = pd.DataFrame({"true_label": ["positive", "positive", "negative", "negative"],
                       "pred_label": ["positive", "invalid", "negative", "positive"]})
    m = score_run(df)
    assert m["accuracy"] == 0.5 and m["n_no_decision"] == 1
    assert m["per_class"]["positive"]["recall"] == 0.5        # invalid = missed positive
    assert m["per_class"]["positive"]["precision"] == 0.5
    assert m["confusion_matrix"]["counts"] == [[1, 1, 0], [0, 1, 1]]


def test_mcnemar():
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    assert mcnemar_exact(a, b)["p_value"] == pytest.approx(2 / 1024)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_resume_repairs_partial_line_and_rejects_changes(tmp_path):
    man = {"created_utc": "t0", "settings_hash": "h", "condition": "c", "split": "s",
           "method": "m", "demo_ids": [], "model_revision_resolved": "r"}
    w = PredictionWriter(tmp_path, copy.deepcopy(man))
    w.write([{"review_id": "a"}, {"review_id": "b"}])
    w.close()
    with open(tmp_path / "predictions.jsonl", "a") as f:
        f.write('{"review_id": "c"')                              # crash mid-write
    w2 = PredictionWriter(tmp_path, {**copy.deepcopy(man), "created_utc": "t1"})
    assert w2.done == {"a", "b"}
    w2.write([{"review_id": "c"}])
    w2.close()
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert [json.loads(l)["review_id"] for l in lines] == ["a", "b", "c"]
    with pytest.raises(RuntimeError, match="settings changed"):
        PredictionWriter(tmp_path, {**copy.deepcopy(man), "settings_hash": "other"})


def test_llm_run_resume_end_to_end(small):
    cfg, paths, _, _ = small
    m1 = run_llm(cfg, paths, "qwen_4shot_set2", "pipeline-test", backend=DummyBackend(),
                 batch_size=16, overwrite=True, max_batches=3)
    assert m1["n_predictions"] == 48 and not m1["complete"]
    m2 = run_llm(cfg, paths, "qwen_4shot_set2", "pipeline-test", backend=DummyBackend(), batch_size=16)
    assert m2["complete"] and m2["n_predictions"] == 200
    df = pd.read_json(paths.run_dir("pipeline-test", "qwen_4shot_set2") / "predictions.jsonl", lines=True)
    assert df["review_id"].is_unique and len(df) == 200
    assert {"review_id", "true_label", "pred_label", "raw_output", "condition", "demo_ids"} <= set(df.columns)
    with pytest.raises(RuntimeError, match="Dummy backend"):
        run_llm(cfg, paths, "qwen_zeroshot", "validation-dev", backend=DummyBackend())


def test_pilot_batch_size_override(small):
    cfg, paths, _, _ = small
    res = pilot_estimate(cfg, paths, ["qwen_zeroshot"], n=6, backend=DummyBackend(), batch_size=2)
    assert res["batch_size"] == 2
    run = paths.run_dir("pilot", "qwen_zeroshot")
    man = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["batch_size"] == 2
    df = pd.read_json(run / "predictions.jsonl", lines=True)
    assert len(df) == 6 and df["batch_index"].nunique() == 3
    assert cfg["llm"]["batch_size"] == 16  # config value is not modified


def test_llm_progress_lines(small, capsys):
    cfg, paths, _, _ = small
    run_llm(cfg, paths, "qwen_zeroshot", "pipeline-test", backend=DummyBackend(),
            batch_size=16, overwrite=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[qwen_zeroshot]")]
    assert "start: 0/200 already done" in lines[0]
    progress = [l for l in lines[1:] if "%)" in l]
    assert 9 <= len(progress) <= 11 and "200/200 (100%)" in progress[-1]


# ---------------------------------------------------------------- order analysis (secondary)
def test_flip_conditions_same_demos_swapped_order(small):
    cfg, paths, _, _ = small
    conds = json.loads(paths.demos_file.read_text())["conditions"]
    names = list(conds)
    # originals first and unchanged in position; flips appended after them
    assert set(names[:7]) == set(primary_condition_names(cfg))
    assert set(names[7:]) == set(flip_condition_names(cfg)) and len(names) == 13
    assert condition_names(cfg) == primary_condition_names(cfg) + flip_condition_names(cfg)
    for flip in flip_condition_names(cfg):
        f, o = conds[flip], conds[flip[:-5]]
        assert f["order_flip_of"] == flip[:-5] and f["shots"] == o["shots"] and f["set"] == o["set"]
        swapped = [x for i in range(0, len(o["demo_ids"]), 2) for x in o["demo_ids"][i:i + 2][::-1]]
        assert f["demo_ids"] == swapped
        assert f["demo_labels"] == [{"positive": "negative", "negative": "positive"}[l] for l in o["demo_labels"]]
    for s in (1, 2, 3):  # nested structure kept
        assert conds[f"qwen_4shot_set{s}_flip"]["demo_ids"][-2:] == conds[f"qwen_2shot_set{s}_flip"]["demo_ids"]
    assert conds["qwen_2shot_set1_flip"]["demo_labels"] == ["negative", "positive"]
    assert conds["qwen_4shot_set1_flip"]["demo_labels"] == ["negative", "positive"] * 2


def test_evaluate_keeps_flips_out_of_primary_analysis(small, tmp_path):
    cfg, paths, _, _ = small
    for c in ("qwen_2shot_set1", "qwen_2shot_set1_flip", "qwen_2shot_set2"):
        run_llm(cfg, paths, c, "pipeline-test", backend=DummyBackend(), batch_size=50, overwrite=True)
    res = evaluate(cfg, paths, "pipeline-test", out_dir=tmp_path)
    assert "qwen_2shot_set1_flip" in res["metrics"]                       # scored like any run
    assert not any(n.endswith("_flip") for n in res["few_shot_spread"]["2shot"]["sets"])
    assert not any(n.endswith("_flip") for n in res["analysis"]["conditions"])
    pairs = res["order_analysis"]["pairs"]
    assert [(p["original"], p["flipped"]) for p in pairs] == [("qwen_2shot_set1", "qwen_2shot_set1_flip")]
    assert pairs[0]["order_original"] == "P-N" and pairs[0]["order_flipped"] == "N-P"
    assert pairs[0]["n"] == 200 and 0.0 <= pairs[0]["p_value"] <= 1.0
    assert (tmp_path / "order_effect.csv").exists()
