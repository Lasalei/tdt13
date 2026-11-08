"""Checkpointed Qwen inference (zero-shot and few-shot).

Runs write to runs/<split_tag>/<condition>/ with one JSONL record per review, flushed and
fsync'ed after every batch. Re-running the same command resumes: finished review IDs are
skipped, a half-written last line is repaired, and the run refuses to resume if the
prompt/decoding settings, demonstrations or resolved model revision changed.

split_tag:
  validation-dev  the fixed 500-review stratified dev subset of validation (default)
  validation      all 2,500 validation reviews
  test            all 25,000 test reviews (only after `imdbsent freeze`)
  pilot           first N reviews of the dev subset, for runtime estimation only

Backends:
  hf     transformers model (the real experiment)
  dummy  deterministic fake outputs for pipeline tests only; refused for validation/test
"""
from __future__ import annotations

import time
from typing import Any, Protocol, Sequence

import pandas as pd

from . import INVALID, LABEL_NAMES
from .config import Paths
from .demos import load_demo_sets
from .guard import load_split
from .predio import PredictionWriter, base_run_manifest
from .prompts import PARSE_RULE, build_messages, parse_output


class Backend(Protocol):
    name: str
    revision: str | None
    info: dict[str, Any]

    def generate(self, batch: Sequence[list[dict[str, str]]]) -> list[tuple[str, int]]:
        """Return (raw_output, prompt_token_count) per conversation."""


class HFBackend:
    name = "hf"

    def __init__(self, cfg: dict[str, Any], device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        lcfg = cfg["llm"]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        if lcfg["dtype"] == "auto":
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            dtype = getattr(torch, lcfg["dtype"])
        torch.manual_seed(cfg["seed"])
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(lcfg["model"], revision=lcfg.get("revision"))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:  # transformers >= 4.56 / 5.x
            self.model = AutoModelForCausalLM.from_pretrained(
                lcfg["model"], revision=lcfg.get("revision"), dtype=dtype)
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(
                lcfg["model"], revision=lcfg.get("revision"), torch_dtype=dtype)
        self.model = self.model.to(device).eval()
        self.shipped_generation_config = self.model.generation_config.to_diff_dict()
        # Explicit decoding config: greedy, and override Qwen's shipped sampling defaults
        # (temperature/top_p/top_k/repetition_penalty) so nothing is inherited implicitly.
        self.gen_cfg = GenerationConfig(
            max_new_tokens=lcfg["max_new_tokens"], do_sample=lcfg["do_sample"],
            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
            pad_token_id=self.tok.pad_token_id, eos_token_id=self.model.generation_config.eos_token_id)
        # Replace the model's own generation config so no shipped default can leak in.
        self.model.generation_config = self.gen_cfg
        self.max_ctx = getattr(self.model.config, "max_position_embeddings", None)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if not self.revision:
            try:
                from huggingface_hub import model_info

                self.revision = model_info(lcfg["model"], revision=lcfg.get("revision")).sha
            except Exception:
                self.revision = None
        self.info = {"device": device, "dtype": str(dtype), "model": lcfg["model"],
                     "revision_requested": lcfg.get("revision"), "revision_resolved": self.revision,
                     "max_position_embeddings": self.max_ctx,
                     "generation_config": self.gen_cfg.to_diff_dict(),
                     "shipped_generation_config_overridden": self.shipped_generation_config}

    def generate(self, batch):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
        lens = enc["attention_mask"].sum(dim=1).tolist()
        if self.max_ctx and max(lens) + self.gen_cfg.max_new_tokens > self.max_ctx:
            raise RuntimeError("Prompt exceeds model context; reviews are never silently truncated.")
        with self.torch.inference_mode():
            # All decoding settings (greedy, max_new_tokens, repetition_penalty=1.0) come from
            # self.gen_cfg only; passing them again as kwargs is deprecated in transformers 5.x.
            out = self.model.generate(**enc, generation_config=self.gen_cfg)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        return list(zip(texts, [int(n) for n in lens]))


class DummyBackend:
    """PIPELINE TEST ONLY. Deterministic fake 'model' that echoes a label or garbage."""
    name = "dummy"
    revision = "dummy"
    info = {"warning": "dummy backend: outputs are fake and carry no information"}

    def generate(self, batch):
        out = []
        for msgs in batch:
            text = msgs[-1]["content"]
            h = sum(map(ord, text)) % 10
            raw = "positive" if h < 4 else "Negative." if h < 8 else "I cannot tell"
            out.append((raw, len(" ".join(m["content"] for m in msgs).split())))
        return out


def _select_rows(cfg, paths, split_tag: str, limit: int | None) -> tuple[str, pd.DataFrame]:
    if split_tag in ("validation-dev", "pilot"):
        df = load_split(cfg, paths, "validation")
        df = df[df["in_dev_subset"]].sort_values("review_id")
        split = "validation"
    elif split_tag in ("validation", "test"):
        split = split_tag
        df = load_split(cfg, paths, split).sort_values("review_id")
    elif split_tag == "pipeline-test":
        split = "validation"
        df = load_split(cfg, paths, "validation").sort_values("review_id")
    else:
        raise ValueError(split_tag)
    if limit is not None:
        # Balanced head: first limit/2 of each label (by review_id), so pilots stay balanced.
        neg, pos = df[df.label == 0], df[df.label == 1]
        k = limit // 2
        df = pd.concat([neg.head(limit - k), pos.head(k)]).sort_values("review_id")
    return split, df.reset_index(drop=True)


def run_llm(cfg: dict[str, Any], paths: Paths, condition: str, split_tag: str = "validation-dev",
            limit: int | None = None, backend: Backend | None = None, batch_size: int | None = None,
            overwrite: bool = False, max_batches: int | None = None) -> dict[str, Any]:
    demo_sets = load_demo_sets(paths)
    if condition not in demo_sets["conditions"]:
        raise KeyError(f"Unknown condition {condition}; options: {list(demo_sets['conditions'])}")
    cond = demo_sets["conditions"][condition]
    backend = backend or HFBackend(cfg)
    if backend.name == "dummy" and split_tag not in ("pipeline-test", "pilot"):
        raise RuntimeError("Dummy backend may only write to the 'pipeline-test' or 'pilot' split tags.")
    split, df = _select_rows(cfg, paths, split_tag, limit)
    bs = batch_size or cfg["llm"]["batch_size"]

    man = base_run_manifest(
        cfg, split, condition, "qwen", split_tag=split_tag, backend=backend.name,
        model=cfg["llm"]["model"], model_revision_resolved=backend.revision, backend_info=backend.info,
        shots=cond["shots"], demo_set=cond["set"], demo_ids=cond["demo_ids"],
        demo_labels=cond["demo_labels"], parse_rule=PARSE_RULE, batch_size=bs,
        text_field="text_eval", n_target=int(len(df)), limit=limit,
        example_prompt=build_messages(cfg, "<REVIEW>", cond["demos"]))
    w = PredictionWriter(paths.run_dir(split_tag, condition), man, overwrite=overwrite)
    todo = df[~df["review_id"].isin(w.done)].copy()
    # Deterministic, length-sorted batching (fewer pad tokens). Order is recorded.
    todo["_len"] = todo["text_eval"].str.len()
    todo = todo.sort_values(["_len", "review_id"], ascending=[False, True])

    t_start = time.perf_counter()
    secs = man.get("seconds_total", 0.0)
    n_batches = 0
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start:start + bs]
        msgs = [build_messages(cfg, t, cond["demos"]) for t in chunk["text_eval"]]
        t0 = time.perf_counter()
        outs = backend.generate(msgs)
        dt = time.perf_counter() - t0
        recs = []
        for row, (raw, ptoks) in zip(chunk.itertuples(index=False), outs):
            pred, true = parse_output(raw), LABEL_NAMES[row.label]
            recs.append({
                "review_id": row.review_id, "split": split, "split_tag": split_tag,
                "condition": condition, "method": "qwen", "true_label": true, "pred_label": pred,
                "correct": pred == true, "valid": pred != INVALID, "score": None,
                "raw_output": raw, "demo_ids": cond["demo_ids"], "prompt_tokens": ptoks,
                "n_words": int(row.n_words), "truncated": bool(row.truncated),
                "batch_index": n_batches, "session": man["created_utc"],
                "batch_seconds": round(dt, 4),
            })
        w.write(recs)
        secs += dt
        n_batches += 1
        w.write_manifest(n_predictions=len(w.done), seconds_total=round(secs, 2),
                         complete=len(w.done) >= len(df))
        if max_batches and n_batches >= max_batches:
            break
    w.write_manifest(n_predictions=len(w.done), complete=len(w.done) >= len(df),
                     wall_seconds_this_session=round(time.perf_counter() - t_start, 2))
    w.close()
    return w.manifest


def pilot_estimate(cfg: dict[str, Any], paths: Paths, conditions: Sequence[str], n: int = 48,
                   backend: Backend | None = None, batch_size: int | None = None) -> dict[str, Any]:
    """Time a few conditions on n dev reviews and extrapolate to the planned runs.

    The first batch of each condition is excluded from the rate (warm-up / compilation).
    Few-shot conditions with the same k have similar prompt lengths, so one measured set per
    k is used for all three sets of that k. batch_size overrides llm.batch_size for the
    pilot only; the value used is recorded in each pilot run manifest and in the result.
    """
    backend = backend or HFBackend(cfg)
    rates: dict[str, float] = {}
    for c in conditions:
        m = run_llm(cfg, paths, c, split_tag="pilot", limit=n, backend=backend,
                    batch_size=batch_size, overwrite=True)
        recs = pd.read_json(paths.run_dir("pilot", c) / "predictions.jsonl", lines=True)
        per_batch = recs.groupby("batch_index").agg(s=("batch_seconds", "first"), k=("review_id", "size"))
        if len(per_batch) > 1:
            per_batch = per_batch.iloc[1:]  # drop warm-up batch
        rates[c] = float(per_batch["s"].sum() / per_batch["k"].sum())
        rates[c + "#invalid_rate"] = float((~recs["valid"]).mean())
        rates[c + "#mean_prompt_tokens"] = float(recs["prompt_tokens"].mean())
    plan = planned_predictions(cfg)
    est = {}
    for cond in plan["test_conditions"]:
        k = "qwen_zeroshot" if "zeroshot" in cond else next(
            (c for c in conditions if c.split("_set")[0] == cond.split("_set")[0]), None)
        if k in rates:
            est[cond] = rates[k] * plan["n_test"] / 3600
    return {"seconds_per_review": {k: v for k, v in rates.items() if "#" not in k},
            "diagnostics": {k: v for k, v in rates.items() if "#" in k},
            "estimated_test_hours_per_condition": est,
            "estimated_test_hours_total": sum(est.values()),
            "backend": backend.info, "plan": plan, "pilot_reviews_per_condition": n,
            "batch_size": batch_size or cfg["llm"]["batch_size"],
            "note": "Extrapolation from a short pilot on validation reviews; actual time varies."}


def planned_predictions(cfg: dict[str, Any]) -> dict[str, Any]:
    from .demos import condition_names

    conds = condition_names(cfg)
    n_test, n_val = 25_000, cfg["data"]["validation_size"]
    n_dev = cfg["data"]["dev_llm_subset_size"]
    return {"test_conditions": conds, "n_test": n_test,
            "llm_test_predictions": len(conds) * n_test,
            "llm_dev_predictions": len(conds) * n_dev,
            "llm_full_validation_predictions": len(conds) * n_val,
            "baseline_test_predictions": 4 * n_test}
