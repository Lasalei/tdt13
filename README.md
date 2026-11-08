# IMDb sentiment: lexicon vs. bag-of-words vs. small instruction-tuned LLM

TDT13 (Large Language Models for Language Understanding and Generation), NTNU. Individual project.

This repository holds a reproducible experiment that compares four approaches to binary
sentiment classification of English movie reviews:

| Method | Conditions |
|---|---|
| VADER (lexicon/rule-based) | `vader` |
| TF-IDF + logistic regression | `tfidf_lr` |
| Qwen2.5-1.5B-Instruct, zero-shot | `qwen_zeroshot` |
| Qwen2.5-1.5B-Instruct, few-shot | `qwen_2shot_set{1,2,3}`, `qwen_4shot_set{1,2,3}` |

Research questions:
1. How do the approaches compare in classification performance?
2. Do the number (2 vs. 4) and the selection (three fixed sets) of few-shot demonstrations affect performance?
3. Which reviews cause errors or disagreement between methods?

> **Status (2026-10-20):** code implemented and technically tested on **synthetic** data only.
> No experiment results exist yet. The test split has not been used. See [Status](#status-and-limitations).

---

## Data

Stanford Large Movie Review Dataset (Maas et al., ACL 2011), loaded as
[`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (config `plain_text`).

* Official split preserved: 25,000 labelled training reviews, 25,000 test reviews. The
  `unsupervised` split is never loaded.
* `train` is split once into **`train_fit`** (22,500) and a stratified **`validation`** subset
  (2,500; 1,250 per class). A fixed stratified **dev subset** of 500 validation reviews is used
  for LLM development runs.
* Review IDs are content-based (`tr-<sha1 prefix>-<duplicate index>`, `te-…`), so they do not
  depend on file row order. The validation selection is also content-based (canonical sort by
  label and text hash, then a seeded draw), so the same seed gives the same split whether the
  data comes from the Hub or from local parquet files.

## Preprocessing and truncation

All methods see the same text in the primary comparison:

1. `<br />` tags are replaced by a space, and whitespace is collapsed (`text_norm`).
2. **Truncation** (`text_eval`): reviews longer than **400 whitespace words** keep the first
   250 and the last 150 words, joined by ` ... `. Head plus tail keeps the opening and the
   closing verdict. The same truncated text is fed to VADER, TF-IDF+LR and Qwen, so no method
   silently sees more of a review than another.
3. Every prediction records `n_words` (original length) and `truncated`. `prepare` reports
   the exact share of truncated reviews per split in `data/processed/data_manifest.json`.
4. To quantify the effect of truncation, the baselines are also run on the full text
   (`vader_fulltext`, `tfidf_lr_fulltext`). These are secondary conditions.
5. The LLM tokenizer is called with `truncation=False`. A prompt that would exceed the model
   context raises an error instead of being cut.

## Methods

### VADER: score-to-label mapping

VADER returns a `compound` score in [-1, 1]. It is mapped to the two dataset labels as follows:

| compound | label |
|---|---|
| `> 0` | positive |
| `< 0` | negative |
| `== 0` exactly | **neutral = no decision**, counted as **incorrect** in accuracy and reported separately |

* VADER's usual neutral band (|compound| < 0.05) is **not** used to withhold labels; those
  reviews are labelled by sign. Each prediction carries `in_vader_neutral_band`, so accuracy
  inside and outside the band is reported in the error analysis.
* A compound of exactly 0 typically means no lexicon word carries sentiment. Treating it as
  no decision (like an invalid LLM output) avoids an arbitrary tie-break toward one class.
* The threshold is fixed at 0 and is **not tuned**.

### TF-IDF + logistic regression

Word 1–2-grams, `min_df=2`, `max_df=0.95`, sublinear TF, liblinear LR. Fitted **only on
`train_fit`**. The regularisation strength C is chosen from `{0.25, 1, 4, 16}` by validation
macro F1 (ties go to the smaller C). The selected model is saved to `models/` and reused
unchanged on test.

### Qwen2.5-1.5B-Instruct

Chat format, identical across all LLM conditions:

```
system:    You are a sentiment classifier for English movie reviews.
user:      Classify the sentiment of the following movie review as positive or negative.
           Answer with exactly one word: positive or negative.

           Review: <demonstration>
assistant: positive | negative                   (repeated for each demonstration)
user:      <same instruction>

           Review: <target review>
assistant: <generated>
```

Zero-shot is the same prompt with no demonstration turns.

* **Decoding:** greedy (`do_sample=False`), `max_new_tokens=5`, `repetition_penalty=1.0`.
  The model's shipped generation config (which sets sampling parameters and a repetition
  penalty) is replaced explicitly and recorded in the run manifest.
* **Parsing (strict):** lowercase and strip; remove leading/trailing non-letters; take the
  first word. The output is valid only if that word is `positive` or `negative`. Everything
  else is **invalid**, stored with its raw output, counted as incorrect in accuracy, and
  reported separately.

### Few-shot design

* **Pool:** `train_fit` only, 30–150 words (so demonstrations are never truncated), excluding
  any text that also occurs in validation or test or more than once in `train_fit`.
* **Three disjoint, fixed sets**, each with 2 positive and 2 negative reviews (seed 13).
* **Nested:** the 4-shot prompt of set *k* prepends two more demonstrations to the 2-shot
  prompt of set *k*. The two demonstrations nearest the target are therefore identical, so
  2 vs. 4 within a set changes only the number of examples, while sets 1–3 vary the selection.
* **Counterbalanced order** (not random, because the last demonstration's label can bias the answer):

  | set | 2-shot | 4-shot |
  |---|---|---|
  | 1 | P N | P N P N |
  | 2 | N P | N P N P |
  | 3 | P N | P N P N |

  With three sets the final label is negative twice and positive once. Per-set confusion
  matrices make any resulting label bias visible.
* IDs, labels and order of every demonstration are stored in `data/processed/demo_sets.json`
  and in each run manifest.

Analysis for RQ2: mean, standard deviation, minimum and maximum over the three sets for 2-shot
and 4-shot; per-set results; paired comparisons (exact McNemar) of zero-shot vs. each set and
of 2-shot vs. 4-shot within each set.

## Evaluation

Per condition: accuracy, macro F1, precision and recall per class, confusion matrix
(rows = true class; columns = negative, positive, no decision), and the number of invalid LLM
outputs or VADER no-decisions.

* Invalid outputs and no-decisions count as **incorrect** in accuracy. For per-class metrics
  they are a false negative for the true class and a false positive for neither class.
* Cross-method analysis (only on complete runs, on the same review IDs): agreement patterns
  (all correct / all wrong / split); accuracy by length bin (≤100, 101–200, 201–400, >400
  truncated words), by true label and by VADER neutral band; pairwise agreement with exact
  McNemar tests; and a seeded sample of split and all-wrong reviews for manual error
  analysis.
* Each results folder includes `STATUS.txt`: *development results* (validation), *final
  test results*, or *pipeline test on synthetic data, not results*.

### Planned number of predictions

| | per condition | conditions | total |
|---|---:|---:|---:|
| LLM, dev subset (development) | 500 | 7 | 3,500 |
| LLM, full validation (optional) | 2,500 | 7 | 17,500 |
| **LLM, test (final)** | 25,000 | 7 | **175,000** |
| Baselines, test (incl. full-text variants) | 25,000 | 4 | 100,000 |

`python -m imdbsent plan` prints these from the config.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                            # technical tests (synthetic data)

# 1. Data (downloads stanfordnlp/imdb; or use --source parquet --parquet-dir DIR)
python -m imdbsent prepare
# 2. Demonstration sets
python -m imdbsent select-demos
# 3. Baselines on validation (development results)
python -m imdbsent baselines --split validation
python -m imdbsent evaluate --split-tag validation
```

The steps above run on a laptop CPU. VADER and TF-IDF on the full validation split take
about a minute.

### LLM runs (GPU machine: Colab, IDUN, or a workstation)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt -r requirements-llm.txt && pip install -e .
python -m imdbsent prepare && python -m imdbsent select-demos   # or copy data/processed/

# a) Runtime pilot: 48 balanced dev reviews x {zero-shot, 2-shot set1, 4-shot set1}
python -m imdbsent pilot --n 48           # -> results/pilot_estimate_hf.json

# b) Development runs on the 500-review dev subset, all 7 LLM conditions
python -m imdbsent llm --condition all --split-tag validation-dev
python -m imdbsent evaluate --split-tag validation-dev   # baselines are scored on the same 500 reviews
```

**Checkpointing.** Every batch is appended to `runs/<split_tag>/<condition>/predictions.jsonl`
and fsync'ed. If a run is interrupted (for example a Colab disconnect), rerun the same command:
finished reviews are skipped and a half-written last line is repaired. A run refuses to resume
if prompt, decoding, demonstrations or model revision changed (`--overwrite` restarts it). On
Colab, put the project on Google Drive so `runs/` survives a disconnect.

The development machine used to build this project has no GPU (2 CPU cores, 8 GB RAM), and
Hugging Face is blocked there, so no LLM inference has been run yet.

### Running on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lasalei/tdt13/blob/main/notebooks/colab_gpu_runs.ipynb)

[`notebooks/colab_gpu_runs.ipynb`](notebooks/colab_gpu_runs.ipynb) runs the LLM development
steps on a Colab T4 GPU (Runtime → Change runtime type → T4 GPU). The repository and all
outputs live on Google Drive (`MyDrive/tdt13`), so **Run all** after a disconnect resumes
interrupted runs. The notebook:

1. checks for a CUDA GPU, mounts Drive, clones or pulls the repository;
2. installs the requirements (keeping Colab's CUDA build of torch) and runs pytest;
3. runs `prepare` and `select-demos` only if `data/processed/` is missing, then stops unless the
   split file hashes and the `demo_sets.json` hash match the expected values at the top of the
   notebook. If they differ, upload the local `data/processed/` to Drive;
4. runs the validation baselines if missing, the pilot (`--n 48`, config batch size), all seven
   LLM conditions on `validation-dev`, and `evaluate --split-tag validation-dev`.

It never runs `freeze` or touches the test split. Test runs come later, after `freeze`, in a
separate notebook.

### Test-set discipline (freeze, then evaluate once)

```bash
# Pin revisions in configs/experiment.yaml (data.hf_revision, llm.revision), commit, then:
python -m imdbsent freeze
python -m imdbsent baselines --split test
python -m imdbsent llm --condition all --split-tag test
python -m imdbsent evaluate --split-tag test
```

`load_split("test")` refuses to load the test split unless `frozen.lock` exists **and** its
config hash, split files, demonstration file and trained TF-IDF models match the current state.
`freeze` requires pinned model and dataset revisions and a clean git tree (overridable with
`--allow-unpinned`, which is recorded as a warning). It always refuses synthetic data.

## How data leakage is prevented

| Risk | Safeguard |
|---|---|
| Fitting on evaluation data | TF-IDF vocabulary/IDF and LR are fitted on `train_fit` only (`fit_tfidf`). |
| Hyperparameter selection on test | C, prompt and parsing decisions are made on `validation` only; test is locked in code until `freeze`. |
| Demonstrations from evaluated data | Demonstrations come only from `train_fit`; texts that also occur in validation or test are excluded from the pool (exact-text match, no labels used). |
| Validation contaminating training | `train_fit` and `validation` are disjoint by construction; checked in tests and recorded in the manifest. |
| Cross-split duplicates in IMDb | Exact duplicate texts across splits are counted and reported in `data_manifest.json`. They are **not** removed, so the official split stays intact. |
| Changing the setup after seeing test results | `frozen.lock` fingerprints config, data, demonstrations and models; any change blocks further test runs. |
| Silent drift between runs | Every run manifest records config hash, method settings hash, seed, git commit, package versions, hardware, resolved model revision and the full generation config. |

## Reproducibility records

* `data/processed/data_manifest.json`: dataset source and revision, file hashes, split sizes,
  label counts, truncation statistics, duplicate checks.
* `data/processed/demo_sets.json`: demonstration IDs, labels, order and texts.
* `runs/**/run_manifest.json`: per run, as listed above.
* `runs/**/predictions.jsonl`: `review_id`, `true_label`, `pred_label`, `correct`, `valid`,
  `score` (VADER compound / LR probability), `raw_output` (LLM), `condition`, `demo_ids`,
  `prompt_tokens`, `n_words`, `truncated`.
* Seed 13 for the split, demonstration selection and LR. The dev subset uses seed 14.

## Repository layout

```
configs/experiment.yaml       all settings (single source of truth)
src/imdbsent/
  data.py        loading, normalisation, truncation, split, manifest
  baselines.py   VADER mapping, TF-IDF + LR
  demos.py       few-shot demonstration sets
  prompts.py     chat prompts and strict output parser
  llm.py         checkpointed Qwen inference, pilot timing, prediction plan
  evaluate.py    metrics, confusion matrices, disagreement analysis, McNemar
  guard.py       test-split lock (freeze)
  predio.py      append-only, resumable prediction files
  cli.py         `python -m imdbsent <command>`
notebooks/colab_gpu_runs.ipynb      Colab GPU runs (pilot + validation-dev), Drive-backed
scripts/make_synthetic_fixture.py   synthetic data for pipeline tests only
tests/test_pipeline.py              technical tests
```

## Status and limitations

* **Tested:** all code paths except the real Qwen backend, using a synthetic fixture
  (25,000 + 25,000 random word-salad "reviews" marked `SYNTHETIC`) and 22 unit tests. The
  numbers produced on synthetic data are pipeline checks, not results, and must not be reported.
* **Not yet run:** data preparation on the real IMDb data, development baselines on real
  validation data, any Qwen inference, the runtime pilot, and anything on the test split.
* **Qwen backend (`HFBackend`)** was written against the transformers API (5.x, with a fallback
  for 4.x) but has not been executed. Run `pilot` first; it exercises the whole path on 144
  validation reviews.
* Head+tail truncation at 400 words is a design choice; the full-text baseline variants
  measure its effect for the baselines but not for the LLM.
* Few-shot demonstrations are limited to 30–150 words, so they are shorter than a typical
  review. This is a constraint on the selection to keep in mind.
* Batched greedy decoding in reduced precision can differ marginally from unbatched decoding.
  The batch size is fixed in the config and recorded.
