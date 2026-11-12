# Sentiment classification on IMDb: VADER vs. TF-IDF vs. Qwen

Individual project in TDT13 (Large Language Models for Language Understanding and Generation),
NTNU, autumn 2026. Las Alei.

## About the project

I compare three quite different ways of classifying movie reviews as positive or negative:

* **VADER**, a rule-based method built on a sentiment lexicon (no training).
* **TF-IDF + logistic regression**, a classic supervised classifier trained on IMDb.
* **Qwen2.5-1.5B-Instruct**, a small instruction-tuned language model that is only prompted,
  never fine-tuned. I test it zero-shot and with two- and four-shot prompts built from three
  different sets of examples.

The research questions are how the three methods compare (RQ1), and how Qwen changes between
zero-, two- and four-shot prompting with different example sets (RQ2). During development I
noticed that the example sets differed both in which examples they used and in what order, so I
added an exploratory question (RQ3): does reversing the order of the examples change the
predictions? This was added before anything was run on the test set.

**Data:** the labelled IMDb dataset (Maas et al., 2011), loaded from Hugging Face as
`stanfordnlp/imdb`. The official training set (25,000) is split into 22,500 reviews for training
TF-IDF and 2,500 for validation. Prompts were developed on a fixed subset of 500 validation
reviews. All final results are on the full official test set (25,000 reviews). `<br />` tags are
removed, and reviews over 400 words are shortened to the first 250 and last 150 words. All three
methods get exactly the same text.

## Final results

Test set, 25,000 reviews (same as Table 1 in the report):

| Condition | Accuracy (%) | Macro F1 |
|---|---:|---:|
| VADER | 69.708 | 0.6884 |
| TF-IDF + logistic regression | 90.116 | 0.9012 |
| Qwen zero-shot | 94.600 | 0.9460 |
| Qwen two-shot, set 1 | 93.528 | 0.9352 |
| Qwen two-shot, set 2 | 94.836 | 0.9484 |
| Qwen two-shot, set 3 | 93.476 | 0.9347 |
| Qwen four-shot, set 1 | 94.268 | 0.9426 |
| Qwen four-shot, set 2 | 94.868 | 0.9487 |
| Qwen four-shot, set 3 | 94.432 | 0.9443 |

In short: zero-shot Qwen clearly beats both baselines, and adding examples does not reliably
help. Only set 2 beats zero-shot. Reversing the example order changed accuracy by between
−0.57 and +1.15 percentage points (report, Table 3), which is more than the best few-shot gain.
The final runs used the frozen commit `825ab81`, dataset revision `e6281661` and Qwen revision
`989aa798`, on one NVIDIA L4 GPU in bfloat16 with batch size 16.

## How to run

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q          # technical tests on synthetic data
```

The Qwen runs need a GPU and `pip install -r requirements-llm.txt` (install a CUDA build of
torch first). `requirements-lock.txt` lists the exact versions from my local Mac setup.

### Development runs (training and validation data only)

```bash
python -m imdbsent prepare                           # download IMDb, make the splits
python -m imdbsent select-demos                      # fixed few-shot example sets
python -m imdbsent baselines --split validation      # VADER and TF-IDF (also picks C)
python -m imdbsent evaluate --split-tag validation
python -m imdbsent llm --condition all --split-tag validation-dev   # GPU
python -m imdbsent evaluate --split-tag validation-dev
```

I ran the Qwen part on Google Colab with an L4 GPU using
[`notebooks/colab_gpu_runs.ipynb`](notebooks/colab_gpu_runs.ipynb)
([open in Colab](https://colab.research.google.com/github/Lasalei/tdt13/blob/main/notebooks/colab_gpu_runs.ipynb)).
Outputs are stored on Google Drive, and every run saves its predictions batch by batch, so an
interrupted run continues where it stopped when the cells are run again.

### Test runs

The test split is locked in the code. It can only be loaded after `freeze`, which saves
fingerprints of the config, data, few-shot examples and trained TF-IDF models in `frozen.lock`.
If any of these change afterwards, the test runs refuse to start.

```bash
python -m imdbsent freeze
python -m imdbsent baselines --split test
python -m imdbsent llm --condition all --split-tag test     # GPU, takes several hours
python -m imdbsent evaluate --split-tag test
```

I did this in [`notebooks/colab_test_runs.ipynb`](notebooks/colab_test_runs.ipynb), which checks
that the GPU is an L4 and that all development runs are complete before it freezes.

## Project structure

```
configs/experiment.yaml     all settings (seeds, truncation, TF-IDF grid, prompt, model revision)
src/imdbsent/
  data.py                   loading, cleaning, truncation, splits
  baselines.py              VADER and TF-IDF + logistic regression
  demos.py                  few-shot example sets, including the reversed-order versions
  prompts.py                chat prompt and strict answer parsing
  llm.py                    Qwen inference with checkpointing
  evaluate.py               metrics, confusion matrices, McNemar tests, order comparison
  guard.py                  test-split lock (freeze)
  cli.py                    python -m imdbsent <command>
notebooks/                  Colab notebooks for development and test runs
scripts/                    report package and error-analysis sample (read existing results only)
tests/                      technical tests on synthetic data
```

Generated files (`data/`, `models/`, `runs/`, `results/`) are not in git.

### Some details worth knowing

* **VADER:** positive if the compound score is above 0 and negative if it is below. A score of
  exactly 0 counts as an error (7 test reviews). The threshold is not tuned.
* **TF-IDF:** unigrams and bigrams. C is chosen from {0.25, 1, 4, 16, 64, 256} by validation
  macro F1. That gave C = 16 for the shortened text and C = 64 for the full text.
* **Qwen:** greedy decoding with at most 5 new tokens. Any answer other than `positive` or
  `negative` counts as wrong.
* **Two vs. four shots:** the four-shot prompt is the two-shot prompt with two more examples
  added in front. Going from two to four shots therefore changes both the number of examples and
  which examples the prompt contains.
* **Reversed order (`_flip` conditions):** the same examples with the same correct labels, but
  the order of the review–label pairs is swapped inside each pair. For example P–N–P–N becomes
  N–P–N–P. These six conditions are reported separately from RQ2.

## Limitations

* Reviews over 400 words are shortened. This cost TF-IDF a little on the shortened reviews. I did
  not run Qwen on the full text, so I do not know how much the shortening affected it.
* IMDb is public, so Qwen may have seen some of these reviews during pretraining. I cannot check
  this.
* Some reviews appear word for word in more than one split (111 between training and test). I
  kept them so the official split stays intact.
* One English dataset, two classes and one small model. Three example sets and one reversal per
  set cover only a small part of the possible prompts.
* The model and dtype are pinned (bfloat16), and all test runs used the same GPU type and batch
  size. This does not guarantee identical results on another GPU type, with another batch size or
  with other library versions. Small differences are possible.
* The p-values from the McNemar tests are not corrected for multiple comparisons.

## Use of AI tools

As stated in the AI declaration in the report:

* ChatGPT was used to improve grammar and spelling, to discuss the research questions and the
  interpretation of the results, and to help structure the report.
* Claude Code was used to help with debugging the experimental code and fixing mistakes.

All results come from the experimental runs and were not generated by AI tools.

## References

* Maas et al. (2011). Learning word vectors for sentiment analysis. ACL.
* Hutto and Gilbert (2014). VADER: A parsimonious rule-based model for sentiment analysis of
  social media text. ICWSM.
* Qwen Team (2024). Qwen2.5-1.5B-Instruct model card. Hugging Face.
* Lu et al. (2022). Fantastically ordered prompts and where to find them. ACL.
