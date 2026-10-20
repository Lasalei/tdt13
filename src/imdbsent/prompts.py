"""Prompt construction and output parsing for the Qwen conditions.

Every LLM condition uses the same system prompt, the same per-review user template and
the same answer format. Few-shot demonstrations are given as preceding chat turns:

  system:    <system_prompt>
  user:      <instruction>\n\nReview: <demo 1>      } repeated k times
  assistant: positive|negative                     }  (k = 0, 2 or 4)
  user:      <instruction>\n\nReview: <target review>
  assistant: (generated)

Zero-shot is exactly the k = 0 case, so the only difference between conditions is the
presence, number and identity of the demonstrations.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from . import INVALID

PARSE_RULE = (
    "lowercase, strip whitespace; strip leading/trailing non-letters; take the first "
    "word; valid iff it is exactly 'positive' or 'negative'; anything else -> invalid"
)
_EDGE = re.compile(r"^[^a-z]+|[^a-z]+$")


def format_review(cfg: dict[str, Any], review: str) -> str:
    lcfg = cfg["llm"]
    return lcfg["review_template"].format(instruction=lcfg["instruction"].strip(), review=review)


def build_messages(cfg: dict[str, Any], review: str,
                   demos: Sequence[dict[str, Any]] = ()) -> list[dict[str, str]]:
    """demos: sequence of {"text": ..., "label": "positive"|"negative"} in prompt order."""
    msgs = [{"role": "system", "content": cfg["llm"]["system_prompt"].strip()}]
    for d in demos:
        if d["label"] not in ("positive", "negative"):
            raise ValueError(f"bad demo label {d['label']!r}")
        msgs.append({"role": "user", "content": format_review(cfg, d["text"])})
        msgs.append({"role": "assistant", "content": d["label"]})
    msgs.append({"role": "user", "content": format_review(cfg, review)})
    return msgs


def parse_output(raw: str) -> str:
    """Strict parser. Returns 'positive', 'negative' or 'invalid'."""
    s = raw.strip().lower()
    if not s:
        return INVALID
    first = _EDGE.sub("", s.split()[0])
    return first if first in ("positive", "negative") else INVALID
