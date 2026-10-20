"""Reproducible IMDb sentiment experiment for TDT13 (NTNU)."""

__version__ = "0.1.0"

LABEL_NAMES = {0: "negative", 1: "positive"}
LABEL_IDS = {v: k for k, v in LABEL_NAMES.items()}
INVALID = "invalid"   # unparseable LLM output
NEUTRAL = "neutral"   # VADER compound exactly at threshold (no decision)
