"""Pure confidence-thresholded selection over pseudo-labels — no torch, fully unit-testable.

The pseudo-label step emits one prediction per pooled example (`id`, argmax `label`,
softmax `confidence`). `select_confident` keeps only the predictions whose confidence
clears a threshold — these become this round's new labelled batch. Keeping the bar high
is the guard against confirmation bias (a wrong-but-confident pseudo-label poisons the
next round), which is why the threshold is a config dimension, not a constant.
"""
from __future__ import annotations

from typing import Any

from .schemas import Pred

DEFAULT_THRESHOLD = 0.95


def select_confident(preds: list[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> list[dict[str, Any]]:
    """Keep predictions with confidence >= threshold (best → worst, deterministic).

    Args:
        preds: raw prediction dicts, each {id, label, confidence}.
        threshold: minimum softmax max-prob to accept a pseudo-label.

    Returns:
        The kept predictions as plain dicts {id, label, confidence}, sorted by
        confidence descending then id ascending (stable across runs). Predictions
        below the threshold are dropped; ill-formed ones raise via Pred.from_item.
    """
    kept = [Pred.from_item(p) for p in preds if Pred.from_item(p).confidence >= threshold]
    kept.sort(key=lambda p: (-p.confidence, p.id))
    return [p.as_dict() for p in kept]
