"""Pure ranking/selection over sweep trial results — no AWS, fully unit-testable.

The sweep's Map emits a list of per-trial records; `select_best` ranks them by a
metric (default mAP@0.5:0.95, higher is better) and returns the winner. Ties break
deterministically toward the cheaper model (fewer params: n < s < m < l < x), so the
same sweep always picks the same model.
"""
from __future__ import annotations

from typing import Any

from .schemas import Best, TrialResult

DEFAULT_METRIC = "mAP50-95"

# Cheaper-first ordering for tie-breaks (ultralytics size suffix).
_SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def _size_rank(result: TrialResult) -> int:
    weights = str(result.hyperparameters.get("weights", ""))
    for suffix, rank in _SIZE_ORDER.items():
        # match e.g. "yolov8s.pt" → 's'
        if weights[:-3].endswith(suffix) or weights.endswith(suffix):
            return rank
    return len(_SIZE_ORDER)


def rank(results: list[TrialResult], metric: str = DEFAULT_METRIC) -> list[TrialResult]:
    """Best → worst by `metric`; trials missing the metric are dropped."""
    scored = [r for r in results if metric in r.metrics]
    # Sort by (−score, size_rank, name): higher score first, then cheaper model, then stable.
    return sorted(scored, key=lambda r: (-r.metrics[metric], _size_rank(r), r.name))


def select_best(items: list[dict[str, Any]], metric: str = DEFAULT_METRIC) -> Best:
    """Pick the winning trial from raw Map output. Raises if none reported `metric`."""
    results = [TrialResult.from_item(i) for i in items]
    ranked = rank(results, metric)
    if not ranked:
        names = [r.name for r in results]
        raise ValueError(f"no trial reported metric {metric!r}; trials seen: {names}")
    top = ranked[0]
    return Best(
        name=top.name,
        metric=metric,
        value=top.metrics[metric],
        model=top.model,
        hyperparameters=top.hyperparameters,
    )
