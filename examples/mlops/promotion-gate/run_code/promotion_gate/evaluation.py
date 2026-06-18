"""Pure candidate-vs-baseline comparison for the promotion gate — no AWS, fully unit-testable.

A retrain produces a candidate with an eval metric; the gate compares it to the current-prod
baseline and decides whether it is worth promoting (subject to an optional minimum improvement).
`evaluate` is the comparison the Run Code node runs and the local driver runs — identical code.
"""
from __future__ import annotations

from typing import Any


def evaluate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    metric: str = "accuracy",
    min_delta: float = 0.0,
    model_package_arn: str | None = None,
) -> dict[str, Any]:
    """Compare `candidate_metrics[metric]` to `baseline_metrics[metric]`.

    Returns ``{"report": {...}, "beats": bool}``. The candidate ``beats`` when it is at least
    ``min_delta`` better than the baseline (so `min_delta=0` means "ties promote", and a positive
    value enforces a real improvement before a human is even asked to sign off).
    """
    if metric not in candidate_metrics:
        raise ValueError(f"candidate is missing metric {metric!r}; has {sorted(candidate_metrics)}")
    candidate = float(candidate_metrics[metric])
    baseline = float(baseline_metrics.get(metric, 0.0))
    delta = round(candidate - baseline, 6)
    beats = candidate >= baseline + float(min_delta)
    report = {
        "metric": metric,
        "candidate": candidate,
        "baseline": baseline,
        "delta": delta,
        "minDelta": float(min_delta),
        "beats": beats,
        "modelPackageArn": model_package_arn,
    }
    return {"report": report, "beats": beats}
