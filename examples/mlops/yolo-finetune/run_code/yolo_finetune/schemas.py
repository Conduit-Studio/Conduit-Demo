"""Typed structures for the YOLO fine-tune sweep selection.

A Map over the sweep grid produces one item per trial; `TrialResult.from_item`
normalises whatever shape the upstream Train Model + metrics emit into a single
comparable record, and `Best` is the winner select_best returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _is_number(value: Any) -> bool:
    # bool is an int subclass — exclude it so a stray True can't rank as 1.0.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class TrialResult:
    """One trial's outcome: which config ran, where its model landed, how it scored."""

    name: str
    model: dict[str, Any]              # s3-ref to the trained model artifact: {bucket, key}
    metrics: dict[str, float]         # {"mAP50-95": 0.71, "mAP50": 0.89, ...}
    hyperparameters: dict[str, Any]   # {weights, imgsz, lr0, epochs}

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> "TrialResult":
        trial = item.get("trial") or {}
        metrics_in = item.get("metrics") or {}
        return cls(
            name=str(item.get("name") or trial.get("name") or "trial"),
            model=dict(item.get("model") or item.get("modelArtifact") or {}),
            metrics={k: float(v) for k, v in metrics_in.items() if _is_number(v)},
            hyperparameters=dict(item.get("hyperparameters") or trial),
        )


@dataclass(frozen=True)
class Best:
    """The winning trial, ready to hand to a Choice gate or a register step."""

    name: str
    metric: str
    value: float
    model: dict[str, Any]
    hyperparameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "value": self.value,
            "model": self.model,
            "hyperparameters": self.hyperparameters,
        }
