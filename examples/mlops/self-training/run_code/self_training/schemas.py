"""Typed structures for the self-training loop selection + merge.

`pseudo_label.py` runs REAL inference over the unlabeled pool and emits one
`Pred` per pooled example: which example (`id`), the model's argmax `label`, and
the softmax max-probability `confidence`. `Pred.from_item` normalises whatever
shape the inference step emits into a single comparable record. Pure Python, no
torch — so the keep/merge logic is unit-testable off a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _is_number(value: Any) -> bool:
    # bool is an int subclass — exclude it so a stray True can't read as 1.0.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Pred:
    """One pseudo-label: a pooled example, the model's predicted class, its confidence."""

    id: str
    label: int
    confidence: float

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> "Pred":
        if "id" not in item:
            raise ValueError(f"prediction missing required 'id': {item!r}")
        raw_conf = item.get("confidence")
        if not _is_number(raw_conf):
            raise ValueError(f"prediction {item.get('id')!r} has non-numeric confidence {raw_conf!r}")
        raw_label = item.get("label")
        if not _is_number(raw_label):
            raise ValueError(f"prediction {item.get('id')!r} has non-numeric label {raw_label!r}")
        return cls(id=str(item["id"]), label=int(raw_label), confidence=float(raw_conf))

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "confidence": self.confidence}
