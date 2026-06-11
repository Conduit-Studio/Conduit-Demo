"""Small typed structures used by the YOLO demo helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class S3Ref:
    bucket: str
    key: str

    def as_dict(self) -> dict[str, str]:
        return {"bucket": self.bucket, "key": self.key}


@dataclass(frozen=True)
class Prediction:
    image_id: str
    label: str
    confidence: float
    keypoints: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "label": self.label,
            "confidence": self.confidence,
            "keypoints": self.keypoints,
        }
