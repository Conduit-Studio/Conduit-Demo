"""Conduit Run Code step: load a model reference and infer study urgency.

This file is designed to be uploaded directly into a Conduit Run Code node.
It consumes the feature artifact emitted by ``read_study.py`` plus an S3 model
reference, loads the RandomForestClassifier package, and emits rail JSON for
Choice, DynamoDB Put, and Slack notification nodes.

Run the focused local tests:

    uv run python -m unittest examples/medical/test_run_code.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


MODEL_VERSION = "rf-baseline-v1"
URGENCY_LABELS = ("routine", "critical")


class S3Client(Protocol):
    def download_file(self, bucket: str, key: str, filename: str) -> None:
        """Download an S3 object to a local file."""


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    """Conduit entrypoint for producing urgency rail JSON."""
    features_payload = _require_features(inputs.get("features"))
    model_ref = _require_ref(inputs, "model")
    model_path = load_model_ref(model_ref, local_s3_root=inputs.get("_local_s3_root"))
    package = joblib.load(model_path)

    vector = np.asarray(features_payload["features"]["vector"], dtype=np.float32)
    classifier = package["model"]
    probabilities = _predict_probabilities(classifier, vector)
    predicted_class = max(probabilities, key=probabilities.get)
    confidence = round(float(probabilities[predicted_class]), 4)

    urgency = {
        "study_id": str(features_payload["study_id"]),
        "class": predicted_class,
        "confidence": confidence,
        "findings": ["pneumothorax"] if predicted_class == "critical" else [],
        "model": {
            "type": str(package.get("model_type", type(classifier).__name__)),
            "version": str(package.get("model_version", "unknown")),
        },
        "probabilities": {
            label: round(float(probabilities.get(label, 0.0)), 4)
            for label in URGENCY_LABELS
        },
        "stats": {
            "slice_count": int(features_payload.get("stats", {}).get("slice_count", 0)),
        },
    }
    return {"urgency": urgency}


def load_model_ref(model_ref: dict[str, Any], local_s3_root: str | None = None) -> Path:
    """Load a model S3 ref, using a local mock-S3 root during tests."""
    key = str(model_ref["key"]).lstrip("/")
    if local_s3_root:
        return Path(local_s3_root) / key

    destination = Path("/tmp") / Path(key).name
    _new_s3_client().download_file(str(model_ref["bucket"]), key, str(destination))
    return destination


def train_test_model_for_fixture(
    X: np.ndarray,
    y: np.ndarray,
    model_path: Path,
    image_size: int,
) -> dict[str, Any]:
    """Train a tiny deterministic model package for unit-test fixtures only."""
    classifier = RandomForestClassifier(
        n_estimators=20,
        random_state=7,
        class_weight="balanced",
        n_jobs=1,
    )
    classifier.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "model": classifier,
        "model_type": "RandomForestClassifier",
        "model_version": MODEL_VERSION,
        "image_size": image_size,
        "labels": list(URGENCY_LABELS),
        "feature_count": int(X.shape[1]),
    }
    joblib.dump(package, model_path)
    return package


def _predict_probabilities(classifier: Any, features: np.ndarray) -> dict[str, float]:
    probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
    return {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, probabilities, strict=True)
    }


def _require_features(features_payload: Any) -> dict[str, Any]:
    if isinstance(features_payload, str):
        path = Path(features_payload)
        if not path.exists():
            raise ValueError(f"inputs['features'] file does not exist: {path}")
        features_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(features_payload, dict):
        raise ValueError("inputs['features'] must be the feature artifact from read_study")
    vector = features_payload.get("features", {}).get("vector")
    if not isinstance(vector, list) or not vector:
        raise ValueError("inputs['features']['features']['vector'] must be a non-empty list")
    return features_payload


def _require_ref(inputs: dict[str, Any], name: str) -> dict[str, Any]:
    ref = inputs.get(name)
    if not isinstance(ref, dict) or not ref.get("bucket") or not ref.get("key"):
        raise ValueError(f"inputs[{name!r}] must be an S3 ref with bucket and key")
    return ref


def _new_s3_client() -> S3Client:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 in the Run Code requirements to load the model from S3.") from exc
    return boto3.client("s3")
