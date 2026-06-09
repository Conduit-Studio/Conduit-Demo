"""Conduit Run Code step: read one radiology study manifest and extract features.

This file is designed to be uploaded directly into a Conduit Run Code node.
It reads an S3 manifest reference, lists the DICOM files for the referenced
series, extracts deterministic image features, and writes the feature payload
as a file artifact for the next node. Use an s3-ref output port named
``features`` so large feature vectors do not pass through ECS environment
overrides.

Run the focused local tests:

    uv run python -m unittest examples/medical/test_run_code.py
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pydicom


DEFAULT_IMAGE_SIZE = 32


class S3Client(Protocol):
    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        """Return an S3 object body."""

    def list_objects_v2(self, Bucket: str, Prefix: str) -> dict[str, Any]:
        """List objects under an S3 prefix."""


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    """Conduit entrypoint for reading a study and emitting model features."""
    manifest_ref = _require_ref(inputs, "manifest", aliases=("trigger",))
    image_size = int(inputs.get("_image_size", DEFAULT_IMAGE_SIZE))
    local_s3_root = inputs.get("_local_s3_root")

    manifest_bytes = _read_s3_ref_bytes(manifest_ref, local_s3_root=local_s3_root)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    base_prefix = _base_prefix_for_manifest_key(str(manifest_ref["key"]))
    dicom_refs = _dicom_refs_for_manifest(
        bucket=str(manifest_ref["bucket"]),
        manifest=manifest,
        base_prefix=base_prefix,
        local_s3_root=local_s3_root,
    )
    if not dicom_refs:
        raise ValueError(f"Manifest does not reference any DICOM files: {manifest_ref['key']}")

    vectors = []
    for dicom_ref in dicom_refs:
        dicom_bytes = _read_s3_ref_bytes(dicom_ref, local_s3_root=local_s3_root)
        vectors.append(extract_dicom_features_from_bytes(dicom_bytes, image_size=image_size))

    matrix = np.vstack(vectors).astype(np.float32)
    average_vector = np.mean(matrix, axis=0)
    payload = {
        "study_id": str(manifest["study_id"]),
        "series_complete": _series_complete(manifest=manifest, observed_count=len(dicom_refs)),
        "features": {
            "vector": average_vector.astype(float).tolist(),
            "image_size": image_size,
            "source": "dicom-stat-sampled",
        },
        "stats": {
            "slice_count": len(dicom_refs),
            "feature_count": int(average_vector.shape[0]),
            "mean_intensity": round(float(average_vector[0]), 4),
            "std_intensity": round(float(average_vector[1]), 4),
        },
    }
    return {
        "features": str(_write_features_payload(payload)),
        "study_id": payload["study_id"],
        "series_complete": payload["series_complete"],
        "stats": payload["stats"],
    }


def extract_dicom_features(dicom_path: Path, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Return a fixed-length feature vector from one DICOM image path."""
    dataset = pydicom.dcmread(str(dicom_path))
    return _extract_dicom_features_from_dataset(dataset=dataset, image_size=image_size)


def extract_dicom_features_from_bytes(dicom_bytes: bytes, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Return a fixed-length feature vector from one DICOM object body."""
    dataset = pydicom.dcmread(io.BytesIO(dicom_bytes))
    return _extract_dicom_features_from_dataset(dataset=dataset, image_size=image_size)


def _extract_dicom_features_from_dataset(dataset: pydicom.dataset.Dataset, image_size: int) -> np.ndarray:
    pixels = _read_pixels(dataset)
    stats = np.array(
        [
            float(np.mean(pixels)),
            float(np.std(pixels)),
            float(np.min(pixels)),
            float(np.max(pixels)),
            float(np.percentile(pixels, 5)),
            float(np.percentile(pixels, 95)),
        ],
        dtype=np.float32,
    )
    normalized = _normalize_pixels(pixels)
    sampled = _sample_square(normalized, image_size)
    return np.concatenate([stats, sampled.reshape(-1).astype(np.float32)])


def _read_pixels(dataset: pydicom.dataset.Dataset) -> np.ndarray:
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = float(np.max(pixels)) - pixels
    return pixels


def _normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    lower = float(np.percentile(pixels, 1))
    upper = float(np.percentile(pixels, 99))
    if upper <= lower:
        return np.zeros_like(pixels, dtype=np.float32)
    clipped = np.clip(pixels, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def _sample_square(pixels: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")
    row_indexes = np.linspace(0, pixels.shape[0] - 1, image_size).astype(int)
    column_indexes = np.linspace(0, pixels.shape[1] - 1, image_size).astype(int)
    return pixels[np.ix_(row_indexes, column_indexes)]


def _dicom_refs_for_manifest(
    bucket: str,
    manifest: dict[str, Any],
    base_prefix: str,
    local_s3_root: str | None,
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for series in manifest.get("series", []):
        series_key = _join_s3_key(base_prefix, str(series["key"]))
        refs.extend(_list_dicom_refs(bucket=bucket, prefix=series_key, local_s3_root=local_s3_root))
    return refs


def _list_dicom_refs(bucket: str, prefix: str, local_s3_root: str | None) -> list[dict[str, str]]:
    normalized_prefix = prefix.lstrip("/")
    if local_s3_root:
        series_dir = Path(local_s3_root) / normalized_prefix
        return [
            {"bucket": bucket, "key": path.relative_to(local_s3_root).as_posix()}
            for path in sorted(series_dir.glob("*.dcm"))
            if path.is_file()
        ]

    client = _new_s3_client()
    refs: list[dict[str, str]] = []
    continuation_token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": normalized_prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = str(item.get("Key", ""))
            if key.lower().endswith(".dcm"):
                refs.append({"bucket": bucket, "key": key})
        if not response.get("IsTruncated"):
            break
        continuation_token = str(response["NextContinuationToken"])
    return refs


def _read_s3_ref_bytes(ref: dict[str, Any], local_s3_root: str | None) -> bytes:
    key = str(ref["key"]).lstrip("/")
    if local_s3_root:
        return (Path(local_s3_root) / key).read_bytes()

    client = _new_s3_client()
    response = client.get_object(Bucket=str(ref["bucket"]), Key=key)
    return response["Body"].read()


def _write_features_payload(payload: dict[str, Any]) -> Path:
    fd, path = tempfile.mkstemp(prefix="conduit-features-", suffix=".json", dir="/tmp")
    output_path = Path(path)
    with open(fd, "w", encoding="utf-8") as output:
        json.dump(payload, output, separators=(",", ":"))
        output.write("\n")
    return output_path


def _base_prefix_for_manifest_key(manifest_key: str) -> str:
    marker = "studies/incoming/"
    normalized_key = manifest_key.lstrip("/")
    if marker not in normalized_key:
        return ""
    return normalized_key.split(marker, 1)[0]


def _join_s3_key(prefix: str, key: str) -> str:
    normalized_prefix = prefix.strip("/")
    normalized_key = key.lstrip("/")
    if not normalized_prefix:
        return normalized_key
    return f"{normalized_prefix}/{normalized_key}"


def _series_complete(manifest: dict[str, Any], observed_count: int) -> bool:
    expected_count = sum(int(series.get("instance_count", 0)) for series in manifest.get("series", []))
    return expected_count == observed_count


def _require_ref(inputs: dict[str, Any], name: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    ref = inputs.get(name)
    if not isinstance(ref, dict) or not ref.get("bucket") or not ref.get("key"):
        for alias in aliases:
            candidate = inputs.get(alias)
            if isinstance(candidate, dict) and candidate.get("bucket") and candidate.get("key"):
                ref = candidate
                break
    if not isinstance(ref, dict) or not ref.get("bucket") or not ref.get("key"):
        names = ", ".join(repr(value) for value in (name, *aliases))
        raise ValueError(f"one of inputs[{names}] must be an S3 ref with bucket and key")
    return ref


def _new_s3_client() -> S3Client:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 in the Run Code requirements to read S3 objects.") from exc
    return boto3.client("s3")
