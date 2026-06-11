"""S3 and rail-input helpers for fleet Run Code entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class S3Client(Protocol):
    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        """Return an S3 object body."""


def require_ref(inputs: dict[str, Any], name: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the first S3 ref present under name or aliases."""
    for candidate in (name, *aliases):
        ref = inputs.get(candidate)
        if isinstance(ref, dict) and ref.get("bucket") and ref.get("key"):
            return ref
    names = ", ".join(repr(candidate) for candidate in (name, *aliases))
    raise ValueError(f"one of inputs[{names}] must be an S3 ref with bucket and key")


def read_s3_ref_bytes(ref: dict[str, Any], local_s3_root: str | None) -> bytes:
    """Read an S3 ref, using a local mock-S3 root during tests."""
    key = str(ref["key"]).lstrip("/")
    if local_s3_root:
        return (Path(local_s3_root) / key).read_bytes()

    client = _new_s3_client()
    response = client.get_object(Bucket=str(ref["bucket"]), Key=key)
    return response["Body"].read()


def _new_s3_client() -> S3Client:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 in the Run Code requirements to read telematics CSVs from S3.") from exc
    return boto3.client("s3")
