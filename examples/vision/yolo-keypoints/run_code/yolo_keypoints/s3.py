"""S3 and local-file helpers for Conduit Run Code entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schemas import S3Ref


EXAMPLE_MARKER = "examples/vision/yolo-keypoints/"


def require_ref(inputs: dict[str, Any], name: str, aliases: tuple[str, ...] = ()) -> S3Ref:
    for candidate_name in (name, *aliases):
        candidate = inputs.get(candidate_name)
        ref = parse_ref(candidate)
        if ref is not None:
            return ref
    names = ", ".join(repr(value) for value in (name, *aliases))
    raise ValueError(f"one of inputs[{names}] must be an S3 ref with bucket and key")


def parse_ref(value: Any) -> S3Ref | None:
    if isinstance(value, dict) and value.get("bucket") and value.get("key"):
        return S3Ref(bucket=str(value["bucket"]), key=str(value["key"]).lstrip("/"))
    if isinstance(value, str) and value.startswith("s3://"):
        parsed = urlparse(value)
        if parsed.netloc and parsed.path.strip("/"):
            return S3Ref(bucket=parsed.netloc, key=parsed.path.lstrip("/"))
    return None


def output_bucket(inputs: dict[str, Any], fallback: S3Ref) -> str:
    return str(inputs.get("output_bucket") or inputs.get("bucket") or fallback.bucket)


def join_key(prefix: str, filename: str) -> str:
    clean_prefix = str(prefix or "").strip("/")
    clean_name = filename.lstrip("/")
    return f"{clean_prefix}/{clean_name}" if clean_prefix else clean_name


def read_bytes(ref: S3Ref, local_s3_root: str | None = None) -> bytes:
    if local_s3_root:
        return _local_path_for_key(local_s3_root, ref.key).read_bytes()
    client = _new_s3_client()
    response = client.get_object(Bucket=ref.bucket, Key=ref.key)
    return response["Body"].read()


def read_json(ref: S3Ref, local_s3_root: str | None = None) -> Any:
    return json.loads(read_bytes(ref, local_s3_root=local_s3_root).decode("utf-8"))


def write_bytes(
    ref: S3Ref,
    data: bytes,
    *,
    content_type: str,
    local_output_root: str | None = None,
) -> dict[str, str]:
    if local_output_root:
        path = Path(local_output_root) / ref.key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ref.as_dict()
    client = _new_s3_client()
    client.put_object(Bucket=ref.bucket, Key=ref.key, Body=data, ContentType=content_type)
    return ref.as_dict()


def write_json(
    ref: S3Ref,
    payload: Any,
    *,
    local_output_root: str | None = None,
) -> dict[str, str]:
    body = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    return write_bytes(ref, body, content_type="application/json", local_output_root=local_output_root)


def _local_path_for_key(local_s3_root: str, key: str) -> Path:
    root = Path(local_s3_root)
    direct = root / key
    if direct.exists():
        return direct
    if EXAMPLE_MARKER in key:
        stripped = key.split(EXAMPLE_MARKER, 1)[1]
        return root / stripped
    return direct


def _new_s3_client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 in Run Code requirements to read or write S3 objects.") from exc
    return boto3.client("s3")
