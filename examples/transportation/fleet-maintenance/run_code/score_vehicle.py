"""Conduit Run Code step: score one vehicle telematics CSV.

This file is designed to be used as a Conduit Run Code entry. It reads one
CSV object from S3, computes deterministic maintenance-risk features, and
emits a small JSON payload for DynamoDB, Choice, and Notify nodes.

Run the local smoke tests from examples/transportation/fleet-maintenance:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from fleet import (
    read_s3_ref_bytes,
    require_ref,
    score_vehicle_frame,
)


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    """Conduit entrypoint for scoring a single vehicle CSV object."""
    vehicle_file = require_ref(inputs, "vehicle_file", aliases=("object", "item", "trigger"))
    local_s3_root = inputs.get("_local_s3_root")
    csv_bytes = read_s3_ref_bytes(vehicle_file, local_s3_root=local_s3_root)
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    health = score_vehicle_frame(frame=frame, source_ref=vehicle_file)
    return {
        "vehicle_health": health,
        "vehicle": health["vehicle"],
        "risk_tier": health["risk_tier"],
        "risk_score": health["risk_score"],
    }
