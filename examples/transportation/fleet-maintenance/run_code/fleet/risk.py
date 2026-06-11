"""Maintenance-risk scoring for fleet telematics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .parsing import (
    active_faults,
    max_number,
    min_number,
    sum_number,
    vehicle_id,
)


CRITICAL_FAULT_CODES = {"P0301", "P0302", "P0420", "C1234", "C121C", "U0100"}


def score_vehicle_frame(frame: pd.DataFrame, source_ref: dict[str, Any]) -> dict[str, Any]:
    """Return a compact maintenance-risk JSON payload for one vehicle."""
    if frame.empty:
        raise ValueError("telematics CSV has no rows")

    key = str(source_ref["key"]).lstrip("/")
    vehicle = vehicle_id(frame=frame, key=key)
    engine_temp_max = max_number(frame, "engine_temp_c")
    oil_pressure_min = min_number(frame, "oil_pressure_psi")
    brake_pad_min = min_number(frame, "brake_pad_mm")
    odometer_km = max_number(frame, "odometer_km")
    harsh_brake_total = sum_number(frame, "harsh_brake_count")
    harsh_accel_total = sum_number(frame, "harsh_accel_count")
    faults = active_faults(frame.get("dtc_codes"))

    scores = {
        "brakes": _clip((6.0 - brake_pad_min) / 4.0),
        "engine_temp": _clip((engine_temp_max - 95.0) / 25.0),
        "oil_pressure": _clip((35.0 - oil_pressure_min) / 20.0),
        "faults": _clip(len(faults) / 2.0),
        "driving_events": _clip((harsh_brake_total + harsh_accel_total) / 20.0),
    }
    weighted_score = (
        scores["brakes"] * 0.35
        + scores["engine_temp"] * 0.20
        + scores["oil_pressure"] * 0.20
        + scores["faults"] * 0.15
        + scores["driving_events"] * 0.10
    )
    if CRITICAL_FAULT_CODES.intersection(faults):
        weighted_score = max(weighted_score, 0.86)
    if brake_pad_min < 2.5:
        weighted_score = max(weighted_score, 0.82)

    risk_score = round(float(_clip(weighted_score)), 4)
    risk_tier = _risk_tier(risk_score)
    return {
        "pk": f"VEHICLE#{vehicle}",
        "sk": f"HEALTH#{Path(key).stem}",
        "vehicle": vehicle,
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "needs_work_order": risk_tier == "critical",
        "active_faults": faults,
        "scores": {name: round(float(value), 4) for name, value in scores.items()},
        "stats": {
            "rows": int(len(frame)),
            "engine_temp_max_c": round(float(engine_temp_max), 2),
            "oil_pressure_min_psi": round(float(oil_pressure_min), 2),
            "brake_pad_min_mm": round(float(brake_pad_min), 2),
            "odometer_km": round(float(odometer_km), 1),
            "harsh_brake_count": int(harsh_brake_total),
            "harsh_accel_count": int(harsh_accel_total),
        },
        "source": {
            "bucket": str(source_ref["bucket"]),
            "key": key,
        },
    }


def _risk_tier(score: float) -> str:
    if score >= 0.8:
        return "critical"
    if score >= 0.55:
        return "warning"
    return "routine"


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
