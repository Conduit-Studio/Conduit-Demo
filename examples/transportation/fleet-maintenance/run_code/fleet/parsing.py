"""Parsing helpers for fleet telematics records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def vehicle_id(frame: pd.DataFrame, key: str) -> str:
    """Return vehicle_id from the CSV, falling back to the S3 object stem."""
    if "vehicle_id" in frame.columns:
        values = frame["vehicle_id"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return str(values.iloc[0])
    return Path(key).stem


def min_number(frame: pd.DataFrame, column: str) -> float:
    """Return the minimum numeric value for a required column."""
    values = number_series(frame, column)
    if values.empty:
        raise ValueError(f"telematics CSV has no numeric values for column: {column}")
    return float(values.min())


def max_number(frame: pd.DataFrame, column: str) -> float:
    """Return the maximum numeric value for a required column."""
    values = number_series(frame, column)
    if values.empty:
        raise ValueError(f"telematics CSV has no numeric values for column: {column}")
    return float(values.max())


def sum_number(frame: pd.DataFrame, column: str) -> float:
    """Return the numeric sum for a required column."""
    values = number_series(frame, column)
    return float(values.sum()) if not values.empty else 0.0


def active_faults(values: Any) -> list[str]:
    """Return unique DTC fault codes from a pandas Series-like value."""
    if values is None:
        return []
    faults: set[str] = set()
    for raw_value in values.fillna("").astype(str):
        for code in re.split(r"[;,\s|]+", raw_value.upper().strip()):
            if code and code not in {"OK", "NONE", "NAN", "NULL", "-"}:
                faults.add(code)
    return sorted(faults)


def number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return numeric values for a required DataFrame column."""
    if column not in frame.columns:
        raise ValueError(f"telematics CSV missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").dropna()
