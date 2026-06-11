"""Aggregation helpers for fleet maintenance results."""

from __future__ import annotations

from typing import Any


def require_results(inputs: dict[str, Any]) -> Any:
    """Return Map results from one of the accepted rail keys."""
    for key in ("results", "items", "map_results", "vehicle_health"):
        if key in inputs:
            return inputs[key]
    raise ValueError("inputs must include Map results under results, items, map_results, or vehicle_health")


def collect_critical_vehicles(results: Any) -> dict[str, Any]:
    """Return the critical-vehicle summary consumed by Choice and work orders."""
    health_items = [_normalize_health(item) for item in _flatten_results(results)]
    health_items = [item for item in health_items if item]
    critical_items = [
        item
        for item in health_items
        if item.get("risk_tier") == "critical" or float(item.get("risk_score", 0.0)) >= 0.8
    ]
    critical_items.sort(key=lambda item: float(item.get("risk_score", 0.0)), reverse=True)

    risk_counts = {"critical": 0, "warning": 0, "routine": 0}
    for item in health_items:
        tier = str(item.get("risk_tier", "routine"))
        risk_counts[tier] = risk_counts.get(tier, 0) + 1

    vehicles = [str(item["vehicle"]) for item in critical_items]
    return {
        "count": len(critical_items),
        "vehicles": vehicles,
        "items": critical_items,
        "highest_risk": critical_items[0] if critical_items else None,
        "total_scored": len(health_items),
        "risk_counts": risk_counts,
        "summary": _summary_text(vehicles=vehicles, total=len(health_items)),
    }


def _flatten_results(value: Any) -> list[Any]:
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_results(item))
        return flattened
    return [value]


def _normalize_health(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if isinstance(item.get("vehicle_health"), dict):
        return item["vehicle_health"]
    if isinstance(item.get("output"), dict):
        return _normalize_health(item["output"])
    if isinstance(item.get("outputs"), dict):
        return _normalize_health(item["outputs"])
    if item.get("vehicle") and item.get("risk_tier"):
        return item
    return {}


def _summary_text(vehicles: list[str], total: int) -> str:
    if not vehicles:
        return f"All clear: {total} vehicles scored with no critical maintenance risk."
    joined = ", ".join(vehicles)
    return f"{len(vehicles)} critical vehicles need maintenance review: {joined}."
