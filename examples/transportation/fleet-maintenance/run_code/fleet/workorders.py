"""Work-order payload helpers for the fleet maintenance demo."""

from __future__ import annotations

from typing import Any


def build_workorders(critical: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic work-order records for the critical vehicle list."""
    items = critical.get("items")
    if not isinstance(items, list) or not items:
        items = [{"vehicle": vehicle, "risk_score": 0.8, "active_faults": []} for vehicle in critical.get("vehicles", [])]

    workorders = []
    for item in items:
        if not isinstance(item, dict) or not item.get("vehicle"):
            continue
        score = float(item.get("risk_score", 0.8))
        vehicle = str(item["vehicle"])
        workorders.append({
            "id": f"WO-{vehicle}-{int(round(score * 1000)):03d}",
            "vehicle": vehicle,
            "priority": "urgent" if score >= 0.85 else "high",
            "risk_score": round(score, 4),
            "reason": _reason(item),
            "status": "opened",
        })
    return workorders


def workorder_message(workorders: list[dict[str, Any]]) -> str:
    """Return a Slack-friendly one-line summary."""
    if not workorders:
        return "No critical vehicles; no work orders opened."
    vehicles = ", ".join(workorder["vehicle"] for workorder in workorders)
    return f"Opened {len(workorders)} fleet maintenance work orders for {vehicles}."


def _reason(item: dict[str, Any]) -> str:
    faults = item.get("active_faults")
    if isinstance(faults, list) and faults:
        return f"Active fault codes: {', '.join(str(code) for code in faults)}"
    stats = item.get("stats")
    if isinstance(stats, dict) and stats.get("brake_pad_min_mm") is not None:
        return f"Brake pad minimum {stats['brake_pad_min_mm']} mm"
    return "Critical maintenance-risk score"
