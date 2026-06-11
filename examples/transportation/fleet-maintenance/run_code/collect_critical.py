"""Conduit Run Code step: aggregate critical vehicles from Map results.

This entry consumes the small JSON outputs from ``score_vehicle.py`` and emits
a compact summary for a Choice node and optional work-order creation.

Run the local smoke tests from examples/transportation/fleet-maintenance:

    python -m unittest discover -s tests
"""

from __future__ import annotations

from typing import Any

from fleet import collect_critical_vehicles, require_results


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    """Conduit entrypoint for aggregating scored vehicle health records."""
    critical = collect_critical_vehicles(require_results(inputs))
    return {
        "critical": critical,
        "has_critical": bool(critical["count"]),
    }
