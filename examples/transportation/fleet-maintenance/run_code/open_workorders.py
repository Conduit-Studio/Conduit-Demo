"""Conduit Run Code step: create deterministic demo work-order payloads.

This demo does not call a real fleet-management API. In a production workflow,
replace ``build_workorders`` with an API call and keep the returned rail JSON
small.

Run the local smoke tests from examples/transportation/fleet-maintenance:

    python -m unittest discover -s tests
"""

from __future__ import annotations

from typing import Any

from fleet import (
    build_workorders,
    workorder_message,
)


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    """Conduit entrypoint for producing work-order JSON for critical vehicles."""
    critical = _require_critical(inputs)
    workorders = build_workorders(critical)
    return {
        "opened": {
            "count": len(workorders),
            "vehicles": [workorder["vehicle"] for workorder in workorders],
        },
        "workorders": workorders,
        "message": workorder_message(workorders),
    }


def _require_critical(inputs: dict[str, Any]) -> dict[str, Any]:
    critical = inputs.get("critical")
    if not isinstance(critical, dict):
        raise ValueError("inputs['critical'] must be the critical summary from collect_critical")
    return critical
