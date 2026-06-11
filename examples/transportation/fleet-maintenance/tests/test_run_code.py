"""Smoke tests for the public fleet maintenance demo.

Run from examples/transportation/fleet-maintenance:

    python -m unittest discover -s tests
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parents[1]
RUN_CODE_DIR = EXAMPLE_DIR / "run_code"
sys.path.insert(0, str(RUN_CODE_DIR))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


score_vehicle = load_module(EXAMPLE_DIR / "run_code" / "score_vehicle.py", "demo_score_vehicle")
collect_critical = load_module(EXAMPLE_DIR / "run_code" / "collect_critical.py", "demo_collect_critical")
open_workorders = load_module(EXAMPLE_DIR / "run_code" / "open_workorders.py", "demo_open_workorders")


class FleetMaintenanceRunCodeTest(unittest.TestCase):
    def test_scores_routine_vehicle_from_csv_ref(self) -> None:
        result = score_vehicle.main({
            "object": {
                "bucket": "demo-bucket",
                "key": "data/telematics/2026-06-05/veh-003.csv",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })

        health = result["vehicle_health"]
        self.assertEqual(health["vehicle"], "veh-003")
        self.assertEqual(health["risk_tier"], "routine")
        self.assertLess(health["risk_score"], 0.5)
        self.assertEqual(health["active_faults"], [])
        self.assertEqual(health["pk"], "VEHICLE#veh-003")

    def test_scores_critical_vehicle_from_csv_ref(self) -> None:
        result = score_vehicle.main({
            "vehicle_file": {
                "bucket": "demo-bucket",
                "key": "data/telematics/2026-06-05/veh-001.csv",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })

        health = result["vehicle_health"]
        self.assertEqual(health["vehicle"], "veh-001")
        self.assertEqual(health["risk_tier"], "critical")
        self.assertGreaterEqual(health["risk_score"], 0.85)
        self.assertIn("P0301", health["active_faults"])

    def test_collects_critical_vehicles_from_map_results(self) -> None:
        routine = score_vehicle.main({
            "object": {
                "bucket": "demo-bucket",
                "key": "data/telematics/2026-06-05/veh-003.csv",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })["vehicle_health"]
        critical = score_vehicle.main({
            "object": {
                "bucket": "demo-bucket",
                "key": "data/telematics/2026-06-05/veh-001.csv",
            },
            "_local_s3_root": str(EXAMPLE_DIR),
        })["vehicle_health"]

        result = collect_critical.main({"results": [routine, critical]})

        self.assertEqual(result["critical"]["count"], 1)
        self.assertEqual(result["critical"]["vehicles"], ["veh-001"])
        self.assertEqual(result["critical"]["highest_risk"]["vehicle"], "veh-001")

    def test_open_workorders_returns_a_small_rail_payload(self) -> None:
        result = open_workorders.main({
            "critical": {
                "count": 2,
                "vehicles": ["veh-001", "veh-014"],
                "items": [
                    {
                        "vehicle": "veh-001",
                        "risk_tier": "critical",
                        "risk_score": 0.94,
                        "active_faults": ["P0301"],
                    },
                    {
                        "vehicle": "veh-014",
                        "risk_tier": "critical",
                        "risk_score": 0.91,
                        "active_faults": ["C1234"],
                    },
                ],
            }
        })

        workorders = result["workorders"]
        self.assertEqual(result["opened"]["count"], 2)
        self.assertEqual(workorders[0]["id"], "WO-veh-001-940")
        self.assertEqual(workorders[0]["priority"], "urgent")


if __name__ == "__main__":
    unittest.main()
