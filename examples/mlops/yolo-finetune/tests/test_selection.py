"""Unit tests for the sweep selection logic.

    uv run python -m unittest examples/mlops/yolo-finetune/tests/test_selection.py

Run from the run_code dir so `yolo_finetune` imports, or rely on the sys.path shim below.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow `python -m unittest tests/test_selection.py` from the example root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run_code"))

from yolo_finetune.schemas import TrialResult  # noqa: E402
from yolo_finetune.selection import rank, select_best  # noqa: E402
from select_best import main as select_best_main  # noqa: E402  (Run Code entry: zips Map's separate per-trial lists)

ITEMS = [
    {
        "name": "n-640",
        "model": {"bucket": "b", "key": "n/model.tar.gz"},
        "metrics": {"mAP50-95": 0.62, "mAP50": 0.81},
        "hyperparameters": {"weights": "yolov8n.pt", "imgsz": 640},
    },
    {
        "name": "s-1280",
        "model": {"bucket": "b", "key": "s/model.tar.gz"},
        "metrics": {"mAP50-95": 0.71, "mAP50": 0.89},
        "hyperparameters": {"weights": "yolov8s.pt", "imgsz": 1280},
    },
    {
        "name": "m-640",
        "model": {"bucket": "b", "key": "m/model.tar.gz"},
        "metrics": {"mAP50-95": 0.68},
        "hyperparameters": {"weights": "yolov8m.pt", "imgsz": 640},
    },
]


class TestSelection(unittest.TestCase):
    def test_picks_highest_map(self):
        best = select_best(ITEMS)
        self.assertEqual(best.name, "s-1280")
        self.assertAlmostEqual(best.value, 0.71)
        self.assertEqual(best.metric, "mAP50-95")
        self.assertEqual(best.model["key"], "s/model.tar.gz")

    def test_rank_order(self):
        ranked = rank([TrialResult.from_item(i) for i in ITEMS])
        self.assertEqual([r.name for r in ranked], ["s-1280", "m-640", "n-640"])

    def test_can_rank_by_other_metric(self):
        best = select_best(ITEMS, metric="mAP50")
        self.assertEqual(best.name, "s-1280")

    def test_tie_breaks_to_cheaper_model(self):
        tied = [
            {"name": "m", "model": {}, "metrics": {"mAP50-95": 0.70}, "hyperparameters": {"weights": "yolov8m.pt"}},
            {"name": "n", "model": {}, "metrics": {"mAP50-95": 0.70}, "hyperparameters": {"weights": "yolov8n.pt"}},
        ]
        self.assertEqual(select_best(tied).name, "n")

    def test_missing_metric_raises(self):
        with self.assertRaises(ValueError):
            select_best([{"name": "x", "metrics": {"loss": 1.0}}])

    def test_bool_metric_is_ignored(self):
        # A stray True must not rank as 1.0 above real scores.
        items = [
            {"name": "real", "metrics": {"mAP50-95": 0.5}},
            {"name": "bogus", "metrics": {"mAP50-95": True}},
        ]
        self.assertEqual(select_best(items).name, "real")


class TestSelectBestMain(unittest.TestCase):
    """The Run Code entry (`main`) zips the Map's separate, index-aligned per-trial lists."""

    def test_zips_three_index_aligned_lists_and_threads_model_and_hp(self):
        out = select_best_main({
            "trials": [
                {"name": "n-640", "weights": "yolov8n.pt", "imgsz": "640"},
                {"name": "s-640", "weights": "yolov8s.pt", "imgsz": "640"},
            ],
            "metrics": [{"mAP50-95": 0.31, "mAP50": 0.55}, {"mAP50-95": 0.50, "mAP50": 0.78}],
            "models": [
                {"bucket": "b", "key": "n/model.tar.gz"},
                {"bucket": "b", "key": "s/model.tar.gz"},
            ],
        })
        self.assertEqual(out["best"]["name"], "s-640")          # real name, not the "trial" default
        self.assertAlmostEqual(out["best"]["value"], 0.50)
        self.assertEqual(out["best"]["model"]["key"], "s/model.tar.gz")        # winner's model threads through
        self.assertEqual(out["best"]["hyperparameters"]["weights"], "yolov8s.pt")  # winner's HP threads through

    def test_tolerates_missing_models_at_verify_time(self):
        # codeCheck omits model-artifact ports, so `models` is absent — ranking still works.
        out = select_best_main({
            "trials": [{"name": "n-640", "weights": "yolov8n.pt"}, {"name": "s-640", "weights": "yolov8s.pt"}],
            "metrics": [{"mAP50-95": 0.31}, {"mAP50-95": 0.50}],
        })
        self.assertEqual(out["best"]["name"], "s-640")
        self.assertEqual(out["best"]["model"], {})

    def test_combined_results_still_supported(self):
        out = select_best_main({"results": [
            {"name": "x", "model": {"key": "x.tar.gz"}, "metrics": {"mAP50-95": 0.6}, "hyperparameters": {"weights": "yolov8n.pt"}},
        ]})
        self.assertEqual(out["best"]["name"], "x")
        self.assertEqual(out["best"]["model"]["key"], "x.tar.gz")

    def test_raises_when_no_usable_input(self):
        with self.assertRaises(ValueError):
            select_best_main({"metric": "mAP50-95"})


if __name__ == "__main__":
    unittest.main()
