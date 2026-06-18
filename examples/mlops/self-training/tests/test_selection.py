"""Unit tests for the self-training loop's pure logic (selection + round-merge).

These never import torch — only the deterministic keep/merge bookkeeping is tested
here; the REAL training (train/finetune.py) and REAL inference (run_code/pseudo_label.py)
are exercised by scripts/run_loop_local.py against a real model + dataset + GPU.

    python -m unittest examples/mlops/self-training/tests/test_selection.py

Run from the example root, or rely on the sys.path shim below.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow `python -m unittest tests/test_selection.py` from the example root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run_code"))

from self_training.selection import DEFAULT_THRESHOLD, select_confident  # noqa: E402
from self_training.merge import merge_round  # noqa: E402
from select_confident import main as select_confident_main  # noqa: E402  (Run Code entry)
from merge import main as merge_main  # noqa: E402  (Run Code entry / round summary)

PREDS = [
    {"id": "p0", "label": 3, "confidence": 0.99},
    {"id": "p1", "label": 5, "confidence": 0.80},
    {"id": "p2", "label": 1, "confidence": 0.96},
    {"id": "p3", "label": 7, "confidence": 0.50},
]


class TestSelectConfident(unittest.TestCase):
    def test_keeps_at_or_above_threshold_drops_below(self):
        kept = select_confident(PREDS, threshold=0.95)
        self.assertEqual([k["id"] for k in kept], ["p0", "p2"])  # 0.99, 0.96 kept; 0.80, 0.50 dropped

    def test_boundary_is_inclusive(self):
        kept = select_confident([{"id": "edge", "label": 0, "confidence": 0.95}], threshold=0.95)
        self.assertEqual([k["id"] for k in kept], ["edge"])

    def test_sorted_by_confidence_descending(self):
        kept = select_confident(PREDS, threshold=0.0)
        self.assertEqual([k["confidence"] for k in kept], [0.99, 0.96, 0.80, 0.50])

    def test_low_threshold_keeps_all(self):
        self.assertEqual(len(select_confident(PREDS, threshold=0.0)), 4)

    def test_high_threshold_keeps_none(self):
        self.assertEqual(select_confident(PREDS, threshold=1.01), [])

    def test_default_threshold(self):
        self.assertEqual(DEFAULT_THRESHOLD, 0.95)

    def test_bool_confidence_rejected(self):
        # A stray True must not sneak in as confidence 1.0.
        with self.assertRaises(ValueError):
            select_confident([{"id": "x", "label": 0, "confidence": True}], threshold=0.5)


class TestMergeRound(unittest.TestCase):
    def setUp(self):
        self.labeled = [
            {"id": "L0", "label": 2, "path": "labeled/L0.png"},
            {"id": "L1", "label": 4, "path": "labeled/L1.png"},
        ]
        self.pool = [
            {"id": "p0", "path": "pool/p0.png"},
            {"id": "p1", "path": "pool/p1.png"},
            {"id": "p2", "path": "pool/p2.png"},
        ]
        self.batch = [
            {"id": "p0", "label": 3, "confidence": 0.99},
            {"id": "p2", "label": 1, "confidence": 0.96},
        ]

    def test_labeled_grows_by_batch(self):
        new_labeled, _ = merge_round(self.labeled, self.pool, self.batch)
        self.assertEqual(len(new_labeled), len(self.labeled) + len(self.batch))
        promoted = {r["id"]: r for r in new_labeled if r["id"] in {"p0", "p2"}}
        self.assertEqual(promoted["p0"]["label"], 3)              # pseudo-label stamped
        self.assertEqual(promoted["p0"]["path"], "pool/p0.png")   # pool metadata carried over

    def test_pool_shrinks_by_exactly_the_batch_ids(self):
        _, new_pool = merge_round(self.labeled, self.pool, self.batch)
        self.assertEqual([r["id"] for r in new_pool], ["p1"])     # only the unselected remains

    def test_round_trips_sizes(self):
        new_labeled, new_pool = merge_round(self.labeled, self.pool, self.batch)
        # No example is lost or duplicated across the labelled/pool partition.
        self.assertEqual(len(new_labeled) + len(new_pool), len(self.labeled) + len(self.pool))

    def test_no_duplicates_when_id_already_labeled(self):
        # Idempotent: re-promoting an already-labelled id is a no-op (no dupe).
        labeled = self.labeled + [{"id": "p0", "label": 3, "path": "pool/p0.png"}]
        batch = [{"id": "p0", "label": 3, "confidence": 0.99}]
        new_labeled, new_pool = merge_round(labeled, self.pool, batch)
        ids = [r["id"] for r in new_labeled]
        self.assertEqual(ids.count("p0"), 1)

    def test_no_duplicates_within_batch(self):
        batch = [
            {"id": "p0", "label": 3, "confidence": 0.99},
            {"id": "p0", "label": 3, "confidence": 0.97},
        ]
        new_labeled, _ = merge_round(self.labeled, self.pool, batch)
        self.assertEqual([r["id"] for r in new_labeled].count("p0"), 1)

    def test_empty_batch_is_identity(self):
        new_labeled, new_pool = merge_round(self.labeled, self.pool, [])
        self.assertEqual(new_labeled, self.labeled)
        self.assertEqual([r["id"] for r in new_pool], ["p0", "p1", "p2"])

    def test_batch_id_not_in_pool_raises(self):
        with self.assertRaises(ValueError):
            merge_round(self.labeled, self.pool, [{"id": "ghost", "label": 0, "confidence": 0.99}])


class TestSelectConfidentMain(unittest.TestCase):
    """The Run Code entry: {preds, threshold} -> {batch, new_confident}."""

    def test_filters_and_counts(self):
        out = select_confident_main({"preds": PREDS, "threshold": 0.95})
        self.assertEqual([b["id"] for b in out["batch"]], ["p0", "p2"])
        self.assertEqual(out["new_confident"], 2)

    def test_threshold_defaults(self):
        out = select_confident_main({"preds": PREDS})  # default 0.95
        self.assertEqual(out["new_confident"], 2)

    def test_string_threshold_coerced(self):
        # Config·JSON / loop vars deliver scalars that may be strings.
        out = select_confident_main({"preds": PREDS, "threshold": "0.80"})
        self.assertEqual(out["new_confident"], 3)


class TestMergeMain(unittest.TestCase):
    """The Run Code entry / round summary: carries model + metrics forward as the loop sink."""

    def test_round_summary_updates_and_passes_through(self):
        out = merge_main({
            "labeled": [{"id": "L0", "label": 2, "path": "labeled/L0.png"}],
            "pool": [{"id": "p0", "path": "pool/p0.png"}, {"id": "p1", "path": "pool/p1.png"}],
            "batch": [{"id": "p0", "label": 3, "confidence": 0.99}],
            "model": {"bucket": "b", "key": "round1/model.tar.gz"},
            "metrics": {"accuracy": 0.62},
        })
        self.assertEqual([r["id"] for r in out["labeled"]], ["L0", "p0"])   # grew
        self.assertEqual([r["id"] for r in out["pool"]], ["p1"])           # shrank
        self.assertEqual(out["model"]["key"], "round1/model.tar.gz")       # passthrough (loop carries it forward)
        self.assertEqual(out["metrics"]["accuracy"], 0.62)                 # passthrough (the stop signal reads this)
        self.assertEqual(out["new_confident"], 1)

    def test_empty_batch_plateau(self):
        out = merge_main({
            "labeled": [{"id": "L0", "label": 2, "path": "labeled/L0.png"}],
            "pool": [{"id": "p0", "path": "pool/p0.png"}],
            "batch": [],
            "model": {"key": "m"},
            "metrics": {"accuracy": 0.7},
        })
        self.assertEqual(out["new_confident"], 0)
        self.assertEqual([r["id"] for r in out["pool"]], ["p0"])  # nothing consumed → plateau


if __name__ == "__main__":
    unittest.main()
