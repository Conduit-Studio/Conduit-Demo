"""Shared logic for the self-training loop example.

Imported by the Conduit Run Code entrypoints (../select_confident.py, ../merge.py)
and by the tests. Pure Python, NO torch — so the selection + merge logic is
unit-testable off a laptop with no GPU. The REAL inference lives in ../pseudo_label.py
and the REAL training in ../../train/finetune.py; only the deterministic
keep/merge bookkeeping lives here.
"""
