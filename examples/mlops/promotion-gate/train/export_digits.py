"""Export the scikit-learn ``digits`` dataset to a CSV for upload to S3.

The promotion-gate **candidate** trains on this CSV pulled from the Train Model node's ``dataset``
channel — a REAL S3 input — not sklearn's in-process ``load_digits()``. Run this once, then upload
the CSV to the bucket the Train Model node's ``dataset`` Config·JSON points at:

    python export_digits.py                                        # writes digits.csv next to this file
    aws s3 cp digits.csv s3://<your-bucket>/promotion-gate/digits/

Layout: 64 pixel columns (``pixel_0``..``pixel_63``, integers 0-16) + a final ``label`` column
(the digit 0-9); one header row, then 1797 sample rows. ``finetune.train()`` reads it back with
:func:`finetune._load_digits_csv`, so the export and the training read stay in lock-step.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV_NAME = "digits.csv"


def export(dest: Path) -> int:
    """Write the digits dataset to ``dest`` as CSV; return the row count."""
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    header = [f"pixel_{i}" for i in range(X.shape[1])] + ["label"]
    with dest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row, label in zip(X, y):
            writer.writerow([*(int(v) for v in row), int(label)])
    return len(y)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / CSV_NAME
    n = export(out)
    print(f"wrote {n} rows to {out}")
