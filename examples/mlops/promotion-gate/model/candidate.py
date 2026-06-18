"""The 'candidate' the promotion gate evaluates — a small but REAL model (no faking).

Trains a real scikit-learn classifier on the real `digits` dataset and returns its real held-out
accuracy. This stands in for a Train Model node's output: a metrics object + a model package
reference. The promotion gate is model-agnostic — swap this for any real training run (e.g. the
YOLO fine-tune in ../yolo-finetune, whose winning mAP becomes the candidate metric).
"""
from __future__ import annotations

from typing import Any


def train_and_eval(seed: int = 0) -> dict[str, Any]:
    """Train + evaluate a real LogisticRegression on digits; return {metrics, modelPackageArn}."""
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
    accuracy = float(accuracy_score(y_te, clf.predict(X_te)))
    return {
        "metrics": {"accuracy": round(accuracy, 4)},
        "modelPackageArn": f"arn:aws:sagemaker:us-east-1:000000000000:model-package/digits-classifier/{seed + 1}",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(train_and_eval(), indent=2))
