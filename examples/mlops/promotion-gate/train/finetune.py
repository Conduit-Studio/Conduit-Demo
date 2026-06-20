"""Build-from-code training function for the promotion-gate **candidate** (REAL sklearn).

This is the candidate the model-promotion gate evaluates. It is the WHOLE training contract
under Conduit's **build-from-code** Train Model node: you author `train(hyperparameters, channels)`
and a `requirements.txt`, and Conduit builds + digest-pins the SageMaker training image for you
(curated base + your requirements + this code + Conduit's wrapper). There is **no Dockerfile and
no `/opt/ml` plumbing** — Conduit's wrapper:

  * hands `train()` the job's hyperparameters and the local channel paths;
  * copies the model directory you return into `SM_MODEL_DIR` (`/opt/ml/model` → `model.tar.gz`,
    which becomes the Train Model node's `model` output);
  * scrapes the `metrics` dict you return → the node's `metrics` output → the eval gate → lineage.

The candidate is a **real** model: a scikit-learn ``LogisticRegression`` on the built-in ``digits``
dataset, scored on a 20% held-out split. The reported accuracy is a genuine held-out metric, not a
hardcoded number — and ``digits`` is a built-in sklearn dataset, so this needs **no channel / no
S3 input** (the Train Model node declares only a ``hyperparameters`` json port).

The exact same ``train()`` runs:
  * locally, called by ``_main`` (real fit + held-out score, no faking);
  * in your account, inside the image Conduit builds for the Train Model node.

SageMaker delivers every hyperparameter as a STRING, so we coerce.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _coerce(value: Any, cast, default):
    """Best-effort cast (SageMaker hyperparameters all arrive as strings)."""
    if value is None:
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _model_dir() -> Path:
    """Resolve a writable directory to save the fitted model into.

    Uses the SageMaker convention ``SM_MODEL_DIR`` (``/opt/ml/model`` in the training image) with a
    local fallback so the same code runs locally. We write into a ``candidate/`` subdirectory so the
    path we return is never identical to the wrapper's ``MODEL_DIR`` (the wrapper copies the
    *contents* of the returned dir into ``SM_MODEL_DIR``; returning that dir itself would self-copy).
    """
    base = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        # Probe writability; fall back to a local dir when /opt/ml is not mountable (local runs).
        probe = base / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError:
        base = Path(os.environ.get("CONDUIT_MODEL_DIR", "runs/model"))
    model_dir = (base / "candidate").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def train(hyperparameters: dict[str, Any], channels: dict[str, str]) -> dict[str, Any]:
    """Train a real LogisticRegression on sklearn ``digits``; report REAL held-out accuracy.

    Args:
        hyperparameters: optional keys ``seed`` (held-out split seed, default 0), ``max_iter``
            (solver iterations, default 5000), ``C`` (inverse regularisation, default 1.0). Values
            may be strings (SageMaker hands every HP over as a string).
        channels: maps channel name → local path. Unused here — ``digits`` is a built-in sklearn
            dataset, so the candidate needs no S3 channel (the Train Model node declares only the
            fixed ``hyperparameters`` json port).

    Returns:
        ``{"model": <model dir>, "metrics": {"accuracy": float}}`` — ``model`` is the directory
        Conduit copies into ``SM_MODEL_DIR`` (it holds ``model.joblib``); ``metrics.accuracy`` is the
        REAL held-out accuracy the promotion gate's eval node compares against the baseline.
    """
    seed = _coerce(hyperparameters.get("seed", 0), int, 0)
    max_iter = _coerce(hyperparameters.get("max_iter", 5000), int, 5000)
    C = _coerce(hyperparameters.get("C", 1.0), float, 1.0)

    model_dir = _model_dir()
    print(f"[train] seed={seed} max_iter={max_iter} C={C} model_dir={model_dir}", flush=True)

    # Heavy imports kept inside the function so importing this module (tests, the wrapper
    # discovering `train`) stays cheap.
    import joblib
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    print(f"[train] {len(X_tr)} train / {len(X_te)} held-out", flush=True)

    clf = LogisticRegression(max_iter=max_iter, C=C).fit(X_tr, y_tr)
    accuracy = float(accuracy_score(y_te, clf.predict(X_te)))

    model_path = model_dir / "model.joblib"
    joblib.dump(clf, model_path)

    print(f"accuracy: {accuracy:.5f}", flush=True)
    print(f"[train] model saved to {model_path}", flush=True)

    return {"model": str(model_dir), "metrics": {"accuracy": accuracy}}


def _main() -> int:
    """Train the candidate locally for a quick smoke (real fit, real held-out score)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train the promotion-gate candidate locally.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000, dest="max_iter")
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    out = train({"seed": args.seed, "max_iter": args.max_iter, "C": args.C}, {})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
