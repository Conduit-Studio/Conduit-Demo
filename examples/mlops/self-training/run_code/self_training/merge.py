"""Pure round-merge bookkeeping for the self-training loop — no torch, fully unit-testable.

After `select_confident` picks this round's high-confidence pseudo-labels (`batch`),
`merge_round` folds them back into the dataset for the NEXT round:

  * labelled set GROWS by the batch (each pooled example now carries its pseudo-label);
  * unlabeled pool SHRINKS by exactly the batch's ids (those examples are no longer pooled).

The labelled and pool sets are simple manifest rows — `{id, label, path}` — matching the
on-disk index format (see ../../README.md). A row's `id` is its identity; merge is keyed
on `id` so it's idempotent and never duplicates. This is the deterministic heart of the
loop body that the `flow.loop` re-feeds into the next iteration's `train`.
"""
from __future__ import annotations

from typing import Any


def _id_of(row: dict[str, Any]) -> str:
    if "id" not in row:
        raise ValueError(f"dataset row missing required 'id': {row!r}")
    return str(row["id"])


def merge_round(
    labeled: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    batch: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grow labelled by the batch; shrink pool by exactly the batch's ids.

    Args:
        labeled: current labelled rows ({id, label, path, ...}).
        pool: current unlabeled pool rows ({id, path, ...}; label absent/None).
        batch: this round's accepted pseudo-labels ({id, label, confidence}).

    Returns:
        (labeled', pool'):
          * labeled' = labeled + the batch examples carrying their pseudo-label,
            taking each pooled row's metadata (e.g. path) and stamping the label.
            Ids already in labelled are NOT duplicated (idempotent).
          * pool' = pool minus every id in the batch.

    The batch ids must be a subset of the pool; an id not in the pool raises (the
    loop only ever pseudo-labels pooled examples, so this catches a wiring bug).
    """
    pool_by_id = {_id_of(row): row for row in pool}
    labeled_ids = {_id_of(row) for row in labeled}

    promoted: list[dict[str, Any]] = []
    promoted_ids: set[str] = set()
    for pred in batch:
        pid = _id_of(pred)
        if pid not in pool_by_id:
            raise ValueError(f"batch id {pid!r} is not in the unlabeled pool")
        if pid in labeled_ids or pid in promoted_ids:
            continue  # idempotent: never duplicate an already-labelled example
        source = pool_by_id[pid]
        promoted_row = {**source, "id": pid, "label": int(pred["label"])}
        promoted.append(promoted_row)
        promoted_ids.add(pid)

    new_labeled = list(labeled) + promoted
    new_pool = [row for row in pool if _id_of(row) not in promoted_ids]
    return new_labeled, new_pool
