"""Score SIFT's detectors against the synthetic ground truth.

Only possible because the corruptions were injected on purpose. On real data
there is no answer key; this module is how you earn trust in the detectors
before pointing them at data where you cannot check them.
"""

from __future__ import annotations

import numpy as np

from .data import Dataset
from .duplicates import DuplicateResult


def prf(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = float((pred & truth).sum())
    p = tp / max(pred.sum(), 1)
    r = tp / max(truth.sum(), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "flagged": int(pred.sum()), "actual": int(truth.sum())}


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean()
    eq = (pos[:, None] == neg[None]).mean()
    return float(gt + 0.5 * eq)


def duplicate_pair_prf(ds: Dataset, res: DuplicateResult) -> dict[str, float]:
    tc = np.array([t.true_cluster for t in ds.trajs])
    iu = np.triu_indices(len(ds), 1)
    same_pred = (res.cluster[:, None] == res.cluster[None]) & (res.cluster[:, None] >= 0)
    same_true = (tc[:, None] == tc[None]) & (tc[:, None] >= 0)
    return prf(same_pred[iu], same_true[iu])
