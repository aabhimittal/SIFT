"""Turn a SIFT ranking into a curated dataset file."""

from __future__ import annotations

import numpy as np

from .curate import VERDICTS
from .data import Dataset


def select(results: dict, frac: float | None = None, verdicts=None) -> np.ndarray:
    """Indices to keep, best first.

    ``frac``      take the top fraction of the ranking (any verdict)
    ``verdicts``  keep every trajectory whose verdict is listed, in rank order
    Exactly one of the two must be given.
    """
    if (frac is None) == (verdicts is None):
        raise ValueError("give exactly one of frac or verdicts")
    trajs = sorted(results["trajectories"], key=lambda t: t["rank"])
    if frac is not None:
        if not 0 < frac <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {frac}")
        k = max(1, int(round(frac * len(trajs))))
        return np.array([t["id"] for t in trajs[:k]], dtype=int)
    unknown = set(verdicts) - set(VERDICTS)
    if unknown:
        raise ValueError(f"unknown verdicts {sorted(unknown)}; expected some of {VERDICTS}")
    return np.array([t["id"] for t in trajs if t["verdict"] in verdicts], dtype=int)


def export_subset(ds: Dataset, results: dict, frac: float | None = None, verdicts=None) -> Dataset:
    if len(ds) != results["meta"]["n"]:
        raise ValueError(f"dataset has {len(ds)} trajectories but the results describe {results['meta']['n']}; "
                         "pass the same dataset the results were computed on")
    expected = results["meta"].get("fingerprint")
    if expected is not None and expected != ds.fingerprint():
        raise ValueError("dataset contents differ from the dataset the results were computed on "
                         f"(fingerprint {ds.fingerprint()} != {expected})")
    idx = select(results, frac, verdicts)
    if len(idx) == 0:
        raise ValueError("selection is empty")
    return ds.subset(idx)
