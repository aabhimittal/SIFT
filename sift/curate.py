"""Fuse the signals into one ranking: which trajectories to keep first.

Each trajectory gets exactly one verdict, applied in this order:

``mismatch``   the judge says its instruction does not describe its motion
``noisy``      no instruction explains its actions well (out-of-fold loss is a
               robust outlier): sloppy or failed motion
``harmful``    confidently negative influence: even mean + 1 std across seeds < 0
``redundant``  a near-duplicate of a trajectory that is kept
``keep``       everything else

The final order is keep, then redundant, then harmful, noisy, mismatch. Within
``keep`` the sort key is a lower confidence bound on influence,
``mean - k * std`` across seeds, so a trajectory has to help consistently to
rank high. Taking a prefix of this order gives the curated subset at any budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .duplicates import DuplicateResult
from .influence import InfluenceResult
from .judge import JudgeResult

VERDICTS = ("keep", "redundant", "harmful", "noisy", "mismatch")


def robust_outliers(x: np.ndarray, k: float) -> np.ndarray:
    """Flag values above median + k robust-sd. NaNs are never flagged.

    When more than half the values are identical the MAD is 0, and a naive
    threshold flags any value a hair above the median. Fall back to the mean
    absolute deviation (scaled to sd), and flag nothing if that is 0 too.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), bool)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return out
    v = x[ok]
    med = np.median(v)
    spread = np.median(np.abs(v - med)) * 1.4826
    if spread == 0:
        spread = np.mean(np.abs(v - med)) * 1.2533
    if spread == 0:
        return out
    out[ok] = v > med + k * spread
    return out


@dataclass
class Curation:
    verdict: np.ndarray       # (N,) str
    order: np.ndarray         # (N,) indices, best first
    score: np.ndarray         # (N,) the within-tier sort key (influence LCB)
    representative: np.ndarray  # (N,) for redundant items, the index of the kept twin; else -1

    def subset(self, frac: float) -> np.ndarray:
        """Top-``frac`` prefix of the ranking (at least one trajectory)."""
        if not 0 < frac <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {frac}")
        return self.order[: max(1, int(round(frac * len(self.order))))]

    def counts(self) -> dict[str, int]:
        return {v: int((self.verdict == v).sum()) for v in VERDICTS}


def curate(infl: InfluenceResult, dups: DuplicateResult, judge: JudgeResult,
           k_std: float = 1.0, noisy_k: float = 4.0) -> Curation:
    n = len(infl.mean)
    if not (len(dups.cluster) == len(judge.mismatch) == n):
        raise ValueError("influence, duplicate and judge results cover different numbers of trajectories")
    lcb = infl.mean - k_std * infl.std
    ucb = infl.mean + k_std * infl.std
    verdict = np.array(["keep"] * n, dtype=object)
    verdict[ucb < 0] = "harmful"
    verdict[robust_outliers(judge.oof_loss, noisy_k)] = "noisy"
    verdict[judge.mismatch] = "mismatch"

    # Within each duplicate cluster keep the member with the best LCB among
    # those still eligible; the rest become redundant.
    rep = np.full(n, -1)
    for members in dups.clusters().values():
        eligible = [i for i in members if verdict[i] == "keep"]
        if len(eligible) < 2:
            continue
        best = max(eligible, key=lambda i: lcb[i])
        for i in eligible:
            if i != best:
                verdict[i] = "redundant"
                rep[i] = best

    tier = np.array([VERDICTS.index(v) for v in verdict])
    order = np.lexsort((-lcb, tier))
    return Curation(verdict.astype(str), order, lcb, rep)


def tracin_only_order(infl: InfluenceResult) -> np.ndarray:
    """Ablation: rank by mean influence alone, no dedup, no judge."""
    return np.argsort(-infl.mean, kind="stable")


def filters_only_order(cur: Curation, seed: int) -> np.ndarray:
    """Ablation: same verdict tiers as SIFT, random order inside each tier.

    If this matches SIFT, the influence ranking adds nothing beyond the
    filters, and you can skip the expensive part. ``harmful`` comes from
    influence, so it is folded back into ``keep`` here.
    """
    rng = np.random.default_rng(seed)
    tier = np.array([VERDICTS.index("keep" if v == "harmful" else v) for v in cur.verdict])
    return np.lexsort((rng.random(len(tier)), tier))
