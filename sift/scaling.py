"""Data-scaling curves: closed-loop success vs. fraction of data kept.

Every subset trains for the same number of gradient steps. That isolates data
quality from compute: a curated 30% subset is not allowed to win by being
trained longer per example, nor to lose by being trained less in total.
Evaluation episodes use a seed range disjoint from the validation rollouts
that influence was computed on, so the curve is not graded on its own exam.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .data import Dataset
from .env import ReachEnv
from .model import train_bc

EVAL_SEED = 100_000


def train_and_eval(ds: Dataset, idx, env: ReachEnv, seed: int, steps: int, n_eval: int) -> float:
    X, Y, _ = ds.arrays(idx)
    pol, _ = train_bc(X, Y, steps=steps, seed=1000 + seed)
    return env.success_rate(pol, n=n_eval, seed=EVAL_SEED + seed)


def scaling_curve(ds: Dataset, env: ReachEnv, orders: dict[str, Callable[[int], np.ndarray]],
                  fractions=(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0), seeds=(0, 1, 2), steps: int = 2500,
                  n_eval: int = 300, log=print) -> dict[str, np.ndarray]:
    """``orders[name](seed)`` returns a full ranking; the top-f prefix is trained on.

    Returns name -> success array of shape (len(fractions), len(seeds)).
    """
    out = {}
    for name, order_fn in orders.items():
        res = np.zeros((len(fractions), len(seeds)))
        for si, s in enumerate(seeds):
            order = order_fn(s)
            for fi, f in enumerate(fractions):
                k = max(1, int(round(f * len(order))))
                res[fi, si] = train_and_eval(ds, order[:k], env, s, steps, n_eval)
        out[name] = res
        log(f"  {name:>10}: " + "  ".join(f"{f:.0%}={m:.2f}" for f, m in zip(fractions, res.mean(1))))
    return out


def fixed_subset_success(ds: Dataset, idx, env: ReachEnv, seeds=(0, 1, 2), steps: int = 2500,
                         n_eval: int = 300) -> np.ndarray:
    return np.array([train_and_eval(ds, idx, env, s, steps, n_eval) for s in seeds])


def matching_fraction(fractions, curve: np.ndarray, baseline: np.ndarray) -> float | None:
    """Smallest fraction whose mean success is at least the full-data mean.

    Deliberately strict: no tolerance band. If curated data never matches,
    report None rather than shopping for a threshold that makes it match.
    """
    target = baseline.mean()
    for f, row in zip(fractions, curve):
        if row.mean() >= target:
            return float(f)
    return None
