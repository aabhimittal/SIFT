"""Near-duplicate detection in action-trajectory space.

Why not image embeddings? Two demos filmed on the same table layout look
identical to a vision encoder even when the robot does something different in
each (reaches a different object). That is a *distinct* demo, and deleting it
removes the very contrast that teaches language grounding. Conversely, a
redundant demo is one where the *behaviour* repeats: same proprioceptive path,
same commands. So SIFT embeds each trajectory as its effector path plus its
action sequence, both resampled to a fixed length over normalised time.

Two stages, so the quadratic part stays cheap:

1. Blocking: Euclidean distance between resampled embeddings for all pairs
   (one matrix product). Keep pairs under a loose threshold as candidates.
2. Verification: dynamic time warping on the raw sequences for candidates
   only, which tolerates the same motion executed at a slightly different pace.

Pairs that pass are joined with union-find into clusters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset


def resample(seq: np.ndarray, length: int) -> np.ndarray:
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, length)
    return np.stack([np.interp(t_new, t_old, seq[:, d]) for d in range(seq.shape[1])], axis=1)


def behaviour_embedding(ds: Dataset, length: int = 32) -> np.ndarray:
    """(N, length*4): effector path and normalised actions, resampled."""
    v = ds.cfg.v_max
    feats = [np.concatenate([resample(t.pos[:-1], length), resample(t.actions / v, length) * 0.1], 1).ravel()
             for t in ds.trajs]
    return np.array(feats) / np.sqrt(length)


def scene_embedding(ds: Dataset) -> np.ndarray:
    """Baseline: what a first-frame image embedding sees (layout + start pose)."""
    return np.array([np.concatenate([t.pos[0], t.targets.ravel()]) for t in ds.trajs])


def pairwise_dist(F: np.ndarray) -> np.ndarray:
    sq = (F ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * F @ F.T, 0))


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """DTW with Euclidean step cost, normalised by max length so a perfect
    diagonal alignment returns the mean per-step distance (same scale as the
    blocking embedding's RMS per-step distance)."""
    n, m = len(a), len(b)
    C = np.linalg.norm(a[:, None] - b[None], axis=-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        row = D[i - 1, :-1].copy()          # diagonal predecessors
        up = D[i - 1, 1:]
        best = np.minimum(row, up) + C[i - 1]
        # left dependency is sequential; resolve with a running minimum
        for j in range(1, m + 1):
            D[i, j] = min(best[j - 1], D[i, j - 1] + C[i - 1, j - 1])
    return float(D[n, m] / max(n, m))


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


@dataclass
class DuplicateResult:
    cluster: np.ndarray          # (N,) cluster id, -1 for singletons
    pairs: list[tuple[int, int, float]]
    threshold: float
    nn_dist: np.ndarray          # (N,) nearest-neighbour embedding distance

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i, c in enumerate(self.cluster):
            if c >= 0:
                out.setdefault(int(c), []).append(i)
        return out


def _cluster_from_pairs(n: int, pairs) -> np.ndarray:
    uf = UnionFind(n)
    for i, j, _ in pairs:
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    sizes = np.bincount(roots, minlength=n)
    cluster = np.full(n, -1)
    ids = {r: k for k, r in enumerate(sorted({r for r in roots if sizes[r] > 1}))}
    for i, r in enumerate(roots):
        if sizes[r] > 1:
            cluster[i] = ids[r]
    return cluster


def auto_threshold(nn: np.ndarray, ratio: float = 0.25) -> float:
    """Duplicates sit far below the typical nearest-neighbour gap between independent demos."""
    return float(ratio * np.median(nn))


def find_duplicates(ds: Dataset, threshold: float | None = None, block_factor: float = 2.0,
                    length: int = 32, use_dtw: bool = True) -> DuplicateResult:
    F = behaviour_embedding(ds, length)
    D = pairwise_dist(F)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    thr = auto_threshold(nn) if threshold is None else threshold
    v = ds.cfg.v_max
    pairs = []
    for i, j in zip(*np.where(np.triu(D < block_factor * thr, 1))):
        if use_dtw:
            a = np.concatenate([ds[i].pos[:-1], ds[i].actions / v * 0.1], 1)
            b = np.concatenate([ds[j].pos[:-1], ds[j].actions / v * 0.1], 1)
            d = dtw(a, b)
        else:
            d = D[i, j]
        if d < thr:
            pairs.append((int(i), int(j), float(d)))
    return DuplicateResult(_cluster_from_pairs(len(ds), pairs), pairs, thr, nn)


def find_scene_duplicates(ds: Dataset, threshold: float | None = None) -> DuplicateResult:
    """The image-space baseline SIFT argues against, for comparison."""
    F = scene_embedding(ds)
    D = pairwise_dist(F)
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    thr = auto_threshold(nn) if threshold is None else threshold
    pairs = [(int(i), int(j), float(D[i, j])) for i, j in zip(*np.where(np.triu(D < thr, 1)))]
    return DuplicateResult(_cluster_from_pairs(len(ds), pairs), pairs, thr, nn)
