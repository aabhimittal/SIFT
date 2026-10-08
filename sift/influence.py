"""TracIn-style trajectory influence for behaviour cloning.

For each training trajectory z_i and saved checkpoint theta_t::

    influence(z_i) = sum_t  eta_t * < grad L(z_i; theta_t),  P_t grad L(D_val; theta_t) >

``L(z_i)`` is the mean BC loss over the trajectory's steps, so a trajectory is
the unit of credit (what a curator keeps or drops), not a single frame.
``P_t`` is identity for vanilla TracIn, or Adam's diagonal preconditioner
``1/sqrt(v_t)`` when ``precondition=True``; since the optimiser actually steps
along ``P_t g``, the preconditioned dot product is the closer first-order
estimate of how one step on z_i moves validation loss.

Positive influence: a step on z_i lowered validation loss (a proponent).
Negative: it raised it (an opponent: conflicting labels, bad motion).

Two choices of validation gradient:

``heldout``  BC loss on a small trusted set of clean demos. Cheap, needs no sim,
             but measures imitation of the expert on the *expert's* state
             distribution.
``rollout``  Roll the checkpoint policy out in the sim from fixed validation
             scenes, relabel every visited state with the expert action, and
             take the BC gradient there (one DAgger step). This measures what
             the policy should have done in the states it actually reaches,
             which is what closed-loop success depends on. It costs a sim
             rollout per checkpoint and needs a relabelling oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .env import COLORS, ReachEnv, clip_norm
from .model import Checkpoint, MLPPolicy, train_bc


def expert_relabel(obs: np.ndarray, v_max: float, gain: float = 0.5) -> np.ndarray:
    """Expert action (normalised) at arbitrary observations; reads the goal offset from obs."""
    k = obs[..., 2 + 2 * len(COLORS):].argmax(-1)
    rel = obs[..., 2:2 + 2 * len(COLORS)].reshape(*obs.shape[:-1], len(COLORS), 2)
    goal_rel = np.take_along_axis(rel, k[..., None, None].repeat(2, -1), axis=-2)[..., 0, :]
    return clip_norm(gain * goal_rel, v_max) / v_max


def trajectory_grads(model: MLPPolicy, ds: Dataset, theta: np.ndarray, idx=None) -> np.ndarray:
    """(N, P) matrix of per-trajectory mean-loss gradients at theta."""
    idx = range(len(ds)) if idx is None else idx
    v = ds.cfg.v_max
    return np.stack([model.loss_grad(ds[i].obs, ds[i].actions / v, theta)[1] for i in idx])


def validation_grad(model: MLPPolicy, theta: np.ndarray, mode: str, val: Dataset, env: ReachEnv,
                    n_rollouts: int = 64, seed: int = 4242) -> np.ndarray:
    v = env.cfg.v_max
    if mode == "heldout":
        X, Y, _ = val.arrays()
        return model.loss_grad(X, Y, theta)[1]
    if mode == "rollout":
        eps = env.sample_episodes(n_rollouts, seed)
        rec = env.rollout(lambda o: model.predict(o, theta), eps, record=True)
        X = rec["obs"].reshape(-1, rec["obs"].shape[-1])
        return model.loss_grad(X, expert_relabel(X, v), theta)[1]
    raise ValueError(f"unknown validation mode {mode!r}")


def tracin(model: MLPPolicy, ckpts: list[Checkpoint], ds: Dataset, val: Dataset, env: ReachEnv,
           mode: str = "rollout", precondition: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (influence, self_influence), each shape (N,)."""
    infl = np.zeros(len(ds))
    self_infl = np.zeros(len(ds))
    for ck in ckpts:
        G = trajectory_grads(model, ds, ck.theta)
        gv = validation_grad(model, ck.theta, mode, val, env)
        P = ck.adam_v if precondition else 1.0
        infl += ck.lr * G @ (P * gv)
        self_infl += ck.lr * np.einsum("np,np->n", G * P, G)
    return infl, self_infl


def rankdata(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


@dataclass
class InfluenceResult:
    per_seed: np.ndarray        # (S, N)
    self_per_seed: np.ndarray   # (S, N)
    mode: str
    precondition: bool

    @property
    def mean(self) -> np.ndarray:
        return self.per_seed.mean(0)

    @property
    def std(self) -> np.ndarray:
        return self.per_seed.std(0, ddof=1) if len(self.per_seed) > 1 else np.zeros(self.per_seed.shape[1])

    @property
    def self_mean(self) -> np.ndarray:
        return self.self_per_seed.mean(0)

    def seed_spearman(self) -> np.ndarray:
        S = len(self.per_seed)
        M = np.eye(S)
        for a in range(S):
            for b in range(a + 1, S):
                M[a, b] = M[b, a] = spearman(self.per_seed[a], self.per_seed[b])
        return M

    def topk_overlap(self, frac: float = 0.3) -> float:
        """Mean pairwise Jaccard overlap of the top-``frac`` sets across seeds."""
        k = max(1, int(frac * self.per_seed.shape[1]))
        tops = [set(np.argsort(-s)[:k]) for s in self.per_seed]
        vals = [len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                for a in range(len(tops)) for b in range(a + 1, len(tops))]
        return float(np.mean(vals)) if vals else 1.0


def run_influence(ds: Dataset, val: Dataset, env: ReachEnv, seeds=(0, 1, 2), steps: int = 2500,
                  n_checkpoints: int = 10, mode: str = "rollout", precondition: bool = True,
                  log=print) -> InfluenceResult:
    """Train one policy per seed on the full dataset and score every trajectory against it.

    Seed variance is the honest error bar: TracIn in BC is noisy, and a
    ranking that flips between seeds should not drive deletion decisions.
    """
    X, Y, _ = ds.arrays()
    per, selfs = [], []
    for s in seeds:
        model, ckpts = train_bc(X, Y, steps=steps, seed=s, n_checkpoints=n_checkpoints)
        infl, si = tracin(model, ckpts, ds, val, env, mode=mode, precondition=precondition)
        per.append(infl)
        selfs.append(si)
        log(f"  influence seed {s}: policy success {env.success_rate(model):.2f}")
    return InfluenceResult(np.stack(per), np.stack(selfs), mode, precondition)
