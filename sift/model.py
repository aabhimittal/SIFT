"""A NumPy behaviour-cloning policy with explicit gradients.

Influence methods need per-example gradients, and a real VLA hides those
behind an autodiff framework. Here the MLP is small and hand-differentiated so
every quantity SIFT uses is a plain vector you can inspect. To use SIFT with a
real policy, implement the same three methods (``flat``, ``loss_grad``,
``predict``) on top of your framework; nothing downstream depends on NumPy MLP
internals.

Loss for a batch is ``mean_s 0.5 * ||f(x_s) - y_s||^2``. All parameters live
in one flat vector ``theta`` with views per layer, which makes checkpointing,
Adam, and gradient dot-products one-liners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import ACT_DIM, OBS_DIM


class MLPPolicy:
    def __init__(self, in_dim: int = OBS_DIM, hidden: int = 64, out_dim: int = ACT_DIM, seed: int = 0):
        self.shapes = [(in_dim, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, out_dim), (out_dim,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.theta = np.zeros(sum(self.sizes))
        rng = np.random.default_rng(seed)
        W1, _, W2, _, W3, _ = self._views(self.theta)
        W1[:] = rng.normal(0, 1 / np.sqrt(in_dim), W1.shape)
        W2[:] = rng.normal(0, 1 / np.sqrt(hidden), W2.shape)
        W3[:] = rng.normal(0, 0.1 / np.sqrt(hidden), W3.shape)

    @property
    def n_params(self) -> int:
        return self.theta.size

    def _views(self, vec: np.ndarray) -> list[np.ndarray]:
        out, o = [], 0
        for shp, sz in zip(self.shapes, self.sizes):
            out.append(vec[o:o + sz].reshape(shp))
            o += sz
        return out

    def flat(self) -> np.ndarray:
        return self.theta.copy()

    def load(self, theta: np.ndarray) -> "MLPPolicy":
        self.theta[:] = theta
        return self

    def _forward(self, X: np.ndarray, theta: np.ndarray | None = None):
        W1, b1, W2, b2, W3, b3 = self._views(self.theta if theta is None else theta)
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h1, h2, h2 @ W3 + b3

    def predict(self, X: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        return self._forward(X, theta)[2]

    __call__ = predict

    def loss(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray | None = None) -> float:
        return float(0.5 * np.mean(np.sum((self.predict(X, theta) - Y) ** 2, axis=-1)))

    def loss_grad(self, X: np.ndarray, Y: np.ndarray, theta: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        """Mean loss over the rows of X and its gradient w.r.t. theta (flat)."""
        th = self.theta if theta is None else theta
        W1, b1, W2, b2, W3, b3 = self._views(th)
        h1, h2, out = self._forward(X, th)
        n = len(X)
        err = out - Y
        loss = float(0.5 * np.mean(np.sum(err ** 2, axis=-1)))
        r = err / n                            # dL/dout
        g = np.empty_like(th)
        gW1, gb1, gW2, gb2, gW3, gb3 = self._views(g)
        gW3[:] = h2.T @ r
        gb3[:] = r.sum(0)
        d2 = (r @ W3.T) * (1 - h2 ** 2)
        gW2[:] = h1.T @ d2
        gb2[:] = d2.sum(0)
        d1 = (d2 @ W2.T) * (1 - h1 ** 2)
        gW1[:] = X.T @ d1
        gb1[:] = d1.sum(0)
        return loss, g


@dataclass
class Checkpoint:
    step: int
    theta: np.ndarray
    lr: float
    adam_v: np.ndarray  # Adam second moment, for preconditioned influence


class Adam:
    def __init__(self, n: int, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0

    def step(self, theta: np.ndarray, g: np.ndarray) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        theta -= self.lr * mh / (np.sqrt(vh) + self.eps)

    def preconditioner(self) -> np.ndarray:
        vh = self.v / (1 - self.b2 ** max(self.t, 1))
        return 1.0 / (np.sqrt(vh) + 1e-6)


def train_bc(
    X: np.ndarray,
    Y: np.ndarray,
    steps: int = 2500,
    batch: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    n_checkpoints: int = 0,
    hidden: int = 64,
) -> tuple[MLPPolicy, list[Checkpoint]]:
    """Plain BC with Adam. Fixed step budget so data subsets get equal compute."""
    rng = np.random.default_rng(seed)
    pol = MLPPolicy(X.shape[1], hidden, Y.shape[1], seed=seed)
    opt = Adam(pol.n_params, lr=lr)
    every = steps // n_checkpoints if n_checkpoints else 0
    ckpts: list[Checkpoint] = []
    for s in range(1, steps + 1):
        idx = rng.integers(0, len(X), size=min(batch, len(X)))
        _, g = pol.loss_grad(X[idx], Y[idx])
        opt.step(pol.theta, g)
        if every and s % every == 0:
            ckpts.append(Checkpoint(s, pol.flat(), lr, opt.preconditioner()))
    return pol, ckpts
