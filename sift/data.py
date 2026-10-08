"""Trajectory containers and a synthetic 'collected, not curated' dataset.

The generator reproduces the failure modes SIFT is meant to find, and records
ground truth for each so detectors can be scored rather than eyeballed:

``clean``        expert demo, operator speed varies per demo
``duplicate``    an operator re-recorded the same scene and motion several times
                 (the first recording of each cluster stays ``clean``)
``mislabeled``   motion goes to one target, instruction names another
``failed``       noisy, hesitant demo that stalls short of the goal
``scene_reuse``  same scene as another demo but a different instruction: looks
                 identical in image space, behaves differently in action space
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import COLORS, INSTRUCTIONS, EnvConfig, clip_norm, expert_action, observe, sample_scene, step

TAGS = ("clean", "duplicate", "mislabeled", "failed", "scene_reuse")


@dataclass
class Trajectory:
    obs: np.ndarray          # (T, OBS_DIM)
    actions: np.ndarray      # (T, 2), raw velocity commands
    pos: np.ndarray          # (T+1, 2), effector path (proprioception)
    targets: np.ndarray      # (3, 2), scene layout
    instr: int               # index into INSTRUCTIONS: the *label* as collected
    executed: int = -1       # target actually reached for (ground truth, synthetic only)
    tag: str = "clean"       # ground-truth corruption tag (synthetic only)
    true_cluster: int = -1   # ground-truth duplicate cluster (synthetic only)
    id: int = -1

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[self.instr]

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class Dataset:
    trajs: list[Trajectory]
    cfg: EnvConfig = field(default_factory=EnvConfig)

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, i: int) -> Trajectory:
        return self.trajs[i]

    def subset(self, idx) -> "Dataset":
        return Dataset([self.trajs[i] for i in idx], self.cfg)

    def arrays(self, idx=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack steps into (X, Y, owner) where Y is normalised by v_max."""
        idx = range(len(self)) if idx is None else idx
        X, Y, own = [], [], []
        for i in idx:
            t = self.trajs[i]
            X.append(t.obs)
            Y.append(t.actions / self.cfg.v_max)
            own.append(np.full(len(t), i))
        return np.concatenate(X), np.concatenate(Y), np.concatenate(own)

    def tags(self) -> np.ndarray:
        return np.array([t.tag for t in self.trajs])

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            obs=np.stack([t.obs for t in self.trajs]),
            actions=np.stack([t.actions for t in self.trajs]),
            pos=np.stack([t.pos for t in self.trajs]),
            targets=np.stack([t.targets for t in self.trajs]),
            instr=np.array([t.instr for t in self.trajs]),
            executed=np.array([t.executed for t in self.trajs]),
            tag=np.array([t.tag for t in self.trajs]),
            true_cluster=np.array([t.true_cluster for t in self.trajs]),
        )

    @classmethod
    def load(cls, path: str, cfg: EnvConfig | None = None) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        trajs = [
            Trajectory(z["obs"][i], z["actions"][i], z["pos"][i], z["targets"][i], int(z["instr"][i]),
                       int(z["executed"][i]), str(z["tag"][i]), int(z["true_cluster"][i]), id=i)
            for i in range(len(z["instr"]))
        ]
        return cls(trajs, cfg or EnvConfig())


def record_demo(start, targets, goal_idx, label_idx, cfg: EnvConfig, rng: np.random.Generator,
                gain: float = 0.5, noise: float = 0.004, stall_at: float | None = None) -> Trajectory:
    """Roll the scripted teleoperator. ``stall_at`` in (0,1) makes it give up partway."""
    pos = np.array(start, dtype=np.float64)
    goal = targets[goal_idx]
    d0 = np.linalg.norm(goal - pos)
    obs, acts, path = [], [], [pos.copy()]
    for _ in range(cfg.horizon):
        o = observe(pos, targets, label_idx)
        a = expert_action(pos, goal, cfg, gain) + rng.normal(0, noise, size=2)
        if stall_at is not None and np.linalg.norm(goal - pos) < (1 - stall_at) * d0:
            a = rng.normal(0, noise, size=2)  # operator hesitates / gives up
        a = clip_norm(a, cfg.v_max)
        obs.append(o)
        acts.append(a)
        pos = step(pos, a, cfg)
        path.append(pos.copy())
    return Trajectory(np.array(obs), np.array(acts), np.array(path), np.array(targets),
                      instr=label_idx, executed=goal_idx)


def make_synthetic(
    n_clean: int = 150,
    n_dup_clusters: int = 8,
    dup_cluster_size: int = 8,
    n_mislabeled: int = 20,
    n_failed: int = 20,
    n_scene_reuse: int = 20,
    seed: int = 0,
    cfg: EnvConfig | None = None,
) -> Dataset:
    cfg = cfg or EnvConfig()
    rng = np.random.default_rng(seed)
    k = len(COLORS)
    out: list[Trajectory] = []

    def clean_demo():
        s, tg = sample_scene(rng, cfg)
        g = int(rng.integers(k))
        return record_demo(s, tg, g, g, cfg, rng, gain=rng.uniform(0.35, 0.8))

    clean = [clean_demo() for _ in range(n_clean)]
    out += clean

    for c in range(n_dup_clusters):
        s, tg = sample_scene(rng, cfg)
        g = int(rng.integers(k))
        gain = rng.uniform(0.35, 0.8)
        for j in range(dup_cluster_size):
            jitter = rng.normal(0, 0.004, size=2) if j else 0.0
            t = record_demo(s + jitter, tg, g, g, cfg, rng, gain=gain, noise=0.002)
            t.tag = "clean" if j == 0 else "duplicate"
            t.true_cluster = c
            out.append(t)

    for _ in range(n_mislabeled):
        s, tg = sample_scene(rng, cfg)
        g = int(rng.integers(k))
        wrong = int((g + rng.integers(1, k)) % k)
        t = record_demo(s, tg, g, wrong, cfg, rng, gain=rng.uniform(0.35, 0.8))
        t.tag = "mislabeled"
        out.append(t)

    for _ in range(n_failed):
        s, tg = sample_scene(rng, cfg)
        g = int(rng.integers(k))
        t = record_demo(s, tg, g, g, cfg, rng, gain=rng.uniform(0.3, 0.6), noise=0.03,
                        stall_at=rng.uniform(0.25, 0.6))
        t.tag = "failed"
        out.append(t)

    for i in rng.choice(n_clean, size=n_scene_reuse, replace=False):
        base = clean[i]
        g = int((base.instr + rng.integers(1, k)) % k)
        t = record_demo(base.pos[0], base.targets, g, g, cfg, rng, gain=rng.uniform(0.35, 0.8))
        t.tag = "scene_reuse"
        out.append(t)

    order = rng.permutation(len(out))
    trajs = [out[i] for i in order]
    for i, t in enumerate(trajs):
        t.id = i
    return Dataset(trajs, cfg)


def make_validation(n: int = 48, seed: int = 777, cfg: EnvConfig | None = None) -> Dataset:
    """A small trusted set of clean expert demos (the 'gold' validation split)."""
    cfg = cfg or EnvConfig()
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(n):
        s, tg = sample_scene(rng, cfg)
        g = int(rng.integers(len(COLORS)))
        t = record_demo(s, tg, g, g, cfg, rng, gain=0.5)
        t.id = i
        trajs.append(t)
    return Dataset(trajs, cfg)
