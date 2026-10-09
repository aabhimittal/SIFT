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

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .env import ACT_DIM, COLORS, INSTRUCTIONS, OBS_DIM, EnvConfig, clip_norm, expert_action, observe, sample_scene, step

TAGS = ("clean", "duplicate", "mislabeled", "failed", "scene_reuse")
UNKNOWN = "unknown"  # tag for data without ground truth (any real dataset)
FORMAT_VERSION = 2


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

    def __post_init__(self):
        self.obs = np.asarray(self.obs, dtype=np.float64)
        self.actions = np.asarray(self.actions, dtype=np.float64)
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.targets = np.asarray(self.targets, dtype=np.float64)
        T = len(self.actions)
        if T < 1:
            raise ValueError("trajectory has no steps")
        if self.obs.shape != (T, OBS_DIM):
            raise ValueError(f"obs shape {self.obs.shape} != ({T}, {OBS_DIM})")
        if self.actions.shape != (T, ACT_DIM):
            raise ValueError(f"actions shape {self.actions.shape} != ({T}, {ACT_DIM})")
        if self.pos.shape != (T + 1, 2):
            raise ValueError(f"pos shape {self.pos.shape} != ({T + 1}, 2)")
        if self.targets.shape != (len(COLORS), 2):
            raise ValueError(f"targets shape {self.targets.shape} != ({len(COLORS)}, 2)")
        if not 0 <= int(self.instr) < len(INSTRUCTIONS):
            raise ValueError(f"instruction index {self.instr} outside 0..{len(INSTRUCTIONS) - 1}")
        if not (np.isfinite(self.obs).all() and np.isfinite(self.actions).all()):
            raise ValueError("trajectory contains NaN or inf")
        self.instr = int(self.instr)

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
        if not X:
            raise ValueError("no trajectories selected")
        return np.concatenate(X), np.concatenate(Y), np.concatenate(own)

    def tags(self) -> np.ndarray:
        return np.array([t.tag for t in self.trajs])

    def fingerprint(self) -> str:
        """Content hash of what training sees (lengths, observations, actions, labels).

        Stored in results so a ranking is never applied to a different dataset
        that merely has the same number of trajectories.
        """
        h = hashlib.sha256()
        for t in self.trajs:
            h.update(np.int64(len(t)).tobytes())
            h.update(np.ascontiguousarray(t.obs, dtype=np.float64).tobytes())
            h.update(np.ascontiguousarray(t.actions, dtype=np.float64).tobytes())
            h.update(np.int64(t.instr).tobytes())
        return h.hexdigest()[:16]

    @property
    def has_ground_truth(self) -> bool:
        return len(self) > 0 and all(t.tag != UNKNOWN for t in self.trajs)

    @classmethod
    def from_arrays(cls, obs, actions, instr, cfg: EnvConfig | None = None) -> "Dataset":
        """Import demos of any (possibly different) lengths from per-trajectory arrays.

        Effector path and scene layout are read back from the observation
        layout, so only observations, actions and instruction indices are needed.
        No ground truth is attached: every tag is ``unknown``.
        """
        cfg = cfg or EnvConfig()
        if not (len(obs) == len(actions) == len(instr)):
            raise ValueError("obs, actions and instr must have one entry per trajectory")
        trajs = []
        for i, (o, a, k) in enumerate(zip(obs, actions, instr)):
            o, a = np.asarray(o, dtype=np.float64), np.asarray(a, dtype=np.float64)
            if o.ndim != 2 or len(o) == 0:
                raise ValueError(f"trajectory {i}: obs must be a non-empty (T, {OBS_DIM}) array")
            p0 = o[:, :2]
            if a.shape != (len(o), ACT_DIM):
                raise ValueError(f"trajectory {i}: actions shape {a.shape} != ({len(o)}, {ACT_DIM})")
            pos = np.vstack([p0, step(p0[-1], a[-1], cfg)])
            targets = o[0, 2:2 + 2 * len(COLORS)].reshape(len(COLORS), 2) + p0[0]
            trajs.append(Trajectory(o, a, pos, targets, int(k), tag=UNKNOWN, id=i))
        return cls(trajs, cfg)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        """Ragged-safe: steps are concatenated, ``lengths`` splits them back."""
        if not self.trajs:
            raise ValueError("refusing to save an empty dataset")
        np.savez_compressed(
            path,
            format_version=np.array(FORMAT_VERSION),
            lengths=np.array([len(t) for t in self.trajs]),
            obs=np.concatenate([t.obs for t in self.trajs]),
            actions=np.concatenate([t.actions for t in self.trajs]),
            pos=np.concatenate([t.pos for t in self.trajs]),
            targets=np.stack([t.targets for t in self.trajs]),
            instr=np.array([t.instr for t in self.trajs]),
            executed=np.array([t.executed for t in self.trajs]),
            tag=np.array([t.tag for t in self.trajs]),
            true_cluster=np.array([t.true_cluster for t in self.trajs]),
        )

    @classmethod
    def load(cls, path: str, cfg: EnvConfig | None = None) -> "Dataset":
        """Load either format. Ground-truth fields are optional (absent -> ``unknown``)."""
        z = np.load(path, allow_pickle=False)
        for key in ("obs", "actions", "instr"):
            if key not in z.files:
                raise ValueError(f"{path}: missing required array {key!r}")
        n = len(z["instr"])
        if "lengths" in z.files:
            lengths = z["lengths"].astype(int)
            cut = np.cumsum(lengths)[:-1]
            obs, actions = np.split(z["obs"], cut), np.split(z["actions"], cut)
            pos = np.split(z["pos"], np.cumsum(lengths + 1)[:-1]) if "pos" in z.files else None
        else:  # legacy: stacked equal-length arrays
            obs, actions = list(z["obs"]), list(z["actions"])
            pos = list(z["pos"]) if "pos" in z.files else None
        if pos is None or "targets" not in z.files:
            ds = cls.from_arrays(obs, actions, z["instr"], cfg)
        else:
            ds = cls([Trajectory(obs[i], actions[i], pos[i], z["targets"][i], int(z["instr"][i]), tag=UNKNOWN, id=i)
                      for i in range(n)], cfg or EnvConfig())
        if "tag" in z.files:
            for i, t in enumerate(ds.trajs):
                t.tag = str(z["tag"][i])
                t.executed = int(z["executed"][i]) if "executed" in z.files else -1
                t.true_cluster = int(z["true_cluster"][i]) if "true_cluster" in z.files else -1
        return ds


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
