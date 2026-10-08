"""A deliberately small language-conditioned manipulation stand-in.

``ReachEnv`` is a 2D tabletop: a point "end effector" and three coloured
targets. An instruction names one colour; the episode succeeds if the effector
ends within ``success_radius`` of that target. It is cheap enough that every
SIFT claim (influence, curation, scaling curve) can be checked in closed loop
on a laptop in minutes, which a real VLA setup cannot offer.

Observation layout (``OBS_DIM = 11``)::

    [ p_x, p_y,                      # effector position
      (t_red - p), (t_green - p),    # target offsets, 2 each
      (t_blue - p),
      onehot(instruction) ]          # 3

Actions are 2D velocity commands, clipped to ``v_max`` in norm. Models are
trained on actions scaled by ``1 / v_max`` so targets are O(1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COLORS = ("red", "green", "blue")
INSTRUCTIONS = tuple(f"reach the {c} target" for c in COLORS)
OBS_DIM = 2 + 2 * len(COLORS) + len(COLORS)
ACT_DIM = 2


@dataclass(frozen=True)
class EnvConfig:
    horizon: int = 30
    v_max: float = 0.06
    success_radius: float = 0.06
    min_target_sep: float = 0.28
    min_start_dist: float = 0.30


def clip_norm(a: np.ndarray, v_max: float) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    scale = np.minimum(1.0, v_max / np.maximum(n, 1e-12))
    return a * scale


def sample_scene(rng: np.random.Generator, cfg: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (start position (2,), target positions (3, 2))."""
    while True:
        targets = rng.uniform(0.1, 0.9, size=(len(COLORS), 2))
        d = np.linalg.norm(targets[:, None] - targets[None], axis=-1)
        if d[np.triu_indices(len(COLORS), 1)].min() < cfg.min_target_sep:
            continue
        start = rng.uniform(0.05, 0.95, size=2)
        if np.linalg.norm(targets - start, axis=-1).min() < cfg.min_start_dist:
            continue
        return start, targets


def observe(pos: np.ndarray, targets: np.ndarray, instr: np.ndarray | int) -> np.ndarray:
    """Vectorised observation. ``pos`` (..., 2), ``targets`` (..., 3, 2), ``instr`` (...)."""
    pos = np.asarray(pos, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    instr = np.asarray(instr)
    rel = (targets - pos[..., None, :]).reshape(*pos.shape[:-1], -1)
    onehot = np.eye(len(COLORS))[instr]
    return np.concatenate([pos, rel, onehot], axis=-1)


def expert_action(pos: np.ndarray, goal: np.ndarray, cfg: EnvConfig, gain: float = 0.5) -> np.ndarray:
    """Proportional controller with a speed cap: the scripted 'teleoperator'."""
    return clip_norm(gain * (goal - pos), cfg.v_max)


def step(pos: np.ndarray, action: np.ndarray, cfg: EnvConfig) -> np.ndarray:
    return np.clip(pos + clip_norm(action, cfg.v_max), 0.0, 1.0)


class ReachEnv:
    """Batched closed-loop evaluator. Policies map obs (B, OBS_DIM) -> actions (B, 2)."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()

    def sample_episodes(self, n: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        starts, targets = zip(*(sample_scene(rng, self.cfg) for _ in range(n)))
        return {
            "start": np.stack(starts),
            "targets": np.stack(targets),
            "instr": rng.integers(0, len(COLORS), size=n),
        }

    def rollout(self, policy, episodes: dict[str, np.ndarray], record: bool = False) -> dict[str, np.ndarray]:
        pos = episodes["start"].copy()
        tg, ins = episodes["targets"], episodes["instr"]
        obs_hist, pos_hist = [], [pos.copy()]
        for _ in range(self.cfg.horizon):
            obs = observe(pos, tg, ins)
            act = policy(obs) * self.cfg.v_max  # policies emit normalised actions
            if record:
                obs_hist.append(obs)
            pos = step(pos, act, self.cfg)
            pos_hist.append(pos.copy())
        goal = tg[np.arange(len(ins)), ins]
        final_dist = np.linalg.norm(pos - goal, axis=-1)
        out = {"success": final_dist < self.cfg.success_radius, "final_dist": final_dist}
        if record:
            out["obs"] = np.stack(obs_hist, axis=1)  # (B, T, OBS_DIM)
            out["pos"] = np.stack(pos_hist, axis=1)  # (B, T+1, 2)
        return out

    def success_rate(self, policy, n: int = 300, seed: int = 10_000) -> float:
        return float(self.rollout(policy, self.sample_episodes(n, seed))["success"].mean())
