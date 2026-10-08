"""Instruction / trajectory consistency judges.

Every judge answers one question per trajectory: *given the motion, how likely
is the instruction it was labelled with?* Two implementations:

``CounterfactualJudge`` (default, no API, no privileged state)
    Cross-fitted counterfactual likelihood. Split the data into K folds; train
    a policy on K-1 folds; for each held-out trajectory, score its actions
    under *every* instruction in the vocabulary by swapping the language
    conditioning. If the motion is far better explained by "reach the blue
    target" than by its label "reach the red target", the label is suspect.
    Cross-fitting matters: a policy that trained on the trajectory has partly
    memorised its wrong label and will vouch for it.

    Side product: the best-case loss over all instructions (``oof_loss``). A
    demo that no instruction explains well is not mislabelled, it is just bad
    motion (a failed or hesitant demo).

    Limit: needs an enumerable instruction set. Open-vocabulary data needs a
    judge that reads language, which is the next class.

``ClaudeVLMJudge`` (optional, needs ``anthropic`` and credentials)
    Renders the trajectory as a top-down image and asks Claude whether the
    motion accomplishes the instruction, returning structured JSON. On a real
    robot you would pass the first/last camera frames instead of a render.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .data import Dataset, Trajectory
from .env import COLORS, INSTRUCTIONS
from .model import train_bc


@dataclass
class JudgeResult:
    p_label: np.ndarray        # (N,) probability the collected label is right
    excess: np.ndarray         # (N,) label loss / best counterfactual loss (1.0 = label is best)
    predicted: np.ndarray      # (N,) most likely instruction index
    probs: np.ndarray          # (N, K) distribution over instructions
    oof_loss: np.ndarray       # (N,) best-case out-of-fold BC loss (motion quality)
    mismatch: np.ndarray       # (N,) bool flag
    rationale: list[str]


def _swap_instruction(obs: np.ndarray, k: int) -> np.ndarray:
    o = obs.copy()
    o[:, -len(COLORS):] = 0.0
    o[:, -len(COLORS) + k] = 1.0
    return o


class CounterfactualJudge:
    """``ratio``: flag when the label's loss is this many times the best counterfactual loss.

    A ratio, not a loss difference or a softmax over raw losses: noisy demos
    have large losses under every instruction, so absolute gaps flag them as
    mislabelled when they are only sloppy. The ratio is scale-free.
    """

    def __init__(self, folds: int = 3, steps: int = 2500, ratio: float = 3.0, temperature: float = 0.5,
                 seed: int = 0):
        self.folds, self.steps, self.ratio, self.temperature, self.seed = folds, steps, ratio, temperature, seed

    def judge(self, ds: Dataset) -> JudgeResult:
        n, K = len(ds), len(COLORS)
        rng = np.random.default_rng(self.seed)
        fold = rng.permutation(n) % self.folds
        losses = np.zeros((n, K))
        v = ds.cfg.v_max
        for f in range(self.folds):
            train_idx = np.where(fold != f)[0]
            X, Y, _ = ds.arrays(train_idx)
            pol, _ = train_bc(X, Y, steps=self.steps, seed=self.seed + f)
            for i in np.where(fold == f)[0]:
                t = ds[i]
                for k in range(K):
                    losses[i, k] = pol.loss(_swap_instruction(t.obs, k), t.actions / v)
        # p_k proportional to L_k^(-1/T): a likelihood-ratio view that is invariant to loss scale
        logits = -np.log(np.maximum(losses, 1e-12)) / self.temperature
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        label = np.array([t.instr for t in ds.trajs])
        p_label = probs[np.arange(n), label]
        pred = losses.argmin(1)
        excess = losses[np.arange(n), label] / np.maximum(losses.min(1), 1e-12)
        mismatch = excess > self.ratio
        why = [
            f"actions fit '{INSTRUCTIONS[pred[i]]}' {excess[i]:.1f}x better than the label "
            f"'{INSTRUCTIONS[label[i]]}'" if mismatch[i]
            else f"label '{INSTRUCTIONS[label[i]]}' explains the actions "
                 + ("best" if pred[i] == label[i] else f"within {excess[i]:.1f}x of the best alternative")
            for i in range(n)
        ]
        return JudgeResult(p_label, excess, pred, probs, losses.min(1), mismatch, why)


# ---------------------------------------------------------------------------
# Rendering (dependency-free PNG) for image-based judges
# ---------------------------------------------------------------------------

_RGB = {"red": (214, 54, 64), "green": (34, 150, 90), "blue": (46, 92, 210)}


def render_png(t: Trajectory, size: int = 256) -> bytes:
    img = np.full((size, size, 3), 248, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]

    def disk(c, r, color):
        cx, cy = c[0] * (size - 1), (1 - c[1]) * (size - 1)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color

    for k, name in enumerate(COLORS):
        disk(t.targets[k], size * 0.045, _RGB[name])
    for a, b in zip(t.pos[:-1], t.pos[1:]):
        for s in np.linspace(0, 1, 6):
            disk(a + s * (b - a), 1.6, (20, 24, 30))
    disk(t.pos[0], 5, (120, 120, 120))
    disk(t.pos[-1], 4, (0, 0, 0))

    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "confidence": {"type": "number"},
        "best_instruction": {"type": "string", "enum": list(INSTRUCTIONS)},
        "rationale": {"type": "string"},
    },
    "required": ["consistent", "confidence", "best_instruction", "rationale"],
    "additionalProperties": False,
}

_VLM_PROMPT = (
    "You are auditing robot demonstration data. The image is a top-down view of a tabletop. "
    "Coloured disks are targets. The thin dark line is the end-effector path; the grey dot is "
    "where it started and the black dot is where it stopped.\n\n"
    "The demonstration was labelled with the instruction: \"{instruction}\".\n\n"
    "Decide whether the motion accomplishes that instruction. Pick which of these instructions the "
    "motion best matches: {options}. Give a confidence between 0 and 1 and a one-sentence rationale."
)


class ClaudeVLMJudge:
    """Consistency judge backed by Claude vision. Each call costs API tokens."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", client=None):
        self.model, self.effort = model, effort
        if client is None:
            import anthropic  # optional dependency: pip install sift-curation[vlm]
            client = anthropic.Anthropic()
        self.client = client

    def build_request(self, t: Trajectory) -> dict:
        img = base64.standard_b64encode(render_png(t)).decode("ascii")
        text = _VLM_PROMPT.format(instruction=t.instruction, options="; ".join(INSTRUCTIONS))
        return dict(
            model=self.model,
            max_tokens=1024,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": _VLM_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
                {"type": "text", "text": text},
            ]}],
        )

    def judge_one(self, t: Trajectory) -> dict:
        resp = self.client.beta.messages.create(**self.build_request(t))
        if resp.stop_reason == "refusal":
            return {"consistent": True, "confidence": 0.0, "best_instruction": t.instruction,
                    "rationale": "judge declined; treated as no evidence"}
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def judge(self, ds: Dataset) -> JudgeResult:
        n, K = len(ds), len(COLORS)
        probs = np.zeros((n, K))
        why = []
        for i, t in enumerate(ds.trajs):
            r = self.judge_one(t)
            best = INSTRUCTIONS.index(r["best_instruction"])
            c = float(np.clip(r["confidence"], 0, 1))
            probs[i] = (1 - c) / K
            probs[i, best] += c
            why.append(r["rationale"])
        label = np.array([t.instr for t in ds.trajs])
        p_label = probs[np.arange(n), label]
        pred = probs.argmax(1)
        excess = probs.max(1) / np.maximum(p_label, 1e-6)
        return JudgeResult(p_label, excess, pred, probs, np.full(n, np.nan), (pred != label) & (p_label < 0.5), why)
