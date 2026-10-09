"""End-to-end run: generate data, score it, curate it, test the curation in closed loop."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .curate import curate, filters_only_order, tracin_only_order
from .data import Dataset, make_synthetic, make_validation
from .duplicates import find_duplicates, find_scene_duplicates
from .env import INSTRUCTIONS, ReachEnv
from .evaluate import auroc, duplicate_pair_prf, prf
from .influence import VAL_MODES, run_influence
from .judge import CounterfactualJudge
from .scaling import fixed_subset_success, matching_fraction, scaling_curve


@dataclass
class RunConfig:
    data_seed: int = 0
    seeds: tuple = (0, 1, 2)
    steps: int = 2500
    n_checkpoints: int = 10
    val_mode: str = "rollout"
    precondition: bool = True
    fractions: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
    n_eval: int = 300
    ablate_influence: bool = True
    project_dim: int | None = None
    judge_folds: int = 3
    synthetic: dict = field(default_factory=dict)

    def __post_init__(self):
        self.seeds = tuple(self.seeds)
        self.fractions = tuple(sorted(set(float(f) for f in self.fractions)))
        if not self.seeds:
            raise ValueError("need at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct (repeated seeds understate variance)")
        if not self.fractions or not all(0 < f <= 1 for f in self.fractions):
            raise ValueError("fractions must be non-empty and each in (0, 1]")
        if self.steps < 1 or self.n_eval < 1 or self.n_checkpoints < 1:
            raise ValueError("steps, n_eval and n_checkpoints must all be >= 1")
        if self.val_mode not in VAL_MODES:
            raise ValueError(f"val_mode must be one of {VAL_MODES}")
        if self.project_dim is not None and self.project_dim < 1:
            raise ValueError("project_dim must be >= 1")


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def json_safe(x):
    """Replace NaN/inf with None so the result is valid JSON (JS JSON.parse rejects NaN)."""
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(cfg: RunConfig, ds: Dataset | None = None, log=print) -> dict:
    t0 = time.time()
    timings = {}

    def tick(name, start):
        timings[name] = round(time.time() - start, 1)

    ds = ds if ds is not None else make_synthetic(seed=cfg.data_seed, **cfg.synthetic)
    if len(ds) < max(3, cfg.judge_folds):
        raise ValueError(f"need at least {max(3, cfg.judge_folds)} trajectories, got {len(ds)}")
    gt = ds.has_ground_truth
    val = make_validation(cfg=ds.cfg)
    env = ReachEnv(ds.cfg)
    tags = ds.tags()
    bad = np.isin(tags, ["mislabeled", "failed"])
    if not gt:
        log("no ground-truth tags: detector accuracy and the oracle reference are skipped")
    log(f"dataset: {len(ds)} trajectories  " + ", ".join(f"{k}={int((tags == k).sum())}" for k in dict.fromkeys(tags)))

    s = time.time()
    log(f"[1/5] TracIn influence ({cfg.val_mode} validation gradient, {len(cfg.seeds)} seeds)")
    infl = run_influence(ds, val, env, seeds=cfg.seeds, steps=cfg.steps, n_checkpoints=cfg.n_checkpoints,
                         mode=cfg.val_mode, precondition=cfg.precondition, project_dim=cfg.project_dim, log=log)
    tick("influence", s)

    ablation = []
    if cfg.ablate_influence:
        s = time.time()
        log("      ablation: validation gradient x preconditioning")
        for mode in ("rollout", "heldout"):
            for pre in (True, False):
                r = infl if (mode, pre) == (cfg.val_mode, cfg.precondition) else run_influence(
                    ds, val, env, seeds=cfg.seeds, steps=cfg.steps, n_checkpoints=cfg.n_checkpoints,
                    mode=mode, precondition=pre, project_dim=cfg.project_dim, log=lambda *a: None)
                ablation.append({
                    "val_mode": mode, "precondition": pre,
                    "auroc_bad": auroc(-r.mean, bad) if gt else None,
                    "auroc_mislabeled": auroc(-r.mean, tags == "mislabeled") if gt else None,
                    "auroc_failed": auroc(-r.mean, tags == "failed") if gt else None,
                    "seed_spearman": r.mean_seed_spearman(),
                    "top30_jaccard": r.topk_overlap(0.3),
                })
        tick("ablation", s)

    s = time.time()
    log("[2/5] near-duplicates in action space")
    dups = find_duplicates(ds)
    scene = find_scene_duplicates(ds)
    tick("duplicates", s)

    s = time.time()
    log("[3/5] instruction consistency (cross-fitted counterfactual judge)")
    judge = CounterfactualJudge(folds=cfg.judge_folds, steps=cfg.steps).judge(ds)
    tick("judge", s)

    log("[4/5] curation")
    cur = curate(infl, dups, judge)
    log("      " + ", ".join(f"{k}={v}" for k, v in cur.counts().items()))

    s = time.time()
    log("[5/5] scaling curves (fixed step budget, held-out eval episodes)")
    full_idx = np.arange(len(ds))
    oracle_idx = np.array([i for i in full_idx if tags[i] in ("clean", "scene_reuse")], dtype=int) if gt \
        else np.array([], dtype=int)
    full = fixed_subset_success(ds, full_idx, env, cfg.seeds, cfg.steps, cfg.n_eval)
    oracle = fixed_subset_success(ds, oracle_idx, env, cfg.seeds, cfg.steps, cfg.n_eval) if len(oracle_idx) else None
    log(f"  full data: {full.mean():.2f} ± {full.std():.2f}" + (
        f"   oracle-clean ({len(oracle_idx) / len(ds):.0%}): {oracle.mean():.2f} ± {oracle.std():.2f}"
        if oracle is not None else ""))
    tr_order = tracin_only_order(infl)
    curves = scaling_curve(ds, env, {
        "SIFT": lambda seed: cur.order,
        "Filters only": lambda seed: filters_only_order(cur, seed),
        "TracIn only": lambda seed: tr_order,
        "Random": lambda seed: np.random.default_rng(seed).permutation(len(ds)),
    }, cfg.fractions, cfg.seeds, cfg.steps, cfg.n_eval, log=log)
    tick("scaling", s)
    timings["total"] = round(time.time() - t0, 1)

    match = {k: matching_fraction(cfg.fractions, v, full) for k, v in curves.items()}
    rank = np.empty(len(ds), int)
    rank[cur.order] = np.arange(len(ds))

    detection = {} if not gt else {
        "mismatch_judge": prf(judge.mismatch, tags == "mislabeled"),
        "duplicates_action_space": duplicate_pair_prf(ds, dups),
        "duplicates_scene_space": duplicate_pair_prf(ds, scene),
        "noisy_flag": prf(cur.verdict == "noisy", tags == "failed"),
        "removed_vs_bad": prf(np.isin(cur.verdict, ["mismatch", "noisy", "harmful"]), bad),
    }
    clusters = {}
    for i, t in enumerate(ds.trajs):
        if t.true_cluster >= 0:
            clusters.setdefault(t.true_cluster, []).append(i)
    kept_one = sum(1 for m in clusters.values() if sum(cur.verdict[i] != "redundant" for i in m) == 1)
    if gt:
        detection["clusters_collapsed"] = {"collapsed": kept_one, "clusters": len(clusters)}

    trajs = []
    for i, t in enumerate(ds.trajs):
        trajs.append({
            "id": i, "instr": int(t.instr), "executed": int(t.executed), "tag": t.tag,
            "true_cluster": int(t.true_cluster),
            "pos": _r(t.pos, 3), "targets": _r(t.targets, 3),
            "infl_mean": float(infl.mean[i]), "infl_std": float(infl.std[i]),
            "infl_seeds": _r(infl.per_seed[:, i], 6), "self_infl": float(infl.self_mean[i]),
            "judge_probs": _r(judge.probs[i], 3), "judge_excess": round(float(judge.excess[i]), 2),
            "judge_pred": int(judge.predicted[i]), "judge_why": judge.rationale[i],
            "oof_loss": round(float(judge.oof_loss[i]), 4),
            "dup_cluster": int(dups.cluster[i]), "scene_cluster": int(scene.cluster[i]),
            "verdict": str(cur.verdict[i]), "rank": int(rank[i]), "representative": int(cur.representative[i]),
            "lcb": float(cur.score[i]),
        })

    return json_safe({
        "meta": {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 "n": len(ds), "instructions": list(INSTRUCTIONS), "timings": timings,
                 "success_radius": ds.cfg.success_radius, "horizon": ds.cfg.horizon,
                 "n_val": len(val), "ground_truth": gt, "fingerprint": ds.fingerprint()},
        "counts": cur.counts(),
        "tags": {k: int((tags == k).sum()) for k in dict.fromkeys(tags)},
        "detection": detection,
        "ablation": ablation,
        "stability": {"spearman": infl.seed_spearman().tolist(), "top30_jaccard": infl.topk_overlap(0.3)},
        "scaling": {
            "fractions": list(cfg.fractions),
            "curves": {k: {"mean": _r(v.mean(1)), "std": _r(v.std(1)), "raw": _r(v)} for k, v in curves.items()},
            "full": {"mean": float(full.mean()), "std": float(full.std()), "raw": _r(full)},
            "oracle": None if oracle is None else {
                "mean": float(oracle.mean()), "std": float(oracle.std()), "raw": _r(oracle),
                "fraction": len(oracle_idx) / len(ds)},
            "matching_fraction": match,
        },
        "trajectories": trajs,
    })
