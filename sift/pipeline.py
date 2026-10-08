"""End-to-end run: generate data, score it, curate it, test the curation in closed loop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .curate import curate, filters_only_order, tracin_only_order
from .data import Dataset, make_synthetic, make_validation
from .duplicates import find_duplicates, find_scene_duplicates
from .env import INSTRUCTIONS, ReachEnv
from .evaluate import auroc, duplicate_pair_prf, prf
from .influence import run_influence
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
    synthetic: dict = field(default_factory=dict)


def _r(x, nd=4):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def run(cfg: RunConfig, ds: Dataset | None = None, log=print) -> dict:
    t0 = time.time()
    timings = {}

    def tick(name, start):
        timings[name] = round(time.time() - start, 1)

    ds = ds or make_synthetic(seed=cfg.data_seed, **cfg.synthetic)
    val = make_validation(cfg=ds.cfg)
    env = ReachEnv(ds.cfg)
    tags = ds.tags()
    bad = np.isin(tags, ["mislabeled", "failed"])
    log(f"dataset: {len(ds)} trajectories  " + ", ".join(f"{k}={int((tags == k).sum())}" for k in dict.fromkeys(tags)))

    s = time.time()
    log(f"[1/5] TracIn influence ({cfg.val_mode} validation gradient, {len(cfg.seeds)} seeds)")
    infl = run_influence(ds, val, env, seeds=cfg.seeds, steps=cfg.steps, n_checkpoints=cfg.n_checkpoints,
                         mode=cfg.val_mode, precondition=cfg.precondition, log=log)
    tick("influence", s)

    ablation = []
    if cfg.ablate_influence:
        s = time.time()
        log("      ablation: validation gradient x preconditioning")
        for mode in ("rollout", "heldout"):
            for pre in (True, False):
                r = infl if (mode, pre) == (cfg.val_mode, cfg.precondition) else run_influence(
                    ds, val, env, seeds=cfg.seeds, steps=cfg.steps, n_checkpoints=cfg.n_checkpoints,
                    mode=mode, precondition=pre, log=lambda *a: None)
                sp = r.seed_spearman()
                ablation.append({
                    "val_mode": mode, "precondition": pre,
                    "auroc_bad": auroc(-r.mean, bad),
                    "auroc_mislabeled": auroc(-r.mean, tags == "mislabeled"),
                    "auroc_failed": auroc(-r.mean, tags == "failed"),
                    "seed_spearman": float(sp[np.triu_indices(len(sp), 1)].mean()),
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
    judge = CounterfactualJudge(steps=cfg.steps).judge(ds)
    tick("judge", s)

    log("[4/5] curation")
    cur = curate(infl, dups, judge)
    log("      " + ", ".join(f"{k}={v}" for k, v in cur.counts().items()))

    s = time.time()
    log("[5/5] scaling curves (fixed step budget, held-out eval episodes)")
    full_idx = np.arange(len(ds))
    oracle_idx = np.array([i for i in full_idx if tags[i] in ("clean", "scene_reuse")])
    full = fixed_subset_success(ds, full_idx, env, cfg.seeds, cfg.steps, cfg.n_eval)
    oracle = fixed_subset_success(ds, oracle_idx, env, cfg.seeds, cfg.steps, cfg.n_eval)
    log(f"  full data: {full.mean():.2f} ± {full.std():.2f}   oracle-clean ({len(oracle_idx) / len(ds):.0%}): "
        f"{oracle.mean():.2f} ± {oracle.std():.2f}")
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

    detection = {
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

    return {
        "meta": {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                 "n": len(ds), "instructions": list(INSTRUCTIONS), "timings": timings,
                 "success_radius": ds.cfg.success_radius, "horizon": ds.cfg.horizon,
                 "n_val": len(val)},
        "counts": cur.counts(),
        "tags": {k: int((tags == k).sum()) for k in dict.fromkeys(tags)},
        "detection": detection,
        "ablation": ablation,
        "stability": {"spearman": infl.seed_spearman().tolist(), "top30_jaccard": infl.topk_overlap(0.3)},
        "scaling": {
            "fractions": list(cfg.fractions),
            "curves": {k: {"mean": _r(v.mean(1)), "std": _r(v.std(1)), "raw": _r(v)} for k, v in curves.items()},
            "full": {"mean": float(full.mean()), "std": float(full.std()), "raw": _r(full)},
            "oracle": {"mean": float(oracle.mean()), "std": float(oracle.std()), "raw": _r(oracle),
                       "fraction": len(oracle_idx) / len(ds)},
            "matching_fraction": match,
        },
        "trajectories": trajs,
    }
