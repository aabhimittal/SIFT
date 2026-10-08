"""Command line entry point: ``python -m sift {demo,generate,curate}``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import Dataset, make_synthetic
from .pipeline import RunConfig, run
from .report import write_report


def _config(args) -> RunConfig:
    cfg = RunConfig(data_seed=args.data_seed, seeds=tuple(range(args.seeds)), val_mode=args.val_mode,
                    precondition=not args.no_precondition)
    if args.quick:
        cfg.steps, cfg.n_checkpoints, cfg.n_eval = 1500, 6, 200
        cfg.fractions = (0.1, 0.3, 0.5, 0.8)
        cfg.ablate_influence = False
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sift", description="Rank robot demonstrations by marginal value.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="report", help="output directory for index.html and results.json")
        p.add_argument("--seeds", type=int, default=3, help="training seeds for influence and evaluation")
        p.add_argument("--val-mode", choices=["rollout", "heldout"], default="rollout",
                       help="validation gradient: sim rollouts relabelled by the expert, or held-out demos")
        p.add_argument("--no-precondition", action="store_true", help="vanilla TracIn dot products")
        p.add_argument("--quick", action="store_true", help="smaller run (~40 s) for a first look")
        p.add_argument("--data-seed", type=int, default=0)

    common(sub.add_parser("demo", help="generate a corrupted dataset, curate it, write the report"))
    c = sub.add_parser("curate", help="curate a dataset saved with `sift generate`")
    c.add_argument("--data", required=True)
    common(c)
    g = sub.add_parser("generate", help="write a synthetic dataset to .npz")
    g.add_argument("--out", default="demos.npz")
    g.add_argument("--data-seed", type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == "generate":
        make_synthetic(seed=args.data_seed).save(args.out)
        print(f"wrote {args.out}")
        return
    ds = Dataset.load(args.data) if args.cmd == "curate" else None
    results = run(_config(args), ds=ds)
    page = write_report(results, Path(args.out))
    m = results["scaling"]["matching_fraction"]["SIFT"]
    print(f"\nreport: {page.resolve()}")
    print("SIFT subset matching full-data success: " + (f"{m:.0%} of the data" if m is not None else "none"))
