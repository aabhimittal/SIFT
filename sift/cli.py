"""Command line entry point: ``python -m sift {demo,curate,generate,export,report}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import Dataset, make_synthetic
from .export import export_subset
from .pipeline import RunConfig, run
from .report import load_results, write_report


def _config(args) -> RunConfig:
    kw = dict(data_seed=args.data_seed, seeds=tuple(range(args.seeds)), val_mode=args.val_mode,
              precondition=not args.no_precondition, project_dim=args.project_dim)
    if args.quick:
        kw.update(steps=1500, n_checkpoints=6, n_eval=200, fractions=(0.1, 0.3, 0.5, 0.8), ablate_influence=False)
    for name in ("steps", "n_eval", "n_checkpoints"):
        if getattr(args, name) is not None:
            kw[name] = getattr(args, name)
    if args.no_ablation:
        kw["ablate_influence"] = False
    return RunConfig(**kw)


def _positive(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _fraction(v: str) -> float:
    f = float(v)
    if not 0 < f <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return f


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sift", description="Rank robot demonstrations by marginal value.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="report", help="output directory for index.html, results.json, ranking.csv")
        p.add_argument("--seeds", type=_positive, default=3, help="training seeds for influence and evaluation")
        p.add_argument("--val-mode", choices=["rollout", "heldout"], default="rollout",
                       help="validation gradient: sim rollouts relabelled by the expert, or held-out demos")
        p.add_argument("--no-precondition", action="store_true", help="vanilla TracIn dot products")
        p.add_argument("--project-dim", type=_positive, default=None,
                       help="sketch gradients to this many dimensions (for large policies)")
        p.add_argument("--steps", type=_positive, default=None, help="gradient steps per trained policy")
        p.add_argument("--n-checkpoints", type=_positive, default=None, help="TracIn checkpoints per policy")
        p.add_argument("--n-eval", type=_positive, default=None, help="sim episodes per evaluation")
        p.add_argument("--no-ablation", action="store_true", help="skip the validation-gradient ablation")
        p.add_argument("--quick", action="store_true", help="smaller run (~40 s) for a first look")
        p.add_argument("--data-seed", type=int, default=0)

    common(sub.add_parser("demo", help="generate a corrupted dataset, curate it, write the report"))
    c = sub.add_parser("curate", help="curate a dataset saved as .npz")
    c.add_argument("--data", required=True)
    common(c)
    g = sub.add_parser("generate", help="write a synthetic dataset to .npz")
    g.add_argument("--out", default="demos.npz")
    g.add_argument("--data-seed", type=int, default=0)
    e = sub.add_parser("export", help="write the curated subset of a dataset to .npz")
    e.add_argument("--data", required=True, help="the dataset the results were computed on")
    e.add_argument("--results", required=True, help="results.json from demo/curate")
    sel = e.add_mutually_exclusive_group(required=True)
    sel.add_argument("--frac", type=_fraction, help="keep the top fraction of the ranking")
    sel.add_argument("--verdicts", help="comma-separated verdicts to keep, e.g. keep or keep,redundant")
    e.add_argument("--out", default="curated.npz")
    r = sub.add_parser("report", help="re-render the HTML report from a results.json")
    r.add_argument("--results", required=True)
    r.add_argument("--out", default="report")

    args = ap.parse_args(argv)
    try:
        _dispatch(args)
    except (ValueError, FileNotFoundError) as err:
        print(f"sift {args.cmd}: error: {err}", file=sys.stderr)
        raise SystemExit(2) from err


def _dispatch(args) -> None:
    if args.cmd == "generate":
        make_synthetic(seed=args.data_seed).save(args.out)
        print(f"wrote {args.out}")
        return
    if args.cmd == "export":
        verdicts = [v.strip() for v in args.verdicts.split(",") if v.strip()] if args.verdicts else None
        sub = export_subset(Dataset.load(args.data), load_results(args.results), args.frac, verdicts)
        sub.save(args.out)
        print(f"wrote {len(sub)} trajectories to {args.out}")
        return
    if args.cmd == "report":
        page = write_report(load_results(args.results), Path(args.out))
        print(f"report: {page.resolve()}")
        return
    cfg = _config(args)
    ds = Dataset.load(args.data) if args.cmd == "curate" else make_synthetic(seed=args.data_seed)
    results = run(cfg, ds=ds)
    out = Path(args.out)
    page = write_report(results, out)
    if args.cmd == "demo":
        ds.save(out / "demos.npz")
    m = results["scaling"]["matching_fraction"]["SIFT"]
    print(f"\nreport: {page.resolve()}  (ranking: {(out / 'ranking.csv').resolve()})")
    print("SIFT subset matching full-data success: " + (f"{m:.0%} of the data" if m is not None else "none"))
