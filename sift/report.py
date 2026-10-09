"""Render a pipeline result into a single self-contained HTML page, plus machine-readable outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .pipeline import json_safe

TEMPLATE = Path(__file__).with_name("report_template.html")

RANKING_FIELDS = ("rank", "id", "instruction", "verdict", "influence_mean", "influence_std", "influence_lcb",
                  "judge_excess", "judge_best_instruction", "oof_loss", "dup_cluster", "representative", "tag")


def _dumps(results: dict) -> str:
    # allow_nan=False: a NaN that slipped past json_safe must fail here, not in the reader's browser
    return json.dumps(json_safe(results), separators=(",", ":"), allow_nan=False)


def render_fragment(results: dict) -> str:
    """The page body without a document skeleton (what an Artifact host wraps)."""
    data = _dumps(results).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*__SIFT_DATA__*/", data)


def render(results: dict) -> str:
    return '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">\n' + render_fragment(results) + "\n</html>\n"


def ranking_rows(results: dict) -> list[dict]:
    instr = results["meta"]["instructions"]
    rows = []
    for t in sorted(results["trajectories"], key=lambda t: t["rank"]):
        rows.append({
            "rank": t["rank"] + 1, "id": t["id"], "instruction": instr[t["instr"]], "verdict": t["verdict"],
            "influence_mean": t["infl_mean"], "influence_std": t["infl_std"], "influence_lcb": t["lcb"],
            "judge_excess": t["judge_excess"], "judge_best_instruction": instr[t["judge_pred"]],
            "oof_loss": t["oof_loss"], "dup_cluster": t["dup_cluster"], "representative": t["representative"],
            "tag": t["tag"],
        })
    return rows


def write_ranking_csv(results: dict, path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RANKING_FIELDS)
        w.writeheader()
        w.writerows(ranking_rows(results))
    return path


def write_report(results: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(json_safe(results), indent=1, allow_nan=False))
    write_ranking_csv(results, out / "ranking.csv")
    page = out / "index.html"
    page.write_text(render(results))
    return page


def load_results(path: str | Path) -> dict:
    results = json.loads(Path(path).read_text())
    for key in ("meta", "trajectories", "scaling", "counts"):
        if key not in results:
            raise ValueError(f"{path} is not a SIFT results file (missing {key!r})")
    return results
