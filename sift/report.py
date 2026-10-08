"""Render a pipeline result into a single self-contained HTML page."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def render_fragment(results: dict) -> str:
    """The page body without a document skeleton (what an Artifact host wraps)."""
    data = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("/*__SIFT_DATA__*/", data)


def render(results: dict) -> str:
    return '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">\n' + render_fragment(results) + "\n</html>\n"


def write_report(results: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    page = out / "index.html"
    page.write_text(render(results))
    return page
