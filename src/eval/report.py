"""
Reporting utilities for the NutriGraph evaluation framework.

print_summary()            — pretty-print all metric sections to stdout
generate_markdown_report() — write a full Markdown report to disk
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any, precision: int = 3) -> str:
    """Format a number for display; returns 'N/A' for NaN / None."""
    if value is None:
        return "N/A"
    try:
        f = float(value)
        if math.isnan(f):
            return "N/A"
        return f"{f:.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _section(title: str, width: int = 60) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def _col_widths(rows: list[list[str]]) -> list[int]:
    """Compute max column width for each column across all rows."""
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    widths = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a plain-text table with aligned columns."""
    all_rows = [headers] + rows
    widths = _col_widths(all_rows)
    sep = "  ".join("-" * w for w in widths)
    lines: list[str] = []
    for i, row in enumerate(all_rows):
        line = "  ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append(sep)
    return "\n".join(lines)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a GitHub-flavoured Markdown table."""
    def row_str(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join("---" for _ in headers) + " |"
    lines = [row_str(headers), sep] + [row_str(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual section renderers
# ---------------------------------------------------------------------------

def _mae_section(mae: dict, mape: dict) -> tuple[str, str]:
    """Return (plain-text, markdown) for the MAE & MAPE table."""
    headers = ["Nutrient", "MAE", "MAPE (%)"]
    nutrients = [
        ("calories_kcal", "Calories (kcal)"),
        ("protein_g", "Protein (g)"),
        ("carbs_g", "Carbs (g)"),
        ("fat_g", "Fat (g)"),
        ("fiber_g", "Fiber (g)"),
    ]
    rows = [
        [label, _fmt(mae.get(key)), _fmt(mape.get(key))]
        for key, label in nutrients
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


def _efficiency_section(eff: dict) -> tuple[str, str]:
    hist = eff.get("hist", {})
    headers = ["Metric", "Value"]
    rows = [
        ["Mean questions / dish", _fmt(eff.get("mean_questions"))],
        ["Median questions / dish", _fmt(eff.get("median_questions"))],
        ["Dishes with 0 questions", str(hist.get("0", "N/A"))],
        ["Dishes with 1 question", str(hist.get("1", "N/A"))],
        ["Dishes with 2 questions", str(hist.get("2", "N/A"))],
        ["Dishes with 3+ questions", str(hist.get("3+", "N/A"))],
        ["High-conf & 0 questions (fraction)", _fmt(eff.get("high_confidence_zero_q_fraction"))],
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


def _judge_section(js: dict) -> tuple[str, str]:
    buckets = js.get("buckets", {})
    headers = ["Metric", "Value"]
    rows = [
        ["Mean score", _fmt(js.get("mean"))],
        ["Median score", _fmt(js.get("median"))],
        ["Std dev", _fmt(js.get("std"))],
        ["Scores 0–3 (count)", str(buckets.get("0-3", "N/A"))],
        ["Scores 3–7 (count)", str(buckets.get("3-7", "N/A"))],
        ["Scores 7–10 (count)", str(buckets.get("7-10", "N/A"))],
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


def _consistency_section(cs: dict) -> tuple[str, str]:
    headers = ["Metric", "Value"]
    rows = [
        ["Mean calorie variance", _fmt(cs.get("mean_variance"))],
        ["High-variance dish fraction (>5% GT)", _fmt(cs.get("high_variance_fraction"))],
        ["Note", str(cs.get("note", ""))],
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


def _latency_section(ls: dict) -> tuple[str, str]:
    headers = ["Percentile", "Latency (s)"]
    rows = [
        ["Mean", _fmt(ls.get("mean"))],
        ["Median (p50)", _fmt(ls.get("median"))],
        ["p90", _fmt(ls.get("p90"))],
        ["p95", _fmt(ls.get("p95"))],
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


def _calibration_section(cal: dict) -> tuple[str, str]:
    headers = ["Confidence range", "Count", "MAE calories"]
    rows = [
        [
            str(info.get("range", key)),
            str(info.get("count", "N/A")),
            _fmt(info.get("mae_calories")),
        ]
        for key, info in cal.items()
    ]
    return _format_table(headers, rows), _md_table(headers, rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_summary(summary: dict[str, Any]) -> None:
    """
    Pretty-print all evaluation metric sections to stdout.

    Uses only stdlib — no external dependencies.
    """
    total = summary.get("total_records", "?")
    ok = summary.get("successful_records", "?")
    coverage = summary.get("coverage", float("nan"))

    print(_section("NutriGraph Evaluation Summary"))
    print(f"\n  Total records  : {total}")
    print(f"  Successful     : {ok}")
    print(f"  Coverage       : {_fmt(coverage, 4)} ({_fmt(float(coverage) * 100 if isinstance(coverage, (int, float)) else float('nan'), 1)} %)")

    print(_section("MAE & MAPE"))
    plain, _ = _mae_section(summary.get("mae", {}), summary.get("mape", {}))
    print(plain)

    print(_section("Agent Efficiency (Clarification Questions)"))
    plain, _ = _efficiency_section(summary.get("agent_efficiency", {}))
    print(plain)

    print(_section("LLM-as-a-Judge Scores (0–10)"))
    plain, _ = _judge_section(summary.get("judge_stats", {}))
    print(plain)

    print(_section("Consistency (Cross-Repeat Variance)"))
    plain, _ = _consistency_section(summary.get("consistency", {}))
    print(plain)

    print(_section("Latency (wall-clock, seconds)"))
    plain, _ = _latency_section(summary.get("latency_stats", {}))
    print(plain)

    print(_section("Confidence Calibration"))
    plain, _ = _calibration_section(summary.get("confidence_calibration", {}))
    print(plain)

    print()


def generate_markdown_report(
    summary: dict[str, Any],
    output_path: str | Path = "artifacts/eval_report.md",
) -> Path:
    """
    Write a Markdown evaluation report and return the resolved output path.

    Parameters
    ----------
    summary:
        The ``summary_metrics`` dict returned by ``run_evaluation()``.
    output_path:
        Destination path for the ``.md`` file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = summary.get("total_records", "?")
    ok = summary.get("successful_records", "?")
    coverage = summary.get("coverage", float("nan"))

    _, mae_md = _mae_section(summary.get("mae", {}), summary.get("mape", {}))
    _, eff_md = _efficiency_section(summary.get("agent_efficiency", {}))
    _, judge_md = _judge_section(summary.get("judge_stats", {}))
    _, cons_md = _consistency_section(summary.get("consistency", {}))
    _, lat_md = _latency_section(summary.get("latency_stats", {}))
    _, cal_md = _calibration_section(summary.get("confidence_calibration", {}))

    lines: list[str] = [
        "# NutriGraph Evaluation Report",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Total records | {total} |",
        f"| Successful records | {ok} |",
        f"| Coverage | {_fmt(coverage, 4)} ({_fmt(float(coverage) * 100 if isinstance(coverage, (int, float)) else float('nan'), 1)} %) |",
        "",
        "## MAE & MAPE",
        "",
        mae_md,
        "",
        "## Agent Efficiency (Clarification Questions)",
        "",
        eff_md,
        "",
        "## LLM-as-a-Judge Scores (0–10)",
        "",
        judge_md,
        "",
        "## Consistency (Cross-Repeat Variance)",
        "",
        cons_md,
        "",
        "## Latency (wall-clock, seconds)",
        "",
        lat_md,
        "",
        "## Confidence Calibration",
        "",
        cal_md,
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
