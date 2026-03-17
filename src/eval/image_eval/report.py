"""
Console summary and Markdown report generators for the two image-based
evaluation frameworks.

print_image_ingredients_summary(metrics)
    Pretty-print image→ingredients metrics to stdout.

print_full_pipeline_summary(metrics)
    Pretty-print full-pipeline macro metrics to stdout.

generate_image_ingredients_report(records, metrics, path)
    Write a Markdown report for the image→ingredients eval.

generate_full_pipeline_report(records, metrics, path)
    Write a Markdown report for the full-pipeline eval.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import ImageEvalRecord

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(val: Any, decimals: int = 2, pct: bool = False) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if pct:
        return f"{float(val):.{decimals}f}%"
    return f"{float(val):.{decimals}f}"


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------

def print_image_ingredients_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print image → ingredients evaluation metrics."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  IMAGE → INGREDIENTS EVALUATION SUMMARY")
    print(sep)

    n_t = metrics.get("n_total", "?")
    n_s = metrics.get("n_success", "?")
    cov = metrics.get("coverage", math.nan)
    print(f"  Coverage       : {n_s}/{n_t} dishes ({_fmt(cov * 100, pct=True)})")

    rec  = (metrics.get("ingredient_recall",   {}) or {}).get("mean_recall", math.nan)
    prec = (metrics.get("ingredient_precision",{}) or {}).get("mean_precision", math.nan)
    f1   = (metrics.get("ingredient_f1",       {}) or {}).get("mean_f1", math.nan)
    print(f"\n  Ingredient Recall    : {_fmt(rec * 100 if not math.isnan(rec) else rec, pct=True)}")
    print(f"  Ingredient Precision : {_fmt(prec * 100 if not math.isnan(prec) else prec, pct=True)}")
    print(f"  Ingredient F1        : {_fmt(f1 * 100 if not math.isnan(f1) else f1, pct=True)}")

    dn = metrics.get("dish_name_accuracy", {}) or {}
    print(f"\n  Dish Name Exact Match : {_fmt((dn.get('exact_match_rate', math.nan) or math.nan) * 100, pct=True)}")
    print(f"  Dish Name Fuzzy Match : {_fmt((dn.get('fuzzy_match_rate', math.nan) or math.nan) * 100, pct=True)}")

    lat = metrics.get("latency", {}) or {}
    print(f"\n  Latency (mean/P50/P90): "
          f"{_fmt(lat.get('mean'))}s / {_fmt(lat.get('p50'))}s / {_fmt(lat.get('p90'))}s")

    if "judge" in metrics:
        j = metrics["judge"]
        print(f"\n  Judge score (mean)   : {_fmt(j.get('mean_score'))} / 10"
              f"  (n={j.get('n_judged', 0)})")

    print(sep + "\n")


def print_full_pipeline_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print full image-to-macros pipeline evaluation metrics."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  FULL PIPELINE (IMAGE → MACROS) EVALUATION SUMMARY")
    print(sep)

    n_t = metrics.get("n_total", "?")
    n_s = metrics.get("n_success", "?")
    cov = metrics.get("coverage", math.nan)
    print(f"  Coverage       : {n_s}/{n_t} dishes ({_fmt(cov * 100, pct=True)})")

    mae = metrics.get("mae", {}) or {}
    print(f"\n  MAE Calories  : {_fmt(mae.get('calories_kcal'))} kcal")
    print(f"  MAE Protein   : {_fmt(mae.get('protein_g'))} g")
    print(f"  MAE Carbs     : {_fmt(mae.get('carbs_g'))} g")
    print(f"  MAE Fat       : {_fmt(mae.get('fat_g'))} g")
    print(f"  MAE Fiber     : {_fmt(mae.get('fiber_g'))} g")

    mape = metrics.get("mape_calories_pct", math.nan)
    print(f"\n  MAPE Calories : {_fmt(mape, pct=True)}")

    lat = metrics.get("latency", {}) or {}
    print(f"\n  Latency (mean/P50/P90): "
          f"{_fmt(lat.get('mean'))}s / {_fmt(lat.get('p50'))}s / {_fmt(lat.get('p90'))}s")

    if "judge" in metrics:
        j = metrics["judge"]
        print(f"\n  Judge score (mean)   : {_fmt(j.get('mean_score'))} / 10"
              f"  (n={j.get('n_judged', 0)})")

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------

def generate_image_ingredients_report(
    records: list[ImageEvalRecord],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a Markdown report for the image → ingredients evaluation."""
    lines: list[str] = [
        "# NutriGraph — Image → Ingredients Evaluation Report\n",
        f"**Total dishes evaluated:** {metrics.get('n_total', '?')}  ",
        f"**Successful calls:** {metrics.get('n_success', '?')}  ",
        f"**Coverage:** {_fmt((metrics.get('coverage') or math.nan) * 100, pct=True)}\n",
        "## Ingredient Identification Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    rec  = (metrics.get("ingredient_recall",   {}) or {}).get("mean_recall",   math.nan)
    prec = (metrics.get("ingredient_precision",{}) or {}).get("mean_precision", math.nan)
    f1   = (metrics.get("ingredient_f1",       {}) or {}).get("mean_f1",       math.nan)
    dn   = metrics.get("dish_name_accuracy", {}) or {}

    def _pct_row(label: str, val: float) -> str:
        return f"| {label} | {_fmt(val * 100 if not math.isnan(val) else val, pct=True)} |"

    lines += [
        _pct_row("Ingredient Recall",        rec),
        _pct_row("Ingredient Precision",     prec),
        _pct_row("Ingredient F1",            f1),
        _pct_row("Dish Name Exact Match",    dn.get("exact_match_rate", math.nan) or math.nan),
        _pct_row("Dish Name Fuzzy Match",    dn.get("fuzzy_match_rate", math.nan) or math.nan),
    ]

    lat = metrics.get("latency", {}) or {}
    lines += [
        "\n## Latency\n",
        "| Percentile | Seconds |",
        "|------------|---------|",
        f"| Mean | {_fmt(lat.get('mean'))} |",
        f"| P50  | {_fmt(lat.get('p50'))} |",
        f"| P90  | {_fmt(lat.get('p90'))} |",
        f"| P99  | {_fmt(lat.get('p99'))} |",
    ]

    if "judge" in metrics:
        j = metrics["judge"]
        lines += [
            "\n## LLM Judge Score\n",
            f"Mean: **{_fmt(j.get('mean_score'))} / 10** "
            f"(min {_fmt(j.get('min_score'))}, max {_fmt(j.get('max_score'))}, "
            f"n={j.get('n_judged', 0)})\n",
        ]

    lines.append("\n## Per-Dish Results\n")
    lines.append("| Dish ID | Dish Name | #Predicted Ingredients | Latency (s) | Success |")
    lines.append("|---------|-----------|----------------------|-------------|---------|")
    for r in records:
        n_pred = len(r.prediction.extracted_ingredients) if r.prediction else "-"
        lat_s  = _fmt(r.prediction.latency_seconds) if r.prediction else "-"
        ok     = "✓" if r.success else "✗"
        lines.append(
            f"| {r.golden.dish_id} | {r.golden.dish_name} "
            f"| {n_pred} | {lat_s} | {ok} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_full_pipeline_report(
    records: list[ImageEvalRecord],
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a Markdown report for the full image → macros pipeline evaluation."""
    lines: list[str] = [
        "# NutriGraph — Full Pipeline (Image → Macros) Evaluation Report\n",
        f"**Total dishes evaluated:** {metrics.get('n_total', '?')}  ",
        f"**Successful calls:** {metrics.get('n_success', '?')}  ",
        f"**Coverage:** {_fmt((metrics.get('coverage') or math.nan) * 100, pct=True)}\n",
        "## Macro-Level Accuracy (MAE)\n",
        "| Macro | MAE |",
        "|-------|-----|",
    ]
    mae = metrics.get("mae", {}) or {}
    lines += [
        f"| Calories (kcal) | {_fmt(mae.get('calories_kcal'))} |",
        f"| Protein (g)     | {_fmt(mae.get('protein_g'))} |",
        f"| Carbs (g)       | {_fmt(mae.get('carbs_g'))} |",
        f"| Fat (g)         | {_fmt(mae.get('fat_g'))} |",
        f"| Fiber (g)       | {_fmt(mae.get('fiber_g'))} |",
        f"\n**MAPE (Calories):** {_fmt(metrics.get('mape_calories_pct', math.nan), pct=True)}\n",
    ]

    lat = metrics.get("latency", {}) or {}
    lines += [
        "## Latency\n",
        "| Percentile | Seconds |",
        "|------------|---------|",
        f"| Mean | {_fmt(lat.get('mean'))} |",
        f"| P50  | {_fmt(lat.get('p50'))} |",
        f"| P90  | {_fmt(lat.get('p90'))} |",
        f"| P99  | {_fmt(lat.get('p99'))} |",
    ]

    if "judge" in metrics:
        j = metrics["judge"]
        lines += [
            "\n## LLM Judge Score\n",
            f"Mean: **{_fmt(j.get('mean_score'))} / 10** "
            f"(min {_fmt(j.get('min_score'))}, max {_fmt(j.get('max_score'))}, "
            f"n={j.get('n_judged', 0)})\n",
        ]

    lines.append("\n## Per-Dish Results\n")
    lines.append(
        "| Dish ID | Dish Name | Pred Cal | True Cal | Pred Pro | True Pro | Latency (s) |"
    )
    lines.append(
        "|---------|-----------|----------|----------|----------|----------|-------------|"
    )
    for r in records:
        if r.prediction:
            p = r.prediction
            lines.append(
                f"| {r.golden.dish_id} | {r.golden.dish_name} "
                f"| {_fmt(p.calories_kcal, 0)} | {_fmt(r.golden.calories_kcal, 0)} "
                f"| {_fmt(p.protein_g, 1)} | {_fmt(r.golden.protein_g, 1)} "
                f"| {_fmt(p.latency_seconds)} |"
            )
        else:
            lines.append(
                f"| {r.golden.dish_id} | {r.golden.dish_name} "
                f"| - | {_fmt(r.golden.calories_kcal, 0)} | - | {_fmt(r.golden.protein_g, 1)} | - |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
