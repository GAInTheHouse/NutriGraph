"""
Evaluation runners for the two image-based NutriGraph eval frameworks.

run_image_ingredients_eval()
    Calls POST /api/v1/analyze-dish for every dish that has a matching image,
    collects ImageEvalRecord objects, and computes image→ingredients metrics
    (recall, precision, F1, dish-name accuracy, coverage, latency).

run_full_pipeline_eval()
    Same API calls; computes full-pipeline metrics instead
    (MAE, MAPE, coverage, latency, LLM-as-a-judge quality score).

Both functions
    • save raw records to artifacts/image_eval_records.csv
    • save a JSON metrics summary
    • return (records, metrics_dict)
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.models import GoldenDish, JudgeResult
from src.eval.metrics import (
    compute_coverage,
    compute_latency_stats,
    compute_mae,
    compute_mape,
    compute_judge_stats,
)
from .client import ImageEvalClient, ImageEvalClientError
from .metrics import (
    compute_dish_name_accuracy,
    compute_ingredient_f1,
    compute_ingredient_precision,
    compute_ingredient_recall,
    records_to_macro_lists,
)
from .models import ImageEvalRecord, ImagePredictionResult

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path("artifacts")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _dish_id_to_image_num(dish_id: str) -> int:
    """
    Convert a dish ID to the corresponding image number.

    ``'D001'`` → 1, ``'D100'`` → 100, ``'D200'`` → 200.
    """
    return int(dish_id[1:])  # strip leading 'D', parse rest as int


def _find_image(images_dir: Path, image_num: int) -> Path | None:
    """
    Return the first file matching ``{image_num}.{jpg,jpeg,png,webp}`` in
    *images_dir*, or *None* if no such file exists.
    """
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidate = images_dir / f"{image_num}.{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_golden_set(golden_csv: Path) -> list[GoldenDish]:
    """Load and parse the golden set CSV into a list of GoldenDish objects."""
    df = pd.read_csv(golden_csv, dtype=str)
    numeric_cols = [
        "serving_size_grams",
        "calories_kcal", "protein_g", "carbs_g",
        "fat_g", "fiber_g",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    dishes: list[GoldenDish] = []
    for _, row in df.iterrows():
        try:
            dishes.append(GoldenDish(**row.to_dict()))
        except Exception as exc:
            logger.warning("Skipping malformed row %s: %s", row.get("dish_id"), exc)
    return dishes


def _records_to_df(records: list[ImageEvalRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in records:
        base = {
            "dish_id":             r.golden.dish_id,
            "dish_name_golden":    r.golden.dish_name,
            "image_path":          r.image_path,
            "success":             r.success,
            "error_message":       r.error_message,
        }
        if r.prediction is not None:
            p = r.prediction
            base.update({
                "dish_name_predicted":   p.dish_name_predicted,
                "extracted_ingredients": "; ".join(p.extracted_ingredients),
                "pred_calories":         p.calories_kcal,
                "pred_protein":          p.protein_g,
                "pred_carbs":            p.carbs_g,
                "pred_fat":              p.fat_g,
                "pred_fiber":            p.fiber_g,
                "confidence":            p.confidence,
                "latency_seconds":       p.latency_seconds,
            })
        if r.judge is not None:
            base["judge_score"] = r.judge.score
        rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Image → Ingredients runner
# ---------------------------------------------------------------------------

def run_image_ingredients_eval(
    golden_csv: Path,
    images_dir: Path,
    client: ImageEvalClient,
    *,
    artifacts_dir: Path = _ARTIFACTS_DIR,
    run_judge: bool = False,
) -> tuple[list[ImageEvalRecord], dict[str, Any]]:
    """
    Evaluate how accurately the model identifies ingredients from images.

    Parameters
    ----------
    golden_csv:    Path to the 200-row golden set CSV.
    images_dir:    Directory containing images named ``1.jpg``…``200.jpg``.
    client:        Configured :class:`ImageEvalClient`.
    artifacts_dir: Where to write ``image_ingredients_records.csv`` and
                   ``image_ingredients_metrics.json``.
    run_judge:     Also call the LLM judge (adds cost / latency).

    Returns
    -------
    (records, metrics_dict)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dishes = _load_golden_set(golden_csv)
    logger.info(
        "Image → Ingredients eval: %d dishes, images_dir=%s",
        len(dishes), images_dir,
    )

    records: list[ImageEvalRecord] = []
    for dish in dishes:
        img_num   = _dish_id_to_image_num(dish.dish_id)
        img_path  = _find_image(images_dir, img_num)

        if img_path is None:
            logger.warning(
                "No image found for %s (number %d) — skipping.",
                dish.dish_id, img_num,
            )
            records.append(
                ImageEvalRecord(
                    golden=dish,
                    image_path=str(images_dir / f"{img_num}.jpg"),
                    success=False,
                    error_message="Image file not found",
                )
            )
            continue

        logger.debug("Analyzing %s → %s", dish.dish_id, img_path.name)
        try:
            prediction = client.analyze_image(dish, img_path)
        except ImageEvalClientError as exc:
            logger.error("analyze_image failed for %s: %s", dish.dish_id, exc)
            records.append(
                ImageEvalRecord(
                    golden=dish,
                    image_path=str(img_path),
                    success=False,
                    error_message=str(exc),
                )
            )
            continue

        judge_result: JudgeResult | None = None
        if run_judge:
            try:
                judge_result = client.judge_prediction(dish, prediction)
            except ImageEvalClientError as exc:
                logger.warning("Judge failed for %s: %s", dish.dish_id, exc)

        records.append(
            ImageEvalRecord(
                golden=dish,
                image_path=str(img_path),
                prediction=prediction,
                judge=judge_result,
                success=True,
            )
        )
        logger.info(
            "  %s OK — %d ingredients predicted, latency=%.2fs",
            dish.dish_id, len(prediction.extracted_ingredients),
            prediction.latency_seconds,
        )

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    n_total    = len(records)
    n_success  = sum(1 for r in records if r.success)
    coverage   = n_success / n_total if n_total else 0.0

    recall     = compute_ingredient_recall(records)
    precision  = compute_ingredient_precision(records)
    f1         = compute_ingredient_f1(records)
    dish_name  = compute_dish_name_accuracy(records)
    latencies  = [
        r.prediction.latency_seconds
        for r in records
        if r.success and r.prediction is not None
    ]
    latency_stats = _latency_stats(latencies)

    metrics: dict[str, Any] = {
        "eval_type":         "image_ingredients",
        "n_total":           n_total,
        "n_success":         n_success,
        "coverage":          coverage,
        "ingredient_recall": recall,
        "ingredient_precision": precision,
        "ingredient_f1":     f1,
        "dish_name_accuracy": dish_name,
        "latency":           latency_stats,
    }
    if run_judge:
        judge_scores = [
            r.judge.score for r in records if r.success and r.judge is not None
        ]
        metrics["judge"] = _judge_stats(judge_scores)

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    df = _records_to_df(records)
    records_path = artifacts_dir / "image_ingredients_records.csv"
    metrics_path = artifacts_dir / "image_ingredients_metrics.json"
    df.to_csv(records_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=_json_safe), encoding="utf-8"
    )
    logger.info("Artifacts saved → %s, %s", records_path, metrics_path)

    return records, metrics


# ---------------------------------------------------------------------------
# Full Pipeline (Image → Macros) runner
# ---------------------------------------------------------------------------

def run_full_pipeline_eval(
    golden_csv: Path,
    images_dir: Path,
    client: ImageEvalClient,
    *,
    artifacts_dir: Path = _ARTIFACTS_DIR,
    run_judge: bool = True,
) -> tuple[list[ImageEvalRecord], dict[str, Any]]:
    """
    Evaluate the full image-to-macros pipeline accuracy.

    Uses the same backend call as :func:`run_image_ingredients_eval` but
    computes macro-level metrics (MAE, MAPE, coverage, latency, judge score).

    Parameters
    ----------
    golden_csv:    Path to the 200-row golden set CSV.
    images_dir:    Directory containing images named ``1.jpg``…``200.jpg``.
    client:        Configured :class:`ImageEvalClient`.
    artifacts_dir: Where to write ``full_pipeline_records.csv`` and
                   ``full_pipeline_metrics.json``.
    run_judge:     Also call the LLM judge (default True).

    Returns
    -------
    (records, metrics_dict)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dishes = _load_golden_set(golden_csv)
    logger.info(
        "Full pipeline eval: %d dishes, images_dir=%s",
        len(dishes), images_dir,
    )

    records: list[ImageEvalRecord] = []
    for dish in dishes:
        img_num  = _dish_id_to_image_num(dish.dish_id)
        img_path = _find_image(images_dir, img_num)

        if img_path is None:
            logger.warning("No image for %s — skipping.", dish.dish_id)
            records.append(
                ImageEvalRecord(
                    golden=dish,
                    image_path=str(images_dir / f"{img_num}.jpg"),
                    success=False,
                    error_message="Image file not found",
                )
            )
            continue

        logger.debug("Pipeline: %s → %s", dish.dish_id, img_path.name)
        try:
            prediction = client.analyze_image(dish, img_path)
        except ImageEvalClientError as exc:
            logger.error("analyze_image failed for %s: %s", dish.dish_id, exc)
            records.append(
                ImageEvalRecord(
                    golden=dish,
                    image_path=str(img_path),
                    success=False,
                    error_message=str(exc),
                )
            )
            continue

        judge_result: JudgeResult | None = None
        if run_judge:
            try:
                judge_result = client.judge_prediction(dish, prediction)
            except ImageEvalClientError as exc:
                logger.warning("Judge failed for %s: %s", dish.dish_id, exc)

        records.append(
            ImageEvalRecord(
                golden=dish,
                image_path=str(img_path),
                prediction=prediction,
                judge=judge_result,
                success=True,
            )
        )
        logger.info(
            "  %s OK — cal=%.0f (golden=%.0f), latency=%.2fs",
            dish.dish_id, prediction.calories_kcal,
            dish.calories_kcal, prediction.latency_seconds,
        )

    # ------------------------------------------------------------------
    # Compute metrics using src.eval.metrics helpers
    # ------------------------------------------------------------------
    lists = records_to_macro_lists(records)
    n_total   = len(records)
    n_success = sum(1 for r in records if r.success)
    coverage  = n_success / n_total if n_total else 0.0

    def _mae(actual: list[float], pred: list[float]) -> float:
        if not actual:
            return math.nan
        return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)

    def _mape(actual: list[float], pred: list[float]) -> float:
        pairs = [(a, p) for a, p in zip(actual, pred) if a != 0]
        if not pairs:
            return math.nan
        return sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs) * 100

    def _latency_pct(lats: list[float]) -> dict[str, float]:
        if not lats:
            return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
        s = sorted(lats)
        n = len(s)
        def pct(q: float) -> float:
            idx = q * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return s[lo] + (s[hi] - s[lo]) * (idx - lo)
        return {
            "mean": sum(s) / n,
            "p50":  pct(0.50),
            "p90":  pct(0.90),
            "p99":  pct(0.99),
        }

    metrics: dict[str, Any] = {
        "eval_type":   "full_pipeline",
        "n_total":     n_total,
        "n_success":   n_success,
        "coverage":    coverage,
        "mae": {
            "calories_kcal": _mae(lists["actual_calories"], lists["pred_calories"]),
            "protein_g":     _mae(lists["actual_protein"],  lists["pred_protein"]),
            "carbs_g":       _mae(lists["actual_carbs"],    lists["pred_carbs"]),
            "fat_g":         _mae(lists["actual_fat"],      lists["pred_fat"]),
            "fiber_g":       _mae(lists["actual_fiber"],    lists["pred_fiber"]),
        },
        "mape_calories_pct": _mape(lists["actual_calories"], lists["pred_calories"]),
        "latency": _latency_pct(lists["latencies"]),
    }

    if run_judge and lists["judge_scores"]:
        j = lists["judge_scores"]
        metrics["judge"] = {
            "mean_score":   sum(j) / len(j),
            "min_score":    min(j),
            "max_score":    max(j),
            "n_judged":     len(j),
        }

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    df = _records_to_df(records)
    records_path = artifacts_dir / "full_pipeline_records.csv"
    metrics_path = artifacts_dir / "full_pipeline_metrics.json"
    df.to_csv(records_path, index=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=_json_safe), encoding="utf-8"
    )
    logger.info("Artifacts saved → %s, %s", records_path, metrics_path)

    return records, metrics


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _latency_stats(lats: list[float]) -> dict[str, float]:
    if not lats:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
    s = sorted(lats)
    n = len(s)
    def pct(q: float) -> float:
        idx = q * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return {"mean": sum(s) / n, "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}


def _judge_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"mean_score": math.nan, "n_judged": 0}
    return {
        "mean_score": sum(scores) / len(scores),
        "min_score":  min(scores),
        "max_score":  max(scores),
        "n_judged":   len(scores),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
