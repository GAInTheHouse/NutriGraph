"""
Evaluation runner for the NutriGraph framework.

run_evaluation() orchestrates the full evaluation loop:
  1. Load and parse the golden set CSV.
  2. For each dish (× repeats), call the analysis endpoint then the judge.
  3. Wrap every outcome in an EvalRecord (success or failure).
  4. Compute all metrics.
  5. Persist artifacts to disk.
  6. Return (records, summary_metrics).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .client import NutriGraphEvalClient
from .metrics import (
    compute_agent_efficiency,
    compute_confidence_calibration,
    compute_consistency,
    compute_coverage,
    compute_judge_stats,
    compute_latency_stats,
    compute_mae,
    compute_mape,
)
from .models import EvalRecord, GoldenDish

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV → GoldenDish
# ---------------------------------------------------------------------------

def _load_golden_set(csv_path: str | Path) -> list[GoldenDish]:
    """Read the golden set CSV and convert every row to a GoldenDish."""
    df = pd.read_csv(csv_path, dtype=str)

    numeric_cols = {
        "serving_size_grams": float,
        "calories_kcal": float,
        "protein_g": float,
        "carbs_g": float,
        "fat_g": float,
        "fiber_g": float,
    }

    dishes: list[GoldenDish] = []
    for _, row in df.iterrows():
        data: dict[str, Any] = row.to_dict()
        for col, cast in numeric_cols.items():
            raw = data.get(col, "")
            try:
                data[col] = cast(raw)
            except (ValueError, TypeError):
                data[col] = 0.0
                logger.warning("Could not parse %s='%s' for dish %s; defaulting to 0.", col, raw, data.get("dish_id"))

        # Fill any missing string fields with empty string
        for col in GoldenDish.model_fields:
            if col not in data or pd.isna(data[col]):
                data[col] = ""

        dishes.append(GoldenDish(**data))

    logger.info("Loaded %d dishes from %s.", len(dishes), csv_path)
    return dishes


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _records_to_df(records: list[EvalRecord]) -> pd.DataFrame:
    """Flatten EvalRecords into a tidy DataFrame for CSV export."""
    rows = []
    for rec in records:
        g = rec.golden
        p = rec.prediction
        j = rec.judge
        row: dict[str, Any] = {
            "dish_id": g.dish_id,
            "dish_name": g.dish_name,
            "success": rec.success,
            "error_message": rec.error_message,
            # Ground truth
            "gt_calories": g.calories_kcal,
            "gt_protein": g.protein_g,
            "gt_carbs": g.carbs_g,
            "gt_fat": g.fat_g,
            "gt_fiber": g.fiber_g,
            # Prediction
            "pred_calories": p.calories_kcal if p else None,
            "pred_protein": p.protein_g if p else None,
            "pred_carbs": p.carbs_g if p else None,
            "pred_fat": p.fat_g if p else None,
            "pred_fiber": p.fiber_g if p else None,
            "confidence": p.confidence if p else None,
            "num_questions": p.num_questions if p else None,
            "latency_seconds": p.latency_seconds if p else None,
            # Judge
            "judge_score": j.score if j else None,
            "judge_explanation": j.explanation if j else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _save_artifacts(
    records: list[EvalRecord],
    summary: dict[str, Any],
    artifacts_dir: Path,
) -> None:
    """Persist eval_records.csv and summary_metrics.json."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "eval_records.csv"
    _records_to_df(records).to_csv(csv_path, index=False)
    logger.info("Saved eval records → %s", csv_path)

    json_path = artifacts_dir / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Saved summary metrics → %s", json_path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(
    golden_csv: str | Path,
    client: NutriGraphEvalClient,
    repeats: int = 1,
    artifacts_dir: str | Path = "artifacts",
) -> tuple[list[EvalRecord], dict[str, Any]]:
    """
    Run the full offline evaluation loop.

    Parameters
    ----------
    golden_csv:
        Path to ``data/golden_set.csv``.
    client:
        Instantiated ``NutriGraphEvalClient``.
    repeats:
        Number of times to evaluate each dish (use > 1 for consistency analysis).
    artifacts_dir:
        Directory where ``eval_records.csv`` and ``summary_metrics.json``
        are written.

    Returns
    -------
    tuple[list[EvalRecord], dict[str, Any]]
        All evaluation records and the aggregated summary metrics dict.
    """
    dishes = _load_golden_set(golden_csv)
    artifacts_path = Path(artifacts_dir)
    records: list[EvalRecord] = []

    total = len(dishes) * repeats
    logger.info(
        "Starting evaluation: %d dish(es) × %d repeat(s) = %d calls.",
        len(dishes),
        repeats,
        total,
    )

    for repeat_idx in range(repeats):
        for dish_idx, golden in enumerate(dishes):
            label = (
                f"[repeat {repeat_idx + 1}/{repeats}] "
                f"dish {dish_idx + 1}/{len(dishes)} ({golden.dish_id})"
            )
            logger.info("Evaluating %s …", label)

            prediction = None
            judge = None
            error_message: str | None = None
            success = False

            try:
                prediction = client.analyze_dish(golden)
                judge = client.judge_prediction(golden, prediction)
                success = True
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                logger.warning("Failed %s: %s", label, exc)

            records.append(
                EvalRecord(
                    golden=golden,
                    prediction=prediction,
                    judge=judge,
                    success=success,
                    error_message=error_message,
                )
            )

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    logger.info("Computing metrics over %d record(s) …", len(records))

    summary: dict[str, Any] = {
        "total_records": len(records),
        "successful_records": sum(1 for r in records if r.success),
        "coverage": compute_coverage(records),
        "mae": compute_mae(records),
        "mape": compute_mape(records),
        "agent_efficiency": compute_agent_efficiency(records),
        "judge_stats": compute_judge_stats(records),
        "consistency": compute_consistency(records),
        "latency_stats": compute_latency_stats(records),
        "confidence_calibration": compute_confidence_calibration(records),
    }

    _save_artifacts(records, summary, artifacts_path)
    return records, summary
