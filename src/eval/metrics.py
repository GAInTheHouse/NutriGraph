"""
Metric computation functions for the NutriGraph evaluation framework.

All functions accept ``list[EvalRecord]`` and operate only on the subset
of records where ``success=True`` (and ``prediction`` / ``judge`` are
not None) unless otherwise noted.

Functions
---------
compute_mae                 — Mean Absolute Error for all five macros
compute_mape                — Mean Absolute Percentage Error for calories
compute_agent_efficiency    — Statistics on clarification questions asked
compute_coverage            — Fraction of dishes processed successfully
compute_judge_stats         — Aggregated LLM-as-a-judge scores
compute_consistency         — Cross-repeat variance of calorie predictions
compute_latency_stats       — Percentile latency over successful records
compute_confidence_calibration — MAE per confidence bin (optional)
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .models import EvalRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _successful(records: list[EvalRecord]) -> list[EvalRecord]:
    """Return only records whose pipeline completed without error."""
    return [r for r in records if r.success and r.prediction is not None]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_values:
        return float("nan")
    n = len(sorted_values)
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ---------------------------------------------------------------------------
# 1. MAE
# ---------------------------------------------------------------------------

def compute_mae(records: list[EvalRecord]) -> dict[str, float]:
    """
    Compute Mean Absolute Error for calories, protein, carbs, fat, and fiber.

    Returns a dict with keys matching the GoldenDish / PredictionResult
    field names: ``calories_kcal``, ``protein_g``, ``carbs_g``, ``fat_g``,
    ``fiber_g``.  Returns NaN for any nutrient if no successful records exist.
    """
    fields = ("calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")
    errors: dict[str, list[float]] = {f: [] for f in fields}

    for rec in _successful(records):
        p = rec.prediction
        g = rec.golden
        assert p is not None  # narrowing for type checker
        errors["calories_kcal"].append(abs(p.calories_kcal - g.calories_kcal))
        errors["protein_g"].append(abs(p.protein_g - g.protein_g))
        errors["carbs_g"].append(abs(p.carbs_g - g.carbs_g))
        errors["fat_g"].append(abs(p.fat_g - g.fat_g))
        errors["fiber_g"].append(abs(p.fiber_g - g.fiber_g))

    return {
        f: (statistics.mean(v) if v else float("nan"))
        for f, v in errors.items()
    }


# ---------------------------------------------------------------------------
# 2. MAPE
# ---------------------------------------------------------------------------

def compute_mape(records: list[EvalRecord]) -> dict[str, float]:
    """
    Compute Mean Absolute Percentage Error for macros where ground truth > 0.

    Primary key is ``calories_kcal``; additional keys are provided for the
    other macros when computable.
    """
    fields = ("calories_kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")
    pct_errors: dict[str, list[float]] = {f: [] for f in fields}

    for rec in _successful(records):
        p = rec.prediction
        g = rec.golden
        assert p is not None

        pairs: list[tuple[str, float, float]] = [
            ("calories_kcal", p.calories_kcal, g.calories_kcal),
            ("protein_g", p.protein_g, g.protein_g),
            ("carbs_g", p.carbs_g, g.carbs_g),
            ("fat_g", p.fat_g, g.fat_g),
            ("fiber_g", p.fiber_g, g.fiber_g),
        ]
        for field, pred_val, gt_val in pairs:
            if gt_val > 0:
                pct_errors[field].append(abs(pred_val - gt_val) / gt_val * 100.0)

    return {
        f: (statistics.mean(v) if v else float("nan"))
        for f, v in pct_errors.items()
    }


# ---------------------------------------------------------------------------
# 3. Agent efficiency
# ---------------------------------------------------------------------------

def compute_agent_efficiency(records: list[EvalRecord]) -> dict[str, Any]:
    """
    Aggregate statistics on clarification questions asked per dish.

    Keys in the returned dict
    -------------------------
    mean_questions : float
    median_questions : float
    hist : dict  — counts for exactly 0 / 1 / 2 / 3+ questions
    high_confidence_zero_q_fraction : float
        Fraction of successful records where ``confidence >= 0.5``
        and ``num_questions == 0`` (agent resolved without asking anything).
    """
    ok = _successful(records)
    if not ok:
        return {
            "mean_questions": float("nan"),
            "median_questions": float("nan"),
            "hist": {"0": 0, "1": 0, "2": 0, "3+": 0},
            "high_confidence_zero_q_fraction": float("nan"),
        }

    counts = [r.prediction.num_questions for r in ok]  # type: ignore[union-attr]

    hist: dict[str, int] = {"0": 0, "1": 0, "2": 0, "3+": 0}
    high_conf_zero = 0
    for rec, nq in zip(ok, counts):
        if nq == 0:
            hist["0"] += 1
        elif nq == 1:
            hist["1"] += 1
        elif nq == 2:
            hist["2"] += 1
        else:
            hist["3+"] += 1

        assert rec.prediction is not None
        if rec.prediction.confidence >= 0.5 and nq == 0:
            high_conf_zero += 1

    return {
        "mean_questions": statistics.mean(counts),
        "median_questions": statistics.median(counts),
        "hist": hist,
        "high_confidence_zero_q_fraction": high_conf_zero / len(ok),
    }


# ---------------------------------------------------------------------------
# 4. Coverage
# ---------------------------------------------------------------------------

def compute_coverage(records: list[EvalRecord]) -> float:
    """Return the fraction of records that completed successfully."""
    if not records:
        return float("nan")
    return len(_successful(records)) / len(records)


# ---------------------------------------------------------------------------
# 5. Judge statistics
# ---------------------------------------------------------------------------

def compute_judge_stats(records: list[EvalRecord]) -> dict[str, Any]:
    """
    Aggregate LLM-as-a-judge scores (0–10) over successful records that
    also have a judge result.

    Keys: ``mean``, ``median``, ``std``, ``buckets`` (counts per band).
    """
    scored = [
        r for r in records
        if r.success and r.judge is not None
    ]
    if not scored:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "buckets": {"0-3": 0, "3-7": 0, "7-10": 0},
        }

    scores = [r.judge.score for r in scored]  # type: ignore[union-attr]
    buckets = {"0-3": 0, "3-7": 0, "7-10": 0}
    for s in scores:
        if s < 3:
            buckets["0-3"] += 1
        elif s < 7:
            buckets["3-7"] += 1
        else:
            buckets["7-10"] += 1

    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "buckets": buckets,
    }


# ---------------------------------------------------------------------------
# 6. Consistency
# ---------------------------------------------------------------------------

def compute_consistency(records: list[EvalRecord]) -> dict[str, Any]:
    """
    Measure prediction variance across repeated evaluations of the same dish.

    Records are grouped by ``golden.dish_id``.  For each dish that has more
    than one successful prediction, variance of ``calories_kcal`` is computed.

    Keys: ``mean_variance``, ``high_variance_fraction``.
    ``high_variance_fraction`` is the fraction of multi-repeat dishes where
    variance exceeds 5 % of the ground-truth calorie value.

    Note: This metric is only meaningful when the runner is called with
    ``repeats > 1``.  With a single repeat every dish has variance = 0.
    """
    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for rec in _successful(records):
        grouped[rec.golden.dish_id].append(rec)

    variances: list[float] = []
    high_var_count = 0
    multi_count = 0

    for dish_id, recs in grouped.items():
        if len(recs) < 2:
            continue
        multi_count += 1
        cal_preds = [r.prediction.calories_kcal for r in recs]  # type: ignore[union-attr]
        var = statistics.variance(cal_preds)
        variances.append(var)
        gt_cal = recs[0].golden.calories_kcal
        threshold = (gt_cal * 0.05) ** 2  # 5 % of GT, squared to compare with variance
        if var > threshold:
            high_var_count += 1

    if not variances:
        return {
            "mean_variance": float("nan"),
            "high_variance_fraction": float("nan"),
            "note": "Requires repeats > 1 for meaningful results.",
        }

    return {
        "mean_variance": statistics.mean(variances),
        "high_variance_fraction": high_var_count / multi_count,
        "note": f"Based on {multi_count} dish(es) with multiple repeats.",
    }


# ---------------------------------------------------------------------------
# 7. Latency statistics
# ---------------------------------------------------------------------------

def compute_latency_stats(records: list[EvalRecord]) -> dict[str, float]:
    """
    Compute percentile latency statistics over successful records.

    Keys: ``mean``, ``median``, ``p90``, ``p95`` (all in seconds).
    Latency covers the entire multi-turn exchange.
    """
    latencies = sorted(
        r.prediction.latency_seconds  # type: ignore[union-attr]
        for r in _successful(records)
    )
    if not latencies:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }

    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p90": _percentile(latencies, 90),
        "p95": _percentile(latencies, 95),
    }


# ---------------------------------------------------------------------------
# 8. Confidence calibration (optional)
# ---------------------------------------------------------------------------

def compute_confidence_calibration(records: list[EvalRecord]) -> dict[str, Any]:
    """
    Bin records by predicted confidence and compute calorie MAE per bin.

    Bins: [0, 0.3), [0.3, 0.6), [0.6, 1.0].

    Keys: ``bin_0_03``, ``bin_03_06``, ``bin_06_10`` — each containing
    ``mae_calories`` and ``count``.
    """
    bins: dict[str, list[float]] = {
        "bin_0_03": [],
        "bin_03_06": [],
        "bin_06_10": [],
    }

    for rec in _successful(records):
        p = rec.prediction
        assert p is not None
        err = abs(p.calories_kcal - rec.golden.calories_kcal)
        if p.confidence < 0.3:
            bins["bin_0_03"].append(err)
        elif p.confidence < 0.6:
            bins["bin_03_06"].append(err)
        else:
            bins["bin_06_10"].append(err)

    result: dict[str, Any] = {}
    labels = {
        "bin_0_03": "[0, 0.3)",
        "bin_03_06": "[0.3, 0.6)",
        "bin_06_10": "[0.6, 1.0]",
    }
    for key, errs in bins.items():
        result[key] = {
            "range": labels[key],
            "count": len(errs),
            "mae_calories": statistics.mean(errs) if errs else float("nan"),
        }

    return result
