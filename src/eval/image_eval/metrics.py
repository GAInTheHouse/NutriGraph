"""
Metric computation for the two image-based evaluation frameworks.

Image → Ingredients metrics
----------------------------
compute_ingredient_recall   — what fraction of golden ingredients are found
compute_ingredient_precision — what fraction of predicted ingredients are golden
compute_ingredient_f1       — harmonic mean of the above
compute_dish_name_accuracy  — exact-match and fuzzy-match fractions

Full Pipeline (Image → Macros) metrics
---------------------------------------
Delegates to src.eval.metrics for MAE, MAPE, coverage, judge stats, and
latency; the helpers here only adapt the ImageEvalRecord format.

Ingredient matching uses token-Jaccard overlap (same algorithm as the
clarification graph), with a configurable threshold (default 0.30).
"""
from __future__ import annotations

import math
import re
from typing import Sequence

from .models import ImageEvalRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QTY_PATTERN = re.compile(
    r"""
    (?<!\w)              # not preceded by a word char — avoids matching
                         # numbers inside tokens like "B12" or "7-spice"
    \d+(?:\.\d+)?        # number (e.g. 150, 0.5)
    \s*                  # optional space between number and unit
    (?:ml|g|oz|lb|cups?|tbsp|tsp|pieces?|pcs?|slices?|cloves?)
                         # unit is now REQUIRED so bare numbers ("7", "12")
                         # that are part of an ingredient name are preserved
    \b                   # word boundary — prevents "g" matching mid-word
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SPLIT_PATTERN = re.compile(r"[;,\n]+")


def _parse_ingredient_tokens(ingredients_list: str) -> list[set[str]]:
    """
    Parse the semicolon-separated golden ingredient string into a list of
    token sets, one per ingredient (quantities stripped).

    Example
    -------
    "romaine lettuce 150g; grilled chicken breast 120g"
    → [{"romaine", "lettuce"}, {"grilled", "chicken", "breast"}]
    """
    result: list[set[str]] = []
    for part in _SPLIT_PATTERN.split(ingredients_list):
        cleaned = _QTY_PATTERN.sub("", part).lower().strip()
        tokens  = {t for t in re.split(r"\s+", cleaned) if len(t) > 1}
        if tokens:
            result.append(tokens)
    return result


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ingredient_matches(
    golden_token_sets: list[set[str]],
    predicted_names: list[str],
    threshold: float = 0.30,
) -> tuple[int, int, int]:
    """
    Return (matched_golden, matched_predicted, len(predicted)) counts.

    An ingredient is considered matched if its best Jaccard overlap with
    any counterpart exceeds *threshold*.
    """
    pred_token_sets = [
        {t for t in re.split(r"\s+", p.lower().strip()) if len(t) > 1}
        for p in predicted_names
    ]

    matched_golden = 0
    for g_toks in golden_token_sets:
        best = max(
            (_jaccard(g_toks, p_toks) for p_toks in pred_token_sets),
            default=0.0,
        )
        if best >= threshold:
            matched_golden += 1

    matched_pred = 0
    for p_toks in pred_token_sets:
        best = max(
            (_jaccard(p_toks, g_toks) for g_toks in golden_token_sets),
            default=0.0,
        )
        if best >= threshold:
            matched_pred += 1

    return matched_golden, matched_pred, len(pred_token_sets)


# ---------------------------------------------------------------------------
# Image → Ingredients metrics
# ---------------------------------------------------------------------------

def compute_ingredient_recall(
    records: Sequence[ImageEvalRecord],
    threshold: float = 0.30,
) -> dict[str, float]:
    """
    Mean ingredient recall across all successful records.
    Recall = matched_golden / total_golden_ingredients.
    """
    scores: list[float] = []
    for r in records:
        if not r.success or r.prediction is None:
            continue
        golden_sets = _parse_ingredient_tokens(r.golden.ingredients_list)
        if not golden_sets:
            continue
        matched_g, _, _ = _ingredient_matches(
            golden_sets,
            r.prediction.extracted_ingredients,
            threshold,
        )
        scores.append(matched_g / len(golden_sets))

    if not scores:
        return {"mean_recall": math.nan, "n": 0}
    return {
        "mean_recall": sum(scores) / len(scores),
        "n": len(scores),
    }


def compute_ingredient_precision(
    records: Sequence[ImageEvalRecord],
    threshold: float = 0.30,
) -> dict[str, float]:
    """
    Mean ingredient precision across all successful records.
    Precision = matched_predicted / total_predicted_ingredients.
    Returns NaN when the model predicted zero ingredients.
    """
    scores: list[float] = []
    for r in records:
        if not r.success or r.prediction is None:
            continue
        predicted = r.prediction.extracted_ingredients
        if not predicted:
            scores.append(0.0)
            continue
        golden_sets = _parse_ingredient_tokens(r.golden.ingredients_list)
        _, matched_p, n_pred = _ingredient_matches(
            golden_sets,
            predicted,
            threshold,
        )
        scores.append(matched_p / n_pred if n_pred else 0.0)

    if not scores:
        return {"mean_precision": math.nan, "n": 0}
    return {
        "mean_precision": sum(scores) / len(scores),
        "n": len(scores),
    }


def compute_ingredient_f1(
    records: Sequence[ImageEvalRecord],
    threshold: float = 0.30,
) -> dict[str, float]:
    """
    Mean ingredient F1 across all successful records.
    F1 = 2·P·R / (P+R), or 0 when both P and R are zero.
    """
    scores: list[float] = []
    for r in records:
        if not r.success or r.prediction is None:
            continue
        golden_sets = _parse_ingredient_tokens(r.golden.ingredients_list)
        if not golden_sets:
            continue
        predicted = r.prediction.extracted_ingredients
        if not predicted:
            scores.append(0.0)
            continue
        matched_g, matched_p, n_pred = _ingredient_matches(
            golden_sets, predicted, threshold
        )
        n_golden = len(golden_sets)
        recall    = matched_g / n_golden if n_golden else 0.0
        precision = matched_p / n_pred   if n_pred   else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    if not scores:
        return {"mean_f1": math.nan, "n": 0}
    return {
        "mean_f1": sum(scores) / len(scores),
        "n": len(scores),
    }


def compute_dish_name_accuracy(
    records: Sequence[ImageEvalRecord],
    fuzzy_threshold: float = 0.40,
) -> dict[str, float]:
    """
    Dish-name accuracy (two variants).

    exact_match_rate  — case-insensitive exact match
    fuzzy_match_rate  — token-Jaccard ≥ fuzzy_threshold
    """
    exact_hits = 0
    fuzzy_hits = 0
    n = 0
    for r in records:
        if not r.success or r.prediction is None:
            continue
        n += 1
        golden_toks = {
            t for t in r.golden.dish_name.lower().split() if len(t) > 1
        }
        pred_toks = {
            t
            for t in r.prediction.dish_name_predicted.lower().split()
            if len(t) > 1
        }
        if r.golden.dish_name.strip().lower() == r.prediction.dish_name_predicted.strip().lower():
            exact_hits += 1
        if _jaccard(golden_toks, pred_toks) >= fuzzy_threshold:
            fuzzy_hits += 1

    if n == 0:
        return {"exact_match_rate": math.nan, "fuzzy_match_rate": math.nan, "n": 0}
    return {
        "exact_match_rate": exact_hits / n,
        "fuzzy_match_rate": fuzzy_hits / n,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Full Pipeline (Image → Macros) metric adapters
# ---------------------------------------------------------------------------
# These thin wrappers convert ImageEvalRecord lists into the flat format
# expected by src.eval.metrics functions.

def records_to_macro_lists(
    records: Sequence[ImageEvalRecord],
) -> dict[str, list[float]]:
    """
    Extract parallel lists of (golden, predicted) values for each macro
    from successful records — for use with src.eval.metrics functions.

    Returns
    -------
    dict with keys:
      actual_calories, pred_calories,
      actual_protein,  pred_protein,
      actual_carbs,    pred_carbs,
      actual_fat,      pred_fat,
      actual_fiber,    pred_fiber,
      latencies,
      judge_scores
    """
    out: dict[str, list[float]] = {k: [] for k in (
        "actual_calories", "pred_calories",
        "actual_protein",  "pred_protein",
        "actual_carbs",    "pred_carbs",
        "actual_fat",      "pred_fat",
        "actual_fiber",    "pred_fiber",
        "latencies",
        "judge_scores",
    )}
    for r in records:
        if not r.success or r.prediction is None:
            continue
        p = r.prediction
        g = r.golden
        out["actual_calories"].append(g.calories_kcal)
        out["pred_calories"].append(p.calories_kcal)
        out["actual_protein"].append(g.protein_g)
        out["pred_protein"].append(p.protein_g)
        out["actual_carbs"].append(g.carbs_g)
        out["pred_carbs"].append(p.carbs_g)
        out["actual_fat"].append(g.fat_g)
        out["pred_fat"].append(p.fat_g)
        out["actual_fiber"].append(g.fiber_g)
        out["pred_fiber"].append(p.fiber_g)
        out["latencies"].append(p.latency_seconds)
        if r.judge is not None:
            out["judge_scores"].append(r.judge.score)
    return out
