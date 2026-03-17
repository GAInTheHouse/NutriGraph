"""
Pydantic models for the NutriGraph evaluation framework.

GoldenDish    — one row from the golden set CSV
PredictionResult — mapped response from the backend analyze endpoint
JudgeResult   — response from the judge endpoint
EvalRecord    — pairs a golden dish with its prediction and judge result
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GoldenDish(BaseModel):
    """One row from the golden set CSV, with all 14 ground-truth columns."""

    dish_id: str
    dish_name: str
    context_type: str
    cuisine: str
    serving_description: str
    serving_size_grams: float
    ingredients_list: str
    preparation_notes: str
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    source_confidence: str


class PredictionResult(BaseModel):
    """
    Nutritional prediction returned by the backend after the full
    multi-turn clarification exchange.

    ``num_questions`` is the total count of clarification questions asked
    across all rounds.  ``latency_seconds`` is the wall-clock time for the
    entire exchange (first POST → final complete response).
    """

    dish_id: str
    dish_name: str
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    confidence: float = Field(ge=0.0, le=1.0, description="Normalized to [0, 1]")
    num_questions: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)
    raw_response: dict[str, Any] | None = None


class JudgeResult(BaseModel):
    """Score and explanation from the LLM-as-a-judge endpoint."""

    score: float = Field(ge=0.0, le=10.0)
    explanation: str


class EvalRecord(BaseModel):
    """
    A single evaluation record binding the ground-truth dish to its
    prediction and judge score.

    ``success`` is False when any step in the pipeline (analysis or judging)
    raised an exception; in that case ``prediction`` and ``judge`` are None
    and ``error_message`` contains a description of the failure.
    """

    golden: GoldenDish
    prediction: PredictionResult | None = None
    judge: JudgeResult | None = None
    success: bool
    error_message: str | None = None
