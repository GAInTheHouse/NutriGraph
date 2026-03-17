"""
Pydantic models for the NutriGraph image-based evaluation frameworks.

Used by both:
  - Image → Ingredients eval  (compares predicted ingredient names)
  - Full pipeline eval        (compares predicted macro totals)

Both evaluations call the same backend endpoint
(POST /api/v1/analyze-dish) so they share a single prediction model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.eval.models import GoldenDish, JudgeResult


class ImagePredictionResult(BaseModel):
    """
    Nutritional prediction returned by the full-image pipeline
    (POST /api/v1/analyze-dish).

    ``extracted_ingredients`` holds the raw ingredient names the LLM
    identified from the photo — used for image→ingredients metrics.
    The ``*_kcal / *_g`` fields are the ChromaDB-computed macro totals
    — used for full-pipeline metrics.
    """

    dish_id: str
    dish_name_predicted: str
    extracted_ingredients: list[str]        # ingredient names as seen by the LLM
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float = 0.0                    # ChromaDB does not index fiber
    confidence: float = Field(ge=0.0, le=1.0)
    latency_seconds: float = Field(ge=0.0)
    raw_response: dict[str, Any] | None = None


class ImageEvalRecord(BaseModel):
    """
    A single image evaluation record.

    ``success`` is False when the backend call failed or the image was not
    found; in that case ``prediction`` and ``judge`` are None.
    """

    golden: GoldenDish
    image_path: str                         # absolute path used for the call
    prediction: ImagePredictionResult | None = None
    judge: JudgeResult | None = None
    success: bool
    error_message: str | None = None
