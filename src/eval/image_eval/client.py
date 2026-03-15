"""
HTTP client for the NutriGraph image-based evaluation frameworks.

ImageEvalClient wraps two backend endpoints:

1. POST /api/v1/analyze-dish   (multipart/form-data image upload)
   Returns dish name, per-ingredient breakdown, and macro totals.

2. POST /api/v1/judge-nutrition
   Same judge endpoint reused from the ingredient-list eval.

All HTTP errors and timeouts are converted to ImageEvalClientError so that
the evaluation loop never crashes on a single bad image.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from src.eval.models import GoldenDish, JudgeResult
from .models import ImagePredictionResult

logger = logging.getLogger(__name__)

_ANALYZE_PATH = "/api/v1/analyze-dish"
_JUDGE_PATH   = "/api/v1/judge-nutrition"


class ImageEvalClientError(Exception):
    """Raised when the backend returns an error or the request times out."""


class ImageEvalClient:
    """
    HTTP client for image-based evaluation.

    Parameters
    ----------
    api_base_url:
        Base URL of the NutriGraph backend, e.g. ``http://localhost:8000``.
    judge_base_url:
        Base URL of the judge backend (may be the same).
    timeout:
        Per-request HTTP timeout in seconds (default 120 — images are larger).
    """

    def __init__(
        self,
        api_base_url: str,
        judge_base_url: str,
        timeout: int = 120,
    ) -> None:
        self._api_base   = api_base_url.rstrip("/")
        self._judge_base = judge_base_url.rstrip("/")
        self._timeout    = timeout
        self._session    = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        golden: GoldenDish,
        image_path: Path,
    ) -> ImagePredictionResult:
        """
        Upload ``image_path`` to the analyze endpoint and return the
        structured prediction result.

        The call is intentionally blind — no ``dish_name`` or
        ``restaurant_context`` hints are supplied — so the model must
        identify everything from the image alone.
        """
        if not image_path.exists():
            raise ImageEvalClientError(
                f"Image not found: {image_path}"
            )

        t_start = time.time()
        url = self._api_base + _ANALYZE_PATH

        suffix = image_path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        try:
            with open(image_path, "rb") as fh:
                resp = self._session.post(
                    url,
                    files={"file": (image_path.name, fh, mime)},
                    timeout=self._timeout,
                )
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise ImageEvalClientError(
                f"Image upload to {url} timed out after {self._timeout}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise ImageEvalClientError(
                f"HTTP {exc.response.status_code} from {url}: "
                f"{exc.response.text[:500]}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ImageEvalClientError(
                f"Request to {url} failed: {exc}"
            ) from exc

        latency = time.time() - t_start

        try:
            data: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise ImageEvalClientError(
                f"Non-JSON response from {url}: {resp.text[:500]}"
            ) from exc

        # Extract ingredient names from the per-ingredient breakdown
        ingredients_raw: list[dict] = data.get("ingredients", [])
        extracted_names = [
            str(ing.get("name", "")).strip()
            for ing in ingredients_raw
            if ing.get("name")
        ]

        # Confidence: average of per-ingredient confidence scores
        conf_scores = [
            float(ing.get("confidence_score", 0.0))
            for ing in ingredients_raw
            if ing.get("confidence_score") is not None
        ]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        return ImagePredictionResult(
            dish_id=golden.dish_id,
            dish_name_predicted=str(data.get("dish_name", "")).strip(),
            extracted_ingredients=extracted_names,
            calories_kcal=float(data.get("total_calories", 0.0)),
            protein_g=float(data.get("total_protein", 0.0)),
            carbs_g=float(data.get("total_carbs", 0.0)),
            fat_g=float(data.get("total_fat", 0.0)),
            fiber_g=0.0,
            confidence=max(0.0, min(1.0, avg_conf)),
            latency_seconds=latency,
            raw_response=data,
        )

    def judge_prediction(
        self,
        golden: GoldenDish,
        prediction: ImagePredictionResult,
    ) -> JudgeResult:
        """
        Call the LLM-as-a-judge endpoint and return a quality score [0, 10].
        Reuses the same /api/v1/judge-nutrition endpoint as the ingredient
        eval client.
        """
        payload = {
            "dish_id": golden.dish_id,
            "ground_truth": {
                "calories_kcal": golden.calories_kcal,
                "protein_g":     golden.protein_g,
                "carbs_g":       golden.carbs_g,
                "fat_g":         golden.fat_g,
                "fiber_g":       golden.fiber_g,
            },
            "prediction": {
                "calories_kcal": prediction.calories_kcal,
                "protein_g":     prediction.protein_g,
                "carbs_g":       prediction.carbs_g,
                "fat_g":         prediction.fat_g,
                "fiber_g":       prediction.fiber_g,
            },
        }
        url = self._judge_base + _JUDGE_PATH
        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ImageEvalClientError(
                f"Judge request failed: {exc}"
            ) from exc

        return JudgeResult(
            score=float(data.get("score", 0.0)),
            explanation=str(data.get("explanation", "")),
        )
