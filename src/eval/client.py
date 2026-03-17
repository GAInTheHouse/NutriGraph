"""
HTTP client for the NutriGraph evaluation framework.

NutriGraphEvalClient wraps two backend endpoints:

1. POST /api/v1/analyze-dish-from-ingredients
   Supports a multi-turn clarification protocol:
   - Round 1: sends the dish payload; backend may respond with
     ``status == "needs_clarification"`` containing questions + session_id.
   - Subsequent rounds: sends session_id + answers until the backend
     responds with ``status == "complete"``.
   The AutoResponder generates answers for each intermediate round.

2. POST /api/v1/judge-nutrition
   Sends ground-truth macros and predicted macros; returns a 0-10 score.

All HTTP errors and timeouts are converted to NutriGraphClientError so that
the evaluation loop never crashes on a single bad call.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .auto_responder import AutoResponder
from .models import GoldenDish, JudgeResult, PredictionResult

logger = logging.getLogger(__name__)

_ANALYZE_PATH = "/api/v1/analyze-dish-from-ingredients"
_RESPOND_PATH = "/api/v1/analyze-dish-from-ingredients/respond"
_JUDGE_PATH = "/api/v1/judge-nutrition"


class NutriGraphClientError(Exception):
    """Raised when the backend returns an error or the request times out."""


class NutriGraphEvalClient:
    """
    Typed HTTP client for the two NutriGraph evaluation endpoints.

    Parameters
    ----------
    api_base_url:
        Base URL of the analysis backend, e.g. ``http://localhost:8000``.
    judge_base_url:
        Base URL of the judge backend (may be the same as ``api_base_url``).
    auto_responder:
        AutoResponder instance used to answer clarification questions.
    timeout:
        Per-request HTTP timeout in seconds (default 60).
    max_turns:
        Maximum number of clarification rounds before giving up (default 5).
    """

    def __init__(
        self,
        api_base_url: str,
        judge_base_url: str,
        auto_responder: AutoResponder,
        timeout: int = 60,
        max_turns: int = 5,
    ) -> None:
        self._api_base = api_base_url.rstrip("/")
        self._judge_base = judge_base_url.rstrip("/")
        self._auto_responder = auto_responder
        self._timeout = timeout
        self._max_turns = max_turns
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_dish(self, golden: GoldenDish) -> PredictionResult:
        """
        Run the full multi-turn clarification exchange for one dish and
        return the nutritional prediction.

        Latency covers the entire wall-clock time from the first POST to
        the final ``complete`` response.  ``num_questions`` accumulates
        the total number of clarification questions asked across all rounds.
        """
        t_start = time.time()
        num_questions = 0

        # --- Round 1: initial payload -----------------------------------
        payload: dict[str, Any] = {
            "dish_id": golden.dish_id,
            "dish_name": golden.dish_name,
            "context_type": golden.context_type,
            "cuisine": golden.cuisine,
            "serving_description": golden.serving_description,
            "serving_size_grams": golden.serving_size_grams,
            "ingredients_list": golden.ingredients_list,
            "preparation_notes": golden.preparation_notes,
        }

        data = self._post(_ANALYZE_PATH, payload)

        # --- Multi-turn clarification loop ------------------------------
        turns = 0
        while data.get("status") == "needs_clarification":
            if turns >= self._max_turns:
                raise NutriGraphClientError(
                    f"Dish {golden.dish_id}: max_turns ({self._max_turns}) "
                    "reached without a complete response from the backend."
                )

            questions: list[str] = data.get("questions", [])
            session_id: str = data.get("session_id", "")

            if not questions:
                raise NutriGraphClientError(
                    f"Dish {golden.dish_id}: backend returned "
                    "'needs_clarification' but included no questions."
                )

            logger.debug(
                "Dish %s turn %d: %d question(s) received, generating answers.",
                golden.dish_id,
                turns + 1,
                len(questions),
            )

            answers = self._auto_responder.answer_questions(questions, golden)
            num_questions += len(questions)
            turns += 1

            data = self._post(
                _RESPOND_PATH,
                {"session_id": session_id, "answers": answers},
            )

        # --- Map final response to PredictionResult ---------------------
        latency = time.time() - t_start

        if data.get("status") not in ("complete", None):
            raise NutriGraphClientError(
                f"Dish {golden.dish_id}: unexpected response status "
                f"'{data.get('status')}' from backend."
            )

        # Backend may also report num_questions; prefer our counted total
        # unless it explicitly returns a higher value (shouldn't happen).
        backend_nq: int = data.get("num_questions", 0)
        total_questions = max(num_questions, backend_nq)

        raw_confidence: float = float(data.get("confidence", 1.0))
        confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence

        return PredictionResult(
            dish_id=str(data.get("dish_id", golden.dish_id)),
            dish_name=str(data.get("dish_name", golden.dish_name)),
            calories_kcal=float(data.get("total_calories", 0.0)),
            protein_g=float(data.get("total_protein", 0.0)),
            carbs_g=float(data.get("total_carbs", 0.0)),
            fat_g=float(data.get("total_fat", 0.0)),
            fiber_g=float(data.get("total_fiber", 0.0)),
            confidence=confidence,
            num_questions=total_questions,
            latency_seconds=latency,
            raw_response=data,
        )

    def judge_prediction(
        self,
        golden: GoldenDish,
        prediction: PredictionResult,
    ) -> JudgeResult:
        """
        Call the LLM-as-a-judge endpoint and return a quality score [0, 10].
        """
        payload = {
            "dish_id": golden.dish_id,
            "ground_truth": {
                "calories_kcal": golden.calories_kcal,
                "protein_g": golden.protein_g,
                "carbs_g": golden.carbs_g,
                "fat_g": golden.fat_g,
                "fiber_g": golden.fiber_g,
            },
            "prediction": {
                "calories_kcal": prediction.calories_kcal,
                "protein_g": prediction.protein_g,
                "carbs_g": prediction.carbs_g,
                "fat_g": prediction.fat_g,
                "fiber_g": prediction.fiber_g,
            },
        }
        data = self._post(_JUDGE_PATH, payload, base=self._judge_base)
        return JudgeResult(
            score=float(data.get("score", 0.0)),
            explanation=str(data.get("explanation", "")),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        base: str | None = None,
    ) -> dict[str, Any]:
        """POST JSON to ``base + path`` and return the parsed response dict."""
        url = (base or self._api_base) + path
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise NutriGraphClientError(
                f"Request to {url} timed out after {self._timeout}s."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise NutriGraphClientError(
                f"HTTP {exc.response.status_code} from {url}: "
                f"{exc.response.text[:500]}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise NutriGraphClientError(
                f"Request to {url} failed: {exc}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise NutriGraphClientError(
                f"Non-JSON response from {url}: {response.text[:500]}"
            ) from exc
