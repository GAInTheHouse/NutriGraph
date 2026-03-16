"""
LLM-backed auto-responder for the NutriGraph evaluation framework.

During offline evaluation the backend may ask clarification questions that a
real user would normally answer interactively.  AutoResponder simulates that
user: it receives the list of questions together with the ground-truth dish
data (ingredients_list + preparation_notes) and calls Gemini via the Vertex AI
REST API (same pattern as src/ml/extract_ingredients.py) to produce concise,
faithful answers.

The LLM is explicitly instructed to answer *only* from the provided context,
preventing hallucination of ingredient details that are not in the golden set.
No additional SDK dependencies are required — only ``requests`` (already in
requirements.txt).
"""
from __future__ import annotations

import logging
import os
import re

import requests

from .models import GoldenDish

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash-lite"
_VERTEX_AI_URL = (
    "https://aiplatform.googleapis.com/v1/publishers/google/models/"
    "{model}:generateContent?key={api_key}"
)

_SYSTEM_INSTRUCTION = (
    "You are simulating a diner or cook who has precise knowledge of exactly "
    "what went into a dish.  You will be given the dish name, the exact "
    "ingredient list (with quantities), and any preparation notes.  You must "
    "answer every question using ONLY the information provided — do not invent "
    "or assume anything that is not stated.  Keep each answer to 1–2 sentences."
)

_PROMPT_TEMPLATE = """\
{system}

Dish: {dish_name}
Ingredients: {ingredients_list}
Preparation notes: {preparation_notes}

Please answer each of the following questions using only the information above.
Return your answers as a numbered list that matches the question numbering exactly.

Questions:
{numbered_questions}

Answers:"""


class AutoResponder:
    """
    Calls the Vertex AI REST API (Gemini) to answer backend clarification
    questions during offline evaluation.

    Uses the same ``VERTEXAI_API_KEY`` + ``requests`` pattern as
    ``src/ml/extract_ingredients.py`` — no additional SDK required.

    Parameters
    ----------
    api_key:
        Vertex AI API key.  Defaults to the ``VERTEXAI_API_KEY`` environment
        variable when not supplied explicitly.
    model:
        Gemini model name served through the Vertex AI publisher endpoint.
        Defaults to ``gemini-2.5-flash-lite`` (matches the rest of the project).
    timeout:
        Per-request HTTP timeout in seconds (default 60).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout: int = 60,
    ) -> None:
        resolved_key = api_key or os.environ.get("VERTEXAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Vertex AI API key is required.  Pass it as ``api_key`` or set "
                "the VERTEXAI_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self._model = model
        self._timeout = timeout
        self._url = _VERTEX_AI_URL.format(model=model, api_key=resolved_key)
        logger.debug("AutoResponder initialised with model=%s", model)

    def answer_questions(
        self,
        questions: list[str],
        golden: GoldenDish,
    ) -> list[str]:
        """
        Generate one answer per question using the golden dish as context.

        Parameters
        ----------
        questions:
            Clarification questions emitted by the backend.
        golden:
            The ground-truth dish record whose ``ingredients_list`` and
            ``preparation_notes`` serve as the authoritative context.

        Returns
        -------
        list[str]
            One answer string per question, in the same order.  If the model
            returns fewer answers than questions, the remaining positions are
            filled with ``"No additional information available."``.
        """
        if not questions:
            return []

        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        prompt = _PROMPT_TEMPLATE.format(
            system=_SYSTEM_INSTRUCTION,
            dish_name=golden.dish_name,
            ingredients_list=golden.ingredients_list,
            preparation_notes=golden.preparation_notes,
            numbered_questions=numbered,
        )

        logger.debug(
            "AutoResponder: sending %d question(s) for dish %s",
            len(questions),
            golden.dish_id,
        )

        raw_text = self._call_vertex_ai(prompt)
        answers = self._parse_numbered_answers(raw_text, len(questions))

        logger.debug(
            "AutoResponder: received %d answer(s) for dish %s",
            len(answers),
            golden.dish_id,
        )
        return answers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_vertex_ai(self, prompt: str) -> str:
        """
        POST a text-only prompt to the Vertex AI generateContent endpoint
        and return the raw text of the first candidate.

        Raises
        ------
        RuntimeError
            If the API returns a non-2xx status.
        ValueError
            If the response JSON has an unexpected structure.
        """
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }

        response = requests.post(
            self._url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self._timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Vertex AI API error: {response.status_code} - {response.text[:500]}"
            )

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Unexpected response structure from Vertex AI: {data}"
            ) from exc

    @staticmethod
    def _parse_numbered_answers(text: str, expected: int) -> list[str]:
        """
        Extract answers from a numbered-list response.

        Handles formats like::

            1. The chicken is skinless.
            2. The dressing is mixed in.

        Falls back to splitting on blank lines if numbered markers are absent.
        """
        pattern = re.compile(r"^\s*\d+\.\s+(.+)", re.MULTILINE)
        matches = pattern.findall(text)

        if matches:
            answers = [m.strip() for m in matches]
        else:
            chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
            answers = [re.sub(r"^\d+[\.\)]\s*", "", c) for c in chunks]

        fallback = "No additional information available."
        while len(answers) < expected:
            answers.append(fallback)

        return answers[:expected]
