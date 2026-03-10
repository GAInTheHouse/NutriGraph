"""
Generate context-aware clarification questions for low-confidence ingredients
using Gemini via the Vertex AI REST API.

Uses the same VERTEXAI_API_KEY as extract_ingredients. The LLM receives dish
context, low-confidence ingredients, their scores, and best matches to produce
natural, targeted follow-up questions.
"""

import json
import logging
import os
from typing import List, Optional, Union

import requests

logger = logging.getLogger(__name__)

# Model used for clarification questions. Set NUTRIGRAPH_CLARIFICATION_MODEL to override.
# gemini-2.0-flash is more capable than gemini-2.5-flash-lite and follows instructions better.
DEFAULT_CLARIFICATION_MODEL = "gemini-2.0-flash"


def _resolve_api_key(api_key: Union[str, None]) -> str:
    """Load .env and return the effective API key, raising ValueError if absent."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = api_key or os.environ.get("VERTEXAI_API_KEY")
    if not key:
        raise ValueError(
            "Vertex AI API Key required. Set VERTEXAI_API_KEY in your .env file or environment."
        )
    return key


def _call_gemini_text_only(
    prompt: str, api_key: str, model: Optional[str] = None
) -> str:
    """
    Send a text prompt to Gemini and return the raw text response.

    Raises:
        RuntimeError: If the API returns a non-2xx status.
        ValueError: If the response JSON has an unexpected structure.
    """
    model = model or os.environ.get("NUTRIGRAPH_CLARIFICATION_MODEL", DEFAULT_CLARIFICATION_MODEL)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"{model}:generateContent?key={api_key}"
    )
    response = requests.post(
        url, headers={"Content-Type": "application/json"}, json=payload, timeout=30
    )

    if not response.ok:
        raise RuntimeError(f"Vertex AI API error: {response.status_code} - {response.text}")

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected response structure from Vertex AI: {data}") from exc


def generate_clarification_questions(
    dish_name: str,
    low_conf_ingredients: List[str],
    scores: List[float],
    best_matches: List[str],
    *,
    threshold: float = 0.50,
    api_key: Union[str, None] = None,
) -> List[str]:
    """
    Use an LLM to generate context-aware clarification questions for each
    low-confidence ingredient.

    Args:
        dish_name: Name of the dish (e.g. "Chicken Caesar Salad").
        low_conf_ingredients: Ingredient strings whose retrieval score is below threshold.
        scores: The combined match score for each low-confidence ingredient (0-1).
        best_matches: The top ChromaDB match name for each low-confidence ingredient.
        threshold: Score threshold below which ingredients are considered low-confidence (must match graph).
        api_key: Vertex AI API Key. Falls back to VERTEXAI_API_KEY env var.

    Returns:
        A list of natural-language questions, one per low-confidence ingredient,
        in the same order. If the LLM fails or returns invalid JSON, falls back
        to templated questions (with "Fallback template response - " prefix).

    Raises:
        None. All exceptions are caught and fallback questions are returned.
    """
    _FALLBACK_PREFIX = "Fallback template response - "
    _GENERIC_PHRASE = "can you clarify details such as"

    def _fallback_questions() -> List[str]:
        return [
            f"{_FALLBACK_PREFIX}For the ingredient '{ing}', can you clarify details such as brand, "
            f"cooking method (e.g., grilled vs fried), or any sauces/seasonings?"
            for ing in low_conf_ingredients
        ]

    logger.info(
        "clarification_questions: called dish_name=%r, low_conf_ingredients=%s",
        dish_name, low_conf_ingredients,
    )
    raw_text = ""
    try:
        key = _resolve_api_key(api_key)
    except ValueError as e:
        logger.warning("clarification_questions: API key missing or invalid (%s), using fallback", e)
        return _fallback_questions()

    # Build context for the LLM
    context_parts = []
    for i, (ing, score, match) in enumerate(
        zip(low_conf_ingredients, scores, best_matches)
    ):
        context_parts.append(
            f"  - '{ing}' (score: {score:.2f}, best DB match: '{match}')"
        )

    context_str = "\n".join(context_parts) if context_parts else "  (none)"

    prompt = f"""You are helping improve a nutrition lookup system. The user uploads a dish photo; we extract ingredients and match them to a nutrition database (USDA, OpenFoodFacts). Each match has a confidence score (0-1). When the score is low, we need to ask the user for details that will help us find a BETTER database match so the score goes UP.

IMPORTANT: The ingredient strings below may ALREADY include details the user gave in a previous round (e.g. "penne pasta, semolina, al dente" means they already said it was semolina and al dente). Do NOT ask for information that is already in the string. If the string already says "semolina", do not ask "was it semolina or whole wheat?". If it says "sautéed", do not ask how it was cooked. Only ask for details that are still MISSING and would help the database lookup.

GOAL: Your question should elicit information that will improve the vector/lexical match. Think about what the database entries look like: they use terms like "Pasta, penne, cooked", "Chicken, breast, grilled", "Oil, olive". We need cooking method, preparation, type/variety—anything that narrows the match.

Dish: "{dish_name}"

Low-confidence ingredients (score < {threshold}) and their current best DB match:
{context_str}

For EACH ingredient above, write ONE short question tailored to that ingredient and its current best match. The question should:
- Reference the dish and ingredient so it feels contextual.
- Ask ONLY for details that are NOT already in the ingredient string (read the string carefully—if it already has semolina, cooked, sautéed, etc., do not ask for that).
- Be specific—not generic. Avoid "can you clarify details such as…" unless you add a concrete example.

BAD (repeated): Ingredient string is "penne pasta, semolina, al dente" but you ask "was it semolina or whole wheat?" — semolina is already there, so do not ask.
GOOD: Ingredient is "penne pasta, semolina, al dente" — ask only what is missing, e.g. "Was the penne boiled or baked?" or "Any sauce or oil used?"

Respond with ONLY a valid JSON object:
{{"questions": ["question 1", "question 2", ...]}}

The number of questions must match the number of low-confidence ingredients listed above."""

    model_used = os.environ.get("NUTRIGRAPH_CLARIFICATION_MODEL", DEFAULT_CLARIFICATION_MODEL)
    logger.info(
        "clarification_questions: calling LLM model=%s. Prompt (first 500 chars): %s",
        model_used,
        (prompt[:500] + "..." if len(prompt) > 500 else prompt),
    )
    raw_text = ""
    try:
        text = _call_gemini_text_only(prompt, key)
        text = text.strip()
        raw_text = text
        logger.info(
            "clarification_questions: LLM raw response (first 400 chars): %s",
            (text[:400] + "..." if len(text) > 400 else text),
        )
        if "```json" in text:
            text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        questions = result.get("questions", [])
        if isinstance(questions, list) and len(questions) == len(low_conf_ingredients):
            # Reject generic-looking questions so we use fallback and user sees the prefix
            if any(_GENERIC_PHRASE in (str(q) or "") for q in questions):
                logger.warning(
                    "clarification_questions: LLM returned generic questions (contain '%s'), using fallback",
                    _GENERIC_PHRASE,
                )
            else:
                logger.info("clarification_questions: using LLM-generated questions")
                return [str(q).strip() for q in questions]
        logger.warning(
            "clarification_questions: LLM returned wrong count (got %s, need %s), using fallback",
            len(questions) if isinstance(questions, list) else 0,
            len(low_conf_ingredients),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            "clarification_questions: LLM parse failed (%s), using fallback. Raw (first 300): %s",
            type(e).__name__,
            (raw_text[:300] + "..." if len(raw_text) > 300 else raw_text or "n/a"),
        )
    except (ValueError, RuntimeError) as e:
        logger.warning(
            "clarification_questions: LLM API call failed (%s), using fallback",
            type(e).__name__,
        )

    # Fallback to templated questions if LLM output is invalid or generic
    logger.info(
        "clarification_questions: returning FALLBACK template for %d ingredient(s)",
        len(low_conf_ingredients),
    )
    return _fallback_questions()


# Sentence fragments and stopwords that must NOT appear in refined ingredient phrases
_SENTENCE_NOISE = frozenset({
    "it", "was", "is", "are", "the", "a", "an", "they", "this", "that",
    "i", "you", "we", "he", "she", "to", "of", "in", "on", "at", "for",
    "with", "and", "or", "but", "so", "as", "if", "when", "can", "could",
})


def _heuristic_extract_keywords(user_answer: str, max_words: int = 5) -> List[str]:
    """
    Extract nutrition-relevant keywords from free text without an LLM.
    Removes stopwords and keeps cooking/preparation terms and ingredient words.
    """
    import re
    text = (user_answer or "").strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w and w not in _SENTENCE_NOISE and len(w) > 1]
    # Prefer: cooking methods, types, forms (common in nutrition DBs)
    cooking_terms = {"boiled", "grilled", "fried", "baked", "raw", "cooked", "steamed",
                     "roasted", "sautéed", "sauté", "sautéed", "poached", "broiled",
                     "semolina", "whole", "wheat", "organic", "diced", "sliced",
                     "minced", "chopped", "crushed", "skinless", "boneless"}
    # Sort: cooking/type terms first (more useful for DB match), then rest
    ordered = []
    for w in words:
        if w in cooking_terms and w not in ordered:
            ordered.append(w)
    for w in words:
        if w not in ordered:
            ordered.append(w)
    return ordered[:max_words]


def refine_ingredient_from_clarification(
    original_ingredient: str,
    user_answer: str,
    dish_name: str = "",
    *,
    api_key: Union[str, None] = None,
) -> str:
    """
    Turn the user's clarification into a single, short ingredient descriptor
    suitable for vector-DB lookup.

    Uses a two-step approach for robustness:
    1. LLM extracts keywords only (no full sentences) - we control the final format
    2. Validate output; if LLM returns sentence fragments, use heuristic extraction
    3. Merge original + keywords into "original, kw1, kw2" (max 8 words total)

    Args:
        original_ingredient: The ingredient that had low confidence (e.g. "penne pasta").
        user_answer: The user's free-text reply (e.g. "It was boiled semolina pasta.").
        dish_name: Optional dish name for context.
        api_key: Vertex AI API Key. Falls back to VERTEXAI_API_KEY env var.

    Returns:
        A short phrase (e.g. "penne pasta, boiled, semolina") for DB lookup.
    """
    user_answer = (user_answer or "").strip()
    if not user_answer:
        return original_ingredient.strip()

    original = (original_ingredient or "").strip()
    original_tokens = set(original.lower().split())

    # Step 1: Try LLM keyword extraction (structured output - harder to mess up)
    keywords: List[str] = []
    try:
        key = _resolve_api_key(api_key)
        prompt = f"""You are converting user clarifications into keywords for a nutrition database search. Our search index is built from USDA FoodData Central and OpenFoodFacts. In these datasets, ingredient names often follow patterns like: "Pasta, penne, cooked", "Peppers, bell, raw", "Chicken, breast, grilled", "Onions, raw", "Tomato sauce". Use similar terms so the vector search finds a better match.

Original ingredient: "{original}"
User said: "{user_answer}"
Dish: "{dish_name or "unknown"}"

Output a JSON object with a "keywords" array of 2-6 words or short phrases that match how USDA/OpenFoodFacts name ingredients: cooking state (cooked, raw, boiled, grilled, sautéed), type (semolina, whole wheat, bell), form (fresh, canned). NO pronouns or articles. Prefer terms that appear in nutrition databases.

Example for "It was boiled semolina pasta" -> {{"keywords": ["cooked", "semolina", "boiled"]}}
Example for "sautéed in oil" -> {{"keywords": ["sautéed", "cooked"]}}

Respond with ONLY valid JSON: {{"keywords": ["word1", "word2", ...]}}"""

        text = _call_gemini_text_only(prompt, key)
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        raw = result.get("keywords") or []
        if isinstance(raw, list):
            for k in raw[:6]:
                k_str = str(k).strip().lower()
                if k_str and k_str not in _SENTENCE_NOISE and len(k_str) <= 25:
                    keywords.append(k_str)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
        pass

    # Step 2: If LLM failed or returned noise, use heuristic
    if not keywords:
        keywords = _heuristic_extract_keywords(user_answer)

    # Step 3: Build final phrase - original + new keywords (avoid duplicates)
    seen = original_tokens.copy()
    parts = [original]
    for kw in keywords:
        kw_tokens = set(kw.split())
        if kw_tokens and not (kw_tokens <= seen):
            parts.append(kw)
            seen.update(kw_tokens)

    result = ", ".join(parts)
    return result[:120] if len(result) > 120 else result


def refine_ingredients_batch(
    ingredients_with_answers: List[tuple],
    dish_name: str = "",
    *,
    api_key: Union[str, None] = None,
) -> List[str]:
    """
    Refine multiple (original_ingredient, user_answer) pairs in one LLM call.
    Returns a list of short DB-friendly phrases in the same order.

    Each tuple is (original_ingredient: str, user_answer: str).
    If the LLM fails or returns invalid data, falls back to per-item heuristic
    refinement for each pair.
    """
    if not ingredients_with_answers:
        return []

    try:
        key = _resolve_api_key(api_key)
    except ValueError:
        # No API key: use heuristic keyword extraction for each
        out = []
        for orig, ans in ingredients_with_answers:
            orig = (orig or "").strip()
            ans = (ans or "").strip()
            if not ans:
                out.append(orig)
                continue
            kw = _heuristic_extract_keywords(ans)
            seen = set((orig or "").lower().split())
            parts = [orig]
            for k in kw:
                if k not in seen:
                    parts.append(k)
                    seen.update(k.split())
            out.append(", ".join(parts)[:120])
        return out

    # Only send non-empty answers to the LLM; preserve original ingredient unchanged for blank answers.
    non_empty: List[tuple] = [(i, (orig, (ans or "").strip())) for i, (orig, ans) in enumerate(ingredients_with_answers) if (ans or "").strip()]
    if not non_empty:
        return [(orig or "").strip() for orig, _ in ingredients_with_answers]

    indices_with_answers = [i for i, _ in non_empty]
    items_for_prompt = [item for _, item in non_empty]
    n_refine = len(items_for_prompt)

    lines = []
    for i, (orig, ans) in enumerate(items_for_prompt):
        lines.append(f"  {i + 1}. Original: \"{orig}\" | User said: \"{ans}\"")

    prompt = f"""You are converting user clarifications into short phrases for a nutrition database search. Our search index uses USDA FoodData Central and OpenFoodFacts. In these datasets, names often look like: "Pasta, penne, cooked", "Peppers, bell, raw", "Chicken, breast, grilled", "Onions, raw", "Tomato sauce". Use similar vocabulary so the vector search gets a better match.

Dish: "{dish_name or "unknown"}"

Items:
{chr(10).join(lines)}

Output a JSON object with a "refined" array of {n_refine} strings in the SAME ORDER. For each item: combine the original ingredient with keywords from the user's answer, using terms that appear in USDA/OpenFoodFacts (e.g. cooked, raw, boiled, grilled, sautéed, semolina, bell, fresh). Each output string should look like a database-style name: "original, state_or_type, preparation" (e.g. "penne pasta, cooked, semolina"). Max 8-10 words per string. No pronouns or full sentences.

Example format: {{"refined": ["phrase1", "phrase2", ...]}}

Respond with ONLY valid JSON: {{"refined": [...]}}"""

    try:
        text = _call_gemini_text_only(prompt, key)
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        refined_list = result.get("refined") or []
        if isinstance(refined_list, list) and len(refined_list) == n_refine:
            refined_strs = [str(s).strip()[:120] for s in refined_list]
            if all(refined_strs):
                # Stitch back: use refined for indices that had answers, original for blank
                out = [(orig or "").strip() for orig, _ in ingredients_with_answers]
                for idx, refined in zip(indices_with_answers, refined_strs):
                    if idx < len(out):
                        out[idx] = refined
                logger.info("clarification_questions: batch refined %d ingredients in one LLM call", n_refine)
                return out
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
        pass

    # Fallback: refine each separately (heuristic or single-item LLM); blank answers stay as original
    return [
        refine_ingredient_from_clarification(orig, (ans or "").strip(), dish_name, api_key=key)
        for orig, ans in ingredients_with_answers
    ]
