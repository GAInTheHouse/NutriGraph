"""
LangGraph-based clarification state machine for NutriGraph.

High-level logic:
    1. Take a list of ingredient strings.
    2. Retrieve nearest neighbors from the ingredient vector DB (Chroma).
    3. For each ingredient, compute a match score that blends:
         - Vector distance (from the embedding model/Chroma)
         - Simple lexical token overlap between query and candidate name.
    4. If any best match has score < threshold, generate clarification questions
       and mark those ingredients as low-confidence.
    5. Otherwise, mark the state as done with high-confidence matches.

This module is invoked directly from ``src/ui/diner.py`` after a successful
dish image analysis.  It can also be called from FastAPI endpoints or CLI
scripts via ``build_clarification_graph().invoke(...)``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .retrieval_server import _get_collection, _get_embedding_model

logger = logging.getLogger(__name__)

# Number of candidates to retrieve per query; higher gives a better chance of a good match
RETRIEVAL_TOP_K = 10

# Synonyms that often appear in USDA/OpenFoodFacts (key -> add this term to query for lexical/embedding match)
# So "boiled" query also gets "cooked" and is more likely to match DB "Pasta, penne, cooked"
_COOKING_SYNONYMS: Dict[str, str] = {
    "boiled": "cooked",
    "al dente": "cooked",
    "sautéed": "cooked",
    "sauteed": "cooked",
    "pan-fried": "cooked",
    "pan fried": "cooked",
    "steamed": "cooked",
    "poached": "cooked",
    "broiled": "cooked",
    "baked": "cooked",
    "roasted": "cooked",
    "grilled": "cooked",
    "fried": "cooked",
}

# Confidence threshold used by the clarification graph.
# Ingredients whose best combined score (vector + lexical) falls below this
# value are considered low-confidence and will trigger a clarifying question.
DEFAULT_THRESHOLD: float = 0.50


class RetrievalMatch(TypedDict, total=False):
    """Lightweight view of a retrieval match used in the agent state."""

    id: str
    name: str
    source: str
    distance: float
    score: float
    energy_kcal: Optional[float]
    protein_g: Optional[float]
    carbohydrates_g: Optional[float]
    fat_g: Optional[float]
    fdc_id: Optional[int]


class ClarificationState(TypedDict, total=False):
    """
    LangGraph state for clarification.

    Fields:
        ingredients: Raw ingredient query strings.
        dish_name: Name of the dish (for LLM context when generating questions).
        fallback_queries: Optional map {idx: query} — when refining, we try both
            the refined query and the previous query, keeping the best match.
        threshold: Match score threshold in [0, 1]; below -> ask clarification.
        matches: Per-ingredient list of RetrievalMatch from vector DB (aligned by index).
        scores: Per-ingredient best combined score (embedding + lexical), aligned by index.
        low_conf_indices: Ingredient indices whose best score < threshold.
        low_conf_ingredients: Ingredient strings corresponding to low_conf_indices.
        questions: List of clarification questions to ask the user.
        used_fallback_indices: Indices where we used the fallback query (better score);
            callers can keep that query for the next round so overall confidence does not drop.
    """

    ingredients: List[str]
    dish_name: str
    fallback_queries: Dict[int, str]
    used_fallback_indices: List[int]
    threshold: float
    matches: List[List[RetrievalMatch]]
    scores: List[float]
    low_conf_indices: List[int]
    low_conf_ingredients: List[str]
    questions: List[str]


def _compute_score(distance: float) -> float:
    """
    Convert a Chroma distance into a simple similarity score in (0, 1].

    We use 1 / (1 + distance) so that:
        - distance 0   -> score 1.0 (perfect match)
        - distance 0.5 -> ~0.67
        - distance 1.0 -> 0.5
    """
    return 1.0 / (1.0 + max(distance, 0.0))


def _normalize_for_match(text: str) -> str:
    """Normalize for matching: ASCII fold diacritics so 'sautéed' matches 'sauteed' in DB."""
    if not text:
        return ""
    nfd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfd if not unicodedata.combining(c))


def _normalize_tokens(text: str) -> set:
    """Tokenize and normalize: split on whitespace and commas, strip punctuation, ASCII-fold."""
    normalized = _normalize_for_match((text or "").lower())
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return {t for t in normalized.split() if len(t) > 1}


def _lexical_overlap(query: str, candidate: str) -> float:
    """
    Compute a simple token-overlap score between query and candidate name.

    Score is |intersection(tokens)| / |union(tokens)| in [0, 1].
    Tokens are normalized (punctuation stripped) so "pasta," matches "pasta".
    """
    q_tokens = _normalize_tokens(query)
    c_tokens = _normalize_tokens(candidate)
    if not q_tokens or not c_tokens:
        return 0.0
    inter = q_tokens & c_tokens
    union = q_tokens | c_tokens
    if not union:
        return 0.0
    return len(inter) / len(union)


def _combined_match_score(distance: float, query: str, candidate_name: str) -> float:
    """
    Blend vector similarity with lexical overlap into a single score in [0, 1].

    - sim_dist: from distance via 1 / (1 + d)  (0..1, higher is better)
    - lex:      token Jaccard overlap between query and candidate name (0..1)

    The weights (0.7, 0.3) can be tuned later once we have empirical data.
    """
    sim_dist = _compute_score(distance)
    lex = _lexical_overlap(query, candidate_name)
    return 0.7 * sim_dist + 0.3 * lex


def _expand_query_synonyms(query: str) -> str:
    """
    Append synonym terms that appear in USDA/OpenFoodFacts so the query embedding
    and lexical overlap are more likely to match DB entries (e.g. 'cooked').
    """
    q_lower = (query or "").lower().strip()
    if not q_lower:
        return query
    added: List[str] = []
    for term, synonym in _COOKING_SYNONYMS.items():
        if term in q_lower and synonym not in q_lower and synonym not in added:
            added.append(synonym)
    if not added:
        return query
    return f"{query}, {', '.join(added)}"


def _retrieve_for_query(
    query_text: str,
    collection,
    model,
    n_results: int = RETRIEVAL_TOP_K,
) -> tuple[List[RetrievalMatch], float]:
    """Run retrieval for a single query; return matches and best score."""
    # Expand with cooking synonyms so we match DB terms like "cooked" when user said "boiled"
    query_expanded = _expand_query_synonyms(query_text)
    emb = model.encode([query_expanded], show_progress_bar=False).tolist()
    result = collection.query(query_embeddings=emb, n_results=n_results)
    ids = result.get("ids", [[]])[0]
    dists = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    ing_matches: List[RetrievalMatch] = []
    best_score = 0.0
    for m_idx, mid in enumerate(ids):
        if m_idx >= len(dists):
            continue
        distance = float(dists[m_idx])
        meta = metadatas[m_idx] if m_idx < len(metadatas) else {}
        meta = meta or {}
        name = str(meta.get("name", ""))
        score = _combined_match_score(distance, query_text, name)
        best_score = max(best_score, score)
        ing_matches.append(
            RetrievalMatch(
                id=str(mid),
                name=name,
                source=str(meta.get("source", "")),
                distance=distance,
                score=score,
                energy_kcal=meta.get("energy_kcal"),
                protein_g=meta.get("protein_g"),
                carbohydrates_g=meta.get("carbohydrates_g"),
                fat_g=meta.get("fat_g"),
                fdc_id=meta.get("fdc_id"),
            )
        )
    ing_matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)
    return ing_matches, best_score


def retrieve_node(state: ClarificationState) -> ClarificationState:
    """
    Node: run vector retrieval for each ingredient and populate `matches` + `scores`.

    When fallback_queries is provided for an index (after user refinement), we
    run retrieval for BOTH the refined query and the original query, then keep
    the best match. This prevents confidence from dropping when refinement
    produces a noisier embedding.
    """
    collection = _get_collection()
    model = _get_embedding_model()

    ingredients = [text.strip() for text in state.get("ingredients", []) if text and text.strip()]
    state["ingredients"] = ingredients
    fallback_queries = state.get("fallback_queries") or {}

    if not ingredients:
        state["matches"] = []
        state["scores"] = []
        state["low_conf_indices"] = []
        state["low_conf_ingredients"] = []
        state["used_fallback_indices"] = []
        return state

    all_matches: List[List[RetrievalMatch]] = []
    scores: List[float] = []
    used_fallback_indices: List[int] = []

    for idx, query_text in enumerate(ingredients):
        fallback = fallback_queries.get(idx) if isinstance(fallback_queries, dict) else None

        if fallback and fallback.strip() != query_text:
            # Multi-query: try both refined and original, keep best
            matches_a, score_a = _retrieve_for_query(query_text, collection, model)
            matches_b, score_b = _retrieve_for_query(fallback.strip(), collection, model)
            if score_b > score_a:
                logger.info(
                    "clarification_graph: multi-query idx=%s refined_score=%.3f fallback_score=%.3f -> using FALLBACK (better). "
                    "refined_query=%r fallback_query=%r",
                    idx, score_a, score_b, query_text, fallback.strip(),
                )
                all_matches.append(matches_b)
                scores.append(score_b)
                used_fallback_indices.append(idx)
            else:
                logger.info(
                    "clarification_graph: multi-query idx=%s refined_score=%.3f fallback_score=%.3f -> using REFINED. "
                    "refined_query=%r fallback_query=%r",
                    idx, score_a, score_b, query_text, fallback.strip(),
                )
                all_matches.append(matches_a)
                scores.append(score_a)
        else:
            matches, best_score = _retrieve_for_query(query_text, collection, model)
            all_matches.append(matches)
            scores.append(best_score)

    state["matches"] = all_matches
    state["scores"] = scores
    state["used_fallback_indices"] = used_fallback_indices
    return state


def decide_low_conf_node(
    state: ClarificationState, default_threshold: float = DEFAULT_THRESHOLD
) -> ClarificationState:
    """
    Node: determine which ingredients are low-confidence based on threshold.
    """
    # If the caller provided an explicit threshold in state, use it.
    # Otherwise, fall back to the default configured at graph build time.
    threshold = state.get("threshold")
    if threshold is None:
        threshold = default_threshold
    scores = state.get("scores", [])
    ingredients = state.get("ingredients", [])

    low_indices: List[int] = [idx for idx, s in enumerate(scores) if s < threshold]
    state["low_conf_indices"] = low_indices
    state["low_conf_ingredients"] = [
        ingredients[idx] for idx in low_indices if idx < len(ingredients)
    ]
    # Persist the effective threshold into state so callers can inspect it
    state["threshold"] = threshold
    return state


def router(state: ClarificationState) -> str:
    """
    Conditional edge: if there are low-confidence ingredients, go to 'ask',
    otherwise end the graph.
    """
    if state.get("low_conf_indices"):
        return "ask"
    return END


def ask_node(state: ClarificationState) -> ClarificationState:
    """
    Node: generate context-aware clarification questions for each low-confidence
    ingredient using an LLM (Gemini via Vertex AI).

    The LLM receives dish name, low-confidence ingredients, their scores, and
    best database matches to produce natural, targeted follow-up questions.
    Falls back to templated questions if the LLM call fails.
    """
    low_conf = state.get("low_conf_ingredients", [])
    low_conf_indices = state.get("low_conf_indices", [])
    matches = state.get("matches", [])
    scores = state.get("scores", [])

    if not low_conf:
        state["questions"] = []
        return state

    # Extract best match name for each low-confidence ingredient
    best_matches: List[str] = []
    low_conf_scores: List[float] = []
    for idx in low_conf_indices:
        if idx < len(scores):
            low_conf_scores.append(scores[idx])
        else:
            low_conf_scores.append(0.0)
        if idx < len(matches) and matches[idx]:
            best = matches[idx][0]
            best_matches.append(str(best.get("name", "")))
        else:
            best_matches.append("")

    dish_name = state.get("dish_name") or "the dish"

    try:
        from src.ml.clarification_questions import generate_clarification_questions

        logger.info("clarification_graph ask_node: calling generate_clarification_questions")
        questions = generate_clarification_questions(
            dish_name=dish_name,
            low_conf_ingredients=low_conf,
            scores=low_conf_scores,
            best_matches=best_matches,
            threshold=state.get("threshold", DEFAULT_THRESHOLD),
        )
        logger.info("clarification_graph ask_node: got %d questions", len(questions))
    except Exception as e:
        # Should be rare now that generate_clarification_questions catches internally
        logger.warning(
            "clarification_graph ask_node: generate_clarification_questions raised %s, using fallback",
            type(e).__name__,
            exc_info=True,
        )
        questions = [
            "Fallback template response - For the ingredient '%s', can you clarify details such as brand, "
            "cooking method (e.g., grilled vs fried), or any sauces/seasonings?" % ing
            for ing in low_conf
        ]

    state["questions"] = questions
    return state


def build_clarification_graph(default_threshold: float = DEFAULT_THRESHOLD):
    """
    Build and compile the clarification LangGraph.

    Usage (example):
        from src.backend.clarification_graph import build_clarification_graph
        graph = build_clarification_graph()
        result = graph.invoke({
            "ingredients": ["chicken breast", "mystery sauce"],
            "dish_name": "Chicken Caesar Salad",
            "threshold": 0.7
        })

    The `result` will contain:
        - matches: retrieval results per ingredient
        - scores: best score per ingredient
        - low_conf_ingredients: list[str]
        - questions: list[str] (LLM-generated if available, else templated)
    """
    graph = StateGraph(ClarificationState)
    graph.add_node("retrieve", retrieve_node)
    # Bind the default threshold into the node via a closure so it is applied
    # whenever the state does not contain an explicit threshold.
    graph.add_node(
        "decide_low_conf",
        lambda s, dt=default_threshold: decide_low_conf_node(s, dt),
    )
    graph.add_node("ask", ask_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide_low_conf")
    graph.add_conditional_edges("decide_low_conf", router, {"ask": "ask", END: END})

    return graph.compile()


