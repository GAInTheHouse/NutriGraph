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

This module does NOT integrate with the UI yet. It is intended to be called from
FastAPI endpoints or scripts, and then wired into the Streamlit app later.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .retrieval_server import _get_collection, _get_embedding_model


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
        threshold: Match score threshold in [0, 1]; below -> ask clarification.
        matches: Per-ingredient list of RetrievalMatch from vector DB (aligned by index).
        scores: Per-ingredient best combined score (embedding + lexical), aligned by index.
        low_conf_indices: Ingredient indices whose best score < threshold.
        low_conf_ingredients: Ingredient strings corresponding to low_conf_indices.
        questions: List of clarification questions to ask the user.
    """

    ingredients: List[str]
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


def _lexical_overlap(query: str, candidate: str) -> float:
    """
    Compute a simple token-overlap score between query and candidate name.

    Score is |intersection(tokens)| / |union(tokens)| in [0, 1].
    """
    q_tokens = {t for t in query.lower().split() if t}
    c_tokens = {t for t in candidate.lower().split() if t}
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


def retrieve_node(state: ClarificationState) -> ClarificationState:
    """
    Node: run vector retrieval for each ingredient and populate `matches` + `scores`.
    """
    collection = _get_collection()
    model = _get_embedding_model()

    ingredients = [text.strip() for text in state.get("ingredients", []) if text.strip()]
    if not ingredients:
        state["matches"] = []
        state["scores"] = []
        state["low_conf_indices"] = []
        state["low_conf_ingredients"] = []
        return state

    embeddings = model.encode(ingredients, show_progress_bar=False).tolist()
    result = collection.query(query_embeddings=embeddings, n_results=5)

    all_matches: List[List[RetrievalMatch]] = []
    scores: List[float] = []

    for idx, query_text in enumerate(ingredients):
        ids = result.get("ids", [[]])[idx]
        dists = result.get("distances", [[]])[idx]
        metadatas = result.get("metadatas", [[]])[idx]

        ing_matches: List[RetrievalMatch] = []
        best_score = 0.0

        for m_idx, mid in enumerate(ids):
            if m_idx >= len(dists):
                # Skip inconsistent result rows (distance missing)
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

        # Sort matches by their combined score (descending) so index 0 is best
        ing_matches.sort(key=lambda m: m.get("score", 0.0), reverse=True)

        all_matches.append(ing_matches)
        scores.append(best_score)

    state["matches"] = all_matches
    state["scores"] = scores
    return state


def decide_low_conf_node(
    state: ClarificationState, default_threshold: float = 0.7
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

    low_indices: List[int] = [
        idx for idx, s in enumerate(scores) if s < threshold
    ]
    state["low_conf_indices"] = low_indices
    state["low_conf_ingredients"] = [
        ingredients[idx] for idx in low_indices if idx < len(ingredients)
    ]
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
    Node: generate simple clarification questions for each low-confidence ingredient.

    This intentionally does not call an LLM yet; instead it creates templated
    questions that can be surfaced to the user. Later, this can be swapped out
    to use Gemini or another LLM for more natural questions.
    """
    low_conf = state.get("low_conf_ingredients", [])
    questions: List[str] = []

    for ing in low_conf:
        questions.append(
            f"For the ingredient '{ing}', can you clarify details such as brand, "
            f"cooking method (e.g., grilled vs fried), or any sauces/seasonings?"
        )

    state["questions"] = questions
    return state


def build_clarification_graph(default_threshold: float = 0.7):
    """
    Build and compile the clarification LangGraph.

    Usage (example):
        from src.backend.clarification_graph import build_clarification_graph
        graph = build_clarification_graph()
        result = graph.invoke({
            "ingredients": ["chicken breast", "mystery sauce"],
            "threshold": 0.7
        })

    The `result` will contain:
        - matches: retrieval results per ingredient
        - scores: best score per ingredient
        - low_conf_ingredients: list[str]
        - questions: list[str] (if any low-confidence ingredients)
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


