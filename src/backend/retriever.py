"""
Hybrid nutrition retriever for NutriGraph.

Provides HybridNutritionRetriever, which combines ChromaDB semantic vector
search with optional metadata filtering (brand exact-match) and a keyword
boost post-processing step.

Called by the FastAPI endpoint /api/v1/retrieve-ingredient, which is in turn
invoked by LangGraph agent nodes after the clarification loop resolves user
ambiguities (e.g., brand name, cooking method).

ChromaDB metadata schema (per document in 'nutrigraph_ingredients'):
    exact_name      : str   – canonical ingredient name used for keyword boost
    brand           : str   – brand name (present for OpenFoodFacts rows only;
                              NOTE: brand must be added to index_ingredients.py
                              before brand-filtered queries return results)
    energy_kcal     : float
    protein_g       : float
    carbs_g         : float (stored as 'carbohydrates_g' in some rows – both
                              keys are checked defensively)
    fat_g           : float
    source          : str   – 'usda_foundation' | 'usda_sr_legacy' | 'off'
"""

import sys
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import RetrievedIngredient  # noqa: E402

CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "nutrigraph_ingredients"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Score boost applied to any result whose exact_name contains the query string
# as a substring (case-insensitive).  Keeps the boost modest so it re-ranks
# near-ties rather than overriding clearly better semantic matches.
_KEYWORD_BOOST = 0.15


def _distance_to_score(distance: float) -> float:
    """
    Convert a ChromaDB L2/cosine distance to a similarity score in (0, 1].

    distance 0.0 → score 1.0 (perfect match)
    distance 1.0 → score 0.5
    distance 2.0 → score 0.33
    """
    return round(1.0 / (1.0 + distance), 6)


def _parse_float(value: Any) -> float | None:
    """Return float or None, swallowing conversion errors."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HybridNutritionRetriever:
    """
    Hybrid semantic + keyword retriever over the NutriGraph ChromaDB index.

    Initialisation is intentionally lazy: the embedding model and ChromaDB
    client are loaded on first use so that importing this module is cheap.
    Subsequent calls reuse the same objects (singleton pattern, same as the
    rest of retrieval_server.py).

    Usage
    -----
    retriever = HybridNutritionRetriever()
    results = retriever.search_ingredient("grilled chicken breast", brand="Tyson", top_k=5)
    """

    def __init__(
        self,
        chroma_dir: Path | str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self._chroma_dir = Path(chroma_dir)
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name

        # Lazily-loaded singletons
        self._model: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            self._collection = client.get_collection(name=self._collection_name)
        return self._collection

    def _build_ingredient(
        self,
        doc_id: str,
        distance: float,
        meta: dict,
        score_override: float | None = None,
    ) -> RetrievedIngredient:
        """Assemble a RetrievedIngredient from raw ChromaDB result fields."""
        base_score = score_override if score_override is not None else _distance_to_score(distance)
        # carbs may be stored under either key depending on the indexer version
        carbs_raw = meta.get("carbs_g")
        if carbs_raw is None and "carbohydrates_g" in meta:
            carbs_raw = meta.get("carbohydrates_g")
        carbs = _parse_float(carbs_raw)
        return RetrievedIngredient(
            id=str(doc_id),
            name=str(meta.get("exact_name") or meta.get("name") or ""),
            brand=meta.get("brand") or None,
            similarity_score=round(base_score, 6),
            calories=_parse_float(meta.get("energy_kcal")),
            protein=_parse_float(meta.get("protein_g")),
            carbs=carbs,
            fat=_parse_float(meta.get("fat_g")),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def search_ingredient(
        self,
        query: str,
        brand: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedIngredient]:
        """
        Perform a hybrid search for a single ingredient query.

        Two modes depending on whether a brand was provided by the LangGraph
        clarification agent:

        Brand-filtered mode (brand is not None)
        ----------------------------------------
        Runs a semantic vector search with a ChromaDB ``where`` pre-filter so
        that only documents whose ``brand`` metadata exactly matches the
        provided value are ranked.  This is the "precision path" – used when
        the user has confirmed a specific brand during the clarification loop.

        Fallback: if the brand filter produces zero results (common while the
        brand field is not yet fully populated in the index), the query
        automatically retries without the filter so the caller always gets a
        useful response.  A ``brand_filter_fallback=True`` flag is NOT set on
        the results because the Pydantic model is kept minimal; the LangGraph
        node should treat low similarity_score values as a signal that the
        brand match was uncertain.

        Unfiltered + keyword-boost mode (brand is None)
        ------------------------------------------------
        Runs a standard semantic search for ``top_k`` results, then iterates
        the returned metadatas and adds ``+{_KEYWORD_BOOST}`` to the
        similarity_score of any document whose ``exact_name`` field contains
        ``query.lower()`` as a substring.  This re-ranks near-ties in favour
        of exact lexical matches without discarding semantically similar results.
        Results are re-sorted by the adjusted score descending before returning.

        Parameters
        ----------
        query:
            Free-text ingredient description (e.g. "grilled chicken breast").
        brand:
            Optional brand string extracted by the LangGraph agent
            (e.g. "Tyson").  When provided, ChromaDB's ``where`` filter
            restricts the vector search to documents with a matching brand.
        top_k:
            Maximum number of results to return (1–50).

        Returns
        -------
        list[RetrievedIngredient]
            Up to ``top_k`` results ordered by similarity_score descending.
            Returns an empty list if the index contains no documents.

        Raises
        ------
        Exception
            Re-raises ChromaDB or model initialisation errors so the calling
            FastAPI endpoint can translate them into appropriate HTTP codes.
        """
        collection = self._get_collection()
        model = self._get_model()

        query_embedding: list[float] = model.encode(
            query.strip(), show_progress_bar=False
        ).tolist()

        if brand is not None:
            results = self._search_with_brand(
                collection, query_embedding, brand, top_k
            )
        else:
            results = self._search_semantic_boosted(
                collection, query_embedding, query, top_k
            )

        return results

    def _search_with_brand(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        brand: str,
        top_k: int,
    ) -> list[RetrievedIngredient]:
        """
        Vector search pre-filtered by brand exact-match.

        Falls back to an unfiltered search if the brand filter returns no
        results, so the caller always receives candidates even when brand
        coverage in the index is sparse.
        """
        try:
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"brand": {"$eq": brand}},
                include=["metadatas", "distances"],
            )
            ids = raw.get("ids", [[]])[0]
            dists = raw.get("distances", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]

            if ids:
                return [
                    self._build_ingredient(ids[i], float(dists[i]), metas[i] or {})
                    for i in range(len(ids))
                ]
        except Exception:
            # ChromaDB raises if the where clause references a field that does
            # not exist in any document; treat this as "no results".
            pass

        # Fallback: unfiltered semantic search
        return self._search_semantic_boosted(
            collection, query_embedding, brand, top_k
        )

    def _search_semantic_boosted(
        self,
        collection: chromadb.Collection,
        query_embedding: list[float],
        query: str,
        top_k: int,
    ) -> list[RetrievedIngredient]:
        """
        Semantic vector search with a post-hoc keyword-substring score boost.

        After the standard ChromaDB ranking, any result whose ``exact_name``
        contains the lowercased query as a substring receives a +0.15 bonus
        to its similarity_score.  The final list is re-sorted by this adjusted
        score so that lexically close matches surface above semantically
        similar but textually distant ones.
        """
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        query_lower = query.strip().lower()
        ingredients: list[RetrievedIngredient] = []

        for i in range(len(ids)):
            meta = metas[i] or {}
            base_score = _distance_to_score(float(dists[i]))

            exact_name = str(meta.get("exact_name") or meta.get("name") or "")
            keyword_hit = query_lower in exact_name.lower()
            adjusted_score = base_score + (_KEYWORD_BOOST if keyword_hit else 0.0)

            ingredients.append(
                self._build_ingredient(
                    ids[i], float(dists[i]), meta, score_override=adjusted_score
                )
            )

        ingredients.sort(key=lambda r: r.similarity_score, reverse=True)
        return ingredients
