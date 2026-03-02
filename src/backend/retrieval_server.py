"""
FastAPI retrieval server for NutriGraph.

Exposes a retrieval endpoint that takes a list of ingredient texts and returns
the closest matches from the ChromaDB ingredient index.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

# Ensure the project root is on sys.path so src.* imports resolve correctly
# when the server is launched from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.models import (  # noqa: E402
    AnalyzedIngredient,
    DishAnalysisResponse,
    IngredientQuery,
    RetrievalResponse,
)
from src.ml.extract_ingredients import extract_ingredients_from_image  # noqa: E402
from src.backend.retriever import HybridNutritionRetriever  # noqa: E402


PROJECT_ROOT = _PROJECT_ROOT
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "nutrigraph_ingredients"


class IngredientRetrievalRequest(BaseModel):
    """Request body for ingredient retrieval."""

    ingredients: List[str] = Field(
        ...,
        min_length=1,
        description="List of ingredient names or phrases to search for.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of closest matches to return per query ingredient.",
    )

    @field_validator("ingredients")
    @classmethod
    def ingredients_non_empty(cls, v: List[str]) -> List[str]:
        """Reject blank or whitespace-only ingredients (422)."""
        for i, s in enumerate(v):
            if not s or not s.strip():
                raise ValueError(
                    f"Ingredient at index {i} is empty or whitespace-only; "
                    "each ingredient must be non-empty after trimming."
                )
        return v


class IngredientMatch(BaseModel):
    """Single match from the ingredient index."""

    id: str
    name: str
    source: str
    distance: float = Field(
        ...,
        description="Chroma distance (lower = closer match).",
    )
    energy_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    fdc_id: Optional[int] = None


class IngredientRetrievalItem(BaseModel):
    """One entry in the response: one input ingredient and its matches."""

    query: str = Field(..., description="Input ingredient string (preserves order and duplicates).")
    matches: List[IngredientMatch] = Field(default_factory=list)


class IngredientRetrievalResponse(BaseModel):
    """Response: ordered list of (query, matches), one per input ingredient."""

    results: List[IngredientRetrievalItem]


app = FastAPI(
    title="NutriGraph Retrieval API",
    version="0.1.0",
    description="Retrieval endpoints for NutriGraph ingredient index.",
)


_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_hybrid_retriever: HybridNutritionRetriever | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_hybrid_retriever() -> HybridNutritionRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridNutritionRetriever()
    return _hybrid_retriever


def _get_collection_or_raise() -> chromadb.Collection:
    """Return the Chroma collection or raise HTTP 503 with setup instructions."""
    try:
        return _get_collection()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ingredient index not available. Run the indexing step first: "
                "python scripts/dataset/index_ingredients.py (after download and clean)."
            ),
        ) from e


@app.get("/health", tags=["health"])
def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/api/v1/ingredients/retrieve",
    response_model=IngredientRetrievalResponse,
    tags=["retrieval"],
)
def retrieve_ingredients(payload: IngredientRetrievalRequest) -> IngredientRetrievalResponse:
    """
    Retrieve closest ingredient matches from the vector index.

    Returns one result per input ingredient (order and duplicates preserved).
    Uses Chroma distance: lower = closer match.
    """
    collection = _get_collection_or_raise()
    model = _get_embedding_model()

    # Preserve order and duplicates; validator ensures each item non-empty
    queries = [s.strip() for s in payload.ingredients]
    query_embeddings = model.encode(queries, show_progress_bar=False).tolist()
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=payload.top_k,
    )

    out: List[IngredientRetrievalItem] = []

    for q_idx, query_text in enumerate(payload.ingredients):
        ids = result.get("ids", [[]])[q_idx]
        dists = result.get("distances", [[]])[q_idx]
        metadatas = result.get("metadatas", [[]])[q_idx]

        matches: List[IngredientMatch] = []
        for idx, doc_id in enumerate(ids):
            if idx >= len(dists):
                continue
            distance = float(dists[idx])
            meta = metadatas[idx] if idx < len(metadatas) else {}
            meta = meta or {}

            matches.append(
                IngredientMatch(
                    id=str(doc_id),
                    name=str(meta.get("name", "")),
                    source=str(meta.get("source", "")),
                    distance=distance,
                    energy_kcal=meta.get("energy_kcal"),
                    protein_g=meta.get("protein_g"),
                    carbohydrates_g=meta.get("carbohydrates_g"),
                    fat_g=meta.get("fat_g"),
                    fdc_id=meta.get("fdc_id"),
                )
            )

        out.append(IngredientRetrievalItem(query=query_text, matches=matches))

    return IngredientRetrievalResponse(results=out)


# ── Hybrid single-ingredient retrieval (called by the LangGraph agent) ────────

@app.post(
    "/api/v1/retrieve-ingredient",
    response_model=RetrievalResponse,
    tags=["retrieval"],
    summary="Hybrid semantic + brand-filtered ingredient lookup",
)
async def retrieve_ingredient(payload: IngredientQuery) -> RetrievalResponse:
    """
    Hybrid retrieval endpoint for the LangGraph clarification agent.

    Accepts a single ingredient query and an optional brand name resolved
    during the clarification loop.  Delegates to
    :class:`~src.backend.retriever.HybridNutritionRetriever` which applies
    one of two strategies:

    - **Brand-filtered** (``brand`` present): ChromaDB ``where`` pre-filter
      restricts the vector search to documents whose ``brand`` metadata
      matches exactly.  Automatically falls back to unfiltered search if no
      brand-tagged documents exist in the index yet.

    - **Semantic + keyword boost** (``brand`` absent): Standard vector search
      followed by a ``+0.15`` score boost for any result whose ``exact_name``
      contains the query as a substring.  Re-ranked by adjusted score.

    Returns up to ``top_k`` results ordered by similarity_score descending.

    Raises
    ------
    HTTP 404
        When the retriever returns no results at all (index may be empty or
        the query is too far from any indexed document).
    HTTP 500
        On ChromaDB initialisation or query errors (e.g. index not built yet,
        corrupt data directory).
    """
    try:
        retriever = _get_hybrid_retriever()
        results = retriever.search_ingredient(
            query=payload.query,
            brand=payload.brand,
            top_k=payload.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "ChromaDB index not found. Run the indexing step first: "
                "python scripts/dataset/index_ingredients.py"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval error: {exc}",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No close matches found for '{payload.query}'"
                + (f" (brand: '{payload.brand}')" if payload.brand else "")
                + ". Try a broader query or omit the brand filter."
            ),
        )

    return RetrievalResponse(results=results)


# ── Dish image analysis ───────────────────────────────────────────────────────

def _distance_to_confidence(distance: float) -> float:
    """
    Convert a ChromaDB distance score to a confidence value in [0, 1].

    Uses a sigmoid-style mapping that works for both L2 and cosine distances:
    - distance 0.0  → confidence ~1.0  (perfect match)
    - distance 1.0  → confidence ~0.5
    - distance 2.0+ → confidence approaching 0
    """
    return round(1.0 / (1.0 + distance), 4)


def _lookup_nutrition(
    ingredient_names: List[str],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> Dict[str, dict]:
    """
    Query ChromaDB for the best nutritional match for each ingredient name.

    Returns a mapping of ``{ingredient_name: {energy_kcal, protein_g, carbohydrates_g, fat_g, confidence}}``.
    Ingredients with no index match default to zeros.
    """
    if not ingredient_names:
        return {}

    embeddings = model.encode(ingredient_names, show_progress_bar=False).tolist()
    result = collection.query(query_embeddings=embeddings, n_results=1)

    nutrition_map: Dict[str, dict] = {}
    for idx, name in enumerate(ingredient_names):
        distances = result.get("distances", [[]])[idx]
        metadatas = result.get("metadatas", [[]])[idx]

        if distances and metadatas:
            distance = float(distances[0])
            meta = metadatas[0] or {}
            nutrition_map[name] = {
                "energy_kcal": float(meta.get("energy_kcal") or 0.0),
                "protein_g": float(meta.get("protein_g") or 0.0),
                "carbohydrates_g": float(meta.get("carbohydrates_g") or 0.0),
                "fat_g": float(meta.get("fat_g") or 0.0),
                "confidence": _distance_to_confidence(distance),
            }
        else:
            nutrition_map[name] = {
                "energy_kcal": 0.0,
                "protein_g": 0.0,
                "carbohydrates_g": 0.0,
                "fat_g": 0.0,
                "confidence": 0.0,
            }

    return nutrition_map


@app.post(
    "/api/v1/analyze-dish",
    response_model=DishAnalysisResponse,
    tags=["analysis"],
    summary="Analyze a dish photo and return a full nutritional breakdown",
)
async def analyze_dish(
    file: UploadFile = File(..., description="JPEG or PNG photo of the dish to analyze"),
) -> DishAnalysisResponse:
    """
    Full image-to-nutrition pipeline:

    1. **Gemini 2.5 Flash Lite** identifies the dish name and ingredient list from the photo.
    2. **ChromaDB RAG retrieval** looks up the best nutritional match for each ingredient.
    3. Returns a :class:`DishAnalysisResponse` with per-ingredient macros and dish-level totals.

    Requires `VERTEXAI_API_KEY` to be set in the environment (or `.env` file).
    """
    # ── 1. Read image bytes ───────────────────────────────────────────────────
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    mime_type = file.content_type or "image/jpeg"

    # ── 2. Gemini: extract dish name + ingredients ────────────────────────────
    try:
        dish_info = extract_ingredients_from_image(image_bytes, mime_type=mime_type)
    except ValueError as exc:
        # Missing API key or unparseable model response
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Vertex AI returned a non-2xx response
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error during image analysis: {exc}"
        ) from exc

    dish_name: str = dish_info.get("dish_name", "Analyzed Dish")
    ingredient_names: List[str] = [
        i.strip() for i in dish_info.get("ingredients", []) if i and i.strip()
    ]

    # ── 3. ChromaDB: look up nutrition for each ingredient ────────────────────
    collection = _get_collection_or_raise()
    embed_model = _get_embedding_model()
    nutrition_map = _lookup_nutrition(ingredient_names, collection, embed_model)

    # ── 4. Assemble response ──────────────────────────────────────────────────
    analyzed: List[AnalyzedIngredient] = []
    for name in ingredient_names:
        n = nutrition_map.get(name, {})
        analyzed.append(
            AnalyzedIngredient(
                name=name,
                confidence_score=n.get("confidence", 0.0),
                calories=n.get("energy_kcal", 0.0),
                protein=n.get("protein_g", 0.0),
                carbs=n.get("carbohydrates_g", 0.0),
                fat=n.get("fat_g", 0.0),
            )
        )

    return DishAnalysisResponse(
        dish_name=dish_name,
        total_calories=round(sum(a.calories for a in analyzed), 1),
        total_protein=round(sum(a.protein for a in analyzed), 1),
        total_carbs=round(sum(a.carbs for a in analyzed), 1),
        total_fat=round(sum(a.fat for a in analyzed), 1),
        ingredients=analyzed,
    )

