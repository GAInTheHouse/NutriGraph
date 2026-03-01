"""
FastAPI retrieval server for NutriGraph.

Exposes a retrieval endpoint that takes a list of ingredient texts and returns
the closest matches from the ChromaDB ingredient index.
"""

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


