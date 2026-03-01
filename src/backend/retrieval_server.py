"""
FastAPI retrieval server for NutriGraph.

Exposes a retrieval endpoint that takes a list of ingredient texts and returns
the closest matches from the ChromaDB ingredient index.
"""

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION_NAME = "nutrigraph_ingredients"


class IngredientRetrievalRequest(BaseModel):
    """Request body for ingredient retrieval."""

    ingredients: List[str] = Field(
        ...,
        min_items=1,
        description="List of ingredient names or phrases to search for.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of closest matches to return per query ingredient.",
    )


class IngredientMatch(BaseModel):
    """Single match from the ingredient index."""

    id: str
    name: str
    source: str
    score: float = Field(
        ...,
        description="Similarity score (lower distance = closer match).",
    )
    energy_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    fat_g: Optional[float] = None
    fdc_id: Optional[int] = None


class IngredientRetrievalResponse(BaseModel):
    """Response mapping each query ingredient to its matches."""

    results: Dict[str, List[IngredientMatch]]


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

    For each input ingredient string, returns up to `top_k` closest matches
    from the `nutrigraph_ingredients` ChromaDB collection.
    """
    collection = _get_collection()
    model = _get_embedding_model()

    queries = [text.strip() for text in payload.ingredients if text.strip()]
    if not queries:
        return IngredientRetrievalResponse(results={})

    query_embeddings = model.encode(queries, show_progress_bar=False).tolist()
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=payload.top_k,
    )

    out: Dict[str, List[IngredientMatch]] = {}

    for q_idx, query_text in enumerate(queries):
        ids = result.get("ids", [[]])[q_idx]
        dists = result.get("distances", [[]])[q_idx]
        metadatas = result.get("metadatas", [[]])[q_idx]

        matches: List[IngredientMatch] = []
        for idx, doc_id in enumerate(ids):
            meta = metadatas[idx] or {}
            distance = float(dists[idx]) if idx < len(dists) else 0.0

            matches.append(
                IngredientMatch(
                    id=str(doc_id),
                    name=str(meta.get("name", "")),
                    source=str(meta.get("source", "")),
                    score=distance,
                    energy_kcal=meta.get("energy_kcal"),
                    protein_g=meta.get("protein_g"),
                    carbohydrates_g=meta.get("carbohydrates_g"),
                    fat_g=meta.get("fat_g"),
                    fdc_id=meta.get("fdc_id"),
                )
            )

        out[query_text] = matches

    return IngredientRetrievalResponse(results=out)


