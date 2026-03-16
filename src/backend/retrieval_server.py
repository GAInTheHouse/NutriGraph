"""
FastAPI retrieval server for NutriGraph.

Exposes a retrieval endpoint that takes a list of ingredient texts and returns
the closest matches from the ChromaDB ingredient index.
"""

import json
import logging
import os
import re
import sys
import threading
import time as _time
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import requests

import chromadb
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

# Ensure the project root is on sys.path so src.* imports resolve correctly
# when the server is launched from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

from src.core.models import AnalyzedIngredient, Dish, DishAnalysisResponse, NutritionEstimate, PlacesResponse  # noqa: E402
from src.ml.extract_ingredients import extract_ingredients_from_image, _HISTORICAL_CONTEXT_SNIPPET  # noqa: E402
from src.backend.places_client import GooglePlacesClient, PlacesAPIError  # noqa: E402
from src.backend.database import engine, get_db  # noqa: E402
from src.backend.db_models import Base, DishRecord  # noqa: E402
from src.backend.crud import get_dish_record, get_dishes_for_place, save_dish_record  # noqa: E402


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


# ── Create DB tables on startup (no-op if they already exist) ─────────────────
@app.on_event("startup")
def _create_tables() -> None:
    Base.metadata.create_all(bind=engine)


# ── Pydantic models for the new persistence endpoints ─────────────────────────

class SaveDishRequest(BaseModel):
    """Payload sent by a diner when they explicitly save their analysis."""
    analysis: DishAnalysisResponse
    place_id: Optional[str] = None


class PublishDishRequest(BaseModel):
    """Payload sent by a restaurant owner to publish verified dish macros."""
    dish_name: str = Field(..., min_length=1)
    place_id: str = Field(..., min_length=1)
    calories: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)
    carbs: float = Field(..., ge=0)
    fat: float = Field(..., ge=0)
    ingredients: List[dict] = Field(default_factory=list)


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


@app.get(
    "/api/v1/places/search",
    response_model=PlacesResponse,
    tags=["places"],
    summary="Search for restaurants via the Google Places API",
)
def search_places(
    query: str = Query(..., min_length=1, description="Restaurant name or search phrase"),
) -> PlacesResponse:
    """
    Proxy the Google Places Text Search API and return matching restaurant establishments.

    Results are filtered to ``type=restaurant`` so non-food businesses are excluded.
    Requires ``GOOGLE_PLACES_API_KEY`` to be set in the environment; returns HTTP 503
    with a clear message when the key is absent.
    """
    try:
        places_client = GooglePlacesClient()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Places integration is not configured. "
                "Set GOOGLE_PLACES_API_KEY in your environment or .env file."
            ),
        ) from exc

    try:
        results = places_client.search_restaurants(query)
    except PlacesAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error during Places search: {exc}"
        ) from exc

    return PlacesResponse(results=results)


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


# ── Restaurant builder: ingredient-list → NutritionEstimate ──────────────────

# Maps unit strings (matching settings.DEFAULT_UNITS) to their gram equivalent.
# Used to scale per-100g ChromaDB macro values by each ingredient's actual quantity.
_UNIT_TO_GRAMS: Dict[str, float] = {
    "g": 1.0,
    "oz": 28.35,
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "piece": 100.0,
    "ml": 1.0,
}


@app.post(
    "/api/v1/builder/generate",
    response_model=NutritionEstimate,
    tags=["builder"],
    summary="Calculate dish-level macros from an explicit ingredient list via ChromaDB RAG",
)
def builder_generate_profile(payload: Dish) -> NutritionEstimate:
    """
    Restaurant builder endpoint: given a :class:`Dish` with named ingredients,
    quantities, and units, look up per-100g nutritional values from ChromaDB for
    each ingredient, scale by the actual quantity, sum to dish-level totals, and
    return a :class:`NutritionEstimate`.

    Unlike the image-analysis pipeline, no LLM is involved — the calculation is
    deterministic and reproducible from the ingredient list.

    Confidence is the average of the individual ChromaDB retrieval confidence
    scores, giving the restaurant owner a signal about how well each ingredient
    was matched in the index.
    """
    collection = _get_collection_or_raise()
    embed_model = _get_embedding_model()

    ingredient_names = [ing.name.strip() for ing in payload.ingredients if ing.name.strip()]
    nutrition_map = _lookup_nutrition(ingredient_names, collection, embed_model)

    total_cal = total_pro = total_carb = total_fat = 0.0
    confidence_scores: List[float] = []

    for ing in payload.ingredients:
        name = ing.name.strip()
        if not name:
            continue
        if ing.unit not in _UNIT_TO_GRAMS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unsupported unit '{ing.unit}' for ingredient '{name}'. "
                    f"Accepted units: {sorted(_UNIT_TO_GRAMS)}."
                ),
            )
        n = nutrition_map.get(name, {})
        # Scale the per-100g ChromaDB values by the actual quantity in grams.
        factor = ing.quantity * _UNIT_TO_GRAMS[ing.unit] / 100.0
        total_cal   += n.get("energy_kcal",     0.0) * factor
        total_pro   += n.get("protein_g",        0.0) * factor
        total_carb  += n.get("carbohydrates_g",  0.0) * factor
        total_fat   += n.get("fat_g",            0.0) * factor
        confidence_scores.append(n.get("confidence", 0.0))

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )

    return NutritionEstimate(
        calories=round(total_cal, 1),
        protein_g=round(total_pro, 1),
        carbs_g=round(total_carb, 1),
        fat_g=round(total_fat, 1),
        confidence=round(avg_confidence, 4),
    )


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
def analyze_dish(
    file: UploadFile = File(..., description="JPEG or PNG photo of the dish to analyze"),
    restaurant_context: Optional[str] = Form(
        None,
        description=(
            "Optional restaurant name (or 'Home Cooked') selected by the user. "
            "When provided, the Gemini prompt is augmented with establishment context "
            "to improve dish-name and ingredient accuracy."
        ),
    ),
    dish_name: Optional[str] = Form(
        None,
        description=(
            "Optional dish-name hint provided by the user before analysis. "
            "When supplied, the database is checked first: a restaurant-verified record "
            "is returned immediately (fast path); diner records are used as coaching "
            "context for the Gemini prompt (context path)."
        ),
    ),
    restaurant_place_id: Optional[str] = Form(
        None,
        description="Google Places place_id of the selected restaurant, or None for home-cooked.",
    ),
    db: Session = Depends(get_db),
) -> DishAnalysisResponse:
    """
    Full image-to-nutrition pipeline with database-backed caching.

    **Routing logic** (only applied when ``dish_name`` is provided):

    1. **Fast path** — if the database contains a restaurant-verified record for this
       ``(dish_name, place_id)`` pair, it is returned immediately without calling Gemini.
    2. **Context path** — if the database contains diner records, their macro averages
       are injected into the Gemini prompt as coaching context for consistency.
    3. **Standard path** — no prior records; Gemini runs with no extra context.

    After the LLM pipeline runs (paths 2 & 3), the result is returned to the UI.
    The user decides whether to save it via the separate ``/diner/save-dish`` endpoint.

    Requires ``VERTEXAI_API_KEY`` to be set in the environment (or ``.env`` file).
    """
    # ── 1. Read image bytes ───────────────────────────────────────────────────
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    mime_type = file.content_type or "image/jpeg"
    resolved_context = restaurant_context.strip() if restaurant_context else None
    resolved_dish_name = dish_name.strip() if dish_name else None
    resolved_place_id = restaurant_place_id.strip() if restaurant_place_id else None

    # ── 2. Database routing (only when the user supplied a dish name hint) ────
    historical_context_str: Optional[str] = None
    data_source = "ai_generated"

    if resolved_dish_name:
        records: List[DishRecord] = get_dish_record(db, resolved_dish_name, resolved_place_id)

        if records and records[0].source == "restaurant":
            # ── Fast path: return the restaurant-verified record immediately ──
            record = records[0]
            try:
                cached_ingredients: List[AnalyzedIngredient] = [
                    AnalyzedIngredient(**ing)
                    for ing in json.loads(record.ingredients_json or "[]")
                ]
            except Exception:
                cached_ingredients = []

            return DishAnalysisResponse(
                dish_name=record.dish_name,
                total_calories=record.calories,
                total_protein=record.protein,
                total_carbs=record.carbs,
                total_fat=record.fat,
                ingredients=cached_ingredients,
                is_cached=True,
                data_source="restaurant_verified",
            )

        if records:
            # ── Context path: build historical coaching string from diner records ──
            avg_cal = sum(r.calories for r in records) / len(records)
            avg_pro = sum(r.protein for r in records) / len(records)
            avg_carb = sum(r.carbs for r in records) / len(records)
            avg_fat = sum(r.fat for r in records) / len(records)

            # Collect all previously identified ingredient names (deduplicated).
            all_ing_names: list[str] = []
            for r in records:
                try:
                    for ing in json.loads(r.ingredients_json or "[]"):
                        name = ing.get("name", "")
                        if name and name not in all_ing_names:
                            all_ing_names.append(name)
                except Exception:
                    pass
            ing_list_str = ", ".join(all_ing_names[:15]) if all_ing_names else "not available"

            historical_context_str = _HISTORICAL_CONTEXT_SNIPPET.format(
                calories=round(avg_cal, 1),
                protein=round(avg_pro, 1),
                carbs=round(avg_carb, 1),
                fat=round(avg_fat, 1),
                ingredients=ing_list_str,
            )
            data_source = "diner_cached"

    # ── 3. Gemini: extract dish name + ingredients ────────────────────────────
    try:
        dish_info = extract_ingredients_from_image(
            image_bytes,
            mime_type=mime_type,
            restaurant_context=resolved_context,
            historical_context=historical_context_str,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error during image analysis: {exc}"
        ) from exc

    llm_dish_name: str = dish_info.get("dish_name", "Analyzed Dish").strip() or "Analyzed Dish"
    ingredient_names: List[str] = [
        i.strip() for i in dish_info.get("ingredients", []) if i and i.strip()
    ]

    # ── 4. ChromaDB: look up nutrition for each ingredient ────────────────────
    collection = _get_collection_or_raise()
    embed_model = _get_embedding_model()
    nutrition_map = _lookup_nutrition(ingredient_names, collection, embed_model)

    # ── 5. Assemble and return response (no auto-save) ────────────────────────
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
        dish_name=llm_dish_name,
        total_calories=round(sum(a.calories for a in analyzed), 1),
        total_protein=round(sum(a.protein for a in analyzed), 1),
        total_carbs=round(sum(a.carbs for a in analyzed), 1),
        total_fat=round(sum(a.fat for a in analyzed), 1),
        ingredients=analyzed,
        is_cached=False,
        data_source=data_source,
    )


# ── Diner: explicit save endpoint ─────────────────────────────────────────────

@app.post(
    "/api/v1/diner/save-dish",
    tags=["analysis"],
    summary="Explicitly save a diner's reviewed nutritional analysis to the database",
)
def save_diner_dish(
    payload: SaveDishRequest,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Called by the Diner UI when the user clicks "Save Results" or
    "Save Refined Results".  Persists the provided :class:`DishAnalysisResponse`
    as a ``source="diner"`` record.

    The LLM pipeline never auto-saves; this endpoint gives the user full
    control over when their data enters the database.
    """
    save_dish_record(db, payload.analysis, payload.place_id, source="diner")
    db.commit()
    return {"status": "saved"}


# ── Restaurant: publish endpoint ──────────────────────────────────────────────

@app.post(
    "/api/v1/restaurant/publish-dish",
    tags=["restaurant"],
    summary="Publish verified dish macros from a restaurant owner",
)
def publish_restaurant_dish(
    payload: PublishDishRequest,
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """
    Called by the Restaurant UI when the owner clicks "Publish to Global Catalog".
    Stores the dish as ``source="restaurant"`` so future diner analyses for the same
    ``(dish_name, place_id)`` pair are served from this verified record instantly.
    """
    # Build a DishAnalysisResponse so we can reuse save_dish_record.
    try:
        ingredients = [AnalyzedIngredient(**ing) for ing in payload.ingredients]
    except Exception:
        ingredients = []

    analysis = DishAnalysisResponse(
        dish_name=payload.dish_name.strip(),
        total_calories=payload.calories,
        total_protein=payload.protein,
        total_carbs=payload.carbs,
        total_fat=payload.fat,
        ingredients=ingredients,
        is_cached=False,
        data_source="restaurant_verified",
    )
    save_dish_record(db, analysis, payload.place_id.strip(), source="restaurant")
    db.commit()
    return {"status": "published"}


# ── Restaurant: list published dishes ─────────────────────────────────────────

@app.get(
    "/api/v1/restaurant/dishes",
    tags=["restaurant"],
    summary="List all restaurant-verified dishes for a given place_id",
)
def list_restaurant_dishes(
    place_id: str = Query(..., description="Google Places place_id for the restaurant"),
    db: Session = Depends(get_db),
) -> Dict:
    """
    Returns all ``source="restaurant"`` records stored for the given place_id.
    Used by the Restaurant UI to reload the catalog across sessions.
    """
    records = get_dishes_for_place(db, place_id, source="restaurant")
    dishes = [
        {
            "name": r.dish_name,
            "serving_size": None,
            "ingredient_count": len(json.loads(r.ingredients_json or "[]")),
            "calories": r.calories,
            "protein_g": r.protein,
            "carbs_g": r.carbs,
            "fat_g": r.fat,
            "confidence": None,
        }
        for r in records
    ]
    return {"dishes": dishes}


# ── Eval: ingredient-list analysis with multi-turn clarification ──────────────
#
# These three endpoints power the offline evaluation framework (src/eval/).
# They implement the multi-turn clarification protocol defined in the eval plan:
#
#   Round 1  POST /api/v1/analyze-dish-from-ingredients
#            → {status: "needs_clarification", session_id, questions}
#            → {status: "complete", dish_id, total_*, confidence, num_questions}
#
#   Round N  POST /api/v1/analyze-dish-from-ingredients/respond
#            → same two shapes as above
#
#   Judge    POST /api/v1/judge-nutrition
#            → {score: float 0-10, explanation: str}

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

@dataclass
class _AnalysisSession:
    """State persisted between clarification rounds for a single dish."""

    dish_id: str
    dish_name: str
    current_ingredients: List[str]   # refined after each round
    grams: List[float]               # gram quantities per ingredient (fixed)
    low_conf_indices: List[int]      # ingredient indices still low-confidence
    num_questions_total: int         # cumulative questions asked so far
    created_at: float = dc_field(default_factory=_time.time)


_analysis_sessions: Dict[str, _AnalysisSession] = {}
_sessions_lock = threading.Lock()
_SESSION_TTL = 600.0  # seconds; sessions older than this are purged


def _evict_expired_sessions() -> None:
    now = _time.time()
    with _sessions_lock:
        expired = [k for k, v in _analysis_sessions.items() if now - v.created_at > _SESSION_TTL]
        for k in expired:
            del _analysis_sessions[k]


def _get_session(session_id: str) -> _AnalysisSession:
    with _sessions_lock:
        s = _analysis_sessions.get(session_id)
    if s is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired (TTL={int(_SESSION_TTL)}s).",
        )
    return s


def _save_session(session_id: str, session: _AnalysisSession) -> None:
    with _sessions_lock:
        _analysis_sessions[session_id] = session


def _delete_session(session_id: str) -> None:
    with _sessions_lock:
        _analysis_sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Ingredient-list parser
# ---------------------------------------------------------------------------

def _parse_ingredients_list(ingredients_list: str) -> tuple:
    """
    Split a semicolon-separated ingredient string into parallel name / gram lists.

    Each segment is expected to end with a gram quantity such as ``150g`` or
    ``50.5 g``.  Segments without a quantity default to 100 g.

    Segments whose name is empty after the quantity is stripped (e.g. a bare
    ``"150g"`` token) are silently skipped so that ``names`` and ``grams``
    always remain strictly index-aligned.

    Example::

        'romaine lettuce 150g; grilled chicken breast (no skin) 120g'
        → (['romaine lettuce', 'grilled chicken breast (no skin)'], [150.0, 120.0])
    """
    names: List[str] = []
    grams: List[float] = []
    for raw in ingredients_list.split(";"):
        part = raw.strip()
        if not part:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*g\s*$", part, re.IGNORECASE)
        if m:
            name = part[: m.start()].strip()
            gram = float(m.group(1))
        else:
            name = part
            gram = 100.0
        if not name:
            # Segment was a bare quantity with no ingredient name; skip both
            # values so names↔grams indices never diverge.
            continue
        names.append(name)
        grams.append(gram)
    return names, grams


# ---------------------------------------------------------------------------
# Clarification graph (lazy singleton)
# ---------------------------------------------------------------------------

_clarification_graph = None
_clarification_graph_lock = threading.Lock()


def _get_clarification_graph():
    """Return a compiled LangGraph clarification graph (built once, reused)."""
    global _clarification_graph
    if _clarification_graph is not None:
        return _clarification_graph
    with _clarification_graph_lock:
        if _clarification_graph is None:
            from src.backend.clarification_graph import build_clarification_graph
            _clarification_graph = build_clarification_graph()
    return _clarification_graph


# ---------------------------------------------------------------------------
# Nutrition aggregation
# ---------------------------------------------------------------------------

def _compute_nutrition_from_graph_state(state: dict, grams: List[float]) -> dict:
    """
    Scale each ingredient's best ChromaDB match macros by its actual gram
    quantity, then sum to dish-level totals.

    Returns a dict with keys ``total_calories``, ``total_protein``,
    ``total_carbs``, ``total_fat``, ``total_fiber``, ``confidence``.
    """
    matches_list: List[List[dict]] = state.get("matches", [])
    scores: List[float] = state.get("scores", [])

    total_cal = total_pro = total_carb = total_fat = 0.0
    conf_scores: List[float] = []

    for idx, ingredient_matches in enumerate(matches_list):
        gram_qty = grams[idx] if idx < len(grams) else 100.0
        factor = gram_qty / 100.0
        if idx < len(scores):
            conf_scores.append(scores[idx])
        if not ingredient_matches:
            continue
        best = ingredient_matches[0]  # already sorted by combined score descending
        total_cal += float(best.get("energy_kcal") or 0.0) * factor
        total_pro += float(best.get("protein_g") or 0.0) * factor
        total_carb += float(best.get("carbohydrates_g") or 0.0) * factor
        total_fat += float(best.get("fat_g") or 0.0) * factor
        # fiber_g is not stored in the ChromaDB index; it defaults to 0

    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
    return {
        "total_calories": round(total_cal, 1),
        "total_protein": round(total_pro, 1),
        "total_carbs": round(total_carb, 1),
        "total_fat": round(total_fat, 1),
        "total_fiber": 0.0,
        "confidence": round(avg_conf, 4),
    }


# ---------------------------------------------------------------------------
# Pydantic request models for the eval endpoints
# ---------------------------------------------------------------------------

class AnalyzeDishFromIngredientsRequest(BaseModel):
    """Round 1 request payload for ingredient-list dish analysis."""

    dish_id: str
    dish_name: str
    context_type: str = ""
    cuisine: str = ""
    serving_description: str = ""
    serving_size_grams: float = 0.0
    ingredients_list: str
    preparation_notes: str = ""


class AnalyzeDishRespondRequest(BaseModel):
    """Follow-up request carrying answers to the previous round's questions."""

    session_id: str
    answers: List[str]


class JudgeNutritionRequest(BaseModel):
    """Request to the LLM judge: ground-truth and predicted macro dicts."""

    dish_id: Optional[str] = None
    ground_truth: Dict[str, float]
    prediction: Dict[str, float]


# ---------------------------------------------------------------------------
# Shared orchestration helper
# ---------------------------------------------------------------------------

def _run_graph_and_respond(
    dish_id: str,
    dish_name: str,
    ingredients: List[str],
    grams: List[float],
    num_questions_so_far: int,
    existing_session_id: Optional[str] = None,
) -> dict:
    """
    Invoke the clarification graph and return the appropriate response dict.

    If the graph produces clarification questions, a session is created (or
    updated) and ``{"status": "needs_clarification", ...}`` is returned.
    When the graph is satisfied, the session is cleaned up and
    ``{"status": "complete", ...}`` with full nutritional totals is returned.
    """
    try:
        graph = _get_clarification_graph()
        state: dict = graph.invoke({"ingredients": ingredients, "dish_name": dish_name})
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Clarification graph failed: {exc}. "
                "Ensure the ingredient index is built: "
                "python scripts/dataset/index_ingredients.py"
            ),
        ) from exc

    questions: List[str] = state.get("questions", [])
    low_conf_indices: List[int] = state.get("low_conf_indices", [])

    if questions and low_conf_indices:
        # Agent needs more information — persist session and ask
        session_id = existing_session_id or str(uuid.uuid4())
        _save_session(
            session_id,
            _AnalysisSession(
                dish_id=dish_id,
                dish_name=dish_name,
                current_ingredients=list(ingredients),
                grams=list(grams),
                low_conf_indices=low_conf_indices,
                num_questions_total=num_questions_so_far + len(questions),
            ),
        )
        return {
            "status": "needs_clarification",
            "session_id": session_id,
            "questions": questions,
        }

    # Agent is satisfied — compute nutrition and return final result
    if existing_session_id:
        _delete_session(existing_session_id)

    nutrition = _compute_nutrition_from_graph_state(state, grams)
    return {
        "status": "complete",
        "dish_id": dish_id,
        "dish_name": dish_name,
        **nutrition,
        "num_questions": num_questions_so_far,
    }


# ---------------------------------------------------------------------------
# Route: Round 1
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze-dish-from-ingredients",
    tags=["eval"],
    summary="[Eval] Analyze a dish from a structured ingredients list (Round 1)",
)
def analyze_dish_from_ingredients(payload: AnalyzeDishFromIngredientsRequest) -> dict:
    """
    Entry point for the evaluation clarification loop.

    Parses the semi-colon separated ``ingredients_list``, runs the LangGraph
    clarification agent, and returns one of:

    * ``{"status": "needs_clarification", "session_id": "...", "questions": [...]}``
      — send answers to ``/api/v1/analyze-dish-from-ingredients/respond``.
    * ``{"status": "complete", "dish_id": ..., "total_calories": ..., ...}``
      — analysis finished with no clarification needed.
    """
    _evict_expired_sessions()
    ingredients, grams = _parse_ingredients_list(payload.ingredients_list)
    if not ingredients:
        raise HTTPException(
            status_code=422,
            detail="ingredients_list is empty or could not be parsed.",
        )
    return _run_graph_and_respond(
        dish_id=payload.dish_id,
        dish_name=payload.dish_name,
        ingredients=ingredients,
        grams=grams,
        num_questions_so_far=0,
    )


# ---------------------------------------------------------------------------
# Route: Follow-up rounds
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze-dish-from-ingredients/respond",
    tags=["eval"],
    summary="[Eval] Submit clarification answers and continue analysis",
)
def analyze_dish_respond(payload: AnalyzeDishRespondRequest) -> dict:
    """
    Submit answers to the questions returned by the previous round.

    The low-confidence ingredient strings are refined using the answers via
    ``refine_ingredients_batch``, then the clarification graph is re-run.
    Returns the same two shapes as the Round 1 endpoint until ``status == "complete"``.
    """
    _evict_expired_sessions()
    session = _get_session(payload.session_id)

    low_indices = session.low_conf_indices
    if len(payload.answers) != len(low_indices):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {len(low_indices)} answer(s) (one per question asked), "
                f"received {len(payload.answers)}."
            ),
        )

    # Refine the low-confidence ingredients using the answers
    refinement_warning: str | None = None
    try:
        from src.ml.clarification_questions import refine_ingredients_batch

        pairs = [
            (session.current_ingredients[idx], payload.answers[i])
            for i, idx in enumerate(low_indices)
            if idx < len(session.current_ingredients)
        ]
        refined = refine_ingredients_batch(pairs, dish_name=session.dish_name)
        updated = list(session.current_ingredients)
        for i, idx in enumerate(low_indices):
            if idx < len(updated) and i < len(refined):
                updated[idx] = refined[i]
    except Exception as exc:
        logger.warning(
            "refine_ingredients_batch failed for session %s (dish=%r); "
            "continuing with unrefined ingredients. Error: %s",
            payload.session_id,
            session.dish_name,
            exc,
            exc_info=True,
        )
        updated = list(session.current_ingredients)
        refinement_warning = (
            f"Ingredient refinement failed ({type(exc).__name__}: {exc}); "
            "analysis continued with unrefined ingredient strings."
        )

    response = _run_graph_and_respond(
        dish_id=session.dish_id,
        dish_name=session.dish_name,
        ingredients=updated,
        grams=session.grams,
        num_questions_so_far=session.num_questions_total,
        existing_session_id=payload.session_id,
    )
    if refinement_warning:
        response["refinement_warning"] = refinement_warning
    return response


# ---------------------------------------------------------------------------
# Route: Judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert nutritionist evaluating the accuracy of an AI nutritional analysis system.
Score the predicted macronutrient values against the ground truth on a scale from 0 to 10.

Scoring guide:
- 10  : All macros within 5 % of ground truth (excellent)
- 8-9 : Most macros within 10-15 %, only minor deviations (very good)
- 6-7 : Some macros off by 15-25 % (acceptable)
- 4-5 : Significant deviations in several macros, 25-40 % error range (below average)
- 2-3 : Large errors in most macros, > 40 % average error (poor)
- 0-1 : Completely inaccurate (very poor)

Weight each macro by nutritional importance when forming your overall score:
  Calories 40 %, Protein 20 %, Fat 20 %, Carbs 15 %, Fiber 5 %.

Ground truth : calories={gt_cal} kcal, protein={gt_pro}g, carbs={gt_carb}g, fat={gt_fat}g, fiber={gt_fiber}g
Prediction   : calories={pred_cal} kcal, protein={pred_pro}g, carbs={pred_carb}g, fat={pred_fat}g, fiber={pred_fiber}g

Respond with ONLY a valid JSON object (no markdown fences):
{{"score": <float 0-10>, "explanation": "<2-3 sentence explanation>"}}"""


def _call_vertex_ai_judge(prompt: str) -> str:
    """POST a text prompt to Gemini via Vertex AI REST and return the raw text."""
    api_key = os.environ.get("VERTEXAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="VERTEXAI_API_KEY is not set; the judge endpoint requires Vertex AI.",
        )
    model = os.environ.get("NUTRIGRAPH_JUDGE_MODEL", "gemini-2.0-flash")
    url = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    resp = requests.post(url, headers={"Content-Type": "application/json"}, json=body, timeout=30)
    if not resp.ok:
        raise HTTPException(
            status_code=502,
            detail=f"Vertex AI API error {resp.status_code}: {resp.text[:300]}",
        )
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Vertex AI response structure: {data}",
        ) from exc


@app.post(
    "/api/v1/judge-nutrition",
    tags=["eval"],
    summary="[Eval] Score predicted macros against ground truth using an LLM judge",
)
def judge_nutrition(payload: JudgeNutritionRequest) -> dict:
    """
    LLM-as-a-judge endpoint for the NutriGraph evaluation framework.

    Accepts ``ground_truth`` and ``prediction`` dicts (keys: ``calories_kcal``,
    ``protein_g``, ``carbs_g``, ``fat_g``, ``fiber_g``), calls Gemini to produce
    a quality score in [0, 10], and returns ``{"score": float, "explanation": str}``.
    """
    gt = payload.ground_truth
    pred = payload.prediction

    def _get(d: dict, *keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in d:
                return float(d[k])
        return default

    prompt = _JUDGE_PROMPT.format(
        gt_cal=_get(gt, "calories_kcal", "calories"),
        gt_pro=_get(gt, "protein_g", "protein"),
        gt_carb=_get(gt, "carbs_g", "carbs"),
        gt_fat=_get(gt, "fat_g", "fat"),
        gt_fiber=_get(gt, "fiber_g", "fiber"),
        pred_cal=_get(pred, "calories_kcal", "calories"),
        pred_pro=_get(pred, "protein_g", "protein"),
        pred_carb=_get(pred, "carbs_g", "carbs"),
        pred_fat=_get(pred, "fat_g", "fat"),
        pred_fiber=_get(pred, "fiber_g", "fiber"),
    )

    raw_text = _call_vertex_ai_judge(prompt)
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
        score = float(result.get("score", 0.0))
        score = max(0.0, min(10.0, score))
        explanation = str(result.get("explanation", ""))
        return {"score": score, "explanation": explanation}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Judge LLM returned unparseable response: {raw_text[:300]}",
        ) from exc

