"""
FastAPI retrieval server for NutriGraph.

Exposes a retrieval endpoint that takes a list of ingredient texts and returns
the closest matches from the ChromaDB ingredient index.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    # Migrate existing DBs: add columns that may not exist yet.
    with engine.connect() as conn:
        for ddl in (
            "ALTER TABLE dish_records ADD COLUMN serving_size TEXT",
            "ALTER TABLE dish_records ADD COLUMN confidence REAL",
        ):
            try:
                conn.execute(__import__("sqlalchemy").text(ddl))
                conn.commit()
            except Exception:
                pass  # column already exists


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
    serving_size: Optional[str] = None
    confidence: Optional[float] = None


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
    record = DishRecord(
        dish_name=payload.dish_name.strip(),
        restaurant_place_id=payload.place_id.strip(),
        source="restaurant",
        calories=payload.calories,
        protein=payload.protein,
        carbs=payload.carbs,
        fat=payload.fat,
        ingredients_json=json.dumps(payload.ingredients),
        serving_size=payload.serving_size,
        confidence=payload.confidence,
    )
    db.add(record)
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
            "serving_size": r.serving_size,
            "ingredient_count": len(json.loads(r.ingredients_json or "[]")),
            "calories": r.calories,
            "protein_g": r.protein,
            "carbs_g": r.carbs,
            "fat_g": r.fat,
            "confidence": r.confidence,
        }
        for r in records
    ]
    return {"dishes": dishes}

