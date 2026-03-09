"""
CRUD helpers for the NutriGraph persistent dish-record store.

All functions accept an open SQLAlchemy Session and leave transaction
management to the caller (FastAPI endpoints use Depends(get_db) which
commits/rolls back at request boundary).
"""
import json
from typing import Optional

from sqlalchemy.orm import Session

from .db_models import DishRecord


def get_dish_record(
    db: Session,
    dish_name: str,
    place_id: Optional[str],
) -> list[DishRecord]:
    """
    Return all saved records that match the given dish name and optional
    restaurant place-ID.

    Matching is case-insensitive on dish_name.  Results are ordered so
    that ``source="restaurant"`` rows appear before ``source="diner"``
    rows, giving callers an easy way to detect a verified fast-path entry
    by checking ``records[0].source == "restaurant"``.

    Args:
        db:        Active SQLAlchemy session.
        dish_name: The dish name to look up (case-insensitive).
        place_id:  Google Places place_id for the restaurant, or ``None``
                   for home-cooked / untagged queries.  When ``None``,
                   only rows with a NULL ``restaurant_place_id`` are
                   returned so that home-cooked records don't pollute
                   restaurant lookups and vice-versa.

    Returns:
        Ordered list of matching :class:`DishRecord` instances; empty list
        when nothing is found.
    """
    query = db.query(DishRecord).filter(
        DishRecord.dish_name.ilike(dish_name),
        DishRecord.restaurant_place_id == place_id,
    )

    # Restaurant-verified rows first so callers can short-circuit on index 0.
    records = (
        query
        .order_by(
            # "restaurant" sorts before "diner" alphabetically — that happens
            # to be wrong, so use an explicit CASE-style sort via Python after
            # fetching (small result sets make this fine for a course project).
        )
        .all()
    )

    # Sort in Python: restaurant rows first, then by created_at descending.
    records.sort(key=lambda r: (0 if r.source == "restaurant" else 1, -r.created_at.timestamp()))
    return records


def save_dish_record(
    db: Session,
    dish_data,  # DishAnalysisResponse — avoid circular import by typing loosely
    place_id: Optional[str],
    source: str,
) -> DishRecord:
    """
    Persist a nutritional analysis result as a :class:`DishRecord`.

    Args:
        db:        Active SQLAlchemy session.
        dish_data: A :class:`~src.core.models.DishAnalysisResponse` instance
                   whose ``ingredients`` list will be serialised to JSON.
        place_id:  Google Places place_id (or ``None`` for home-cooked).
        source:    ``"diner"`` or ``"restaurant"``.

    Returns:
        The newly created and committed :class:`DishRecord`.
    """
    ingredients_payload = [
        ing.model_dump() for ing in (dish_data.ingredients or [])
    ]

    record = DishRecord(
        dish_name=dish_data.dish_name,
        restaurant_place_id=place_id,
        source=source,
        calories=dish_data.total_calories,
        protein=dish_data.total_protein,
        carbs=dish_data.total_carbs,
        fat=dish_data.total_fat,
        ingredients_json=json.dumps(ingredients_payload),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
