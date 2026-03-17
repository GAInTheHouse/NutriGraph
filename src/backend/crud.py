"""
CRUD helpers for the NutriGraph persistent dish-record store.

All functions accept an open SQLAlchemy Session.  Transaction management
is the caller's responsibility — none of the helpers here call
``db.commit()`` or ``db.refresh()``.  FastAPI endpoints must call
``db.commit()`` (and optionally ``db.refresh(record)``) after any
write operation, which keeps CRUD helpers composable and allows callers
to batch multiple operations into a single atomic transaction.
"""
import json
from typing import Optional

from sqlalchemy import func
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
        func.lower(DishRecord.dish_name) == dish_name.lower(),
        DishRecord.restaurant_place_id == place_id,
    )

    # Fetch all matching records; ordering is applied in Python below.
    records = query.all()

    # Sort in Python: restaurant rows first, then by created_at descending.
    records.sort(key=lambda r: (0 if r.source == "restaurant" else 1, -r.created_at.timestamp()))
    return records


def get_dishes_for_place(
    db: Session,
    place_id: str,
    source: str = "restaurant",
) -> list[DishRecord]:
    """
    Return all saved records for a given restaurant place-ID and source.

    Args:
        db:       Active SQLAlchemy session.
        place_id: Google Places place_id for the restaurant.
        source:   ``"restaurant"`` or ``"diner"``; defaults to ``"restaurant"``.

    Returns:
        List of matching :class:`DishRecord` instances ordered by most recent first.
    """
    return (
        db.query(DishRecord)
        .filter(
            DishRecord.restaurant_place_id == place_id,
            DishRecord.source == source,
        )
        .order_by(DishRecord.created_at.desc())
        .all()
    )


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
        The newly created (but not yet committed) :class:`DishRecord`.
        The caller must call ``db.commit()`` to persist the row.
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
    return record
