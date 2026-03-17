"""
SQLAlchemy ORM models for NutriGraph persistent storage.
"""
import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DishRecord(Base):
    """
    Persisted nutritional record for a dish.

    Rows are written either by a diner (source="diner") after they review
    and explicitly save their analysis, or by a restaurant owner
    (source="restaurant") when they publish the official macros for a menu
    item.  Restaurant rows act as a verified "fast-path" cache: if one
    exists for a given (dish_name, restaurant_place_id) pair the LLM is
    skipped entirely.  Diner rows are surfaced as historical context to
    coach the model toward consistency.
    """

    __tablename__ = "dish_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # The canonical dish name returned by Gemini, stored as-is for display.
    # Case-insensitive lookups are performed at query time via func.lower().
    dish_name = Column(String, index=True, nullable=False)

    # Google Places place_id for the restaurant; NULL for home-cooked meals.
    restaurant_place_id = Column(String, index=True, nullable=True)

    # "diner" — saved by a consumer after reviewing AI results.
    # "restaurant" — published by the restaurant owner as ground truth.
    source = Column(String, nullable=False)

    # Dish-level macro totals (kcal / grams).
    calories = Column(Float, nullable=False, default=0.0)
    protein = Column(Float, nullable=False, default=0.0)
    carbs = Column(Float, nullable=False, default=0.0)
    fat = Column(Float, nullable=False, default=0.0)

    # JSON-serialised list of AnalyzedIngredient dicts (per-ingredient breakdown).
    ingredients_json = Column(Text, nullable=False, default="[]")

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<DishRecord id={self.id} dish='{self.dish_name}' "
            f"source='{self.source}' place_id='{self.restaurant_place_id}'>"
        )
