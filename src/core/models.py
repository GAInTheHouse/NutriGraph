"""
Pydantic models for NutriGraph data structures.
"""
from pydantic import BaseModel, Field
from typing import Optional
import hashlib
import random


class Ingredient(BaseModel):
    """Represents a single ingredient in a dish."""
    name: str = Field(..., description="Name of the ingredient")
    quantity: float = Field(..., ge=0, description="Amount of the ingredient")
    unit: str = Field(..., description="Unit of measurement (g, oz, cup, etc.)")


class Dish(BaseModel):
    """Represents a dish with its ingredients."""
    name: str = Field(..., description="Name of the dish")
    restaurant: Optional[str] = Field(None, description="Restaurant name if applicable")
    serving_size: str = Field("1 serving", description="Serving size description")
    ingredients: list[Ingredient] = Field(default_factory=list, description="List of ingredients")
    
    def get_seed(self) -> int:
        """Generate a stable seed based on dish name for reproducible mock data."""
        hash_str = hashlib.md5(self.name.lower().encode()).hexdigest()
        return int(hash_str[:8], 16)


class NutritionEstimate(BaseModel):
    """Nutrition estimation results."""
    calories: float = Field(..., ge=0, description="Total calories (kcal)")
    protein_g: float = Field(..., ge=0, description="Protein in grams")
    carbs_g: float = Field(..., ge=0, description="Carbohydrates in grams")
    fat_g: float = Field(..., ge=0, description="Fat in grams")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    
    @classmethod
    def mock_from_dish(cls, dish: Dish) -> "NutritionEstimate":
        """
        Generate mock nutrition estimate based on dish name.
        Uses dish name as seed for reproducible results.
        """
        seed = dish.get_seed()
        rng = random.Random(seed)
        
        # Generate plausible nutrition values
        calories = rng.uniform(200, 800)
        protein_g = rng.uniform(10, 40)
        carbs_g = rng.uniform(20, 80)
        fat_g = rng.uniform(5, 35)
        confidence = rng.uniform(0.65, 0.95)
        
        return cls(
            calories=round(calories, 1),
            protein_g=round(protein_g, 1),
            carbs_g=round(carbs_g, 1),
            fat_g=round(fat_g, 1),
            confidence=round(confidence, 2)
        )


class FeedbackSubmission(BaseModel):
    """User feedback for incorrect estimates."""
    dish_name: str
    feedback_text: str
    submitted_at: Optional[str] = None


class AnalyzedIngredient(BaseModel):
    """A single ingredient identified and analyzed from a dish photo."""
    name: str = Field(..., description="Name of the identified ingredient")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence (0–1)")
    calories: float = Field(..., ge=0, description="Calories contributed by this ingredient (kcal)")
    protein: float = Field(..., ge=0, description="Protein in grams")
    carbs: float = Field(..., ge=0, description="Carbohydrates in grams")
    fat: float = Field(..., ge=0, description="Fat in grams")


class DishAnalysisResponse(BaseModel):
    """Full nutritional analysis returned by the image-to-ingredient pipeline."""
    dish_name: str = Field(..., description="Name inferred from the dish photo")
    total_calories: float = Field(..., ge=0, description="Sum of calories across all ingredients")
    total_protein: float = Field(..., ge=0, description="Sum of protein (g) across all ingredients")
    total_carbs: float = Field(..., ge=0, description="Sum of carbohydrates (g) across all ingredients")
    total_fat: float = Field(..., ge=0, description="Sum of fat (g) across all ingredients")
    ingredients: list[AnalyzedIngredient] = Field(
        default_factory=list,
        description="Per-ingredient breakdown with individual macros and confidence",
    )


# ── Hybrid retrieval models (used by /api/v1/retrieve-ingredient) ─────────────

class IngredientQuery(BaseModel):
    """
    Request payload for the hybrid ingredient retrieval endpoint.

    When ``brand`` is supplied (extracted by the LangGraph clarification agent
    after asking the user), the retriever applies a ChromaDB pre-filter so only
    documents whose ``brand`` metadata matches are returned alongside the
    semantic ranking.
    """

    query: str = Field(..., description="Free-text ingredient name or description.")
    brand: str | None = Field(
        None,
        description=(
            "Optional brand name provided by the LangGraph clarification agent. "
            "When present, restricts results to documents with a matching brand."
        ),
    )
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return.")


class RetrievedIngredient(BaseModel):
    """A single ingredient result returned by the hybrid retriever."""

    id: str = Field(..., description="ChromaDB document ID.")
    name: str = Field(..., description="Canonical ingredient name from the index.")
    brand: str | None = Field(None, description="Brand name, if present in the index.")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        description=(
            "Similarity score in (0, 1]: higher = closer match. "
            "Derived from ChromaDB distance; may include a +0.15 keyword boost "
            "when the query is a substring of the document's exact_name."
        ),
    )
    calories: float | None = Field(None, description="Energy in kcal per 100 g.")
    protein: float | None = Field(None, description="Protein in grams per 100 g.")
    carbs: float | None = Field(None, description="Carbohydrates in grams per 100 g.")
    fat: float | None = Field(None, description="Fat in grams per 100 g.")


class RetrievalResponse(BaseModel):
    """Response envelope returned by /api/v1/retrieve-ingredient."""

    results: list[RetrievedIngredient] = Field(
        default_factory=list,
        description="Ordered list of ingredient matches, highest similarity first.",
    )


# ── Mock data helpers ──────────────────────────────────────────────────────────

def generate_mock_ingredients(dish_name: str, count: int = 5) -> list[Ingredient]:
    """
    Generate mock ingredients for a dish based on its name.
    
    Args:
        dish_name: Name of the dish to generate ingredients for.
        count: Number of ingredients to generate.
    
    Returns:
        List of mock Ingredient objects.
    """
    # Sample ingredient pools
    proteins = ["Chicken Breast", "Beef", "Salmon", "Tofu", "Eggs", "Shrimp"]
    carbs = ["Rice", "Pasta", "Bread", "Potatoes", "Quinoa", "Noodles"]
    vegetables = ["Broccoli", "Spinach", "Bell Pepper", "Onion", "Tomato", "Carrots"]
    fats = ["Olive Oil", "Butter", "Avocado", "Cheese", "Coconut Oil"]
    seasonings = ["Salt", "Pepper", "Garlic", "Herbs", "Soy Sauce", "Lemon Juice"]
    
    all_ingredients = proteins + carbs + vegetables + fats + seasonings
    
    # Use dish name as seed for reproducibility
    seed = int(hashlib.md5(dish_name.lower().encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    
    selected = rng.sample(all_ingredients, min(count, len(all_ingredients)))
    units = ["g", "oz", "cup", "tbsp", "piece"]
    
    return [
        Ingredient(
            name=name,
            quantity=round(rng.uniform(10, 200), 1),
            unit=rng.choice(units)
        )
        for name in selected
    ]
