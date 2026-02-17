"""
Shared UI components for NutriGraph Streamlit application.
"""
import streamlit as st
import pandas as pd
from typing import Optional

from ..core.models import NutritionEstimate, Ingredient, Dish


def render_macro_card(estimate: NutritionEstimate) -> None:
    """
    Render a card layout showing macro nutrition breakdown.
    
    Args:
        estimate: NutritionEstimate object with nutrition values.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔥 Calories",
            value=f"{estimate.calories:.0f}",
            help="Total energy (kcal)"
        )
    
    with col2:
        st.metric(
            label="🥩 Protein",
            value=f"{estimate.protein_g:.1f}g",
            help="Protein content in grams"
        )
    
    with col3:
        st.metric(
            label="🍞 Carbs",
            value=f"{estimate.carbs_g:.1f}g",
            help="Carbohydrate content in grams"
        )
    
    with col4:
        st.metric(
            label="🧈 Fat",
            value=f"{estimate.fat_g:.1f}g",
            help="Fat content in grams"
        )


def render_confidence_indicator(confidence: float) -> None:
    """
    Render a confidence indicator for nutrition estimates.
    
    Args:
        confidence: Confidence score between 0 and 1.
    """
    # Determine color based on confidence level
    if confidence >= 0.8:
        status = "High confidence"
        color = "green"
    elif confidence >= 0.6:
        status = "Medium confidence"
        color = "orange"
    else:
        status = "Low confidence"
        color = "red"
    
    st.progress(confidence, text=f"Confidence: {confidence:.0%} ({status})")


def render_ingredients_table(ingredients: list[Ingredient]) -> None:
    """
    Render ingredients as a formatted table.
    
    Args:
        ingredients: List of Ingredient objects to display.
    """
    if not ingredients:
        st.info("No ingredients available.")
        return
    
    df = pd.DataFrame([
        {
            "Ingredient": ing.name,
            "Quantity": ing.quantity,
            "Unit": ing.unit
        }
        for ing in ingredients
    ])
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def render_dish_catalog_table(catalog: list[dict]) -> None:
    """
    Render the dish catalog as a table.
    
    Args:
        catalog: List of dish dictionaries with nutrition data.
    """
    if not catalog:
        st.info("No dishes in catalog yet. Create your first dish above!")
        return
    
    df = pd.DataFrame(catalog)
    
    # Reorder and rename columns for display
    display_columns = {
        "name": "Dish Name",
        "serving_size": "Serving Size",
        "ingredient_count": "# Ingredients",
        "calories": "Calories",
        "protein_g": "Protein (g)",
        "carbs_g": "Carbs (g)",
        "fat_g": "Fat (g)",
        "confidence": "Confidence"
    }
    
    # Select only columns that exist
    available_cols = [col for col in display_columns.keys() if col in df.columns]
    df_display = df[available_cols].rename(columns=display_columns)
    
    # Format confidence as percentage
    if "Confidence" in df_display.columns:
        df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x:.0%}")
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )


def render_ingredient_editor(key_prefix: str = "ing") -> Optional[Ingredient]:
    """
    Render an ingredient input form.
    
    Args:
        key_prefix: Prefix for widget keys to avoid conflicts.
    
    Returns:
        Ingredient if valid input provided, None otherwise.
    """
    from ..core.config import settings
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        name = st.text_input(
            "Ingredient name",
            key=f"{key_prefix}_name",
            placeholder="e.g., Chicken Breast"
        )
    
    with col2:
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            value=100.0,
            step=10.0,
            key=f"{key_prefix}_qty"
        )
    
    with col3:
        unit = st.selectbox(
            "Unit",
            options=settings.DEFAULT_UNITS,
            key=f"{key_prefix}_unit"
        )
    
    if name and quantity > 0:
        return Ingredient(name=name, quantity=quantity, unit=unit)
    return None


def export_catalog_to_csv(catalog: list[dict]) -> bytes:
    """
    Convert catalog to CSV bytes for download.
    
    Args:
        catalog: List of dish dictionaries.
    
    Returns:
        CSV data as bytes.
    """
    if not catalog:
        return b""
    
    df = pd.DataFrame(catalog)
    return df.to_csv(index=False).encode("utf-8")


def initialize_session_state() -> None:
    """Initialize session state with default values."""
    if "catalog" not in st.session_state:
        st.session_state.catalog = []
    
    if "last_estimate" not in st.session_state:
        st.session_state.last_estimate = None
    
    if "diner_ingredients" not in st.session_state:
        st.session_state.diner_ingredients = []
    
    if "restaurant_ingredients" not in st.session_state:
        st.session_state.restaurant_ingredients = []


def reset_session_state() -> None:
    """Reset all session state to defaults."""
    st.session_state.catalog = []
    st.session_state.last_estimate = None
    st.session_state.diner_ingredients = []
    st.session_state.restaurant_ingredients = []
