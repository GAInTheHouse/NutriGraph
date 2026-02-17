"""
Diner tab UI for NutriGraph.

This module handles the consumer-facing interface for searching dishes
and estimating their nutrition.
"""
import streamlit as st
from datetime import date

from ..core.models import Dish, NutritionEstimate, generate_mock_ingredients
from ..core.api_client import NutriGraphClient
from .components import (
    render_macro_card,
    render_confidence_indicator,
    render_ingredients_table
)


def render_diner(client: NutriGraphClient) -> None:
    """
    Render the Diner tab interface.
    
    Args:
        client: NutriGraphClient instance for API calls.
    """
    st.header("🍽️ Diner View")
    st.caption("Search for dishes and estimate their nutrition")
    
    # Section 1: Dish Search / Log
    _render_dish_search_section(client)
    
    st.divider()
    
    # Section 2: Dish Detail View
    _render_dish_detail_section()
    
    st.divider()
    
    # Section 3: Personalized Tracking
    _render_tracking_section()
    
    st.divider()
    
    # Feedback Section
    _render_feedback_section()


def _render_dish_search_section(client: NutriGraphClient) -> None:
    """Render the dish search and estimation form."""
    st.subheader("🔍 Dish Search / Log")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dish_name = st.text_input(
            "Dish name",
            placeholder="e.g., Chicken Alfredo Pasta",
            key="diner_dish_name"
        )
    
    with col2:
        restaurant_name = st.text_input(
            "Restaurant (optional)",
            placeholder="e.g., Olive Garden",
            key="diner_restaurant"
        )
    
    if st.button("🔮 Estimate Nutrition", type="primary", use_container_width=True):
        if not dish_name:
            st.warning("Please enter a dish name.")
            return
        
        with st.spinner("Estimating nutrition..."):
            # Create dish object
            dish = Dish(
                name=dish_name,
                restaurant=restaurant_name if restaurant_name else None
            )
            
            # Get estimate from client (currently mocked)
            estimate = client.estimate_nutrition(dish)
            
            # Generate mock ingredients for display
            mock_ingredients = generate_mock_ingredients(dish_name)
            
            # Store in session state
            st.session_state.last_estimate = {
                "dish": dish.model_dump(),
                "estimate": estimate.model_dump(),
                "ingredients": [ing.model_dump() for ing in mock_ingredients]
            }
        
        st.success(f"Nutrition estimated for '{dish_name}'!")


def _render_dish_detail_section() -> None:
    """Render the dish detail view with macro cards and ingredients."""
    st.subheader("📊 Dish Detail View")
    
    if st.session_state.last_estimate is None:
        st.info("Search for a dish above to see nutrition details.")
        return
    
    data = st.session_state.last_estimate
    dish_data = data["dish"]
    estimate_data = data["estimate"]
    ingredients_data = data["ingredients"]
    
    # Dish info
    dish_info = f"**{dish_data['name']}**"
    if dish_data.get("restaurant"):
        dish_info += f" from {dish_data['restaurant']}"
    st.markdown(dish_info)
    
    # Macro cards
    estimate = NutritionEstimate(**estimate_data)
    render_macro_card(estimate)
    
    # Confidence indicator
    st.markdown("#### Estimation Confidence")
    render_confidence_indicator(estimate.confidence)
    
    # Ingredients table
    st.markdown("#### Estimated Ingredients")
    st.caption("⚠️ Ingredients are estimated and may not be accurate")
    
    from ..core.models import Ingredient
    ingredients = [Ingredient(**ing) for ing in ingredients_data]
    render_ingredients_table(ingredients)


def _render_tracking_section() -> None:
    """Render the personalized tracking section."""
    st.subheader("📈 Personalized Tracking")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_date = st.date_input(
            "Select date",
            value=date.today(),
            key="tracking_date"
        )
    
    with col2:
        st.markdown("#### Daily Totals (Mock)")
        
        # Mock daily totals
        daily_cols = st.columns(4)
        with daily_cols[0]:
            st.metric("Calories", "1,847")
        with daily_cols[1]:
            st.metric("Protein", "89g")
        with daily_cols[2]:
            st.metric("Carbs", "204g")
        with daily_cols[3]:
            st.metric("Fat", "72g")
    
    st.info("📌 **Feature coming soon:** Full meal logging, daily/weekly trends, "
            "and personalized nutrition goals.")


def _render_feedback_section() -> None:
    """Render the feedback expander for flagging incorrect estimates."""
    with st.expander("📝 Feedback - Flag Incorrect Estimate"):
        st.caption("Help us improve by reporting inaccurate nutrition estimates")
        
        feedback_text = st.text_area(
            "What was incorrect?",
            placeholder="e.g., The calorie estimate seems too high. "
                       "The actual dish is smaller than estimated...",
            key="diner_feedback"
        )
        
        if st.button("Submit Feedback", key="submit_feedback"):
            if feedback_text:
                # In production, this would send to backend
                st.success("Thank you for your feedback! "
                          "Our team will review this estimate.")
            else:
                st.warning("Please enter your feedback before submitting.")
