"""
Restaurant tab UI for NutriGraph.

This module handles the restaurant-facing interface for creating dishes
and generating nutrition profiles.
"""
import streamlit as st

from ..core.models import Dish, Ingredient, NutritionEstimate
from ..core.api_client import NutriGraphClient, NutriGraphAPIError
from ..core.config import settings
from .components import (
    render_macro_card,
    render_confidence_indicator,
    render_ingredients_table,
    render_dish_catalog_table,
    export_catalog_to_csv
)


def render_restaurant(client: NutriGraphClient) -> None:
    """
    Render the Restaurant tab interface.
    
    Args:
        client: NutriGraphClient instance for API calls.
    """
    st.header("🍳 Restaurant View")
    st.caption("Create dishes and generate nutrition profiles for your menu")

    # Section 0: Restaurant Profile Setup (must complete before building catalog)
    _render_restaurant_profile_section(client)

    st.divider()

    if not st.session_state.get("current_restaurant_profile"):
        st.info(
            "Search for your restaurant above to get started. "
            "Once you've selected your business, the dish builder and catalog will appear here."
        )
        return

    # Section 1: Create / Edit Dish
    _render_dish_builder_section(client)
    
    st.divider()
    
    # Section 2: Nutrition Catalog
    _render_catalog_section()
    
    st.divider()
    
    # Section 3: Export
    _render_export_section()


def _render_restaurant_profile_section(client: NutriGraphClient) -> None:
    """
    Two-step Google Places search that lets a restaurant owner find and claim
    their business before building the nutrition catalog.

    Stores ``{"place_id": ..., "name": ...}`` in
    ``st.session_state.current_restaurant_profile`` once a location is confirmed.
    """
    st.session_state.setdefault("current_restaurant_profile", None)
    st.session_state.setdefault("restaurant_profile_results", [])

    st.subheader("🏪 Restaurant Profile Setup")

    profile = st.session_state.get("current_restaurant_profile")
    if profile:
        st.success(f"Managing catalog for: **{profile['name']}**")
        if st.button("🔄 Change Restaurant", key="change_restaurant_profile"):
            st.session_state.current_restaurant_profile = None
            st.session_state.restaurant_profile_results = []
            st.rerun()
        return

    st.caption(
        "Search for your restaurant by name to claim your profile. "
        "All dishes you create will be linked to this location."
    )

    search_col, btn_col = st.columns([3, 1])
    with search_col:
        profile_query = st.text_input(
            "Restaurant name",
            placeholder="e.g., The Spotted Pig New York",
            key="restaurant_profile_query",
        )
    with btn_col:
        st.write("")  # vertical alignment nudge
        search_clicked = st.button(
            "Search",
            key="restaurant_profile_search",
            use_container_width=True,
        )

    if search_clicked:
        if not profile_query.strip():
            st.warning("Enter a restaurant name to search.")
        else:
            with st.spinner("Searching for your restaurant…"):
                try:
                    results = client.search_restaurants(profile_query)
                    st.session_state.restaurant_profile_results = results
                    if not results:
                        st.info(
                            "No restaurants found matching that name. "
                            "Try including the city or neighbourhood."
                        )
                except NutriGraphAPIError as exc:
                    st.error(f"Restaurant search failed: {exc}")
                    st.session_state.restaurant_profile_results = []

    options = st.session_state.restaurant_profile_results
    if options:
        labels = [f"{p['name']} — {p['address']}" for p in options]
        selected_idx = st.selectbox(
            "Select your business",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="restaurant_profile_selectbox",
        )
        chosen = options[selected_idx]

        if st.button("Confirm Selection", key="restaurant_profile_confirm", type="primary"):
            st.session_state.current_restaurant_profile = {
                "place_id": chosen["place_id"],
                "name": chosen["name"],
            }
            st.rerun()


def _render_dish_builder_section(client: NutriGraphClient) -> None:
    """Render the dish creation form with ingredient editor."""
    st.subheader("🆕 Create / Edit Dish")
    
    # Dish basic info
    col1, col2 = st.columns(2)
    
    with col1:
        dish_name = st.text_input(
            "Dish name",
            placeholder="e.g., Grilled Salmon Bowl",
            key="restaurant_dish_name"
        )
    
    with col2:
        serving_size = st.text_input(
            "Serving size",
            value=settings.DEFAULT_SERVING_SIZE,
            placeholder="e.g., 1 bowl, 350g",
            key="restaurant_serving_size"
        )
    
    # Ingredient list editor
    st.markdown("#### Ingredients")
    st.caption("Add ingredients to calculate nutrition profile")
    
    # Initialize ingredients list in session state
    if "restaurant_ingredients" not in st.session_state:
        st.session_state.restaurant_ingredients = []
    
    # Display current ingredients
    if st.session_state.restaurant_ingredients:
        st.markdown("**Current ingredients:**")
        render_ingredients_table(st.session_state.restaurant_ingredients)
        
        # Option to clear all
        if st.button("🗑️ Clear All Ingredients", key="clear_ingredients"):
            st.session_state.restaurant_ingredients = []
            st.rerun()
    
    # Add ingredient form
    st.markdown("**Add ingredient:**")
    _render_add_ingredient_form()
    
    st.markdown("---")
    
    # Generate profile button
    col_gen1, col_gen2 = st.columns([2, 1])
    
    with col_gen1:
        generate_clicked = st.button(
            "🧪 Generate Nutrition Profile",
            type="primary",
            use_container_width=True
        )
    
    with col_gen2:
        add_to_catalog = st.checkbox(
            "Add to catalog",
            value=True,
            key="add_to_catalog"
        )
    
    if generate_clicked:
        _handle_generate_profile(client, dish_name, serving_size, add_to_catalog)


def _render_add_ingredient_form() -> None:
    """Render the form for adding a new ingredient."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        new_ing_name = st.text_input(
            "Ingredient name",
            key="new_ing_name",
            placeholder="e.g., Chicken Breast",
            label_visibility="collapsed"
        )
    
    with col2:
        new_ing_qty = st.number_input(
            "Quantity",
            min_value=0.0,
            value=100.0,
            step=10.0,
            key="new_ing_qty",
            label_visibility="collapsed"
        )
    
    with col3:
        new_ing_unit = st.selectbox(
            "Unit",
            options=settings.DEFAULT_UNITS,
            key="new_ing_unit",
            label_visibility="collapsed"
        )
    
    with col4:
        if st.button("➕ Add", key="add_ingredient", use_container_width=True):
            if new_ing_name:
                new_ingredient = Ingredient(
                    name=new_ing_name,
                    quantity=new_ing_qty,
                    unit=new_ing_unit
                )
                st.session_state.restaurant_ingredients.append(new_ingredient)
                st.rerun()
            else:
                st.warning("Enter ingredient name")


def _handle_generate_profile(
    client: NutriGraphClient,
    dish_name: str,
    serving_size: str,
    add_to_catalog: bool
) -> None:
    """Handle the generate profile button click."""
    if not dish_name:
        st.warning("Please enter a dish name.")
        return
    
    if not st.session_state.restaurant_ingredients:
        st.warning("Please add at least one ingredient.")
        return
    
    with st.spinner("Generating nutrition profile..."):
        # Create dish object
        dish = Dish(
            name=dish_name,
            serving_size=serving_size,
            ingredients=st.session_state.restaurant_ingredients
        )
        
        # Generate profile (currently mocked)
        estimate = client.builder_generate_profile(dish)
        
        # Display results
        st.success(f"Nutrition profile generated for '{dish_name}'!")
        
        st.markdown("#### Generated Nutrition Profile")
        render_macro_card(estimate)
        render_confidence_indicator(estimate.confidence)
        
        # Add to catalog if requested
        if add_to_catalog:
            catalog_entry = {
                "name": dish_name,
                "serving_size": serving_size,
                "ingredient_count": len(st.session_state.restaurant_ingredients),
                "calories": estimate.calories,
                "protein_g": estimate.protein_g,
                "carbs_g": estimate.carbs_g,
                "fat_g": estimate.fat_g,
                "confidence": estimate.confidence
            }
            st.session_state.catalog.append(catalog_entry)
            st.info(f"✅ '{dish_name}' added to catalog")
            
            # Clear ingredients for next dish
            st.session_state.restaurant_ingredients = []


def _render_catalog_section() -> None:
    """Render the nutrition catalog section."""
    st.subheader("📚 Nutrition Catalog")
    st.caption(f"Dishes created in this session: {len(st.session_state.catalog)}")
    
    render_dish_catalog_table(st.session_state.catalog)


def _render_export_section() -> None:
    """Render the export section."""
    st.subheader("📤 Export")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        csv_data = export_catalog_to_csv(st.session_state.catalog)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="nutrigraph_catalog.csv",
            mime="text/csv",
            disabled=len(st.session_state.catalog) == 0
        )
    
    with col2:
        if st.session_state.catalog:
            st.success(f"Ready to export {len(st.session_state.catalog)} dish(es)")
        else:
            st.info("Create dishes above to enable export")
