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

# Session-state keys that are logically scoped to a single restaurant.
# They must be cleared whenever the active restaurant changes so that data
# from one restaurant cannot leak into — or be published under — another.
_RESTAURANT_SCOPED_KEYS: list[str] = [
    "last_generated_profile",
    "dish_published",
    "restaurant_ingredients",
    "catalog",
    "catalog_loaded_for",
]


def _load_catalog_from_db(client: NutriGraphClient) -> None:
    """
    Populate ``st.session_state.catalog`` from the DB for the active restaurant.

    Runs at most once per confirmed profile per session: the ``catalog_loaded_for``
    key tracks which place_id we last loaded so repeated Streamlit rerenders don't
    trigger redundant network calls.
    """
    profile = st.session_state.get("current_restaurant_profile")
    if not profile:
        return

    place_id = profile.get("place_id", "")
    if not place_id:
        return

    if st.session_state.get("catalog_loaded_for") == place_id:
        return

    st.session_state.setdefault("catalog", [])
    try:
        dishes = client.get_restaurant_dishes(place_id)
    except NutriGraphAPIError as e:
        # Surface a user-visible error and allow retries on subsequent rerenders.
        st.error(
            "We couldn't load your existing catalog right now. "
            "You can still create dishes; we'll retry loading automatically."
        )
        st.session_state["catalog_load_error"] = str(e)
        return
    else:
        st.session_state.catalog = dishes
        st.session_state.catalog_loaded_for = place_id
        # Clear any previous load error on success.
        st.session_state.pop("catalog_load_error", None)


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

    # Load the persisted catalog from the DB the first time a profile is confirmed
    # in this session (guarded so we don't re-fetch on every Streamlit rerender).
    _load_catalog_from_db(client)

    # Section 1: Create / Edit Dish
    _render_dish_builder_section(client)

    # Section 1b: Publish to Global Catalog (appears after a profile is generated)
    _render_publish_section(client)
    
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
    st.session_state.setdefault("last_generated_profile", None)
    st.session_state.setdefault("dish_published", False)

    st.subheader("🏪 Restaurant Profile Setup")

    profile = st.session_state.get("current_restaurant_profile")
    if profile:
        st.success(f"Managing catalog for: **{profile['name']}**")
        if st.button("🔄 Change Restaurant", key="change_restaurant_profile"):
            st.session_state.current_restaurant_profile = None
            st.session_state.restaurant_profile_results = []
            for key in _RESTAURANT_SCOPED_KEYS:
                st.session_state.pop(key, None)
            st.rerun()
        return

    st.caption(
        "Search for your restaurant by name to set the active profile. "
        "The dish builder and catalog will be enabled once a location is confirmed."
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
                    st.session_state.pop("restaurant_profile_selectbox", None)
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
            for key in _RESTAURANT_SCOPED_KEYS:
                st.session_state.pop(key, None)
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
            width='stretch'
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
        if st.button("➕ Add", key="add_ingredient", width='stretch'):
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
        
        # Generate nutrition profile via NutriGraph API
        try:
            estimate = client.builder_generate_profile(dish)
        except NutriGraphAPIError as exc:
            st.error(f"Failed to generate nutrition profile: {exc}")
            return
        
        # Persist to session state so the Publish button can reference it
        st.session_state.last_generated_profile = {
            "dish_name": dish_name,
            "calories": estimate.calories,
            "protein": estimate.protein_g,
            "carbs": estimate.carbs_g,
            "fat": estimate.fat_g,
        }
        st.session_state.dish_published = False
        
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


def _render_publish_section(client: NutriGraphClient) -> None:
    """
    Render the 'Publish to Global Catalog' panel.

    Appears only after a nutrition profile has been generated in the current
    session.  Lets the restaurant owner push their verified macros to the
    shared database so future diner analyses for this dish are served from
    the authoritative record rather than the LLM.
    """
    profile = st.session_state.get("last_generated_profile")
    if not profile:
        return

    st.divider()
    st.subheader("🌐 Publish to Global Catalog")
    st.caption(
        "Once you're happy with the macros above, publish them as the official "
        "ground truth for this dish. Diners who order **"
        + profile["dish_name"]
        + "** at your restaurant will receive these verified numbers instantly."
    )

    already_published = st.session_state.get("dish_published", False)
    if already_published:
        st.success(
            f"✅ **{profile['dish_name']}** has been published to the global catalog. "
            "Future diner requests will be served from this verified record."
        )
        return

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🔥 Calories", f"{profile['calories']:.0f} kcal")
        with m2:
            st.metric("💪 Protein", f"{profile['protein']:.1f} g")
        with m3:
            st.metric("🌾 Carbs", f"{profile['carbs']:.1f} g")
        with m4:
            st.metric("🥑 Fat", f"{profile['fat']:.1f} g")

    with col_btn:
        st.write("")  # vertical alignment nudge
        if st.button(
            "🌐 Publish",
            key="publish_dish_btn",
            type="primary",
            use_container_width=True,
            help="Publish these macros as the restaurant-verified ground truth.",
        ):
            place_id = st.session_state.get("current_restaurant_profile", {}).get("place_id", "")
            if not place_id:
                st.error("Restaurant profile not found. Please re-select your restaurant.")
                return
            try:
                client.publish_dish(
                    dish_name=profile["dish_name"],
                    place_id=place_id,
                    calories=profile["calories"],
                    protein=profile["protein"],
                    carbs=profile["carbs"],
                    fat=profile["fat"],
                )
                st.session_state.dish_published = True
                # Add to the local catalog immediately so it shows up without a
                # full session reload, and force catalog_loaded_for to refresh
                # from the DB next render so the entry survives future sessions.
                catalog_entry = {
                    "name": profile["dish_name"],
                    "serving_size": None,
                    "ingredient_count": None,
                    "calories": profile["calories"],
                    "protein_g": profile["protein"],
                    "carbs_g": profile["carbs"],
                    "fat_g": profile["fat"],
                    "confidence": None,
                }
                st.session_state.setdefault("catalog", [])
                existing_names = {d.get("name") for d in st.session_state.catalog}
                if profile["dish_name"] not in existing_names:
                    st.session_state.catalog.append(catalog_entry)
                st.session_state.catalog_loaded_for = None
                st.rerun()
            except Exception as exc:
                st.error(f"⚠️ Publish failed: {exc}")


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
