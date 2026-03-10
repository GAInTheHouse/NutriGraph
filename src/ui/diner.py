"""
Diner tab UI for NutriGraph.

This module handles the consumer-facing interface for:
  - Uploading a dish photo and triggering the Gemini vision pipeline (Step 1)
  - Viewing the one-shot AI nutritional breakdown immediately after analysis
  - Agent-assisted clarification loop (Step 2): the LangGraph clarification graph
    automatically fires on the identified ingredients, asking the user targeted
    questions for any ingredient whose retrieval confidence falls below the
    DEFAULT_THRESHOLD defined in src/backend/clarification_graph.py
  - Displaying the refined final estimate once all ingredients converge (Step 3)
  - Searching dishes by name (legacy text-based flow, kept for compatibility)
  - Personalised daily tracking (placeholder)
  - Submitting accuracy feedback
"""
import streamlit as st
from datetime import date

import pandas as pd

from ..core.models import (
    AnalyzedIngredient,
    Dish,
    DishAnalysisResponse,
    NutritionEstimate,
    generate_mock_ingredients,
    Ingredient,
)
from ..core.api_client import NutriGraphClient, NutriGraphAPIError
from .components import (
    render_macro_card,
    render_confidence_indicator,
    render_ingredients_table,
)

# ── Threshold mirrored from clarification_graph.py (imported lazily to avoid
#    loading heavy ML dependencies at Streamlit startup time).
_CONFIDENCE_THRESHOLD: float = 0.7

# ── Clarification: stop asking after N rounds with no/small improvement ─────
# Improvement below this is considered negligible (e.g. 1%).
_NEGLIGIBLE_IMPROVEMENT: float = 0.01
# After this many follow-up rounds with negligible improvement for an ingredient, stop asking.
_MAX_NO_IMPROVEMENT_ROUNDS: int = 2

# ── Issue types offered in the feedback form ─────────────────────────────────
_FEEDBACK_ISSUE_TYPES = [
    "Missing Ingredient",
    "Wrong Portion Size",
    "Incorrect Macros",
    "Incorrect Dish Name",
    "Other",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_diner(client: NutriGraphClient) -> None:
    """
    Render the Diner tab.

    Args:
        client: NutriGraphClient instance used for backend calls.
    """
    st.header("🍽️ Diner View")
    st.caption("Upload a photo of your dish for an AI-powered nutritional breakdown")

    _init_session_state()

    # ── Step 1: Image upload & one-shot AI analysis ───────────────────────────
    _render_image_analysis_section(client)

    st.divider()

    # ── Step 1 result: one-shot Dish Detail View ──────────────────────────────
    _render_analysis_detail_section(client)

    st.divider()

    # ── Step 2 + 3: Agent clarification loop & refined final result ───────────
    _render_clarification_section(client)

    st.divider()

    # ── Feedback form ─────────────────────────────────────────────────────────
    _render_feedback_section()

    st.divider()

    # ── Legacy text-search (kept for backward compatibility) ──────────────────
    with st.expander("🔍 Search by Dish Name (Legacy)", expanded=False):
        _render_dish_search_section(client)
        if st.session_state.last_estimate is not None:
            st.divider()
            _render_dish_detail_section()

    st.divider()

    # ── Personalised tracking placeholder ────────────────────────────────────
    _render_tracking_section()


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Initialise all session-state keys used by the Diner tab."""
    # One-shot image analysis
    st.session_state.setdefault("current_dish_analysis", None)
    st.session_state.setdefault("last_estimate", None)

    # Restaurant tagging (image upload section)
    st.session_state.setdefault("restaurant_home_cooked", False)
    st.session_state.setdefault("restaurant_results", [])
    st.session_state.setdefault("selected_restaurant", None)

    # Persistence: tracks whether the current result has been saved to avoid duplicates
    st.session_state.setdefault("dish_saved", False)

    # Clarification agent state
    st.session_state.setdefault("clar_active", False)
    # Dish name from the image analysis (for LLM context when generating questions)
    st.session_state.setdefault("clar_dish_name", "")
    # Original ingredient names extracted from the image (used for display)
    st.session_state.setdefault("clar_original_names", [])
    # Ingredient query strings sent to ChromaDB; may be enriched with user answers
    st.session_state.setdefault("clar_query_names", [])
    # Latest raw ClarificationState dict returned by the LangGraph graph
    st.session_state.setdefault("clar_state", None)
    # Ordered list of {"question", "answer", "scores_before", "scores_after"} turns
    st.session_state.setdefault("clar_history", [])
    # True once all ingredients are ≥ threshold
    st.session_state.setdefault("clar_done", False)
    # Serialised DishAnalysisResponse built from the converged graph state
    st.session_state.setdefault("clar_refined_result", None)
    # Non-empty string if the graph raised an exception
    st.session_state.setdefault("clar_error", None)
    # Per-ingredient scores from the very first graph run (baseline for delta display)
    st.session_state.setdefault("clar_initial_scores", [])
    # Per-ingredient scores from the previous graph run (for round-over-round delta)
    st.session_state.setdefault("clar_prev_scores", [])
    # For multi-query retrieval: {idx: pre-refinement query} so we try both and keep best
    st.session_state.setdefault("clar_fallback_queries", {})
    # Per-ingredient count of rounds with no/small improvement; stop asking after _MAX_NO_IMPROVEMENT_ROUNDS
    st.session_state.setdefault("clar_no_improvement_count", {})
    # True when we stopped because no ingredients had questions left (max no-improvement rounds reached)
    st.session_state.setdefault("clar_stopped_max_rounds", False)


# ─────────────────────────────────────────────────────────────────────────────
# Shared dish-detail renderer (one-shot & agent-refined paths both use this)
# ─────────────────────────────────────────────────────────────────────────────

def render_dish_detail(result: DishAnalysisResponse, *, label: str = "Dish Detail View") -> None:
    """
    Render a :class:`DishAnalysisResponse` as a structured nutritional breakdown.

    Decoupled from session state so it can be reused for both the one-shot
    image-analysis result and the agent-refined final result.

    Args:
        result: The nutritional analysis to display.
        label:  Section heading shown above the breakdown.
    """
    st.subheader(f"📊 {label}")
    st.markdown(f"### {result.dish_name}")

    st.markdown("#### Nutritional Totals")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔥 Calories", f"{result.total_calories:.0f} kcal")
    with col2:
        st.metric("💪 Protein", f"{result.total_protein:.1f} g")
    with col3:
        st.metric("🌾 Carbs", f"{result.total_carbs:.1f} g")
    with col4:
        st.metric("🥑 Fat", f"{result.total_fat:.1f} g")

    # Overall confidence summary derived from per-ingredient scores
    if result.ingredients:
        avg_conf = sum(ing.confidence_score for ing in result.ingredients) / len(result.ingredients)
        st.markdown("#### Overall Confidence")
        conf_col, bar_col = st.columns([1, 3])
        with conf_col:
            st.metric("🎯 Average", f"{avg_conf:.1%}")
        with bar_col:
            st.progress(
                min(avg_conf, 1.0),
                text=f"{avg_conf:.1%} mean confidence across {len(result.ingredients)} ingredient(s)",
            )

    st.markdown("#### Identified Ingredients")
    if result.ingredients:
        df = pd.DataFrame(
            [
                {
                    "Ingredient": ing.name,
                    "Confidence": f"{ing.confidence_score:.0%}",
                    "Calories (kcal)": round(ing.calories, 1),
                    "Protein (g)": round(ing.protein, 1),
                    "Carbs (g)": round(ing.carbs, 1),
                    "Fat (g)": round(ing.fat, 1),
                }
                for ing in result.ingredients
            ]
        )
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.caption("No per-ingredient breakdown was returned by the model.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — image upload & analysis
# ─────────────────────────────────────────────────────────────────────────────

def _render_image_analysis_section(client: NutriGraphClient) -> None:
    """Image upload widget, restaurant tagging, and 'Analyze Dish' button (Step 1)."""
    st.subheader("📸 Step 1 — Upload Dish Photo")

    uploaded_file = st.file_uploader(
        "Select an image of your dish",
        type=["png", "jpg", "jpeg"],
        key="dish_image_upload",
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        col_img, _ = st.columns([1, 2])
        with col_img:
            st.image(uploaded_file, caption=uploaded_file.name, width='stretch')

        # ── Restaurant tagging ─────────────────────────────────────────────────
        st.markdown("#### Where is this meal from?")
        _render_restaurant_tagging(client)

        # ── Optional dish-name hint ────────────────────────────────────────────
        st.markdown("#### Dish name hint (optional)")
        st.caption(
            "If you already know the dish name, enter it here to check our database "
            "first — you may get an instant restaurant-verified result or AI coaching "
            "from past analyses."
        )
        dish_name_hint = st.text_input(
            "Dish name",
            placeholder="e.g., Grilled Salmon Bowl",
            key="diner_dish_name_hint",
            label_visibility="collapsed",
        )

        st.write("")  # spacing before the action button

        if st.button("🔍 Analyze Dish", type="primary", width='stretch'):
            # Resolve the restaurant selection to a plain string and place_id.
            _selected = st.session_state.get("selected_restaurant")
            if isinstance(_selected, dict):
                _restaurant_context: str | None = _selected.get("name")
                _place_id: str | None = _selected.get("place_id")
            elif isinstance(_selected, str):
                _restaurant_context = _selected  # "Home Cooked"
                _place_id = None
            else:
                _restaurant_context = None
                _place_id = None

            _hint = dish_name_hint.strip() if dish_name_hint else None

            with st.spinner("Analyzing image and retrieving nutritional data…"):
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    response: DishAnalysisResponse = client.analyze_dish_image(
                        image_bytes,
                        uploaded_file.name,
                        _restaurant_context,
                        dish_name=_hint,
                        place_id=_place_id,
                    )
                    st.session_state.current_dish_analysis = response.model_dump()
                    # Store place_id so save buttons can forward it
                    st.session_state.current_place_id = _place_id
                    # Reset the saved flag whenever a new analysis comes in
                    st.session_state.dish_saved = False

                except NutriGraphAPIError as exc:
                    st.error(f"⚠️ Analysis failed: {exc}")
                    return

                except Exception as exc:
                    st.error(f"⚠️ An unexpected error occurred: {exc}")
                    return

            st.success(f"Analysis complete for **{response.dish_name}**!")

            # Automatically start the agent clarification loop on the
            # identified ingredients (Step 2).
            _init_clarification(response)

    else:
        st.info("Upload a dish photo above and click **Analyze Dish** to get started.")


def _render_restaurant_tagging(client: NutriGraphClient) -> None:
    """
    Home Cooked Meal checkbox + two-step Google Places restaurant search.

    Writes the resolved selection into ``st.session_state.selected_restaurant``:
    - ``"Home Cooked"`` when the checkbox is ticked.
    - A ``dict`` with ``place_id``, ``name``, ``address`` when a Places result
      is picked from the selectbox.
    - ``None`` when the user hasn't completed the search yet.
    """
    home_cooked = st.checkbox(
        "Home Cooked Meal",
        value=st.session_state.restaurant_home_cooked,
        key="restaurant_home_cooked",
    )

    if home_cooked:
        st.session_state.selected_restaurant = "Home Cooked"
        return

    # Checkbox was just unchecked (or was never checked): make sure any
    # previously stored "Home Cooked" value doesn't leak into a new analysis.
    if st.session_state.get("selected_restaurant") == "Home Cooked":
        st.session_state.selected_restaurant = None
        st.session_state.restaurant_results = []
        st.session_state.pop("diner_restaurant_selectbox", None)

    # Two-step Places search when not home cooked
    rest_col1, rest_col2 = st.columns([3, 1])
    with rest_col1:
        rest_query = st.text_input(
            "Search Restaurant Name",
            placeholder="e.g., Shake Shack Manhattan",
            key="diner_restaurant_query",
        )
    with rest_col2:
        st.write("")  # vertical alignment nudge
        find_clicked = st.button(
            "Find Restaurant",
            key="diner_find_restaurant",
            use_container_width=True,
        )

    if find_clicked:
        if not rest_query.strip():
            st.warning("Enter a restaurant name to search.")
        else:
            with st.spinner("Searching for restaurants…"):
                try:
                    results = client.search_restaurants(rest_query)
                    st.session_state.restaurant_results = results
                    st.session_state.selected_restaurant = None
                    st.session_state.pop("diner_restaurant_selectbox", None)
                    if not results:
                        st.info("No restaurants found. Try a different search term.")
                except NutriGraphAPIError as exc:
                    st.error(f"Restaurant search failed: {exc}")
                    st.session_state.restaurant_results = []

    if st.session_state.restaurant_results:
        options = st.session_state.restaurant_results
        labels = [f"{p['name']} — {p['address']}" for p in options]
        selected_idx = st.selectbox(
            "Select your restaurant",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="diner_restaurant_selectbox",
        )
        chosen = options[selected_idx]
        st.session_state.selected_restaurant = chosen


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 result — one-shot dish detail view
# ─────────────────────────────────────────────────────────────────────────────

def _render_analysis_detail_section(client: NutriGraphClient) -> None:
    """One-shot Dish Detail View rendered right after the image is analysed."""
    if not st.session_state.get("current_dish_analysis"):
        st.subheader("📊 Dish Detail View")
        st.info("Nutritional details will appear here after you analyze a dish photo.")
        return

    analysis = DishAnalysisResponse(**st.session_state.current_dish_analysis)

    # ── Cache / data-source badge ──────────────────────────────────────────────
    data_source = analysis.data_source
    if data_source == "restaurant_verified":
        st.info(
            "✅ **Verified by Restaurant** — these macros were published by the restaurant "
            "owner and are served directly from our database. No AI estimation was performed."
        )
    elif data_source == "diner_cached":
        st.info(
            "🔄 **Enhanced by Past Diner Data** — previous analyses of this dish were used "
            "to coach the AI for greater consistency. The image is still the primary source."
        )

    render_dish_detail(analysis, label="Dish Detail View (one-shot estimate)")

    # ── Save Results button (enabled as soon as macros are populated) ──────────
    col_save, col_clear = st.columns([2, 1])
    with col_save:
        already_saved = st.session_state.get("dish_saved", False)
        if already_saved:
            st.success("✅ Saved to your history!")
        elif not analysis.is_cached:
            # Only offer to save AI-generated results (cached restaurant results
            # are already in the DB as the authoritative source).
            if st.button(
                "💾 Save Results",
                key="save_oneshot_results",
                help="Save this nutritional analysis to your history. "
                     "You can improve accuracy first using the clarification agent below.",
                use_container_width=True,
            ):
                _save_analysis(client, analysis)

    with col_clear:
        if st.button("🗑️ Clear Analysis", key="clear_analysis", use_container_width=True):
            st.session_state.current_dish_analysis = None
            st.session_state.dish_saved = False
            st.session_state.pop("current_place_id", None)
            _reset_clarification()
            st.rerun()


def _save_analysis(client: NutriGraphClient, analysis: DishAnalysisResponse) -> None:
    """Helper: call the save endpoint and update session state."""
    place_id = st.session_state.get("current_place_id")
    try:
        client.save_dish_result(analysis, place_id=place_id)
        st.session_state.dish_saved = True
        st.rerun()
    except NutriGraphAPIError as exc:
        st.error(f"⚠️ Could not save: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Clarification agent helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_clarification(analysis: DishAnalysisResponse) -> None:
    """
    Reset clarification session state and run the first graph pass.

    Called automatically after a successful image analysis.  Populates
    ``clar_original_names`` and ``clar_query_names`` from the identified
    ingredients, then invokes the LangGraph clarification graph.

    Args:
        analysis: The :class:`DishAnalysisResponse` returned by the vision pipeline.
    """
    names = [ing.name for ing in analysis.ingredients]

    # Short-circuit: if the vision pipeline returned no ingredients there is
    # nothing for the clarification graph to work with.  Setting clar_active
    # here would show a misleading "all ingredients meet threshold" banner.
    if not names:
        st.session_state.clar_active = False
        return

    # Reset ALL clarification state for the new dish — including the
    # baseline/delta score lists so that metrics from a previous dish do
    # not bleed into the new session when a second image is analyzed without
    # clicking "Clear Analysis".
    st.session_state.clar_active = True
    st.session_state.clar_dish_name = analysis.dish_name
    st.session_state.clar_original_names = names
    st.session_state.clar_query_names = list(names)   # mutable copy
    st.session_state.clar_history = []
    st.session_state.clar_done = False
    st.session_state.clar_refined_result = None
    st.session_state.clar_error = None
    st.session_state.clar_state = None
    st.session_state.clar_initial_scores = []   # reset baseline for this dish
    st.session_state.clar_prev_scores = []      # reset round-over-round delta
    st.session_state.clar_fallback_queries = {}  # multi-query retrieval
    st.session_state.clar_no_improvement_count = {}
    st.session_state.clar_stopped_max_rounds = False
    _run_clarification_graph()


def _run_clarification_graph() -> None:
    """
    Invoke the LangGraph clarification graph on the current query ingredient
    strings and persist the result in session state.

    Uses the current ``clar_query_names`` list (which may have been enriched
    by previous user answers).  If the graph finds no low-confidence
    ingredients it marks the conversation as done and builds the refined
    :class:`DishAnalysisResponse`.

    Errors from ChromaDB (e.g. index not yet built) are caught and stored in
    ``clar_error`` so the UI can surface a helpful message without crashing.
    """
    names = st.session_state.get("clar_query_names", [])
    if not names:
        st.session_state.clar_done = True
        return

    try:
        # Lazy import keeps heavy ML dependencies out of the module-level import
        # so that Streamlit startup stays fast even when the backend is offline.
        from ..backend.clarification_graph import build_clarification_graph, DEFAULT_THRESHOLD

        # Sync the module-level constant so the UI displays the correct value.
        global _CONFIDENCE_THRESHOLD
        _CONFIDENCE_THRESHOLD = DEFAULT_THRESHOLD

        # Snapshot current scores as "previous" before overwriting clar_state.
        existing = st.session_state.get("clar_state") or {}
        st.session_state.clar_prev_scores = list(existing.get("scores", []))

        graph = build_clarification_graph()
        dish_name = st.session_state.get("clar_dish_name", "")
        fallback_queries = st.session_state.get("clar_fallback_queries", {})
        result: dict = graph.invoke({
            "ingredients": names,
            "dish_name": dish_name,
            "fallback_queries": fallback_queries,
        })
        st.session_state.clar_state = result

        # Persist the very first set of scores as the baseline for total-improvement display.
        if not st.session_state.get("clar_initial_scores"):
            st.session_state.clar_initial_scores = list(result.get("scores", []))

        # Treat as converged only when the graph EXPLICITLY returned an empty
        # low_conf_indices list — a missing key means something went wrong.
        low_conf = result.get("low_conf_indices")
        if low_conf is not None and len(low_conf) == 0:
            st.session_state.clar_done = True
            refined = _build_refined_result(
                st.session_state.clar_original_names, result
            )
            st.session_state.clar_refined_result = refined.model_dump()
        elif low_conf is None:
            # Graph ran but decision node never set low_conf_indices — unexpected state.
            st.session_state.clar_error = (
                "The clarification graph returned an incomplete state "
                "(low_conf_indices missing). This usually means ChromaDB returned "
                "no results. Check that the ingredient index is built and non-empty."
            )

    except Exception as exc:
        # Keep clar_active = True so the section still renders and shows the error.
        st.session_state.clar_error = (
            f"{type(exc).__name__}: {exc}. "
            "Make sure the ChromaDB index has been built "
            "(run scripts/dataset/index_ingredients.py) and that all "
            "backend dependencies are installed."
        )


def _build_refined_result(
    original_names: list[str],
    clar_state: dict,
) -> DishAnalysisResponse:
    """
    Construct a :class:`DishAnalysisResponse` from a converged
    :class:`ClarificationState`.

    For each ingredient the best ChromaDB match's macro values are used.
    The combined retrieval score becomes the ``confidence_score`` on each
    :class:`AnalyzedIngredient`.

    Args:
        original_names: Display names from the initial image analysis.
        clar_state:     The raw dict returned by the LangGraph graph after
                        convergence (i.e. ``low_conf_indices`` is empty).

    Returns:
        A :class:`DishAnalysisResponse` ready to be rendered.
    """
    matches: list[list[dict]] = clar_state.get("matches", [])
    scores: list[float] = clar_state.get("scores", [])

    analyzed: list[AnalyzedIngredient] = []
    for i, name in enumerate(original_names):
        if i < len(matches) and matches[i]:
            best = matches[i][0]
            calories = float(best.get("energy_kcal") or 0.0)
            protein = float(best.get("protein_g") or 0.0)
            carbs = float(best.get("carbohydrates_g") or 0.0)
            fat = float(best.get("fat_g") or 0.0)
            confidence = float(scores[i]) if i < len(scores) else 0.0
        else:
            calories = protein = carbs = fat = confidence = 0.0

        analyzed.append(
            AnalyzedIngredient(
                name=name,
                confidence_score=round(confidence, 3),
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
            )
        )

    dish_name = (
        (st.session_state.get("current_dish_analysis") or {})
        .get("dish_name", "Analyzed Dish")
    )

    return DishAnalysisResponse(
        dish_name=dish_name,
        total_calories=round(sum(a.calories for a in analyzed), 1),
        total_protein=round(sum(a.protein for a in analyzed), 1),
        total_carbs=round(sum(a.carbs for a in analyzed), 1),
        total_fat=round(sum(a.fat for a in analyzed), 1),
        ingredients=analyzed,
    )


def _reset_clarification() -> None:
    """Clear all clarification-related session state keys, including widget state.

    Widget keys (prefixed ``clar_``) must be explicitly popped so that
    Streamlit does not reuse stale values to prefill inputs on the next render.
    For example, ``clar_user_answer_input`` would otherwise carry the previous
    reply text into the first question of a new analysis session.
    """
    for key in (
        # Agent state
        "clar_active",
        "clar_dish_name",
        "clar_original_names",
        "clar_query_names",
        "clar_state",
        "clar_history",
        "clar_done",
        "clar_refined_result",
        "clar_error",
        "clar_initial_scores",
        "clar_prev_scores",
        "clar_fallback_queries",
        "clar_no_improvement_count",
        "clar_stopped_max_rounds",
        # Persistence flags
        "dish_saved",
        "current_place_id",
        # Widget state — must be cleared to prevent Streamlit from prefilling
        "clar_user_answer_input",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("clar_answer_"):
            st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 + 3 — clarification chat loop and refined final result
# ─────────────────────────────────────────────────────────────────────────────

def _render_clarification_section(client: NutriGraphClient) -> None:
    """
    Render the agent-assisted clarification chat (Steps 2 & 3).

    Layout:
      - Confidence-threshold info bar
      - Per-ingredient confidence scoreboard
      - Chat history (alternating assistant / user bubbles)
      - Current clarifying question + answer form  (Step 2, while open)
      - OR: success banner + agent-refined final result (Step 3, when done)

    The section is a no-op when no image analysis has been run yet.
    """
    if not st.session_state.get("clar_active"):
        return

    st.subheader("🤖 Step 2 — Agent Clarification")

    # ── Error state ───────────────────────────────────────────────────────────
    if st.session_state.get("clar_error"):
        st.error(f"⚠️ Clarification agent unavailable: {st.session_state.clar_error}")
        st.caption(
            "The one-shot estimate above is still valid. "
            "Agent clarification requires the ChromaDB ingredient index to be built."
        )
        return

    threshold = _CONFIDENCE_THRESHOLD
    clar_state: dict | None = st.session_state.get("clar_state")
    if clar_state:
        threshold = float(clar_state.get("threshold", threshold))

    st.caption(
        f"Confidence threshold: **{threshold:.0%}** — the agent asks clarifying questions "
        f"for ingredients whose retrieval score is below this value."
    )

    original_names: list[str] = st.session_state.get("clar_original_names", [])
    scores: list[float] = clar_state.get("scores", []) if clar_state else []
    initial_scores: list[float] = st.session_state.get("clar_initial_scores", [])
    prev_scores: list[float] = st.session_state.get("clar_prev_scores", [])

    # ── Overall confidence banner ──────────────────────────────────────────────
    if scores:
        avg_now = sum(scores) / len(scores)
        avg_initial = sum(initial_scores) / len(initial_scores) if initial_scores else avg_now
        total_delta = avg_now - avg_initial

        conf_col, bar_col = st.columns([1, 3])
        with conf_col:
            st.metric(
                "🎯 Overall confidence",
                f"{avg_now:.1%}",
                delta=f"{total_delta:+.1%} from initial" if initial_scores else None,
                delta_color="normal",
            )
        with bar_col:
            st.progress(
                min(avg_now, 1.0),
                text=f"Target: **{threshold:.0%}** — currently at **{avg_now:.1%}**",
            )

    # ── Per-ingredient confidence scoreboard ──────────────────────────────────
    if original_names and scores:
        with st.expander("📊 Per-ingredient confidence scores", expanded=True):
            cols = st.columns(min(len(original_names), 4))
            for i, name in enumerate(original_names):
                score = scores[i] if i < len(scores) else 0.0
                prev = prev_scores[i] if i < len(prev_scores) else None
                icon = "✅" if score >= threshold else "🔴"
                # delta shows round-over-round improvement for this ingredient
                delta_str = f"{score - prev:+.0%}" if prev is not None else None
                with cols[i % len(cols)]:
                    st.metric(
                        label=f"{icon} {name}",
                        value=f"{score:.0%}",
                        delta=delta_str,
                        delta_color="normal",
                        help=(
                            "Above threshold — high confidence"
                            if score >= threshold
                            else "Below threshold — clarification needed"
                        ),
                    )

    # ── Conversation history with per-turn confidence change ──────────────────
    history: list[dict] = st.session_state.get("clar_history", [])
    if history:
        st.markdown("#### Conversation")
        for turn in history:
            # Support both single Q&A and batch (multiple questions/answers per turn)
            q_list = turn.get("questions") if isinstance(turn.get("questions"), list) else ([turn["question"]] if turn.get("question") else [])
            a_list = turn.get("answers") if isinstance(turn.get("answers"), list) else ([turn["answer"]] if turn.get("answer") else [])
            for q, a in zip(q_list, a_list):
                with st.chat_message("assistant"):
                    st.markdown(q)
                with st.chat_message("user"):
                    st.markdown(a)
            # Show the confidence shift produced by this answer
            sb: list[float] = turn.get("scores_before", [])
            sa: list[float] = turn.get("scores_after", [])
            if sb and sa:
                avg_b = sum(sb) / len(sb)
                avg_a = sum(sa) / len(sa)
                diff = avg_a - avg_b
                arrow = "📈" if diff > 0.001 else ("📉" if diff < -0.001 else "➡️")
                color = "green" if diff > 0.001 else ("red" if diff < -0.001 else "gray")
                status = ""
                if diff < -0.001:
                    status = " ⚠️ <b>DROPPED</b>"
                elif diff > 0.001:
                    status = " ✓ improved"
                st.markdown(
                    f"<p style='color:{color};font-size:0.82em;margin:2px 0 8px 0'>"
                    f"{arrow}&nbsp;Overall confidence after this answer: "
                    f"<b>{avg_b:.1%} → {avg_a:.1%}</b> ({diff:+.1%}){status}</p>",
                    unsafe_allow_html=True,
                )

    # ── Step 3: converged — summary metrics + refined result ─────────────────
    if st.session_state.get("clar_done"):
        final_scores: list[float] = clar_state.get("scores", []) if clar_state else []
        final_avg = sum(final_scores) / len(final_scores) if final_scores else 0.0
        init_avg = sum(initial_scores) / len(initial_scores) if initial_scores else final_avg
        rounds = len(history)
        stopped_max_rounds = st.session_state.get("clar_stopped_max_rounds", False)

        if stopped_max_rounds:
            st.success(
                "✅ Reached the limit of follow-up questions (no improvement after 2 rounds for remaining ingredients). "
                "Here is our best estimate from the matches we have."
            )
        else:
            st.success(
                f"✅ All ingredients now meet the **{threshold:.0%}** confidence threshold!"
            )

        # Journey summary
        j1, j2, j3 = st.columns(3)
        with j1:
            st.metric("Initial confidence", f"{init_avg:.1%}")
        with j2:
            st.metric(
                "Final confidence",
                f"{final_avg:.1%}",
                delta=f"{final_avg - init_avg:+.1%}",
                delta_color="normal",
            )
        with j3:
            st.metric("Clarification rounds", str(rounds))

        refined_data = st.session_state.get("clar_refined_result")
        if refined_data:
            refined = DishAnalysisResponse(**refined_data)
            render_dish_detail(
                refined,
                label="Agent-refined Nutritional Estimate (after clarifications)",
            )

            # ── Save Refined Results button ────────────────────────────────────
            already_saved = st.session_state.get("dish_saved", False)
            if already_saved:
                st.success("✅ Saved to your history!")
            else:
                if st.button(
                    "💾 Save Refined Results",
                    key="save_refined_results",
                    type="primary",
                    help="Save the agent-refined estimate — this supersedes any earlier save.",
                    use_container_width=True,
                ):
                    _save_analysis(client, refined)

        if st.button("🔄 Clear & Start Over", key="clar_reset"):
            _reset_clarification()
            st.rerun()
        return

    # ── Step 2: ongoing — show the next clarifying question ───────────────────
    if not clar_state:
        st.info("Running agent analysis…")
        return

    questions: list[str] = clar_state.get("questions", [])
    low_conf_ingredients: list[str] = clar_state.get("low_conf_ingredients", [])
    low_conf_indices: list[int] = clar_state.get("low_conf_indices", [])

    if not questions:
        # The graph routed to 'ask' but produced no question text — unexpected.
        st.warning(
            "⚠️ The agent identified low-confidence ingredients but could not "
            "generate clarifying questions. Check that `low_conf_ingredients` is "
            "populated in the graph state."
        )
        return

    # Exclude ingredients that have had 2+ follow-up rounds with no/small improvement.
    no_improvement_count = st.session_state.get("clar_no_improvement_count") or {}
    filtered_tuples = [
        (idx, q, ing)
        for idx, q, ing in zip(low_conf_indices, questions, low_conf_ingredients)
        if no_improvement_count.get(idx, 0) < _MAX_NO_IMPROVEMENT_ROUNDS
    ]
    filtered_indices = [t[0] for t in filtered_tuples]
    filtered_questions = [t[1] for t in filtered_tuples]
    filtered_ingredients = [t[2] for t in filtered_tuples]

    # If no questions left after filtering, treat as done and show refined result.
    if not filtered_indices:
        st.session_state.clar_done = True
        st.session_state.clar_stopped_max_rounds = True  # stopped due to 2 no-improvement rounds per ingredient
        st.session_state.clar_refined_result = _build_refined_result(
            st.session_state.get("clar_original_names", []), clar_state
        ).model_dump()
        st.rerun()

    # Show ALL remaining low-confidence questions at once so the user can provide info for each.
    with st.chat_message("assistant"):
        st.markdown("To improve our match, please add details for each ingredient below:")

    with st.form("clar_reply_form", clear_on_submit=True):
        answer_keys: list[int] = []
        for i, (target_idx, q, ing) in enumerate(zip(filtered_indices, filtered_questions, filtered_ingredients)):
            placeholder_ing = original_names[target_idx] if target_idx < len(original_names) else ing
            st.markdown(f"**{ing}**")
            st.caption(q)
            st.text_input(
                "Your answer",
                placeholder=f"e.g. boiled, semolina, no sauce",
                key=f"clar_answer_{target_idx}",
                label_visibility="collapsed",
            )
            answer_keys.append(target_idx)

        submitted = st.form_submit_button("Send all answers ➤", type="primary", width='stretch')

    if submitted:
        # Collect answers in order of low_conf_indices (same order as questions)
        answers: list[str] = []
        for target_idx in answer_keys:
            val = st.session_state.get(f"clar_answer_{target_idx}", "") or ""
            answers.append(val.strip())

        if not any(answers):
            st.warning("Please fill in at least one answer.")
        else:
            # Capture scores BEFORE re-running
            scores_before = list(clar_state.get("scores", []))

            # One LLM call to get refined phrases for all (ingredient, answer) pairs
            dish_name = st.session_state.get("clar_dish_name", "")
            from ..ml.clarification_questions import refine_ingredients_batch

            ingredients_with_answers = [
                (st.session_state.clar_query_names[target_idx], answers[i])
                for i, target_idx in enumerate(answer_keys)
            ]
            refined_list = refine_ingredients_batch(ingredients_with_answers, dish_name)

            for i, target_idx in enumerate(answer_keys):
                current_query = st.session_state.clar_query_names[target_idx]
                st.session_state.clar_fallback_queries[target_idx] = current_query
                st.session_state.clar_query_names[target_idx] = refined_list[i] if i < len(refined_list) else current_query

            # Re-run the full clarification graph once with all updated queries
            with st.spinner("Re-analysing all ingredients…"):
                _run_clarification_graph()

            new_state = st.session_state.get("clar_state") or {}
            scores_after = list(new_state.get("scores", []))

            # Keep the winning query for indices where fallback was better, so next round
            # we don't drop below the best score we already had (overall confidence won't decrease).
            used_fallback = new_state.get("used_fallback_indices") or []
            for idx in used_fallback:
                if idx in st.session_state.clar_fallback_queries:
                    st.session_state.clar_query_names[idx] = st.session_state.clar_fallback_queries[idx]

            # Update per-ingredient no-improvement count: stop asking after 2 rounds with negligible improvement.
            no_improvement = st.session_state.get("clar_no_improvement_count") or {}
            for i, target_idx in enumerate(answer_keys):
                sb = scores_before[target_idx] if target_idx < len(scores_before) else 0.0
                sa = scores_after[target_idx] if target_idx < len(scores_after) else 0.0
                delta = sa - sb
                if delta < _NEGLIGIBLE_IMPROVEMENT:
                    no_improvement[target_idx] = no_improvement.get(target_idx, 0) + 1
                else:
                    no_improvement[target_idx] = 0
            st.session_state.clar_no_improvement_count = no_improvement

            st.session_state.clar_history.append(
                {
                    "questions": filtered_questions,
                    "answers": [answers[i] if i < len(answers) else "" for i in range(len(filtered_questions))],
                    "scores_before": scores_before,
                    "scores_after": scores_after,
                }
            )
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Feedback section
# ─────────────────────────────────────────────────────────────────────────────

def _render_feedback_section() -> None:
    """Feedback form for flagging incorrect AI-generated data."""
    with st.expander("🚩 Flag Incorrect Data or Suggest an Edit"):
        st.caption("Help NutriGraph's AI improve by reporting inaccuracies")

        st.selectbox(
            "Issue type",
            options=_FEEDBACK_ISSUE_TYPES,
            key="feedback_issue_type",
        )

        st.text_area(
            "Additional details",
            placeholder=(
                "e.g., The grilled salmon portion should be 180 g, not 100 g. "
                "The avocado was missing from the ingredient list."
            ),
            key="feedback_details",
        )

        if st.button("Submit Feedback", key="submit_feedback"):
            st.success(
                "Thank you! Your feedback will be used to improve our AI's accuracy."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers — text-search workflow
# ─────────────────────────────────────────────────────────────────────────────

def _render_dish_search_section(client: NutriGraphClient) -> None:
    """Text-based dish search using the mock/RAG estimation pipeline."""
    st.subheader("🔍 Dish Search / Log")

    col1, col2 = st.columns([2, 1])
    with col1:
        dish_name = st.text_input(
            "Dish name",
            placeholder="e.g., Chicken Alfredo Pasta",
            key="diner_dish_name",
        )
    with col2:
        restaurant_name = st.text_input(
            "Restaurant (optional)",
            placeholder="e.g., Olive Garden",
            key="diner_restaurant",
        )

    if st.button("🔮 Estimate Nutrition", type="primary", width='stretch'):
        if not dish_name:
            st.warning("Please enter a dish name.")
            return

        with st.spinner("Estimating nutrition…"):
            dish = Dish(
                name=dish_name,
                restaurant=restaurant_name if restaurant_name else None,
            )
            estimate = client.estimate_nutrition(dish)
            mock_ingredients = generate_mock_ingredients(dish_name)

            st.session_state.last_estimate = {
                "dish": dish.model_dump(),
                "estimate": estimate.model_dump(),
                "ingredients": [ing.model_dump() for ing in mock_ingredients],
            }

        st.success(f"Nutrition estimated for **{dish_name}**!")


def _render_dish_detail_section() -> None:
    """Detail view for the legacy text-search result stored in session state."""
    st.subheader("📋 Estimated Nutrition")

    if st.session_state.last_estimate is None:
        st.info("Search for a dish above to see nutrition details.")
        return

    data = st.session_state.last_estimate
    dish_data = data["dish"]
    estimate_data = data["estimate"]
    ingredients_data = data["ingredients"]

    dish_label = f"**{dish_data['name']}**"
    if dish_data.get("restaurant"):
        dish_label += f" from {dish_data['restaurant']}"
    st.markdown(dish_label)

    estimate = NutritionEstimate(**estimate_data)
    render_macro_card(estimate)

    st.markdown("#### Estimation Confidence")
    render_confidence_indicator(estimate.confidence)

    st.markdown("#### Estimated Ingredients")
    st.caption("⚠️ Ingredients are estimated and may not reflect the actual dish")
    render_ingredients_table([Ingredient(**ing) for ing in ingredients_data])


def _render_tracking_section() -> None:
    """Personalised daily tracking placeholder."""
    st.subheader("📈 Personalised Tracking")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.date_input("Select date", value=date.today(), key="tracking_date")
    with col2:
        st.markdown("#### Daily Totals (Mock)")
        daily_cols = st.columns(4)
        with daily_cols[0]:
            st.metric("Calories", "1,847")
        with daily_cols[1]:
            st.metric("Protein", "89g")
        with daily_cols[2]:
            st.metric("Carbs", "204g")
        with daily_cols[3]:
            st.metric("Fat", "72g")

    st.info(
        "📌 **Feature coming soon:** Full meal logging, daily/weekly trends, "
        "and personalised nutrition goals."
    )
