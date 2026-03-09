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
    _render_analysis_detail_section()

    st.divider()

    # ── Step 2 + 3: Agent clarification loop & refined final result ───────────
    _render_clarification_section()

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

    # Clarification agent state
    st.session_state.setdefault("clar_active", False)
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
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No per-ingredient breakdown was returned by the model.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — image upload & analysis
# ─────────────────────────────────────────────────────────────────────────────

def _render_image_analysis_section(client: NutriGraphClient) -> None:
    """Image upload widget and 'Analyze Dish' action button (Step 1)."""
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
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if st.button("🔍 Analyze Dish", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and retrieving nutritional data…"):
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    response: DishAnalysisResponse = client.analyze_dish_image(
                        image_bytes, uploaded_file.name
                    )
                    st.session_state.current_dish_analysis = response.model_dump()

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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 result — one-shot dish detail view
# ─────────────────────────────────────────────────────────────────────────────

def _render_analysis_detail_section() -> None:
    """One-shot Dish Detail View rendered right after the image is analysed."""
    if not st.session_state.get("current_dish_analysis"):
        st.subheader("📊 Dish Detail View")
        st.info("Nutritional details will appear here after you analyze a dish photo.")
        return

    analysis = DishAnalysisResponse(**st.session_state.current_dish_analysis)
    render_dish_detail(analysis, label="Dish Detail View (one-shot estimate)")

    if st.button("🗑️ Clear Analysis", key="clear_analysis"):
        st.session_state.current_dish_analysis = None
        _reset_clarification()
        st.rerun()


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
    st.session_state.clar_original_names = names
    st.session_state.clar_query_names = list(names)   # mutable copy
    st.session_state.clar_history = []
    st.session_state.clar_done = False
    st.session_state.clar_refined_result = None
    st.session_state.clar_error = None
    st.session_state.clar_state = None
    st.session_state.clar_initial_scores = []   # reset baseline for this dish
    st.session_state.clar_prev_scores = []      # reset round-over-round delta
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
        result: dict = graph.invoke({"ingredients": names})
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
    """Clear all clarification-related session state keys."""
    for key in (
        "clar_active",
        "clar_original_names",
        "clar_query_names",
        "clar_state",
        "clar_history",
        "clar_done",
        "clar_refined_result",
        "clar_error",
        "clar_initial_scores",
        "clar_prev_scores",
    ):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 + 3 — clarification chat loop and refined final result
# ─────────────────────────────────────────────────────────────────────────────

def _render_clarification_section() -> None:
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
            with st.chat_message("assistant"):
                st.markdown(turn["question"])
            with st.chat_message("user"):
                st.markdown(turn["answer"])
            # Show the confidence shift produced by this answer
            sb: list[float] = turn.get("scores_before", [])
            sa: list[float] = turn.get("scores_after", [])
            if sb and sa:
                avg_b = sum(sb) / len(sb)
                avg_a = sum(sa) / len(sa)
                diff = avg_a - avg_b
                arrow = "📈" if diff > 0.001 else ("📉" if diff < -0.001 else "➡️")
                color = "green" if diff > 0.001 else ("red" if diff < -0.001 else "gray")
                st.markdown(
                    f"<p style='color:{color};font-size:0.82em;margin:2px 0 8px 0'>"
                    f"{arrow}&nbsp;Overall confidence after this answer: "
                    f"<b>{avg_b:.1%} → {avg_a:.1%}</b> ({diff:+.1%})</p>",
                    unsafe_allow_html=True,
                )

    # ── Step 3: converged — summary metrics + refined result ─────────────────
    if st.session_state.get("clar_done"):
        final_scores: list[float] = clar_state.get("scores", []) if clar_state else []
        final_avg = sum(final_scores) / len(final_scores) if final_scores else 0.0
        init_avg = sum(initial_scores) / len(initial_scores) if initial_scores else final_avg
        rounds = len(history)

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

    # Always surface the FIRST pending question; the rest follow in subsequent reruns.
    current_question = questions[0]
    current_ingredient = low_conf_ingredients[0] if low_conf_ingredients else "ingredient"
    target_idx = low_conf_indices[0] if low_conf_indices else 0

    with st.chat_message("assistant"):
        st.markdown(current_question)

    with st.form("clar_reply_form", clear_on_submit=True):
        user_answer = st.text_input(
            "Your answer",
            placeholder=f"Describe the {current_ingredient} in more detail…",
            key="clar_user_answer_input",
        )
        submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

    if submitted:
        if not user_answer.strip():
            st.warning("Please type an answer before sending.")
        else:
            # Capture scores BEFORE re-running so we can show the delta in history.
            scores_before = list(clar_state.get("scores", []))

            # Enrich the query string so the next graph pass gets more context.
            current_query = st.session_state.clar_query_names[target_idx]
            st.session_state.clar_query_names[target_idx] = (
                f"{current_query} {user_answer.strip()}"
            )

            # Re-run the full clarification graph with the updated ingredient list.
            with st.spinner("Agent is re-analysing…"):
                _run_clarification_graph()

            # Capture scores AFTER re-running and store the full turn record.
            new_state = st.session_state.get("clar_state") or {}
            scores_after = list(new_state.get("scores", []))
            st.session_state.clar_history.append(
                {
                    "question": current_question,
                    "answer": user_answer.strip(),
                    "scores_before": scores_before,
                    "scores_after": scores_after,
                }
            )

            # Force a re-render so updated history and new question appear immediately.
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

    if st.button("🔮 Estimate Nutrition", type="primary", use_container_width=True):
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
