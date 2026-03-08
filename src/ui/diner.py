"""
Diner tab UI for NutriGraph.

This module handles the consumer-facing interface for:
  - Uploading a dish photo and triggering the Gemini vision pipeline
  - Viewing the AI-generated nutritional breakdown (Dish Detail View)
  - Agent-assisted clarification conversation for high-confidence estimates
  - Searching dishes by name (legacy text-based flow, kept for compatibility)
  - Personalised daily tracking (placeholder)
  - Submitting accuracy feedback
"""
import streamlit as st
from datetime import date

import pandas as pd

from ..core.models import (
    ConversationState,
    ConversationTurn,
    Dish,
    DishAnalysisResponse,
    MessageType,
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

    # Ensure all session-state keys exist regardless of app boot order
    st.session_state.setdefault("current_dish_analysis", None)
    st.session_state.setdefault("last_estimate", None)
    st.session_state.setdefault("current_conversation", None)
    st.session_state.setdefault("conversation_messages", [])

    # ── 1. Image upload & AI analysis ────────────────────────────────────────
    _render_image_analysis_section(client)

    st.divider()

    # ── 2. Dish Detail View (one-shot AI result) ──────────────────────────────
    _render_analysis_detail_section()

    st.divider()

    # ── 3. Agent-assisted conversation ───────────────────────────────────────
    _render_agent_conversation_section(client)

    st.divider()

    # ── 4. Feedback form ─────────────────────────────────────────────────────
    _render_feedback_section()

    st.divider()

    # ── 5. Legacy text-search (kept for backward compatibility) ──────────────
    with st.expander("🔍 Search by Dish Name (Legacy)", expanded=False):
        _render_dish_search_section(client)
        if st.session_state.last_estimate is not None:
            st.divider()
            _render_dish_detail_section()

    st.divider()

    # ── 6. Personalised tracking placeholder ─────────────────────────────────
    _render_tracking_section()


# ─────────────────────────────────────────────────────────────────────────────
# Shared detail-view helper (used by both one-shot and agent-refined paths)
# ─────────────────────────────────────────────────────────────────────────────

def render_dish_detail(result: DishAnalysisResponse, *, label: str = "Dish Detail View") -> None:
    """
    Render a :class:`DishAnalysisResponse` as a structured nutritional breakdown.

    This helper is intentionally decoupled from session state so it can be
    called both from the one-shot image analysis flow and from the agent-refined
    final-result flow.

    Args:
        result: The nutritional analysis to display.
        label:  Section heading shown above the breakdown.  Callers can pass a
                custom label (e.g. "Agent-refined Nutritional Estimate (after
                clarifications)") to distinguish sources.
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
# Section helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_image_analysis_section(client: NutriGraphClient) -> None:
    """Image upload widget + 'Analyze Dish' action button."""
    st.subheader("📸 Upload Dish Photo")

    uploaded_file = st.file_uploader(
        "Select an image of your dish",
        type=["png", "jpg", "jpeg"],
        key="dish_image_upload",
        help="Supported formats: PNG, JPG, JPEG",
    )

    if uploaded_file is not None:
        col_img, col_spacer = st.columns([1, 2])
        with col_img:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if st.button("🔍 Analyze Dish", type="primary", use_container_width=True):
            with st.spinner("Analyzing image and retrieving nutritional data..."):
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    response: DishAnalysisResponse = client.analyze_dish_image(
                        image_bytes, uploaded_file.name
                    )
                    st.session_state.current_dish_analysis = response.model_dump()
                    st.success(f"Analysis complete for **{response.dish_name}**!")

                except NutriGraphAPIError as exc:
                    st.error(f"⚠️ Analysis failed: {exc}")

                except Exception as exc:
                    st.error(f"⚠️ An unexpected error occurred: {exc}")
    else:
        st.info("Upload a dish photo above and click **Analyze Dish** to get started.")


def _render_analysis_detail_section() -> None:
    """
    One-shot Dish Detail View — rendered when a successful DishAnalysisResponse is
    stored in ``st.session_state.current_dish_analysis``.
    """
    if not st.session_state.get("current_dish_analysis"):
        st.subheader("📊 Dish Detail View")
        st.info("Nutritional details will appear here after you analyze a dish photo.")
        return

    analysis = DishAnalysisResponse(**st.session_state.current_dish_analysis)
    render_dish_detail(analysis, label="Dish Detail View")

    if st.button("🗑️ Clear Analysis", key="clear_analysis"):
        st.session_state.current_dish_analysis = None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Agent conversation section
# ─────────────────────────────────────────────────────────────────────────────

def _render_agent_conversation_section(client: NutriGraphClient) -> None:
    """
    Agent-assisted Nutrition Estimate section.

    Allows the user to start a multi-turn clarification dialogue with the
    LangGraph backend agent.  The section handles:

    * Starting a new conversation (POST /api/v1/agent/start).
    * Rendering all past turns as styled chat bubbles.
    * Collecting and submitting user replies (POST /api/v1/agent/continue).
    * Displaying the agent-refined final result once the agent converges.
    """
    st.subheader("🤖 Agent-assisted Nutrition Estimate")
    st.caption(
        "Let the AI agent ask clarifying questions to produce a high-confidence "
        "nutritional estimate for your dish."
    )

    # ── Start-conversation form ───────────────────────────────────────────────
    with st.form("agent_start_form", clear_on_submit=False):
        agent_dish_name = st.text_input(
            "Dish name *",
            placeholder="e.g., Chicken Tikka Masala",
            key="agent_dish_name_input",
        )
        agent_restaurant = st.text_input(
            "Restaurant (optional)",
            placeholder="e.g., Dishoom",
            key="agent_restaurant_input",
        )
        start_clicked = st.form_submit_button(
            "▶️ Start Agent Conversation", type="primary", use_container_width=True
        )

    if start_clicked:
        if not agent_dish_name.strip():
            st.warning("Please enter a dish name to start the conversation.")
        else:
            # Build initial payload; attach image analysis ID if one exists
            initial_input: dict = {"dish_name": agent_dish_name.strip()}
            if agent_restaurant.strip():
                initial_input["restaurant_name"] = agent_restaurant.strip()
            if st.session_state.get("current_dish_analysis"):
                # Pass the dish name from the prior image analysis as a soft hint.
                # When the backend assigns UUIDs to image analyses, swap this for the real ID.
                initial_input["image_analysis_id"] = st.session_state.current_dish_analysis.get(
                    "dish_name", ""
                )

            # Reset conversation state
            st.session_state.current_conversation = None
            st.session_state.conversation_messages = []

            with st.spinner("Starting agent conversation…"):
                try:
                    state: ConversationState = client.start_dish_conversation(initial_input)
                    st.session_state.current_conversation = state.model_dump()
                    st.session_state.conversation_messages = [
                        {"role": t.role, "type": t.type, "message": t.message}
                        for t in state.history
                    ]
                except NutriGraphAPIError as exc:
                    st.error(f"⚠️ Could not start conversation: {exc}")
                except Exception as exc:
                    st.error(f"⚠️ Unexpected error: {exc}")

    # ── Chat display ─────────────────────────────────────────────────────────
    if not st.session_state.conversation_messages:
        return

    st.markdown("---")
    st.markdown("#### Conversation")

    for turn in st.session_state.conversation_messages:
        role = turn.get("role", "agent")
        message = turn.get("message", "")
        avatar = "🤖" if role == "agent" else "🧑"
        with st.chat_message(role if role == "user" else "assistant"):
            st.markdown(message)

    # ── Reply input (only while the agent still has an open question) ─────────
    conv_state = st.session_state.get("current_conversation")
    if conv_state is None:
        return

    state_obj = ConversationState(**conv_state)

    # Check whether the most recent agent turn is a question
    last_agent_turn = next(
        (t for t in reversed(state_obj.history) if t.role == "agent"), None
    )
    conversation_open = (
        last_agent_turn is not None
        and last_agent_turn.type == MessageType.question
        and state_obj.final_result is None
    )

    if conversation_open:
        with st.form("agent_reply_form", clear_on_submit=True):
            user_reply = st.text_input(
                "Your answer",
                placeholder="Type your reply here…",
                key="agent_user_reply",
            )
            send_clicked = st.form_submit_button(
                "📨 Send", type="primary", use_container_width=True
            )

        if send_clicked:
            if not user_reply.strip():
                st.warning("Please type an answer before sending.")
            else:
                # Optimistically add the user turn to the visible message list
                st.session_state.conversation_messages.append(
                    {"role": "user", "type": MessageType.answer, "message": user_reply.strip()}
                )

                with st.spinner("Agent is thinking…"):
                    try:
                        updated: ConversationState = client.continue_dish_conversation(
                            dish_id=state_obj.dish_id,
                            user_message=user_reply.strip(),
                        )
                        st.session_state.current_conversation = updated.model_dump()
                        # Rebuild the full message list from the authoritative backend state
                        st.session_state.conversation_messages = [
                            {"role": t.role, "type": t.type, "message": t.message}
                            for t in updated.history
                        ]
                    except NutriGraphAPIError as exc:
                        st.error(f"⚠️ Could not send reply: {exc}")
                        # Roll back the optimistic user turn so the UI stays consistent
                        st.session_state.conversation_messages.pop()
                    except Exception as exc:
                        st.error(f"⚠️ Unexpected error: {exc}")
                        st.session_state.conversation_messages.pop()

                st.rerun()

    # ── Final result ──────────────────────────────────────────────────────────
    if state_obj.final_result is not None:
        st.success("✅ High-confidence estimate reached!")
        render_dish_detail(
            state_obj.final_result,
            label="Agent-refined Nutritional Estimate (after clarifications)",
        )

        if st.button("🔄 Start New Conversation", key="reset_conversation"):
            st.session_state.current_conversation = None
            st.session_state.conversation_messages = []
            st.rerun()


def _render_feedback_section() -> None:
    """Feedback form for flagging incorrect AI-generated data."""
    with st.expander("🚩 Flag Incorrect Data or Suggest an Edit"):
        st.caption("Help NutriGraph's AI improve by reporting inaccuracies")

        issue_type = st.selectbox(
            "Issue type",
            options=_FEEDBACK_ISSUE_TYPES,
            key="feedback_issue_type",
        )

        additional_details = st.text_area(
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
# Legacy helpers (text-search workflow)
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

        with st.spinner("Estimating nutrition..."):
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
