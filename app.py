"""
NutriGraph - Streamlit Application Entrypoint

A nutrition estimation and tracking application with separate interfaces
for diners (consumers) and restaurants.
"""
import streamlit as st

from src.core.config import settings
from src.core.api_client import NutriGraphClient
from src.ui.components import initialize_session_state, reset_session_state
from src.ui.diner import render_diner
from src.ui.restaurant import render_restaurant


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon=settings.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_sidebar() -> NutriGraphClient:
    """
    Render the sidebar with configuration options.
    
    Returns:
        Configured NutriGraphClient instance.
    """
    with st.sidebar:
        st.title(f"{settings.APP_ICON} {settings.APP_TITLE}")
        st.caption("Nutrition Estimation & Tracking")
        
        st.divider()
        
        # Environment selector
        st.subheader("⚙️ Configuration")
        
        environment = st.selectbox(
            "Environment",
            options=settings.ENVIRONMENTS,
            index=settings.ENVIRONMENTS.index(
                settings.ENVIRONMENT.capitalize()
            ) if settings.ENVIRONMENT.capitalize() in settings.ENVIRONMENTS else 0,
            help="Select the deployment environment"
        )
        
        # Backend URL input
        default_url = settings.BACKEND_URL
        if environment == "Staging":
            default_url = "https://staging-api.nutrigraph.io"
        
        backend_url = st.text_input(
            "Backend URL",
            value=default_url,
            help="URL of the NutriGraph backend API"
        )
        
        st.divider()
        
        # Reset session button
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_session_state()
            st.success("Session reset!")
            st.rerun()
        
        st.divider()
        
        # Status info
        st.subheader("📊 Session Info")
        st.caption(f"Catalog items: {len(st.session_state.get('catalog', []))}")
        st.caption(f"Environment: {environment}")
        
        # Footer
        st.divider()
        st.caption("NutriGraph v0.1.0")
        st.caption("Course Project - 2026")
    
    return NutriGraphClient(base_url=backend_url)


def render_main_content(client: NutriGraphClient) -> None:
    """
    Render the main content area with tabs.
    
    Args:
        client: NutriGraphClient instance for API calls.
    """
    # Create top-level tabs
    diner_tab, restaurant_tab = st.tabs(["🍽️ Diner", "🍳 Restaurant"])
    
    with diner_tab:
        render_diner(client)
    
    with restaurant_tab:
        render_restaurant(client)


def main() -> None:
    """Main application entry point."""
    # Configure page (must be first Streamlit command)
    configure_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar and get configured client
    client = render_sidebar()
    
    # Render main content
    render_main_content(client)


if __name__ == "__main__":
    main()
