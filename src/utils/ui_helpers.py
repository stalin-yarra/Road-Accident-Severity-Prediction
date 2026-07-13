"""
src/utils/ui_helpers.py
========================
Shared Streamlit UI building blocks — page configuration, custom CSS, sidebar
branding, and small display widgets (severity badges, metric cards) — reused
by `app/streamlit_app.py` and every page under `app/pages/`. Centralizing
these avoids duplicating the same `st.markdown(<style>...)` block six times.
"""

from __future__ import annotations

import streamlit as st

from src.config.settings import ASSETS_DIR, LOGO_PATH, SEVERITY_COLORS
from src.utils.io_helpers import image_exists

_CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        font-weight: 700;
    }
    .app-hero {
        padding: 1.75rem 2rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #f9fafb;
        margin-bottom: 1.5rem;
    }
    .app-hero h1 {
        color: #f9fafb;
        margin-bottom: 0.25rem;
    }
    .app-hero p {
        color: #d1d5db;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #f8f9fb;
        border: 1px solid #e5e7eb;
        border-radius: 0.6rem;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .severity-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #f3f4f6 !important;
    }
</style>
"""


def configure_page(page_title: str, page_icon: str = "🚦") -> None:
    """
    Apply consistent `st.set_page_config` and inject the shared custom CSS.
    Must be the first Streamlit call on any page.

    Args:
        page_title: Title shown in the browser tab.
        page_icon: Emoji or path used as the browser tab icon.
    """
    st.set_page_config(
        page_title=f"{page_title} | Accident Severity Prediction",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_branding() -> None:
    """
    Render the logo (if present) and a short project caption at the top of
    the sidebar, above Streamlit's auto-generated page navigation links.
    """
    with st.sidebar:
        if image_exists(LOGO_PATH):
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.markdown("### 🚦 Accident Severity Prediction")
        st.caption("Road Accident Severity Prediction using Machine Learning")
        st.divider()


def render_hero(title: str, subtitle: str) -> None:
    """
    Render the large gradient "hero" header used at the top of every page.

    Args:
        title: Main page title.
        subtitle: One-line description shown beneath the title.
    """
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, column=None) -> None:
    """
    Render a single styled metric card (label + large value).

    Args:
        label: Short label shown above the value (e.g., "Accuracy").
        value: The formatted metric value to display (e.g., "84.2%").
        column: An optional Streamlit column/container to render into;
            defaults to the main flow if not provided.
    """
    target = column if column is not None else st
    target.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_severity_badge(severity: str) -> str:
    """
    Build the HTML for a colored severity badge, using the shared
    Slight/Serious/Fatal color palette from `src/config/settings.py`.

    Args:
        severity: One of the class labels (e.g., "Fatal").

    Returns:
        An HTML string ready to pass to `st.markdown(..., unsafe_allow_html=True)`.
    """
    color = SEVERITY_COLORS.get(severity, "#6b7280")
    return f'<span class="severity-badge" style="background-color:{color};">{severity}</span>'


def render_missing_artifact_notice(artifact_description: str, generating_notebook: str) -> None:
    """
    Render a consistent, friendly notice when a report/figure/CSV this page
    wants to display has not been generated yet, instead of a raw
    file-not-found error.

    Args:
        artifact_description: Human-readable description of what's missing
            (e.g., "the model comparison table").
        generating_notebook: The notebook filename that produces it (e.g.,
            "06_Model_Training_and_Comparison.ipynb").
    """
    st.info(
        f"ℹ️ {artifact_description} hasn't been generated yet. "
        f"Run `notebooks/{generating_notebook}` to produce it, then refresh this page."
    )
