"""
app/streamlit_app.py
======================
Main entry point for the Road Accident Severity Prediction Streamlit
application. Run with:

    streamlit run app/streamlit_app.py

Streamlit automatically discovers every script under `app/pages/` and lists
them in the sidebar navigation. This file renders the same content as
`pages/1_Home.py` so a user landing on the bare entry point still sees a
complete, useful page rather than a blank shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
# regardless of the working directory Streamlit is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_helpers import configure_page, render_sidebar_branding  # noqa: E402
from app.pages_content.home_content import render_home_page  # noqa: E402

configure_page(page_title="Home", page_icon="🚦")
render_sidebar_branding()
render_home_page()
