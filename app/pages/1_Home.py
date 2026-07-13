"""
app/pages/1_Home.py
======================
Home page: project overview, objectives, architecture, and workflow diagram.
Rendering logic lives in `app/pages_content/home_content.py` and is shared
with `app/streamlit_app.py` to avoid duplicated code.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ui_helpers import configure_page, render_sidebar_branding  # noqa: E402
from app.pages_content.home_content import render_home_page  # noqa: E402

configure_page(page_title="Home", page_icon="🚦")
render_sidebar_branding()
render_home_page()
