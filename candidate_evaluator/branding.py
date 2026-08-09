"""Shared Catalyst branding (logo path + CSS) for the web apps.

Both the local (`web_app.py`) and cloud (`web_app_cloud.py`) interfaces import
from here so the look-and-feel stays in sync. Colors are derived from the
Catalyst @ MIT linQ logo/brand:

- Teal / cyan  : #00B2CA  (primary accent)
- Magenta      : #9F2D7D  (secondary accent)
- Navy wordmark: #1C1C34  (headings / text)

The theme base colors (primaryColor, sidebar background, text) live in
`.streamlit/config.toml`; this module layers on the font, heading accents, and
metric-card styling that Streamlit's theme options can't express.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Absolute path to the logo so it resolves regardless of the working directory.
LOGO_PATH = str(Path(__file__).parent / "assets" / "catalyst_logo.png")

# Brand palette.
BRAND_TEAL = "#00B2CA"
BRAND_TEAL_DARK = "#0091A6"
BRAND_MAGENTA = "#9F2D7D"
BRAND_NAVY = "#1C1C34"
BRAND_SIDEBAR_TINT = "#EEF9FB"

BRAND_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Base font across the whole app */
html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

/* Headings in brand navy with a teal accent under the page title */
h1, h2, h3, h4 {{
    color: {BRAND_NAVY};
    font-weight: 700;
    letter-spacing: -0.01em;
}}
h1 {{
    border-bottom: 3px solid {BRAND_TEAL};
    padding-bottom: 0.3rem;
    display: inline-block;
}}
h2 {{
    color: {BRAND_TEAL_DARK};
}}

/* Sidebar tint + divider */
[data-testid="stSidebar"] {{
    background-color: {BRAND_SIDEBAR_TINT};
    border-right: 1px solid #D6EEF2;
}}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h1 {{
    color: {BRAND_NAVY};
    border-bottom: none;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid #E3E8EF;
    border-left: 4px solid {BRAND_TEAL};
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(28, 28, 52, 0.06);
}}
[data-testid="stMetricValue"] {{
    color: {BRAND_NAVY};
    font-weight: 700;
}}
[data-testid="stMetricLabel"] {{
    color: #5A6472;
}}

/* Selected tab underline in brand teal */
[data-baseweb="tab-highlight"] {{
    background-color: {BRAND_TEAL} !important;
}}
</style>
"""


def apply_branding() -> None:
    """Render the Catalyst logo (top-left) and inject brand CSS.

    Call once near the top of each app's ``main()``.
    """
    try:
        st.logo(LOGO_PATH, size="large")
    except Exception:
        # st.logo requires a recent Streamlit; never let branding break the app.
        pass
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
