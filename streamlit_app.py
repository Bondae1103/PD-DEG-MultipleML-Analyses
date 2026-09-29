"""
streamlit_app.py
----------------
Main entry point for the PD-DEG ML-Methods Dashboard.
Configures page settings and registers native multipage navigation.
Follows Plan v2 Section 2 and Section 6.1.
"""

import streamlit as st

# Configure page settings once per application session
st.set_page_config(
    page_title="PD Biomarker Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Multi-page navigation structure (Streamlit >= 1.36)
landing_page = st.Page("pages/1_Landing.py", title="1. Setup & Run", icon="🚀", default=True)
output_page = st.Page("pages/2_Output.py", title="2. Results & Validation", icon="📊")
about_page = st.Page("pages/3_About.py", title="3. Methodology & Docs", icon="📖")

pg = st.navigation({
    "Pipeline Workbench": [landing_page, output_page],
    "Documentation": [about_page]
})

pg.run()
