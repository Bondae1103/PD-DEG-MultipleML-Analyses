"""
app/ui/nav.py
-------------
Top navigation bar component for the PD-DEG ML-Methods Dashboard.
"""

import streamlit as st

def render_top_nav(current_page: str = "Landing"):
    """
    Renders clean top-bar page links across pages.
    Guarded with try/except so AppTest / bare environments don't crash on uninitialized page registries.
    """
    col1, col2, col3, col_fill = st.columns([1.5, 1.8, 1.8, 3.5])
    try:
        with col1:
            st.page_link("pages/1_Landing.py", label="🚀 1. Setup & Run", use_container_width=True)
        with col2:
            st.page_link("pages/2_Output.py", label="📊 2. Results & Validation", use_container_width=True)
        with col3:
            st.page_link("pages/3_About.py", label="📖 3. Methodology & Docs", use_container_width=True)
    except Exception:
        pass

    st.markdown("<hr style='margin-top: 0.2rem; margin-bottom: 1.2rem; border-color: #E2E8F0;'>", unsafe_allow_html=True)
