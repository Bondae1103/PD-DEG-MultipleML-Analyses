"""
app/ui/components.py
--------------------
Reusable visual components, disclaimers, and badges for the PD-DEG ML-Methods Dashboard.
"""

import streamlit as st
from app.config import FALLBACK_FILTER_NAME, GEO_ACCESSION

def render_header(title: str, subtitle: str):
    """Renders consistent page titles with subtitles."""
    st.markdown(f"<h1 style='margin-bottom: 0.2rem; font-size: 1.85rem; font-weight: 800; color: #0F172A;'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #475569; font-size: 0.95rem; margin-bottom: 1.25rem;'>{subtitle}</p>", unsafe_allow_html=True)

def render_provenance_badge(data_source_label: str):
    """Displays dataset accession badge and cohort size."""
    st.markdown(f"""
    <div style='margin-bottom: 1rem;'>
        <span class='badge-primary'>GEO: {GEO_ACCESSION}</span>
        <span class='badge-primary'>Human Frontal Cortex (BA9)</span>
        <span class='badge-primary'>72 Samples (44 Control / 28 PD)</span>
        <span class='badge-success'>Source: {data_source_label}</span>
    </div>
    """, unsafe_allow_html=True)

def render_fallback_warning():
    """Prominent persistent warning banner when the fallback DEG filter is used."""
    st.warning(
        f"⚠️ **Notice: Using {FALLBACK_FILTER_NAME}**\n\n"
        "This analysis was generated using a vectorized two-sample Welch's t-test with Benjamini-Hochberg FDR correction. "
        "This is an **approximate substitute** provided for user exploration on raw counts, and is **not** the `limma-voom` "
        "empirical Bayes linear modeling method used in the original research paper."
    )

def render_disclaimer():
    """Research use only disclaimer."""
    st.markdown("""
    <div class='footer-text'>
        🔬 <strong>For Research Demonstration Only.</strong> This application showcases a 5-method feature selection, consensus, 
        and leave-one-out cross-validation pipeline on postmortem Parkinson's disease transcriptomics. 
        It is not a diagnostic tool and must not be used for clinical decision-making.
    </div>
    """, unsafe_allow_html=True)
