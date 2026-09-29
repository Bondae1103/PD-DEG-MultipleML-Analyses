"""
app/ui/styles.py
----------------
CSS injection and typography styling for the PD-DEG ML-Methods Dashboard.
Single indigo accent palette, clean clinical typography.
"""

import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
    /* Global Container styling */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Badges */
    .badge-primary {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #C7D2FE;
        margin-right: 0.4rem;
    }

    .badge-success {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #A7F3D0;
    }

    .badge-warning {
        display: inline-block;
        background-color: #FFFBEB;
        color: #92400E;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid #FDE68A;
    }

    /* Top Navigation bar */
    .top-nav {
        display: flex;
        gap: 1.5rem;
        align-items: center;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }

    /* Metric card */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.25rem;
    }

    /* Notice banner */
    .notice-banner {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
        font-size: 0.875rem;
        color: #334155;
    }

    /* Regulatory footer */
    .footer-text {
        font-size: 0.75rem;
        color: #94A3B8;
        border-top: 1px solid #E2E8F0;
        padding-top: 1rem;
        margin-top: 2.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
