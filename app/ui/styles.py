import streamlit as st

CUSTOM_CSS = """
<style>
/* Main container layout and font styling */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1380px;
}

/* Metric card styling */
div[data-testid="stMetric"] {
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 10px 16px;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

div[data-testid="stMetric"] label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #4B5563;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
    font-weight: 700;
    color: #111827;
}

/* Section headers */
h2 {
    border-bottom: 2px solid #F3F4F6;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
    font-size: 1.35rem;
}

h3 {
    font-size: 1.15rem;
    margin-top: 1rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FAFAFA;
    border-right: 1px solid #E5E7EB;
}

/* Table and dataframe borders */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""

def inject_styles():
    """Injects custom application stylesheet once."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
