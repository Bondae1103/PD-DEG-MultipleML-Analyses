import streamlit as st

# Configure page settings strictly once as the first Streamlit call
st.set_page_config(
    page_title="PD Transcriptomics Biomarker Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.ui.main import run

if __name__ == "__main__":
    run()
