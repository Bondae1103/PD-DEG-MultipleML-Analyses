"""
pages/1_Landing.py
------------------
Landing and Pipeline Configuration page.
Allows choosing sample data or uploading custom DEG/counts, selecting feature selection methods,
and executing the pipeline with per-method progress indication.
"""

import os
import io
import hashlib
import pandas as pd
import streamlit as st

from app.config import (
    SAMPLE_DEG_PATH,
    COUNTS_MATRIX_PATH,
    SAMPLE_METADATA_PATH,
    METHOD_NAMES,
    METHOD_INFO,
    RANDOM_SEED,
    FALLBACK_FILTER_NAME,
    SS_DATA_SOURCE,
    SS_DEG_HASH,
    SS_SELECTED_METHODS,
    SS_ANALYSIS_READY,
    SS_RESULTS,
    SS_CONSENSUS,
    SS_VALIDATION,
    SS_CUSTOM_DEG_DF,
    SS_RAW_COUNTS_DF,
    SS_RAW_META_DF
)
from app.core.io import load_deg_list, load_optional_raw_data, UserInputError
from app.core.preprocess import clean_deg_list, subset_and_normalize, build_X_y
from app.core.deg_fallback import run_fallback_deg_filter
from app.core.methods import run_selected_methods
from app.core.consensus import build_consensus
from app.core.validate import run_loocv_validation
from app.ui.styles import apply_custom_styles
from app.ui.nav import render_top_nav
from app.ui.components import render_header, render_provenance_badge, render_disclaimer, render_fallback_warning

# Page setup
apply_custom_styles()
render_top_nav("Landing")

render_header(
    title="Parkinson's Disease DEG Feature-Selection & Validation Pipeline",
    subtitle="A portfolio demonstration of a 5-method feature-selection + consensus + leave-one-out cross-validation pipeline for Parkinson's biomarker discovery (GEO GSE68719)."
)

# Initialize Session State
if SS_DATA_SOURCE not in st.session_state:
    st.session_state[SS_DATA_SOURCE] = "sample"
if SS_SELECTED_METHODS not in st.session_state:
    st.session_state[SS_SELECTED_METHODS] = list(METHOD_NAMES)
if SS_ANALYSIS_READY not in st.session_state:
    st.session_state[SS_ANALYSIS_READY] = False
if SS_DEG_HASH not in st.session_state:
    st.session_state[SS_DEG_HASH] = ""

# -------------------------------------------------------------
# Section 1: Data Source Selection
# -------------------------------------------------------------
st.markdown("### 1. Select Input Data Source")

source_choice = st.radio(
    "Choose how to supply differentially expressed genes (DEGs):",
    options=[
        "Use sample data (GSE68719 DEG list)",
        "Upload my own DEG list",
        "I don't have a DEG list — filter from raw counts"
    ],
    index=0 if st.session_state[SS_DATA_SOURCE] == "sample" else (1 if st.session_state[SS_DATA_SOURCE] == "upload_deg" else 2),
    key="radio_data_source"
)

# Detect source changes and reset downstream analysis
current_source_key = "sample" if "sample" in source_choice else ("upload_deg" if "Upload" in source_choice else "upload_raw")
if current_source_key != st.session_state[SS_DATA_SOURCE]:
    st.session_state[SS_DATA_SOURCE] = current_source_key
    st.session_state[SS_ANALYSIS_READY] = False
    st.session_state[SS_RESULTS] = None
    st.session_state[SS_CONSENSUS] = None
    st.session_state[SS_VALIDATION] = None

deg_df = None
counts_matrix = None
sample_metadata = None
is_fallback_used = False

# Path A: Sample Data
if st.session_state[SS_DATA_SOURCE] == "sample":
    render_provenance_badge("Bundled GSE68719 Ground Truth")
    st.caption(
        "ℹ️ **Dataset Note:** This is the **actual DEG list** from the GSE68719 postmortem prefrontal cortex study "
        "(430 significant genes, filtered from 21,537 via `limma-voom`). Uploading a custom DEG list "
        "**derived from GSE68719** is the validated way to run alternative feature selection against the bundled counts matrix."
    )
    deg_df = pd.read_csv(SAMPLE_DEG_PATH)
    counts_matrix = pd.read_parquet(COUNTS_MATRIX_PATH)
    sample_metadata = pd.read_csv(SAMPLE_METADATA_PATH)

    with st.expander("Preview sample DEG list (first 5 rows)", expanded=False):
        st.dataframe(deg_df.head(), use_container_width=True)

# Path B: Custom DEG List Upload
elif st.session_state[SS_DATA_SOURCE] == "upload_deg":
    st.markdown(
        "Upload a custom candidate DEG list in **CSV** or **Excel** format. "
        "The table must contain columns: `ENTREZID`, `SYMBOL`, and `GENENAME`."
    )

    col_up, col_dl = st.columns([3, 1.2])
    with col_up:
        uploaded_deg_file = st.file_uploader(
            "Choose DEG file (.csv, .xlsx)",
            type=["csv", "xlsx", "xls"],
            key="file_deg_upload"
        )
    with col_dl:
        sample_deg_bytes = open(SAMPLE_DEG_PATH, "rb").read()
        st.download_button(
            "📥 Download Sample DEG Template",
            data=sample_deg_bytes,
            file_name="sample_deg_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_deg_file is not None:
        try:
            deg_df = load_deg_list(uploaded_deg_file)
            st.success(f"Loaded custom DEG table: {len(deg_df):,} genes found.")
            st.session_state[SS_CUSTOM_DEG_DF] = deg_df
            counts_matrix = pd.read_parquet(COUNTS_MATRIX_PATH)
            sample_metadata = pd.read_csv(SAMPLE_METADATA_PATH)
            with st.expander("Preview uploaded DEG list", expanded=False):
                st.dataframe(deg_df.head(), use_container_width=True)
        except UserInputError as e:
            st.error(f"❌ Input Validation Error: {str(e)}")
            deg_df = None

# Path C: Fallback Filter from Raw Counts
else:
    render_fallback_warning()
    st.markdown(
        "Upload raw RNA-Seq read counts and matching sample metadata. "
        "The pipeline will execute an in-browser two-sample Welch's t-test with Benjamini-Hochberg FDR correction."
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        up_counts = st.file_uploader("Upload Raw Counts Matrix (.parquet, .csv, .xlsx)", type=["parquet", "csv", "xlsx"], key="up_raw_counts")
    with col_c2:
        up_meta = st.file_uploader("Upload Sample Metadata (.csv, .xlsx)", type=["csv", "xlsx"], key="up_raw_meta")

    if up_counts is not None and up_meta is not None:
        try:
            counts_matrix, sample_metadata = load_optional_raw_data(up_counts, up_meta)
            st.success(f"Loaded counts ({counts_matrix.shape[0]:,} genes, {counts_matrix.shape[1]-1} samples) and metadata ({len(sample_metadata)} samples).")

            with st.spinner(f"Running {FALLBACK_FILTER_NAME}..."):
                sig_deg, full_deg, dt = run_fallback_deg_filter(counts_matrix, sample_metadata)
            st.info(f"Fallback filter identified {len(sig_deg)} significant DEGs in {dt:.3f} s.")
            deg_df = sig_deg
            is_fallback_used = True
        except UserInputError as e:
            st.error(f"❌ Input Validation Error: {str(e)}")
            deg_df = None

# -------------------------------------------------------------
# Section 2: Method Selection
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 2. Select Machine Learning Feature Selection Methods")
st.write("Select any combination of the 5 algorithms to execute in parallel:")

method_cols = st.columns(5)
selected_methods = []

for idx, m_name in enumerate(METHOD_NAMES):
    info = METHOD_INFO[m_name]
    with method_cols[idx]:
        is_checked = st.checkbox(
            label=f"**{info['title']}**",
            value=(m_name in st.session_state[SS_SELECTED_METHODS]),
            key=f"chk_method_{m_name}",
            help=info["description"]
        )
        st.caption(f"_{info['metric']}_")
        if is_checked:
            selected_methods.append(m_name)

st.session_state[SS_SELECTED_METHODS] = selected_methods

# Validation & Execution readiness
can_run = (deg_df is not None and len(selected_methods) > 0)
if len(selected_methods) == 0:
    st.warning("⚠️ Please select at least one feature selection method.")

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Section 3: Run Pipeline Action
# -------------------------------------------------------------
if st.button("🚀 Run Feature Selection & Validation", type="primary", disabled=not can_run, use_container_width=True):
    try:
        progress_text = st.empty()
        bar = st.progress(0)

        # 1. Clean DEG list
        progress_text.text("Step 1/5: Cleaning candidate DEG list...")
        bar.progress(10)
        deg_clean = clean_deg_list(deg_df)

        # 2. Subset and normalize
        progress_text.text("Step 2/5: Normalizing CPM from full library size...")
        bar.progress(25)
        log_cpm, id_to_symbol = subset_and_normalize(deg_clean, counts_matrix)

        # 3. Build X, y
        progress_text.text("Step 3/5: Building ML feature matrix and encoding labels...")
        bar.progress(35)
        X, y, y_raw, le = build_X_y(log_cpm, sample_metadata)

        # 4. Run selected methods with individual progress
        total_methods = len(selected_methods)
        def progress_cb(method_name, current, total):
            pct = 35 + int((current / total) * 40)
            bar.progress(pct)
            if method_name == "Boruta":
                progress_text.text(f"Step 4/5: Running Boruta (all-relevant selection with 200 iterations — may take ~20-60s)...")
            else:
                progress_text.text(f"Step 4/5: Running {method_name} ({current}/{total})...")

        results = run_selected_methods(
            X, y, id_to_symbol,
            selected_method_names=selected_methods,
            seed=RANDOM_SEED,
            progress_callback=progress_cb
        )

        # 5. Build Consensus Panel
        progress_text.text("Step 5/5: Computing consensus panel & running LOOCV cross-validation...")
        bar.progress(85)
        consensus = build_consensus(results, id_to_symbol, selected_methods_user=selected_methods)

        # 6. Validate with LOOCV
        validation = run_loocv_validation(X, y, consensus.panel, seed=RANDOM_SEED)

        bar.progress(100)
        progress_text.empty()

        # Save to session state
        st.session_state[SS_RESULTS] = results
        st.session_state[SS_CONSENSUS] = consensus
        st.session_state[SS_VALIDATION] = validation
        st.session_state["id_to_symbol"] = id_to_symbol
        st.session_state["X_final"] = X[validation.panel_features]
        st.session_state["y_raw"] = y_raw
        st.session_state["le_classes"] = list(le.classes_)
        st.session_state["is_fallback_used"] = is_fallback_used
        st.session_state[SS_ANALYSIS_READY] = True

        st.success("✅ Analysis completed successfully! Redirecting to Output page...")
        st.switch_page("pages/2_Output.py")

    except UserInputError as e:
        st.error(f"❌ Input Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Pipeline Execution Error: {str(e)}")
        with st.expander("Technical details"):
            st.exception(e)

render_disclaimer()
