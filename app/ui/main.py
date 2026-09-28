import io
import os
import json
import hashlib
import traceback
from collections import Counter
import numpy as np
import pandas as pd
import streamlit as st

from app.config import (
    MODEL_PATH, FEATURES_PATH, PREPROC_PATH, DEG_PATH, METADATA_PATH,
    COHORTS, CLASS_MAP, POSITIVE_LABEL_DISPLAY, NEGATIVE_LABEL_DISPLAY,
    DEFAULT_PVAL_THRESH, DEFAULT_LOG2FC_THRESH, TOP_N_SHAP,
    SS_DATA_SOURCE, SS_ACTIVE_COHORT, SS_UPLOAD_HASH, SS_SELECTED_SAMPLE,
    SS_P_THRESH, SS_LFC_THRESH, SS_USE_PADJ
)
from app.core import UserInputError
from app.core.io import load_matrix, validate_upload
from app.core.preprocess import align_to_features
from app.core.model import load_model, predict
from app.core.deg import load_deg, filter_deg, compute_deg_fallback
from app.core.explain import get_explainer, explain_sample, waterfall_frame
from app.core.plots import volcano_figure, waterfall_figure, prob_figure
from app.ui.styles import inject_styles
from app.ui.components import render_header, render_disclaimer, render_training_notice

# -------------------------------------------------------------
# Cached Resources
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def get_explainer_cached(_model):
    return get_explainer(_model)

@st.cache_resource(show_spinner=False)
def get_deg_table():
    return load_deg(DEG_PATH)

@st.cache_resource(show_spinner=False)
def get_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def get_preproc():
    with open(PREPROC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def get_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def get_methods_data():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "methods_data.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_cohort_data(cohort_name: str):
    info = COHORTS[cohort_name]
    matrix = pd.read_parquet(info["matrix_path"])
    meta = pd.read_csv(info["meta_path"])
    return matrix, meta

# -------------------------------------------------------------
# Main Application
# -------------------------------------------------------------
def run():
    inject_styles()

    model = get_model()
    explainer = get_explainer_cached(model)
    deg_base = get_deg_table()
    features = get_features()
    preproc = get_preproc()
    meta = get_metadata()
    methods_data = get_methods_data()

    # Initialize state
    st.session_state.setdefault(SS_DATA_SOURCE, "Sample data (GSE68719)")
    st.session_state.setdefault(SS_ACTIVE_COHORT, "GSE68719")
    st.session_state.setdefault(SS_SELECTED_SAMPLE, None)
    st.session_state.setdefault(SS_P_THRESH, DEFAULT_PVAL_THRESH)
    st.session_state.setdefault(SS_LFC_THRESH, DEFAULT_LOG2FC_THRESH)
    st.session_state.setdefault(SS_USE_PADJ, True)

    render_header()
    render_disclaimer(meta)

    # ---------------------------------------------------------
    # SECTION 1: Pipeline Architecture & Purpose
    # ---------------------------------------------------------
    st.header("1. Pipeline Architecture & Project Overview")
    
    st.markdown("""
    This project discovers, validates, and interprets a compact transcriptomic biomarker panel for **Parkinson's Disease (PD)**.
    It combines classical RNA-Seq differential expression modeling with **five independent machine learning selection paradigms** 
    to isolate high-confidence molecular drivers from human postmortem prefrontal cortex tissue.
    """)

    arch_col1, arch_col2, arch_col3, arch_col4, arch_col5 = st.columns(5)
    with arch_col1:
        st.markdown("**Step 1: DEG Discovery**")
        st.caption("RNA-Seq empirical Bayes linear modeling (`limma-voom`) identifies significant DEGs ($p_{\\text{adj}} < 0.05, |\\text{log}_2\\text{FC}| > 1$).")
    with arch_col2:
        st.markdown("**Step 2: Quality Cleaning**")
        st.caption("Removal of uncharacterized `LOC*` loci, pseudogenes, microRNAs, and snRNAs to isolate biologically actionable transcripts.")
    with arch_col3:
        st.markdown("**Step 3: 5-Method Selection**")
        st.caption("Parallel feature selection across LASSO, Boruta, SVM-RFE, XGBoost, and Mutual Information.")
    with arch_col4:
        st.markdown("**Step 4: Consensus Panel**")
        st.caption("Dual-algorithm consensus (LASSO $\\cup$ Boruta = 18 genes) minimizes individual algorithm selection bias.")
    with arch_col5:
        st.markdown("**Step 5: Validation & SHAP**")
        st.caption("Leave-One-Out Cross-Validation (LOOCV: 86.1% Accuracy, 0.912 AUC) and exact SHAP tree attributions.")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2: Multi-Method Feature Selection & DEG Analysis Hub
    # ---------------------------------------------------------
    st.header("2. Multi-Method Feature Selection & DEG Analysis Hub")
    st.markdown(
        "Upload a candidate DEG list or explore with preloaded sample data. "
        "Select any combination of the **5 machine learning algorithms** to dynamically evaluate their selections and generate consensus panels."
    )

    data_source_mode = st.radio(
        "Select Data Source:",
        options=["Sample data (GSE68719 Discovery Cohort)", "Upload custom DEG list"],
        horizontal=True
    )

    active_gene_pool = list(methods_data.get("methods", {}).get("LASSO", []))
    active_deg_df = deg_base.copy()

    if data_source_mode == "Upload custom DEG list":
        upload_col1, upload_col2 = st.columns([3, 1])
        with upload_col1:
            deg_file = st.file_uploader(
                "Upload DEG List (CSV / TSV / Excel / TXT)",
                type=["csv", "tsv", "txt", "xlsx"],
                help="File with gene symbols (column 'gene' or 'SYMBOL') and optionally log2fc, pvalue, padj."
            )
        with upload_col2:
            st.write("")
            st.write("")
            sample_deg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "sample_deg_list.csv")
            if os.path.exists(sample_deg_path):
                with open(sample_deg_path, "rb") as f:
                    st.download_button(
                        "📥 Download Sample DEG CSV",
                        data=f.read(),
                        file_name="sample_deg_list.csv",
                        mime="text/csv"
                    )

        if deg_file is not None:
            try:
                if deg_file.name.endswith(".xlsx"):
                    user_deg = pd.read_excel(deg_file)
                else:
                    user_deg = pd.read_csv(deg_file)
                st.success(f"✅ Loaded {len(user_deg):,} uploaded DEGs.")
                # Look for gene column
                gene_col = next((c for c in user_deg.columns if c.lower() in ["gene", "symbol", "geneid"]), user_deg.columns[0])
                user_genes = set(user_deg[gene_col].astype(str).str.strip().str.upper())
                active_deg_df = user_deg.rename(columns={gene_col: "gene"})
            except Exception as e:
                st.error(f"Error parsing uploaded DEG file: {e}")

    st.subheader("Select Machine Learning Feature Selection Methods")
    st.caption("Choose any number of algorithms to inspect their candidates and compute consensus statistics:")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        use_lasso = st.checkbox("LASSO (L1 Logistic)", value=True, help="L1-penalized sparse regression with 5-fold CV")
    with mc2:
        use_boruta = st.checkbox("Boruta (Random Forest)", value=True, help="Shadow-feature confirmed Random Forest selection")
    with mc3:
        use_svm = st.checkbox("SVM-RFE", value=False, help="Support Vector Machine Recursive Feature Elimination")
    with mc4:
        use_xgb = st.checkbox("XGBoost Importance", value=False, help="Gradient Boosted Tree Gini feature importance ranking")
    with mc5:
        use_mi = st.checkbox("Mutual Information", value=False, help="Information-theoretic entropy dependency scoring")

    # Determine active method lists
    selected_methods = {}
    repo_methods = methods_data.get("methods", {})
    if use_lasso and "LASSO" in repo_methods:
        selected_methods["LASSO"] = repo_methods["LASSO"]
    if use_boruta and "Boruta" in repo_methods:
        selected_methods["Boruta"] = repo_methods["Boruta"]
    if use_svm and "SVM_RFE" in repo_methods:
        selected_methods["SVM_RFE"] = repo_methods["SVM_RFE"]
    if use_xgb and "XGBoost" in repo_methods:
        selected_methods["XGBoost"] = repo_methods["XGBoost"]
    if use_mi and "MutualInfo" in repo_methods:
        selected_methods["MutualInfo"] = repo_methods["MutualInfo"]

    if not selected_methods:
        st.warning("⚠️ Please select at least one feature selection method above to view consensus results.")
    else:
        # Tally vote counts
        all_selected_genes = [g for genes in selected_methods.values() for g in genes]
        vote_counter = Counter(all_selected_genes)
        n_methods = len(selected_methods)

        consensus_mode = st.radio(
            "Consensus Strategy:",
            options=["Union (Any selected method)", "Majority Vote (>= 50% of selected methods)", "Strict Intersect (All selected methods)"],
            horizontal=True
        )

        if consensus_mode == "Strict Intersect (All selected methods)":
            consensus_panel = [g for g, count in vote_counter.items() if count == n_methods]
        elif consensus_mode == "Majority Vote (>= 50% of selected methods)":
            min_votes = max(1, (n_methods + 1) // 2)
            consensus_panel = [g for g, count in vote_counter.items() if count >= min_votes]
        else:
            consensus_panel = list(vote_counter.keys())

        # Consensus Summary Cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Selected Algorithms", f"{n_methods} / 5")
        sc2.metric("Total Unique Candidates", f"{len(vote_counter)}")
        sc3.metric("Consensus Panel Size", f"{len(consensus_panel)} genes")
        match_overlap = [g for g, c in vote_counter.items() if c >= 2]
        sc4.metric("Multi-Algorithm Overlap", f"{len(match_overlap)} genes")

        # Build dynamic voting table
        vote_records = []
        gene_info_dict = methods_data.get("gene_info", {})
        for g, votes in vote_counter.most_common():
            info = gene_info_dict.get(g, {})
            row = {
                "Gene Symbol": g,
                "Votes": f"{votes} / {n_methods}",
                "In Consensus Panel": "✅ Yes" if g in consensus_panel else "No",
                "LASSO": "✓" if use_lasso and g in selected_methods.get("LASSO", []) else "—",
                "Boruta": "✓" if use_boruta and g in selected_methods.get("Boruta", []) else "—",
                "SVM-RFE": "✓" if use_svm and g in selected_methods.get("SVM_RFE", []) else "—",
                "XGBoost": "✓" if use_xgb and g in selected_methods.get("XGBoost", []) else "—",
                "Mutual Info": "✓" if use_mi and g in selected_methods.get("MutualInfo", []) else "—",
                "log2FC (PD vs Ctrl)": f"{info.get('log2fc', 0.0):+.2f}",
                "FDR padj": f"{info.get('padj', 1.0):.2e}"
            }
            vote_records.append(row)

        vote_df = pd.DataFrame(vote_records)
        st.dataframe(vote_df, use_container_width=True)

        # Download consensus panel
        csv_panel = pd.DataFrame({"Gene Symbol": consensus_panel}).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Consensus Panel (CSV)",
            data=csv_panel,
            file_name="consensus_gene_panel.csv",
            mime="text/csv"
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 3: Analysis Plots Showcase Gallery
    # ---------------------------------------------------------
    st.header("3. Analysis Plots Showcase (Research Gallery)")
    st.markdown(
        "Inspect the published figures and diagnostic plots generated by this transcriptomics machine learning pipeline. "
        "Each tab presents a key figure with biological and statistical context."
    )

    image_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "images")

    tab_roc, tab_heatmap, tab_boxplots, tab_shap, tab_volcano = st.tabs([
        "📈 LOOCV ROC Curve",
        "🧬 Clustered Heatmap",
        "📊 Per-Gene Boxplots",
        "🐝 Global SHAP Summary",
        "🌋 Transcriptome Volcano"
    ])

    with tab_roc:
        st.subheader("Model Discrimination: Leave-One-Out Cross-Validation (LOOCV)")
        roc_img_path = os.path.join(image_dir, "roc_loocv.png")
        if os.path.exists(roc_img_path):
            r_col1, r_col2 = st.columns([3, 2])
            with r_col1:
                st.image(roc_img_path, caption="LOOCV ROC Curve for the 18-gene consensus panel", use_container_width=True)
            with r_col2:
                st.markdown("""
                **Key Performance Indicators:**
                - **ROC-AUC:** `0.912` (Random Forest) / `0.939` (XGBoost)
                - **LOOCV Accuracy:** `86.1%` (62 / 72 correct classifications)
                - **Sensitivity (PD Recall):** `71.4%` (20 / 28)
                - **Specificity (Control Recall):** `95.5%` (42 / 44)
                
                **Methodological Rationale:**
                With $n=72$ biospecimens, single train/test splits are prone to selection variance. 
                Leave-One-Out Cross-Validation trains on $n-1$ samples and tests on the single held-out subject across 72 iterations, 
                providing the most rigorous performance estimate for small cohorts without data leakage.
                """)

    with tab_heatmap:
        st.subheader("Unsupervised Hierarchical Clustering of the 18-Gene Biomarker Panel")
        hm_img_path = os.path.join(image_dir, "panel_heatmap.png")
        if os.path.exists(hm_img_path):
            h_col1, h_col2 = st.columns([3, 2])
            with h_col1:
                st.image(hm_img_path, caption="Clustered heatmap across 72 patients (Z-scored expression)", use_container_width=True)
            with h_col2:
                st.markdown("""
                **Orthogonal Unsupervised Validation:**
                - **Sample Clustering:** Without receiving disease labels, unsupervised dendrogram clustering largely separates Parkinson's disease (orange) from Controls (blue).
                - **Co-expression Modules:** 
                  - *Upregulated Module in PD:* `ADAM33`, `DNAJB1`, `RIPOR3`, `SIX5`, `MT1A`, `SMTN`, `SERPINF2`, `CSF1`, `SYTL4`, `PRELP`, `HSPB1`.
                  - *Downregulated Module in PD:* `CELF2-AS1`, `C1QL3`, `PRAP1`, `CISTR`.
                - Demonstrates that the feature selection converged on coordinated regulatory programs rather than isolated noisy genes.
                """)

    with tab_boxplots:
        st.subheader("Grouped Expression Distributions (PD vs. Control)")
        bp_img_path = os.path.join(image_dir, "panel_boxplots.png")
        if os.path.exists(bp_img_path):
            st.image(bp_img_path, caption="Normalized log2(CPM+1) expression across individual panel genes", use_container_width=True)
            st.caption("Boxplots depict median and interquartile range; individual overlaid dots represent distinct brain tissue samples.")

    with tab_shap:
        st.subheader("Global Feature Importance: SHAP Beeswarm Summary Plot")
        shap_img_path = os.path.join(image_dir, "shap_summary.png")
        if os.path.exists(shap_img_path):
            s_col1, s_col2 = st.columns([3, 2])
            with s_col1:
                st.image(shap_img_path, caption="SHAP summary plot ranking panel biomarkers by mean absolute impact", use_container_width=True)
            with s_col2:
                st.markdown("""
                **Key Interpretability Insights:**
                - **Top Biomarker (`ADAM33`):** Selected by both LASSO and Boruta; demonstrates the highest magnitude impact in driving predictions toward Parkinson's disease.
                - **Proteostasis & Heat Shock (`DNAJB1`, `HSPB1`):** Central chaperones involved in protein quality control and $\\alpha$-synuclein aggregation response.
                - **Directional Consistency:** High expression of `ADAM33`, `RIPOR3`, and `DNAJB1` (red points) strictly increases the log-odds of a Parkinson's diagnosis.
                """)

    with tab_volcano:
        st.subheader("Transcriptome-Wide Differential Expression")
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            st.markdown("""
            **Filters & Settings:**
            Adjust significance and fold-change boundaries to interact with the full 21,537-gene differential expression landscape.
            """)
            p_val_slider = st.select_slider("FDR Significance Threshold:", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.05)
            lfc_slider = st.slider("Minimum |log2FC| Threshold:", 0.0, 3.0, 1.0, 0.1)

            filtered_deg = filter_deg(deg_base, p_col="padj", p_thresh=p_val_slider, lfc_thresh=lfc_slider)
            st.metric("Significantly Upregulated", f"{(filtered_deg['category']=='Up').sum():,}")
            st.metric("Significantly Downregulated", f"{(filtered_deg['category']=='Down').sum():,}")

        with v_col2:
            fig_v = volcano_figure(filtered_deg, p_thresh=p_val_slider, lfc_thresh=lfc_slider, model_features=features)
            st.plotly_chart(fig_v, use_container_width=True, key="gallery_volcano")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 4: Biomarker Model Inference & Patient Decision Audit
    # ---------------------------------------------------------
    st.header("4. Biomarker Inference & Patient Decision Audit (SHAP)")
    st.markdown(
        "Evaluate the validated XGBoost biomarker classifier across cohorts, "
        "and select individual patients to inspect exact SHAP waterfall attributions."
    )

    cohort_selector = st.selectbox(
        "Evaluate Cohort:",
        options=list(COHORTS.keys()),
        format_func=lambda k: COHORTS[k]["display_name"],
        key="eval_cohort_select"
    )

    curr_matrix, curr_meta = load_cohort_data(cohort_selector)
    aligned_df, _ = align_to_features(curr_matrix, features, fill_values=preproc.get("feature_medians"))
    pred_df = predict(model, aligned_df, CLASS_MAP, meta_df=curr_meta)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Parkinson's (PD)", f"{(pred_df['predicted_label']=='PD').sum()}")
    c2.metric("Predicted Control (NO_PD)", f"{(pred_df['predicted_label']=='NO_PD').sum()}")
    if "true_label" in pred_df.columns:
        cohort_acc = (pred_df["predicted_label"] == pred_df["true_label"]).mean()
        is_held_out = pred_df.get("held_out", pd.Series([False]*len(pred_df))).all()
        c3.metric("Held-out Validation Accuracy" if is_held_out else "Cohort Accuracy", f"{cohort_acc*100:.1f}%")

    if COHORTS[cohort_selector]["used_in_training"]:
        render_training_notice()

    # Patient Table & Dropdown Selection
    display_df = pred_df.sort_values("confidence", ascending=False).reset_index(drop=True)
    sample_ids = display_df["sample_id"].tolist()

    sel_sample = st.selectbox("Select Patient to Audit with SHAP:", options=sample_ids)

    # Compute SHAP for selected patient
    sample_row = aligned_df.loc[sel_sample]
    sample_info = pred_df[pred_df["sample_id"] == sel_sample].iloc[0]
    
    shap_res = explain_sample(explainer, model, sample_row, sample_id=sel_sample)
    wf_frame = waterfall_frame(shap_res, top_n=TOP_N_SHAP)

    fig_wf = waterfall_figure(
        wf_frame,
        base_value=shap_res.base_value,
        final_margin=shap_res.margin,
        sample_id=sel_sample,
        predicted_label=sample_info["predicted_label"],
        probability=shap_res.probability
    )
    st.plotly_chart(fig_wf, use_container_width=True, key=f"waterfall_plot_{sel_sample}")

    top_feature = wf_frame.iloc[0]
    direction_desc = POSITIVE_LABEL_DISPLAY if top_feature["shap"] > 0 else NEGATIVE_LABEL_DISPLAY
    st.info(
        f"💡 **Top Biomarker Driver:** For patient **{sel_sample}**, the most impactful gene was "
        f"**{top_feature['gene']}** ({top_feature['feature_value']:.2f}), contributing **{top_feature['shap']:+.3f} log-odds** toward **{direction_desc}**."
    )
