"""
pages/2_Output.py
-----------------
Output and validation results page displaying per-method top-10 rankings,
dynamic consensus panel synthesis, LOOCV diagnostic evaluation, and explainability plots.
"""

import io
import pandas as pd
import streamlit as st

from app.config import (
    METHOD_NAMES,
    METHOD_INFO,
    FALLBACK_FILTER_NAME,
    SS_DATA_SOURCE,
    SS_ANALYSIS_READY,
    SS_RESULTS,
    SS_CONSENSUS,
    SS_VALIDATION
)
from app.core.plots import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_gene_boxplots,
    plot_clustered_heatmap,
    plot_shap_summary
)
from app.ui.styles import apply_custom_styles
from app.ui.nav import render_top_nav
from app.ui.components import render_header, render_disclaimer, render_fallback_warning

apply_custom_styles()
render_top_nav("Output")

# Guard: Check if analysis is ready
if not st.session_state.get(SS_ANALYSIS_READY, False) or SS_RESULTS not in st.session_state:
    st.info("ℹ️ **No analysis results to display yet.**")
    st.write("Please visit the Setup page, select your data source and methods, and click **Run Feature Selection & Validation**.")
    if st.button("👈 Go to Setup & Run Page", type="primary"):
        st.switch_page("pages/1_Landing.py")
    st.stop()

# Retrieve analysis state
results = st.session_state[SS_RESULTS]
consensus = st.session_state[SS_CONSENSUS]
validation = st.session_state[SS_VALIDATION]
id_to_symbol = st.session_state["id_to_symbol"]
X_final = st.session_state["X_final"]
y_raw = st.session_state["y_raw"]
le_classes = st.session_state["le_classes"]
data_source = st.session_state.get(SS_DATA_SOURCE, "sample")
is_fallback_used = st.session_state.get("is_fallback_used", False)

source_names = {
    "sample": "Preloaded GSE68719 Discovery DEGs",
    "upload_deg": "Custom Uploaded Candidate DEG List",
    "upload_raw": f"Raw Counts Filtered via {FALLBACK_FILTER_NAME}"
}
current_source_label = source_names.get(data_source, "GSE68719")

render_header(
    title="Pipeline Results & Cross-Validation",
    subtitle=f"Results for consensus biomarker panel evaluated across {len(validation.y_true)} postmortem samples."
)

if is_fallback_used:
    render_fallback_warning()

# -------------------------------------------------------------
# Section 1: Per-Method Top 10 Rankings
# -------------------------------------------------------------
st.markdown("### 1. Individual Method Feature Rankings")
st.write("Top candidate genes selected and ranked by each executed machine learning algorithm:")

ordered_methods = [m for m in METHOD_NAMES if m in results]
tab_list = [METHOD_INFO[m]["title"] for m in ordered_methods]
tabs = st.tabs(tab_list)

method_export_frames = []

for idx, m_name in enumerate(ordered_methods):
    res = results[m_name]
    with tabs[idx]:
        st.caption(f"**Ranking Metric:** {res.metric_name} | **Total Evaluated Candidates:** {res.n_available}")

        df_rank = pd.DataFrame({
            "Rank": list(range(1, len(res.top10_ids) + 1)),
            "Gene Symbol": res.top10_symbols,
            "Entrez ID": res.top10_ids,
            "Score": [res.scores.get(gid, 0.0) for gid in res.top10_ids]
        })
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

        df_export = df_rank.copy()
        df_export["Method"] = m_name
        method_export_frames.append(df_export)

# CSV Export for method tables
if method_export_frames:
    all_methods_df = pd.concat(method_export_frames, ignore_index=True)
    csv_methods = all_methods_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export All Method Rankings (CSV)",
        data=csv_methods,
        file_name="pd_feature_selection_methods_top10.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# Section 2: Consensus Panel Synthesis
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 2. Consensus Panel Synthesis")

# Rule trigger callout
st.markdown(f"""
<div class='notice-banner'>
    <strong>Consensus Rule Triggered:</strong> <code>{consensus.rule}</code><br>
    {consensus.description}
</div>
""", unsafe_allow_html=True)

col_c1, col_c2 = st.columns([1.5, 2.5])
with col_c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-card-title'>Consensus Biomarkers</div>
        <div class='metric-card-value'>{len(consensus.panel)} Genes</div>
        <div style='font-size: 0.8rem; color: #64748B; margin-top: 0.3rem;'>
            Symbols: <strong>{", ".join(consensus.symbols)}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    consensus_df = pd.DataFrame({
        "Gene Symbol": consensus.symbols,
        "Entrez ID": consensus.panel,
        "Algorithm Agreement": [f"{consensus.all_votes.get(gid, 1)} of {len(results)}" for gid in consensus.panel]
    })
    st.dataframe(consensus_df, use_container_width=True, hide_index=True)

    csv_consensus = consensus_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Consensus Panel (CSV)",
        data=csv_consensus,
        file_name="pd_consensus_biomarker_panel.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# Section 3: Leave-One-Out Cross-Validation (LOOCV)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 3. Cross-Validation & Diagnostic Evaluation")
st.write(f"Leave-One-Out Cross-Validation (LOOCV) with a balanced Random Forest (500 trees) on the {len(consensus.panel)}-gene panel:")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-card-title'>LOOCV Accuracy</div>
        <div class='metric-card-value'>{validation.accuracy:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-card-title'>ROC-AUC Score</div>
        <div class='metric-card-value'>{validation.auc_score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    n_samples = len(validation.y_true)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-card-title'>Cross-Validated Samples</div>
        <div class='metric-card-value'>{n_samples}</div>
    </div>
    """, unsafe_allow_html=True)

# Diagnostic Plots Grid
col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("#### A. Receiver Operating Characteristic (ROC)")
    st.caption("Diagonal gray line indicates chance (AUC = 0.50). High curves indicate strong sensitivity and specificity trade-off.")
    try:
        fig_roc = plot_roc_curve(validation.fpr, validation.tpr, validation.auc_score, len(consensus.panel))
        st.pyplot(fig_roc)
    except Exception as e:
        st.error("Error rendering ROC curve")
        with st.expander("Details"):
            st.exception(e)

with col_p2:
    st.markdown("#### B. Confusion Matrix")
    st.caption("Discrete sample classification distribution under LOOCV hold-out inference.")
    try:
        fig_cm = plot_confusion_matrix(validation.cm, validation.accuracy, le_classes)
        st.pyplot(fig_cm)
    except Exception as e:
        st.error("Error rendering Confusion Matrix")
        with st.expander("Details"):
            st.exception(e)

# -------------------------------------------------------------
# Section 4: Expression Heatmap & Per-Gene Boxplots
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 4. Biomarker Expression & Clustering")

st.markdown("#### A. Hierarchical Clustered Heatmap")
st.caption("Z-scored log2(CPM + 1) normalized expression across samples. Blue header = NO_PD, Orange header = PD.")
try:
    fig_hm = plot_clustered_heatmap(X_final, y_raw, id_to_symbol, le_classes)
    st.pyplot(fig_hm)
except Exception as e:
    st.error("Error rendering Clustered Heatmap")
    with st.expander("Details"):
        st.exception(e)

st.markdown("#### B. Per-Gene Expression Boxplots + Stripplots")
st.caption("Individual sample expression distributions for each biomarker in the consensus panel.")
try:
    fig_bp = plot_gene_boxplots(X_final, y_raw, id_to_symbol)
    st.pyplot(fig_bp)
except Exception as e:
    st.error("Error rendering Gene Boxplots")
    with st.expander("Details"):
        st.exception(e)

# -------------------------------------------------------------
# Section 5: SHAP Interpretability
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 5. Explainable AI: SHAP Feature Attribution")
st.caption(
    "SHAP summary beeswarm plot ranking the consensus biomarkers by their impact on model decisions. "
    "Points represent individual patients; red represents high relative expression, blue represents low."
)
try:
    fig_shap = plot_shap_summary(validation.rf_fitted, X_final, id_to_symbol)
    st.pyplot(fig_shap)
except Exception as e:
    st.error("Error rendering SHAP summary plot")
    with st.expander("Details"):
        st.exception(e)

# Metrics Export
metrics_summary = pd.DataFrame({
    "Metric": ["LOOCV Accuracy", "LOOCV AUC", "Consensus Panel Size", "Total Samples"],
    "Value": [f"{validation.accuracy:.4f}", f"{validation.auc_score:.4f}", len(consensus.panel), len(validation.y_true)]
})
csv_metrics = metrics_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Export Validation Metrics (CSV)",
    data=csv_metrics,
    file_name="pd_pipeline_validation_metrics.csv",
    mime="text/csv"
)

render_disclaimer()
