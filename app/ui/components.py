import streamlit as st
from typing import Dict, Any, List, Optional
from app.core.io import ValidationReport

def render_header():
    """Renders application header and quick start instructions."""
    st.title("🧬 Parkinson's Disease Transcriptomics Biomarker Explorer")
    st.markdown(
        "Explore differentially expressed genes in Parkinson's disease postmortem cortex "
        "and inspect transparent, sample-level XGBoost biomarker predictions powered by SHAP explainability."
    )

    with st.expander("ℹ️ How to Use This Dashboard (4 Quick Steps)", expanded=False):
        st.markdown("""
        1. **Select or Upload Data (Sidebar):** Explore the built-in GSE68719 discovery cohort (72 samples), switch to independent external cohorts, or upload your own normalized expression matrix.
        2. **Inspect Differential Expression (Section 2):** Filter transcriptome-wide DEGs using statistical significance (FDR padj) and effect size (|log2FC|) sliders to pinpoint upregulated and downregulated genes.
        3. **Evaluate Biomarker Predictions (Section 3):** View cohort-level predictions from the 18-gene consensus panel (union of LASSO and Boruta selections).
        4. **Audit Patient Decisions (Section 4):** Select any patient row in the table to generate an interactive SHAP waterfall explaining exactly which genes drove that individual's classification.
        """)

def render_disclaimer(meta: Optional[Dict[str, Any]] = None):
    """Renders mandatory regulatory disclaimer and model specification expander."""
    n_samples = 72
    acc_str = "GSE68719"
    if meta and "cohorts" in meta and "GSE68719" in meta["cohorts"]:
        n_samples = meta["cohorts"]["GSE68719"]["n_samples"]
        acc_str = meta["cohorts"]["GSE68719"]["geo_accession"]

    st.caption(
        f"⚠️ **Research demonstration only.** Not a medical device and not for clinical diagnosis. "
        f"Predictions come from a machine learning model trained on {n_samples} postmortem brain samples from {acc_str}; "
        f"performance on external clinical data is not guaranteed."
    )

    with st.expander("📖 About This Model & Clinical Limitations", expanded=False):
        st.markdown(f"""
        - **Training Cohort:** GSE68719 ({n_samples} QC-passed postmortem prefrontal cortex BA9 samples: 44 Control, 28 Parkinson's).
        - **Biomarker Panel:** 18 consensus genes derived from dual-feature selection (L1 LASSO regression + Boruta random forest).
        - **Algorithm:** Gradient Boosted Decision Trees (`XGBClassifier`) with native JSON serialization.
        - **Internal Validation:** Leave-One-Out Cross-Validation (LOOCV) Accuracy = 86.1%, ROC-AUC = 0.939 (reproducing original notebook metrics).
        - **Limitations:** Postmortem cortical tissue; small sample size (n=72); potential platform and batch discrepancies when applied to peripheral blood or diverse sequencing protocols.
        """)

def render_validation_summary(report: ValidationReport):
    """Renders quality inspection metrics and alerts for uploaded files."""
    st.subheader("Data Validation Report")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Samples Detected", f"{report.n_samples:,}")
    col2.metric("Genes Present", f"{report.n_genes:,}")
    col3.metric("Model Feature Coverage", f"{report.coverage*100:.1f}%")

    if report.warnings:
        for w in report.warnings:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ Dataset meets all quality thresholds and matches expected biomarker features.")

def render_training_notice():
    """Notice rendered when evaluating samples that were part of model training."""
    st.info(
        "ℹ️ **Training Set Notice:** The samples in the GSE68719 discovery cohort were used during model feature selection and training. "
        "Predictions on this cohort represent fitted model behavior; for independent external validation, please select **GSE136666** or **GSE168496** in the sidebar."
    )
