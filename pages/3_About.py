"""
pages/3_About.py
----------------
Comprehensive methodology and documentation page per Plan v2 Section 6.4.
Explains the 5 machine learning methods, the consensus logic, plot interpretations,
and dataset provenance.
"""

import streamlit as st
from app.ui.styles import apply_custom_styles
from app.ui.nav import render_top_nav
from app.ui.components import render_header, render_disclaimer, render_provenance_badge
from app.config import GEO_ACCESSION, FALLBACK_FILTER_NAME

apply_custom_styles()
render_top_nav("About")

render_header(
    title="Pipeline Methodology & Documentation",
    subtitle="Technical background on the 5-method feature selection framework, consensus rules, and cross-validation."
)

render_provenance_badge("Reference Documentation")

# -------------------------------------------------------------
# Section 1: Pipeline Architecture
# -------------------------------------------------------------
st.markdown("### 1. End-to-End Pipeline Architecture")
st.markdown(r"""
High-throughput transcriptomics experiments often suffer from the **curse of dimensionality** ($p \gg n$), where tens of thousands of genes are measured across dozens of patients. Single machine learning models frequently overfit to dataset-specific noise or collinear feature clusters.

This pipeline implements a multi-algorithmic consensus strategy:
""")

st.markdown("""
```
┌─────────────────────────────────┐
│ GSE68719 Prefrontal Cortex      │  72 postmortem RNA-Seq samples (44 Control / 28 PD)
│ Candidate DEGs (limma-voom)     │  430 significant transcripts (|log2FC| > 1, FDR < 0.05)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Transcript Quality Cleaning     │  Drop NA Entrez, uncharacterized LOC*, pseudogenes,
│                                 │  microRNAs (MIR*), and small nucleolar RNAs (SNOR*)
└────────────────┬────────────────┘  ➜ 351 Cleaned Protein-Coding Transcripts
                 │
                 ▼
┌─────────────────────────────────┐
│ Library-Size CPM Normalization  │  Counts normalized using FULL transcriptome library size
│                                 │  Transform: log2(CPM + 1)
└────────────────┬────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────────────┐
│ 5 Parallel Feature Selection Methods                          │
│                                                               │
│  1. LASSO L1      2. SVM-RFE      3. XGBoost                  │
│  (L1 Logistic)    (Margin RFE)    (Gini Gain)                 │
│                                                               │
│          4. Mutual Info         5. Boruta                     │
│          (Entropy Dependency)   (Shadow Feature RF)           │
└────────────────┬──────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Dynamic Consensus Synthesis     │  Intersection ➔ Most-Occurring Tier ➔ Deterministic Fallback
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Cross-Validation & SHAP Audit   │  Leave-One-Out Cross-Validation (LOOCV, 500-Tree RF)
│                                 │  ROC, Confusion Matrix, Boxplots, Heatmap, SHAP Summary
└─────────────────────────────────┘
```
""")

# -------------------------------------------------------------
# Section 2: The 5 Feature Selection Methods
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 2. The 5 Feature Selection Algorithms")
st.markdown("""
To achieve methodological diversity, the pipeline executes five complementary feature selection algorithms spanning linear regularization, margin-based elimination, gradient boosting, information theory, and all-relevant shadow testing:
""")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("#### 1. LASSO (L1 Regularization)")
    st.write(
        "**Estimator:** `LogisticRegressionCV(Cs=20, cv=5, penalty='l1', solver='liblinear')` on z-scored features.\n\n"
        "**Mechanism:** Imposes an L1 penalty on regression coefficients, driving non-informative feature weights to exactly zero. "
        "It acts as a sparse feature selector that penalizes correlated redundancy. The top 10 genes are ranked by absolute coefficient magnitude."
    )

    st.markdown("#### 2. SVM-RFE (Support Vector Machine - RFE)")
    st.write(
        "**Estimator:** `RFE(SVC(kernel='linear', class_weight='balanced'), n_features_to_select=10)` on z-scored features.\n\n"
        "**Mechanism:** Iteratively trains a linear SVM, calculates margin weights, and prunes the lowest-contributing feature until exactly 10 features remain. "
        "A final linear SVM is refit on the subset to establish internal feature rankings."
    )

    st.markdown("#### 3. XGBoost Importance")
    st.write(
        "**Estimator:** `XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)` on unscaled features.\n\n"
        "**Mechanism:** Constructs an ensemble of gradient-boosted decision trees. Feature importance is computed from the total split gain (Gini impurity reduction) "
        "across all boosting stages, capturing non-linear interactions without assuming data normality."
    )

with col_m2:
    st.markdown("#### 4. Mutual Information")
    st.write(
        "**Estimator:** `mutual_info_classif(random_state=123)` on unscaled features.\n\n"
        "**Mechanism:** A non-parametric information-theoretic score measuring the shared mutual dependence between continuous gene expression "
        "and discrete disease status. Unlike linear correlation, Mutual Information captures complex non-linear relationships without distributional assumptions."
    )

    st.markdown("#### 5. Boruta (All-Relevant Feature Selection)")
    st.write(
        "**Estimator:** `BorutaPy(RandomForestClassifier, n_estimators='auto', max_iter=200)` on unscaled features.\n\n"
        "**Mechanism:** Duplicates the dataset, shuffles each feature to create synthetic 'shadow' features, and trains random forests. "
        "A real feature is confirmed only if its importance statistically exceeds the maximum shadow feature across 200 iterations. "
        "A second 500-tree RF refit ranks confirmed genes by mean decrease in impurity."
    )

# -------------------------------------------------------------
# Section 3: Consensus Rule Logic
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 3. Consensus Panel Synthesis Logic")
st.markdown("""
Rather than relying on any single model, the dashboard applies a deterministic consensus rule across all selected methods:

1. **Single Method Selected:** If the user checks only one algorithm, no consensus voting is applied. The panel directly equals that method's top 10 ranked genes.
2. **Strict Intersection:** If all selected algorithms agree on one or more genes (e.g., 3 out of 3 methods choose `ADAM33`), those common genes form the consensus panel.
3. **Most-Occurring Tier:** If no gene is selected by all methods, the pipeline identifies the maximum vote count ($v > 1$) and selects all genes in that highest agreement tier.
4. **Total Disjoint Fallback:** If zero overlap exists across all candidate lists (all genes receive exactly 1 vote), the pipeline deterministically selects the #1 ranked gene from the first selected method in the fixed order: `["LASSO", "SVM_RFE", "XGBoost", "MutualInfo", "Boruta"]`.
""")

# -------------------------------------------------------------
# Section 4: How to Interpret Diagnostic Plots
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 4. Interpretation of Cross-Validation Figures")

col_i1, col_i2 = st.columns(2)

with col_i1:
    st.markdown("#### Receiver Operating Characteristic (ROC) Curve")
    st.write(
        "Evaluates diagnostic sensitivity versus false-positive rate across varying probability thresholds under Leave-One-Out Cross-Validation. "
        "An Area Under the Curve (AUC) of 0.5 represents chance, while 1.0 represents perfect discrimination. "
        "Values above 0.85 indicate strong generalization on held-out samples."
    )

    st.markdown("#### LOOCV Confusion Matrix")
    st.write(
        "Summarizes discrete patient classifications under a standard 0.5 decision threshold. "
        "The diagonal elements show True Negatives (top-left) and True Positives (bottom-right), highlighting balance between disease detection and false alarms."
    )

    st.markdown("#### Hierarchical Clustered Heatmap")
    st.write(
        "Visualizes standardized expression ($z$-score across samples) for all consensus biomarkers. "
        "Dendrograms cluster co-regulated genes and patient cohorts, revealing whether PD and Control samples naturally stratify based on the biomarker panel."
    )

with col_i2:
    st.markdown("#### Per-Gene Boxplots + Stripplots")
    st.write(
        "Displays log2(CPM + 1) normalized expression distributions for each individual biomarker, separated by diagnostic group. "
        "Black dots show individual sample data points, showing variance and confirming differential expression directionality (up- or down-regulated in PD)."
    )

    st.markdown("#### SHAP Summary Beeswarm Plot")
    st.write(
        "Quantifies feature impact on model decisions using Shapley Additive exPlanations (SHAP). "
        "Each point is an individual patient. Points to the right of the vertical zero-line push the prediction toward Parkinson's Disease. "
        "Color represents relative expression (red = high expression, blue = low expression)."
    )

# -------------------------------------------------------------
# Section 5: Dataset Provenance & Narrative
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 5. Dataset Provenance & Research Context")
st.markdown(f"""
- **Primary Study Accession:** **{GEO_ACCESSION}**
- **Tissue Type:** Postmortem human prefrontal cortex (Brodmann Area 9, BA9).
- **Cohort Composition:** 72 QC-passed postmortem donors (44 neurologically normal controls, 28 idiopathic Parkinson's disease).
- **Sequencing Technology:** Illumina HiSeq 2000 RNA-Seq (paired-end read counts mapped to GRCh38).
- **Sample Exclusions:** One sample (`T.20`) was excluded as an extreme multidimensional scaling (MDS) outlier belonging to an isolated cluster (`P_cluster2`).

> **Provenance Guarantee:** The sample DEG list provided on the Setup page is the **actual DEG list used in this analysis** (derived from GSE68719 via `limma-voom`). 
> Uploading alternative candidate DEG lists **derived from GSE68719** is the intended method for testing custom feature selection filters against the bundled counts matrix.

#### Note on the Fallback Filter
When using the "filter from raw counts" option, the dashboard executes `{FALLBACK_FILTER_NAME}`. 
This is a vectorized Welch's t-test with Benjamini-Hochberg FDR correction. It is **not** the `limma-voom` empirical Bayes method used in the published study, 
and is provided solely as a lightweight exploration tool for users without a precomputed DEG table.
""")

render_disclaimer()
