"""
app/config.py
-------------
Single source of truth for paths, thresholds, seeds, method definitions,
and display constants across the PD-DEG ML-Methods Dashboard.
Follows Plan v2 Section 3.1.
"""

import os

# Base directory for the app package
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing persisted machine learning models, gene lists, and cohorts
ARTIFACT_DIR = os.path.join(APP_DIR, "artifacts")

# Bundled GSE68719 asset paths
SAMPLE_DEG_PATH = os.path.join(ARTIFACT_DIR, "sample_deg_list.csv")
COUNTS_MATRIX_PATH = os.path.join(ARTIFACT_DIR, "counts_matrix.parquet")
SAMPLE_METADATA_PATH = os.path.join(ARTIFACT_DIR, "sample_metadata.csv")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

# GEO accession and dataset description
GEO_ACCESSION = "GSE68719"
DATASET_TITLE = "Human Postmortem Frontal Cortex (BA9) RNA-Seq"

# Random seed matching the notebook exactly
RANDOM_SEED = 123

# Fixed order of feature selection methods (used for deterministic fallback ordering)
METHOD_NAMES = ["LASSO", "SVM_RFE", "XGBoost", "MutualInfo", "Boruta"]

# Descriptive labels and tooltips for each method
METHOD_INFO = {
    "LASSO": {
        "title": "LASSO (L1 Regularization)",
        "estimator": "LogisticRegressionCV(Cs=20, cv=5, penalty='l1', solver='liblinear')",
        "metric": "Absolute coefficient (|coef|)",
        "scaling_required": True,
        "description": "L1-penalized sparse logistic regression shrinking non-informative gene weights to exactly zero."
    },
    "SVM_RFE": {
        "title": "SVM-RFE (Recursive Feature Elimination)",
        "estimator": "RFE(SVC(kernel='linear'), n_features_to_select=10)",
        "metric": "Refit linear SVM |coef|",
        "scaling_required": True,
        "description": "Iteratively prunes lowest-weight genes using a linear support vector machine margin."
    },
    "XGBoost": {
        "title": "XGBoost Importance",
        "estimator": "XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05)",
        "metric": "Gini / Split Feature Importance",
        "scaling_required": False,
        "description": "Gradient-boosted decision trees measuring non-linear feature split gain."
    },
    "MutualInfo": {
        "title": "Mutual Information",
        "estimator": "mutual_info_classif(random_state=123)",
        "metric": "Mutual Information score",
        "scaling_required": False,
        "description": "Non-parametric information-theoretic dependency score capturing non-linear relationships."
    },
    "Boruta": {
        "title": "Boruta (All-Relevant Feature Selection)",
        "estimator": "BorutaPy(RandomForestClassifier, max_iter=200)",
        "metric": "Refit Random Forest Gini importance",
        "scaling_required": False,
        "description": "Compares real feature importance against permuted shadow features to retain all statistically relevant biomarkers."
    }
}

# Explicit class mapping from training LabelEncoder
CLASS_MAP = {0: "NO_PD", 1: "PD"}
POSITIVE_LABEL_DISPLAY = "Parkinson's Disease (PD)"
NEGATIVE_LABEL_DISPLAY = "Neurologically Normal Control (NO_PD)"

# Constraints and upload thresholds
MAX_UPLOAD_MB = 20
MAX_DEG_ROWS = 5000
MIN_DEG_ROWS_FOR_METHODS = 15

# Fallback DEG filter constants (Plan v2 Section 1.5 & Section 4.6)
FALLBACK_FILTER_NAME = "Welch's t-test + BH-FDR (Approximate Filter)"
FALLBACK_P_THRESH = 0.05
FALLBACK_LFC_THRESH = 1.0

# Session state keys for Streamlit reactive state management
SS_DATA_SOURCE = "data_source"               # "sample" | "upload_deg" | "upload_raw"
SS_DEG_HASH = "deg_hash"                     # Content hash of DEG data
SS_SELECTED_METHODS = "selected_methods"     # List of selected method names
SS_ANALYSIS_READY = "analysis_ready"         # Bool flag indicating pipeline completion
SS_RESULTS = "results"                       # Dict[str, MethodResult]
SS_CONSENSUS = "consensus"                   # ConsensusResult
SS_VALIDATION = "validation"                 # ValidationResult
SS_CUSTOM_DEG_DF = "custom_deg_df"           # User-uploaded DEG table
SS_RAW_COUNTS_DF = "raw_counts_df"           # User-uploaded raw counts
SS_RAW_META_DF = "raw_meta_df"               # User-uploaded sample metadata
