import os

# Base directory for the app package
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory containing persisted machine learning models, gene lists, and cohorts
ARTIFACT_DIR = os.path.join(APP_DIR, "artifacts")

# Path to the serialized XGBoost model in native JSON format
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.json")

# Path to the ordered JSON list of 18 biomarker gene symbols
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_list.json")

# Path to preprocessing parameters (feature medians, pseudocounts, ID maps)
PREPROC_PATH = os.path.join(ARTIFACT_DIR, "preprocessing.json")

# Path to the differential expression results table
DEG_PATH = os.path.join(ARTIFACT_DIR, "deg_results.csv")

# Path to pipeline metadata, provenance, and evaluation metrics
METADATA_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")

# Available cohorts catalog for exploratory analysis
COHORTS = {
    "GSE68719": {
        "display_name": "GSE68719 — Prefrontal Cortex (Discovery)",
        "matrix_path": os.path.join(ARTIFACT_DIR, "cohort_gse68719_normalized.parquet"),
        "meta_path": os.path.join(ARTIFACT_DIR, "cohort_gse68719_meta.csv"),
        "description": "Human postmortem frontal cortex (BA9); 72 samples (44 Control, 28 Parkinson's).",
        "geo_accession": "GSE68719",
        "tissue": "Frontal Cortex (BA9)",
        "platform": "Illumina HiSeq 2000",
        "n_samples": 72,
        "group_counts": {"NO_PD": 44, "PD": 28},
        "used_in_training": True,
        "contrast": "Parkinson's Disease vs Control (log2FC > 0 is elevated in PD)"
    },
    "GSE136666": {
        "display_name": "GSE136666 — Brain Tissue (External Validation)",
        "matrix_path": os.path.join(ARTIFACT_DIR, "cohort_gse136666_normalized.parquet"),
        "meta_path": os.path.join(ARTIFACT_DIR, "cohort_gse136666_meta.csv"),
        "description": "Independent human postmortem brain cohort; 16 samples (8 Control, 8 Parkinson's).",
        "geo_accession": "GSE136666",
        "tissue": "Brain Tissue",
        "platform": "Illumina NextSeq 500",
        "n_samples": 16,
        "group_counts": {"NO_PD": 8, "PD": 8},
        "used_in_training": False,
        "contrast": "Parkinson's Disease vs Control"
    },
    "GSE168496": {
        "display_name": "GSE168496 — Brain Tissue (External Validation)",
        "matrix_path": os.path.join(ARTIFACT_DIR, "cohort_gse168496_normalized.parquet"),
        "meta_path": os.path.join(ARTIFACT_DIR, "cohort_gse168496_meta.csv"),
        "description": "Independent human postmortem brain cohort; 16 samples (8 Control, 8 Parkinson's).",
        "geo_accession": "GSE168496",
        "tissue": "Brain Tissue",
        "platform": "Illumina NovaSeq 6000",
        "n_samples": 16,
        "group_counts": {"NO_PD": 8, "PD": 8},
        "used_in_training": False,
        "contrast": "Parkinson's Disease vs Control"
    }
}

# Explicit class mapping from training LabelEncoder
CLASS_MAP = {0: "NO_PD", 1: "PD"}

# User-facing display string for the positive target class
POSITIVE_LABEL_DISPLAY = "Parkinson's Disease (PD)"
NEGATIVE_LABEL_DISPLAY = "Neurologically Normal Control (NO_PD)"

# Default statistical thresholds matching the repository analysis
DEFAULT_PVAL_THRESH = 0.05
DEFAULT_LOG2FC_THRESH = 1.0

# Upload constraints and validation limits
MAX_UPLOAD_MB = 50
MIN_FEATURE_COVERAGE = 0.80
MAX_UPLOAD_SAMPLES = 2000

# UI display budgets and limits
TOP_N_SHAP = 15
VOLCANO_LABEL_TOP_N = 10

# Original random seed from the repository training pipeline
RANDOM_SEED = 123

# Standardized Plotly color palette across Volcano and SHAP
COLOR_UP = "#D62728"        # Red for significantly up-regulated genes
COLOR_DOWN = "#1F77B4"      # Blue for significantly down-regulated genes
COLOR_NOT_SIG = "#BDBDBD"   # Neutral gray for non-significant genes
COLOR_SHAP_POS = "#D62728"  # Red pushes toward positive class (PD)
COLOR_SHAP_NEG = "#1F77B4"  # Blue pushes toward negative class (NO_PD)

# Session state keys for Streamlit reactive state management
SS_DATA_SOURCE = "data_source"
SS_ACTIVE_COHORT = "active_cohort"
SS_UPLOAD_HASH = "upload_hash"
SS_SELECTED_SAMPLE = "selected_sample"
SS_P_THRESH = "p_thresh"
SS_LFC_THRESH = "lfc_thresh"
SS_USE_PADJ = "use_padj"
