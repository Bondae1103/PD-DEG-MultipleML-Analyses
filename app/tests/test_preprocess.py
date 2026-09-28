import json
import pytest
import pandas as pd
import numpy as np

from app.config import FEATURES_PATH, PREPROC_PATH, COHORTS
from app.core.preprocess import align_to_features, apply_preprocessing

def test_align_to_features_order_and_imputation():
    features = ["GENE_A", "GENE_B", "GENE_C"]
    medians = {"GENE_A": 2.5, "GENE_B": 4.0, "GENE_C": 1.2}
    
    # Input missing GENE_B and in wrong order
    df = pd.DataFrame({
        "GENE_C": [1.0, 2.0],
        "GENE_A": [3.0, 4.0]
    }, index=["S1", "S2"])

    aligned, missing = align_to_features(df, features, medians)
    
    assert list(aligned.columns) == features
    assert missing == ["GENE_B"]
    assert aligned.loc["S1", "GENE_B"] == 4.0
    assert aligned.dtypes["GENE_A"] == np.float32

def test_builtin_cohort_matches_model_features():
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    with open(PREPROC_PATH, "r") as f:
        preproc = json.load(f)

    cohort_path = COHORTS["GSE68719"]["matrix_path"]
    df = pd.read_parquet(cohort_path)

    aligned, missing = align_to_features(df, features, preproc["feature_medians"])
    assert missing == []
    assert list(aligned.columns) == features
    # Check that values for the first sample match stored parquet exactly
    for col in features:
        diff = abs(aligned.loc["C.1", col] - df.loc["C.1", col])
        assert diff < 1e-5
