import json
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from app.config import MODEL_PATH, FEATURES_PATH, METADATA_PATH, CLASS_MAP, COHORTS
from app.core.model import load_model, predict

def test_model_predictions_format_and_probabilities():
    model = load_model(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    # Test with GSE68719 cohort data
    cohort_path = COHORTS["GSE68719"]["matrix_path"]
    meta_path = COHORTS["GSE68719"]["meta_path"]
    df_expr = pd.read_parquet(cohort_path)[features]
    df_meta = pd.read_csv(meta_path)

    pred_df = predict(model, df_expr, CLASS_MAP, meta_df=df_meta)

    # Check columns
    assert "sample_id" in pred_df.columns
    assert "prob_positive" in pred_df.columns
    assert "predicted_label" in pred_df.columns
    assert "confidence" in pred_df.columns
    assert "margin" in pred_df.columns
    assert "true_label" in pred_df.columns

    # Probabilities must be strictly bounded in [0, 1]
    assert (pred_df["prob_positive"] >= 0.0).all()
    assert (pred_df["prob_positive"] <= 1.0).all()
    assert (pred_df["confidence"] >= 0.5).all()

    # Labels must be in CLASS_MAP values
    valid_labels = set(CLASS_MAP.values())
    assert set(pred_df["predicted_label"]).issubset(valid_labels)

def test_model_accuracy_benchmark():
    model = load_model(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)

    df_expr = pd.read_parquet(COHORTS["GSE68719"]["matrix_path"])[features]
    df_meta = pd.read_csv(COHORTS["GSE68719"]["meta_path"])

    pred_df = predict(model, df_expr, CLASS_MAP, meta_df=df_meta)
    
    # Train accuracy on full dataset should be >= LOOCV accuracy
    acc = accuracy_score(pred_df["true_label"], pred_df["predicted_label"])
    assert acc >= 0.85
