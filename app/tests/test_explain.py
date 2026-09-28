import json
import pytest
import pandas as pd
import numpy as np

from app.config import MODEL_PATH, FEATURES_PATH, COHORTS, TOP_N_SHAP
from app.core.model import load_model
from app.core.explain import get_explainer, explain_sample, waterfall_frame

def test_explain_sample_additivity_20_samples():
    model = load_model(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
        
    df_expr = pd.read_parquet(COHORTS["GSE68719"]["matrix_path"])[features]
    explainer = get_explainer(model)

    # Test 20 samples
    for i in range(min(20, len(df_expr))):
        row = df_expr.iloc[i]
        sample_id = df_expr.index[i]
        result = explain_sample(explainer, model, row, sample_id=sample_id)
        
        # Verify additivity: base + sum(shap) == margin within 1e-3
        recon_margin = result.base_value + float(np.sum(result.shap_values))
        assert abs(recon_margin - result.margin) < 1e-3
        assert 0.0 <= result.probability <= 1.0

def test_waterfall_frame_length_and_cumulative_margin():
    model = load_model(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
        
    df_expr = pd.read_parquet(COHORTS["GSE68719"]["matrix_path"])[features]
    explainer = get_explainer(model)
    
    row = df_expr.iloc[0]
    result = explain_sample(explainer, model, row, sample_id=df_expr.index[0])
    
    frame = waterfall_frame(result, top_n=TOP_N_SHAP)
    # Must have <= TOP_N_SHAP + 1 rows
    assert len(frame) <= TOP_N_SHAP + 1
    
    # Cumulative sum must end at margin within 1e-3
    final_cumulative = frame["cumulative_end"].iloc[-1]
    assert abs(final_cumulative - result.margin) < 1e-3
