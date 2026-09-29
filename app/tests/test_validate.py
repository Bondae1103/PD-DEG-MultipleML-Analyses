"""
app/tests/test_validate.py
--------------------------
Unit tests for LOOCV validation and SHAP additivity per Plan v2 Section 9.4.
Verifies LOOCV execution on panels of size 1, 2, and 10, and tests SHAP additivity.
"""

import numpy as np
import pandas as pd
import shap
from app.core.validate import run_loocv_validation

def get_test_matrix():
    np.random.seed(123)
    n_samples = 20
    n_genes = 15
    X_mat = np.random.randn(n_samples, n_genes)
    y = (X_mat[:, 0] > 0).astype(int)

    feature_ids = [str(i) for i in range(n_genes)]
    X = pd.DataFrame(X_mat, index=[f"S{i}" for i in range(n_samples)], columns=feature_ids)
    return X, y, feature_ids

def test_loocv_on_panel_sizes_1_2_10():
    X, y, feature_ids = get_test_matrix()

    for size in [1, 2, 10]:
        panel = feature_ids[:size]
        res = run_loocv_validation(X, y, panel, seed=123)
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.auc_score <= 1.0
        assert len(res.y_proba_loo) == len(y)
        assert res.cm.shape == (2, 2)
        assert len(res.panel_features) == size

def test_shap_additivity():
    X, y, feature_ids = get_test_matrix()
    panel = feature_ids[:5]
    res = run_loocv_validation(X, y, panel, seed=123)

    explainer = shap.TreeExplainer(res.rf_fitted)
    shap_vals = explainer.shap_values(X[panel])
    probs = res.rf_fitted.predict_proba(X[panel])[:, 1]

    expected_val = explainer.expected_value[1]
    shap_sum = shap_vals[:, :, 1].sum(axis=1) + expected_val
    diff = np.max(np.abs(shap_sum - probs))
    assert diff < 1e-3, f"SHAP additivity difference too large: {diff}"
