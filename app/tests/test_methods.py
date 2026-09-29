"""
app/tests/test_methods.py
-------------------------
Unit tests for the 5 feature selection methods per Plan v2 Section 9.2.
Verifies determinism across reruns and <= 10 gene outputs on a synthetic dataset.
"""

import numpy as np
import pandas as pd
from app.core.methods import (
    run_lasso,
    run_svm_rfe,
    run_xgboost,
    run_mutual_info,
    run_boruta
)


def get_synthetic_data():
    np.random.seed(123)
    n_samples = 30
    n_features = 20
    X_mat = np.random.randn(n_samples, n_features)
    # Give first 3 features genuine signal
    y = ((X_mat[:, 0] * 1.5 + X_mat[:, 1] * -1.2 + np.random.randn(n_samples) * 0.5) > 0).astype(int)

    feature_ids = [str(100 + i) for i in range(n_features)]
    X = pd.DataFrame(X_mat, index=[f"S{i}" for i in range(n_samples)], columns=feature_ids)
    id_to_symbol = {fid: f"GENE_{fid}" for fid in feature_ids}
    return X, y, id_to_symbol

def test_methods_return_valid_shapes():
    X, y, id_to_symbol = get_synthetic_data()

    # 1. LASSO
    res_lasso = run_lasso(X, y, id_to_symbol, seed=123)
    assert len(res_lasso.top10_ids) <= 10
    assert len(res_lasso.top10_symbols) == len(res_lasso.top10_ids)

    # 2. SVM-RFE
    res_svm = run_svm_rfe(X, y, id_to_symbol, seed=123)
    assert len(res_svm.top10_ids) <= 10
    assert len(res_svm.top10_symbols) == len(res_svm.top10_ids)

    # 3. XGBoost
    res_xgb = run_xgboost(X, y, id_to_symbol, seed=123)
    assert len(res_xgb.top10_ids) <= 10
    assert len(res_xgb.top10_symbols) == len(res_xgb.top10_ids)

    # 4. Mutual Info
    res_mi = run_mutual_info(X, y, id_to_symbol, seed=123)
    assert len(res_mi.top10_ids) <= 10
    assert len(res_mi.top10_symbols) == len(res_mi.top10_ids)

    # 5. Boruta
    res_boruta = run_boruta(X, y, id_to_symbol, seed=123)
    assert len(res_boruta.top10_ids) <= 10
    assert len(res_boruta.top10_symbols) == len(res_boruta.top10_ids)

def test_methods_are_deterministic_across_reruns():
    X, y, id_to_symbol = get_synthetic_data()

    # LASSO
    l1 = run_lasso(X, y, id_to_symbol, seed=123).top10_ids
    l2 = run_lasso(X, y, id_to_symbol, seed=123).top10_ids
    assert l1 == l2

    # SVM-RFE
    s1 = run_svm_rfe(X, y, id_to_symbol, seed=123).top10_ids
    s2 = run_svm_rfe(X, y, id_to_symbol, seed=123).top10_ids
    assert s1 == s2

    # XGBoost
    x1 = run_xgboost(X, y, id_to_symbol, seed=123).top10_ids
    x2 = run_xgboost(X, y, id_to_symbol, seed=123).top10_ids
    assert x1 == x2

    # Mutual Info
    m1 = run_mutual_info(X, y, id_to_symbol, seed=123).top10_ids
    m2 = run_mutual_info(X, y, id_to_symbol, seed=123).top10_ids
    assert m1 == m2
