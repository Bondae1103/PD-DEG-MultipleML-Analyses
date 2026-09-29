"""
app/core/methods.py
-------------------
Verbatim ports of the 5 machine learning feature selection methods from notebook cells 4-8.
Follows Plan v2 Section 4.3 & Section 1.3. Contains zero Streamlit imports.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from boruta import BorutaPy
from app.config import RANDOM_SEED, METHOD_NAMES, METHOD_INFO

@dataclass
class MethodResult:
    method_name: str
    metric_name: str
    top10_ids: List[str]
    top10_symbols: List[str]
    scores: Dict[str, float]
    n_available: int

def run_lasso(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    seed: int = RANDOM_SEED
) -> MethodResult:
    """
    LASSO (L1 Regularization) feature selection on StandardScaler-scaled X.
    Exact port of Notebook Cell 4.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso_cv = LogisticRegressionCV(
        Cs=20,
        cv=5,
        penalty="l1",
        solver="liblinear",
        scoring="accuracy",
        max_iter=5000,
        random_state=seed
    )
    lasso_cv.fit(X_scaled, y)

    lasso_coefs = pd.Series(lasso_cv.coef_[0], index=X.columns)
    nonzero_coefs = lasso_coefs[lasso_coefs != 0]

    top_series = nonzero_coefs.abs().sort_values(ascending=False).head(10)
    top_ids = [str(g) for g in top_series.index.tolist()]
    top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
    scores_dict = {str(k): float(v) for k, v in top_series.items()}

    return MethodResult(
        method_name="LASSO",
        metric_name=METHOD_INFO["LASSO"]["metric"],
        top10_ids=top_ids,
        top10_symbols=top_symbols,
        scores=scores_dict,
        n_available=len(nonzero_coefs)
    )

def run_svm_rfe(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    seed: int = RANDOM_SEED
) -> MethodResult:
    """
    SVM-RFE feature selection on StandardScaler-scaled X.
    Selects 10 genes via linear SVM margin, then refits to rank by |coef|.
    Exact port of Notebook Cell 5.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = min(10, X.shape[1])
    svm_linear = SVC(kernel="linear", class_weight="balanced", random_state=seed)
    rfe = RFE(estimator=svm_linear, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled, y)

    selected_ids = X.columns[rfe.support_].tolist()

    # Refit linear SVM on the selected subset to rank by |coef|
    svm_final = SVC(kernel="linear", class_weight="balanced", random_state=seed)
    svm_final.fit(X_scaled[:, rfe.support_], y)

    svm_coefs = pd.Series(svm_final.coef_[0], index=selected_ids)
    top_series = svm_coefs.abs().sort_values(ascending=False).head(10)

    top_ids = [str(g) for g in top_series.index.tolist()]
    top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
    scores_dict = {str(k): float(v) for k, v in top_series.items()}

    return MethodResult(
        method_name="SVM_RFE",
        metric_name=METHOD_INFO["SVM_RFE"]["metric"],
        top10_ids=top_ids,
        top10_symbols=top_symbols,
        scores=scores_dict,
        n_available=len(selected_ids)
    )

def run_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    seed: int = RANDOM_SEED
) -> MethodResult:
    """
    XGBoost importance feature selection on raw (unscaled) X.
    Exact port of Notebook Cell 6.
    """
    neg, pos = np.bincount(y)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1
    )
    xgb_clf.fit(X, y)

    xgb_importance = pd.Series(xgb_clf.feature_importances_, index=X.columns)
    top_series = xgb_importance.sort_values(ascending=False).head(10)

    top_ids = [str(g) for g in top_series.index.tolist()]
    top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
    scores_dict = {str(k): float(v) for k, v in top_series.items()}

    return MethodResult(
        method_name="XGBoost",
        metric_name=METHOD_INFO["XGBoost"]["metric"],
        top10_ids=top_ids,
        top10_symbols=top_symbols,
        scores=scores_dict,
        n_available=int((xgb_importance > 0).sum())
    )

def run_mutual_info(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    seed: int = RANDOM_SEED
) -> MethodResult:
    """
    Mutual Information feature selection on raw (unscaled) X.
    Exact port of Notebook Cell 7.
    """
    mi_scores = mutual_info_classif(X, y, random_state=seed)
    mi_series = pd.Series(mi_scores, index=X.columns)
    top_series = mi_series.sort_values(ascending=False).head(10)

    top_ids = [str(g) for g in top_series.index.tolist()]
    top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
    scores_dict = {str(k): float(v) for k, v in top_series.items()}

    return MethodResult(
        method_name="MutualInfo",
        metric_name=METHOD_INFO["MutualInfo"]["metric"],
        top10_ids=top_ids,
        top10_symbols=top_symbols,
        scores=scores_dict,
        n_available=int((mi_series > 0).sum())
    )

def run_boruta(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    seed: int = RANDOM_SEED
) -> MethodResult:
    """
    Boruta all-relevant feature selection on raw (unscaled) X.
    Uses 200 iterations matching notebook Cell 8, followed by a second 500-tree RF
    to rank confirmed features by Gini importance.
    """
    rf_internal = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=seed)
    boruta_selector = BorutaPy(rf_internal, n_estimators="auto", verbose=0, random_state=seed, max_iter=200)
    boruta_selector.fit(X.values, y)

    selected_ids = X.columns[boruta_selector.support_].tolist()

    if len(selected_ids) == 0:
        # Fallback to tentative support if none confirmed
        selected_ids = X.columns[boruta_selector.support_weak_].tolist()
        if len(selected_ids) == 0:
            # Fallback to top genes from the initial random forest
            rf_fallback = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
            rf_fallback.fit(X, y)
            top_series = pd.Series(rf_fallback.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
            top_ids = [str(g) for g in top_series.index.tolist()]
            top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
            return MethodResult(
                method_name="Boruta",
                metric_name=METHOD_INFO["Boruta"]["metric"],
                top10_ids=top_ids,
                top10_symbols=top_symbols,
                scores={str(k): float(v) for k, v in top_series.items()},
                n_available=0
            )

    # Second RF refit on confirmed genes to rank them
    X_confirmed = X[selected_ids]
    rf_rank = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
    rf_rank.fit(X_confirmed, y)

    boruta_importance = pd.Series(rf_rank.feature_importances_, index=selected_ids)
    top_series = boruta_importance.sort_values(ascending=False).head(10)

    top_ids = [str(g) for g in top_series.index.tolist()]
    top_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_ids]
    scores_dict = {str(k): float(v) for k, v in top_series.items()}

    return MethodResult(
        method_name="Boruta",
        metric_name=METHOD_INFO["Boruta"]["metric"],
        top10_ids=top_ids,
        top10_symbols=top_symbols,
        scores=scores_dict,
        n_available=len(selected_ids)
    )

# Method dispatch mapping
METHOD_FUNCTIONS = {
    "LASSO": run_lasso,
    "SVM_RFE": run_svm_rfe,
    "XGBoost": run_xgboost,
    "MutualInfo": run_mutual_info,
    "Boruta": run_boruta
}

def run_selected_methods(
    X: pd.DataFrame,
    y: np.ndarray,
    id_to_symbol: Dict[str, str],
    selected_method_names: List[str],
    seed: int = RANDOM_SEED,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, MethodResult]:
    """
    Runs only the user-checked methods, strictly in METHOD_NAMES order.
    """
    results: Dict[str, MethodResult] = {}
    ordered_selection = [m for m in METHOD_NAMES if m in selected_method_names]
    total = len(ordered_selection)

    for i, method_name in enumerate(ordered_selection):
        if progress_callback:
            progress_callback(method_name, i + 1, total)

        func = METHOD_FUNCTIONS[method_name]
        res = func(X, y, id_to_symbol, seed=seed)
        results[method_name] = res

    return results
