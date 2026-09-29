"""
app/core/validate.py
--------------------
Verbatim port of Leave-One-Out Cross-Validation (LOOCV) with RandomForestClassifier
from notebook cells 11-13. Follows Plan v2 Section 4.7. Contains zero Streamlit imports.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from app.config import RANDOM_SEED

@dataclass
class ValidationResult:
    accuracy: float
    auc_score: float
    y_true: np.ndarray
    y_proba_loo: np.ndarray
    y_pred_loo: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    cm: np.ndarray
    report_dict: Dict[str, Any]
    rf_fitted: RandomForestClassifier
    panel_features: List[str]

def run_loocv_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    panel_gene_ids: List[str],
    seed: int = RANDOM_SEED
) -> ValidationResult:
    """
    Evaluates the consensus gene panel using Leave-One-Out Cross-Validation (LOOCV)
    with a balanced 500-tree RandomForestClassifier matching notebook Steps 10a-10c.
    Guarded to operate cleanly on panels of any size (from 1 gene to N genes).
    """
    if not panel_gene_ids:
        raise ValueError("Cannot perform validation on an empty gene panel.")

    # Align available features
    available_panel = [str(g) for g in panel_gene_ids if str(g) in X.columns]
    if not available_panel:
        raise ValueError(f"None of the panel gene IDs {panel_gene_ids} were found in the expression matrix columns.")

    X_final = X[available_panel]

    loo = LeaveOneOut()
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)

    # 1. LOOCV Accuracy
    scores = cross_val_score(rf, X_final, y, cv=loo, scoring="accuracy")
    accuracy = float(scores.mean())

    # 2. LOOCV Predicted Probabilities for Class 1 (PD)
    y_proba_loo = cross_val_predict(rf, X_final, y, cv=loo, method="predict_proba")[:, 1]

    # 3. ROC metrics
    fpr, tpr, _ = roc_curve(y, y_proba_loo)
    try:
        auc_score = float(roc_auc_score(y, y_proba_loo))
    except Exception:
        auc_score = 0.5

    # 4. Predictions & Confusion Matrix
    y_pred_loo = (y_proba_loo >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred_loo)
    report_dict = classification_report(y, y_pred_loo, output_dict=True, zero_division=0)

    # 5. Fit final RF instance on all samples for downstream SHAP explainer
    rf_fitted = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
    rf_fitted.fit(X_final, y)

    return ValidationResult(
        accuracy=accuracy,
        auc_score=auc_score,
        y_true=y,
        y_proba_loo=y_proba_loo,
        y_pred_loo=y_pred_loo,
        fpr=fpr,
        tpr=tpr,
        cm=cm,
        report_dict=report_dict,
        rf_fitted=rf_fitted,
        panel_features=available_panel
    )
