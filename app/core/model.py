import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import xgboost as xgb

from app.config import MODEL_PATH, CLASS_MAP

def load_model(path: Optional[str] = None) -> xgb.XGBClassifier:
    """
    Loads trained XGBoost classifier from serialized native JSON.
    """
    if path is None:
        path = MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

def predict(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    class_map: Optional[Dict[int, str]] = None,
    meta_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate predictions, probabilities, and log-odds margins for input samples.
    
    Returns:
        DataFrame with columns:
        ['sample_id', 'prob_positive', 'predicted_label', 'confidence', 'margin', ('true_label', 'held_out')]
    """
    if class_map is None:
        class_map = CLASS_MAP

    sample_ids = X.index.tolist()
    
    # Probabilities
    probas = model.predict_proba(X)
    # Binary classification: positive class is index 1 ('PD')
    prob_pos = probas[:, 1]
    
    # Raw margin (log-odds)
    margins = model.predict(X, output_margin=True)
    
    # Predicted class
    pred_idx = (prob_pos >= 0.5).astype(int)
    pred_labels = [class_map.get(idx, str(idx)) for idx in pred_idx]
    
    # Confidence: probability assigned to the chosen class
    confidence = np.max(probas, axis=1)

    result_df = pd.DataFrame({
        "sample_id": sample_ids,
        "prob_positive": prob_pos,
        "predicted_label": pred_labels,
        "confidence": confidence,
        "margin": margins
    })

    # Join metadata if provided
    if meta_df is not None:
        meta_clean = meta_df.copy()
        if "sample_id" in meta_clean.columns:
            meta_clean = meta_clean.set_index("sample_id")
        
        if "group" in meta_clean.columns:
            result_df["true_label"] = result_df["sample_id"].map(meta_clean["group"].to_dict())
        if "split" in meta_clean.columns:
            result_df["split"] = result_df["sample_id"].map(meta_clean["split"].to_dict())
        if "held_out" in meta_clean.columns:
            result_df["held_out"] = result_df["sample_id"].map(meta_clean["held_out"].to_dict())

    return result_df
