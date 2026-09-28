from dataclasses import dataclass
from typing import List, Union, Tuple, Optional
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from app.core import UserInputError
from app.config import TOP_N_SHAP

@dataclass
class ShapResult:
    """Detailed SHAP attribution result for a single sample."""
    sample_id: str
    base_value: float
    shap_values: np.ndarray
    feature_names: List[str]
    feature_values: np.ndarray
    margin: float
    probability: float

def probability_from_margin(margin: float) -> float:
    """Calculates logistic sigmoid probability from raw log-odds margin."""
    return float(1.0 / (1.0 + np.exp(-margin)))

def get_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Constructs and returns TreeExplainer for the XGBoost model."""
    return shap.TreeExplainer(model)

def explain_sample(
    explainer: shap.TreeExplainer,
    model: xgb.XGBClassifier,
    x_row: Union[pd.DataFrame, pd.Series],
    sample_id: str = "Sample"
) -> ShapResult:
    """
    Computes SHAP feature attributions for a single sample.
    Asserts mathematical additivity within 1e-3.
    """
    if isinstance(x_row, pd.Series):
        x_df = pd.DataFrame([x_row])
    else:
        x_df = x_row.copy()

    # Raw model margin
    raw_margin = float(model.predict(x_df, output_margin=True)[0])
    prob = probability_from_margin(raw_margin)

    # Compute SHAP
    shap_raw = explainer.shap_values(x_df)
    base_val = explainer.expected_value

    # Extract 1D array for positive class (PD)
    if isinstance(shap_raw, list):
        # Multi-class or 2-element list format
        sv = np.array(shap_raw[1][0], dtype=np.float64)
        base = float(base_val[1] if isinstance(base_val, (list, np.ndarray)) else base_val)
    elif shap_raw.ndim == 3:
        # Shape: (samples, features, classes)
        sv = np.array(shap_raw[0, :, 1], dtype=np.float64)
        base = float(base_val[1] if isinstance(base_val, (list, np.ndarray)) else base_val)
    elif shap_raw.ndim == 2:
        # Shape: (samples, features)
        sv = np.array(shap_raw[0], dtype=np.float64)
        base = float(base_val)
    else:
        sv = np.array(shap_raw, dtype=np.float64)
        base = float(base_val)

    # Mathematical additivity assertion
    recon_margin = base + np.sum(sv)
    diff = abs(recon_margin - raw_margin)
    if diff > 1e-3:
        raise ValueError(
            f"SHAP additivity violation: base ({base:.4f}) + sum(SHAP) ({np.sum(sv):.4f}) = {recon_margin:.4f} "
            f"!= margin {raw_margin:.4f} (diff = {diff:.2e} > 1e-3)."
        )

    feat_names = list(x_df.columns)
    feat_values = x_df.values[0]

    return ShapResult(
        sample_id=sample_id,
        base_value=base,
        shap_values=sv,
        feature_names=feat_names,
        feature_values=feat_values,
        margin=raw_margin,
        probability=prob
    )

def waterfall_frame(result: ShapResult, top_n: int = TOP_N_SHAP) -> pd.DataFrame:
    """
    Constructs waterfall DataFrame aggregating top positive/negative drivers
    and grouping remaining genes into a single '<k> other genes' row.
    """
    df = pd.DataFrame({
        "gene": result.feature_names,
        "feature_value": result.feature_values,
        "shap": result.shap_values
    })
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    if len(df) > top_n:
        top_df = df.iloc[:top_n].copy()
        rest_df = df.iloc[top_n:]
        rest_count = len(rest_df)
        rest_shap = float(rest_df["shap"].sum())
        
        other_row = pd.DataFrame({
            "gene": [f"{rest_count} other genes"],
            "feature_value": [np.nan],
            "shap": [float(rest_shap)],
            "abs_shap": [float(abs(rest_shap))]
        })
        plot_df = pd.concat([top_df, other_row], ignore_index=True)
    else:
        plot_df = df.copy()

    # Calculate cumulative positions
    running = result.base_value
    starts = []
    ends = []
    for s in plot_df["shap"]:
        starts.append(running)
        running += s
        ends.append(running)

    plot_df["cumulative_start"] = starts
    plot_df["cumulative_end"] = ends
    return plot_df
