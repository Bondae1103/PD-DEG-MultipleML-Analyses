from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from app.core import UserInputError

def align_to_features(
    df: pd.DataFrame,
    feature_list: List[str],
    fill_values: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aligns and orders DataFrame columns strictly to the model's feature_list.
    Missing features are imputed using training-set feature medians.
    
    Returns:
        (aligned_df, list_of_missing_features)
    """
    if fill_values is None:
        fill_values = {}

    missing_features = [f for f in feature_list if f not in df.columns]
    
    aligned = df.copy()
    for f in missing_features:
        # Fill missing feature with training median if provided, else column median or 0
        fill_val = fill_values.get(f, 0.0)
        aligned[f] = fill_val

    # Strict column re-ordering
    aligned = aligned[feature_list]
    
    # Fill any remaining NaNs in present columns with feature median
    for f in feature_list:
        if aligned[f].isna().any():
            median_val = fill_values.get(f, float(aligned[f].median(skipna=True)))
            aligned[f] = aligned[f].fillna(median_val)

    return aligned.astype(np.float32), missing_features

def apply_preprocessing(
    df: pd.DataFrame,
    preproc: Optional[Dict[str, Any]] = None,
    is_raw_counts: bool = False
) -> pd.DataFrame:
    """
    Applies pipeline transformation to expression matrix.
    If data is raw counts, performs CPM normalization + log2(x + 1).
    If data is already normalized log-scale expression, verifies float32 formatting.
    """
    out = df.copy()
    
    if is_raw_counts:
        pseudocount = preproc.get("pseudocount", 1.0) if preproc else 1.0
        lib_sizes = out.sum(axis=1)
        # Avoid division by zero
        lib_sizes = lib_sizes.replace(0, 1.0)
        cpm = out.divide(lib_sizes, axis=0) * 1e6
        out = np.log2(cpm + pseudocount)

    return out.astype(np.float32)
