"""
app/tests/test_deg_fallback.py
------------------------------
Unit tests for the fallback DEG filter per Plan v2 Section 9.5.
Verifies column alignment, significance filtering, and approximate labeling.
"""

import numpy as np
import pandas as pd
from app.core.deg_fallback import run_fallback_deg_filter
from app.core.preprocess import clean_deg_list
from app.config import FALLBACK_FILTER_NAME

def test_fallback_deg_filter_output_and_labeling():
    np.random.seed(123)
    n_samples = 10
    n_genes = 30

    counts = pd.DataFrame(
        np.random.randint(5, 500, size=(n_genes, n_samples)),
        index=[f"100{i}" for i in range(n_genes)],
        columns=[f"S{i}" for i in range(n_samples)]
    ).reset_index().rename(columns={"index": "GENE ID"})

    # Give first 5 genes elevated counts in group PD
    counts.iloc[:5, 6:] = counts.iloc[:5, 6:] * 10

    meta = pd.DataFrame({
        "COUNT": [f"S{i}" for i in range(n_samples)],
        "STATUS": ["NO_PD"] * 5 + ["PD"] * 5
    })

    sig_df, full_df, dt = run_fallback_deg_filter(counts, meta, p_thresh=0.1, lfc_thresh=0.5)

    # 1. Verification of columns
    expected_cols = {"ENTREZID", "SYMBOL", "GENENAME", "logFC", "P.Value", "adj.P.Val", "Method"}
    assert expected_cols.issubset(set(full_df.columns))
    assert expected_cols.issubset(set(sig_df.columns))

    # 2. Verification of label
    assert (full_df["Method"] == FALLBACK_FILTER_NAME).all()
    assert (sig_df["Method"] == FALLBACK_FILTER_NAME).all()

    # 3. Compatibility with clean_deg_list
    cleaned = clean_deg_list(sig_df)
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty
