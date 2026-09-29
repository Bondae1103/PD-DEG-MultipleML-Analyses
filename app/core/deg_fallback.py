"""
app/core/deg_fallback.py
------------------------
Vectorized Welch's t-test + Benjamini-Hochberg FDR fallback filter for users uploading raw counts.
Follows Plan v2 Section 1.5 & Section 4.6. Labeled explicitly as an approximate substitute.
Contains zero Streamlit imports.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, Optional
from app.config import FALLBACK_FILTER_NAME, FALLBACK_P_THRESH, FALLBACK_LFC_THRESH

def run_fallback_deg_filter(
    counts_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    id_to_symbol: Optional[Dict[str, str]] = None,
    p_thresh: float = FALLBACK_P_THRESH,
    lfc_thresh: float = FALLBACK_LFC_THRESH,
    use_padj: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Runs a vectorized two-sample Welch's t-test on log2(CPM + 1) normalized counts.
    Returns:
      (deg_significant_table, full_deg_results, execution_time_seconds)
    All outputs are explicitly tagged with FALLBACK_FILTER_NAME.
    """
    t0 = time.time()

    counts = counts_df.copy()
    if "GENE ID" in counts.columns:
        counts_indexed = counts.set_index("GENE ID")
    else:
        counts_indexed = counts.copy()

    counts_indexed.index = counts_indexed.index.astype(str)

    # Align metadata
    sample_col = "COUNT" if "COUNT" in meta_df.columns else meta_df.columns[0]
    status_col = "STATUS" if "STATUS" in meta_df.columns else meta_df.columns[1]

    meta = meta_df[[sample_col, status_col]].copy()
    meta[sample_col] = meta[sample_col].astype(str)

    # Find the two groups
    groups = meta[status_col].value_counts().index.tolist()
    group_ctrl, group_pd = groups[0], groups[1]

    samples_ctrl = meta[meta[status_col] == group_ctrl][sample_col].tolist()
    samples_pd = meta[meta[status_col] == group_pd][sample_col].tolist()

    # Subset counts to available samples
    avail_ctrl = [s for s in samples_ctrl if s in counts_indexed.columns]
    avail_pd = [s for s in samples_pd if s in counts_indexed.columns]

    if len(avail_ctrl) < 2 or len(avail_pd) < 2:
        raise ValueError(
            f"Insufficient matching samples between counts and metadata: "
            f"Group '{group_ctrl}' has {len(avail_ctrl)} matched, Group '{group_pd}' has {len(avail_pd)} matched. "
            f"At least 2 samples per group are required."
        )

    mat_ctrl = counts_indexed[avail_ctrl].values.astype(float)
    mat_pd = counts_indexed[avail_pd].values.astype(float)

    # Calculate library sizes from all genes in the input counts
    lib_ctrl = mat_ctrl.sum(axis=0, keepdims=True)
    lib_pd = mat_pd.sum(axis=0, keepdims=True)

    # Avoid zero division
    lib_ctrl = np.where(lib_ctrl == 0, 1.0, lib_ctrl)
    lib_pd = np.where(lib_pd == 0, 1.0, lib_pd)

    log_cpm_ctrl = np.log2((mat_ctrl / lib_ctrl) * 1e6 + 1)
    log_cpm_pd = np.log2((mat_pd / lib_pd) * 1e6 + 1)

    # Vectorized Welch's t-test (unequal variances)
    t_stat, pvals = stats.ttest_ind(log_cpm_pd, log_cpm_ctrl, axis=1, equal_var=False)
    log2fc = log_cpm_pd.mean(axis=1) - log_cpm_ctrl.mean(axis=1)

    clean_pvals = np.nan_to_num(pvals, nan=1.0)
    _, padj, _, _ = multipletests(clean_pvals, method="fdr_bh")

    genes = counts_indexed.index.tolist()
    symbols = [id_to_symbol.get(str(g), str(g)) if id_to_symbol else str(g) for g in genes]

    full_results = pd.DataFrame({
        "ENTREZID": genes,
        "SYMBOL": symbols,
        "GENENAME": [f"Gene {s}" for s in symbols],
        "logFC": log2fc,
        "P.Value": clean_pvals,
        "adj.P.Val": padj,
        "Method": FALLBACK_FILTER_NAME
    })

    # Significance filtering
    p_metric = "adj.P.Val" if use_padj else "P.Value"
    sig_mask = (full_results[p_metric] < p_thresh) & (full_results["logFC"].abs() > lfc_thresh)
    sig_df = full_results[sig_mask].copy().sort_values(p_metric).reset_index(drop=True)

    dt = time.time() - t0
    return sig_df, full_results, dt
