import os
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from app.config import DEG_PATH, DEFAULT_PVAL_THRESH, DEFAULT_LOG2FC_THRESH
from app.core import UserInputError

def load_deg(path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and normalizes the Differential Expression Gene table.
    Ensures canonical columns: ['gene', 'log2fc', 'pvalue', 'padj', 'neglog10_p'].
    """
    if path is None:
        path = DEG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"DEG table not found at: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)

    # Normalize column names
    col_mapping = {
        "SYMBOL": "gene",
        "Symbol": "gene",
        "Gene": "gene",
        "GeneID": "gene",
        "logFC": "log2fc",
        "Log2FC": "log2fc",
        "log2FoldChange": "log2fc",
        "P.Value": "pvalue",
        "p_val": "pvalue",
        "pvalue": "pvalue",
        "adj.P.Val": "padj",
        "padj": "padj",
        "p_adj": "padj",
        "AveExpr": "mean_expr",
        "baseMean": "mean_expr"
    }

    renamed = {c: col_mapping[c] for c in df.columns if c in col_mapping}
    df = df.rename(columns=renamed)

    required = ["gene", "log2fc", "pvalue"]
    for r in required:
        if r not in df.columns:
            raise UserInputError(f"DEG table missing required column: '{r}'. Present columns: {list(df.columns)}")

    if "padj" not in df.columns:
        # Compute BH-FDR if padj is absent
        _, fdr, _, _ = multipletests(df["pvalue"].fillna(1.0), method="fdr_bh")
        df["padj"] = fdr

    # Clean numeric values
    df["log2fc"] = pd.to_numeric(df["log2fc"], errors="coerce").fillna(0.0)
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce").fillna(1.0)
    df["padj"] = pd.to_numeric(df["padj"], errors="coerce").fillna(1.0)

    # Compute -log10(p-value) avoiding infinity
    min_nonzero_float = np.finfo(np.float64).tiny
    safe_p = np.clip(df["pvalue"].values, min_nonzero_float, 1.0)
    df["neglog10_p"] = -np.log10(safe_p)

    return df

def filter_deg(
    deg: pd.DataFrame,
    p_col: str = "padj",
    p_thresh: float = DEFAULT_PVAL_THRESH,
    lfc_thresh: float = DEFAULT_LOG2FC_THRESH
) -> pd.DataFrame:
    """
    Categorizes genes into 'Up', 'Down', and 'Not significant' based on thresholds.
    
    Category definitions:
        Up: log2fc >= lfc_thresh and p <= p_thresh
        Down: log2fc <= -lfc_thresh and p <= p_thresh
        Not significant: all other genes
    """
    out = deg.copy()
    if p_col not in out.columns:
        p_col = "pvalue"

    p_vals = out[p_col].values
    lfc_vals = out["log2fc"].values

    is_sig = p_vals <= p_thresh
    is_up = is_sig & (lfc_vals >= lfc_thresh)
    is_down = is_sig & (lfc_vals <= -lfc_thresh)

    categories = np.full(len(out), "Not significant", dtype=object)
    categories[is_up] = "Up"
    categories[is_down] = "Down"

    out["category"] = categories
    return out

def compute_deg_fallback(
    expr: pd.DataFrame,
    groups: pd.Series,
    group_a: str,
    group_b: str
) -> pd.DataFrame:
    """
    Computes fast Welch's t-test and BH-FDR differential expression between group_a and group_b.
    Contrast: group_a vs group_b (log2FC = mean(A) - mean(B)).
    
    Args:
        expr: Expression matrix with samples as rows and genes as columns.
        groups: Series mapping sample IDs to group names.
        group_a: Numerator group (e.g. 'PD').
        group_b: Reference group (e.g. 'NO_PD').
    """
    common_samples = expr.index.intersection(groups.index)
    if len(common_samples) == 0:
        raise UserInputError("No matching sample IDs between expression matrix and group labels.")

    sub_expr = expr.loc[common_samples]
    sub_groups = groups.loc[common_samples]

    samples_a = sub_groups[sub_groups == group_a].index
    samples_b = sub_groups[sub_groups == group_b].index

    if len(samples_a) < 2 or len(samples_b) < 2:
        raise UserInputError(
            f"Differential expression requires at least 2 samples per group. "
            f"Found {len(samples_a)} for '{group_a}' and {len(samples_b)} for '{group_b}'."
        )

    mat_a = sub_expr.loc[samples_a].values
    mat_b = sub_expr.loc[samples_b].values

    # Means
    mean_a = np.mean(mat_a, axis=0)
    mean_b = np.mean(mat_b, axis=0)
    log2fc = mean_a - mean_b

    # Vectorized Welch's t-test (equal_var=False)
    t_stat, p_values = stats.ttest_ind(mat_a, mat_b, axis=0, equal_var=False, nan_policy="omit")
    p_values = np.nan_to_num(p_values, nan=1.0)

    # Benjamini-Hochberg FDR
    _, padj, _, _ = multipletests(p_values, method="fdr_bh")

    min_nonzero_float = np.finfo(np.float64).tiny
    safe_p = np.clip(p_values, min_nonzero_float, 1.0)
    neglog10_p = -np.log10(safe_p)

    res = pd.DataFrame({
        "gene": sub_expr.columns,
        "log2fc": log2fc,
        "pvalue": p_values,
        "padj": padj,
        "neglog10_p": neglog10_p,
        "mean_expr": (mean_a + mean_b) / 2.0
    })

    return res.sort_values("pvalue").reset_index(drop=True)
