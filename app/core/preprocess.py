"""
app/core/preprocess.py
----------------------
Preprocessing and feature matrix preparation verbatim from notebook cells 1-3.
Follows Plan v2 Section 4.2. Contains zero Streamlit imports.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from app.core.io import UserInputError
from app.config import MIN_DEG_ROWS_FOR_METHODS

def clean_deg_list(deg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the candidate DEG list by dropping NA Entrez IDs, uncharacterized LOC* genes,
    pseudogenes, microRNAs (MIR*), and small nucleolar RNAs (SNOR*).
    Exact port of Notebook Step 2 (Cells 1-2).
    """
    deg_clean = deg_df.copy()
    
    # 1. Drop NA Entrez IDs
    deg_clean = deg_clean[deg_clean["ENTREZID"].notna()].copy()
    
    # 2. Exclude uncharacterized LOC* loci
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("LOC")]
    
    # 3. Exclude pseudogenes
    deg_clean = deg_clean[~deg_clean["GENENAME"].astype(str).str.contains("pseudogene", case=False, na=False)]
    
    # 4. Exclude microRNAs and small nucleolar RNAs
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("MIR")]
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("SNOR")]
    
    return deg_clean.reset_index(drop=True)

def subset_and_normalize(deg_clean: pd.DataFrame, counts_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Subset counts matrix to cleaned DEG Entrez IDs and normalize to log2(CPM + 1).
    IMPORTANT: Library sizes are calculated from the FULL counts matrix, NOT the subset.
    Exact port of Notebook Step 3-4 (Cell 3).
    """
    counts = counts_df.copy()
    if "GENE ID" in counts.columns:
        counts_indexed = counts.set_index("GENE ID")
    else:
        counts_indexed = counts.copy()

    # Ensure index is string format
    counts_indexed.index = counts_indexed.index.astype(str)

    # Clean Entrez IDs to match
    clean_entrez_ids = deg_clean["ENTREZID"].dropna().astype(int).astype(str).tolist()
    available_ids = [g for g in clean_entrez_ids if g in counts_indexed.index]
    missing_ids = [g for g in clean_entrez_ids if g not in counts_indexed.index]

    if len(available_ids) < MIN_DEG_ROWS_FOR_METHODS:
        raise UserInputError(
            f"After matching with the counts matrix, only {len(available_ids)} candidate DEGs were found "
            f"({len(missing_ids)} missing). At least {MIN_DEG_ROWS_FOR_METHODS} genes are required for "
            f"feature selection algorithms to operate reliably. Please verify that your DEG table uses "
            f"NCBI Entrez IDs that match the RNA-Seq platform."
        )

    counts_sub = counts_indexed.loc[available_ids]

    # Calculate library size across ALL genes in the full counts matrix
    lib_sizes = counts_indexed.sum(axis=0)

    # CPM and log2(cpm + 1)
    cpm = counts_sub.divide(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    # Build Entrez ID -> Gene Symbol dictionary
    id_to_symbol = dict(zip(
        deg_clean["ENTREZID"].dropna().astype(int).astype(str),
        deg_clean["SYMBOL"]
    ))

    return log_cpm, id_to_symbol

def build_X_y(
    log_cpm: pd.DataFrame,
    sample_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, LabelEncoder]:
    """
    Transpose expression matrix to samples x genes, merge with sample metadata,
    and encode target status labels.
    Exact port of Notebook Step 5-6 (Cell 3).
    """
    expr_mat = log_cpm.T
    expr_mat.index.name = "SampleID"
    expr_mat = expr_mat.reset_index()

    sample_clean = sample_df.copy()
    if "COUNT" in sample_clean.columns:
        sample_clean = sample_clean.rename(columns={"COUNT": "SampleID"})
    elif "Sample" in sample_clean.columns:
        sample_clean = sample_clean.rename(columns={"Sample": "SampleID"})

    if "Disease_state" in sample_clean.columns and "STATUS" not in sample_clean.columns:
        sample_clean = sample_clean.rename(columns={"Disease_state": "STATUS"})

    sample_clean["SampleID"] = sample_clean["SampleID"].astype(str)
    expr_mat["SampleID"] = expr_mat["SampleID"].astype(str)

    df_ml = expr_mat.merge(sample_clean[["SampleID", "STATUS"]], on="SampleID", how="inner")
    if df_ml.empty:
        raise UserInputError(
            "None of the sample IDs in the counts matrix matched the sample IDs in the metadata table. "
            "Please ensure sample identifiers (e.g. GSM / C.1 / T.1) match between counts and metadata."
        )

    df_ml = df_ml.set_index("SampleID")

    X = df_ml.drop(columns=["STATUS"])
    y_raw = df_ml["STATUS"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return X, y, y_raw, le
