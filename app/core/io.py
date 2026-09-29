"""
app/core/io.py
--------------
File ingestion and validation for custom DEG lists and optional raw counts/metadata.
Follows Plan v2 Section 4.1. Contains zero Streamlit imports.
"""

import os
import io
import pandas as pd
from typing import Tuple, Union, BinaryIO

class UserInputError(Exception):
    """Actionable validation error raised for malformed or missing user uploads."""
    pass

def load_deg_list(file_or_path: Union[str, bytes, BinaryIO, io.BytesIO, io.StringIO], max_rows: int = 5000) -> pd.DataFrame:
    """
    Load and validate a candidate DEG table from CSV or Excel.
    Requires ENTREZID, SYMBOL, and GENENAME columns (case-insensitive).
    """
    try:
        if isinstance(file_or_path, str):
            if file_or_path.endswith(".xlsx") or file_or_path.endswith(".xls"):
                df = pd.read_excel(file_or_path)
            else:
                # Sniff CSV vs TSV
                df = pd.read_csv(file_or_path, sep=None, engine="python")
        else:
            # File-like or bytes object
            # Check filename attribute if available
            filename = getattr(file_or_path, "name", "").lower()
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file_or_path)
            else:
                try:
                    df = pd.read_csv(file_or_path, sep=None, engine="python")
                except Exception:
                    if hasattr(file_or_path, "seek"):
                        file_or_path.seek(0)
                    df = pd.read_excel(file_or_path)
    except Exception as e:
        raise UserInputError(f"Could not parse DEG file: {str(e)}. Please ensure it is a valid CSV or Excel file.")

    if df.empty:
        raise UserInputError("Uploaded DEG table is completely empty.")

    # Standardize column names (case-insensitive match)
    col_mapping = {}
    required_cols = {"ENTREZID", "SYMBOL", "GENENAME"}
    found_cols = set()

    for col in df.columns:
        norm = str(col).strip().upper()
        if norm in required_cols:
            col_mapping[col] = norm
            found_cols.add(norm)
        elif norm in ("ENTREZ", "ENTREZ_ID", "ENTREZID"):
            col_mapping[col] = "ENTREZID"
            found_cols.add("ENTREZID")
        elif norm in ("GENE_SYMBOL", "GENESYMBOL", "SYMBOL"):
            col_mapping[col] = "SYMBOL"
            found_cols.add("SYMBOL")
        elif norm in ("GENE_NAME", "GENENAME", "NAME", "DESCRIPTION"):
            col_mapping[col] = "GENENAME"
            found_cols.add("GENENAME")

    missing = required_cols - found_cols
    if missing:
        missing_str = ", ".join(sorted(list(missing)))
        raise UserInputError(
            f"Uploaded DEG list is missing required column(s): {missing_str}. "
            f"The table must include at least: ENTREZID, SYMBOL, and GENENAME."
        )

    df = df.rename(columns=col_mapping)

    # Optional columns normalization if present
    for col in df.columns:
        norm = str(col).strip().upper()
        if norm in ("LOGFC", "LOG2FC", "LOG2_FC"):
            df = df.rename(columns={col: "logFC"})
        elif norm in ("ADJ.P.VAL", "PADJ", "FDR", "QVALUE"):
            df = df.rename(columns={col: "adj.P.Val"})

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df

def load_optional_raw_data(
    counts_file_or_path: Union[str, bytes, BinaryIO],
    meta_file_or_path: Union[str, bytes, BinaryIO],
    gene_id_col: str = None,
    sample_id_col: str = None,
    status_col: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate raw counts matrix and sample metadata for fallback DEG filtering.
    """
    try:
        if isinstance(counts_file_or_path, str) and counts_file_or_path.endswith(".parquet"):
            counts_df = pd.read_parquet(counts_file_or_path)
        elif isinstance(counts_file_or_path, str) and (counts_file_or_path.endswith(".xlsx") or counts_file_or_path.endswith(".xls")):
            counts_df = pd.read_excel(counts_file_or_path)
        else:
            filename = getattr(counts_file_or_path, "name", "").lower()
            if filename.endswith(".parquet"):
                counts_df = pd.read_parquet(counts_file_or_path)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                counts_df = pd.read_excel(counts_file_or_path)
            else:
                counts_df = pd.read_csv(counts_file_or_path, sep=None, engine="python")
    except Exception as e:
        raise UserInputError(f"Could not parse counts matrix: {str(e)}.")

    try:
        if isinstance(meta_file_or_path, str) and (meta_file_or_path.endswith(".xlsx") or meta_file_or_path.endswith(".xls")):
            meta_df = pd.read_excel(meta_file_or_path)
        else:
            filename = getattr(meta_file_or_path, "name", "").lower()
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                meta_df = pd.read_excel(meta_file_or_path)
            else:
                meta_df = pd.read_csv(meta_file_or_path, sep=None, engine="python")
    except Exception as e:
        raise UserInputError(f"Could not parse sample metadata table: {str(e)}.")

    # Validate gene ID column in counts
    if gene_id_col is None:
        possible_gene_cols = ["GENE ID", "GeneID", "gene_id", "ENTREZID", "ID", "gene"]
        for c in possible_gene_cols:
            if c in counts_df.columns:
                gene_id_col = c
                break
        if gene_id_col is None:
            gene_id_col = counts_df.columns[0]

    # Validate metadata columns
    if sample_id_col is None:
        possible_sample_cols = ["COUNT", "SampleID", "Sample", "sample_id", "sample"]
        for c in possible_sample_cols:
            if c in meta_df.columns:
                sample_id_col = c
                break
        if sample_id_col is None:
            sample_id_col = meta_df.columns[0]

    if status_col is None:
        possible_status_cols = ["STATUS", "Disease_state", "group", "condition", "status"]
        for c in possible_status_cols:
            if c in meta_df.columns:
                status_col = c
                break
        if status_col is None:
            if len(meta_df.columns) > 1:
                status_col = meta_df.columns[1]
            else:
                raise UserInputError("Sample metadata must contain at least two columns: Sample ID and Status/Condition.")

    # Rename to standard internal names
    counts_df = counts_df.rename(columns={gene_id_col: "GENE ID"})
    meta_df = meta_df.rename(columns={sample_id_col: "COUNT", status_col: "STATUS"})

    # Ensure >= 2 samples per status group
    group_counts = meta_df["STATUS"].value_counts()
    if len(group_counts) < 2:
        raise UserInputError(f"Metadata must contain at least 2 distinct groups, found only: {list(group_counts.keys())}.")
    if (group_counts < 2).any():
        too_small = group_counts[group_counts < 2].to_dict()
        raise UserInputError(f"Each experimental group must have at least 2 samples for statistical testing. Insufficient samples: {too_small}.")

    return counts_df, meta_df
