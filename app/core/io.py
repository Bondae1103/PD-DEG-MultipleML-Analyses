import io
import csv
import gzip
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import pandas as pd
import numpy as np

from app.core import UserInputError
from app.config import MAX_UPLOAD_SAMPLES, MIN_FEATURE_COVERAGE

@dataclass
class ValidationReport:
    """Detailed summary of uploaded expression dataset quality and characteristics."""
    n_samples: int
    n_genes: int
    coverage: float
    matched_features: List[str]
    missing_features: List[str]
    duplicate_ids: List[str]
    negative_values: bool
    looks_like_raw_counts: bool
    warnings: List[str] = field(default_factory=list)
    orientation: str = "samples_x_genes"

def load_matrix(file_or_path: Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    Load an expression matrix from CSV, TSV, TXT, GZ, or Parquet.
    Auto-detects delimiter and sets the first column as the index.
    
    Raises:
        UserInputError: For non-numeric matrices, invalid formats, or exceeding sample limits.
    """
    df: Optional[pd.DataFrame] = None
    filename = ""

    if isinstance(file_or_path, str):
        filename = file_or_path.lower()
        if filename.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_or_path)
            except Exception as e:
                raise UserInputError(f"Failed to read Parquet file: {str(e)}")
        else:
            with open(file_or_path, "rb") as f:
                content = f.read()
    else:
        # File-like object (e.g. UploadedFile)
        filename = getattr(file_or_path, "name", "upload.csv").lower()
        if filename.endswith(".parquet"):
            try:
                df = pd.read_parquet(file_or_path)
            except Exception as e:
                raise UserInputError(f"Failed to read Parquet file: {str(e)}")
        else:
            file_or_path.seek(0)
            content = file_or_path.read()

    if df is None:
        # Check gzip compression
        if filename.endswith(".gz") or content[:2] == b"\x1f\x8b":
            try:
                content = gzip.decompress(content)
            except Exception as e:
                raise UserInputError(f"Failed to decompress gzip file: {str(e)}")

        # Delimiter sniffing on first 4 KB
        sample_bytes = content[:4096]
        try:
            sample_text = sample_bytes.decode("utf-8", errors="replace")
            dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t; ")
            sep = dialect.delimiter
        except Exception:
            # Fallback heuristic: comma first, then tab
            sample_str = sample_bytes.decode("utf-8", errors="ignore")
            if "\t" in sample_str and sample_str.count("\t") > sample_str.count(","):
                sep = "\t"
            else:
                sep = ","

        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, index_col=0)
        except Exception as e:
            raise UserInputError(f"Could not parse tabular file with delimiter '{sep}': {str(e)}")

    if df.empty:
        raise UserInputError("The uploaded file contains no data.")

    # Guard against oversized matrices
    if df.shape[0] > MAX_UPLOAD_SAMPLES and df.shape[1] > MAX_UPLOAD_SAMPLES:
        raise UserInputError(
            f"Uploaded matrix has {df.shape[0]} rows and {df.shape[1]} columns, exceeding the maximum allowed {MAX_UPLOAD_SAMPLES} samples."
        )

    # Coerce to numeric
    orig_cells = df.size
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    nan_cells = df_numeric.isna().sum().sum()
    if orig_cells > 0 and (nan_cells / orig_cells) > 0.05:
        raise UserInputError(
            f"More than 5% ({nan_cells}/{orig_cells}) of matrix cells could not be converted to numeric values. Please check that values are numerical expression levels."
        )

    return df_numeric

def detect_orientation(df: pd.DataFrame, feature_list: List[str], id_map: Optional[dict] = None) -> str:
    """
    Determine if expression matrix is samples_x_genes or genes_x_samples.
    Compares overlap with the model feature list.
    
    Returns:
        'samples_x_genes' or 'genes_x_samples'
    Raises:
        UserInputError: If zero target features are matched in either dimension.
    """
    features_clean = {str(f).strip().upper() for f in feature_list}
    if id_map:
        for k, v in id_map.items():
            features_clean.add(str(k).strip().upper())
            features_clean.add(str(v).strip().upper())

    cols_clean = {str(c).strip().upper() for c in df.columns}
    idx_clean = {str(i).strip().upper() for i in df.index}

    col_overlap = len(cols_clean.intersection(features_clean))
    idx_overlap = len(idx_clean.intersection(features_clean))

    if col_overlap == 0 and idx_overlap == 0:
        raise UserInputError(
            "None of the model's biomarker genes were found in the uploaded file. "
            "Please ensure row or column labels match standard HGNC gene symbols (e.g. ADAM33, DNAJB1, RIPOR3)."
        )

    if col_overlap >= idx_overlap:
        return "samples_x_genes"
    else:
        return "genes_x_samples"

def validate_upload(
    df: pd.DataFrame,
    feature_list: List[str],
    id_map: Optional[dict] = None
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Validate, sanitize, orient, and report on uploaded expression data.
    
    Returns:
        (sanitized_samples_x_genes_df, ValidationReport)
    """
    warnings: List[str] = []
    
    # 1. Orientation detection and transposition
    orientation = detect_orientation(df, feature_list, id_map)
    if orientation == "genes_x_samples":
        df = df.T
        warnings.append("Detected genes as rows: matrix was automatically transposed to samples as rows.")

    # 2. Handle duplicate gene columns
    gene_cols = [str(c).strip() for c in df.columns]
    df.columns = gene_cols

    # Map Entrez IDs to Symbols if columns are numeric Entrez IDs
    if id_map:
        renamed_cols = {}
        for c in df.columns:
            if c in id_map:
                renamed_cols[c] = id_map[c]
            elif str(c) in id_map:
                renamed_cols[c] = id_map[str(c)]
        if renamed_cols:
            df = df.rename(columns=renamed_cols)
            warnings.append(f"Mapped {len(renamed_cols)} Entrez IDs to HGNC gene symbols.")

    # Deduplicate genes by keeping the column with highest mean expression
    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        warnings.append(f"Duplicate gene columns found for: {dup_names}. Keeping column with highest average expression.")
        kept_cols = []
        for col_name in df.columns.unique():
            sub = df[[col_name]]
            if sub.shape[1] > 1:
                best_idx = sub.mean().idxmax()
                kept_cols.append(sub[best_idx])
            else:
                kept_cols.append(sub.iloc[:, 0])
        df = pd.concat(kept_cols, axis=1)

    # 3. Deduplicate sample IDs in index
    df.index = [str(idx).strip() for idx in df.index]
    duplicate_samples: List[str] = []
    if df.index.duplicated().any():
        seen = {}
        new_index = []
        for idx in df.index:
            if idx in seen:
                seen[idx] += 1
                duplicate_samples.append(idx)
                new_index.append(f"{idx}_{seen[idx]}")
            else:
                seen[idx] = 1
                new_index.append(idx)
        df.index = new_index
        warnings.append(f"Duplicate sample IDs detected and disambiguated: {list(set(duplicate_samples))}.")

    # 4. Feature coverage
    req_set = set(feature_list)
    present_set = set(df.columns)
    matched = [f for f in feature_list if f in present_set]
    missing = [f for f in feature_list if f not in present_set]
    coverage = len(matched) / len(feature_list) if feature_list else 0.0

    if coverage < MIN_FEATURE_COVERAGE:
        pct = coverage * 100.0
        min_pct = MIN_FEATURE_COVERAGE * 100.0
        raise UserInputError(
            f"Feature coverage is {pct:.1f}%, which is below the required {min_pct:.0f}% threshold ({len(matched)}/{len(feature_list)} genes present). "
            f"Missing required genes: {missing}."
        )

    if missing:
        warnings.append(f"{len(missing)} of {len(feature_list)} model genes missing. Imputing with training medians.")

    # 5. Numerical checks
    vals = df.values.flatten()
    vals_valid = vals[~np.isnan(vals)]
    has_negatives = bool(np.any(vals_valid < 0))
    if has_negatives:
        warnings.append("Matrix contains negative values. Ensure expression data is not heavily z-score centered across non-comparable cohorts.")

    # Check for raw counts heuristic: all integer-valued and max > 1000
    is_integers = bool(np.all(np.equal(np.mod(vals_valid[:1000], 1), 0)))
    max_val = float(np.max(vals_valid)) if len(vals_valid) > 0 else 0.0
    looks_raw = is_integers and max_val > 1000.0
    if looks_raw:
        warnings.append(
            "Values appear to be unnormalized raw read counts (integer-valued, maximum > 1,000). "
            "The model was trained on log2(CPM + 1) normalized expression. Results may be inaccurate unless normalized."
        )

    report = ValidationReport(
        n_samples=df.shape[0],
        n_genes=df.shape[1],
        coverage=coverage,
        matched_features=matched,
        missing_features=missing,
        duplicate_ids=duplicate_samples,
        negative_values=has_negatives,
        looks_like_raw_counts=looks_raw,
        warnings=warnings,
        orientation=orientation
    )

    return df, report
