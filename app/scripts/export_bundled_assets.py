"""
app/scripts/export_bundled_assets.py
------------------------------------
Locates the real GSE68719 files in the repository/parent directory,
prepares the clean bundled assets, and writes app/artifacts/metadata.json.
Follows Plan v2 Section 3.2.
"""

import os
import sys
import json
import platform
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Parent directory for GSE68719
GSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "GSE68719"))
if not os.path.exists(GSE_DIR):
    # Try direct parent
    GSE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "GSE68719"))

if not os.path.exists(GSE_DIR):
    raise FileNotFoundError(f"GSE68719 assets directory not found at {GSE_DIR}")

def main():
    print(f"Reading real GSE68719 source files from: {GSE_DIR}")

    # 1. DEG table (PD_DEG-1.xlsx -> filtered to match PD_DEG.xlsx)
    deg_raw_path = os.path.join(GSE_DIR, "PD_DEG-1.xlsx")
    deg_df = pd.read_excel(deg_raw_path, sheet_name="PD_DEG")
    # Exact filtering used to produce PD_DEG.xlsx: adj.P.Val < 0.05 and |logFC| > 1
    deg_430 = deg_df[(deg_df["adj.P.Val"] < 0.05) & (deg_df["logFC"].abs() > 1)].reset_index(drop=True)
    assert len(deg_430) == 430, f"Expected 430 DEGs, got {len(deg_430)}"
    
    deg_out_path = os.path.join(ARTIFACT_DIR, "sample_deg_list.csv")
    deg_430.to_csv(deg_out_path, index=False)
    print(f"Wrote {deg_out_path} ({deg_430.shape})")

    # 2. Raw counts matrix (wholegene.xlsx -> drop T.20 outlier, rename GeneID -> GENE ID)
    wholegene_path = os.path.join(GSE_DIR, "wholegene.xlsx")
    counts_df = pd.read_excel(wholegene_path)
    if "GeneID" in counts_df.columns:
        counts_df = counts_df.rename(columns={"GeneID": "GENE ID"})
    if "T.20" in counts_df.columns:
        counts_df = counts_df.drop(columns=["T.20"])
    
    assert counts_df.shape == (39376, 73), f"Expected (39376, 73), got {counts_df.shape}"
    
    # Cast sample columns to float32 for compact parquet storage
    sample_cols = [c for c in counts_df.columns if c != "GENE ID"]
    for col in sample_cols:
        counts_df[col] = counts_df[col].astype("float32")
    counts_df["GENE ID"] = counts_df["GENE ID"].astype(str)

    counts_out_path = os.path.join(ARTIFACT_DIR, "counts_matrix.parquet")
    counts_df.to_parquet(counts_out_path, index=False, engine="pyarrow")
    print(f"Wrote {counts_out_path} ({counts_df.shape}, float32)")

    # 3. Sample metadata (metadata.xlsx -> drop T.20, map Disease_state -> STATUS, Sample -> COUNT)
    meta_path = os.path.join(GSE_DIR, "metadata.xlsx")
    meta_df = pd.read_excel(meta_path)
    meta_72 = meta_df[meta_df["Sample"] != "T.20"].copy()
    meta_72 = meta_72.rename(columns={"Sample": "COUNT", "Disease_state": "STATUS"})
    status_map = {"Control": "NO_PD", "Parkinson's disease": "PD"}
    meta_72["STATUS"] = meta_72["STATUS"].map(status_map)
    meta_72 = meta_72[["COUNT", "STATUS"]].reset_index(drop=True)
    assert meta_72.shape == (72, 2), f"Expected (72, 2), got {meta_72.shape}"

    meta_out_path = os.path.join(ARTIFACT_DIR, "sample_metadata.csv")
    meta_72.to_csv(meta_out_path, index=False)
    print(f"Wrote {meta_out_path} ({meta_72.shape})")

    # 4. Provenance metadata.json
    metadata = {
        "geo_accession": "GSE68719",
        "title": "Parkinson's disease postmortem frontal cortex RNA-Seq",
        "tissue": "Human Postmortem Frontal Cortex (BA9)",
        "source_files": {
            "deg_raw": "GSE68719/PD_DEG-1.xlsx [PD_DEG]",
            "counts_raw": "GSE68719/wholegene.xlsx",
            "metadata_raw": "GSE68719/metadata.xlsx"
        },
        "sample_counts": {
            "total_raw": 73,
            "excluded_samples": ["T.20"],
            "exclusion_reason": "MDS outlier (P_cluster2)",
            "total_analyzed": 72,
            "NO_PD": 44,
            "PD": 28
        },
        "gene_counts": {
            "raw_counts_total": 39376,
            "deg_significant_unfiltered": 430,
            "deg_cleaned": 351,
            "cleaning_criteria": "Drop NA Entrez; drop Symbol starting with LOC, MIR, SNOR; drop Genename containing 'pseudogene'"
        },
        "ml_pipeline": {
            "random_seed": 123,
            "methods": ["LASSO", "SVM_RFE", "XGBoost", "MutualInfo", "Boruta"],
            "validation": "RandomForestClassifier(n_estimators=500, class_weight='balanced') with LeaveOneOut CV"
        },
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system()
        }
    }

    meta_json_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {meta_json_path}")

    print("\nBundled asset export completed successfully!")

if __name__ == "__main__":
    main()
