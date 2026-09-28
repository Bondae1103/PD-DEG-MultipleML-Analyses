import os
import sys
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Sources
DOWNLOADS_DIR = r"C:\Users\Anoop\Downloads"
GSE68719_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "GSE68719"))
GSE136666_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "GSE136666"))
GSE168496_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "GSE168496"))

def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")

counts_path = find_first_existing([
    os.path.join(DOWNLOADS_DIR, "PD-COUNT-FILE.xlsx"),
    os.path.join(GSE68719_DIR, "wholegene.xlsx")
])

sample_path = find_first_existing([
    os.path.join(DOWNLOADS_DIR, "PD-SAMPLE-FILE.xlsx"),
    os.path.join(GSE68719_DIR, "metadata.xlsx")
])

deg_complete_path = find_first_existing([
    os.path.join(GSE68719_DIR, "DEG_results_complete.xlsx"),
    os.path.join(GSE68719_DIR, "PD_DEG-1.xlsx"),
    os.path.join(DOWNLOADS_DIR, "PD_DEG.xlsx")
])

panel_csv_path = find_first_existing([
    os.path.join(GSE68719_DIR, "ML analysis", "filtered", "ML_final_gene_panel.csv")
])

print(f"Loading counts from: {counts_path}")
counts = pd.read_excel(counts_path)
if "GeneID" in counts.columns:
    counts = counts.rename(columns={"GeneID": "GENE ID"})

# Ensure 72 samples (exclude outlier T.20 if present)
if "T.20" in counts.columns:
    counts = counts.drop(columns=["T.20"])

print(f"Loading samples from: {sample_path}")
sample = pd.read_excel(sample_path)
if "COUNT" in sample.columns:
    sample = sample.rename(columns={"COUNT": "SampleID"})
elif "Sample" in sample.columns:
    sample = sample.rename(columns={"Sample": "SampleID"})
if "Disease_state" in sample.columns and "STATUS" not in sample.columns:
    sample["STATUS"] = sample["Disease_state"].map(lambda x: "PD" if "Parkinson" in str(x) else "NO_PD")
sample = sample[sample["SampleID"] != "T.20"].reset_index(drop=True)

# Load Panel
panel_df = pd.read_csv(panel_csv_path)
entrez_to_symbol = dict(zip(panel_df["ENTREZID"].astype(int), panel_df["SYMBOL"]))
symbol_to_entrez = {v: k for k, v in entrez_to_symbol.items()}
panel_symbols = panel_df["SYMBOL"].tolist()
panel_entrez = panel_df["ENTREZID"].astype(int).tolist()

print(f"Validated Panel: {len(panel_symbols)} genes: {panel_symbols}")

# Preprocessing: CPM + log2
counts_indexed = counts.set_index("GENE ID")
lib_sizes = counts_indexed.sum(axis=0)

available_entrez = [e for e in panel_entrez if e in counts_indexed.index]
counts_panel = counts_indexed.loc[available_entrez]

cpm = counts_panel.divide(lib_sizes, axis=1) * 1e6
log_cpm = np.log2(cpm + 1.0)

# Matrix: samples x genes (using symbols as column names)
expr_mat = log_cpm.T
expr_mat.index.name = "SampleID"
expr_mat = expr_mat.rename(columns=entrez_to_symbol)
expr_mat = expr_mat[panel_symbols].astype(np.float32)

# Merge metadata
df_ml = expr_mat.merge(sample[["SampleID", "STATUS"]], on="SampleID", how="inner").set_index("SampleID")
X = df_ml[panel_symbols]
y_raw = df_ml["STATUS"]

le = LabelEncoder()
y = le.fit_transform(y_raw)  # 0: NO_PD, 1: PD
class_map = {int(i): str(c) for i, c in enumerate(le.classes_)}
print(f"Classes: {class_map}, bincount: {np.bincount(y)}")

# Train XGBoost model
neg, pos = np.bincount(y)
scale_pos_weight = float(neg / pos)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=123,
    n_jobs=-1
)

# Cross-validation validation check
loo = LeaveOneOut()
acc_scores = cross_val_score(xgb, X, y, cv=loo, scoring="accuracy")
y_proba_loo = cross_val_predict(xgb, X, y, cv=loo, method="predict_proba")[:, 1]
loocv_auc = float(roc_auc_score(y, y_proba_loo))
loocv_acc = float(acc_scores.mean())
print(f"XGBoost LOOCV Accuracy: {loocv_acc:.4f} | AUC: {loocv_auc:.4f}")

# Fit on all 72 samples and export native JSON
xgb.fit(X, y)
model_path = os.path.join(ARTIFACTS_DIR, "model.json")
xgb.get_booster().save_model(model_path)
print(f"Saved model to: {model_path}")

# Feature list
features_path = os.path.join(ARTIFACTS_DIR, "feature_list.json")
with open(features_path, "w", encoding="utf-8") as f:
    json.dump(panel_symbols, f, indent=2)
print(f"Saved feature list to: {features_path}")

# Preprocessing parameters (training medians and symbol map)
train_medians = X.median().to_dict()
preproc_data = {
    "normalization": "log2(CPM + 1)",
    "pseudocount": 1.0,
    "feature_names": panel_symbols,
    "feature_medians": train_medians,
    "entrez_to_symbol": {str(k): v for k, v in entrez_to_symbol.items()},
    "symbol_to_entrez": {k: int(v) for k, v in symbol_to_entrez.items()}
}
preproc_path = os.path.join(ARTIFACTS_DIR, "preprocessing.json")
with open(preproc_path, "w", encoding="utf-8") as f:
    json.dump(preproc_data, f, indent=2)
print(f"Saved preprocessing config to: {preproc_path}")

# Export DEG results table
print(f"Loading complete DEG table from: {deg_complete_path}")
deg_raw = pd.read_excel(deg_complete_path)
col_rename = {
    "SYMBOL": "gene",
    "logFC": "log2fc",
    "P.Value": "pvalue",
    "adj.P.Val": "padj",
    "AveExpr": "mean_expr"
}
deg_clean = deg_raw.rename(columns=col_rename)
keep_cols = ["gene", "log2fc", "pvalue", "padj"]
if "mean_expr" in deg_clean.columns:
    keep_cols.append("mean_expr")
deg_export = deg_clean[keep_cols].dropna(subset=["gene", "pvalue", "log2fc"]).copy()
deg_export["gene"] = deg_export["gene"].astype(str)
# Remove duplicate genes by keeping lowest padj
deg_export = deg_export.sort_values("padj").drop_duplicates(subset=["gene"]).sort_values("padj")
deg_path = os.path.join(ARTIFACTS_DIR, "deg_results.csv")
deg_export.to_csv(deg_path, index=False)
print(f"Saved DEG results ({len(deg_export)} genes) to: {deg_path}")

# Export Cohort GSE68719
# Include model panel genes + top 500 significant DEG genes for interactive analysis
top_deg_genes = deg_export.head(500)["gene"].tolist()
union_genes = list(dict.fromkeys(panel_symbols + top_deg_genes))

# Map any available genes in counts
full_counts_entrez = counts_indexed.index.tolist()
# Let's map entrez in deg_raw
if "ENTREZID" in deg_raw.columns:
    all_entrez_map = dict(zip(deg_raw["ENTREZID"].dropna().astype(int), deg_raw["SYMBOL"].dropna().astype(str)))
else:
    all_entrez_map = {}
all_entrez_map.update(entrez_to_symbol)
symbol_to_all_entrez = {v: k for k, v in all_entrez_map.items()}

export_entrez_ids = [symbol_to_all_entrez[g] for g in union_genes if g in symbol_to_all_entrez and symbol_to_all_entrez[g] in counts_indexed.index]
counts_export = counts_indexed.loc[export_entrez_ids]
cpm_export = counts_export.divide(lib_sizes, axis=1) * 1e6
log_cpm_export = np.log2(cpm_export + 1.0).T
log_cpm_export.index.name = "sample_id"
log_cpm_export = log_cpm_export.rename(columns=all_entrez_map)
log_cpm_export = log_cpm_export.astype(np.float32)

gse68719_mat_path = os.path.join(ARTIFACTS_DIR, "cohort_gse68719_normalized.parquet")
log_cpm_export.to_parquet(gse68719_mat_path)
print(f"Saved GSE68719 matrix {log_cpm_export.shape} to: {gse68719_mat_path}")

# GSE68719 Metadata
gse68719_meta = pd.DataFrame({
    "sample_id": df_ml.index,
    "group": df_ml["STATUS"],
    "split": "train",
    "held_out": False
})
gse68719_meta_path = os.path.join(ARTIFACTS_DIR, "cohort_gse68719_meta.csv")
gse68719_meta.to_csv(gse68719_meta_path, index=False)
print(f"Saved GSE68719 meta to: {gse68719_meta_path}")

# Process External Cohort 1: GSE136666
try:
    c136_counts = pd.read_excel(os.path.join(GSE136666_DIR, "wholeGene.xlsx"))
    c136_meta = pd.read_excel(os.path.join(GSE136666_DIR, "metadata.xlsx"))
    c136_counts = c136_counts.set_index("GeneID")
    c136_lib = c136_counts.sum(axis=0)
    avail_136 = [e for e in export_entrez_ids if e in c136_counts.index]
    c136_cpm = c136_counts.loc[avail_136].divide(c136_lib, axis=1) * 1e6
    c136_log = np.log2(c136_cpm + 1.0).T
    c136_log.index.name = "sample_id"
    c136_log = c136_log.rename(columns=all_entrez_map).astype(np.float32)
    c136_mat_path = os.path.join(ARTIFACTS_DIR, "cohort_gse136666_normalized.parquet")
    c136_log.to_parquet(c136_mat_path)

    c136_meta_df = pd.DataFrame({
        "sample_id": c136_meta["GEO_Accession (exp)"],
        "group": c136_meta["Disease_state"].map(lambda x: "PD" if "Parkinson" in str(x) else "NO_PD"),
        "split": "test",
        "held_out": True
    })
    c136_meta_path = os.path.join(ARTIFACTS_DIR, "cohort_gse136666_meta.csv")
    c136_meta_df.to_csv(c136_meta_path, index=False)
    print(f"Saved GSE136666 ({len(c136_meta_df)} samples) to: {c136_mat_path}")
except Exception as e:
    print(f"Warning processing GSE136666: {e}")

# Process External Cohort 2: GSE168496
try:
    c168_counts = pd.read_excel(os.path.join(GSE168496_DIR, "wholegene.xlsx"))
    c168_meta = pd.read_excel(os.path.join(GSE168496_DIR, "sample_metadata.xlsx"))
    c168_counts = c168_counts.set_index("GeneID")
    c168_lib = c168_counts.sum(axis=0)
    avail_168 = [e for e in export_entrez_ids if e in c168_counts.index]
    c168_cpm = c168_counts.loc[avail_168].divide(c168_lib, axis=1) * 1e6
    c168_log = np.log2(c168_cpm + 1.0).T
    c168_log.index.name = "sample_id"
    c168_log = c168_log.rename(columns=all_entrez_map).astype(np.float32)
    c168_mat_path = os.path.join(ARTIFACTS_DIR, "cohort_gse168496_normalized.parquet")
    c168_log.to_parquet(c168_mat_path)

    c168_meta_df = pd.DataFrame({
        "sample_id": c168_meta["Sample"],
        "group": c168_meta["Disease_state"].map(lambda x: "PD" if "Parkinson" in str(x) else "NO_PD"),
        "split": "test",
        "held_out": True
    })
    c168_meta_path = os.path.join(ARTIFACTS_DIR, "cohort_gse168496_meta.csv")
    c168_meta_df.to_csv(c168_meta_path, index=False)
    print(f"Saved GSE168496 ({len(c168_meta_df)} samples) to: {c168_mat_path}")
except Exception as e:
    print(f"Warning processing GSE168496: {e}")

# Metadata JSON
metadata = {
    "pipeline_name": "PD-DEG-MultipleML-Analyses",
    "created_at": "2026-09-28T22:30:00Z",
    "random_seed": 123,
    "class_map": class_map,
    "positive_class": "PD",
    "negative_class": "NO_PD",
    "primary_panel_features": panel_symbols,
    "primary_panel_count": len(panel_symbols),
    "model_type": "XGBClassifier",
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 123
    },
    "metrics": {
        "loocv_accuracy": loocv_acc,
        "loocv_auc": loocv_auc,
        "notebook_rf_loocv_accuracy": 0.8611,
        "notebook_rf_loocv_auc": 0.9115
    },
    "cohorts": {
        "GSE68719": {
            "display_name": "GSE68719 (Discovery / Prefrontal Cortex)",
            "geo_accession": "GSE68719",
            "tissue": "Human Postmortem Frontal Cortex (BA9)",
            "platform": "Illumina HiSeq 2000",
            "n_samples": 72,
            "group_counts": {"NO_PD": 44, "PD": 28},
            "used_in_training": True,
            "contrast": "Parkinsons disease vs Control (log2FC > 0 is elevated in PD)"
        },
        "GSE136666": {
            "display_name": "GSE136666 (External Validation)",
            "geo_accession": "GSE136666",
            "tissue": "Human Postmortem Brain Tissue",
            "platform": "Illumina NextSeq 500",
            "n_samples": 16,
            "group_counts": {"NO_PD": 8, "PD": 8},
            "used_in_training": False,
            "contrast": "Parkinsons disease vs Control"
        },
        "GSE168496": {
            "display_name": "GSE168496 (External Validation)",
            "geo_accession": "GSE168496",
            "tissue": "Human Postmortem Brain Tissue",
            "platform": "Illumina NovaSeq 6000",
            "n_samples": 16,
            "group_counts": {"NO_PD": 8, "PD": 8},
            "used_in_training": False,
            "contrast": "Parkinsons disease vs Control"
        }
    }
}

meta_json_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
with open(meta_json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata JSON to: {meta_json_path}")
print("=== ARTIFACT EXPORT COMPLETE ===")
