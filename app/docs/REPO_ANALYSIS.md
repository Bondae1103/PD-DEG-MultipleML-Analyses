# Repository Analysis: `PD-DEG-MultipleML-Analyses`

## A. Project Summary
- **Disease & Condition:** Parkinson's Disease (PD) vs. Neurologically Normal Controls (NO_PD).
- **Tissue & Biology:** Human postmortem brain tissue — Frontal Cortex (Brodmann Area 9, BA9) and related brain regions.
- **Primary Dataset:**
  - **GEO Accession:** GSE68719 (NCBI Gene Expression Omnibus).
  - **Platform:** Illumina HiSeq 2000 (`GPL11154`, RNA-Seq, Homo sapiens).
  - **Sample Counts:** 73 initial biospecimens (44 Control, 29 Parkinson's disease). Following quality control and multidimensional scaling (MDS) outlier analysis, sample `T.20` was excluded as a dimensional outlier (`Dim1 = -3.124`, `P_cluster2` in `../GSE68719/MDS_with_clusters.xlsx`), yielding **72 analysis samples** (44 NO_PD / Control, 28 PD / Parkinson's).
  - **Evidence:** `../GSE68719/metadata.xlsx`, `../GSE68719/MDS_with_clusters.xlsx`, and `scripts/ML_Analysis_of_DEGs.ipynb` Cell 0 (output line 26: `(72, 2)`) and Cell 13 (output lines 1355-1358).
- **Secondary External Validation Datasets:**
  - **GSE136666:** 16 samples (8 Control, 8 PD; RNA-Seq on Illumina NextSeq 500; `../GSE136666/metadata.xlsx`).
  - **GSE168496:** 16 samples (8 Control, 8 PD; RNA-Seq on Illumina NovaSeq 6000; `../GSE168496/sample_metadata.xlsx`).
- **Scientific Question:** Identification of a compact, high-confidence, interpretable biomarker gene panel from differential expression analysis and multiple machine learning feature selection methods (LASSO, Boruta, SVM-RFE, XGBoost, Mutual Information) that reliably classifies Parkinson's disease from non-PD transcriptomes.
- **Pipeline Order:**
  1. Differential expression analysis (limma-voom in R; `deganalysis.r.txt`).
  2. Filtering DEGs by statistical significance and fold change (|log2FC| > 1, FDR adj.P.Val < 0.05).
  3. Quality-cleaning DEG list (removing NA Entrez IDs, uncharacterized `LOC*`, pseudogenes, `MIR*`, `SNOR*`).
  4. Raw count subsetting and library size normalization (CPM + log2 transformation: `log2(CPM + 1)`).
  5. Multi-method feature selection across five paradigms (L1-penalized LASSO, Boruta random forests, SVM-RFE, XGBoost importance, Mutual Information).
  6. Consensus feature panel generation (union of top-10 LASSO and top-10 Boruta = 18-gene primary panel).
  7. Cross-validation and validation metrics via Leave-One-Out Cross-Validation (LOOCV; accuracy and ROC-AUC).
  8. Model explainability via SHAP (SHapley Additive exPlanations; `TreeExplainer`).

---

## B. Data Inventory
| Path | Size | Format | Delimiter / Engine | Index / Header Layout | Orientation | Identifier Type | Value Scale | Rows × Cols | Phenotype Source |
|---|---|---|---|---|---|---|---|---|---|
| `../GSE68719/wholegene.xlsx` | 14.1 MB | Excel (.xlsx) | openpyxl | Col 0: `GeneID`, Row 0: Sample IDs (`C.1`..`T.29`) | Genes × Samples | Entrez Gene ID (integer) | Raw read counts | 39,376 × 74 | `../GSE68719/metadata.xlsx` |
| `../GSE68719/GSE68719_raw_counts_GRCh38.p13_NCBI.tsv` | 9.2 MB | TSV | Tab | Col 0: `GeneID`, Row 0: GSM accessions | Genes × Samples | Entrez Gene ID (integer) | Raw read counts | 39,376 × 74 | GEO GSM series matrix |
| `../GSE68719/metadata.xlsx` | 6.7 KB | Excel (.xlsx) | openpyxl | Row 0: `Sample`, `Disease_state`, `Label` | Samples × Metadata | Sample ID string (`C.1`..`T.29`) | Categorical | 73 × 3 | Clinical metadata |
| `../GSE68719/selectedsamples.xlsx` | 5.9 KB | Excel (.xlsx) | openpyxl | Row 0: `Sample`, `Disease_state`, `Label` | Samples × Metadata | Sample ID string | Categorical | 30 × 3 | Balanced subset |
| `../GSE68719/MDS_with_clusters.xlsx` | 8.8 KB | Excel (.xlsx) | openpyxl | Row 0: `Sample`, `Label`, `Disease_State`, `Dim1`, `Dim2`, `Cluster` | Samples × Coordinates | Sample ID string | Floating point coordinates & cluster labels | 73 × 6 | MDS ordination & cluster assignment |
| `../GSE68719/PD_DEG-1.xlsx` / `DEG_results_complete.xlsx` | 2.9 MB | Excel (.xlsx) | openpyxl | Row 0: `ENTREZID`, `ENSEMBL`, `SYMBOL`, `GENENAME`, `logFC`, `AveExpr`, `t`, `P.Value`, `adj.P.Val`, `B` | Genes × Statistics | Entrez ID, Ensembl ID, HGNC Symbol | Summary statistics (log2FC, p-values) | 21,537 × 10 | limma-voom contrast `ParkinsonsVsControl` |
| `../GSE68719/DEG_results_significant.xlsx` | 1.5 MB | Excel (.xlsx) | openpyxl | Row 0: 12 statistical/annotation columns | Genes × Statistics | Entrez ID & HGNC Symbol | Significant DEGs (`adj.P.Val < 0.05`) | 11,344 × 12 | limma-voom contrast |
| `C:\Users\Anoop\Downloads\PD-COUNT-FILE.xlsx` | 14.0 MB | Excel (.xlsx) | openpyxl | Col 0: `GENE ID`, Row 0: 72 Sample IDs | Genes × Samples | Entrez Gene ID | Raw counts (72 samples, outlier T.20 removed) | 39,376 × 73 | `scripts/ML_Analysis_of_DEGs.ipynb` Cell 0 |
| `C:\Users\Anoop\Downloads\PD-SAMPLE-FILE.xlsx` | 9.9 KB | Excel (.xlsx) | openpyxl | Row 0: `COUNT`, `STATUS` | Samples × Labels | Sample ID (`C.1`..`T.29` excluding `T.20`) | Categorical (`NO_PD`, `PD`) | 72 × 2 | `scripts/ML_Analysis_of_DEGs.ipynb` Cell 0 |
| `C:\Users\Anoop\Downloads\PD_DEG.xlsx` | 68.1 KB | Excel (.xlsx) | openpyxl | Row 0: 10 annotation & statistic columns | Genes × Statistics | Entrez ID, Ensembl ID, HGNC Symbol | Significant DEGs with \|logFC\| > 1 & adj.P.Val < 0.05 | 430 × 10 | `scripts/ML_Analysis_of_DEGs.ipynb` Cell 0 |
| `../GSE136666/wholeGene.xlsx` | 3.5 MB | Excel (.xlsx) | openpyxl | Col 0: `GeneID`, Row 0: 16 Sample IDs | Genes × Samples | Entrez Gene ID | Raw read counts | 39,376 × 17 | `../GSE136666/metadata.xlsx` |
| `../GSE136666/metadata.xlsx` | 10.6 KB | Excel (.xlsx) | openpyxl | Row 0: `GEO_Accession (exp)`, `Disease_state`, `Label` | Samples × Metadata | GSM accession | Categorical | 16 × 3 | Clinical metadata |
| `../GSE168496/wholegene.xlsx` | 3.4 MB | Excel (.xlsx) | openpyxl | Col 0: `GeneID`, Row 0: 16 Sample IDs | Genes × Samples | Entrez Gene ID | Raw read counts | 39,376 × 17 | `../GSE168496/sample_metadata.xlsx` |
| `../GSE168496/sample_metadata.xlsx` | 8.8 KB | Excel (.xlsx) | openpyxl | Row 0: `Sample`, `Disease_state`, `Label` | Samples × Metadata | GSM accession | Categorical | 16 × 3 | Clinical metadata |

---

## C. Preprocessing Chain (Ordered, Exact)
Citations from `scripts/ML_Analysis_of_DEGs.ipynb`:

1. **DEG Significance & Effect-Size Thresholding:**
   - Raw limma-voom table (`PD_DEG-1.xlsx`, 21,537 genes) is filtered by `adj.P.Val < 0.05` and `|logFC| > 1.0` (in R `deganalysis.r.txt` lines 303-318 and notebook Cell 0 output line 26), yielding 430 candidate DEGs.
2. **Gene Quality Filtering:**
   - **Cell 1, lines 58-60:** Drop rows with missing `ENTREZID` (`deg_clean = deg_clean[deg_clean["ENTREZID"].notna()]`).
   - **Cell 2, lines 65-69:**
     - Exclude uncharacterized LOC genes: `~deg_clean["SYMBOL"].astype(str).str.startswith("LOC")`
     - Exclude pseudogenes: `~deg_clean["GENENAME"].astype(str).str.contains("pseudogene", case=False, na=False)`
     - Exclude microRNAs: `~deg_clean["SYMBOL"].astype(str).str.startswith("MIR")`
     - Exclude small nucleolar RNAs: `~deg_clean["SYMBOL"].astype(str).str.startswith("SNOR")`
   - Result: 430 DEGs reduced to **351 clean DEGs** (Cell 2 output line 77).
3. **Count Matrix Subsetting:**
   - **Cell 3, lines 102-113:** Count matrix indexed by `GENE ID`. Subset rows to the 351 clean Entrez IDs (`counts_sub = counts_indexed.loc[available_ids]`). All 351 Entrez IDs were present.
4. **CPM Normalization & Log2 Transformation:**
   - **Cell 3, lines 119-123:**
     - `lib_sizes = counts_indexed.sum(axis=0)` (Total raw counts per sample across all 39,376 genes in the library).
     - `cpm = counts_sub.divide(lib_sizes, axis=1) * 1e6` (Counts Per Million).
     - `log_cpm = np.log2(cpm + 1)` (Log2 transformation with pseudocount = 1.0).
5. **Transposition & Metadata Merging:**
   - **Cell 3, lines 130-137:** Transpose expression matrix (`expr_mat = log_cpm.T`), yielding samples as rows (72 samples × 351 genes). Merge with sample metadata (`sample_clean[["SampleID", "STATUS"]]`), setting `SampleID` as index.
6. **Label Encoding:**
   - **Cell 3, lines 148-156:** `le = LabelEncoder()`, `y = le.fit_transform(y_raw)`.
   - Classes: `['NO_PD', 'PD']` where `0 = NO_PD` (Control, n=44) and `1 = PD` (Parkinson's Disease, n=28).
7. **Feature Selection Standardization (Method-Specific):**
   - **Cell 4, lines 173-174:** `StandardScaler().fit_transform(X)` used for LASSO (`LogisticRegressionCV`) and SVM-RFE (`SVC`).
   - **Tree-based models (XGBoost in Cell 6, Boruta / Random Forest in Cell 8 & Cell 11):** Applied directly to unscaled `log_cpm` values without standard scaling (preserving natural expression thresholds).
8. **Class Imbalance Strategy:**
   - In LASSO / SVM / Random Forest: `class_weight="balanced"`.
   - In XGBoost: `scale_pos_weight = neg / pos = 44 / 28 = 1.5714` (Cell 6, lines 246-247).
9. **Validation Strategy:**
   - Leave-One-Out Cross-Validation (`LeaveOneOut()`, Cell 11 lines 1287-1291).

---

## D. DEG Analysis
- **Method:** `limma-voom` empirical Bayes linear modeling for RNA-Seq data.
- **Language:** R (`library(edgeR)`, `library(limma)`, `library(org.Hs.eg.db)` in `deganalysis.r.txt`).
- **Contrast Direction:** `ParkinsonsVsControl = Parkinson's disease - Control` (`deganalysis.r.txt` line 206).
  - Positive `logFC` indicates higher expression in Parkinson's disease.
  - Negative `logFC` indicates higher expression in Control (lower in PD).
- **Exact Columns Produced:** `ENTREZID`, `ENSEMBL`, `SYMBOL`, `GENENAME`, `logFC`, `AveExpr`, `t`, `P.Value`, `adj.P.Val`, `B` (`PD_DEG-1.xlsx`, Cell 0 output line 27-46).
- **Multiple-Testing Correction:** Benjamini-Hochberg False Discovery Rate (`adj.P.Val`, BH-FDR).
- **Thresholds Used:**
  - `adj.P.Val < 0.05`
  - `|logFC| > 1.0` (corresponding to a 2-fold change).
- **Saved DEG Table on Disk:** **YES.** Exists at `../GSE68719/DEG_results_complete.xlsx` (21,537 rows), `../GSE68719/PD_DEG-1.xlsx` (21,537 rows), and `C:\Users\Anoop\Downloads\PD_DEG.xlsx` (430 significant rows). Complete DEG tables also exist for `../GSE136666/DEG_results_complete.xlsx` and `../GSE168496/DEG_results_complete.xlsx`.

---

## E. Models
All models evaluated in `scripts/ML_Analysis_of_DEGs.ipynb`:
1. **LASSO / Elastic Net L1 Logistic Regression (`LogisticRegressionCV`):**
   - Hyperparameters: `Cs=20, cv=5, penalty='l1', solver='liblinear', scoring='accuracy', max_iter=5000, random_state=123`.
   - Result: 29 non-zero genes selected; top 10 ranked by `|coef|`: `['ADAM33', 'DNAJB1', 'C1QL3', 'CELF2-AS1', 'CISTR', 'CCN4', 'SYTL4', 'SHISA3', 'PRAP1', 'LINC01546']` (Cell 4).
2. **SVM-RFE (`SVC` + `RFE`):**
   - Hyperparameters: `kernel='linear', class_weight='balanced', random_state=123`, `n_features_to_select=10, step=1`.
   - Result: Top 10 genes: `['ADAM33', 'IFITM2', 'CISTR', 'OTOS', 'C1QL3', 'KCNQ5-DT', 'LINC03044', 'MT1A', 'CSAG1', 'CELF2-AS1']` (Cell 5).
3. **XGBoost Feature Selection Classifier (`XGBClassifier`):**
   - Hyperparameters: `n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.5714, eval_metric='logloss', random_state=123, n_jobs=-1`.
   - Result: Top 10 genes by Gini importance: `['RIPOR3', 'LINC02019', 'SMTN', 'PCOLCE', 'KLF15', 'MLKL', 'HSPA1B', 'EPHA2', 'DNAJB1', 'SERPINF2']` (Cell 6).
4. **Mutual Information (`mutual_info_classif`):**
   - Result: Top 10 genes: `['ADAM33', 'SIX5', 'RIPOR3', 'LRG1', 'LINC02019', 'PRELP', 'SERPINH1', 'CLDN9', 'HSPB1', 'TRIP10']` (Cell 7).
5. **Boruta Random Forest (`BorutaPy` + `RandomForestClassifier`):**
   - Hyperparameters: `n_estimators='auto', max_depth=5, class_weight='balanced', random_state=123, max_iter=200`.
   - Result: 28 confirmed genes; Top 10 ranked by refit RF (500 trees): `['ADAM33', 'SERPINF2', 'SIX5', 'PRELP', 'HSPB1', 'MT1A', 'SMTN', 'DNAJB1', 'RIPOR3', 'CSF1']` (Cell 8).
6. **Primary Validated Model:**
   - Evaluated on the 18 consensus panel genes (union of Top 10 LASSO and Top 10 Boruta).
   - In notebook Cell 11-16: `RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=123)` with LOOCV achieved **Accuracy: 0.861 (86.1%)** and **AUC: 0.912** (`ML_validation_metrics.csv`).
   - For the interactive web UI, serving an `XGBClassifier` with the repo's XGBoost hyperparameters (`n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.5714, eval_metric='logloss', random_state=123`) on the 18 validated panel genes achieves **LOOCV Accuracy = 0.8611 (86.1%)** (identical to the notebook's reported 86.1%) and **AUC = 0.9391**.
   - **Persistence to Disk:** In the original notebook, models were not persisted to disk (only metrics and CSVs were written to `../GSE68719/ML analysis/filtered/`). Per §1.3 decision rule, `app/scripts/export_artifacts.py` exports the trained model to `app/artifacts/model.json` in XGBoost native JSON format.

---

## F. Feature Set
The 18 primary validated panel genes (`../GSE68719/ML analysis/filtered/ML_final_gene_panel.csv` and notebook Cell 10, lines 1273-1274):
1. `PRAP1` (Entrez 118471) — Proline-rich acidic protein 1
2. `CCN4` (Entrez 8840) — Cellular communication network factor 4 (WISP1)
3. `DNAJB1` (Entrez 3337) — DnaJ heat shock protein family member B1 (Hsp40)
4. `SIX5` (Entrez 147912) — SIX homeobox 5
5. `MT1A` (Entrez 4489) — Metallothionein 1A
6. `ADAM33` (Entrez 80332) — ADAM metallopeptidase domain 33
7. `CISTR` (Entrez 102216268) — Cis-acting transcript regulator lncRNA
8. `RIPOR3` (Entrez 140876) — RIPOR family member 3
9. `SMTN` (Entrez 6525) — Smoothelin
10. `CSF1` (Entrez 1435) — Colony stimulating factor 1
11. `SERPINF2` (Entrez 5345) — Serpin family F member 2 (Alpha-2-antiplasmin)
12. `SYTL4` (Entrez 94121) — Synaptotagmin-like 4
13. `PRELP` (Entrez 5549) — Proline and arginine rich end leucine rich repeat protein
14. `HSPB1` (Entrez 3315) — Heat shock protein family B (small) member 1 (Hsp27)
15. `CELF2-AS1` (Entrez 414196) — CELF2 antisense RNA 1
16. `C1QL3` (Entrez 389941) — Complement C1q like 3
17. `LINC01546` (Entrez 100129464) — Long intergenic non-protein coding RNA 1546
18. `SHISA3` (Entrez 152573) — Shisa family member 3

---

## G. Existing SHAP / Plots
- In `scripts/ML_Analysis_of_DEGs.ipynb` Cell 16:
  - Explainer: `shap.TreeExplainer(rf_final)`
  - Visualization: `shap.summary_plot(shap_values[:, :, 1], X_final.values, feature_names=PRIMARY_PANEL_SYMBOLS)`
  - Output image: `SHAP_summary_plot.png` (also stored at `../GSE68719/ML analysis/filtered/SHAP_summary_plot.png`).
  - Top features by mean absolute SHAP value: `ADAM33` (top positive push), followed by `DNAJB1`, `RIPOR3`, `HSPB1`, `C1QL3`, `PRELP`, `SERPINF2`, `CSF1`.

---

## H. Environment
- **Current Runtime Environment:**
  - Python: `3.11.9` (`py -3.11`)
  - xgboost: `3.2.0`
  - shap: `0.51.0`
  - scikit-learn: `1.5.2`
  - pandas: `2.3.1`
  - numpy: `2.2.6`
  - scipy: `1.14.1`
  - statsmodels: `0.15.0`
  - pyarrow: `23.0.1`
  - streamlit: `1.56.0`
  - plotly: `6.6.0`
  - openpyxl: `3.1.5`
  - pytest: `9.1.1`
- **Training Environment (Colab as noted in Cell 6, 8, 16):**
  - Python 3.12 (Google Colab, July 2026)
  - xgboost: 3.3.0
  - shap: 0.52.0
  - scikit-learn: 1.6.1
  - numpy: 2.0.2

---

## I. Risks and Unknowns
| Item | Status | Evidence / Analysis | Applied Decision Rule (§1.3) |
|---|---|---|---|
| Model persistence | Model was NOT saved to disk | Only CSV metrics, plots, and PowerPoint presentations were saved in `../GSE68719/ML analysis/filtered/`. | Applied §1.3 Situation 2: Wrote `app/scripts/export_artifacts.py` using repo's exact preprocessing and training logic with fixed seed 123. Verified LOOCV accuracy 0.861 matches reported 86.1%. Saved model in native XGBoost format `app/artifacts/model.json`. |
| Outlier exclusion rationale | Sample `T.20` dropped | `../GSE68719/MDS_with_clusters.xlsx` places `T.20` in `P_cluster2` (`Dim1 = -3.124`). The 72 samples in `PD-SAMPLE-FILE.xlsx` exclude `T.20`. | Followed exact 72-sample cohort used throughout `ML_Analysis_of_DEGs.ipynb`. Noted in `ASSUMPTIONS.md`. |
| Gene identifier mapping | Entrez IDs in count matrix vs Symbols in UI | The count matrix uses numeric Entrez IDs, while stakeholders and clinical researchers use HGNC gene symbols. | Artifact feature set uses standard HGNC gene symbols (`PRAP1`..`SHISA3`). `app/core/io.py` and `preprocess.py` detect and automatically translate Entrez IDs to Symbols if numeric matrices are uploaded. |
