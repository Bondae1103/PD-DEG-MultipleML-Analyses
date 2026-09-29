"""
app/scripts/verify_pipeline.py
------------------------------
Mandatory Phase 0 verification gate script per Plan v2 Section 1.6.
Asserts and prints PASS/FAIL for all 5 checks against the real GSE68719 assets.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from boruta import BorutaPy
import shap
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

def run_checks():
    print("=" * 60)
    print("PHASE 0 PIPELINE VERIFICATION GATE")
    print("=" * 60)

    # Load assets
    deg_path = os.path.join(ARTIFACT_DIR, "sample_deg_list.csv")
    counts_path = os.path.join(ARTIFACT_DIR, "counts_matrix.parquet")
    meta_path = os.path.join(ARTIFACT_DIR, "sample_metadata.csv")

    assert os.path.exists(deg_path), f"Missing {deg_path}"
    assert os.path.exists(counts_path), f"Missing {counts_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    deg = pd.read_csv(deg_path)
    counts = pd.read_parquet(counts_path)
    sample = pd.read_csv(meta_path)

    print(f"Loaded DEG table: shape={deg.shape}")
    print(f"Loaded Counts matrix: shape={counts.shape}")
    print(f"Loaded Sample metadata: shape={sample.shape}")

    # ---------------------------------------------------------
    # Check 1: Cleaning reproduces row reduction (430 -> 351)
    # ---------------------------------------------------------
    deg_clean = deg.copy()
    deg_clean = deg_clean[deg_clean["ENTREZID"].notna()]
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("LOC")]
    deg_clean = deg_clean[~deg_clean["GENENAME"].astype(str).str.contains("pseudogene", case=False, na=False)]
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("MIR")]
    deg_clean = deg_clean[~deg_clean["SYMBOL"].astype(str).str.startswith("SNOR")]

    initial_rows = len(deg)
    cleaned_rows = len(deg_clean)
    check1_pass = (initial_rows == 430 and cleaned_rows == 351)
    status1 = "PASS" if check1_pass else "FAIL"
    print(f"Check 1 [Cleaning row reduction: 430 -> {cleaned_rows} (expected 351)]: {status1}")

    # Build X and y
    counts_indexed = counts.set_index("GENE ID")
    counts_indexed.index = counts_indexed.index.astype(str)
    clean_entrez_ids = deg_clean["ENTREZID"].astype(int).astype(str).tolist()
    available_ids = [g for g in clean_entrez_ids if g in counts_indexed.index]
    counts_sub = counts_indexed.loc[available_ids]

    lib_sizes = counts_indexed.sum(axis=0)
    cpm = counts_sub.divide(lib_sizes, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1)

    expr_mat = log_cpm.T
    expr_mat.index.name = "SampleID"
    expr_mat = expr_mat.reset_index()
    sample_clean = sample.rename(columns={"COUNT": "SampleID"})
    df_ml = expr_mat.merge(sample_clean[["SampleID", "STATUS"]], on="SampleID", how="inner").set_index("SampleID")

    id_to_symbol = dict(zip(deg_clean["ENTREZID"].astype(int).astype(str), deg_clean["SYMBOL"]))
    X = df_ml.drop(columns=["STATUS"])
    y_raw = df_ml["STATUS"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # Check 2: Running all 5 methods & notebook match
    # ---------------------------------------------------------
    # Method 1: LASSO
    lasso_cv = LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="liblinear", scoring="accuracy", max_iter=5000, random_state=123)
    lasso_cv.fit(X_scaled, y)
    lasso_coefs = pd.Series(lasso_cv.coef_[0], index=X.columns)
    top10_lasso = lasso_coefs[lasso_coefs != 0].abs().sort_values(ascending=False).head(10).index.tolist()
    top10_lasso_symbols = [id_to_symbol.get(str(g), str(g)) for g in top10_lasso]

    expected_lasso_symbols = ['ADAM33', 'DNAJB1', 'C1QL3', 'CELF2-AS1', 'CISTR', 'CCN4', 'SYTL4', 'SHISA3', 'PRAP1', 'LINC01546']
    lasso_match = (top10_lasso_symbols == expected_lasso_symbols)

    # Method 2: SVM-RFE
    svm_linear = SVC(kernel="linear", class_weight="balanced", random_state=123)
    rfe = RFE(estimator=svm_linear, n_features_to_select=10, step=1)
    rfe.fit(X_scaled, y)
    selected_svm_rfe = X.columns[rfe.support_].tolist()
    svm_final = SVC(kernel="linear", class_weight="balanced", random_state=123)
    svm_final.fit(X_scaled[:, rfe.support_], y)
    svm_coefs = pd.Series(svm_final.coef_[0], index=selected_svm_rfe)
    top10_svm = svm_coefs.abs().sort_values(ascending=False).head(10).index.tolist()
    top10_svm_symbols = [id_to_symbol.get(str(g), str(g)) for g in top10_svm]
    expected_svm_symbols = ['ADAM33', 'IFITM2', 'CISTR', 'OTOS', 'C1QL3', 'KCNQ5-DT', 'LINC03044', 'MT1A', 'CSAG1', 'CELF2-AS1']
    svm_match = (top10_svm_symbols == expected_svm_symbols)

    # Method 3: XGBoost
    neg, pos = np.bincount(y)
    xgb_clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos, eval_metric="logloss", random_state=123, n_jobs=-1)
    xgb_clf.fit(X, y)
    top10_xgb = pd.Series(xgb_clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).index.tolist()
    top10_xgb_symbols = [id_to_symbol.get(str(g), str(g)) for g in top10_xgb]

    # Method 4: Mutual Info
    mi_scores = mutual_info_classif(X, y, random_state=123)
    top10_mi = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False).head(10).index.tolist()
    top10_mi_symbols = [id_to_symbol.get(str(g), str(g)) for g in top10_mi]
    expected_mi_symbols = ['ADAM33', 'SIX5', 'RIPOR3', 'LRG1', 'LINC02019', 'PRELP', 'SERPINH1', 'CLDN9', 'HSPB1', 'TRIP10']
    mi_match = (top10_mi_symbols == expected_mi_symbols)

    # Method 5: Boruta
    rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=123)
    boruta_sel = BorutaPy(rf_boruta, n_estimators="auto", verbose=0, random_state=123, max_iter=200)
    boruta_sel.fit(X.values, y)
    selected_boruta = X.columns[boruta_sel.support_].tolist()
    rf_rank = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=123)
    rf_rank.fit(X[selected_boruta], y)
    top10_boruta = pd.Series(rf_rank.feature_importances_, index=selected_boruta).sort_values(ascending=False).head(10).index.tolist()
    top10_boruta_symbols = [id_to_symbol.get(str(g), str(g)) for g in top10_boruta]
    expected_boruta_symbols = ['ADAM33', 'SERPINF2', 'SIX5', 'PRELP', 'HSPB1', 'MT1A', 'SMTN', 'DNAJB1', 'RIPOR3', 'CSF1']
    boruta_match = (top10_boruta_symbols == expected_boruta_symbols)

    check2_pass = (lasso_match and svm_match and mi_match and boruta_match)
    status2 = "PASS" if check2_pass else "FAIL"
    print(f"Check 2 [Method symbols exact match with notebook]: {status2}")
    print(f"  - LASSO match: {lasso_match} ({top10_lasso_symbols})")
    print(f"  - Boruta match: {boruta_match} ({top10_boruta_symbols})")
    print(f"  - SVM-RFE match: {svm_match} ({top10_svm_symbols})")
    print(f"  - Mutual Info match: {mi_match} ({top10_mi_symbols})")
    print(f"  - XGBoost top 10: {top10_xgb_symbols}")

    # ---------------------------------------------------------
    # Check 3: Determinism across 2 runs
    # ---------------------------------------------------------
    # Rerun LASSO, SVM, XGB, MI
    lasso_cv2 = LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="liblinear", scoring="accuracy", max_iter=5000, random_state=123)
    lasso_cv2.fit(X_scaled, y)
    top10_lasso2 = pd.Series(lasso_cv2.coef_[0], index=X.columns)[lambda s: s != 0].abs().sort_values(ascending=False).head(10).index.tolist()

    rfe2 = RFE(estimator=SVC(kernel="linear", class_weight="balanced", random_state=123), n_features_to_select=10, step=1)
    rfe2.fit(X_scaled, y)
    svm_final2 = SVC(kernel="linear", class_weight="balanced", random_state=123)
    svm_final2.fit(X_scaled[:, rfe2.support_], y)
    top10_svm2 = pd.Series(svm_final2.coef_[0], index=X.columns[rfe2.support_]).abs().sort_values(ascending=False).head(10).index.tolist()

    xgb_clf2 = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg/pos, eval_metric="logloss", random_state=123, n_jobs=-1)
    xgb_clf2.fit(X, y)
    top10_xgb2 = pd.Series(xgb_clf2.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).index.tolist()

    mi_scores2 = mutual_info_classif(X, y, random_state=123)
    top10_mi2 = pd.Series(mi_scores2, index=X.columns).sort_values(ascending=False).head(10).index.tolist()

    determ_pass = (top10_lasso == top10_lasso2 and top10_svm == top10_svm2 and top10_xgb == top10_xgb2 and top10_mi == top10_mi2)
    status3 = "PASS" if determ_pass else "FAIL"
    print(f"Check 3 [Determinism across reruns]: {status3}")

    # ---------------------------------------------------------
    # Check 4: SHAP TreeExplainer additivity on PRIMARY_PANEL
    # ---------------------------------------------------------
    primary_panel = list(set(top10_lasso).union(set(top10_boruta)))
    X_panel = X[primary_panel]
    rf_eval = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=123)
    rf_eval.fit(X_panel, y)

    explainer = shap.TreeExplainer(rf_eval)
    shap_vals = explainer.shap_values(X_panel)
    # Predicted probabilities from RF
    probs = rf_eval.predict_proba(X_panel)[:, 1]
    # In tree classifiers with TreeExplainer, sum of SHAP values + expected_value = predicted probability
    expected_val = explainer.expected_value[1]
    shap_sums = shap_vals[:, :, 1].sum(axis=1) + expected_val
    max_diff = np.max(np.abs(shap_sums - probs))
    check4_pass = (max_diff < 1e-3)
    status4 = "PASS" if check4_pass else "FAIL"
    print(f"Check 4 [SHAP TreeExplainer additivity (max diff = {max_diff:.2e} < 1e-3)]: {status4}")

    # ---------------------------------------------------------
    # Check 5: Fallback DEG filter runs within budget & valid
    # ---------------------------------------------------------
    t_start = time.time()
    # Vectorized Welch's t-test + BH-FDR on all 39,376 genes
    c_sub = counts_indexed
    sample_ctrl = sample[sample["STATUS"] == "NO_PD"]["COUNT"].tolist()
    sample_pd = sample[sample["STATUS"] == "PD"]["COUNT"].tolist()

    counts_ctrl = c_sub[sample_ctrl].values.astype(float)
    counts_pd = c_sub[sample_pd].values.astype(float)

    # Normalize to log2(CPM + 1)
    lib_ctrl = counts_ctrl.sum(axis=0, keepdims=True)
    lib_pd = counts_pd.sum(axis=0, keepdims=True)
    log_cpm_ctrl = np.log2((counts_ctrl / lib_ctrl) * 1e6 + 1)
    log_cpm_pd = np.log2((counts_pd / lib_pd) * 1e6 + 1)

    t_stat, pvals = stats.ttest_ind(log_cpm_pd, log_cpm_ctrl, axis=1, equal_var=False)
    log2fc = log_cpm_pd.mean(axis=1) - log_cpm_ctrl.mean(axis=1)
    clean_pvals = np.nan_to_num(pvals, nan=1.0)
    _, padj, _, _ = multipletests(clean_pvals, method="fdr_bh")
    filter_time = time.time() - t_start

    fallback_df = pd.DataFrame({
        "gene": counts_indexed.index,
        "log2fc": log2fc,
        "pvalue": clean_pvals,
        "padj": padj
    })

    has_no_nans = not fallback_df["padj"].isna().any()
    has_nonzero = (fallback_df["pvalue"] < 0.05).any()
    within_budget = (filter_time < 5.0)

    check5_pass = (has_no_nans and has_nonzero and within_budget)
    status5 = "PASS" if check5_pass else "FAIL"
    print(f"Check 5 [Fallback DEG filter runs in {filter_time:.3f} s (< 5 s) with valid p-values]: {status5}")

    all_pass = (check1_pass and check2_pass and determ_pass and check4_pass and check5_pass)
    print("=" * 60)
    print(f"OVERALL GATE VERIFICATION: {'PASSED' if all_pass else 'FAILED'}")
    print("=" * 60)
    return all_pass

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
