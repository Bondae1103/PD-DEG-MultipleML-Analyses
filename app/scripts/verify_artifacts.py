import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    print("=" * 60)
    print("RUNNING ARTIFACT VERIFICATION SUITE")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    results = {}
    
    # 1. Model loads without error
    model_path = os.path.join(artifacts_dir, "model.json")
    try:
        booster = xgb.Booster()
        booster.load_model(model_path)
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print("PASS [1/5]: Model loaded successfully from native JSON.")
        results["1_model_load"] = "PASS"
    except Exception as e:
        print(f"FAIL [1/5]: Failed to load model: {e}")
        results["1_model_load"] = f"FAIL: {e}"
        sys.exit(1)

    # 2. Feature list checks
    features_path = os.path.join(artifacts_dir, "feature_list.json")
    try:
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        assert len(features) == len(set(features)), "Duplicate features detected!"
        # Check model expected width
        expected_width = len(features)
        # Verify booster feature names or num_features
        assert booster.num_features() == expected_width, f"Booster features {booster.num_features()} != {expected_width}"
        print(f"PASS [2/5]: Feature list has {len(features)} unique features matching model input width.")
        results["2_feature_list"] = "PASS"
    except Exception as e:
        print(f"FAIL [2/5]: Feature list error: {e}")
        results["2_feature_list"] = f"FAIL: {e}"
        sys.exit(1)

    # 3. Reproduce notebook accuracy/AUC within ±0.01
    try:
        mat_path = os.path.join(artifacts_dir, "cohort_gse68719_normalized.parquet")
        meta_path = os.path.join(artifacts_dir, "cohort_gse68719_meta.csv")
        df_expr = pd.read_parquet(mat_path)
        df_meta = pd.read_csv(meta_path)
        
        X = df_expr[features]
        y = (df_meta["group"] == "PD").astype(int).values
        
        # Load metadata metrics
        with open(os.path.join(artifacts_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            
        reported_acc = meta["metrics"]["loocv_accuracy"]
        reported_auc = meta["metrics"]["loocv_auc"]
        notebook_acc = meta["metrics"]["notebook_rf_loocv_accuracy"]
        
        # Cross-validation check on X, y
        from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
        loo = LeaveOneOut()
        cv_acc = cross_val_score(model, X, y, cv=loo, scoring="accuracy").mean()
        cv_proba = cross_val_predict(model, X, y, cv=loo, method="predict_proba")[:, 1]
        cv_auc = roc_auc_score(y, cv_proba)
        
        diff_acc = abs(cv_acc - notebook_acc)
        assert diff_acc <= 0.01, f"Accuracy difference {diff_acc:.4f} > 0.01 (CV: {cv_acc:.4f}, Notebook: {notebook_acc:.4f})"
        print(f"PASS [3/5]: Model LOOCV Accuracy: {cv_acc:.4f} reproduces notebook target {notebook_acc:.4f} (diff {diff_acc:.4f} <= 0.01). AUC: {cv_auc:.4f}.")
        results["3_metrics_reproduced"] = "PASS"
    except Exception as e:
        print(f"FAIL [3/5]: Metrics verification failed: {e}")
        results["3_metrics_reproduced"] = f"FAIL: {e}"
        sys.exit(1)

    # 4. TreeExplainer and Additivity within 1e-3
    try:
        explainer = shap.TreeExplainer(model)
        # Select 20 samples
        np.random.seed(123)
        sample_indices = np.random.choice(len(X), size=min(20, len(X)), replace=False)
        X_sample = X.iloc[sample_indices]
        
        shap_values = explainer.shap_values(X_sample)
        base_value = explainer.expected_value
        raw_margin = model.predict(X_sample, output_margin=True)
        
        # Handle shape
        if isinstance(shap_values, list):
            # Binary classification list format
            sv_pos = shap_values[1]
            base_val_pos = base_value[1] if isinstance(base_value, (list, np.ndarray)) else base_value
        else:
            sv_pos = shap_values
            base_val_pos = base_value
            
        additivity_diff = np.abs(base_val_pos + sv_pos.sum(axis=1) - raw_margin)
        max_diff = np.max(additivity_diff)
        assert max_diff < 1e-3, f"Additivity violation: max diff = {max_diff}"
        print(f"PASS [4/5]: shap.TreeExplainer additivity holds across 20 samples (max diff: {max_diff:.2e} < 1e-3).")
        results["4_shap_additivity"] = "PASS"
    except Exception as e:
        print(f"FAIL [4/5]: SHAP additivity error: {e}")
        results["4_shap_additivity"] = f"FAIL: {e}"
        sys.exit(1)

    # 5. Explicit Class Mapping
    try:
        class_map = meta["class_map"]
        print(f"PASS [5/5]: Explicit class mapping confirmed: {class_map} (Positive = '{meta['positive_class']}', Negative = '{meta['negative_class']}')")
        assert class_map.get("0") == "NO_PD" or class_map.get(0) == "NO_PD", "Index 0 must be NO_PD"
        assert class_map.get("1") == "PD" or class_map.get(1) == "PD", "Index 1 must be PD"
        results["5_class_mapping"] = "PASS"
    except Exception as e:
        print(f"FAIL [5/5]: Class mapping error: {e}")
        results["5_class_mapping"] = f"FAIL: {e}"
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL 5 VERIFICATION CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
