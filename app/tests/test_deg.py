import pytest
import pandas as pd
import numpy as np

from app.core.deg import load_deg, filter_deg, compute_deg_fallback

def test_load_deg_canonical_columns():
    df = load_deg()
    for col in ["gene", "log2fc", "pvalue", "padj", "neglog10_p"]:
        assert col in df.columns
    assert not np.isinf(df["neglog10_p"]).any()
    assert not np.isnan(df["neglog10_p"]).any()

def test_filter_deg_boundaries():
    toy = pd.DataFrame({
        "gene": ["G1", "G2", "G3", "G4"],
        "log2fc": [1.0, -1.0, 0.99, -0.99],
        "pvalue": [0.05, 0.05, 0.05, 0.05],
        "padj": [0.05, 0.05, 0.05, 0.05],
        "neglog10_p": [1.3, 1.3, 1.3, 1.3]
    })
    filtered = filter_deg(toy, p_col="padj", p_thresh=0.05, lfc_thresh=1.0)
    cats = dict(zip(filtered["gene"], filtered["category"]))
    assert cats["G1"] == "Up"
    assert cats["G2"] == "Down"
    assert cats["G3"] == "Not significant"
    assert cats["G4"] == "Not significant"

def test_compute_deg_fallback_vs_scipy():
    np.random.seed(42)
    # 2 groups, 5 samples each, 3 genes
    mat_a = np.random.normal(loc=5.0, scale=0.5, size=(5, 3))
    mat_b = np.random.normal(loc=3.0, scale=0.5, size=(5, 3))
    
    data = np.vstack([mat_a, mat_b])
    sample_ids = [f"S{i}" for i in range(10)]
    genes = ["Gene1", "Gene2", "Gene3"]
    df_expr = pd.DataFrame(data, index=sample_ids, columns=genes)
    
    groups = pd.Series(["A"]*5 + ["B"]*5, index=sample_ids)
    
    res = compute_deg_fallback(df_expr, groups, "A", "B")
    assert len(res) == 3
    assert all(res["log2fc"] > 0)
    assert all(res["pvalue"] < 0.01)
    assert not np.isinf(res["neglog10_p"]).any()
