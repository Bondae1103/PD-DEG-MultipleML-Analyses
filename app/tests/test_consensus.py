"""
app/tests/test_consensus.py
---------------------------
Unit tests for the exact consensus rule per Plan v2 Section 4.5 & Section 9.3.
Verifies all 4 branches and click-order determinism.
"""

from app.core.methods import MethodResult
from app.core.consensus import build_consensus

id_to_symbol = {
    "1": "GENE_1", "2": "GENE_2", "3": "GENE_3",
    "4": "GENE_4", "5": "GENE_5", "6": "GENE_6",
    "7": "GENE_7", "8": "GENE_8", "9": "GENE_9",
    "10": "GENE_10", "11": "GENE_11", "12": "GENE_12"
}

def make_result(method_name: str, top_ids: list) -> MethodResult:
    return MethodResult(
        method_name=method_name,
        metric_name="Score",
        top10_ids=top_ids,
        top10_symbols=[id_to_symbol.get(g, g) for g in top_ids],
        scores={g: 1.0 for g in top_ids},
        n_available=len(top_ids)
    )

def test_branch_1_single_method_selected():
    """Branch 1: Only 1 method selected -> panel = that method's top 10 directly."""
    results = {
        "LASSO": make_result("LASSO", ["1", "2", "3"])
    }
    res = build_consensus(results, id_to_symbol)
    assert res.rule == "single_method"
    assert res.panel == ["1", "2", "3"]
    assert res.source_method == "LASSO"

def test_branch_2_strict_intersection():
    """Branch 2: All selected methods share >= 1 gene -> strict intersection."""
    results = {
        "LASSO": make_result("LASSO", ["1", "2", "3"]),
        "Boruta": make_result("Boruta", ["2", "3", "4"]),
        "SVM_RFE": make_result("SVM_RFE", ["2", "3", "5"])
    }
    res = build_consensus(results, id_to_symbol)
    assert res.rule == "intersection_all_selected"
    assert set(res.panel) == {"2", "3"}
    assert res.vote_count == 3

def test_branch_3_most_occurring_tier():
    """Branch 3: No gene shared by all, but some genes have votes > 1 -> most occurring tier."""
    results = {
        "LASSO": make_result("LASSO", ["1", "2", "3"]),
        "Boruta": make_result("Boruta", ["2", "4", "5"]),
        "SVM_RFE": make_result("SVM_RFE", ["3", "6", "7"])
    }
    # Gene 2 has 2 votes (LASSO, Boruta)
    # Gene 3 has 2 votes (LASSO, SVM)
    # Other genes have 1 vote
    # max_votes = 2
    res = build_consensus(results, id_to_symbol)
    assert res.rule == "most_occurring"
    assert set(res.panel) == {"2", "3"}
    assert res.vote_count == 2

def test_branch_4_total_disjoint_fallback():
    """Branch 4: Zero overlap across all selected methods -> fallback to top-1 of first method in METHOD_NAMES order."""
    results = {
        "XGBoost": make_result("XGBoost", ["10", "11"]),
        "LASSO": make_result("LASSO", ["1", "2"])
    }
    # In METHOD_NAMES, LASSO comes before XGBoost
    res = build_consensus(results, id_to_symbol)
    assert res.rule == "fallback_top1"
    assert res.panel == ["1"]  # Top gene of LASSO
    assert res.source_method == "LASSO"

def test_determinism_under_different_click_orders():
    """Fixed config order METHOD_NAMES defines 'first method', not the order keys were inserted."""
    # Insertion order 1: MutualInfo first
    results_order1 = {
        "MutualInfo": make_result("MutualInfo", ["8", "9"]),
        "SVM_RFE": make_result("SVM_RFE", ["4", "5"])
    }
    # Insertion order 2: SVM_RFE first
    results_order2 = {
        "SVM_RFE": make_result("SVM_RFE", ["4", "5"]),
        "MutualInfo": make_result("MutualInfo", ["8", "9"])
    }

    # SVM_RFE appears before MutualInfo in METHOD_NAMES = ["LASSO", "SVM_RFE", "XGBoost", "MutualInfo", "Boruta"]
    res1 = build_consensus(results_order1, id_to_symbol)
    res2 = build_consensus(results_order2, id_to_symbol)

    assert res1.source_method == "SVM_RFE"
    assert res2.source_method == "SVM_RFE"
    assert res1.panel == ["4"]
    assert res2.panel == ["4"]
