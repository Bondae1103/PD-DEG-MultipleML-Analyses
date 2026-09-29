"""
app/tests/test_preprocess.py
----------------------------
Tests for DEG cleaning and library-size CPM normalization per Plan v2 Section 9.1.
"""

import pytest
import numpy as np
import pandas as pd
from app.core.preprocess import clean_deg_list, subset_and_normalize, build_X_y
from app.core.io import UserInputError

def test_cleaning_drops_expected_categories():
    raw_deg = pd.DataFrame({
        "ENTREZID": [101, None, 103, 104, 105, 106, 107],
        "SYMBOL": ["ADAM33", "SIX5", "LOC1234", "MIR21", "SNORA73", "HSPB1", "NORMAL_GENE"],
        "GENENAME": [
            "ADAM metallopeptidase",
            "SIX homeobox 5",
            "uncharacterized LOC1234",
            "microRNA 21",
            "small nucleolar RNA",
            "heat shock protein (pseudogene)",
            "normal protein coding gene"
        ]
    })

    cleaned = clean_deg_list(raw_deg)
    # Row 1 has None ENTREZID -> dropped
    # Row 2 has LOC1234 -> dropped
    # Row 3 has MIR21 -> dropped
    # Row 4 has SNORA73 -> dropped
    # Row 5 has 'pseudogene' in GENENAME -> dropped
    # Remaining: Row 0 (ADAM33) and Row 6 (NORMAL_GENE)
    assert len(cleaned) == 2
    assert list(cleaned["SYMBOL"]) == ["ADAM33", "NORMAL_GENE"]

def test_cpm_library_size_uses_full_counts_matrix_not_subset():
    """
    Critical regression test: verifying CPM calculation divides by full counts sum,
    NOT the sum of the subsetted DEGs.
    """
    # 5 genes in full counts, each with total count 100 per sample
    full_counts = pd.DataFrame({
        "GENE ID": ["1", "2", "3", "4", "5"],
        "S1": [20, 20, 20, 20, 20],  # sum = 100
        "S2": [10, 10, 10, 10, 60]   # sum = 100
    })

    # Only genes 1 and 2 are in the DEG list
    # If using subset library size (sum = 40 for S1), CPM would be (20/40)*1e6 = 500,000
    # If using full library size (sum = 100 for S1), CPM should be (20/100)*1e6 = 200,000
    deg_list = pd.DataFrame({
        "ENTREZID": [1, 2] + [i for i in range(10, 25)], # At least 15 for threshold check
        "SYMBOL": [f"G{i}" for i in range(17)],
        "GENENAME": [f"Gene {i}" for i in range(17)]
    })

    # Add dummy rows to full_counts to satisfy threshold
    extra_counts = pd.DataFrame({
        "GENE ID": [str(i) for i in range(10, 25)],
        "S1": [10] * 15,
        "S2": [10] * 15
    })
    full_counts = pd.concat([full_counts, extra_counts], ignore_index=True)

    log_cpm, id_to_symbol = subset_and_normalize(deg_list, full_counts)

    # Full library size for S1: 100 + 150 = 250
    # Gene 1 count = 20 -> CPM = (20 / 250) * 1e6 = 80,000
    expected_cpm_s1_g1 = (20.0 / 250.0) * 1e6
    expected_log_cpm_s1_g1 = np.log2(expected_cpm_s1_g1 + 1)

    actual_log_cpm = log_cpm.loc["1", "S1"]
    assert np.isclose(actual_log_cpm, expected_log_cpm_s1_g1, atol=1e-5)

def test_too_few_genes_raises_user_input_error():
    counts = pd.DataFrame({
        "GENE ID": ["1", "2", "3"],
        "S1": [10, 20, 30],
        "S2": [15, 25, 35]
    })
    deg = pd.DataFrame({
        "ENTREZID": [1, 2],
        "SYMBOL": ["A", "B"],
        "GENENAME": ["Gene A", "Gene B"]
    })
    with pytest.raises(UserInputError) as exc_info:
        subset_and_normalize(deg, counts)
    assert "At least 15 genes are required" in str(exc_info.value)
