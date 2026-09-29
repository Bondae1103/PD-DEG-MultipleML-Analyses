"""
app/tests/test_io.py
--------------------
Unit tests for data intake in app/core/io.py per Plan v2 Section 4.1.
Verifies column validation, format handling, and actionable error messages.
"""

import io
import pytest
import pandas as pd
from app.core.io import load_deg_list, load_optional_raw_data, UserInputError

def test_load_deg_list_valid_csv():
    csv_data = "ENTREZID,SYMBOL,GENENAME,logFC,adj.P.Val\n80332,ADAM33,ADAM metallopeptidase,1.5,0.001\n"
    df = load_deg_list(io.StringIO(csv_data))
    assert len(df) == 1
    assert "ENTREZID" in df.columns
    assert "SYMBOL" in df.columns
    assert "GENENAME" in df.columns

def test_load_deg_list_case_insensitive_and_alternate_names():
    csv_data = "entrez_id,gene_symbol,gene_name\n1234,GENE1,Test Gene\n"
    df = load_deg_list(io.StringIO(csv_data))
    assert len(df) == 1
    assert "ENTREZID" in df.columns
    assert "SYMBOL" in df.columns
    assert "GENENAME" in df.columns

def test_load_deg_list_missing_columns_raises_user_input_error():
    csv_bad = "SYMBOL,GENENAME\nGENE1,Test Gene\n"
    with pytest.raises(UserInputError) as exc_info:
        load_deg_list(io.StringIO(csv_bad))
    assert "missing required column(s): ENTREZID" in str(exc_info.value)

def test_load_optional_raw_data_valid():
    counts_csv = "GENE ID,S1,S2,S3,S4\n101,10,20,30,40\n102,5,15,25,35\n"
    meta_csv = "SampleID,STATUS\nS1,NO_PD\nS2,NO_PD\nS3,PD\nS4,PD\n"

    counts_df, meta_df = load_optional_raw_data(
        io.StringIO(counts_csv),
        io.StringIO(meta_csv)
    )
    assert counts_df.shape == (2, 5)
    assert meta_df.shape == (4, 2)
    assert "GENE ID" in counts_df.columns
    assert "STATUS" in meta_df.columns

def test_load_optional_raw_data_insufficient_samples_raises():
    counts_csv = "GENE ID,S1,S2\n101,10,20\n"
    meta_csv = "SampleID,STATUS\nS1,NO_PD\nS2,PD\n" # Only 1 sample per group

    with pytest.raises(UserInputError) as exc_info:
        load_optional_raw_data(io.StringIO(counts_csv), io.StringIO(meta_csv))
    assert "at least 2 samples" in str(exc_info.value)
