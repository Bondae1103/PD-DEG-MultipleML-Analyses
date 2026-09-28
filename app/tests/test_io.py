import io
import pytest
import pandas as pd
import numpy as np

from app.core import UserInputError
from app.core.io import load_matrix, detect_orientation, validate_upload

FEATURES = ["ADAM33", "DNAJB1", "RIPOR3", "HSPB1", "SIX5"]

def test_load_matrix_csv_and_tsv():
    csv_data = "sample_id,ADAM33,DNAJB1\nS1,4.5,2.1\nS2,3.8,1.9\n"
    df = load_matrix(io.BytesIO(csv_data.encode("utf-8")))
    assert df.shape == (2, 2)
    assert "ADAM33" in df.columns

    tsv_data = "sample_id\tADAM33\tDNAJB1\nS1\t4.5\t2.1\nS2\t3.8\t1.9\n"
    df_tsv = load_matrix(io.BytesIO(tsv_data.encode("utf-8")))
    assert df_tsv.shape == (2, 2)

def test_detect_orientation_both():
    # Samples x Genes
    df_samples = pd.DataFrame(np.ones((3, 4)), index=["S1", "S2", "S3"], columns=["ADAM33", "DNAJB1", "G3", "G4"])
    assert detect_orientation(df_samples, FEATURES) == "samples_x_genes"

    # Genes x Samples (case and whitespace variants)
    df_genes = pd.DataFrame(np.ones((4, 3)), index=["  adam33 ", "DNAJB1", "G3", "G4"], columns=["S1", "S2", "S3"])
    assert detect_orientation(df_genes, FEATURES) == "genes_x_samples"

def test_detect_orientation_zero_overlap_raises():
    df_bad = pd.DataFrame(np.ones((2, 2)), index=["S1", "S2"], columns=["UNKNOWN1", "UNKNOWN2"])
    with pytest.raises(UserInputError) as exc:
        detect_orientation(df_bad, FEATURES)
    assert "None of the model's biomarker genes were found" in str(exc.value)

def test_validate_upload_duplicates_and_coverage():
    # Duplicate columns and duplicate samples
    df_dup = pd.DataFrame(
        [[1.0, 2.0, 5.0, 3.0, 4.0], [2.0, 3.0, 6.0, 4.0, 5.0]],
        index=["S1", "S1"],
        columns=["ADAM33", "ADAM33", "DNAJB1", "RIPOR3", "HSPB1"]
    )
    df_clean, rep = validate_upload(df_dup, FEATURES)
    assert len(df_clean.index) == 2
    assert "S1_2" in df_clean.index or "S1" in df_clean.index
    assert rep.coverage >= 0.8
    assert len(rep.warnings) > 0

def test_validate_upload_low_coverage_raises():
    df_low = pd.DataFrame([[1.0]], index=["S1"], columns=["ADAM33"])
    with pytest.raises(UserInputError) as exc:
        validate_upload(df_low, FEATURES)
    assert "below the required" in str(exc.value)

def test_raw_counts_warning():
    # All integer and large
    df_raw = pd.DataFrame(
        np.array([[2000, 3500, 4100, 5000, 6000]], dtype=float),
        index=["S1"],
        columns=FEATURES
    )
    _, rep = validate_upload(df_raw, FEATURES)
    assert rep.looks_like_raw_counts is True
    assert any("raw read counts" in w for w in rep.warnings)

def test_nan_threshold_raises():
    # >5% non-numeric
    bad_csv = "id,A,B\nS1,1.0,abc\nS2,def,2.0\nS3,3.0,4.0\n"
    with pytest.raises(UserInputError) as exc:
        load_matrix(io.BytesIO(bad_csv.encode("utf-8")))
    assert "could not be converted to numeric" in str(exc.value)
