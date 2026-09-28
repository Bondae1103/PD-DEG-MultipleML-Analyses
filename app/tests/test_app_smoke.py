import pytest
from streamlit.testing.v1 import AppTest

def test_app_smoke_runs_without_exceptions():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    
    assert not at.exception
    
    # Assert headers exist
    headers = [h.value for h in at.header]
    assert any("Pipeline Architecture" in h for h in headers)
    assert any("Feature Selection" in h for h in headers)
    assert any("Analysis Plots" in h for h in headers)
    assert any("Biomarker Inference" in h for h in headers)

def test_app_smoke_switch_cohort_and_slider():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception
    
    # Switch cohort in sidebar selectbox
    if at.sidebar.selectbox:
        at.sidebar.selectbox[0].select("GSE136666").run()
        assert not at.exception

    # Adjust volcano slider in sidebar if present
    if at.sidebar.slider:
        at.sidebar.slider[0].set_value(1.5).run()
        assert not at.exception
