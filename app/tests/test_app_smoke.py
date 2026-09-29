"""
app/tests/test_app_smoke.py
---------------------------
AppTest smoke tests for the multipage Streamlit dashboard per Plan v2 Section 9.6.
Tests that the Landing page loads cleanly, the About page renders independently,
and the Output page safely guards against unready analysis state.
"""

from streamlit.testing.v1 import AppTest

def test_landing_page_loads_without_exceptions():
    at = AppTest.from_file("pages/1_Landing.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Landing page raised exception: {at.exception}"
    # Verify radio button exists
    assert len(at.radio) >= 1
    # Verify method checkboxes exist
    assert len(at.checkbox) == 5

def test_about_page_renders_without_prior_state():
    at = AppTest.from_file("pages/3_About.py", default_timeout=30)
    at.run()
    assert not at.exception, f"About page raised exception: {at.exception}"

def test_output_page_guards_against_unready_state():
    at = AppTest.from_file("pages/2_Output.py", default_timeout=30)
    at.run()
    # Should not raise an unhandled exception; should display info message
    assert not at.exception, f"Output page raised exception when state was unready: {at.exception}"
    assert len(at.info) >= 1
    assert "No analysis results to display yet" in at.info[0].value
