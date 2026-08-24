import os

import pytest


def test_vspw_model_class_count_requires_a_verified_candidate():
    model_path = os.environ.get("DRIFTGUARD_R75_MODEL")
    if not model_path:
        pytest.skip("No verified VSPW-compatible checkpoint selected")
    pytest.fail("Candidate validation must be implemented for the selected model")
