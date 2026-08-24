import os

import pytest


def test_vspw_class_alignment_requires_verified_taxonomy():
    alignment_path = os.environ.get("DRIFTGUARD_R75_CLASS_ALIGNMENT")
    if not alignment_path:
        pytest.skip("No verified VSPW class alignment is available")
    pytest.fail("Class alignment must be checked against the selected candidate")
