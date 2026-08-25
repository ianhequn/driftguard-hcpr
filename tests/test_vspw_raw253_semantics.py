import numpy as np

from driftguard.datasets.vspw import normalize_vspw_mask


def test_raw_253_is_void_not_a_semantic_class():
    canonical = normalize_vspw_mask(np.array([253], dtype=np.uint8))
    assert canonical.tolist() == [255]
