import numpy as np

from driftguard.datasets.vspw import normalize_vspw_mask


def test_official_vspw_label_normalization():
    source = np.array([[0, 1, 124, 255]], dtype=np.uint8)
    normalized = normalize_vspw_mask(source)
    assert normalized.tolist() == [[255, 0, 123, 255]]
