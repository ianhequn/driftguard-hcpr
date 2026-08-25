import numpy as np

from driftguard.datasets.vspw import normalize_vspw_mask


def test_round76_training_label_domain():
    raw = np.array([0, 1, 124, 255], dtype=np.uint8)
    assert normalize_vspw_mask(raw).tolist() == [255, 0, 123, 255]
