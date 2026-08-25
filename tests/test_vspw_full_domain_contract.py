import numpy as np

from driftguard.datasets.vspw import VSPW_IGNORE_INDEX, VSPW_NUM_CLASSES, normalize_vspw_mask


def test_canonical_contract_contains_only_classes_or_ignore():
    raw = np.arange(256, dtype=np.uint8)
    canonical = normalize_vspw_mask(raw)
    assert np.all(
        (canonical == VSPW_IGNORE_INDEX)
        | ((canonical >= 0) & (canonical < VSPW_NUM_CLASSES))
    )
