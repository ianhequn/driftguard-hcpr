import numpy as np

from driftguard.datasets.vspw import (
    VSPW_IGNORE_INDEX,
    VSPW_NUM_CLASSES,
    normalize_vspw_mask,
    official_normalize_vspw_mask,
    vspw_official_valid_mask,
)


def test_official_loader_and_evaluator_contract_on_synthetic_values():
    raw = np.array([0, 1, 124, 253, 255], dtype=np.uint8)
    official = official_normalize_vspw_mask(raw)
    canonical = normalize_vspw_mask(raw)

    assert official.tolist() == [255, 0, 123, 252, 255]
    assert canonical.tolist() == [255, 0, 123, 255, 255]
    assert vspw_official_valid_mask(official).tolist() == [False, True, True, False, False]
    assert set(canonical.tolist()) <= set(range(VSPW_NUM_CLASSES)) | {VSPW_IGNORE_INDEX}


def test_canonical_valid_mask_matches_official_on_all_synthetic_predictions():
    raw = np.arange(256, dtype=np.uint8)
    official = official_normalize_vspw_mask(raw)
    canonical = normalize_vspw_mask(raw)
    assert np.array_equal(vspw_official_valid_mask(official), canonical != VSPW_IGNORE_INDEX)
