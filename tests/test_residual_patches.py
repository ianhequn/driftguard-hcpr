import numpy as np

from driftguard.analysis.residual_patches import selected_residual_patches


def test_selected_patch_extraction_returns_only_candidates():
    previous = np.zeros((8, 8, 3), dtype=np.uint8)
    current = previous.copy()
    current[:2, :2] = 255
    current[6:, 6:] = 128
    patches = selected_residual_patches(
        current,
        previous,
        np.asarray([0, 15]),
        grid=4,
        size=4,
    )
    assert patches.shape == (2, 3, 4, 4)
    assert np.allclose(patches[0], 1.0)
    assert np.allclose(patches[1], 128 / 255.0)


def test_selected_patch_extraction_handles_empty_candidates():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    patches = selected_residual_patches(
        frame,
        frame,
        np.empty(0, dtype=np.int64),
        grid=4,
        size=4,
    )
    assert patches.shape == (0, 3, 4, 4)
