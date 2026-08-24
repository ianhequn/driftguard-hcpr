import torch

from driftguard.profiling.selector_gpu import gpu_rgb_residual


def test_gpu_residual_matches_round3_normalization():
    current = torch.tensor(
        [[[[0, 255], [128, 64]]] * 3],
        dtype=torch.uint8,
    )
    reference = torch.zeros_like(current)
    actual = gpu_rgb_residual(current, reference)
    expected = current.float() / 255.0
    assert torch.allclose(actual, expected)


def test_gpu_residual_accepts_unit_float_frames():
    current = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    reference = torch.full_like(current, 0.25)
    assert torch.allclose(
        gpu_rgb_residual(current, reference),
        torch.full_like(current, 0.75),
    )
