import numpy as np
import torch

from driftguard.analysis.residual_patches import selected_residual_patches
from driftguard.models.risk_predictor import TinyRiskCNN
from driftguard.profiling.selector_gpu import GPUAFVTSelector


def _selector(extraction: str) -> GPUAFVTSelector:
    return GPUAFVTSelector(
        TinyRiskCNN(metadata_dim=1),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        extraction=extraction,
    )


def test_vectorized_all_block_patches_match_old_cpu_path():
    previous = np.zeros((16, 20, 3), dtype=np.uint8)
    current = np.arange(16 * 20 * 3, dtype=np.uint16)
    current = (current % 256).astype(np.uint8).reshape(16, 20, 3)
    indices = np.asarray([0, 5, 10, 15], dtype=np.int64)
    expected = selected_residual_patches(
        current,
        previous,
        indices,
        grid=4,
        size=16,
    )
    current_tensor = torch.from_numpy(current).permute(2, 0, 1).unsqueeze(0)
    previous_tensor = torch.from_numpy(previous).permute(2, 0, 1).unsqueeze(0)
    actual = _selector("all_blocks").candidate_patches(
        current_tensor,
        previous_tensor,
        indices,
    )
    assert torch.allclose(actual, torch.from_numpy(expected), atol=1e-6)


def test_direct_candidate_patches_match_all_block_strategy():
    generator = torch.Generator().manual_seed(7)
    current = torch.randint(
        0,
        256,
        (1, 3, 16, 20),
        dtype=torch.uint8,
        generator=generator,
    )
    reference = torch.randint(
        0,
        256,
        (1, 3, 16, 20),
        dtype=torch.uint8,
        generator=generator,
    )
    indices = np.asarray([1, 6, 14], dtype=np.int64)
    all_blocks = _selector("all_blocks").candidate_patches(
        current,
        reference,
        indices,
    )
    direct = _selector("direct").candidate_patches(
        current,
        reference,
        indices,
    )
    assert torch.allclose(direct, all_blocks, atol=1e-6)
