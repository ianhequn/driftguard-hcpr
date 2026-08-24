import numpy as np
import torch

from driftguard.models.risk_predictor import TinyRiskCNN
from driftguard.profiling.selector_gpu import GPUAFVTSelector


class CountingCNN(TinyRiskCNN):
    def __init__(self):
        super().__init__(metadata_dim=1)
        self.calls = 0

    def forward(self, patches, metadata):
        self.calls += 1
        return super().forward(patches, metadata)


def test_fast_path_skips_residual_and_cnn_when_age_choice_is_complete():
    model = CountingCNN()
    selector = GPUAFVTSelector(
        model,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    current = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    reference = torch.zeros_like(current)
    ages = np.arange(16, dtype=np.float32)
    result = selector.select(current, reference, ages, 0.25)
    assert not result.used_visual_path
    assert model.calls == 0
    assert np.flatnonzero(result.decision.mask).tolist() == [12, 13, 14, 15]


def test_fast_path_keeps_exact_budget_when_all_tie_blocks_are_selected():
    model = CountingCNN()
    selector = GPUAFVTSelector(
        model,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    frame = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    ages = np.asarray([3] * 4 + [0] * 12, dtype=np.float32)
    result = selector.select(frame, frame, ages, 0.25)
    assert result.decision.mask.sum() == 4
    assert model.calls == 0
