import numpy as np
import torch

from driftguard.analysis.age_aware_schedulers import select_afvt
from driftguard.models.risk_predictor import TinyRiskCNN
from driftguard.profiling.selector_gpu import GPUAFVTSelector


def test_gpu_selector_matches_frozen_afvt_decision():
    torch.manual_seed(11)
    model = TinyRiskCNN(metadata_dim=1).eval()
    selector = GPUAFVTSelector(
        model,
        np.asarray([5.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
    )
    current = torch.randint(0, 256, (1, 3, 16, 20), dtype=torch.uint8)
    reference = torch.randint(0, 256, (1, 3, 16, 20), dtype=torch.uint8)
    ages = np.asarray([2] * 8 + [1] * 8, dtype=np.float32)

    def score(indices):
        return selector.score_candidates(
            current,
            reference,
            ages,
            indices,
        )

    old = select_afvt(ages, 0.25, score)
    new = selector.select(current, reference, ages, 0.25)
    assert np.array_equal(new.decision.mask, old.mask)
    assert new.decision.mask.sum() == 4
