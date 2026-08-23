import pytest


def test_stage_sensitivity_metrics_are_zero_for_identical_logits():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.stage_sensitivity import (
        logit_relative_l2,
        pixel_disagreement_ratio,
        prediction_kl_divergence,
    )

    logits = torch.randn(1, 5, 8, 8)
    assert prediction_kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)
    assert pixel_disagreement_ratio(logits, logits) == pytest.approx(0.0, abs=1e-6)
    assert logit_relative_l2(logits, logits) == pytest.approx(0.0, abs=1e-6)
