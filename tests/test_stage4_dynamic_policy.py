import numpy as np
import pytest

from driftguard.structured_reuse.policies import (
    StageAgeVisualPolicy,
    VisualRiskPolicy,
    aggregate_frame_risk,
)


@pytest.mark.parametrize("aggregation", ["max", "mean", "p90", "topk_mean"])
def test_frame_risk_aggregation_uses_current_block_scores(aggregation):
    values = np.arange(16, dtype=np.float32)
    risk = aggregate_frame_risk(values, aggregation)
    assert np.isfinite(risk)
    assert values.min() <= risk <= values.max()


def test_visual_policy_requires_current_risk_and_no_future_signal():
    policy = VisualRiskPolicy(threshold=0.5)
    assert policy.decide(0, cache_age=0)
    with pytest.raises(ValueError):
        policy.decide(1, cache_age=0)
    assert not policy.decide(1, cache_age=0, frame_risk=0.4)
    assert policy.decide(1, cache_age=0, frame_risk=0.6)


def test_age_visual_guard_forces_refresh_before_visual_threshold():
    policy = StageAgeVisualPolicy(max_age=2, threshold=0.9)
    assert policy.decide(0, cache_age=0)
    assert not policy.decide(1, cache_age=0, frame_risk=0.1)
    assert policy.decide(2, cache_age=2, frame_risk=0.1)
    assert policy.decide(2, cache_age=0, frame_risk=1.0)
