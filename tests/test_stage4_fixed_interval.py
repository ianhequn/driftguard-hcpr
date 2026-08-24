import pytest

from driftguard.structured_reuse.policies import (
    FixedIntervalPolicy,
    StageAgePolicy,
)


@pytest.mark.parametrize("interval", [2, 3, 4, 6, 8])
def test_fixed_interval_pattern(interval):
    policy = FixedIntervalPolicy(interval)
    actual = [
        policy.decide(index, cache_age=0) for index in range(interval * 3)
    ]
    expected = [index % interval == 0 for index in range(interval * 3)]
    assert actual == expected


@pytest.mark.parametrize("threshold", [1, 2, 3, 5, 7])
def test_stage_age_is_equivalent_to_matching_fixed_interval(threshold):
    age_policy = StageAgePolicy(threshold)
    fixed_policy = FixedIntervalPolicy(threshold + 1)
    cache_age = 0
    for frame_index in range((threshold + 1) * 4):
        age_refresh = age_policy.decide(frame_index, cache_age)
        fixed_refresh = fixed_policy.decide(frame_index, cache_age)
        assert age_refresh == fixed_refresh
        cache_age = 0 if age_refresh else cache_age + 1
