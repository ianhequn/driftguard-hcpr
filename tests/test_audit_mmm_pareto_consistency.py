from scripts.audit_mmm_pareto_consistency import dominates, dominators, make_point


def point(method: str, damage: float, speedup: float) -> dict:
    return make_point(
        model="B0",
        device="GPU",
        method=method,
        boundary="test",
        interval=None,
        quantile=None,
        quality_source="q",
        latency_source="l",
        quality_protocol="p",
        latency_protocol="p",
        damage_pp=damage,
        speedup=speedup,
        ci_low=speedup - 0.01,
        ci_high=speedup + 0.01,
    )


def test_dominance_requires_one_strict_inequality():
    equal_a = point("a", 1.0, 2.0)
    equal_b = point("b", 1.0, 2.0)
    assert not dominates(equal_a, equal_b)
    better = point("better", 0.9, 2.0)
    assert dominates(better, equal_b)


def test_dominators_returns_all_strict_dominators():
    target = point("target", 1.0, 2.0)
    a = point("a", 0.9, 2.0)
    b = point("b", 1.0, 2.1)
    c = point("c", 1.2, 2.2)
    assert dominators(target, [target, a, b, c]) == ["a", "b"]


def test_mixed_protocol_is_not_strict_pareto_eligible():
    mixed = make_point(
        model="B0",
        device="GPU",
        method="mixed",
        boundary="test",
        interval=None,
        quantile=None,
        quality_source="q",
        latency_source="l",
        quality_protocol="quality-p",
        latency_protocol="latency-p",
        damage_pp=1.0,
        speedup=2.0,
        ci_low=1.9,
        ci_high=2.1,
    )
    assert mixed["measurement_protocol_id"] == "MIXED_PROTOCOL"
    assert mixed["valid_for_strict_pareto"] is False
