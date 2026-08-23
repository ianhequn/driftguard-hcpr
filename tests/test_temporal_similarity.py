import pytest


def test_identical_features_have_similarity_one_and_zero_distance():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.temporal_similarity import (
        cosine_similarity,
        relative_l2_distance,
    )

    feature = torch.randn(2, 8, 8)
    assert cosine_similarity(feature, feature) == pytest.approx(1.0, abs=1e-6)
    assert relative_l2_distance(feature, feature) == pytest.approx(0.0, abs=1e-6)


def test_feature_shape_mismatch_is_explicit():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.temporal_similarity import cosine_similarity

    with pytest.raises(ValueError, match="Feature shape mismatch"):
        cosine_similarity(torch.zeros(2, 4, 4), torch.zeros(2, 8, 8))
