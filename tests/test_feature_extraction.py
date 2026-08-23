import pytest


def test_four_stage_features_are_nchw():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from driftguard.models.segformer import get_feature_stats, get_stage_features

    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        decoder_hidden_size=32,
        num_labels=3,
    )
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    with torch.inference_mode():
        outputs = model(
            pixel_values=torch.randn(1, 3, 64, 64),
            output_hidden_states=True,
        )
    features = get_stage_features(outputs, model)
    assert len(features) == 4
    assert all(feature.ndim == 4 for feature in features)
    assert all(torch.isfinite(feature).all() for feature in features)
    stats = get_feature_stats(features)
    assert len(stats) == 4
    assert all(item["all_finite"] for item in stats)
