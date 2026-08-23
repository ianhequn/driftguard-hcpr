import pytest


def test_sequential_continuation_matches_full_encoder_for_current_features():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from driftguard.analysis.sequential_stale import sequential_stale_features
    from driftguard.models.segformer import get_stage_features

    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=8,
        num_labels=3,
        reshape_last_stage=True,
    )
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    pixel_values = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        outputs = model.segformer(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
    current = get_stage_features(outputs, model=model)
    for cut in range(1, 5):
        reconstructed = sequential_stale_features(
            model=model,
            current_features=current,
            cached_features=current,
            cut=cut,
        )
        assert len(reconstructed) == 4
        for expected, actual in zip(current, reconstructed):
            assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


def test_sequential_cut_changes_only_selected_cached_stage_when_cache_differs():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from driftguard.analysis.sequential_stale import sequential_stale_features

    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 1, 1],
        decoder_hidden_size=8,
        num_labels=3,
        reshape_last_stage=True,
    )
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    current = [torch.full((1, channels, size, size), float(index + 1))
               for index, (channels, size) in enumerate(
                   [(8, 16), (16, 8), (32, 4), (64, 2)]
               )]
    cached = [feature + 100.0 for feature in current]
    for cut in range(1, 5):
        result = sequential_stale_features(model, current, cached, cut)
        for index, (expected, actual) in enumerate(zip(current, result), start=1):
            if index == cut:
                assert torch.allclose(actual, cached[index - 1])
            elif index < cut:
                assert torch.allclose(actual, expected)
