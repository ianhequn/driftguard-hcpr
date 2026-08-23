import pytest


def test_tiny_segformer_forward_without_download():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from driftguard.models.segformer import get_stage_features

    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 2, 4, 8],
        sr_ratios=[8, 4, 2, 1],
        hidden_act="gelu",
        decoder_hidden_size=32,
        num_labels=3,
    )
    model = transformers.SegformerForSemanticSegmentation(config).eval()
    pixel_values = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == 3
    assert torch.isfinite(outputs.logits).all()
