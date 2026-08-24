import pytest


@pytest.fixture(scope="session")
def round5_tiny_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[1, 1, 1, 2],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 1, 4],
        sr_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        mlp_ratios=[2, 2, 2, 2],
        decoder_hidden_size=16,
        num_labels=3,
        reshape_last_stage=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.0,
        drop_path_rate=0.0,
    )
    torch.manual_seed(42)
    return transformers.SegformerForSemanticSegmentation(config).eval()
