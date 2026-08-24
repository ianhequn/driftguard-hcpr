import torch

from driftguard.partial.stage4 import PartialStage4Engine


def _fresh_stage_1_to_3(model, pixel_values):
    encoder = model.segformer.encoder
    hidden = pixel_values
    features = []
    for index in range(3):
        hidden, height, width = encoder.patch_embeddings[index](hidden)
        for block in encoder.block[index]:
            hidden = block(hidden, height, width, False)[0]
        hidden = encoder.layer_norm[index](hidden)
        hidden = (
            hidden.reshape(hidden.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        features.append(hidden)
    return features


def test_real_partial_stage4_decoder_integration(round5_tiny_model):
    engine = PartialStage4Engine(round5_tiny_model)
    frame_1 = torch.randn(1, 3, 128, 128)
    frame_2 = frame_1 + torch.randn_like(frame_1) * 0.05
    mask = torch.zeros(16, dtype=torch.bool)
    mask[[0, 3, 5, 10, 12, 15]] = True
    with torch.inference_mode():
        features_1 = _fresh_stage_1_to_3(round5_tiny_model, frame_1)
        _, cache, _ = engine.initialize(features_1[2])
        features_2 = _fresh_stage_1_to_3(round5_tiny_model, frame_2)
        stage4, cache, stats = engine.partial_stage4(
            features_2[2],
            cache,
            mask,
        )
        logits = round5_tiny_model.decode_head([*features_2, stage4])
    assert logits.shape == (1, 3, 32, 32)
    assert stats.logical_refresh_blocks == 6
    assert stats.physical_tokens == 6
    assert cache.output_tokens.shape[1] == 16
