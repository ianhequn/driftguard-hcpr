import torch

from driftguard.structured_reuse.pipeline import StructuredReusePipeline


def test_structured_pipeline_full_refresh_matches_full_model(
    round5_tiny_model,
):
    pipeline = StructuredReusePipeline(round5_tiny_model)
    pixel_values = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        expected = round5_tiny_model(
            pixel_values=pixel_values,
            return_dict=True,
        ).logits
        actual, state, stats = pipeline.forward(
            pixel_values,
            refresh=True,
            state=None,
        )
    difference = actual - expected
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    )
    assert stats["stage4_calls_delta"] == 1
    assert state.age == 0
    assert float(difference.abs().max()) == 0.0
    assert float(difference.abs().mean()) == 0.0
    assert float(relative_l2) == 0.0
