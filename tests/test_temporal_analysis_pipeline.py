import pytest


def test_small_temporal_analysis_pipeline():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.temporal_similarity import analyze_feature_stream

    frames = []
    for frame_id in range(5):
        value = float(frame_id)
        frames.append(
            (
                frame_id,
                [
                    torch.full((1, 8, 8), value),
                    torch.full((1, 8, 8), value),
                    torch.full((1, 8, 8), value),
                    torch.full((1, 8, 8), value),
                ],
            )
        )

    result = analyze_feature_stream(
        frames,
        offsets=[1, 2],
        block_grid=4,
        video_id="synthetic",
    )
    assert result["processed_frames"] == 5
    assert result["buffer_max_size"] == 3
    assert result["buffer_peak_size"] == 3
    assert len(result["pair_rows"]) == (4 + 3) * 4
    assert len(result["block_rows"]) == (4 + 3) * 4 * 16
    assert len(result["summary_rows"]) == 2 * 4
    assert result["heatmap"] is not None
