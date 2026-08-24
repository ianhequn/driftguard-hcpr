from driftguard.structured_reuse.pipeline import Stage4CacheState


def test_new_video_uses_empty_stage4_state():
    first_video = Stage4CacheState()
    second_video = Stage4CacheState()
    assert first_video.initialized is False
    assert second_video.initialized is False
    assert first_video.cached_f4 is None
    assert second_video.cached_f4 is None
