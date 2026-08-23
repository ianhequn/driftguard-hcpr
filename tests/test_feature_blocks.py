import pytest


def test_4x4_and_8x8_block_partition_shapes():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.feature_blocks import split_feature_blocks

    feature = torch.randn(4, 32, 32)
    blocks_4 = split_feature_blocks(feature, grid_size=4)
    blocks_8 = split_feature_blocks(feature, grid_size=8)
    assert len(blocks_4) == 16
    assert len(blocks_8) == 64
    assert blocks_4[(0, 0)].shape == (4, 8, 8)
    assert blocks_8[(7, 7)].shape == (4, 4, 4)


def test_non_divisible_feature_shape_fails():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.feature_blocks import split_feature_blocks

    with pytest.raises(ValueError, match="not divisible"):
        split_feature_blocks(torch.randn(4, 10, 10), grid_size=4)


def test_rolling_feature_buffer_is_bounded():
    torch = pytest.importorskip("torch")
    from driftguard.analysis.feature_blocks import RollingFeatureBuffer

    buffer = RollingFeatureBuffer(max_size=3)
    for frame_id in range(10):
        buffer.add(frame_id, [torch.zeros(2, 4, 4)])
    assert len(buffer) == 3
    assert buffer.frame_ids() == (7, 8, 9)
    assert buffer.peak_size == 3
    assert buffer.get(0) is None
    assert buffer.get(9) is not None
