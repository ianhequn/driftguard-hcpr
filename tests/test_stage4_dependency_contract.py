def test_stage4_dependency_contract(round5_tiny_model):
    encoder = round5_tiny_model.segformer.encoder
    assert encoder.config.sr_ratios[3] == 1
    assert len(encoder.block[3]) == 2
    for block in encoder.block[3]:
        attention = block.attention.self
        assert attention.sr_ratio == 1
        assert not hasattr(attention, "sr")
        depthwise = block.mlp.dwconv.dwconv
        assert depthwise.kernel_size == (3, 3)
        assert depthwise.padding == (1, 1)
        assert depthwise.groups == depthwise.in_channels
