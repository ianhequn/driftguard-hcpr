import pytest


def test_segformer_124_head_requires_cloud_model():
    pytest.importorskip("transformers")
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=124))
    assert model.decode_head.classifier.out_channels == 124
    assert model.config.semantic_loss_ignore_index == 255
