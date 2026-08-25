import pytest
import torch

from driftguard.training.vspw_segformer import StreamingSegmentationMetrics


def test_perfect_prediction_has_unit_miou():
    target = torch.tensor([[0, 1, 255], [1, 0, 255]])
    metrics = StreamingSegmentationMetrics()
    metrics.update(target.clone(), target)
    result = metrics.compute()
    assert result["miou"] == 1.0
    assert result["pixel_accuracy"] == 1.0
    assert result["included_classes"] == [0, 1]


def test_metric_rejects_out_of_range_target_before_bincount():
    metrics = StreamingSegmentationMetrics()
    with pytest.raises(ValueError, match=r"\[252\]"):
        metrics.update(torch.tensor([0]), torch.tensor([252]))
