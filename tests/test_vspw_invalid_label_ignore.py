import torch
import torch.nn.functional as F

from driftguard.training.vspw_segformer import StreamingSegmentationMetrics


def test_canonical_ignore_pixels_do_not_contribute_to_cross_entropy():
    logits_a = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    logits_b = torch.tensor([[4.0, 0.0], [-100.0, 100.0]])
    target = torch.tensor([0, 255])
    loss_a = F.cross_entropy(logits_a, target, ignore_index=255)
    loss_b = F.cross_entropy(logits_b, target, ignore_index=255)
    assert torch.allclose(loss_a, loss_b)


def test_miou_is_invariant_to_prediction_on_canonical_void_pixel():
    target = torch.tensor([0, 1, 255])
    first = StreamingSegmentationMetrics()
    second = StreamingSegmentationMetrics()
    first.update(torch.tensor([0, 1, 0]), target)
    second.update(torch.tensor([0, 1, 123]), target)
    assert first.compute()["miou"] == second.compute()["miou"] == 1.0


def test_miou_changes_when_a_valid_prediction_is_wrong():
    target = torch.tensor([0, 1])
    perfect = StreamingSegmentationMetrics()
    wrong = StreamingSegmentationMetrics()
    perfect.update(torch.tensor([0, 1]), target)
    wrong.update(torch.tensor([0, 0]), target)
    assert perfect.compute()["miou"] > wrong.compute()["miou"]
