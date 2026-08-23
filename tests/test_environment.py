import importlib.util

import pytest


def test_required_project_modules_are_importable():
    assert importlib.util.find_spec("driftguard") is not None
    assert importlib.util.find_spec("driftguard.models.segformer") is not None
    assert importlib.util.find_spec("driftguard.datasets.vspw") is not None


def test_device_selection_is_explicit():
    torch = pytest.importorskip("torch")
    from driftguard.utils.common import choose_device

    assert choose_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert choose_device("auto").type == "cuda"
    else:
        assert choose_device("auto").type == "cpu"
