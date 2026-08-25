from pathlib import Path

import torch
from PIL import Image

from driftguard.training.vspw_segformer import VSPWFrameDataset


def test_training_dataset_shapes_and_nearest_labels(tmp_path: Path):
    root = tmp_path / "vspw"
    video = root / "data" / "v1"
    (video / "origin").mkdir(parents=True)
    (video / "mask").mkdir()
    Image.new("RGB", (7, 5), color=(10, 20, 30)).save(video / "origin" / "1.jpg")
    mask = Image.new("L", (7, 5), color=124)
    mask.putpixel((0, 0), 0)
    mask.save(video / "mask" / "1.png")
    (root / "train.txt").write_text("v1\n", encoding="utf-8")
    dataset = VSPWFrameDataset(root, "train", image_size=16, training=False)
    item = dataset[0]
    assert item["pixel_values"].shape == (3, 16, 16)
    assert item["labels"].shape == (16, 16)
    assert item["labels"].dtype == torch.int64
    assert set(item["labels"].unique().tolist()) <= {123, 255}
