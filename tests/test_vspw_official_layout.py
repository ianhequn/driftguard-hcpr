from pathlib import Path

from PIL import Image

from driftguard.datasets.vspw import discover_vspw_samples, inspect_vspw_structure


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 6), color=1).save(path)


def test_official_data_directory_and_split_file_are_respected(tmp_path):
    root = tmp_path / "VSPW_480p"
    _write_rgb(root / "data/video_train/origin/0001.jpg")
    _write_mask(root / "data/video_train/mask/0001.png")
    _write_rgb(root / "data/video_val/origin/0002.jpg")
    _write_mask(root / "data/video_val/mask/0002.png")
    (root / "train.txt").write_text("video_train\n", encoding="utf-8")
    (root / "val.txt").write_text("video_val\n", encoding="utf-8")

    samples = discover_vspw_samples(root, split="val")

    assert [(sample.video_id, sample.frame_id) for sample in samples] == [("video_val", "0002")]
    report = inspect_vspw_structure(root, split="val")
    assert report["video_count"] == 1
    assert report["sample_count"] == 1
    assert report["official_split_file"] == str(root / "val.txt")
