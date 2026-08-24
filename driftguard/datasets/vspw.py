"""VSPW frame discovery and label normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_IMAGE_DIR_NAMES = ("origin", "images", "img")
DEFAULT_MASK_DIR_NAMES = ("mask", "masks", "label", "labels")


def normalize_vspw_mask(mask):
    """Apply the official VSPW void/class-id normalization rule."""

    normalized = mask.copy()
    normalized[normalized == 0] = 255
    normalized = normalized - 1
    normalized[normalized == 254] = 255
    return normalized


@dataclass(frozen=True)
class VSPWSample:
    video_id: str
    frame_id: str
    image_path: Path
    mask_path: Path | None


def _find_child(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def _image_files(directory: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def resolve_vspw_data_root(root: str | Path) -> Path:
    """Return the directory containing official per-video folders."""

    root = Path(root)
    data_root = root / "data"
    return data_root if data_root.is_dir() else root


def read_vspw_split(root: str | Path, split: str) -> list[str] | None:
    """Read an official VSPW split file when it is present."""

    root = Path(root)
    split_file = root / f"{split}.txt"
    if not split_file.is_file():
        return None
    video_ids = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8-sig").splitlines()
    ]
    return [video_id for video_id in video_ids if video_id]


def discover_vspw_samples(
    root: str | Path,
    split: str = "train",
    image_dir_names: tuple[str, ...] = DEFAULT_IMAGE_DIR_NAMES,
    mask_dir_names: tuple[str, ...] = DEFAULT_MASK_DIR_NAMES,
) -> list[VSPWSample]:
    root = Path(root)
    data_root = resolve_vspw_data_root(root)
    split_video_ids = read_vspw_split(root, split)
    if split_video_ids is None:
        legacy_split_root = root / split
        split_root = legacy_split_root if legacy_split_root.is_dir() else data_root
        video_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    else:
        split_root = data_root
        video_dirs = [data_root / video_id for video_id in split_video_ids]
    samples: list[VSPWSample] = []
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue
        image_dir = _find_child(video_dir, image_dir_names)
        if image_dir is None:
            continue
        mask_dir = _find_child(video_dir, mask_dir_names)
        masks = {path.stem: path for path in _image_files(mask_dir)} if mask_dir else {}
        for image_path in _image_files(image_dir):
            samples.append(
                VSPWSample(
                    video_id=video_dir.name,
                    frame_id=image_path.stem,
                    image_path=image_path,
                    mask_path=masks.get(image_path.stem),
                )
            )
    return samples


def inspect_vspw_structure(root: str | Path, split: str = "train") -> dict[str, Any]:
    root = Path(root)
    exists = root.is_dir()
    data_root = resolve_vspw_data_root(root)
    split_video_ids = read_vspw_split(root, split) if exists else None
    legacy_split_root = root / split
    split_root = legacy_split_root if legacy_split_root.is_dir() else data_root
    videos = []
    if exists:
        if split_video_ids is None:
            video_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        else:
            video_dirs = [data_root / video_id for video_id in split_video_ids]
        for video_dir in video_dirs:
            image_dir = _find_child(video_dir, DEFAULT_IMAGE_DIR_NAMES)
            mask_dir = _find_child(video_dir, DEFAULT_MASK_DIR_NAMES)
            videos.append(
                {
                    "video_id": video_dir.name,
                    "exists": video_dir.is_dir(),
                    "image_dir": str(image_dir) if image_dir else None,
                    "mask_dir": str(mask_dir) if mask_dir else None,
                    "image_count": len(_image_files(image_dir)) if image_dir else 0,
                    "mask_count": len(_image_files(mask_dir)) if mask_dir else 0,
                }
            )
    samples = discover_vspw_samples(root, split=split) if exists else []
    return {
        "root": str(root),
        "split": split,
        "exists": exists,
        "split_root": str(split_root),
        "official_split_file": str(root / f"{split}.txt") if split_video_ids is not None else None,
        "video_count": len(videos),
        "sample_count": len(samples),
        "videos": videos,
        "warnings": (
            []
            if samples
            else [
                "No image samples were discovered. Check the root, split, and "
                "origin/mask directory names."
            ]
        ),
    }


class VSPWDataset(Dataset):
    """A minimal frame-level VSPW dataset.

    Each item returns a dictionary with a PIL RGB image, an optional PIL mask,
    and identifiers needed by later temporal experiments.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        require_masks: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = discover_vspw_samples(self.root, split=split)
        if require_masks:
            self.samples = [sample for sample in self.samples if sample.mask_path is not None]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path) if sample.mask_path else None
        item: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "video_id": sample.video_id,
            "frame_id": sample.frame_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path else None,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
