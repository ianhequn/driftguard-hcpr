"""Build the deterministic Round 7.5 audit artifacts for official VSPW."""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
from collections import Counter
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _split_ids(root: Path, split: str) -> list[str]:
    path = root / f"{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _files_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _jpeg_size(handle: BinaryIO) -> tuple[int, int]:
    if handle.read(2) != b"\xff\xd8":
        raise ValueError("missing JPEG SOI marker")
    while True:
        marker_start = handle.read(1)
        if not marker_start:
            raise ValueError("JPEG SOF marker not found")
        if marker_start != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        marker_value = marker[0]
        if marker_value in {0xD8, 0xD9}:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise ValueError("truncated JPEG segment")
        segment_length = struct.unpack(">H", length_bytes)[0]
        if marker_value in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            payload = handle.read(5)
            if len(payload) != 5:
                raise ValueError("truncated JPEG SOF segment")
            height, width = struct.unpack(">HH", payload[1:5])
            return width, height
        handle.seek(segment_length - 2, 1)


def image_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            if len(signature) < 24:
                raise ValueError("truncated PNG header")
            return struct.unpack(">II", signature[16:24])
        handle.seek(0)
        return _jpeg_size(handle)


def _relative(path: Path | None, root: Path) -> str:
    return path.relative_to(root).as_posix() if path else ""


def _representative_size(
    files: dict[str, Path], problems: list[dict[str, str]]
) -> tuple[int | None, int | None]:
    for path in files.values():
        if path.stat().st_size == 0:
            problems.append({"path": str(path), "error": "zero-byte file"})
            continue
        try:
            return image_size(path)
        except (OSError, ValueError, IndexError) as exc:
            problems.append({"path": str(path), "error": str(exc)})
    return None, None


def _sample_masks(mask_paths: list[tuple[str, str, Path]], count: int, seed: int) -> list[tuple[str, str, Path]]:
    by_video: dict[str, list[tuple[str, str, Path]]] = {}
    for item in mask_paths:
        by_video.setdefault(item[0], []).append(item)
    rng = random.Random(seed)
    videos = sorted(by_video)
    rng.shuffle(videos)
    selected: list[tuple[str, str, Path]] = []
    for video_id in videos[: min(count, len(videos))]:
        selected.append(rng.choice(by_video[video_id]))
    if len(selected) < count:
        remaining = [item for item in mask_paths if item not in selected]
        selected.extend(rng.sample(remaining, min(count - len(selected), len(remaining))))
    return sorted(selected)


def audit(root: Path, output: Path, sample_count: int, seed: int) -> dict[str, object]:
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"official data directory not found: {data_root}")

    taxonomy_path = root / "label_num_dic_final.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy_ids = {int(value) for value in taxonomy.values()}
    taxonomy_valid = len(taxonomy) == 125 and taxonomy_ids == set(range(125))
    num_classes = len(taxonomy_ids - {0})

    split_lists = {split: _split_ids(root, split) for split in SPLITS}
    split_sets = {split: set(ids) for split, ids in split_lists.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("official split files overlap")

    directory_ids = {path.name for path in data_root.iterdir() if path.is_dir()}
    split_union = set().union(*split_sets.values())
    split_errors = {
        "missing_video_directories": sorted(split_union - directory_ids),
        "unassigned_video_directories": sorted(directory_ids - split_union),
    }

    manifest_path = output / "dataset_manifest.csv"
    mask_paths: list[tuple[str, str, Path]] = []
    split_counts = {split: Counter() for split in SPLITS}
    totals = Counter()
    problems: list[dict[str, str]] = []
    fieldnames = [
        "video_id", "frame_id", "rgb_path", "mask_path", "has_rgb", "has_mask",
        "width", "height", "mask_width", "mask_height", "split", "file_size",
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLITS:
            for video_id in split_lists[split]:
                video_dir = data_root / video_id
                rgb_files = _files_by_stem(video_dir / "origin")
                mask_files = _files_by_stem(video_dir / "mask")
                stems = sorted(set(rgb_files) | set(mask_files))
                split_counts[split]["videos"] += 1
                rgb_size = _representative_size(rgb_files, problems)
                mask_size = _representative_size(mask_files, problems)
                for frame_id in stems:
                    rgb_path = rgb_files.get(frame_id)
                    mask_path = mask_files.get(frame_id)
                    rgb_file_size = rgb_path.stat().st_size if rgb_path else 0
                    if rgb_path and rgb_file_size == 0:
                        problems.append({"path": _relative(rgb_path, root), "error": "zero-byte file"})
                    if mask_path and mask_path.stat().st_size == 0:
                        problems.append({"path": _relative(mask_path, root), "error": "zero-byte file"})
                    if mask_path:
                        mask_paths.append((video_id, frame_id, mask_path))
                    has_rgb = rgb_path is not None
                    has_mask = mask_path is not None
                    split_counts[split]["rgb_frames"] += int(has_rgb)
                    split_counts[split]["masks"] += int(has_mask)
                    split_counts[split]["paired_frames"] += int(has_rgb and has_mask)
                    totals["rows"] += 1
                    totals["rgb_frames"] += int(has_rgb)
                    totals["masks"] += int(has_mask)
                    totals["paired_frames"] += int(has_rgb and has_mask)
                    writer.writerow(
                        {
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "rgb_path": _relative(rgb_path, root),
                            "mask_path": _relative(mask_path, root),
                            "has_rgb": int(has_rgb),
                            "has_mask": int(has_mask),
                            "width": rgb_size[0],
                            "height": rgb_size[1],
                            "mask_width": mask_size[0],
                            "mask_height": mask_size[1],
                            "split": split,
                            "file_size": rgb_file_size,
                        }
                    )

    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "videos", "rgb_frames", "masks", "paired_frames"])
        writer.writeheader()
        for split in SPLITS:
            writer.writerow({"split": split, **split_counts[split]})

    sampled = _sample_masks(mask_paths, sample_count, seed)
    observed_raw: set[int] = set()
    observed_normalized: set[int] = set()
    mask_rows = []
    for video_id, frame_id, path in sampled:
        with Image.open(path) as image:
            image.load()
            mask = np.asarray(image)
        raw_values = sorted(int(value) for value in np.unique(mask))
        normalized = mask.copy()
        normalized[normalized == 0] = 255
        normalized = normalized - 1
        normalized[normalized == 254] = 255
        normalized_values = sorted(int(value) for value in np.unique(normalized))
        observed_raw.update(raw_values)
        observed_normalized.update(normalized_values)
        mask_rows.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "mask_path": _relative(path, root),
                "width": mask.shape[1],
                "height": mask.shape[0],
                "raw_unique_labels": " ".join(map(str, raw_values)),
                "normalized_unique_labels": " ".join(map(str, normalized_values)),
            }
        )
    with (output / "mask_semantics_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mask_rows[0]) if mask_rows else [])
        if mask_rows:
            writer.writeheader()
            writer.writerows(mask_rows)

    valid_raw = taxonomy_ids | {255}
    valid_normalized = set(range(num_classes)) | {255}
    summary: dict[str, object] = {
        "dataset": "VSPW",
        "source": "official VSPW_480p package",
        "download_status": "installed",
        "root": str(root),
        "package": "VSPW_480p.zip",
        "total_size_gb": round(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1e9, 6),
        "num_videos": len(directory_ids),
        "num_frames": totals["rgb_frames"],
        "num_masks": totals["masks"],
        "paired_frames": totals["paired_frames"],
        "manifest_rows": totals["rows"],
        "num_classes": num_classes,
        "ignore_index": 255,
        "taxonomy_file": taxonomy_path.name,
        "taxonomy_entries": len(taxonomy),
        "taxonomy_raw_ids": sorted(taxonomy_ids),
        "taxonomy_verified": taxonomy_valid,
        "split_status": "verified" if not any(split_errors.values()) else "failed",
        "manifest_status": "complete" if not problems else "has_file_problems",
        "verification_scope": {
            "all_files": "path, stem pairing, and byte size",
            "per_video": "representative RGB and mask header dimensions",
            "sampled_masks": "full Pillow decode and label-value audit",
        },
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "split_errors": split_errors,
        "unreadable_headers": problems,
        "sampled_mask_count": len(sampled),
        "sampled_video_count": len({row["video_id"] for row in mask_rows}),
        "observed_raw_labels": sorted(observed_raw),
        "observed_normalized_labels": sorted(observed_normalized),
        "raw_labels_valid": observed_raw <= valid_raw,
        "normalized_labels_valid": observed_normalized <= valid_normalized,
        "label_semantics_verified": (
            taxonomy_valid and observed_raw <= valid_raw and observed_normalized <= valid_normalized
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/R75_E00_vspw_dataset")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=75)
    args = parser.parse_args()
    summary = audit(args.root, args.output, args.sample_count, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
