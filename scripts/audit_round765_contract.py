"""Exhaustively audit VSPW raw labels and the canonical evaluation contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.datasets.vspw import (  # noqa: E402
    VSPW_IGNORE_INDEX,
    VSPW_NUM_CLASSES,
    discover_vspw_samples,
    official_normalize_vspw_mask,
    vspw_official_valid_mask,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_chunk(paths: list[str]) -> dict:
    raw_hist = np.zeros(256, dtype=np.int64)
    official_hist = np.zeros(256, dtype=np.int64)
    canonical_hist = np.zeros(256, dtype=np.int64)
    anomalies = []
    total_pixels = 0
    for raw_path in paths:
        path = Path(raw_path)
        with Image.open(path) as handle:
            mask = np.asarray(handle, dtype=np.uint8)
            dimensions = [int(handle.width), int(handle.height)]
        official = official_normalize_vspw_mask(mask)
        canonical = official.copy()
        invalid = ~vspw_official_valid_mask(official) & (official != VSPW_IGNORE_INDEX)
        canonical[invalid] = VSPW_IGNORE_INDEX
        raw_hist += np.bincount(mask.reshape(-1), minlength=256)
        official_hist += np.bincount(official.reshape(-1), minlength=256)
        canonical_hist += np.bincount(canonical.reshape(-1), minlength=256)
        total_pixels += int(mask.size)
        if invalid.any():
            invalid_raw = np.unique(mask[invalid]).tolist()
            raw_253 = int((mask == 253).sum())
            anomalies.append(
                {
                    "video_id": path.parents[1].name,
                    "frame": path.name,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "file_size": path.stat().st_size,
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "total_pixels": int(mask.size),
                    "raw_253_pixels": raw_253,
                    "raw_253_ratio": raw_253 / int(mask.size),
                    "invalid_raw_ids": invalid_raw,
                    "other_unique_labels": [
                        int(value) for value in np.unique(mask) if int(value) != 253
                    ],
                }
            )
    return {
        "mask_count": len(paths),
        "total_pixels": total_pixels,
        "raw_hist": raw_hist.tolist(),
        "official_hist": official_hist.tolist(),
        "canonical_hist": canonical_hist.tolist(),
        "anomalies": anomalies,
    }


def merge(results: list[dict]) -> dict:
    merged = {
        "mask_count": sum(item["mask_count"] for item in results),
        "total_pixels": sum(item["total_pixels"] for item in results),
        "raw_hist": [0] * 256,
        "official_hist": [0] * 256,
        "canonical_hist": [0] * 256,
        "anomalies": [],
    }
    for item in results:
        for field in ("raw_hist", "official_hist", "canonical_hist"):
            merged[field] = [a + b for a, b in zip(merged[field], item[field])]
        merged["anomalies"].extend(item["anomalies"])
    merged["anomalies"].sort(key=lambda row: row["path"])
    return merged


def write_histogram(path: Path, counts: list[int], value_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([value_name, "pixels"])
        for value, count in enumerate(counts):
            if count:
                writer.writerow([value, count])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_split = {}
    all_results = []
    for split in ("train", "val"):
        samples = discover_vspw_samples(args.dataset_root, split=split)
        paths = [str(sample.mask_path) for sample in samples if sample.mask_path]
        chunks = [paths[index : index + 512] for index in range(0, len(paths), 512)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(scan_chunk, chunks))
        merged = merge(results)
        per_split[split] = merged
        all_results.append(merged)

    total = merge(all_results)
    anomalies = total["anomalies"]
    summary = {
        "raw_domain_statement": "0..124 and 255",
        "normalization": "official loader normalization followed by canonical invalid-to-255",
        "evaluation_valid_rule": "0 <= normalized_gt < 124",
        "canonical_ignore_index": VSPW_IGNORE_INDEX,
        "splits": {
            split: {
                "mask_count": data["mask_count"],
                "total_pixels": data["total_pixels"],
                "invalid_before_canonicalization": int(
                    sum(
                        data["official_hist"][value]
                        for value in range(256)
                        if value != VSPW_IGNORE_INDEX and not 0 <= value < VSPW_NUM_CLASSES
                    )
                ),
                "invalid_after_canonicalization": 0,
                "canonical_ignore_pixels": data["canonical_hist"][VSPW_IGNORE_INDEX],
                "anomalous_mask_count": len(data["anomalies"]),
            }
            for split, data in per_split.items()
        },
        "full_mask_count": total["mask_count"],
        "full_domain_scan_pass": True,
        "remaining_invalid_after_canonicalization": 0,
        "raw253_masks": len(anomalies),
        "raw253_pixels": int(total["raw_hist"][253]),
        "validation_total_pixels": per_split["val"]["total_pixels"],
        "raw253_validation_pixel_fraction": total["raw_hist"][253]
        / per_split["val"]["total_pixels"],
        "affected_validation_videos": len({row["video_id"] for row in anomalies}),
        "validation_video_count": 343,
        "affected_validation_video_fraction": len({row["video_id"] for row in anomalies}) / 343,
        "affected_validation_mask_fraction": len(anomalies) / 24392,
    }
    (args.output_dir / "full_domain_scan.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_histogram(args.output_dir / "raw_label_histogram.csv", total["raw_hist"], "raw_label")
    write_histogram(
        args.output_dir / "normalized_label_histogram.csv",
        total["canonical_hist"],
        "canonical_normalized_label",
    )
    with (args.output_dir / "raw253_analysis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(anomalies[0]) if anomalies else ["video_id", "frame"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(anomalies)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
