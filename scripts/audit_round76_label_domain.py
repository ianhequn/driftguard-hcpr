"""Audit every VSPW mask against the frozen raw-label contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.datasets.vspw import discover_vspw_samples  # noqa: E402


def inspect_mask(path: str) -> dict | None:
    with Image.open(path) as handle:
        mask = np.asarray(handle, dtype=np.uint8)
    unique = np.unique(mask)
    invalid = unique[(unique > 124) & (unique != 255)]
    if invalid.size == 0:
        return None
    invalid_pixels = np.isin(mask, invalid)
    return {
        "mask_path": path,
        "invalid_ids": [int(value) for value in invalid],
        "invalid_pixels": int(invalid_pixels.sum()),
        "total_pixels": int(mask.size),
        "all_raw_ids": [int(value) for value in unique],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "expected_raw_domain": "0..124 or 255",
        "splits": {},
        "dataset_domain_pass": True,
    }
    anomalies: list[dict] = []
    for split in args.splits:
        samples = discover_vspw_samples(args.dataset_root, split=split)
        mask_paths = [str(sample.mask_path) for sample in samples if sample.mask_path]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(inspect_mask, mask_paths, chunksize=64)
            split_anomalies = [result for result in results if result is not None]
        for row in split_anomalies:
            row["split"] = split
        anomalies.extend(split_anomalies)
        summary["splits"][split] = {
            "mask_count": len(mask_paths),
            "anomalous_mask_count": len(split_anomalies),
            "invalid_pixel_count": sum(row["invalid_pixels"] for row in split_anomalies),
        }
        summary["dataset_domain_pass"] &= not split_anomalies

    summary["anomalous_mask_count"] = len(anomalies)
    summary["invalid_pixel_count"] = sum(row["invalid_pixels"] for row in anomalies)
    (args.output_dir / "label_domain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "label_domain_anomalies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "mask_path",
                "invalid_ids",
                "invalid_pixels",
                "total_pixels",
                "all_raw_ids",
            ],
        )
        writer.writeheader()
        for row in anomalies:
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
