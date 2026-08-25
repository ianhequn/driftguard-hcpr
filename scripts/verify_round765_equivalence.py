"""Verify project canonical labels reproduce the official evaluator mask."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.datasets.vspw import (
    VSPW_IGNORE_INDEX,
    VSPW_NUM_CLASSES,
    normalize_vspw_mask,
    official_normalize_vspw_mask,
    vspw_official_valid_mask,
)


def confusion(target: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    encoded = target[valid].astype(np.int64) * VSPW_NUM_CLASSES + prediction[valid]
    return np.bincount(encoded, minlength=VSPW_NUM_CLASSES**2).reshape(
        VSPW_NUM_CLASSES, VSPW_NUM_CLASSES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anomalies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.array([0, 1, 124, 253, 255], dtype=np.uint8)
    official = official_normalize_vspw_mask(raw)
    canonical = normalize_vspw_mask(raw)
    official_valid = vspw_official_valid_mask(official)
    project_valid = canonical != VSPW_IGNORE_INDEX
    prediction = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    synthetic_confusion_equal = np.array_equal(
        confusion(official, prediction, official_valid),
        confusion(canonical, prediction, project_valid),
    )
    synthetic = {
        "raw": raw.tolist(),
        "official_normalized": official.tolist(),
        "canonical_normalized": canonical.tolist(),
        "official_valid_mask": official_valid.tolist(),
        "project_valid_mask": project_valid.tolist(),
        "valid_mask_equal": bool(np.array_equal(official_valid, project_valid)),
        "confusion_equal": bool(synthetic_confusion_equal),
    }

    real_rows = []
    with args.anomalies.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            with Image.open(path) as image:
                raw_mask = np.asarray(image, dtype=np.uint8)
            official_mask = official_normalize_vspw_mask(raw_mask)
            canonical_mask = normalize_vspw_mask(raw_mask)
            official_valid = vspw_official_valid_mask(official_mask)
            project_valid = canonical_mask != VSPW_IGNORE_INDEX
            prediction = np.arange(raw_mask.size, dtype=np.int64).reshape(raw_mask.shape) % VSPW_NUM_CLASSES
            official_matrix = confusion(official_mask, prediction, official_valid)
            project_matrix = confusion(canonical_mask, prediction, project_valid)
            real_rows.append(
                {
                    "video_id": row["video_id"],
                    "frame": row["frame"],
                    "path": row["path"],
                    "valid_pixel_count_official": int(official_valid.sum()),
                    "valid_pixel_count_project": int(project_valid.sum()),
                    "valid_mask_equal": bool(np.array_equal(official_valid, project_valid)),
                    "confusion_equal": bool(np.array_equal(official_matrix, project_matrix)),
                }
            )

    real_pass = all(row["valid_mask_equal"] and row["confusion_equal"] for row in real_rows)
    summary = {
        "synthetic": synthetic,
        "real_anomaly_masks": len(real_rows),
        "real_valid_mask_all_equal": real_pass,
        "real_confusion_all_equal": real_pass,
        "equivalence_pass": bool(synthetic["valid_mask_equal"] and synthetic["confusion_equal"] and real_pass),
    }
    (args.output_dir / "evaluator_equivalence.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "real_mask_equivalence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(real_rows[0]))
        writer.writeheader()
        writer.writerows(real_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
