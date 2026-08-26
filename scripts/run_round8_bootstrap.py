"""Video-level bootstrap for the B1 Full versus FI-2 mIoU gap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["video_id"]: {
                "miou": float(row["miou"]),
                "pixel_accuracy": float(row["pixel_accuracy"]),
            }
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-global-miou", type=float)
    parser.add_argument("--reuse-global-miou", type=float)
    args = parser.parse_args()
    full = read_rows(args.full)
    reuse = read_rows(args.reuse)
    if set(full) != set(reuse):
        raise RuntimeError("Full/FI-2 video sets differ")
    videos = sorted(full)
    full_miou = np.asarray([full[key]["miou"] for key in videos], dtype=np.float64)
    reuse_miou = np.asarray([reuse[key]["miou"] for key in videos], dtype=np.float64)
    full_acc = np.asarray([full[key]["pixel_accuracy"] for key in videos], dtype=np.float64)
    reuse_acc = np.asarray([reuse[key]["pixel_accuracy"] for key in videos], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(videos), size=(args.samples, len(videos)))
    full_boot = full_miou[indices].mean(axis=1)
    reuse_boot = reuse_miou[indices].mean(axis=1)
    drop_boot_pp = 100.0 * (full_boot - reuse_boot)
    result = {
        "videos": len(videos),
        "samples": args.samples,
        "seed": args.seed,
        "official_global_miou_full": args.full_global_miou,
        "official_global_miou_fi2": args.reuse_global_miou,
        "official_global_drop_pp": (
            100.0 * (args.full_global_miou - args.reuse_global_miou)
            if args.full_global_miou is not None and args.reuse_global_miou is not None
            else None
        ),
        "video_mean_miou_full": float(full_miou.mean()),
        "video_mean_miou_fi2": float(reuse_miou.mean()),
        "video_mean_drop_pp": float(100.0 * (full_miou.mean() - reuse_miou.mean())),
        "bootstrap_drop_pp_mean": float(drop_boot_pp.mean()),
        "bootstrap_drop_pp_ci95": [
            float(np.quantile(drop_boot_pp, 0.025)),
            float(np.quantile(drop_boot_pp, 0.975)),
        ],
        "bootstrap_full_video_mean_miou_ci95": [
            float(np.quantile(full_boot, 0.025)),
            float(np.quantile(full_boot, 0.975)),
        ],
        "bootstrap_fi2_video_mean_miou_ci95": [
            float(np.quantile(reuse_boot, 0.025)),
            float(np.quantile(reuse_boot, 0.975)),
        ],
        "video_mean_pixel_accuracy_full": float(full_acc.mean()),
        "video_mean_pixel_accuracy_fi2": float(reuse_acc.mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
