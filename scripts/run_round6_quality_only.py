"""Run the frozen-test quality pass without latency instrumentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_round6_formal_test import (  # noqa: E402
    VISUAL_AGGREGATION,
    VISUAL_THRESHOLD,
    STAGE_AGE_VISUAL_MAX_AGE,
    load_predictor,
    load_test_videos,
    make_specs,
    quality_pass,
    read_csv,
)
from driftguard.models.segformer import load_segformer  # noqa: E402
from driftguard.utils.common import set_seed  # noqa: E402


def main() -> None:
    set_seed(42)
    device = torch.device("cuda")
    records = read_csv(
        PROJECT_ROOT / "results/R2_E00_video_dataset/video_inventory.csv"
    )
    processor, model = load_segformer(
        "/root/autodl-tmp/models/segformer-b0",
        device=device,
        local_files_only=True,
    )
    predictor, predictor_mean, predictor_std = load_predictor(
        PROJECT_ROOT / "results/R2_E03_ablation/patch_age.pt",
        device,
    )
    videos = load_test_videos(records, processor, device, max_frames=48)
    rows = quality_pass(
        model,
        predictor,
        predictor_mean,
        predictor_std,
        videos,
        make_specs(),
    )
    root = PROJECT_ROOT / "results/R6_E06_task_quality"
    root.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    import csv

    with (root / "test_frame_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "test_videos": [video.video_id for video in videos],
                "aggregation": VISUAL_AGGREGATION,
                "visual_threshold": VISUAL_THRESHOLD,
                "stage_age_visual_max_age": STAGE_AGE_VISUAL_MAX_AGE,
                "frames": len(rows),
                "quality_only_clean_process": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "split: test\ninput: cuda_resident_fp32_pixel_values\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (root / "runtime.txt").write_text(
        "Completed one frozen-test quality pass in a clean process.\n",
        encoding="utf-8",
    )
    print(f"quality_rows={len(rows)}")


if __name__ == "__main__":
    main()
