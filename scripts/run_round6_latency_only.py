"""Run the clean-process sequential latency pass for Round 6."""

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
    latency_pass,
    make_specs,
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
    rows = latency_pass(
        model,
        predictor,
        predictor_mean,
        predictor_std,
        videos,
        make_specs(),
        warmup=200,
        measurements=1000,
    )
    root = PROJECT_ROOT / "results/R6_E05_sequential_latency"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latency_metrics.csv").write_text(
        "method,mean_ms,median_ms,p90_ms,p95_ms,p99_ms,std_ms,"
        "actual_refresh_rate,speedup,stage4_forward_calls,persistent_cache_mb,"
        "selector_mean_ms,selector_p95_ms,refresh_mean_ms,reuse_mean_ms,fps\n"
        + "\n".join(
            ",".join(
                str(row.get(key, ""))
                for key in (
                    "method",
                    "mean_ms",
                    "median_ms",
                    "p90_ms",
                    "p95_ms",
                    "p99_ms",
                    "std_ms",
                    "actual_refresh_rate",
                    "speedup",
                    "stage4_forward_calls",
                    "persistent_cache_mb",
                    "selector_mean_ms",
                    "selector_p95_ms",
                    "refresh_mean_ms",
                    "reuse_mean_ms",
                    "fps",
                )
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "warmup": 200,
                "measurements": 1000,
                "aggregation": VISUAL_AGGREGATION,
                "visual_threshold": VISUAL_THRESHOLD,
                "stage_age_visual_max_age": STAGE_AGE_VISUAL_MAX_AGE,
                "latency": rows,
                "process_boundary": "clean latency-only process",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "split: test\nwarmup: 200\nmeasurements: 1000\n"
        "input: cuda_resident_fp32_pixel_values\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (root / "runtime.txt").write_text(
        "Latency measured in a clean process before quality evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
