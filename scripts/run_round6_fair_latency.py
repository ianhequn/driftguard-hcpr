"""Fair Round 6 latency: each policy is paired with a fresh Full baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_round6_formal_test import (  # noqa: E402
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
    specs = make_specs()
    full_spec = next(spec for spec in specs if spec["name"] == "Full")
    paired = []
    for spec in specs:
        policy_rows = latency_pass(
            model,
            predictor,
            predictor_mean,
            predictor_std,
            videos,
            [spec],
            warmup=200,
            measurements=1000,
        )
        if spec["name"] == "Full":
            full_row = policy_rows[0]
        else:
            full_row = latency_pass(
                model,
                predictor,
                predictor_mean,
                predictor_std,
                videos,
                [full_spec],
                warmup=200,
                measurements=1000,
            )[0]
        policy_row = dict(policy_rows[0])
        policy_row["paired_full_mean_ms"] = full_row["mean_ms"]
        policy_row["paired_full_p95_ms"] = full_row["p95_ms"]
        policy_row["paired_speedup"] = (
            full_row["mean_ms"] / policy_row["mean_ms"]
        )
        paired.append(policy_row)
        print(
            spec["name"],
            "policy_mean_ms=",
            policy_row["mean_ms"],
            "paired_full_mean_ms=",
            full_row["mean_ms"],
            "paired_speedup=",
            policy_row["paired_speedup"],
            flush=True,
        )
    root = PROJECT_ROOT / "results/R6_E05_sequential_latency"
    root.mkdir(parents=True, exist_ok=True)
    (root / "fair_latency_summary.json").write_text(
        json.dumps(
            {
                "warmup": 200,
                "measurements": 1000,
                "pairing": (
                    "Each policy followed by a fresh Full baseline in the "
                    "same process and same frozen test sequence."
                ),
                "rows": paired,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
