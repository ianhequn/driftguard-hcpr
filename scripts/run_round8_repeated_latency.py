"""Repeated paired CUDA latency for frozen B1 whole-Stage4 reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_round7_latency import (  # noqa: E402
    load_pool,
    measure_fixed,
    measure_full,
    summarize,
)
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Round 8 latency requires CUDA")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixels, video_ids = load_pool(dataset, torch.device("cuda"), args.pool)
    policy_intervals = {"FI-2": 2, "FI-4": 4, "FI-8": 8}
    repetitions = []
    for repetition in range(args.repetitions):
        for policy, interval in policy_intervals.items():
            policy_values, refresh_count, calls = measure_fixed(
                model,
                pixels,
                video_ids,
                interval,
                args.warmup,
                args.measurements,
                args.precision,
            )
            full_values, _ = measure_full(
                model, pixels, args.warmup, args.measurements, args.precision
            )
            policy_summary = summarize(policy_values)
            full_summary = summarize(full_values)
            repetitions.append(
                {
                    "repetition": repetition + 1,
                    "policy": policy,
                    "interval": interval,
                    "refresh_count": refresh_count,
                    "refresh_ratio": refresh_count / args.measurements,
                    "stage4_calls": calls,
                    "policy_latency": policy_summary,
                    "paired_full_latency": full_summary,
                    "paired_speedup": full_summary["mean_ms"] / policy_summary["mean_ms"],
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
                }
            )
            print(json.dumps(repetitions[-1], indent=2), flush=True)
    full_rows = []
    for row in repetitions:
        full_rows.append(
            {
                "repetition": row["repetition"],
                "policy": "Full",
                "latency": row["paired_full_latency"],
                "paired_speedup": 1.0,
            }
        )
    payload = {
        "architecture": "SegFormer-B1",
        "warmup": args.warmup,
        "measurements": args.measurements,
        "repetitions_requested": args.repetitions,
        "pool": args.pool,
        "precision": args.precision,
        "sequence_protocol": "same frozen pool and paired Full immediately after each reuse run",
        "rows": repetitions,
        "paired_full_rows": full_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
