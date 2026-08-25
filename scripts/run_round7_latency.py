"""Run the paired CUDA-resident Round 7 latency benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import (  # noqa: E402
    Stage4CacheState,
    StructuredReusePipeline,
)
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def autocast_context(precision: str):
    if precision == "fp32":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
        enabled=True,
    )


def load_pool(dataset, device, count):
    count = min(count, len(dataset))
    pixels = torch.stack([dataset[index]["pixel_values"] for index in range(count)]).to(device)
    video_ids = [dataset[index]["video_id"] for index in range(count)]
    return pixels, video_ids


def sequence_indices(pool_size: int, count: int):
    for index in range(count):
        yield index % pool_size, index // pool_size


def measure_full(model, pixels, warmup, measurements, precision):
    def run(count, record):
        values = []
        for index, _ in sequence_indices(pixels.shape[0], count):
            pixel = pixels[index : index + 1]
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            with torch.inference_mode(), autocast_context(precision):
                model(pixel_values=pixel).logits
            end.record()
            end.synchronize()
            if record:
                values.append({"cuda_ms": float(start.elapsed_time(end)), "wall_ms": (time.perf_counter() - wall_start) * 1000.0})
        return values

    run(warmup, False)
    torch.cuda.reset_peak_memory_stats()
    values = run(measurements, True)
    return values, int(measurements)


def measure_fixed(model, pixels, video_ids, interval, warmup, measurements, precision):
    pipeline = StructuredReusePipeline(model)

    def run(count, record):
        values = []
        state: Stage4CacheState | None = None
        refresh_count = 0
        pipeline.stage4_cache.reset_counter()
        for emitted, (index, cycle) in enumerate(sequence_indices(pixels.shape[0], count)):
            if cycle > 0 and index == 0:
                state = None
            refresh = index % interval == 0 or state is None
            pixel = pixels[index : index + 1]
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            with torch.inference_mode(), autocast_context(precision):
                _, state, stats = pipeline.forward(pixel, refresh=refresh, state=state)
            end.record()
            end.synchronize()
            actual_refresh = bool(stats["actual_refresh"])
            refresh_count += int(actual_refresh)
            if stats["stage4_calls_delta"] != int(actual_refresh):
                raise RuntimeError("Stage4 counter does not match refresh decision")
            if record:
                values.append(
                    {
                        "cuda_ms": float(start.elapsed_time(end)),
                        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
                        "refresh": int(actual_refresh),
                        "cache_age": int(stats["cache_age"]),
                    }
                )
        return values, refresh_count, pipeline.stage4_cache.stage4_forward_calls

    run(warmup, False)
    torch.cuda.reset_peak_memory_stats()
    values, refresh_count, calls = run(measurements, True)
    return values, refresh_count, calls


def summarize(values):
    cuda = np.asarray([row["cuda_ms"] for row in values], dtype=np.float64)
    wall = np.asarray([row["wall_ms"] for row in values], dtype=np.float64)
    return {
        "mean_ms": float(cuda.mean()),
        "median_ms": float(np.median(cuda)),
        "p95_ms": float(np.quantile(cuda, 0.95)),
        "p99_ms": float(np.quantile(cuda, 0.99)),
        "std_ms": float(cuda.std()),
        "wall_mean_ms": float(wall.mean()),
        "fps": float(1000.0 / cuda.mean()),
        "measurements": len(values),
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixels, video_ids = load_pool(dataset, device, args.pool)
    rows = []
    for policy in ("FI-2", "FI-3", "FI-4", "FI-6", "FI-8"):
        interval = int(policy.split("-")[1])
        policy_values, refresh_count, calls = measure_fixed(
            model, pixels, video_ids, interval, args.warmup, args.measurements, args.precision
        )
        full_values, _ = measure_full(model, pixels, args.warmup, args.measurements, args.precision)
        row = {
            "policy": policy,
            "refresh_interval": interval,
            "refresh_frames": refresh_count,
            "reuse_frames": args.measurements - refresh_count,
            "refresh_ratio": refresh_count / args.measurements,
            "stage4_calls": calls,
            "total_frames": args.measurements,
            "persistent_cache_mb": (256 * 16 * 16 * 2) / (1024**2),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "latency": summarize(policy_values),
            "paired_full_latency": summarize(full_values),
        }
        row["speedup"] = row["paired_full_latency"]["mean_ms"] / row["latency"]["mean_ms"]
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    full_values, _ = measure_full(model, pixels, args.warmup, args.measurements, args.precision)
    rows.insert(
        0,
        {
            "policy": "Full",
            "refresh_interval": 1,
            "refresh_frames": args.measurements,
            "reuse_frames": 0,
            "refresh_ratio": 1.0,
            "stage4_calls": args.measurements,
            "total_frames": args.measurements,
            "persistent_cache_mb": 0.0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "latency": summarize(full_values),
            "paired_full_latency": summarize(full_values),
            "speedup": 1.0,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"warmup": args.warmup, "measurements": args.measurements, "pool": args.pool, "precision": args.precision, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
