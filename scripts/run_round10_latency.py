"""Five-repeat paired CUDA latency matrix for Round 10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from driftguard.structured_reuse.boundaries import BoundaryCacheState, StructuredBoundaryReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--pool", type=int, default=256)
    return parser.parse_args()


def load_pool(dataset, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    return torch.stack([sample["pixel_values"] for sample in samples]).to("cuda"), [sample["video_id"] for sample in samples]


def timing_call(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record(); end.synchronize()
    return float(start.elapsed_time(end))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "p99_ms": float(np.quantile(values, .99)),
            "std_ms": float(values.std()), "fps": float(1000.0 / values.mean()), "measurements": len(values)}


def run_sequence(model, pipeline, pixels, video_ids, boundary, interval, count, record):
    values = []
    state = None
    previous_video = None
    positions = {}
    refreshes = 0
    for emitted in range(count):
        index = emitted % pixels.shape[0]
        video_id = video_ids[index]
        if video_id != previous_video:
            state = None
            positions[video_id] = 0
            previous_video = video_id
        position = positions[video_id]
        positions[video_id] += 1
        refresh = boundary == "full" or position % interval == 0 or state is None
        pixel = pixels[index:index + 1]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if boundary == "full":
                elapsed = timing_call(lambda: model(pixel_values=pixel).logits)
            else:
                result = {}
                elapsed = timing_call(lambda: result.setdefault("value", pipeline.forward(pixel, refresh=refresh, state=state)))
                _, state, _ = result["value"]
        refreshes += int(refresh)
        if record:
            values.append(elapsed)
    return values, refreshes


def bootstrap(values, seed=42, samples=10000):
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main():
    args = parse_args()
    if args.repetitions != 5:
        raise ValueError("Round 10 requires five latency repetitions")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError(f"Round 10 requires NVIDIA GeForce RTX 4090 D, got {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixels, video_ids = load_pool(dataset, args.pool)
    rows = []
    for repetition in range(args.repetitions):
        for boundary in ("s4", "s3_s4", "s2_s4"):
            for interval in (2, 4, 8):
                pipeline = StructuredBoundaryReusePipeline(model, boundary)
                run_sequence(model, pipeline, pixels, video_ids, boundary, interval, args.warmup, False)
                torch.cuda.reset_peak_memory_stats()
                policy_values, refreshes = run_sequence(model, pipeline, pixels, video_ids, boundary, interval, args.measurements, True)
                full_values, _ = run_sequence(model, None, pixels, video_ids, "full", 1, args.measurements, True)
                policy = summarize(policy_values); full = summarize(full_values)
                row = {"model": args.model_dir.name, "boundary": boundary, "interval": interval,
                       "repetition": repetition + 1, "refreshes": refreshes,
                       "actual_refresh": refreshes / args.measurements, "policy": policy,
                       "paired_full": full, "speedup": full["mean_ms"] / policy["mean_ms"],
                       "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3)}
                rows.append(row); print(json.dumps(row), flush=True)
    summary = {}
    for boundary in ("s4", "s3_s4", "s2_s4"):
        for interval in (2, 4, 8):
            key = f"{boundary}/FI-{interval}"
            values = [row["speedup"] for row in rows if row["boundary"] == boundary and row["interval"] == interval]
            summary[key] = {"repetitions": len(values), "mean_speedup": float(np.mean(values)),
                            "std_speedup": float(np.std(values)), "min_speedup": float(np.min(values)),
                            "max_speedup": float(np.max(values)), "ci95": bootstrap(values)}
    payload = {"model": args.model_dir.name, "warmup": args.warmup, "measurements": args.measurements,
               "repetitions": args.repetitions, "pool": args.pool, "precision": "bf16",
               "timing_boundary": "CUDA-resident pool input to logits, CUDA Events, paired Full immediately after policy",
               "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
