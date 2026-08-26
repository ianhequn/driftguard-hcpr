"""Formal Round 11A paired latency for HCPR and the Round-10 frontier."""

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
from driftguard.structured_reuse.boundaries import StructuredBoundaryReusePipeline  # noqa: E402
from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "p99_ms": float(np.quantile(values, .99)),
            "std_ms": float(values.std()), "fps": float(1000.0 / values.mean()), "measurements": len(values)}


def bootstrap(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def event_call(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); fn(); end.record(); end.synchronize()
    return float(start.elapsed_time(end))


def run_stream(model, kind, config, pixels, video_ids, warmup, measurements, record):
    pipeline = None
    if kind == "hcpr":
        pipeline = HardwareCalibratedProgressiveReusePipeline(
            model, stage_costs=config["hardware"], drift_scales=config["scales"],
            lambda_value=config["lambda"], max_stale_run=config["max_stale_run"],
        )
    elif kind == "fixed":
        pipeline = StructuredBoundaryReusePipeline(model, config["boundary"])
    state = None
    previous_video = None
    positions = {}
    calls = {f"stage{i}": 0 for i in range(1, 5)} | {"decoder": 0}
    routes = {}
    values = []

    def one(emitted, collect):
        nonlocal state, previous_video
        index = emitted % pixels.shape[0]
        video_id = video_ids[index]
        if video_id != previous_video:
            state = None
            positions[video_id] = 0
            previous_video = video_id
        position = positions[video_id]
        positions[video_id] += 1
        pixel = pixels[index:index + 1]
        result = {}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if kind == "hcpr":
                elapsed = event_call(lambda: result.setdefault("value", pipeline.forward(pixel, state=state)))
                _, state, stats = result["value"]
                route = stats["route"]
                for key, value in stats["stage_calls"].items():
                    calls[key] += int(value)
            else:
                refresh = position % config["interval"] == 0 or state is None
                elapsed = event_call(lambda: result.setdefault("value", pipeline.forward(pixel, refresh=refresh, state=state)))
                _, state, stats = result["value"]
                route = config["boundary"].upper()
                for key, value in stats["stage_calls"].items():
                    calls[key] += int(value)
        routes[route] = routes.get(route, 0) + 1
        if collect:
            values.append(elapsed)

    for emitted in range(warmup):
        one(emitted, False)
    if kind == "hcpr":
        pipeline = HardwareCalibratedProgressiveReusePipeline(
            model, stage_costs=config["hardware"], drift_scales=config["scales"],
            lambda_value=config["lambda"], max_stale_run=config["max_stale_run"],
        )
        state = None; previous_video = None; positions = {}; calls = {f"stage{i}": 0 for i in range(1, 5)} | {"decoder": 0}; routes = {}
    elif kind == "fixed":
        pipeline = StructuredBoundaryReusePipeline(model, config["boundary"])
        state = None; previous_video = None; positions = {}; calls = {f"stage{i}": 0 for i in range(1, 5)} | {"decoder": 0}; routes = {}
    for emitted in range(measurements):
        one(emitted, record)
    return values, {"stage_calls": calls, "route_counts": routes}


def paired_full(model, pixels, video_ids, warmup, measurements):
    state = None
    values = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for emitted in range(warmup):
            pixel = pixels[emitted % pixels.shape[0]:emitted % pixels.shape[0] + 1]
            event_call(lambda: model(pixel_values=pixel).logits)
        for emitted in range(measurements):
            pixel = pixels[emitted % pixels.shape[0]:emitted % pixels.shape[0] + 1]
            values.append(event_call(lambda: model(pixel_values=pixel).logits))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hardware-costs", type=Path, required=True)
    parser.add_argument("--drift-statistics", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--no-age-guard", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError("Round 11A formal latency requires RTX 4090 D")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    samples = [dataset[index] for index in range(min(args.pool, len(dataset)))]
    pixels = torch.stack([sample["pixel_values"] for sample in samples]).to("cuda")
    video_ids = [sample["video_id"] for sample in samples]
    hardware = json.loads(args.hardware_costs.read_text(encoding="utf-8"))
    scales_raw = json.loads(args.drift_statistics.read_text(encoding="utf-8"))["scales"]
    scales = {int(key[-1]) + 1: value for key, value in scales_raw.items()}
    grid = json.loads(args.lambda_grid.read_text(encoding="utf-8"))["values"]
    configs = [("fixed", {"name": "S4/FI2", "boundary": "s4", "interval": 2}),
               ("fixed", {"name": "S3-S4/FI2", "boundary": "s3_s4", "interval": 2}),
               ("fixed", {"name": "S2-S4/FI2", "boundary": "s2_s4", "interval": 2}),
               ("fixed", {"name": "S2-S4/FI4", "boundary": "s2_s4", "interval": 4}),
               ("fixed", {"name": "S2-S4/FI8", "boundary": "s2_s4", "interval": 8})]
    max_stale_run = None if args.no_age_guard else 7
    for entry in grid:
        configs.append(("hcpr", {"name": f"HCPR/lambda-{float(entry['lambda']):.8g}", "lambda": float(entry["lambda"]), "quantile": entry.get("quantile"), "hardware": hardware, "scales": scales, "max_stale_run": max_stale_run}))
    rows = []
    for repetition in range(args.repetitions):
        for kind, config in configs:
            policy_values, diagnostics = run_stream(model, kind, config, pixels, video_ids, args.warmup, args.measurements, True)
            torch.cuda.synchronize()
            full_values = paired_full(model, pixels, video_ids, args.warmup, args.measurements)
            policy = summarize(policy_values); full = summarize(full_values)
            row = {"method": config["name"], "kind": kind, "repetition": repetition + 1,
                   "lambda": config.get("lambda"), "quantile": config.get("quantile"),
                   "paired_full": full, "policy": policy,
                   "paired_speedup": full["mean_ms"] / policy["mean_ms"], **diagnostics}
            rows.append(row)
            print(json.dumps({"method": row["method"], "repetition": row["repetition"], "policy_mean_ms": policy["mean_ms"], "full_mean_ms": full["mean_ms"], "paired_speedup": row["paired_speedup"]}), flush=True)
    summary = {}
    for kind, config in configs:
        values = [row["paired_speedup"] for row in rows if row["method"] == config["name"]]
        summary[config["name"]] = {"kind": kind, "repetitions": len(values), "mean_speedup": float(np.mean(values)), "std_speedup": float(np.std(values)), "min_speedup": float(np.min(values)), "max_speedup": float(np.max(values)), "ci95": bootstrap(values)}
    payload = {"model": args.model_dir.name, "warmup": args.warmup, "measurements": args.measurements, "repetitions": args.repetitions, "pool": args.pool, "precision": "bf16", "timing_boundary": "CUDA-resident input to logits, controller included, paired Full immediately after each policy", "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
