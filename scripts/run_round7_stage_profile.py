"""Profile Full, Stage 1-3, Stage 4, and decoder on the frozen VSPW model."""

from __future__ import annotations

import argparse
import csv
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

from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to(device)
    model.eval()
    sample = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]
    pixels = sample["pixel_values"].unsqueeze(0).to(device)
    pipeline = StructuredReusePipeline(model)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    def context():
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32")

    def measure(fn):
        for _ in range(args.warmup):
            with torch.inference_mode(), context():
                fn()
        values = []
        for _ in range(args.measurements):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode(), context():
                fn()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "median_ms": float(np.median(array)),
            "p95_ms": float(np.quantile(array, 0.95)),
            "p99_ms": float(np.quantile(array, 0.99)),
            "std_ms": float(array.std()),
        }

    full = measure(lambda: model(pixel_values=pixels).logits)
    features = {}

    def stage13():
        features["f1"], features["f2"], features["f3"] = pipeline.fresh_stage_1_to_3(pixels)

    stage13_stats = measure(stage13)
    stage13()
    stage4_stats = measure(lambda: pipeline.stage4_cache.run_dense(features["f3"]))
    stage4 = pipeline.stage4_cache.run_dense(features["f3"])
    decoder_stats = measure(lambda: model.decode_head([features["f1"], features["f2"], features["f3"], stage4]))
    rows = [
        {"component": "Full", **full},
        {"component": "Stage1-3", **stage13_stats},
        {"component": "Stage4", **stage4_stats},
        {"component": "Decoder", **decoder_stats},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "warmup": args.warmup,
        "measurements": args.measurements,
        "precision": args.precision,
        "input_shape": list(pixels.shape),
        "rows": rows,
        "stage4_fraction_of_full": stage4_stats["mean_ms"] / full["mean_ms"],
        "amdahl_upper_bound_at_zero_refresh": full["mean_ms"] / (full["mean_ms"] - stage4_stats["mean_ms"]),
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
