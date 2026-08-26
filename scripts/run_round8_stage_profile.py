"""Repeated B1 component profile for the frozen whole-Stage4 pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to("cuda").eval()
    sample = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]
    pixels = sample["pixel_values"].unsqueeze(0).to("cuda")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    enabled = args.precision != "fp32"
    rows = []
    for repetition in range(args.repetitions):
        pipeline = StructuredReusePipeline(model)
        features: dict[str, torch.Tensor] = {}

        def run(fn):
            for _ in range(args.warmup):
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=enabled
                ):
                    fn()
            values = []
            for _ in range(args.measurements):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=dtype, enabled=enabled
                ):
                    fn()
                end.record()
                end.synchronize()
                values.append(float(start.elapsed_time(end)))
            return summarize(values)

        full = run(lambda: model(pixel_values=pixels).logits)
        stage13 = run(
            lambda: features.update(
                zip(("f1", "f2", "f3"), pipeline.fresh_stage_1_to_3(pixels))
            )
        )
        stage13_values = pipeline.fresh_stage_1_to_3(pixels)
        stage4 = run(lambda: pipeline.stage4_cache.run_dense(stage13_values[2]))
        stage4_value = pipeline.stage4_cache.run_dense(stage13_values[2])
        decoder = run(
            lambda: model.decode_head(
                [stage13_values[0], stage13_values[1], stage13_values[2], stage4_value]
            )
        )
        rows.append(
            {
                "repetition": repetition + 1,
                "Full": full,
                "Stage1-3": stage13,
                "Stage4": stage4,
                "Decoder": decoder,
                "stage4_fraction_of_full": stage4["mean_ms"] / full["mean_ms"],
            }
        )
        print(json.dumps(rows[-1], indent=2), flush=True)
    summary = {
        "architecture": "SegFormer-B1",
        "warmup": args.warmup,
        "measurements": args.measurements,
        "repetitions": args.repetitions,
        "precision": args.precision,
        "input_shape": list(pixels.shape),
        "rows": rows,
        "mean_stage4_fraction": float(np.mean([row["stage4_fraction_of_full"] for row in rows])),
        "f4_shape": [1, 512, 16, 16],
        "f4_dtype": "torch.bfloat16" if args.precision == "bf16" else ("torch.float16" if args.precision == "fp16" else "torch.float32"),
        "persistent_cache_mb": (512 * 16 * 16 * 2) / (1024**2),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
