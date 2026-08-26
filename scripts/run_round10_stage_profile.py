"""Profile individual SegFormer stages for the Round-10 GPU."""

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
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError(f"Round 10 requires NVIDIA GeForce RTX 4090 D, got {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA'}")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    pixels = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]["pixel_values"].unsqueeze(0).to("cuda")
    pipeline = StructuredBoundaryReusePipeline(model, "s4")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = pipeline._run_fresh_prefix(pixels, 4)
    inputs = {"Stage1": pixels, "Stage2": features[0], "Stage3": features[1], "Stage4": features[2]}
    def run_stage(index, value):
        return pipeline._run_stage(index, value)
    def measure(fn):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.measurements):
                fn()
            end.record(); end.synchronize()
            total = start.elapsed_time(end)
        value = total / args.measurements
        return {"mean_ms": float(value), "median_ms": float(value), "p95_ms": float(value), "p99_ms": float(value), "std_ms": 0.0}
    rows = []
    for repetition in range(args.repetitions):
        rows.append({"repetition": repetition + 1, "component": "Full", **measure(lambda: model(pixel_values=pixels).logits)})
        for index, component in enumerate(("Stage1", "Stage2", "Stage3", "Stage4")):
            rows.append({"repetition": repetition + 1, "component": component, **measure(lambda i=index, v=inputs[component]: run_stage(i, v))})
        rows.append({"repetition": repetition + 1, "component": "Decoder", **measure(lambda: model.decode_head(features))})
    summary = {}
    for component in ("Full", "Stage1", "Stage2", "Stage3", "Stage4", "Decoder"):
        values = [row["mean_ms"] for row in rows if row["component"] == component]
        summary[component] = {"mean_ms": float(np.mean(values)), "std_across_repetitions_ms": float(np.std(values)), "repetitions": len(values)}
    full = summary["Full"]["mean_ms"]
    s4 = summary["Stage4"]["mean_ms"]
    s3s4 = s4 + summary["Stage3"]["mean_ms"]
    s2s4 = s3s4 + summary["Stage2"]["mean_ms"]
    payload = {"model": args.model_dir.name, "warmup": args.warmup, "measurements": args.measurements,
               "repetitions": args.repetitions, "precision": "bf16", "input_shape": list(pixels.shape),
               "rows": rows, "summary": summary, "shares": {"S4": s4 / full, "S3-S4": s3s4 / full, "S2-S4": s2s4 / full},
               "cache_shapes": {"F2": list(features[1].shape), "F3": list(features[2].shape), "F4": list(features[3].shape)},
               "cache_bytes": {"S4": features[3].numel() * features[3].element_size(),
                               "S3-S4": sum(features[index].numel() * features[index].element_size() for index in (2, 3)),
                               "S2-S4": sum(features[index].numel() * features[index].element_size() for index in (1, 2, 3))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": payload["model"], "summary": summary, "shares": payload["shares"]}, indent=2))


if __name__ == "__main__":
    main()
