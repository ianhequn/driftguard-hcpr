"""Measure the GPU-resident HCPR drift/score/branch overhead separately."""

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
from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hardware-costs", type=Path, required=True)
    parser.add_argument("--drift-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError("controller profiling requires RTX 4090 D")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixels = dataset[0]["pixel_values"].unsqueeze(0).to("cuda")
    hardware = json.loads(args.hardware_costs.read_text(encoding="utf-8"))
    raw_scales = json.loads(args.drift_statistics.read_text(encoding="utf-8"))["scales"]
    scales = {int(key[-1]) + 1: value for key, value in raw_scales.items()}
    pipeline = HardwareCalibratedProgressiveReusePipeline(model, hardware, scales)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = pipeline._full_frame(pixels, pipeline.new_state())
    references = {2: pipeline._pool(features[0]), 3: pipeline._pool(features[1]), 4: pipeline._pool(features[2])}
    signals = {2: (features[0], references[2]), 3: (features[1], references[3]), 4: (features[2], references[4])}
    def measure(stage):
        current, reference = signals[stage]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(args.warmup):
                value = pipeline._score(pipeline._drift(current, reference), stage)
                bool((value <= 1.0).item())
            torch.cuda.synchronize()
            values = []
            for _ in range(args.measurements):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                value = pipeline._score(pipeline._drift(current, reference), stage)
                bool((value <= 1.0).item())
                end.record(); end.synchronize()
                values.append(float(start.elapsed_time(end)))
        array = np.asarray(values)
        return {"mean_ms": float(array.mean()), "median_ms": float(np.median(array)), "p95_ms": float(np.quantile(array, .95)), "p99_ms": float(np.quantile(array, .99)), "std_ms": float(array.std()), "measurements": len(values)}
    result = {f"boundary_{stage}": measure(stage) for stage in (2, 3, 4)}
    result["sum_mean_ms"] = sum(result[f"boundary_{stage}"]["mean_ms"] for stage in (2, 3, 4))
    result["definition"] = "GAP + cosine drift + normalization + scalar synchronization + Python branch; feature maps remain on GPU"
    result["model"] = args.model_dir.name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
