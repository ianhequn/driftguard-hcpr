"""Measure HCPR controller overhead on the RTX 3080 Ti."""

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


def measure(pipeline, stage: int, current: torch.Tensor, reference: torch.Tensor, warmup: int, measurements: int) -> dict:
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            value = pipeline._score(pipeline._drift(current, reference), stage)
            bool((value <= 1.0).item())
        torch.cuda.synchronize()
        values = []
        for _ in range(measurements):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            value = pipeline._score(pipeline._drift(current, reference), stage)
            bool((value <= 1.0).item())
            end.record(); end.synchronize()
            values.append(float(start.elapsed_time(end)))
    arr = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(arr.mean()), "median_ms": float(np.median(arr)), "p95_ms": float(np.quantile(arr, .95)), "p99_ms": float(np.quantile(arr, .99)), "std_ms": float(arr.std()), "measurements": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-key", choices=("B0", "B1"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--round11a-state", type=Path, required=True)
    parser.add_argument("--new-cost", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "3080 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Round 11B controller profiling requires RTX 3080 Ti")
    state = json.loads(args.round11a_state.read_text(encoding="utf-8"))
    costs = json.loads(args.new_cost.read_text(encoding="utf-8"))
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    sample = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]["pixel_values"].unsqueeze(0).to("cuda")
    pipeline = HardwareCalibratedProgressiveReusePipeline(model, {2: costs["k2"], 3: costs["k3"], 4: costs["k4"]}, {2: state["drift_scales"][args.model_key]["d1"], 3: state["drift_scales"][args.model_key]["d2"], 4: state["drift_scales"][args.model_key]["d3"]})
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = pipeline._full_frame(sample, pipeline.new_state())
    references = {2: pipeline._pool(features[0]), 3: pipeline._pool(features[1]), 4: pipeline._pool(features[2])}
    result = {f"boundary_{stage}": measure(pipeline, stage, features[stage - 2], references[stage], args.warmup, args.measurements) for stage in (2, 3, 4)}
    result["sum_mean_ms"] = sum(result[f"boundary_{stage}"]["mean_ms"] for stage in (2, 3, 4))
    result["gpu"] = torch.cuda.get_device_name(0)
    result["definition"] = "GPU-resident GAP + cosine drift + score + scalar branch synchronization"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
