"""Profile B0/B1 stage costs on the second GPU under the frozen protocol."""

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


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.quantile(a, .95)), "p99_ms": float(np.quantile(a, .99)), "std_ms": float(a.std()), "measurements": len(a)}


def event_call(fn) -> float:
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); fn(); end.record(); end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "3080 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Round 11B profile requires the RTX 3080 Ti")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    sample = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]
    pixel = sample["pixel_values"].unsqueeze(0).to("cuda")
    helper = HardwareCalibratedProgressiveReusePipeline(model)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        f1 = helper._run_stage(0, pixel)
        f2 = helper._run_stage(1, f1)
        f3 = helper._run_stage(2, f2)
    functions = {
        "Stage2": lambda: helper._run_stage(1, f1),
        "Stage3": lambda: helper._run_stage(2, f2),
        "Stage4": lambda: helper._run_stage(3, f3),
        "Full": lambda: model(pixel_values=pixel, return_dict=True).logits,
    }
    timings = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for name, fn in functions.items():
            values = []
            for _ in range(args.repetitions):
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                values.extend(event_call(fn) for _ in range(args.measurements))
            timings[name] = summarize(values)
    full_ms = timings["Full"]["mean_ms"]
    payload = {
        "model": args.model_dir.name, "gpu": torch.cuda.get_device_name(0),
        "precision": "bf16", "resolution": [512, 512], "batch_size": 1,
        "warmup": args.warmup, "measurements": args.measurements, "repetitions": args.repetitions,
        "timing_boundary": "CUDA-resident stage input to stage output / logits",
        "timings": timings,
        "c2_ms": timings["Stage2"]["mean_ms"], "c3_ms": timings["Stage3"]["mean_ms"], "c4_ms": timings["Stage4"]["mean_ms"], "full_ms": full_ms,
        "k2": timings["Stage2"]["mean_ms"] / full_ms, "k3": timings["Stage3"]["mean_ms"] / full_ms, "k4": timings["Stage4"]["mean_ms"] / full_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
