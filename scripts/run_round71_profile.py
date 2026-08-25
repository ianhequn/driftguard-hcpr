"""Run three independent Round 7.1 structured-stage profiles."""

from __future__ import annotations

import argparse
import hashlib
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


EXPECTED_SHA256 = "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def summarize(values: list[float]) -> dict[str, float]:
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    args = parser.parse_args()
    actual_sha256 = sha256_file(args.model_dir / "model.safetensors")
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"MODEL ARTIFACT MISMATCH: {actual_sha256}")

    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixel = dataset[0]["pixel_values"].unsqueeze(0).to(device)
    pipeline = StructuredReusePipeline(model)
    rows = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for repetition in range(1, args.repetitions + 1):
            # Materialize component inputs once per repetition, then profile only the requested component.
            f1, f2, f3 = pipeline.fresh_stage_1_to_3(pixel)
            f4 = pipeline.stage4_cache.run_dense(f3)
            warmup_fns = {
                "Full": lambda: pipeline.forward(pixel, True, None),
                "Stage1-3": lambda: pipeline.fresh_stage_1_to_3(pixel),
                "Stage4": lambda: pipeline.stage4_cache.run_dense(f3),
                "Decoder": lambda: model.decode_head([f1, f2, f3, f4]),
            }
            measure_fns = warmup_fns
            for component in ("Full", "Stage1-3", "Stage4", "Decoder"):
                for _ in range(args.warmup):
                    warmup_fns[component]()
                torch.cuda.synchronize()
                values = [timed(measure_fns[component]) for _ in range(args.measurements)]
                summary = summarize(values)
                rows.append({"repetition": repetition, "component": component, **summary})
                print(json.dumps(rows[-1]), flush=True)

    summary_by_component = {}
    for component in ("Full", "Stage1-3", "Stage4", "Decoder"):
        values = [row["mean_ms"] for row in rows if row["component"] == component]
        summary_by_component[component] = {
            "mean_ms": float(np.mean(values)),
            "std_across_repetitions_ms": float(np.std(values, ddof=1)),
            "repetitions": len(values),
        }
    fraction = summary_by_component["Stage4"]["mean_ms"] / summary_by_component["Full"]["mean_ms"]
    payload = {
        "model_sha256": actual_sha256,
        "warmup": args.warmup,
        "measurements": args.measurements,
        "repetitions": args.repetitions,
        "precision": "bf16",
        "input_shape": [1, 3, 512, 512],
        "rows": rows,
        "summary_by_component": summary_by_component,
        "stage4_fraction_of_full": fraction,
        "profile_boundary": "CUDA-resident structured reuse pipeline, CUDA Event",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
