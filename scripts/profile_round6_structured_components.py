"""Profile structured-pipeline full, Stage-4, and reuse components."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer, prepare_inputs  # noqa: E402
from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std()),
    }


def timed(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> None:
    device = torch.device("cuda")
    processor, model = load_segformer(
        "/root/autodl-tmp/models/segformer-b0",
        device=device,
        local_files_only=True,
    )
    image = Image.open(
        PROJECT_ROOT / "data/samples/segmentation_input.jpg"
    ).convert("RGB")
    pixel = prepare_inputs(processor, image, device)["pixel_values"]
    pipeline = StructuredReusePipeline(model)
    with torch.inference_mode():
        f3 = pipeline.fresh_stage_1_to_3(pixel)[2]
    for _ in range(200):
        with torch.inference_mode():
            pipeline.forward(pixel, True, None)
    torch.cuda.synchronize()
    values = {"structured_full": [], "stage4": [], "reuse": []}
    with torch.inference_mode():
        _, state, _ = pipeline.forward(pixel, True, None)
        for _ in range(1000):
            values["structured_full"].append(
                timed(lambda: pipeline.forward(pixel, True, state))
            )
        for _ in range(1000):
            values["stage4"].append(
                timed(lambda: pipeline.stage4_cache.run_dense(f3))
            )
        for _ in range(1000):
            values["reuse"].append(
                timed(lambda: pipeline.forward(pixel, False, state))
            )
    summary = {
        "structured_full": distribution(values["structured_full"]),
        "stage4": distribution(values["stage4"]),
        "reuse": distribution(values["reuse"]),
        "amdahl_upper_bound": (
            distribution(values["structured_full"])["mean_ms"]
            / (
                distribution(values["structured_full"])["mean_ms"]
                - distribution(values["stage4"])["mean_ms"]
            )
        ),
        "warmup": 200,
        "measurements": 1000,
        "boundary": "CUDA-resident pixel_values, structured pipeline",
    }
    root = PROJECT_ROOT / "results/R6_E00_full_reprofile"
    root.mkdir(parents=True, exist_ok=True)
    (root / "structured_components.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
