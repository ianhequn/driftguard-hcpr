"""Measure full and cache-cut continuation latency with real model execution."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.sequential_stale import (  # noqa: E402
    decode_features,
    sequential_stale_features,
)
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/samples/segmentation_input.jpg")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="results/E03_sequential_stale/cache_cut_latency.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _event_time(fn, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    start_time = time.perf_counter()
    fn()
    return (time.perf_counter() - start_time) * 1000.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    processor, model = load_segformer(args.model, device=device)
    image = Image.open(PROJECT_ROOT / args.image).convert("RGB")
    inputs = prepare_inputs(processor, image, device)
    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True,
        )
    current_features = get_stage_features(outputs, model=model)
    cached_features = [feature.detach().clone() for feature in current_features]

    operations = {"full": lambda: model(pixel_values=inputs["pixel_values"], return_dict=True)}
    for cut in range(1, 5):
        operations[f"cut_{cut}"] = lambda cut=cut: decode_features(
            model,
            sequential_stale_features(
                model=model,
                current_features=current_features,
                cached_features=cached_features,
                cut=cut,
            ),
        )

    rows = []
    for name, operation in operations.items():
        for _ in range(args.warmup):
            with torch.no_grad():
                operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(args.iterations):
            with torch.no_grad():
                timings.append(_event_time(operation, device))
        values = torch.tensor(timings, dtype=torch.float64)
        rows.append(
            {
                "cut": name,
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iterations,
                "mean_latency_ms": float(values.mean().item()),
                "median_latency_ms": float(values.median().item()),
                "p95_latency_ms": float(torch.quantile(values, 0.95).item()),
                "p99_latency_ms": float(torch.quantile(values, 0.99).item()),
                "timing_backend": "cuda_event" if device.type == "cuda" else "perf_counter_cpu",
            }
        )
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"device={device}")
    for row in rows:
        print(f"{row['cut']}: {row['mean_latency_ms']:.6f} ms")
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
