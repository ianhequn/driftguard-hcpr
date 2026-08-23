"""Profile full SegFormer inference with CUDA Events or CPU timing."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer  # noqa: E402
from driftguard.profiling.benchmark import benchmark_model  # noqa: E402
from driftguard.utils.common import choose_device, set_seed, start_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", default="results/baseline_profile.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, logger = start_run("profile_baseline", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    processor, model = load_segformer(args.model, device=device)
    del processor
    pixel_values = torch.randn(
        1,
        3,
        args.input_size,
        args.input_size,
        device=device,
    )
    metrics = benchmark_model(
        model=model,
        pixel_values=pixel_values,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": args.model,
        "device": str(device),
        "input_size": args.input_size,
        **metrics,
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    for label, key in (
        ("Mean Latency", "mean_latency_ms"),
        ("Median Latency", "median_latency_ms"),
        ("P95", "p95_latency_ms"),
        ("P99", "p99_latency_ms"),
        ("FPS", "fps"),
        ("Peak VRAM", "peak_vram_mb"),
    ):
        print(f"{label}: {metrics[key]:.6f}")
    print(f"Timing backend: {metrics['timing_backend']}")
    print(f"Output: {output_path.resolve()}")
    logger.info("Baseline profile complete: %s", metrics)


if __name__ == "__main__":
    main()
