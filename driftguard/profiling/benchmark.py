"""CUDA-event and CPU fallback benchmarking."""

from __future__ import annotations

import time
from typing import Any

import torch


def _forward(model, pixel_values):
    with torch.inference_mode():
        return model(pixel_values=pixel_values, return_dict=True)


def benchmark_model(
    model,
    pixel_values: torch.Tensor,
    device: torch.device,
    warmup: int = 100,
    iterations: int = 500,
) -> dict[str, Any]:
    """Benchmark a model with synchronized CUDA timing when available."""

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")

    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            _forward(model, pixel_values)
        torch.cuda.synchronize(device)
        timings_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _forward(model, pixel_values)
            end.record()
            end.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        for _ in range(warmup):
            _forward(model, pixel_values)
        timings_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            _forward(model, pixel_values)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
        peak_vram_mb = 0.0

    values = torch.tensor(timings_ms, dtype=torch.float64)
    mean_latency = float(values.mean().item())
    median_latency = float(values.median().item())
    p95 = float(torch.quantile(values, 0.95).item())
    p99 = float(torch.quantile(values, 0.99).item())
    return {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "fps": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        "peak_vram_mb": float(peak_vram_mb),
        "timing_backend": "cuda_event" if use_cuda_events else "perf_counter_cpu",
    }
