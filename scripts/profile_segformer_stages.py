"""Profile SegFormer encoder stages and decoder with CUDA-aware timing."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import (  # noqa: E402
    DEFAULT_IMAGE_URL,
    get_stage_features,
    load_image,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402


def _percentile(values: list[float], q: float) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q).item())


def _summarize(measurements: list[float], device: torch.device) -> dict:
    return {
        "mean_latency_ms": sum(measurements) / len(measurements),
        "median_latency_ms": float(torch.tensor(measurements).median().item()),
        "p95_latency_ms": _percentile(measurements, 0.95),
        "p99_latency_ms": _percentile(measurements, 0.99),
        "timing_backend": "cuda_event" if device.type == "cuda" else "perf_counter_cpu",
    }


def _run_encoder_with_stage_timing(
    encoder,
    pixel_values: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[torch.Tensor]]:
    """Run the Hugging Face encoder once and time each real stage."""

    batch_size = pixel_values.shape[0]
    hidden_states = pixel_values
    stage_times: list[float] = []
    all_hidden_states: list[torch.Tensor] = []

    for index, (embedding_layer, block_layer, norm_layer) in enumerate(
        zip(encoder.patch_embeddings, encoder.block, encoder.layer_norm)
    ):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            started = time.perf_counter()

        hidden_states, height, width = embedding_layer(hidden_states)
        for block in block_layer:
            hidden_states = block(hidden_states, height, width, False)[0]
        hidden_states = norm_layer(hidden_states)
        if index != len(encoder.patch_embeddings) - 1 or encoder.config.reshape_last_stage:
            hidden_states = hidden_states.reshape(
                batch_size,
                height,
                width,
                -1,
            ).permute(0, 3, 1, 2).contiguous()
        all_hidden_states.append(hidden_states)

        if device.type == "cuda":
            end_event.record()
            end_event.synchronize()
            stage_times.append(float(start_event.elapsed_time(end_event)))
        else:
            stage_times.append((time.perf_counter() - started) * 1000.0)

    return stage_times, all_hidden_states


def _profile_encoder_stages(
    encoder,
    pixel_values: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[dict]:
    for _ in range(warmup):
        _run_encoder_with_stage_timing(encoder, pixel_values, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measurements = [[] for _ in range(4)]
    last_features: list[torch.Tensor] = []
    for _ in range(iterations):
        stage_times, last_features = _run_encoder_with_stage_timing(
            encoder,
            pixel_values,
            device,
        )
        for index, value in enumerate(stage_times):
            measurements[index].append(value)

    rows = []
    for index, values in enumerate(measurements):
        output = last_features[index]
        rows.append(
            {
                "component": f"stage_{index + 1}",
                "output_shape": "x".join(str(item) for item in output.shape),
                **_summarize(values, device),
            }
        )
    return rows


def _profile_decoder(
    model,
    pixel_values: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict:
    def run_decoder():
        with torch.inference_mode():
            encoder_outputs = model.segformer.encoder(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            features = get_stage_features(encoder_outputs, model=model)
            return model.decode_head(features)

    for _ in range(warmup):
        output = run_decoder()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measurements = []
    for _ in range(iterations):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = run_decoder()
            end_event.record()
            end_event.synchronize()
            measurements.append(float(start_event.elapsed_time(end_event)))
        else:
            started = time.perf_counter()
            output = run_decoder()
            measurements.append((time.perf_counter() - started) * 1000.0)
    return {
        "component": "decoder",
        "output_shape": "x".join(str(item) for item in output.shape),
        **_summarize(measurements, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", default="results/E02_stage_profiling/stage_latency.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    image = load_image(args.image, PROJECT_ROOT / "data" / "samples")
    processor, model = load_segformer(args.model, device=device)
    inputs = prepare_inputs(processor, image, device)
    pixel_values = inputs["pixel_values"]

    encoder = model.segformer.encoder
    stage_timings = _profile_encoder_stages(
        encoder,
        pixel_values,
        device,
        args.warmup,
        args.iterations,
    )
    stage_timings.append(
        _profile_decoder(
            model,
            pixel_values,
            device,
            args.warmup,
            args.iterations,
        )
    )

    total_latency = sum(row["mean_latency_ms"] for row in stage_timings)
    for row in stage_timings:
        row["runtime_percent_of_reported_components"] = (
            row["mean_latency_ms"] / total_latency * 100.0 if total_latency else 0.0
        )
        row["model"] = args.model
        row["device"] = str(device)
        row["warmup_iterations"] = args.warmup
        row["measured_iterations"] = args.iterations

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stage_timings[0]))
        writer.writeheader()
        writer.writerows(stage_timings)
    print(f"device={device}")
    print(f"timing_backend={stage_timings[0]['timing_backend']}")
    for row in stage_timings:
        print(
            f"{row['component']}: mean={row['mean_latency_ms']:.6f} ms "
            f"p95={row['p95_latency_ms']:.6f} ms "
            f"runtime_percent={row['runtime_percent_of_reported_components']:.2f}% "
            f"shape={row['output_shape']}"
        )
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
