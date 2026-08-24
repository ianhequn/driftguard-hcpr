"""Reprofile dense full/refresh/reuse execution for Round 6."""

from __future__ import annotations

import argparse
import csv
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
from driftguard.structured_reuse.pipeline import (  # noqa: E402
    StructuredReusePipeline,
)
from driftguard.utils.common import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="data/samples/segmentation_input.jpg",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/segformer-b0",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        default="results/R6_E00_full_reprofile",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stats(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(array.mean()),
        f"{prefix}_median_ms": float(np.median(array)),
        f"{prefix}_p90_ms": float(np.quantile(array, 0.90)),
        f"{prefix}_p95_ms": float(np.quantile(array, 0.95)),
        f"{prefix}_p99_ms": float(np.quantile(array, 0.99)),
        f"{prefix}_std_ms": float(array.std()),
    }


def event_and_wall(fn) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return (
        float(start.elapsed_time(end)),
        (time.perf_counter() - wall_start) * 1000.0,
    )


def fresh_stage_1_to_3(model, pixel_values: torch.Tensor) -> list[torch.Tensor]:
    return StructuredReusePipeline(model).fresh_stage_1_to_3(pixel_values)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def benchmark(name: str, fn, warmup: int, measurements: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    event_values = []
    wall_values = []
    for _ in range(measurements):
        event_ms, wall_ms = event_and_wall(fn)
        event_values.append(event_ms)
        wall_values.append(wall_ms)
    return {
        "method": name,
        **stats(event_values, "cuda"),
        **stats(wall_values, "wall"),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.measurements < 1000:
        raise ValueError("Round 6 requires warmup>=200 and measurements>=1000")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda")
    processor, model = load_segformer(
        args.model,
        device=device,
        local_files_only=True,
    )
    image = Image.open(PROJECT_ROOT / args.image).convert("RGB")
    pixel_a = prepare_inputs(processor, image, device)["pixel_values"]
    pixel_b = torch.roll(pixel_a, shifts=1, dims=-1)
    pipeline = StructuredReusePipeline(model)

    with torch.inference_mode():
        full_reference = model(
            pixel_values=pixel_b,
            return_dict=True,
        ).logits
        structured, _, _ = pipeline.forward(
            pixel_b,
            refresh=True,
            state=None,
        )
    difference = structured - full_reference
    equivalence = {
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference.float())
            / torch.linalg.vector_norm(full_reference.float()).clamp_min(1e-12)
        ),
        "allclose": bool(
            torch.allclose(
                structured,
                full_reference,
                rtol=1e-5,
                atol=1e-6,
            )
        ),
    }
    if not equivalence["allclose"]:
        raise RuntimeError(f"100% refresh mismatch: {equivalence}")

    with torch.inference_mode():
        f3_a = fresh_stage_1_to_3(model, pixel_a)[2]
        f3_b = fresh_stage_1_to_3(model, pixel_b)[2]

    rows = []
    with torch.inference_mode():
        rows.append(
            benchmark(
                "direct_full",
                lambda: model(pixel_values=pixel_b, return_dict=True),
                args.warmup,
                args.measurements,
            )
        )
        rows.append(
            benchmark(
                "isolated_stage4",
                lambda: pipeline.stage4_cache.run_dense(f3_b),
                args.warmup,
                args.measurements,
            )
        )

        refresh_state = None
        refresh_index = 0

        def refresh_frame():
            nonlocal refresh_state, refresh_index
            pixel = pixel_a if refresh_index % 2 == 0 else pixel_b
            _, refresh_state, _ = pipeline.forward(
                pixel,
                refresh=True,
                state=refresh_state,
            )
            refresh_index += 1

        rows.append(
            benchmark(
                "structured_refresh_frame",
                refresh_frame,
                args.warmup,
                args.measurements,
            )
        )

        _, reuse_state, _ = pipeline.forward(
            pixel_a,
            refresh=True,
            state=None,
        )
        calls_before_reuse = pipeline.stage4_cache.stage4_forward_calls
        reuse_index = 0

        def reuse_frame():
            nonlocal reuse_state, reuse_index
            pixel = pixel_a if reuse_index % 2 == 0 else pixel_b
            _, reuse_state, frame_stats = pipeline.forward(
                pixel,
                refresh=False,
                state=reuse_state,
            )
            if frame_stats["stage4_calls_delta"] != 0:
                raise RuntimeError("reuse frame executed Stage 4")
            reuse_index += 1

        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = torch.cuda.memory_allocated(device)
        rows.append(
            benchmark(
                "structured_reuse_frame",
                reuse_frame,
                args.warmup,
                args.measurements,
            )
        )
        reuse_peak_mb = (
            torch.cuda.max_memory_allocated(device) - start_allocated
        ) / (1024**2)
        calls_after_reuse = pipeline.stage4_cache.stage4_forward_calls

    full_mean = next(
        row["wall_mean_ms"] for row in rows if row["method"] == "direct_full"
    )
    reuse_mean = next(
        row["wall_mean_ms"]
        for row in rows
        if row["method"] == "structured_reuse_frame"
    )
    stage4_mean = next(
        row["cuda_mean_ms"]
        for row in rows
        if row["method"] == "isolated_stage4"
    )
    summary = {
        "device": torch.cuda.get_device_name(device),
        "model": args.model,
        "input_shape": list(pixel_a.shape),
        "warmup": args.warmup,
        "measurements": args.measurements,
        "equivalence": equivalence,
        "rows": rows,
        "formal_full_wall_mean_ms": full_mean,
        "formal_reuse_wall_mean_ms": reuse_mean,
        "isolated_stage4_cuda_mean_ms": stage4_mean,
        "measured_full_skip_speedup": full_mean / reuse_mean,
        "amdahl_upper_bound": full_mean / (full_mean - stage4_mean),
        "stage4_calls_during_reuse_benchmark": (
            calls_after_reuse - calls_before_reuse
        ),
        "persistent_cache_mb": reuse_state.persistent_bytes / (1024**2),
        "reuse_peak_temporary_mb": float(reuse_peak_mb),
        "boundary": (
            "CUDA-resident pixel_values to native logits; no disk I/O, video "
            "decode, CPU preprocessing, pageable H2D, logging, or plotting."
        ),
    }
    if summary["stage4_calls_during_reuse_benchmark"] != 0:
        raise RuntimeError("Stage 4 was not fully skipped")

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "metrics.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "config.yaml").write_text(
        f"warmup: {args.warmup}\nmeasurements: {args.measurements}\n"
        "input: cuda_resident_fp32_pixel_values\nprecision: fp32\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime.txt").write_text(
        "Completed formal CUDA Event and synchronized wall-clock profile.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
