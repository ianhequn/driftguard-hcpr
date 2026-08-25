"""Run five independent paired CUDA latency repetitions for Round 7.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import Stage4CacheState, StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


EXPECTED_SHA256 = "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b"
POLICIES = ("FI-2", "FI-3", "FI-4", "FI-6", "FI-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_telemetry() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,memory.used,clocks.sm,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        values = subprocess.check_output(command, text=True).strip().split(",")
        return {
            "gpu_temperature": values[0].strip(),
            "gpu_memory_used": values[1].strip(),
            "gpu_clock_sm": values[2].strip(),
            "gpu_utilization": values[3].strip(),
        }
    except (OSError, subprocess.CalledProcessError, IndexError):
        return {"gpu_temperature": "unavailable", "gpu_memory_used": "unavailable", "gpu_clock_sm": "unavailable", "gpu_utilization": "unavailable"}


def autocast_context(precision: str):
    if precision == "fp32":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16, enabled=True)


def load_pool(dataset: VSPWFrameDataset, device: torch.device, count: int):
    count = min(count, len(dataset))
    samples = [dataset[index] for index in range(count)]
    pixels = torch.stack([sample["pixel_values"] for sample in samples]).to(device)
    video_ids = [sample["video_id"] for sample in samples]
    frame_ids = [sample["frame_id"] for sample in samples]
    positions: list[int] = []
    previous = None
    position = 0
    for video_id in video_ids:
        if video_id != previous:
            position = 0
            previous = video_id
        positions.append(position)
        position += 1
    return pixels, video_ids, frame_ids, positions


def sequence_indices(pool_size: int, count: int):
    for emitted in range(count):
        yield emitted % pool_size, emitted // pool_size


def timed_model(fn) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std()),
        "measurements": int(array.size),
        "fps": float(1000.0 / array.mean()),
    }


def run_full(model, pixels, warmup: int, measurements: int, precision: str) -> tuple[list[float], dict]:
    def invoke(index: int):
        pixel = pixels[index : index + 1]
        with torch.inference_mode(), autocast_context(precision):
            model(pixel_values=pixel).logits

    for index, _ in sequence_indices(pixels.shape[0], warmup):
        invoke(index)
    torch.cuda.synchronize()
    values = [timed_model(lambda index=index: invoke(index)) for index, _ in sequence_indices(pixels.shape[0], measurements)]
    return values, gpu_telemetry()


def warmup_full(model, pixels, warmup: int, precision: str) -> None:
    for index, _ in sequence_indices(pixels.shape[0], warmup):
        pixel = pixels[index : index + 1]
        with torch.inference_mode(), autocast_context(precision):
            model(pixel_values=pixel).logits
    torch.cuda.synchronize()


def measure_full_only(model, pixels, measurements: int, precision: str) -> tuple[list[float], dict]:
    def invoke(index: int):
        pixel = pixels[index : index + 1]
        with torch.inference_mode(), autocast_context(precision):
            model(pixel_values=pixel).logits

    values = [timed_model(lambda index=index: invoke(index)) for index, _ in sequence_indices(pixels.shape[0], measurements)]
    return values, gpu_telemetry()


def run_fixed(model, pixels, video_ids, positions, interval: int, warmup: int, measurements: int, precision: str):
    pipeline = StructuredReusePipeline(model)

    def invoke(index: int, cycle: int):
        if cycle > 0 and index == 0:
            state_holder[0] = None
        if state_holder[1] != video_ids[index]:
            state_holder[0] = None
            state_holder[1] = video_ids[index]
        state = state_holder[0]
        refresh = state is None or positions[index] % interval == 0
        pixel = pixels[index : index + 1]
        with torch.inference_mode(), autocast_context(precision):
            _, next_state, stats = pipeline.forward(pixel, refresh=refresh, state=state)
        state_holder[0] = next_state
        state_holder[2] += int(stats["actual_refresh"])
        if stats["stage4_calls_delta"] != int(stats["actual_refresh"]):
            raise RuntimeError("Stage4 counter mismatch")

    state_holder: list[Stage4CacheState | str | int | None] = [None, None, 0]
    for index, cycle in sequence_indices(pixels.shape[0], warmup):
        invoke(index, cycle)
    torch.cuda.synchronize()
    state_holder[:] = [None, None, 0]
    values: list[float] = []
    for index, cycle in sequence_indices(pixels.shape[0], measurements):
        values.append(timed_model(lambda index=index, cycle=cycle: invoke(index, cycle)))
    return values, int(state_holder[2]), gpu_telemetry()


def bootstrap_ci(values: list[float], seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = rng.choice(array, size=(10000, array.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    actual_sha256 = sha256_file(args.model_dir / "model.safetensors")
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"MODEL ARTIFACT MISMATCH: {actual_sha256}")

    device = torch.device("cuda")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pixels, video_ids, frame_ids, positions = load_pool(dataset, device, args.pool)
    rng = np.random.default_rng(args.seed)
    orders = [list(rng.permutation(POLICIES)) for _ in range(args.repetitions)]
    rows: list[dict] = []

    for repetition in range(args.repetitions):
        full_values, telemetry = run_full(model, pixels, args.warmup, args.measurements, args.precision)
        full_summary = summarize(full_values)
        rows.append({
            "repetition": repetition + 1,
            "policy": "Full",
            "order_position": 0,
            **full_summary,
            "paired_full_mean_ms": full_summary["mean_ms"],
            "speedup": 1.0,
            **telemetry,
        })
        for order_position, policy in enumerate(orders[repetition], start=1):
            interval = int(policy.split("-")[1])
            # Warm both paths first. The paired Full measurement follows the FI measurement with no intervening warmup.
            warmup_full(model, pixels, args.warmup, args.precision)
            fixed_values, refresh_count, telemetry = run_fixed(model, pixels, video_ids, positions, interval, args.warmup, args.measurements, args.precision)
            full_values, _ = measure_full_only(model, pixels, args.measurements, args.precision)
            policy_summary = summarize(fixed_values)
            paired_summary = summarize(full_values)
            rows.append({
                "repetition": repetition + 1,
                "policy": policy,
                "order_position": order_position,
                **policy_summary,
                "refresh_frames": refresh_count,
                "refresh_ratio": refresh_count / args.measurements,
                "paired_full_mean_ms": paired_summary["mean_ms"],
                "paired_full_median_ms": paired_summary["median_ms"],
                "paired_full_p95_ms": paired_summary["p95_ms"],
                "paired_full_p99_ms": paired_summary["p99_ms"],
                "speedup": paired_summary["mean_ms"] / policy_summary["mean_ms"],
                **telemetry,
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    summary_by_policy = {}
    for policy in ("Full", *POLICIES):
        policy_rows = [row for row in rows if row["policy"] == policy]
        speedups = [float(row["speedup"]) for row in policy_rows]
        array = np.asarray(speedups, dtype=np.float64)
        low, high = bootstrap_ci(speedups, args.seed)
        summary_by_policy[policy] = {
            "repetitions": len(policy_rows),
            "mean_speedup": float(array.mean()),
            "std_speedup": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "cv_speedup": float(array.std(ddof=1) / array.mean()) if len(array) > 1 else 0.0,
            "min_speedup": float(array.min()),
            "max_speedup": float(array.max()),
            "ci95_low": low,
            "ci95_high": high,
            "mean_latency_ms": float(np.mean([row["mean_ms"] for row in policy_rows])),
        }

    payload = {
        "model_sha256": actual_sha256,
        "dataset_split": "VSPW official validation",
        "pool": args.pool,
        "precision": args.precision,
        "warmup": args.warmup,
        "measurements": args.measurements,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "policy_orders": [[str(policy) for policy in order] for order in orders],
        "rows": rows,
        "summary_by_policy": summary_by_policy,
        "timing_boundary": "CUDA-resident 1x3x512x512 input, CUDA Event, paired Full immediately after each FI measurement",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_by_policy, indent=2))


if __name__ == "__main__":
    main()
