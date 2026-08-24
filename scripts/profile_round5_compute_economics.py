"""Profile Round 5 SegFormer compute economics with CUDA events."""

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

from driftguard.models.segformer import (  # noqa: E402
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import set_seed  # noqa: E402


COMPONENTS = (
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "decoder",
)


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
        default="results/R5_E00_compute_economics",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "p99_ms": float(np.quantile(array, 0.99)),
        "std_ms": float(array.std()),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_stages(model, pixel_values: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    encoder = model.segformer.encoder
    batch_size = pixel_values.shape[0]
    hidden_states = pixel_values
    features = []
    for index, (embedding, blocks, norm) in enumerate(
        zip(encoder.patch_embeddings, encoder.block, encoder.layer_norm)
    ):
        hidden_states, height, width = embedding(hidden_states)
        for block in blocks:
            hidden_states = block(
                hidden_states,
                height,
                width,
                output_attentions=False,
            )[0]
        hidden_states = norm(hidden_states)
        if index != len(encoder.patch_embeddings) - 1 or encoder.config.reshape_last_stage:
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        features.append(hidden_states)
    return features, model.decode_head(features)


def _timed_manual_forward(
    model,
    pixel_values: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    encoder = model.segformer.encoder
    batch_size = pixel_values.shape[0]
    hidden_states = pixel_values
    features = []
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in (*COMPONENTS, "manual_total")
    }
    events["manual_total"][0].record()
    for index, (embedding, blocks, norm) in enumerate(
        zip(encoder.patch_embeddings, encoder.block, encoder.layer_norm)
    ):
        name = f"stage_{index + 1}"
        events[name][0].record()
        hidden_states, height, width = embedding(hidden_states)
        for block in blocks:
            hidden_states = block(
                hidden_states,
                height,
                width,
                output_attentions=False,
            )[0]
        hidden_states = norm(hidden_states)
        if index != len(encoder.patch_embeddings) - 1 or encoder.config.reshape_last_stage:
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        features.append(hidden_states)
        events[name][1].record()
    events["decoder"][0].record()
    logits = model.decode_head(features)
    events["decoder"][1].record()
    events["manual_total"][1].record()
    events["manual_total"][1].synchronize()
    timings = {
        name: float(start.elapsed_time(end))
        for name, (start, end) in events.items()
    }
    return timings, logits


def _time_full_forward(model, pixel_values: torch.Tensor) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model(pixel_values=pixel_values, return_dict=True)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _profile_cpu_preprocessing(
    processor,
    image: Image.Image,
    warmup: int,
    measurements: int,
) -> list[float]:
    for _ in range(warmup):
        processor(images=image, return_tensors="pt")
    values = []
    for _ in range(measurements):
        started = time.perf_counter()
        processor(images=image, return_tensors="pt")
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.measurements < 1000:
        raise ValueError("Round 5 requires warmup>=200 and measurements>=1000")
    if not torch.cuda.is_available():
        raise RuntimeError("Round 5 compute economics requires CUDA")
    set_seed(args.seed)
    device = torch.device("cuda")
    processor, model = load_segformer(
        args.model,
        device=device,
        local_files_only=True,
    )
    image_path = PROJECT_ROOT / args.image
    image = Image.open(image_path).convert("RGB")
    pixel_values = prepare_inputs(processor, image, device)["pixel_values"]

    with torch.inference_mode():
        direct = model(pixel_values=pixel_values, return_dict=True).logits
        features, manual = _run_stages(model, pixel_values)
    max_abs_error = float((manual - direct).abs().max().item())
    mean_abs_error = float((manual - direct).abs().mean().item())
    relative_l2 = float(
        torch.linalg.vector_norm((manual - direct).float())
        / torch.linalg.vector_norm(direct.float()).clamp_min(1e-12)
    )
    if not torch.allclose(manual, direct, rtol=1e-5, atol=1e-6):
        raise RuntimeError(
            "Manual staged path does not match model forward: "
            f"max_abs_error={max_abs_error}"
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _run_stages(model, pixel_values)
            model(pixel_values=pixel_values, return_dict=True)
    torch.cuda.synchronize(device)

    component_values = {name: [] for name in (*COMPONENTS, "manual_total")}
    direct_values = []
    with torch.inference_mode():
        for _ in range(args.measurements):
            timings, _ = _timed_manual_forward(model, pixel_values)
            for name, value in timings.items():
                component_values[name].append(value)
        for _ in range(args.measurements):
            direct_values.append(_time_full_forward(model, pixel_values))

    preprocessing_values = _profile_cpu_preprocessing(
        processor,
        image,
        warmup=args.warmup,
        measurements=args.measurements,
    )
    rows = []
    for component in COMPONENTS:
        rows.append(
            {
                "component": component,
                **_stats(component_values[component]),
                "timing_backend": "cuda_event",
                "included_in_model_total": True,
            }
        )
    rows.extend(
        [
            {
                "component": "manual_staged_total",
                **_stats(component_values["manual_total"]),
                "timing_backend": "cuda_event",
                "included_in_model_total": True,
            },
            {
                "component": "direct_full_model",
                **_stats(direct_values),
                "timing_backend": "cuda_event",
                "included_in_model_total": True,
            },
            {
                "component": "input_preprocessing_cpu",
                **_stats(preprocessing_values),
                "timing_backend": "perf_counter_cpu",
                "included_in_model_total": False,
            },
        ]
    )
    direct_mean = _stats(direct_values)["mean_ms"]
    stage4_mean = _stats(component_values["stage_4"])["mean_ms"]
    component_sum = sum(
        _stats(component_values[name])["mean_ms"] for name in COMPONENTS
    )
    summary = {
        "torch_version": torch.__version__,
        "model": args.model,
        "device": torch.cuda.get_device_name(device),
        "input_shape": list(pixel_values.shape),
        "input_dtype": str(pixel_values.dtype),
        "warmup": args.warmup,
        "measurements": args.measurements,
        "manual_equivalence": {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "relative_l2": relative_l2,
            "allclose_rtol": 1e-5,
            "allclose_atol": 1e-6,
            "passed": True,
        },
        "stage_shapes": [list(feature.shape) for feature in features],
        "direct_full_mean_ms": direct_mean,
        "manual_total_mean_ms": _stats(
            component_values["manual_total"]
        )["mean_ms"],
        "component_sum_mean_ms": component_sum,
        "stage4_mean_ms": stage4_mean,
        "stage4_fraction_of_direct_full": stage4_mean / direct_mean,
        "preprocessing_boundary": (
            "CPU AutoImageProcessor is reported separately and excluded from "
            "the CUDA-resident model-input-to-logits total."
        ),
    }

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "stage_latency.csv", rows)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    for row in rows:
        print(
            f"{row['component']}: mean={row['mean_ms']:.6f} ms "
            f"p95={row['p95_ms']:.6f} ms"
        )
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
