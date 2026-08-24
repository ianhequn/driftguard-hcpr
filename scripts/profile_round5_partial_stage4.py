"""Profile cached-context partial Stage 4 against full Stage 4."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer, prepare_inputs  # noqa: E402
from driftguard.partial.stage4 import PartialStage4Engine  # noqa: E402
from driftguard.utils.common import set_seed  # noqa: E402


BUDGET_BLOCKS = {
    0.125: 2,
    0.25: 4,
    0.375: 6,
    0.5: 8,
    0.75: 12,
    1.0: 16,
}


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
        default="results/R5_E03_approx_partial",
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
    }


def _fresh_stage_1_to_3(model, pixel_values: torch.Tensor) -> list[torch.Tensor]:
    encoder = model.segformer.encoder
    hidden = pixel_values
    features = []
    for index in range(3):
        hidden, height, width = encoder.patch_embeddings[index](hidden)
        for block in encoder.block[index]:
            hidden = block(hidden, height, width, False)[0]
        hidden = encoder.layer_norm[index](hidden)
        hidden = (
            hidden.reshape(hidden.shape[0], height, width, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        features.append(hidden)
    return features


def _full_stage4(model, stage3_feature: torch.Tensor) -> torch.Tensor:
    encoder = model.segformer.encoder
    hidden, height, width = encoder.patch_embeddings[3](stage3_feature)
    for block in encoder.block[3]:
        hidden = block(hidden, height, width, False)[0]
    hidden = encoder.layer_norm[3](hidden)
    return (
        hidden.reshape(hidden.shape[0], height, width, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _event_time(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _mask(blocks: int) -> np.ndarray:
    order = np.asarray(
        [0, 15, 3, 12, 5, 10, 6, 9, 1, 14, 2, 13, 4, 11, 7, 8],
        dtype=np.int64,
    )
    mask = np.zeros(16, dtype=bool)
    mask[order[:blocks]] = True
    return mask


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.measurements < 1000:
        raise ValueError("Round 5 requires warmup>=200 and measurements>=1000")
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
    pixel_values_a = prepare_inputs(processor, image, device)["pixel_values"]
    pixel_values_b = torch.roll(pixel_values_a, shifts=1, dims=-1)
    with torch.inference_mode():
        stage3_a = _fresh_stage_1_to_3(model, pixel_values_a)[2]
        stage3_b = _fresh_stage_1_to_3(model, pixel_values_b)[2]
    engine = PartialStage4Engine(model)

    with torch.inference_mode():
        for index in range(args.warmup):
            _full_stage4(model, stage3_a if index % 2 == 0 else stage3_b)
    torch.cuda.synchronize(device)
    full_values = []
    with torch.inference_mode():
        for index in range(args.measurements):
            feature = stage3_a if index % 2 == 0 else stage3_b
            full_values.append(
                _event_time(lambda feature=feature: _full_stage4(model, feature))
            )

    rows = [
        {
            "method": "full_stage4",
            "budget": 1.0,
            "logical_refresh_blocks": 16,
            "physical_tokens": 256,
            "full_tokens": 256,
            "physical_compute_ratio": 1.0,
            "halo_tokens": 0,
            "persistent_cache_mb": 0.0,
            **_stats(full_values),
        }
    ]
    full_mean = rows[0]["mean_ms"]
    full_refresh_error = None
    for budget, blocks in BUDGET_BLOCKS.items():
        mask = _mask(blocks)
        with torch.inference_mode():
            _, cache, _ = engine.initialize(stage3_a)
            if budget == 1.0:
                expected = _full_stage4(model, stage3_b)
                actual, cache, stats = engine.partial_stage4(
                    stage3_b, cache, mask
                )
                full_refresh_error = {
                    "max_abs_error": float((actual - expected).abs().max().item()),
                    "mean_abs_error": float((actual - expected).abs().mean().item()),
                    "relative_l2": float(
                        torch.linalg.vector_norm((actual - expected).float())
                        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                    ),
                    "allclose": bool(
                        torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
                    ),
                }
            for index in range(args.warmup):
                feature = stage3_a if index % 2 == 0 else stage3_b
                _, cache, stats = engine.partial_stage4(
                    feature, cache, mask
                )
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = torch.cuda.memory_allocated(device)
        values = []
        with torch.inference_mode():
            for index in range(args.measurements):
                feature = stage3_a if index % 2 == 0 else stage3_b
                values.append(
                    _event_time(
                        lambda feature=feature: engine.partial_stage4(
                            feature, cache, mask
                        )
                    )
                )
        peak_temporary_mb = (
            torch.cuda.max_memory_allocated(device) - start_allocated
        ) / (1024**2)
        row = {
            "method": "cached_context_approx",
            "budget": budget,
            "logical_refresh_blocks": stats.logical_refresh_blocks,
            "physical_tokens": stats.physical_tokens,
            "full_tokens": stats.full_tokens,
            "physical_compute_ratio": stats.physical_compute_ratio,
            "halo_tokens": stats.halo_tokens,
            "persistent_cache_mb": cache.persistent_bytes / (1024**2),
            "peak_temporary_mb": float(peak_temporary_mb),
            **_stats(values),
        }
        row["stage4_speedup"] = full_mean / row["mean_ms"]
        rows.append(row)
        print(row, flush=True)

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "partial_stage4_latency.csv", rows)
    summary = {
        "warmup": args.warmup,
        "measurements": args.measurements,
        "full_stage4": rows[0],
        "partial": rows[1:],
        "full_refresh_equivalence": full_refresh_error,
        "early_stop_b": {
            "condition": "all real partial core budgets are >= full Stage 4",
            "triggered": all(
                row["mean_ms"] >= full_mean
                for row in rows[1:]
                if row["budget"] in {0.25, 0.375, 0.5}
            ),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
