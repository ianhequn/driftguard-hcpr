"""Profile the Round 4 GPU-resident AFVT optimization waterfall."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.age_aware_schedulers import (  # noqa: E402
    partition_afvt,
    select_afvt,
)
from driftguard.analysis.residual_patches import selected_residual_patches  # noqa: E402
from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.profiling.selector_gpu import (  # noqa: E402
    GPUAFVTSelector,
    gpu_rgb_residual,
    resize_candidate_blocks,
    vectorized_all_blocks,
)
from driftguard.utils.video import iter_video_frames  # noqa: E402


CORE_RATIOS = (0.25, 0.375, 0.5)
ORIGINAL_37P5_MS = 38.40393970999867


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_predictor(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = TinyRiskCNN(metadata_dim=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return model, mean, std


def _frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(device)
    )


def _load_samples(
    inventory_path: Path,
    diagnostics_path: Path,
    device: torch.device,
    split: str,
) -> dict[float, list[dict]]:
    inventory = {
        row["video_id"]: row
        for row in _read_csv(inventory_path)
        if row["split"] == split
    }
    frame_maps = {}
    for video_id, record in inventory.items():
        frames = {
            frame_id: frame
            for frame_id, frame in iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=48,
            )
        }
        frame_maps[video_id] = frames
    rows = [
        row
        for row in _read_csv(diagnostics_path)
        if row["method"] == "afvt"
        and float(row["budget"]) in CORE_RATIOS
        and row["video_id"] in inventory
    ]
    samples = {ratio: [] for ratio in CORE_RATIOS}
    gpu_frames: dict[tuple[str, int], torch.Tensor] = {}
    for row in rows:
        video_id = row["video_id"]
        frame_id = int(row["frame_id"])
        frames = frame_maps[video_id]
        for value in (frame_id - 1, frame_id):
            key = (video_id, value)
            if key not in gpu_frames:
                gpu_frames[key] = _frame_tensor(frames[value], device)
        samples[float(row["budget"])].append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "current_np": frames[frame_id],
                "reference_np": frames[frame_id - 1],
                "current": gpu_frames[(video_id, frame_id)],
                "reference": gpu_frames[(video_id, frame_id - 1)],
                "ages": np.asarray(
                    [float(value) for value in row["ages_before"].split()],
                    dtype=np.float32,
                ),
            }
        )
    return samples


def _metadata(
    ages: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    values = ((ages[indices, None] - mean) / std).astype(np.float32)
    return torch.from_numpy(values).to(device)


def _rank(
    ages: np.ndarray,
    ratio: float,
    candidates: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    score_map = {
        int(index): float(score)
        for index, score in zip(candidates, scores)
    }

    def score_fn(indices):
        return np.asarray(
            [score_map[int(index)] for index in indices],
            dtype=np.float32,
        )

    return select_afvt(ages, ratio, score_fn).mask


def _old_selector(model, mean, std, device, sample, ratio):
    def score(indices):
        patches = selected_residual_patches(
            sample["current_np"],
            sample["reference_np"],
            indices,
            grid=4,
            size=16,
        )
        metadata = _metadata(
            sample["ages"], indices, mean, std, device
        )
        with torch.inference_mode():
            return (
                model(torch.from_numpy(patches).to(device), metadata)
                .detach()
                .cpu()
                .numpy()
            )

    return select_afvt(sample["ages"], ratio, score).mask


def _visual_candidates(ages: np.ndarray, ratio: float):
    partition = partition_afvt(ages, ratio)
    return partition, partition.tie.astype(np.int64, copy=False)


def _stage_selector(
    stage: str,
    model,
    mean,
    std,
    device,
    all_selector: GPUAFVTSelector,
    direct_selector: GPUAFVTSelector,
    no_fast_selector: GPUAFVTSelector,
    sample,
    ratio,
):
    if stage == "O0 Original":
        return _old_selector(model, mean, std, device, sample, ratio)
    if stage == "O4 Candidate Batch":
        return no_fast_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        ).decision.mask
    if stage == "O5 Fast Path":
        return all_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        ).decision.mask
    if stage == "O6 Direct Gather":
        return direct_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        ).decision.mask
    if stage == "O7 Final":
        return all_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        ).decision.mask

    partition, candidates = _visual_candidates(sample["ages"], ratio)
    mask = np.zeros(16, dtype=bool)
    mask[partition.older] = True
    if not partition.needs_visual_ranking:
        mask[partition.tie[: partition.remaining]] = True
        return mask

    residual = gpu_rgb_residual(sample["current"], sample["reference"])
    if stage == "O1 GPU Residual":
        height, width = residual.shape[-2:]
        block_h, block_w = height // 4, width // 4
        values = []
        for index in candidates:
            row, col = divmod(int(index), 4)
            crop = residual[
                :,
                :,
                row * block_h : (row + 1) * block_h,
                col * block_w : (col + 1) * block_w,
            ]
            values.append(
                F.interpolate(
                    crop,
                    size=(16, 16),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
        patches = torch.stack(values)
    else:
        all_blocks = vectorized_all_blocks(residual, 4)
        indices = torch.as_tensor(
            candidates, dtype=torch.int64, device=device
        )
        if stage == "O2 GPU Patch":
            values = [
                F.interpolate(
                    all_blocks[index : index + 1],
                    size=(16, 16),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                for index in indices.tolist()
            ]
            patches = torch.stack(values)
        elif stage == "O3 GPU Resize":
            patches = resize_candidate_blocks(all_blocks, 16).index_select(
                0, indices
            )
        else:
            raise ValueError(stage)
    metadata = _metadata(
        sample["ages"], candidates, mean, std, device
    )
    with torch.inference_mode():
        scores = model(patches, metadata).detach().cpu().numpy()
    return _rank(sample["ages"], ratio, candidates, scores)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "Mean": float(array.mean()),
        "Median": float(np.median(array)),
        "P90": float(np.quantile(array, 0.90)),
        "P95": float(np.quantile(array, 0.95)),
        "P99": float(np.quantile(array, 0.99)),
        "Min": float(array.min()),
        "Max": float(array.max()),
        "Std": float(array.std()),
    }


def _benchmark(
    fn,
    samples: list[dict],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, float], float]:
    if not samples:
        empty = {
            "Mean": float("nan"),
            "Median": float("nan"),
            "P90": float("nan"),
            "P95": float("nan"),
            "P99": float("nan"),
            "Min": float("nan"),
            "Max": float("nan"),
            "Std": float("nan"),
        }
        return empty, float("nan")
    for index in range(warmup):
        fn(samples[index % len(samples)])
    torch.cuda.synchronize(device)
    wall = []
    gpu_span = []
    for index in range(repeats):
        sample = samples[index % len(samples)]
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_event.record()
        fn(sample)
        end_event.record()
        end_event.synchronize()
        wall.append((time.perf_counter() - start_wall) * 1000.0)
        gpu_span.append(float(start_event.elapsed_time(end_event)))
    return _stats(wall), float(np.mean(gpu_span))


def _benchmark_h2d(
    selector: GPUAFVTSelector,
    samples: list[dict],
    ratio: float,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float]:
    def run(sample):
        current = _frame_tensor(sample["current_np"], device)
        reference = _frame_tensor(sample["reference_np"], device)
        return selector.select(current, reference, sample["ages"], ratio)

    stats, _ = _benchmark(run, samples, warmup, repeats, device)
    return stats


def _prime_cuda(device: torch.device) -> None:
    values = torch.randn((2048, 2048), device=device)
    for _ in range(20):
        values = values @ values
    torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--diagnostics",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "scheduler_frame_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--output-root",
        default="results/R4_E07_profiling",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.repeats < 500:
        raise ValueError("Round 4 requires warmup>=200 and repeats>=500")
    device = torch.device("cuda")
    model, mean, std = _load_predictor(
        PROJECT_ROOT / args.predictor, device
    )
    samples = _load_samples(
        PROJECT_ROOT / args.inventory,
        PROJECT_ROOT / args.diagnostics,
        device,
        args.split,
    )
    all_selector = GPUAFVTSelector(
        model, mean, std, extraction="all_blocks", fast_path=True
    )
    direct_selector = GPUAFVTSelector(
        model, mean, std, extraction="direct", fast_path=True
    )
    no_fast_selector = GPUAFVTSelector(
        model, mean, std, extraction="all_blocks", fast_path=False
    )
    _prime_cuda(device)
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    stages = (
        "O0 Original",
        "O1 GPU Residual",
        "O2 GPU Patch",
        "O3 GPU Resize",
        "O4 Candidate Batch",
        "O5 Fast Path",
        "O6 Direct Gather",
        "O7 Final",
    )
    waterfall = []
    reference_masks = [
        _old_selector(model, mean, std, device, sample, 0.375)
        for sample in samples[0.375]
    ]
    for stage in stages:
        fn = lambda sample, stage=stage: _stage_selector(
            stage,
            model,
            mean,
            std,
            device,
            all_selector,
            direct_selector,
            no_fast_selector,
            sample,
            0.375,
        )
        stats, gpu_span = _benchmark(
            fn, samples[0.375], args.warmup, args.repeats, device
        )
        matches = [
            np.array_equal(fn(sample), expected)
            for sample, expected in zip(samples[0.375], reference_masks)
        ]
        waterfall.append(
            {
                "OptimizationStage": stage,
                **stats,
                "GPUEventSpan": gpu_span,
                "SpeedupVsO0": 0.0,
                "SpeedupVsFrozenRound3": (
                    ORIGINAL_37P5_MS / stats["Mean"]
                ),
                "SelectedBlockEquivalence": float(np.mean(matches)),
            }
        )
        print(stage, waterfall[-1], flush=True)

    measured_o0 = float(waterfall[0]["Mean"])
    for row in waterfall:
        row["SpeedupVsO0"] = measured_o0 / float(row["Mean"])

    latency_rows = []
    for ratio in CORE_RATIOS:
        ratio_samples = samples[ratio]
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = torch.cuda.memory_allocated(device)
        fn = lambda sample, ratio=ratio: all_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        )
        stats, gpu_span = _benchmark(
            fn, ratio_samples, args.warmup, args.repeats, device
        )
        peak = (
            torch.cuda.max_memory_allocated(device) - start_allocated
        ) / (1024**2)
        h2d = _benchmark_h2d(
            all_selector,
            ratio_samples,
            ratio,
            args.warmup,
            args.repeats,
            device,
        )
        visual = [
            sample
            for sample in ratio_samples
            if partition_afvt(
                sample["ages"], ratio
            ).needs_visual_ranking
        ]
        fast = [
            sample
            for sample in ratio_samples
            if not partition_afvt(
                sample["ages"], ratio
            ).needs_visual_ranking
        ]
        visual_stats, _ = _benchmark(
            fn, visual, args.warmup, args.repeats, device
        )
        fast_stats, _ = _benchmark(
            fn, fast, args.warmup, args.repeats, device
        )
        latency_rows.append(
            {
                "Implementation": "GPU-Resident AFVT FP32",
                "Budget": ratio,
                **stats,
                "GPUActiveTime": gpu_span,
                "WallClockTime": stats["Mean"],
                "IncludingH2DMean": h2d["Mean"],
                "PeakMemoryMB": float(peak),
                "FastPathMean": fast_stats["Mean"],
                "VisualPathMean": visual_stats["Mean"],
                "FastPathFrames": len(fast),
                "VisualPathFrames": len(visual),
            }
        )
        print("final", latency_rows[-1], flush=True)

    breakdown = []
    ratio = 0.375
    ratio_samples = samples[ratio]
    visual_samples = [
        sample
        for sample in ratio_samples
        if partition_afvt(sample["ages"], ratio).needs_visual_ranking
    ]
    prepared = []
    for sample in visual_samples:
        partition = partition_afvt(sample["ages"], ratio)
        candidates = partition.tie.astype(np.int64, copy=False)
        indices = torch.as_tensor(
            candidates, dtype=torch.int64, device=device
        )
        residual = gpu_rgb_residual(
            sample["current"], sample["reference"]
        )
        blocks = vectorized_all_blocks(residual, 4)
        patches = resize_candidate_blocks(
            blocks.index_select(0, indices), 16
        )
        metadata = _metadata(
            sample["ages"], candidates, mean, std, device
        )
        with torch.inference_mode():
            scores = model(patches, metadata).detach().cpu().numpy()
        prepared.append(
            {
                **sample,
                "partition": partition,
                "candidates": candidates,
                "indices": indices,
                "residual": residual,
                "blocks": blocks,
                "patches": patches,
                "metadata": metadata,
                "scores": scores,
            }
        )

    component_functions = {
        "AgeLogic": lambda sample: partition_afvt(
            sample["ages"], ratio
        ),
        "Residual": lambda sample: gpu_rgb_residual(
            sample["current"], sample["reference"]
        ),
        "BlockExtraction": lambda sample: vectorized_all_blocks(
            sample["residual"], 4
        ).index_select(0, sample["indices"]),
        "Resize": lambda sample: resize_candidate_blocks(
            sample["blocks"].index_select(0, sample["indices"]), 16
        ),
        "Metadata": lambda sample: _metadata(
            sample["ages"],
            sample["candidates"],
            mean,
            std,
            device,
        ),
        "CNN": lambda sample: model(
            sample["patches"], sample["metadata"]
        ),
        "Ranking": lambda sample: _rank(
            sample["ages"],
            ratio,
            sample["candidates"],
            sample["scores"],
        ),
        "Total": lambda sample: all_selector.select(
            sample["current"],
            sample["reference"],
            sample["ages"],
            ratio,
        ),
    }
    for component, component_fn in component_functions.items():
        component_samples = (
            ratio_samples
            if component in {"AgeLogic", "Total"}
            else prepared
        )
        stats, gpu_span = _benchmark(
            component_fn,
            component_samples,
            args.warmup,
            args.repeats,
            device,
        )
        breakdown.append(
            {
                "Component": component,
                "Budget": ratio,
                **stats,
                "GPUEventSpan": gpu_span,
            }
        )

    _write_csv(output_root / "optimization_waterfall.csv", waterfall)
    _write_csv(output_root / "selector_latency.csv", latency_rows)
    _write_csv(output_root / "selector_breakdown.csv", breakdown)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "final_extraction": "all_blocks",
                "original_37p5_ms": ORIGINAL_37P5_MS,
                "waterfall": waterfall,
                "latency": latency_rows,
                "breakdown": breakdown,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "config.yaml").write_text(
        f"split: {args.split}\nwarmup: {args.warmup}\n"
        f"repeats: {args.repeats}\nprecision: fp32\n"
        "final_extraction: all_blocks\ninput: cuda_resident_bchw_uint8\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )

    plt.figure(figsize=(9, 5.4))
    plt.bar(
        np.arange(len(waterfall)),
        [row["Mean"] for row in waterfall],
    )
    plt.xticks(
        np.arange(len(waterfall)),
        [row["OptimizationStage"].replace(" ", "\n", 1) for row in waterfall],
        fontsize=8,
    )
    plt.ylabel("Mean selector latency (ms/frame)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_root / "selector_optimization_waterfall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.2))
    x = [row["Budget"] * 100 for row in latency_rows]
    plt.plot(x, [row["Mean"] for row in latency_rows], marker="o", label="Mean")
    plt.plot(x, [row["P95"] for row in latency_rows], marker="o", label="P95")
    plt.axhline(2.0, color="red", linestyle="--", label="2 ms readiness")
    plt.xlabel("Refresh budget (%)")
    plt.ylabel("Selector latency (ms/frame)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "selector_latency_by_budget.png", dpi=180)
    plt.close()

    row = next(value for value in latency_rows if value["Budget"] == 0.375)
    plt.figure(figsize=(7.2, 5.0))
    plt.bar(
        ["Fast path", "Visual path", "Overall"],
        [row["FastPathMean"], row["VisualPathMean"], row["Mean"]],
    )
    plt.ylabel("Mean latency (ms/frame)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_root / "fast_path_vs_visual_path_latency.png", dpi=180)
    plt.close()

    fn_37 = lambda sample: all_selector.select(
        sample["current"],
        sample["reference"],
        sample["ages"],
        0.375,
    )
    counts = []
    latencies = []
    for sample in samples[0.375]:
        count = len(partition_afvt(sample["ages"], 0.375).tie)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn_37(sample)
        torch.cuda.synchronize(device)
        counts.append(count)
        latencies.append((time.perf_counter() - start) * 1000.0)
    plt.figure(figsize=(7.2, 5.0))
    plt.scatter(counts, latencies, alpha=0.65)
    plt.xlabel("CNN candidate blocks")
    plt.ylabel("Selector latency (ms/frame)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_root / "cnn_candidate_blocks_vs_latency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.hist(latencies, bins=20)
    plt.xlabel("37.5% selector latency (ms/frame)")
    plt.ylabel("Frames")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        output_root / "selector_latency_distribution_37p5.png",
        dpi=180,
    )
    plt.close()


if __name__ == "__main__":
    main()
