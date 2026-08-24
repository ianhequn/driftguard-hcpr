"""Run the single frozen-test evaluation for Round 6 structured reuse."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.risk_predictor import TinyRiskCNN  # noqa: E402
from driftguard.models.segformer import load_segformer, prepare_inputs  # noqa: E402
from driftguard.profiling.selector_gpu import (  # noqa: E402
    gpu_rgb_residual,
    resize_candidate_blocks,
    vectorized_all_blocks,
)
from driftguard.structured_reuse.pipeline import (  # noqa: E402
    Stage4CacheState,
    StructuredReusePipeline,
)
from driftguard.structured_reuse.policies import (  # noqa: E402
    FixedIntervalPolicy,
    StageAgeVisualPolicy,
    VisualRiskPolicy,
    aggregate_frame_risk,
)
from driftguard.utils.common import set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _task_metrics  # noqa: E402


VISUAL_AGGREGATION = "max"
VISUAL_THRESHOLD = -2.2600908279418945
STAGE_AGE_VISUAL_MAX_AGE = 5
FIXED_INTERVALS = (1, 2, 3, 4, 6, 8)


@dataclass
class TestVideo:
    video_id: str
    motion_level: str
    frames: list[np.ndarray]
    pixel_values: list[torch.Tensor]
    rgb_gpu: list[torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/segformer-b0",
    )
    parser.add_argument(
        "--predictor",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float], prefix: str = "") -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    key = f"{prefix}_" if prefix else ""
    if not len(array):
        return {
            f"{key}mean_ms": float("nan"),
            f"{key}median_ms": float("nan"),
            f"{key}p90_ms": float("nan"),
            f"{key}p95_ms": float("nan"),
            f"{key}p99_ms": float("nan"),
            f"{key}std_ms": float("nan"),
        }
    return {
        f"{key}mean_ms": float(array.mean()),
        f"{key}median_ms": float(np.median(array)),
        f"{key}p90_ms": float(np.quantile(array, 0.90)),
        f"{key}p95_ms": float(np.quantile(array, 0.95)),
        f"{key}p99_ms": float(np.quantile(array, 0.99)),
        f"{key}std_ms": float(array.std()),
    }


def load_predictor(
    path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    checkpoint = torch.load(path, map_location="cpu")
    predictor = TinyRiskCNN(metadata_dim=1)
    predictor.load_state_dict(checkpoint["state_dict"])
    predictor.to(device).eval()
    mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
    std[std < 1e-8] = 1.0
    return predictor, mean, std


def load_test_videos(
    records: list[dict],
    processor,
    device: torch.device,
    max_frames: int,
) -> list[TestVideo]:
    videos = []
    for record in records:
        if record["split"] != "test":
            continue
        frames = [
            frame
            for _, frame in iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=max_frames,
            )
        ]
        pixel_values = []
        rgb_gpu = []
        for frame in frames:
            image = Image.fromarray(frame, mode="RGB")
            pixel_values.append(
                prepare_inputs(processor, image, device)["pixel_values"]
            )
            rgb_gpu.append(
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
                .to(device)
            )
        videos.append(
            TestVideo(
                video_id=record["video_id"],
                motion_level=record["motion_level"],
                frames=frames,
                pixel_values=pixel_values,
                rgb_gpu=rgb_gpu,
            )
        )
    if len(videos) != 3:
        raise RuntimeError("Round 6 requires the frozen three test videos")
    return videos


def block_scores(
    predictor,
    mean: np.ndarray,
    std: np.ndarray,
    current: torch.Tensor,
    previous: torch.Tensor,
    age: int,
) -> np.ndarray:
    residual = gpu_rgb_residual(current, previous)
    patches = resize_candidate_blocks(
        vectorized_all_blocks(residual, grid=4),
        size=16,
    )
    metadata_np = np.full((16, 1), float(age), dtype=np.float32)
    metadata = torch.from_numpy((metadata_np - mean) / std).to(
        current.device
    )
    with torch.inference_mode():
        return (
            predictor(patches, metadata)
            .detach()
            .float()
            .cpu()
            .numpy()
        )


def make_specs() -> list[dict]:
    specs = [{"name": "Full", "kind": "fixed", "interval": 1}]
    for interval in FIXED_INTERVALS[1:]:
        specs.append(
            {
                "name": f"FixedInterval-{interval}",
                "kind": "fixed",
                "interval": interval,
            }
        )
    specs.extend(
        [
            {
                "name": "BestStageAge",
                "kind": "fixed",
                "interval": 2,
                "alias": "StageAge-H1-equivalent",
            },
            {
                "name": "BestVisualRisk",
                "kind": "visual",
                "threshold": VISUAL_THRESHOLD,
            },
            {
                "name": "BestStageAgeVisual",
                "kind": "age_visual",
                "threshold": VISUAL_THRESHOLD,
                "max_age": STAGE_AGE_VISUAL_MAX_AGE,
            },
        ]
    )
    return specs


def decide(
    spec: dict,
    frame_index: int,
    state: Stage4CacheState | None,
    current_rgb: torch.Tensor,
    previous_rgb: torch.Tensor | None,
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
) -> tuple[bool, float, float]:
    age = 0 if state is None else state.age
    if frame_index == 0 or state is None or not state.initialized:
        return True, 0.0, 0.0
    if spec["kind"] == "fixed":
        refresh = frame_index % int(spec["interval"]) == 0
        return refresh, 0.0, 0.0
    if previous_rgb is None:
        return True, 0.0, 0.0
    selector_start = torch.cuda.Event(enable_timing=True)
    selector_end = torch.cuda.Event(enable_timing=True)
    selector_start.record()
    scores = block_scores(
        predictor,
        predictor_mean,
        predictor_std,
        current_rgb,
        previous_rgb,
        age,
    )
    risk = aggregate_frame_risk(scores, VISUAL_AGGREGATION)
    if spec["kind"] == "visual":
        refresh = risk > float(spec["threshold"])
    else:
        refresh = (
            age >= int(spec["max_age"])
            or risk > float(spec["threshold"])
        )
    selector_end.record()
    selector_end.synchronize()
    return refresh, risk, float(selector_start.elapsed_time(selector_end))


def quality_pass(
    model,
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
    videos: list[TestVideo],
    specs: list[dict],
) -> list[dict]:
    rows = []
    for spec in specs:
        pipeline = StructuredReusePipeline(model)
        for video in videos:
            state = None
            previous_rgb = None
            for frame_index, (pixel_values, current_rgb) in enumerate(
                zip(video.pixel_values, video.rgb_gpu)
            ):
                with torch.inference_mode():
                    full_logits = model(
                        pixel_values=pixel_values,
                        return_dict=True,
                    ).logits
                age_before = 0 if state is None else state.age
                refresh, risk, _ = decide(
                    spec,
                    frame_index,
                    state,
                    current_rgb,
                    previous_rgb,
                    predictor,
                    predictor_mean,
                    predictor_std,
                )
                with torch.inference_mode():
                    policy_logits, state, frame_stats = pipeline.forward(
                        pixel_values,
                        refresh,
                        state,
                    )
                metrics = _task_metrics(
                    full_logits,
                    policy_logits,
                    grid=4,
                )
                rows.append(
                    {
                        "method": spec["name"],
                        "video_id": video.video_id,
                        "motion_level": video.motion_level,
                        "frame": frame_index,
                        "refresh": int(frame_stats["actual_refresh"]),
                        "stage_cache_age_before": age_before,
                        "stage_cache_age": frame_stats["cache_age"],
                        "frame_risk": risk,
                        "logit_relative_l2": float(
                            metrics["logit_relative_l2"][0]
                        ),
                        "kl": float(metrics["global_kl"][0]),
                        "pixel_disagreement": float(
                            metrics["global_pixel_disagreement"][0]
                        ),
                        "stage4_forward_calls": frame_stats[
                            "stage4_forward_calls"
                        ],
                    }
                )
                previous_rgb = current_rgb
    return rows


def sequence_samples(videos: list[TestVideo], count: int):
    video_index = 0
    frame_index = 0
    for _ in range(count):
        video = videos[video_index]
        yield video, frame_index
        frame_index += 1
        if frame_index >= len(video.frames):
            frame_index = 0
            video_index = (video_index + 1) % len(videos)


def latency_pass(
    model,
    predictor,
    predictor_mean: np.ndarray,
    predictor_std: np.ndarray,
    videos: list[TestVideo],
    specs: list[dict],
    warmup: int,
    measurements: int,
) -> list[dict]:
    rows = []
    for spec in specs:
        pipeline = StructuredReusePipeline(model)
        state_by_video: dict[str, Stage4CacheState | None] = {
            video.video_id: None for video in videos
        }
        previous_by_video: dict[str, torch.Tensor | None] = {
            video.video_id: None for video in videos
        }

        def run_one(video: TestVideo, frame_index: int) -> dict:
            state = state_by_video[video.video_id]
            previous_rgb = previous_by_video[video.video_id]
            current_rgb = video.rgb_gpu[frame_index]
            pixel_values = video.pixel_values[frame_index]
            if frame_index == 0:
                state = None
                previous_rgb = None
            outer_start = torch.cuda.Event(enable_timing=True)
            outer_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            outer_start.record()
            if spec["kind"] == "fixed":
                refresh = frame_index % int(spec["interval"]) == 0
                risk = 0.0
                selector_ms = 0.0
            else:
                selector_wall_start = time.perf_counter()
                age = 0 if state is None else state.age
                if previous_rgb is None:
                    refresh = True
                    risk = 0.0
                else:
                    scores = block_scores(
                        predictor,
                        predictor_mean,
                        predictor_std,
                        current_rgb,
                        previous_rgb,
                        age,
                    )
                    risk = aggregate_frame_risk(scores, VISUAL_AGGREGATION)
                    if spec["kind"] == "visual":
                        refresh = risk > float(spec["threshold"])
                    else:
                        refresh = (
                            age >= int(spec["max_age"])
                            or risk > float(spec["threshold"])
                        )
                torch.cuda.synchronize()
                selector_ms = (
                    time.perf_counter() - selector_wall_start
                ) * 1000.0
            with torch.inference_mode():
                _, state, frame_stats = pipeline.forward(
                    pixel_values,
                    refresh,
                    state,
                )
            torch.cuda.synchronize()
            outer_end.record()
            outer_end.synchronize()
            state_by_video[video.video_id] = state
            previous_by_video[video.video_id] = current_rgb
            return {
                "refresh": int(frame_stats["actual_refresh"]),
                "cache_age": frame_stats["cache_age"],
                "risk": risk,
                "cuda_ms": float(outer_start.elapsed_time(outer_end)),
                "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
                "selector_ms": selector_ms,
            }

        for video, frame_index in sequence_samples(videos, warmup):
            run_one(video, frame_index)
        state_by_video = {video.video_id: None for video in videos}
        previous_by_video = {video.video_id: None for video in videos}
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        measured = [
            run_one(video, frame_index)
            for video, frame_index in sequence_samples(videos, measurements)
        ]
        peak_mb = (
            torch.cuda.max_memory_allocated() - start_allocated
        ) / (1024**2)
        all_values = [row["wall_ms"] for row in measured]
        refresh_values = [row["wall_ms"] for row in measured if row["refresh"]]
        reuse_values = [row["wall_ms"] for row in measured if not row["refresh"]]
        selector_values = [row["selector_ms"] for row in measured]
        rows.append(
            {
                "method": spec["name"],
                "config": spec.get("interval", spec.get("threshold", "")),
                "actual_refresh_rate": float(
                    np.mean([row["refresh"] for row in measured])
                ),
                "stage4_forward_calls": int(
                    sum(row["refresh"] for row in measured)
                ),
                "persistent_cache_mb": 0.25,
                "peak_temporary_mb": float(peak_mb),
                "fps": 1000.0 / float(np.mean(all_values)),
                **distribution(all_values),
                **distribution(refresh_values, "refresh"),
                **distribution(reuse_values, "reuse"),
                **distribution(selector_values, "selector"),
            }
        )
        print(rows[-1], flush=True)
    full_rows = [row for row in rows if row["method"] == "Full"]
    if full_rows:
        full_mean = full_rows[0]["mean_ms"]
        for row in rows:
            row["speedup"] = full_mean / row["mean_ms"]
    return rows


def add_summary_fields(
    quality_rows: list[dict],
    latency_rows: list[dict],
) -> list[dict]:
    quality_groups: dict[str, list[dict]] = {}
    for row in quality_rows:
        quality_groups.setdefault(row["method"], []).append(row)
    latency_by_method = {row["method"]: row for row in latency_rows}
    combined = []
    for method, values in quality_groups.items():
        latency = latency_by_method[method]
        combined.append(
            {
                "Method": method,
                "ActualRefreshRate": float(
                    np.mean([row["refresh"] for row in values])
                ),
                "MeanLatency": latency["mean_ms"],
                "P95Latency": latency["p95_ms"],
                "FPS": latency["fps"],
                "Speedup": latency["speedup"],
                "LogitRelativeL2": float(
                    np.mean([row["logit_relative_l2"] for row in values])
                ),
                "KL": float(np.mean([row["kl"] for row in values])),
                "PixelDisagreement": float(
                    np.mean([row["pixel_disagreement"] for row in values])
                ),
                "PersistentCacheMB": latency["persistent_cache_mb"],
                "P95Refresh": latency["refresh_p95_ms"],
                "P95Reuse": latency["reuse_p95_ms"],
                "SelectorMeanMS": latency["selector_mean_ms"],
                "SelectorP95MS": latency["selector_p95_ms"],
                "Stage4ForwardCalls": latency["stage4_forward_calls"],
            }
        )
    return combined


def make_plots(
    root: Path,
    main_rows: list[dict],
    quality_rows: list[dict],
    timeline_method: str,
) -> None:
    plt.figure(figsize=(8, 5.5))
    for row in main_rows:
        plt.scatter(
            row["MeanLatency"],
            row["LogitRelativeL2"],
            s=55,
            label=row["Method"],
        )
        plt.annotate(
            row["Method"].replace("FixedInterval-", "FI"),
            (row["MeanLatency"], row["LogitRelativeL2"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("Measured end-to-end latency (ms)")
    plt.ylabel("Logit Relative L2")
    plt.title("Structured reuse accuracy / latency Pareto")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "structured_accuracy_latency_pareto.png", dpi=180)
    plt.close()

    rates = [row["ActualRefreshRate"] * 100 for row in main_rows]
    latencies = [row["MeanLatency"] for row in main_rows]
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(rates, latencies, "o-")
    plt.xlabel("Actual Stage-4 refresh rate (%)")
    plt.ylabel("Measured end-to-end latency (ms)")
    plt.title("Refresh rate vs latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "refresh_rate_vs_latency.png", dpi=180)
    plt.close()

    damages = [row["LogitRelativeL2"] for row in main_rows]
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(rates, damages, "o-")
    plt.xlabel("Actual Stage-4 refresh rate (%)")
    plt.ylabel("Logit Relative L2")
    plt.title("Refresh rate vs task damage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "refresh_rate_vs_task_damage.png", dpi=180)
    plt.close()

    age_rows = [
        row
        for row in quality_rows
        if row["method"] == timeline_method and not row["refresh"]
    ]
    bins: dict[str, list[float]] = {}
    for row in age_rows:
        age = int(row["stage_cache_age_before"])
        key = str(age) if age < 5 else "5+"
        bins.setdefault(key, []).append(row["logit_relative_l2"])
    labels = list(bins)
    values = [float(np.mean(bins[label])) for label in labels]
    plt.figure(figsize=(7.5, 5.0))
    plt.bar(labels, values)
    plt.xlabel("Stage cache age before reuse")
    plt.ylabel("Mean Logit Relative L2")
    plt.title("Stage cache age vs task damage")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "stage_cache_age_vs_task_damage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(
        [row["frame_risk"] for row in age_rows],
        [row["logit_relative_l2"] for row in age_rows],
        alpha=0.65,
    )
    plt.xlabel("Frame visual risk")
    plt.ylabel("Task damage on reuse frames")
    plt.title("Frame risk vs task damage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "frame_risk_vs_task_damage.png", dpi=180)
    plt.close()

    timeline = [
        row
        for row in quality_rows
        if row["method"] == timeline_method
        and row["motion_level"] == "high"
    ]
    if timeline:
        timeline = timeline[: len(timeline) // 3]
        x = [row["frame"] for row in timeline]
        figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(x, [row["frame_risk"] for row in timeline])
        axes[0].set_ylabel("Risk")
        axes[1].step(
            x,
            [row["stage_cache_age"] for row in timeline],
            where="post",
        )
        axes[1].set_ylabel("Age")
        axes[2].step(
            x,
            [row["refresh"] for row in timeline],
            where="post",
        )
        axes[2].set_ylabel("Refresh")
        axes[2].set_xlabel("Frame")
        figure.suptitle(f"{timeline_method} high-motion timeline")
        figure.tight_layout()
        figure.savefig(root / "timeline_refresh_decisions.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.measurements < 1000:
        raise ValueError("Round 6 requires warmup>=200 and measurements>=1000")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    device = torch.device("cuda")
    records = read_csv(PROJECT_ROOT / args.inventory)
    processor, model = load_segformer(
        args.model,
        device=device,
        local_files_only=True,
    )
    predictor, predictor_mean, predictor_std = load_predictor(
        PROJECT_ROOT / args.predictor,
        device,
    )
    videos = load_test_videos(
        records,
        processor,
        device,
        args.max_frames,
    )
    specs = make_specs()
    quality_rows = quality_pass(
        model,
        predictor,
        predictor_mean,
        predictor_std,
        videos,
        specs,
    )
    latency_rows = latency_pass(
        model,
        predictor,
        predictor_mean,
        predictor_std,
        videos,
        specs,
        args.warmup,
        args.measurements,
    )
    root = PROJECT_ROOT / "results/R6_E07_accuracy_latency"
    root.mkdir(parents=True, exist_ok=True)
    main_rows = add_summary_fields(quality_rows, latency_rows)
    write_csv(root / "round6_main_table.csv", main_rows)
    write_csv(root / "structured_reuse_main_table.csv", main_rows)
    write_csv(root / "test_frame_metrics.csv", quality_rows)
    write_csv(root / "latency_metrics.csv", latency_rows)

    fixed_rows = [
        row
        for row in main_rows
        if row["Method"].startswith("FixedInterval-")
    ]
    dynamic_rows = [
        row
        for row in main_rows
        if row["Method"] in {"BestVisualRisk", "BestStageAgeVisual"}
    ]
    matched_rows = []
    for dynamic in dynamic_rows:
        closest = min(
            fixed_rows,
            key=lambda fixed: abs(
                fixed["ActualRefreshRate"]
                - dynamic["ActualRefreshRate"]
            ),
        )
        matched_rows.append(
            {
                "dynamic_method": dynamic["Method"],
                "dynamic_refresh_rate": dynamic["ActualRefreshRate"],
                "dynamic_latency_ms": dynamic["MeanLatency"],
                "dynamic_logit_l2": dynamic["LogitRelativeL2"],
                "fixed_method": closest["Method"],
                "fixed_refresh_rate": closest["ActualRefreshRate"],
                "fixed_latency_ms": closest["MeanLatency"],
                "fixed_logit_l2": closest["LogitRelativeL2"],
            }
        )
    write_csv(root / "refresh_matched_comparison.csv", matched_rows)
    write_csv(
        PROJECT_ROOT / "results/R6_E06_task_quality/stage_age_vs_task_damage.csv",
        [
            {
                "method": row["Method"],
                "stage_cache_age_before": row["stage_cache_age_before"],
                "refresh": row["refresh"],
                "logit_relative_l2": row["logit_relative_l2"],
            }
            for row in quality_rows
            if not row["refresh"]
        ],
    )
    write_csv(
        PROJECT_ROOT / "results/R6_E06_task_quality/risk_damage_test.csv",
        [
            {
                "method": row["method"],
                "frame_risk": row["frame_risk"],
                "task_damage": row["logit_relative_l2"],
                "refresh": row["refresh"],
            }
            for row in quality_rows
        ],
    )
    make_plots(
        root,
        main_rows,
        quality_rows,
        "BestStageAgeVisual",
    )
    (PROJECT_ROOT / "results/R6_E04_age_visual").mkdir(
        parents=True,
        exist_ok=True,
    )
    (PROJECT_ROOT / "results/R6_E04_age_visual/selected_config.json").write_text(
        json.dumps(
            {
                "aggregation": VISUAL_AGGREGATION,
                "visual_threshold": VISUAL_THRESHOLD,
                "stage_age_visual_max_age": STAGE_AGE_VISUAL_MAX_AGE,
                "source": "validation_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "test_videos": [video.video_id for video in videos],
        "selected_visual_config": {
            "aggregation": VISUAL_AGGREGATION,
            "threshold": VISUAL_THRESHOLD,
        },
        "selected_stage_age_visual_config": {
            "aggregation": VISUAL_AGGREGATION,
            "threshold": VISUAL_THRESHOLD,
            "max_age": STAGE_AGE_VISUAL_MAX_AGE,
        },
        "main_table": main_rows,
        "refresh_matched": matched_rows,
        "measured_real_speedup": True,
        "structured_reuse_result": "pending_decision",
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "split: test\n"
        "aggregation: max\n"
        f"visual_threshold: {VISUAL_THRESHOLD}\n"
        f"stage_age_visual_max_age: {STAGE_AGE_VISUAL_MAX_AGE}\n"
        f"warmup: {args.warmup}\nmeasurements: {args.measurements}\n"
        "input: cuda_resident_fp32_pixel_values\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (root / "runtime.txt").write_text(
        "Completed one frozen-test quality pass and sequential latency pass.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
