"""Evaluate fixed-interval whole-Stage-4 reuse on frozen videos."""

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
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.structured_reuse.pipeline import (  # noqa: E402
    Stage4CacheState,
    StructuredReusePipeline,
)
from driftguard.structured_reuse.policies import (  # noqa: E402
    FixedIntervalPolicy,
)
from driftguard.utils.common import set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402
from generate_round2_task_dataset import _task_metrics  # noqa: E402


INTERVALS = (1, 2, 3, 4, 6, 8)


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
    if array.size == 0:
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


def prepare_split(
    records: list[dict],
    split: str,
    processor,
    device: torch.device,
    max_frames: int,
) -> list[dict]:
    videos = []
    for record in records:
        if record["split"] != split:
            continue
        frames = list(
            iter_video_frames(
                PROJECT_ROOT / record["path"],
                max_frames=max_frames,
            )
        )
        pixel_values = []
        for _, frame in frames:
            image = Image.fromarray(frame, mode="RGB")
            pixel_values.append(
                prepare_inputs(processor, image, device)["pixel_values"]
            )
        videos.append(
            {
                "video_id": record["video_id"],
                "motion_level": record["motion_level"],
                "frames": frames,
                "pixel_values": pixel_values,
            }
        )
    return videos


def quality_pass(model, videos: list[dict], split: str) -> list[dict]:
    policies = {
        interval: FixedIntervalPolicy(interval) for interval in INTERVALS
    }
    rows = []
    pipeline = StructuredReusePipeline(model)
    for video in videos:
        states = {
            interval: Stage4CacheState() for interval in INTERVALS
        }
        refresh_counts = {interval: 0 for interval in INTERVALS}
        for local_index, pixel_values in enumerate(video["pixel_values"]):
            with torch.inference_mode():
                outputs = model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                features = get_stage_features(outputs, model=model)
                full_logits = model.decode_head(features)
            variant_f4 = []
            descriptors = []
            for interval, policy in policies.items():
                state = states[interval]
                refresh = policy.decide(local_index, state.age)
                if not state.initialized:
                    refresh = True
                if refresh:
                    state.cached_f4 = features[3].detach()
                    state.age = 0
                    state.initialized = True
                    refresh_counts[interval] += 1
                else:
                    state.age += 1
                variant_f4.append(state.cached_f4)
                descriptors.append(
                    {
                        "interval": interval,
                        "refresh": refresh,
                        "cache_age": state.age,
                    }
                )
            batch_size = len(variant_f4)
            batch_features = [
                feature.expand(batch_size, -1, -1, -1)
                for feature in features
            ]
            batch_features[3] = torch.cat(variant_f4, dim=0)
            with torch.inference_mode():
                variants = model.decode_head(batch_features)
            metrics = _task_metrics(full_logits, variants, grid=4)
            for index, descriptor in enumerate(descriptors):
                rows.append(
                    {
                        "split": split,
                        "video_id": video["video_id"],
                        "motion_level": video["motion_level"],
                        "frame": local_index,
                        "method": (
                            "Full"
                            if descriptor["interval"] == 1
                            else "FixedInterval"
                        ),
                        "config": descriptor["interval"],
                        "refresh": int(descriptor["refresh"]),
                        "stage_cache_age": descriptor["cache_age"],
                        "logit_relative_l2": float(
                            metrics["logit_relative_l2"][index]
                        ),
                        "kl": float(metrics["global_kl"][index]),
                        "pixel_disagreement": float(
                            metrics["global_pixel_disagreement"][index]
                        ),
                    }
                )
            del outputs, features, full_logits, batch_features, variants
        frame_count = len(video["pixel_values"])
        for interval in INTERVALS:
            actual = refresh_counts[interval] / frame_count
            expected = sum(
                index % interval == 0 for index in range(frame_count)
            ) / frame_count
            if actual != expected:
                raise RuntimeError(
                    f"FixedInterval-{interval} refresh mismatch: "
                    f"{actual} vs {expected}"
                )
    return rows


def summarize_quality(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (row["split"], int(row["config"])),
            [],
        ).append(row)
    summary = []
    for (split, interval), values in sorted(groups.items()):
        summary.append(
            {
                "split": split,
                "method": "Full" if interval == 1 else "FixedInterval",
                "config": interval,
                "actual_refresh_rate": float(
                    np.mean([value["refresh"] for value in values])
                ),
                "logit_relative_l2": float(
                    np.mean(
                        [value["logit_relative_l2"] for value in values]
                    )
                ),
                "kl": float(np.mean([value["kl"] for value in values])),
                "pixel_disagreement": float(
                    np.mean(
                        [value["pixel_disagreement"] for value in values]
                    )
                ),
                "frames": len(values),
            }
        )
    return summary


def event_wall(fn) -> tuple[float, float, dict]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    stats = fn()
    end.record()
    end.synchronize()
    return (
        float(start.elapsed_time(end)),
        (time.perf_counter() - wall_start) * 1000.0,
        stats,
    )


def latency_pass(
    model,
    videos: list[dict],
    warmup: int,
    measurements: int,
) -> list[dict]:
    rows = []
    for interval in INTERVALS:
        policy = FixedIntervalPolicy(interval)
        pipeline = StructuredReusePipeline(model)

        def sequence_frames(count: int):
            emitted = 0
            while emitted < count:
                for video in videos:
                    state = None
                    for local_index, pixel_values in enumerate(
                        video["pixel_values"]
                    ):
                        if emitted >= count:
                            return
                        yield local_index, pixel_values, state
                        emitted += 1
                        state = yield

        def run_count(count: int, record: bool) -> list[dict]:
            values = []
            emitted = 0
            while emitted < count:
                for video in videos:
                    state = None
                    for local_index, pixel_values in enumerate(
                        video["pixel_values"]
                    ):
                        if emitted >= count:
                            return values
                        refresh = policy.decide(
                            local_index,
                            0 if state is None else state.age,
                        )

                        def run():
                            nonlocal state
                            with torch.inference_mode():
                                _, state, frame_stats = pipeline.forward(
                                    pixel_values,
                                    refresh,
                                    state,
                                )
                            return frame_stats

                        cuda_ms, wall_ms, frame_stats = event_wall(run)
                        if frame_stats["stage4_calls_delta"] != int(
                            frame_stats["actual_refresh"]
                        ):
                            raise RuntimeError(
                                "Stage-4 call counter violated refresh decision"
                            )
                        if record:
                            values.append(
                                {
                                    "interval": interval,
                                    "frame": emitted,
                                    "local_frame": local_index,
                                    "refresh": int(
                                        frame_stats["actual_refresh"]
                                    ),
                                    "cache_age": frame_stats["cache_age"],
                                    "cuda_ms": cuda_ms,
                                    "wall_ms": wall_ms,
                                }
                            )
                        emitted += 1
            return values

        run_count(warmup, record=False)
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        frame_rows = run_count(measurements, record=True)
        peak_mb = (
            torch.cuda.max_memory_allocated() - start_allocated
        ) / (1024**2)
        overall = [row["wall_ms"] for row in frame_rows]
        refresh_values = [
            row["wall_ms"] for row in frame_rows if row["refresh"]
        ]
        reuse_values = [
            row["wall_ms"] for row in frame_rows if not row["refresh"]
        ]
        rows.append(
            {
                "method": "Full" if interval == 1 else "FixedInterval",
                "config": interval,
                "actual_refresh_rate": float(
                    np.mean([row["refresh"] for row in frame_rows])
                ),
                **distribution(overall),
                **distribution(refresh_values, "refresh"),
                **distribution(reuse_values, "reuse"),
                "fps": 1000.0 / float(np.mean(overall)),
                "stage4_forward_calls": int(
                    sum(row["refresh"] for row in frame_rows)
                ),
                "persistent_cache_mb": 0.25,
                "peak_temporary_mb": float(peak_mb),
                "measurements": len(frame_rows),
            }
        )
        print(rows[-1], flush=True)
    full_mean = next(
        row["mean_ms"] for row in rows if int(row["config"]) == 1
    )
    for row in rows:
        row["speedup"] = full_mean / row["mean_ms"]
    return rows


def write_experiment_metadata(
    root: Path,
    args: argparse.Namespace,
    summary: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        f"max_frames: {args.max_frames}\nwarmup: {args.warmup}\n"
        f"measurements: {args.measurements}\n"
        "intervals: [1, 2, 3, 4, 6, 8]\n"
        "input: cuda_resident_fp32_pixel_values\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (root / "runtime.txt").write_text(
        "Completed fixed-interval quality and sequential latency closure.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 200 or args.measurements < 1000:
        raise ValueError("Formal fixed-interval profile requires 200/1000")
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
    val_videos = prepare_split(
        records, "val", processor, device, args.max_frames
    )
    test_videos = prepare_split(
        records, "test", processor, device, args.max_frames
    )
    if len(val_videos) != 3 or len(test_videos) != 3:
        raise RuntimeError("Round 6 requires frozen 3 validation and 3 test videos")

    quality_rows = quality_pass(model, val_videos, "val")
    quality_rows.extend(quality_pass(model, test_videos, "test"))
    quality_summary = summarize_quality(quality_rows)
    latency_rows = latency_pass(
        model,
        test_videos,
        warmup=args.warmup,
        measurements=args.measurements,
    )

    fixed_root = PROJECT_ROOT / "results/R6_E01_fixed_interval"
    age_root = PROJECT_ROOT / "results/R6_E02_stage_age"
    latency_root = PROJECT_ROOT / "results/R6_E05_sequential_latency"
    quality_root = PROJECT_ROOT / "results/R6_E06_task_quality"
    for root in (fixed_root, age_root, latency_root, quality_root):
        root.mkdir(parents=True, exist_ok=True)

    write_csv(fixed_root / "metrics.csv", quality_summary)
    write_csv(quality_root / "fixed_interval_frame_metrics.csv", quality_rows)
    write_csv(latency_root / "fixed_interval_latency.csv", latency_rows)
    age_equivalence = [
        {
            "stage_age_threshold": interval - 1,
            "fixed_interval": interval,
            "mathematically_equivalent": True,
        }
        for interval in INTERVALS
        if interval > 1
    ]
    write_csv(age_root / "metrics.csv", age_equivalence)

    combined = []
    latency_by_config = {
        int(row["config"]): row for row in latency_rows
    }
    for row in quality_summary:
        if row["split"] != "test":
            continue
        latency = latency_by_config[int(row["config"])]
        combined.append({**row, **latency})
    full = next(row for row in combined if int(row["config"]) == 1)
    candidates = [row for row in combined if int(row["config"]) != 1]
    viable = [
        row
        for row in candidates
        if row["speedup"] >= 1.05
    ]
    best_speed = max(candidates, key=lambda row: row["speedup"])
    summary = {
        "validation_videos": [video["video_id"] for video in val_videos],
        "test_videos": [video["video_id"] for video in test_videos],
        "quality": quality_summary,
        "latency": latency_rows,
        "combined_test": combined,
        "full": full,
        "best_speed_fixed": best_speed,
        "fixed_points_speedup_ge_1_05": len(viable),
        "stage_age_equivalence": "H is exactly equivalent to interval H+1",
        "early_stop_a": {
            "triggered": len(viable) == 0,
            "reason": (
                "No Fixed Interval point achieved 1.05x measured speedup."
                if len(viable) == 0
                else "Fixed Interval has measured points above 1.05x."
            ),
        },
    }
    write_csv(fixed_root / "structured_reuse_main_table.csv", combined)
    write_experiment_metadata(fixed_root, args, summary)
    write_experiment_metadata(
        age_root,
        args,
        {
            "equivalence": age_equivalence,
            "decision": "Stage Age merged with Fixed Interval",
        },
    )
    write_experiment_metadata(
        latency_root,
        args,
        {"latency": latency_rows},
    )
    write_experiment_metadata(
        quality_root,
        args,
        {"quality": quality_summary},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
