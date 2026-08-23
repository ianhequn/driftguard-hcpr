"""Analyze SegFormer temporal redundancy across a small real-video collection."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
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

from driftguard.analysis.temporal_similarity import (  # noqa: E402
    analyze_feature_stream,
    cosine_similarity,
    relative_l2_distance,
)
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import get_video_info, iter_video_frames  # noqa: E402


PER_VIDEO_FIELDS = [
    "video_id",
    "motion_mean",
    "motion_p90",
    "motion_group",
    "frames",
    "resolution",
    "stage",
    "offset",
    "block_grid",
    "cosine_mean",
    "relative_l2_mean",
    "num_pairs",
    "random_cosine_mean",
    "random_relative_l2_mean",
]
SUMMARY_FIELDS = [
    "stage",
    "offset",
    "block_grid",
    "video_count",
    "pair_count",
    "cosine_mean",
    "cosine_std",
    "relative_l2_mean",
    "relative_l2_std",
    "random_cosine_mean",
    "random_relative_l2_mean",
]
MOTION_FIELDS = [
    "motion_group",
    "stage",
    "offset",
    "block_grid",
    "video_count",
    "cosine_mean",
    "relative_l2_mean",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _random_baseline(
    features: list[list[torch.Tensor]],
    max_offset: int,
    seed: int,
) -> dict[int, tuple[float, float]]:
    rng = random.Random(seed)
    frame_count = len(features)
    min_gap = max(max_offset + 1, frame_count // 3)
    pairs = [
        (left, right)
        for left in range(frame_count)
        for right in range(left + min_gap, frame_count)
    ]
    if len(pairs) > 32:
        pairs = rng.sample(pairs, 32)
    result: dict[int, tuple[float, float]] = {}
    for stage in range(4):
        cosine_values = [
            cosine_similarity(features[left][stage], features[right][stage])
            for left, right in pairs
        ]
        l2_values = [
            relative_l2_distance(features[left][stage], features[right][stage])
            for left, right in pairs
        ]
        result[stage + 1] = (
            float(np.mean(cosine_values)) if cosine_values else 0.0,
            float(np.mean(l2_values)) if l2_values else 0.0,
        )
    return result


def _motion_group(values: list[float], value: float) -> str:
    if not values:
        return "unknown"
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def _aggregate(rows: list[dict], fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        cosine = [float(row["cosine_mean"]) for row in group]
        l2 = [float(row["relative_l2_mean"]) for row in group]
        random_cosine = [float(row["random_cosine_mean"]) for row in group]
        random_l2 = [float(row["random_relative_l2_mean"]) for row in group]
        output.append(
            {
                **dict(zip(fields, key)),
                "video_count": len({row["video_id"] for row in group}),
                "pair_count": sum(int(row["num_pairs"]) for row in group),
                "cosine_mean": float(np.mean(cosine)),
                "cosine_std": float(np.std(cosine)),
                "relative_l2_mean": float(np.mean(l2)),
                "relative_l2_std": float(np.std(l2)),
                "random_cosine_mean": float(np.mean(random_cosine)),
                "random_relative_l2_mean": float(np.mean(random_l2)),
            }
        )
    return output


def _plot_summary(summary: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    global_rows = [row for row in summary if int(row["block_grid"]) == 4]
    for metric, ylabel, filename in (
        ("cosine_mean", "Global cosine similarity", "cosine_vs_offset_real.png"),
        ("relative_l2_mean", "Global relative L2", "relative_l2_vs_offset_real.png"),
    ):
        plt.figure(figsize=(8, 5))
        for stage in range(1, 5):
            rows = sorted(
                [row for row in global_rows if int(row["stage"]) == stage],
                key=lambda row: int(row["offset"]),
            )
            if not rows:
                continue
            x = [int(row["offset"]) for row in rows]
            y = [float(row[metric]) for row in rows]
            plt.plot(x, y, marker="o", label=f"Stage {stage}")
        plt.xlabel("Temporal offset")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=160)
        plt.close()


def _plot_block_heatmap(block_rows: list[dict], figures_dir: Path) -> None:
    selected = [
        row
        for row in block_rows
        if int(row["stage"]) == 1
        and int(row["offset"]) == 1
        and int(row["block_grid"]) == 8
    ]
    if not selected:
        return
    max_row = max(int(row["block_row"]) for row in selected)
    max_col = max(int(row["block_col"]) for row in selected)
    values = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)
    for row in selected:
        values[int(row["block_row"]), int(row["block_col"])] = float(row["cosine"])
    plt.figure(figsize=(6, 5))
    plt.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(label="Block cosine similarity")
    plt.title("Stage 1, offset 1, 8x8 block mean")
    plt.xlabel("Block column")
    plt.ylabel("Block row")
    plt.tight_layout()
    plt.savefig(figures_dir / "block_similarity_examples.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--block-grids", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="results/E02_real_video_redundancy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    video_dir = PROJECT_ROOT / args.video_dir
    output_root = PROJECT_ROOT / args.output_root
    figures_dir = PROJECT_ROOT / "figures" / "E02_real_video_redundancy"
    videos = sorted(video_dir.glob("*"))
    videos = [path for path in videos if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    offsets = sorted({int(value) for value in args.offsets})
    grids = sorted({int(value) for value in args.block_grids})
    if not offsets or any(value <= 0 for value in offsets):
        raise ValueError("--offsets must contain positive integers")
    if not grids or any(value <= 0 for value in grids):
        raise ValueError("--block-grids must contain positive integers")

    processor, model = load_segformer(args.model, device=device)
    video_records: list[dict] = []
    all_summary_rows: list[dict] = []
    all_block_rows: list[dict] = []
    start_time = time.perf_counter()

    for video_path in videos:
        info = get_video_info(video_path)
        features: list[list[torch.Tensor]] = []
        motion_values: list[float] = []
        previous_frame: np.ndarray | None = None
        for _, frame_rgb in iter_video_frames(video_path, max_frames=args.max_frames):
            if previous_frame is not None:
                motion_values.append(
                    float(
                        np.abs(
                            frame_rgb.astype(np.float32)
                            - previous_frame.astype(np.float32)
                        ).mean()
                        / 255.0
                    )
                )
            previous_frame = frame_rgb
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.inference_mode():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            features.append(
                [
                    feature.detach().cpu()
                    for feature in get_stage_features(outputs, model=model)
                ]
            )
        if len(features) < max(offsets) + 1:
            raise RuntimeError(f"{video_path.name} has too few decoded frames")

        motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
        motion_p90 = float(np.quantile(motion_values, 0.9)) if motion_values else 0.0
        video_records.append(
            {
                "video_id": video_path.stem,
                "motion_mean": motion_mean,
                "motion_p90": motion_p90,
                "frames": len(features),
                "resolution": f"{info.width}x{info.height}",
            }
        )
        random_metrics = _random_baseline(features, max(offsets), args.seed)
        for grid in grids:
            analysis = analyze_feature_stream(
                ((index, value) for index, value in enumerate(features)),
                offsets=offsets,
                block_grid=grid,
                video_id=video_path.stem,
            )
            all_block_rows.extend(
                {
                    **row,
                    "block_grid": grid,
                }
                for row in analysis["block_rows"]
            )
            for row in analysis["summary_rows"]:
                stage = int(row["stage"])
                random_cosine, random_l2 = random_metrics[stage]
                all_summary_rows.append(
                    {
                        "video_id": video_path.stem,
                        "motion_mean": motion_mean,
                        "motion_p90": motion_p90,
                        "frames": len(features),
                        "resolution": f"{info.width}x{info.height}",
                        "stage": stage,
                        "offset": int(row["offset"]),
                        "block_grid": grid,
                        "cosine_mean": float(row["cosine_mean"]),
                        "relative_l2_mean": float(row["l2_mean"]),
                        "num_pairs": int(row["num_pairs"]),
                        "random_cosine_mean": random_cosine,
                        "random_relative_l2_mean": random_l2,
                    }
                )

    motion_values = [row["motion_mean"] for row in video_records]
    for row in video_records:
        row["motion_group"] = _motion_group(motion_values, row["motion_mean"])
    per_video_rows = [
        {
            **row,
            "motion_group": next(
                item["motion_group"]
                for item in video_records
                if item["video_id"] == row["video_id"]
            ),
            "stage": summary["stage"],
            "offset": summary["offset"],
            "block_grid": summary["block_grid"],
            "cosine_mean": summary["cosine_mean"],
            "relative_l2_mean": summary["relative_l2_mean"],
            "num_pairs": summary["num_pairs"],
            "random_cosine_mean": summary["random_cosine_mean"],
            "random_relative_l2_mean": summary["random_relative_l2_mean"],
        }
        for summary in all_summary_rows
        for row in video_records
        if row["video_id"] == summary["video_id"]
    ]
    summary_rows = _aggregate(
        all_summary_rows,
        ["stage", "offset", "block_grid"],
    )
    motion_rows = _aggregate(
        [
            {**row, "motion_group": next(
                item["motion_group"]
                for item in video_records
                if item["video_id"] == row["video_id"]
            )}
            for row in all_summary_rows
        ],
        ["motion_group", "stage", "offset", "block_grid"],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "per_video_metrics.csv", PER_VIDEO_FIELDS, per_video_rows)
    _write_csv(output_root / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(output_root / "motion_group_summary.csv", MOTION_FIELDS, motion_rows)
    _write_csv(
        output_root / "block_metrics.csv",
        [
            "video_id",
            "current_frame",
            "reference_frame",
            "offset",
            "stage",
            "block_grid",
            "block_row",
            "block_col",
            "cosine",
            "relative_l2",
        ],
        all_block_rows,
    )
    _plot_summary(summary_rows, figures_dir)
    _plot_block_heatmap(all_block_rows, figures_dir)
    runtime = time.perf_counter() - start_time
    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"video_dir: {args.video_dir}",
                f"model: {args.model}",
                f"max_frames: {args.max_frames}",
                f"offsets: {offsets}",
                f"block_grids: {grids}",
                f"device: {device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime.txt").write_text(f"{runtime:.6f}\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "videos": [row["video_id"] for row in video_records],
                "video_count": len(video_records),
                "frames_per_video": args.max_frames,
                "device": str(device),
                "runtime_sec": runtime,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"videos={len(video_records)}")
    print(f"frames_per_video={args.max_frames}")
    print(f"device={device}")
    print(f"runtime_sec={runtime:.3f}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
