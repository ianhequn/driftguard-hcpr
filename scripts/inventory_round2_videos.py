"""Build a real-video inventory and deterministic video-level split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.utils.video import get_video_info, iter_video_frames  # noqa: E402


def _motion_group(values: list[float], value: float) -> str:
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--output-root", default="results/R2_E00_video_dataset")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = PROJECT_ROOT / args.video_dir
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        and not path.name.endswith(".partial")
    )
    if len(videos) < 20:
        raise RuntimeError(f"Round 2 requires at least 20 complete videos, found {len(videos)}")

    rows = []
    for path in videos:
        info = get_video_info(path)
        frames = list(iter_video_frames(path, max_frames=args.frames))
        if len(frames) < 32:
            raise RuntimeError(f"{path.name} has only {len(frames)} decodable frames")
        residuals = []
        for (_, current), (_, previous) in zip(frames[1:], frames[:-1]):
            residuals.append(
                float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0)
            )
        rows.append(
            {
                "video_id": path.stem,
                "source": "intel-iot-devkit/sample-videos",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "frame_count": len(frames),
                "reported_frame_count": info.frame_count,
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "mean_global_residual": float(np.mean(residuals)),
                "motion_p90": float(np.quantile(residuals, 0.9)),
            }
        )

    motion_values = [row["mean_global_residual"] for row in rows]
    for row in rows:
        row["motion_level"] = _motion_group(motion_values, row["mean_global_residual"])

    test_count = max(3, round(len(rows) * 0.15))
    val_count = max(3, round(len(rows) * 0.15))
    train_count = len(rows) - test_count - val_count
    if train_count < 1:
        raise RuntimeError("Video split leaves no training videos")

    # Stratify the held-out videos by the motion groups computed from all
    # videos, while keeping the split video-level and deterministic.
    rng = np.random.default_rng(args.seed)
    grouped = {group: [] for group in ("low", "medium", "high")}
    for row in rows:
        grouped[row["motion_level"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    test_rows = [grouped[group].pop() for group in ("low", "medium", "high")]
    val_rows = []
    for group in ("low", "medium", "high"):
        if grouped[group] and len(val_rows) < val_count:
            val_rows.append(grouped[group].pop())
    remaining = [row for values in grouped.values() for row in values]
    rng.shuffle(remaining)
    test_rows.extend(remaining[: max(0, test_count - len(test_rows))])
    remaining = remaining[max(0, test_count - 3):]
    val_rows.extend(remaining[: max(0, val_count - len(val_rows))])
    val_ids = {row["video_id"] for row in val_rows}
    test_ids = {row["video_id"] for row in test_rows}
    for row in rows:
        row["split"] = "test" if row["video_id"] in test_ids else "val" if row["video_id"] in val_ids else "train"

    actual_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    train_count, val_count, test_count = (actual_counts["train"], actual_counts["val"], actual_counts["test"])
    test_groups = {row["motion_level"] for row in rows if row["split"] == "test"}
    if test_groups != {"low", "medium", "high"}:
        raise RuntimeError(f"Test split must contain low/medium/high motion, got {sorted(test_groups)}")

    fields = list(rows[0])
    _write_csv(output_root / "video_inventory.csv", rows)
    (output_root / "video_split.json").write_text(
        json.dumps({row["video_id"]: row["split"] for row in rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "config.yaml").write_text(
        f"video_dir: {args.video_dir}\nframes: {args.frames}\nseed: {args.seed}\n"
        f"video_count: {len(rows)}\ntrain: {train_count}\nval: {val_count}\ntest: {test_count}\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "videos": len(rows),
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "frames_per_video": args.frames,
                "motion_distribution": {
                    group: sum(row["motion_level"] == group for row in rows)
                    for group in ("low", "medium", "high")
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(rows), "train": train_count, "val": val_count, "test": test_count}))


if __name__ == "__main__":
    main()
