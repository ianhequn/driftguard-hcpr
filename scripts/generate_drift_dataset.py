"""Generate a video-level-split dataset for the tiny DriftGuard estimator."""

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

from driftguard.analysis.cheap_observables import (  # noqa: E402
    block_relative_l2,
    entropy_block_observables,
    rgb_block_observables,
)
from driftguard.analysis.sequential_stale import (  # noqa: E402
    decode_features,
    sequential_stale_features,
)
from driftguard.analysis.stage_sensitivity import stale_feature_metrics  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


FEATURE_COLUMNS = [
    "local_residual",
    "cache_age",
    "previous_entropy",
    "global_residual",
    "local_variance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--cut", type=int, default=None)
    parser.add_argument("--state", default="reports/research_state.json")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--ages", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--block-grid", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="results/E05_drift_dataset")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_cut(args: argparse.Namespace) -> int:
    if args.cut is not None:
        return args.cut
    state = json.loads((PROJECT_ROOT / args.state).read_text(encoding="utf-8"))
    value = state.get("selected_cache_cut")
    if value is None:
        raise ValueError("No selected cache cut in research_state.json")
    return int(value)


def _split_videos(videos: list[Path], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)
    count = len(shuffled)
    test_count = max(1, int(count * 0.15))
    val_count = max(1, int(count * 0.15))
    train_count = count - test_count - val_count
    if train_count < 1:
        raise ValueError("Need at least three videos for train/val/test split")
    return {
        path.stem: (
            "train"
            if index < train_count
            else "val"
            if index < train_count + val_count
            else "test"
        )
        for index, path in enumerate(shuffled)
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cut = _read_cut(args)
    if cut < 1 or cut > 4:
        raise ValueError("--cut must be between 1 and 4")
    ages = sorted({int(value) for value in args.ages})
    if not ages or any(value <= 0 for value in ages):
        raise ValueError("--ages must contain positive integers")
    device = choose_device(args.device)
    video_dir = PROJECT_ROOT / args.video_dir
    videos = sorted(
        path
        for path in video_dir.glob("*")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    if len(videos) < 3:
        raise ValueError("Need at least three valid videos for video-level split")
    split_by_video = _split_videos(videos, args.seed)
    processor, model = load_segformer(args.model, device=device)
    samples: list[dict] = []
    start_time = time.perf_counter()

    for video_path in videos:
        history: dict[int, list[torch.Tensor]] = {}
        frame_history: dict[int, np.ndarray] = {}
        logits_history: dict[int, torch.Tensor] = {}
        max_age = max(ages)
        previous_frame: np.ndarray | None = None
        for frame_id, frame_rgb in iter_video_frames(
            video_path,
            max_frames=args.max_frames,
        ):
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                current_features = [
                    feature.detach()
                    for feature in get_stage_features(outputs, model=model)
                ]
                full_logits = decode_features(model, current_features)
                for age in ages:
                    reference_frame = frame_id - age
                    cached = history.get(reference_frame)
                    reference_rgb = frame_history.get(reference_frame)
                    if cached is None or reference_rgb is None:
                        continue
                    cheap = rgb_block_observables(
                        frame_rgb,
                        reference_rgb,
                        grid=args.block_grid,
                    )
                    entropy = entropy_block_observables(
                        logits_history.get(reference_frame),
                        image_height=frame_rgb.shape[0],
                        image_width=frame_rgb.shape[1],
                        grid=args.block_grid,
                    )
                    labels = block_relative_l2(
                        current_features[cut - 1],
                        cached[cut - 1],
                        grid=args.block_grid,
                    )
                    stale = sequential_stale_features(
                        model=model,
                        current_features=current_features,
                        cached_features=cached,
                        cut=cut,
                    )
                    task = stale_feature_metrics(
                        model=model,
                        current_features=current_features,
                        stale_features=stale,
                        full_logits=full_logits,
                    )
                    for index, label in enumerate(labels):
                        samples.append(
                            {
                                "video_id": video_path.stem,
                                "split": split_by_video[video_path.stem],
                                "frame_id": frame_id,
                                "reference_frame": reference_frame,
                                "block_index": index,
                                "block_row": index // args.block_grid,
                                "block_col": index % args.block_grid,
                                "feature_drift": float(label),
                                "task_damage": float(task["logit_relative_l2"]),
                                "features": [
                                    float(cheap[index, 0]),
                                    float(age),
                                    float(entropy[index]),
                                    float(cheap[index, 2]),
                                    float(cheap[index, 3]),
                                ],
                            }
                        )
            history[frame_id] = [feature.detach() for feature in current_features]
            frame_history[frame_id] = frame_rgb.copy()
            logits_history[frame_id] = full_logits.detach()
            for old_frame in list(history):
                if old_frame < frame_id - max_age:
                    del history[old_frame]
                    frame_history.pop(old_frame, None)
                    logits_history.pop(old_frame, None)

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for split in ("train", "val", "test"):
        split_samples = [sample for sample in samples if sample["split"] == split]
        x = np.asarray([sample["features"] for sample in split_samples], dtype=np.float32)
        y = np.asarray([sample["feature_drift"] for sample in split_samples], dtype=np.float32)
        task = np.asarray([sample["task_damage"] for sample in split_samples], dtype=np.float32)
        np.savez_compressed(
            output_root / f"{split}.npz",
            X=x,
            y=y,
            task_damage=task,
            video_id=np.asarray([sample["video_id"] for sample in split_samples]),
            frame_id=np.asarray([sample["frame_id"] for sample in split_samples]),
            reference_frame=np.asarray(
                [sample["reference_frame"] for sample in split_samples]
            ),
            block_index=np.asarray([sample["block_index"] for sample in split_samples]),
            feature_columns=np.asarray(FEATURE_COLUMNS),
        )
        summary_rows.append(
            {
                "split": split,
                "video_count": len(
                    {sample["video_id"] for sample in split_samples}
                ),
                "sample_count": len(split_samples),
                "mean_feature_drift": float(y.mean()) if len(y) else 0.0,
                "p90_feature_drift": float(np.quantile(y, 0.9)) if len(y) else 0.0,
                "videos": ",".join(
                    sorted({sample["video_id"] for sample in split_samples})
                ),
            }
        )
    _write_csv(
        output_root / "dataset_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    (output_root / "video_split.json").write_text(
        json.dumps(split_by_video, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = time.perf_counter() - start_time
    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"video_dir: {args.video_dir}",
                f"model: {args.model}",
                f"cut: {cut}",
                f"max_frames: {args.max_frames}",
                f"ages: {ages}",
                f"block_grid: {args.block_grid}",
                f"device: {device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "runtime.txt").write_text(f"{runtime:.6f}\n", encoding="utf-8")
    print(f"cut={cut}")
    print(f"videos={len(videos)}")
    print(f"samples={len(samples)}")
    print(f"runtime_sec={runtime:.3f}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
