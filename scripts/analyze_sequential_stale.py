"""Measure true sequential stale propagation for SegFormer cache cuts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from driftguard.utils.video import get_video_info, iter_video_frames  # noqa: E402


FIELDS = [
    "video_id",
    "current_frame",
    "reference_frame",
    "offset",
    "cut",
    "kl_divergence",
    "pixel_disagreement_ratio",
    "logit_relative_l2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/real_videos")
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="results/E03_sequential_stale")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    video_dir = PROJECT_ROOT / args.video_dir
    videos = sorted(
        path
        for path in video_dir.glob("*")
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    )
    offsets = sorted({int(value) for value in args.offsets})
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}")
    if not offsets or any(value <= 0 for value in offsets):
        raise ValueError("--offsets must contain positive integers")

    processor, model = load_segformer(args.model, device=device)
    rows: list[dict] = []
    start_time = time.perf_counter()

    for video_path in videos:
        info = get_video_info(video_path)
        history: dict[int, list[torch.Tensor]] = {}
        max_offset = max(offsets)
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

                for offset in offsets:
                    reference_frame = frame_id - offset
                    cached_features = history.get(reference_frame)
                    if cached_features is None:
                        continue
                    for cut in range(1, 5):
                        stale_features = sequential_stale_features(
                            model=model,
                            current_features=current_features,
                            cached_features=cached_features,
                            cut=cut,
                        )
                        metrics = stale_feature_metrics(
                            model=model,
                            current_features=current_features,
                            stale_features=stale_features,
                            full_logits=full_logits,
                        )
                        rows.append(
                            {
                                "video_id": video_path.stem,
                                "current_frame": frame_id,
                                "reference_frame": reference_frame,
                                "offset": offset,
                                "cut": cut,
                                **metrics,
                            }
                        )

            history[frame_id] = [feature.detach() for feature in current_features]
            for old_frame in list(history):
                if old_frame < frame_id - max_offset:
                    del history[old_frame]

    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "sequential_metrics.csv", rows)
    summary: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        summary.setdefault((int(row["cut"]), int(row["offset"])), []).append(row)
    summary_rows = []
    for (cut, offset), group in sorted(summary.items()):
        summary_rows.append(
            {
                "cut": cut,
                "offset": offset,
                "num_pairs": len(group),
                "kl_divergence_mean": sum(
                    float(row["kl_divergence"]) for row in group
                )
                / len(group),
                "pixel_disagreement_mean": sum(
                    float(row["pixel_disagreement_ratio"]) for row in group
                )
                / len(group),
                "logit_relative_l2_mean": sum(
                    float(row["logit_relative_l2"]) for row in group
                )
                / len(group),
            }
        )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "cut",
            "offset",
            "num_pairs",
            "kl_divergence_mean",
            "pixel_disagreement_mean",
            "logit_relative_l2_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    runtime = time.perf_counter() - start_time
    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"video_dir: {args.video_dir}",
                f"model: {args.model}",
                f"max_frames: {args.max_frames}",
                f"offsets: {offsets}",
                f"device: {device}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "runtime.txt").write_text(f"{runtime:.6f}\n", encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "video_count": len(videos),
                "frames_per_video": args.max_frames,
                "row_count": len(rows),
                "runtime_sec": runtime,
                "resolution": {
                    path.stem: f"{get_video_info(path).width}x{get_video_info(path).height}"
                    for path in videos
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"videos={len(videos)}")
    print(f"rows={len(rows)}")
    print(f"device={device}")
    print(f"runtime_sec={runtime:.3f}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
