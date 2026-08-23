"""Measure temporal redundancy of SegFormer stage features on a video stream."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.temporal_similarity import analyze_feature_stream  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed, start_run  # noqa: E402
from driftguard.utils.video import get_video_info, iter_video_frames  # noqa: E402


PAIR_FIELDS = [
    "video_id",
    "current_frame",
    "reference_frame",
    "offset",
    "stage",
    "global_cosine",
    "relative_l2",
    "block_grid",
]
BLOCK_FIELDS = [
    "video_id",
    "current_frame",
    "reference_frame",
    "offset",
    "stage",
    "block_row",
    "block_col",
    "cosine",
    "relative_l2",
]
SUMMARY_FIELDS = [
    "stage",
    "offset",
    "num_pairs",
    "cosine_mean",
    "cosine_std",
    "cosine_median",
    "cosine_p25",
    "cosine_p75",
    "l2_mean",
    "l2_std",
    "l2_median",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--block-grid", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-root",
        default="results/E01_temporal_redundancy",
    )
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, logger = start_run("analyze_temporal_redundancy", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    offsets = sorted({int(offset) for offset in args.offsets})
    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("--offsets must contain positive integers.")
    if args.block_grid <= 0:
        raise ValueError("--block-grid must be positive.")

    video_info = get_video_info(video_path)
    video_id = args.video_id or video_path.stem
    processor, model = load_segformer(args.model, device=device)
    extraction_times: list[float] = []
    frame_iterator = iter_video_frames(video_path, max_frames=args.max_frames)
    total_start = time.perf_counter()

    def feature_stream():
        from PIL import Image

        for frame_id, frame_rgb in frame_iterator:
            image = Image.fromarray(frame_rgb, mode="RGB")
            inputs = prepare_inputs(processor, image, device)
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=True,
                    return_dict=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            extraction_times.append((time.perf_counter() - start) * 1000.0)
            features = [
                feature.detach()
                for feature in get_stage_features(outputs, model=model)
            ]
            logger.info("Extracted frame=%d", frame_id)
            yield frame_id, features

    analysis = analyze_feature_stream(
        feature_frames=feature_stream(),
        offsets=offsets,
        block_grid=args.block_grid,
        video_id=video_id,
    )
    total_time = time.perf_counter() - total_start
    output_root = PROJECT_ROOT / args.output_root
    _write_csv(output_root / "frame_pair_metrics.csv", PAIR_FIELDS, analysis["pair_rows"])
    _write_csv(output_root / "block_metrics.csv", BLOCK_FIELDS, analysis["block_rows"])
    _write_csv(output_root / "temporal_redundancy_summary.csv", SUMMARY_FIELDS, analysis["summary_rows"])

    metrics = {
        "video": str(video_path.resolve()),
        "video_id": video_id,
        "frames_requested": args.max_frames,
        "frames_processed": analysis["processed_frames"],
        "resolution": f"{video_info.width}x{video_info.height}",
        "source_fps": video_info.fps,
        "model": args.model,
        "device": str(device),
        "offsets": offsets,
        "block_grid": args.block_grid,
        "total_analysis_time_sec": total_time,
        "per_frame_feature_extraction_time_ms": (
            sum(extraction_times) / len(extraction_times) if extraction_times else 0.0
        ),
        "buffer_max_size": analysis["buffer_max_size"],
        "buffer_peak_size": analysis["buffer_peak_size"],
        "pair_count": len(analysis["pair_rows"]),
        "block_metric_count": len(analysis["block_rows"]),
    }
    import json

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"video={video_path.resolve()}")
    print(f"video_id={video_id}")
    print(f"frames_processed={analysis['processed_frames']}")
    print(f"resolution={video_info.width}x{video_info.height}")
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"offsets={offsets}")
    print(f"block_grid={args.block_grid}")
    print(f"pair_count={len(analysis['pair_rows'])}")
    print(f"block_metric_count={len(analysis['block_rows'])}")
    print(f"total_analysis_time_sec={total_time:.3f}")
    print(f"per_frame_feature_extraction_time_ms={metrics['per_frame_feature_extraction_time_ms']:.3f}")
    print(f"rolling_buffer_peak={analysis['buffer_peak_size']}/{analysis['buffer_max_size']}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
