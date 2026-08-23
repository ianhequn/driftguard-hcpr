"""Run frame-by-frame SegFormer inference and save masks/overlays."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import (  # noqa: E402
    load_segformer,
    mask_to_images,
    prepare_inputs,
    predict_segmentation,
)
from driftguard.utils.common import choose_device, set_seed, start_run  # noqa: E402
from driftguard.utils.video import get_video_info, iter_video_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/segmentation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, logger = start_run("infer_video", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    info = get_video_info(video_path)
    output_dir = PROJECT_ROOT / args.output_root / video_path.stem
    original_dir = output_dir / "original"
    mask_dir = output_dir / "mask"
    overlay_dir = output_dir / "overlay"
    for directory in (original_dir, mask_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processor, model = load_segformer(args.model, device=device)
    rows = []
    for frame_id, frame_rgb in iter_video_frames(video_path, max_frames=args.max_frames):
        image = Image.fromarray(frame_rgb, mode="RGB")
        inputs = prepare_inputs(processor, image, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(pixel_values=inputs["pixel_values"], return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        _, mask = predict_segmentation(outputs, image.size)
        mask_image, overlay = mask_to_images(
            image,
            mask[0],
            int(model.config.num_labels),
        )
        stem = f"frame_{frame_id:06d}"
        cv2.imwrite(
            str(original_dir / f"{stem}.jpg"),
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        )
        mask_image.save(mask_dir / f"{stem}.png")
        overlay.save(overlay_dir / f"{stem}.jpg")
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        rows.append(
            {
                "frame_id": frame_id,
                "latency_ms": latency_ms,
                "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
                "peak_vram_mb": peak_vram_mb,
            }
        )
        if frame_id == 0 or frame_id % 10 == 0:
            logger.info("frame=%d latency_ms=%.3f", frame_id, latency_ms)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_id", "latency_ms", "fps", "peak_vram_mb"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"video={video_path.resolve()}")
    print(f"size={info.width}x{info.height} source_fps={info.fps:.3f}")
    print(f"frames_processed={len(rows)}")
    print(f"device={device}")
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
