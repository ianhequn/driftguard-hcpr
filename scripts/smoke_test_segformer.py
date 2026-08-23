"""Run a single SegFormer semantic segmentation smoke test."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import (  # noqa: E402
    DEFAULT_IMAGE_URL,
    load_image,
    load_segformer,
    mask_to_images,
    prepare_inputs,
    predict_segmentation,
)
from driftguard.utils.common import choose_device, set_seed, start_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="outputs/segmentation/smoke_test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, logger = start_run("smoke_test_segformer", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    logger.info("Device: %s", device)
    image = load_image(args.image, cache_dir=PROJECT_ROOT / "data" / "samples")
    processor, model = load_segformer(args.model, device=device)
    inputs = prepare_inputs(processor, image, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs = model(
        pixel_values=inputs["pixel_values"],
        output_hidden_states=False,
        return_dict=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    _, mask = predict_segmentation(outputs, image.size)
    mask_image, overlay = mask_to_images(
        image,
        mask[0],
        num_labels=int(model.config.num_labels),
    )
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "input.png")
    mask_image.save(output_dir / "mask.png")
    overlay.save(output_dir / "overlay.png")
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    finite = bool(torch.isfinite(outputs.logits).all().item())
    print(f"Input shape: {tuple(inputs['pixel_values'].shape)}")
    print(f"Output logits shape: {tuple(outputs.logits.shape)}")
    print(f"Latency: {latency_ms:.3f} ms")
    print(f"Peak VRAM: {peak_vram_mb:.3f} MB")
    print(f"No NaN / Inf: {finite}")
    print(f"Output directory: {output_dir.resolve()}")
    if device.type == "cuda":
        print("CUDA inference successful")
    else:
        print("CPU inference completed; CUDA unavailable on this machine")
    logger.info("Smoke test finished in %.3f ms", latency_ms)
    (run_dir / "metrics.csv").write_text(
        "latency_ms,peak_vram_mb,logits_shape,finite,device\n"
        f"{latency_ms:.6f},{peak_vram_mb:.6f},\"{tuple(outputs.logits.shape)}\","
        f"{finite},{device}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
