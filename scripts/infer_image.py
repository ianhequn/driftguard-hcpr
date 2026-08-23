"""Convenience command for SegFormer inference on one local or remote image."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="outputs/segmentation/image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, logger = start_run("infer_image", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    image = load_image(args.image, cache_dir=PROJECT_ROOT / "data" / "samples")
    processor, model = load_segformer(args.model, device=device)
    inputs = prepare_inputs(processor, image, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs = model(pixel_values=inputs["pixel_values"], return_dict=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    _, mask = predict_segmentation(outputs, image.size)
    mask_image, overlay = mask_to_images(image, mask[0], int(model.config.num_labels))
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "input.png")
    mask_image.save(output_dir / "mask.png")
    overlay.save(output_dir / "overlay.png")
    logger.info("Image inference device=%s latency_ms=%.3f", device, latency_ms)
    print(f"device={device}")
    print(f"input_shape={tuple(inputs['pixel_values'].shape)}")
    print(f"logits_shape={tuple(outputs.logits.shape)}")
    print(f"latency_ms={latency_ms:.3f}")
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
