"""Measure a controlled VSPW-124 validation pass for the B1 resume budget."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import (  # noqa: E402
    NUM_CLASSES,
    StreamingSegmentationMetrics,
    VSPWFrameDataset,
    build_vspw_segformer,
    load_vspw_class_names,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("validation timing requires CUDA")
    class_names = load_vspw_class_names(args.dataset_root)
    model, _ = build_vspw_segformer(args.source, class_names)
    model = model.to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    with torch.inference_mode():
        warmup_batch = next(iter(loader))
        warmup_images = warmup_batch["pixel_values"].to("cuda", non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32"):
            _ = model(pixel_values=warmup_images).logits
        torch.cuda.synchronize()

        metric = StreamingSegmentationMetrics()
        batches = 0
        frames = 0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for batch in loader:
            images = batch["pixel_values"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32"):
                logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            metric.update(logits.argmax(dim=1), labels)
            batches += 1
            frames += int(images.shape[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    result = metric.compute()
    result.pop("confusion")
    payload = {
        "architecture": "SegFormer-B1",
        "source": str(args.source),
        "dataset_split": "val",
        "frames": frames,
        "batches": batches,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "precision": args.precision,
        "elapsed_seconds": elapsed,
        "frames_per_second": frames / elapsed,
        "milliseconds_per_frame": 1000.0 * elapsed / frames,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "miou": result["miou"],
        "pixel_accuracy": result["pixel_accuracy"],
        "num_classes": NUM_CLASSES,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
