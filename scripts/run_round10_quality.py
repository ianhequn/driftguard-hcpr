"""Run one Round-10 VSPW quality configuration in chronological order."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftguard.structured_reuse.boundaries import StructuredBoundaryReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import (  # noqa: E402
    NUM_CLASSES,
    StreamingSegmentationMetrics,
    VSPWFrameDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", choices=("full", "s4", "s3_s4", "s2_s4"), default="full")
    parser.add_argument("--interval", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-frames", type=int)
    return parser.parse_args()


def limit_dataset(dataset, limit_videos, limit_frames):
    if limit_videos is None and limit_frames is None:
        return dataset
    selected = []
    order = []
    counts = defaultdict(int)
    for sample in dataset.samples:
        if sample.video_id not in order:
            order.append(sample.video_id)
        if limit_videos is not None and order.index(sample.video_id) >= limit_videos:
            continue
        if limit_frames is not None and counts[sample.video_id] >= limit_frames:
            continue
        selected.append(sample)
        counts[sample.video_id] += 1
    dataset.samples = selected
    return dataset


def update(metric: StreamingSegmentationMetrics, logits: torch.Tensor, labels: torch.Tensor) -> None:
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.argmax(dim=1).to(torch.int64)
    labels = labels.to(torch.int64)
    valid = labels == 255
    if bool(((labels != 255) & ((labels < 0) | (labels >= NUM_CLASSES))).any().item()):
        raise ValueError("canonical VSPW labels contain an out-of-range ID")
    encoded = labels[~valid] * NUM_CLASSES + prediction[~valid]
    if encoded.numel():
        counts = torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
        metric.confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


class SegmentSampler:
    def __init__(self, dataset, size: int):
        self.segments = []
        start = 0
        while start < len(dataset.samples):
            video_id = dataset.samples[start].video_id
            end = start
            while end < len(dataset.samples) and dataset.samples[end].video_id == video_id and end < start + size:
                end += 1
            self.segments.append(list(range(start, end)))
            start = end

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Round 10 quality requires CUDA")
    if args.boundary != "full" and args.interval <= 0:
        raise ValueError("interval must be positive")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    if int(model.config.num_labels) != NUM_CLASSES:
        raise RuntimeError("model is not VSPW-124")
    dataset = limit_dataset(VSPWFrameDataset(args.dataset_root, "val", training=False), args.limit_videos, args.limit_frames)
    batch_size = 8 if args.boundary == "full" else args.interval
    loader = DataLoader(dataset, batch_sampler=SegmentSampler(dataset, batch_size),
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    metric = StreamingSegmentationMetrics()
    positions = defaultdict(int)
    state = None
    pipeline = None if args.boundary == "full" else StructuredBoundaryReusePipeline(model, args.boundary)
    refresh_frames = 0
    with torch.inference_mode():
        for batch in loader:
            video_id = batch["video_id"][0]
            if any(value != video_id for value in batch["video_id"]):
                raise RuntimeError("segment crosses a video boundary")
            positions[video_id] += len(batch["video_id"])
            pixels = batch["pixel_values"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.boundary == "full":
                    logits = model(pixel_values=pixels).logits
                    refresh = True
                else:
                    state = None
                    logits, state, _ = pipeline.forward_segment(pixels, state=state)
                    refresh = True
            update(metric, logits, labels)
            refresh_frames += len(batch["video_id"]) if args.boundary == "full" else 1
    result = metric.compute()
    confusion = result.pop("confusion")
    total_frames = len(dataset)
    if args.boundary == "full":
        counts = {f"stage{i}": total_frames for i in range(1, 5)}
        counts["decoder"] = total_frames
        cache_bytes = 0
    else:
        counts = {"stage1": total_frames, "stage2": total_frames,
                  "stage3": total_frames, "stage4": refresh_frames,
                  "decoder": total_frames}
        if args.boundary == "s3_s4":
            counts["stage3"] = refresh_frames
        elif args.boundary == "s2_s4":
            counts["stage2"] = refresh_frames
            counts["stage3"] = refresh_frames
        cache_bytes = state.persistent_bytes if state is not None else 0
    payload = {
        "model": args.model_dir.name,
        "boundary": args.boundary,
        "interval": 1 if args.boundary == "full" else args.interval,
        "nominal_refresh": 1.0 if args.boundary == "full" else 1.0 / args.interval,
        "actual_refresh": refresh_frames / total_frames if total_frames else float("nan"),
        "videos": len(positions), "frames": total_frames,
        "S1_calls": counts["stage1"], "S2_calls": counts["stage2"],
        "S3_calls": counts["stage3"], "S4_calls": counts["stage4"],
        "decoder_calls": counts["decoder"],
        "mIoU": result["miou"], "pixel_accuracy": result["pixel_accuracy"],
        "mean_class_accuracy": result["mean_class_accuracy"],
        "cache_bytes": cache_bytes, "cache_mib": cache_bytes / (1024 * 1024),
        "included_classes": result["included_classes"],
        "confusion": confusion.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("model", "boundary", "interval", "frames", "mIoU", "actual_refresh", "S4_calls")}, indent=2))


if __name__ == "__main__":
    main()
