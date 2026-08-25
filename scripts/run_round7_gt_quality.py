"""Run the Round 7 VSPW ground-truth Full/FI quality benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import (  # noqa: E402
    Stage4CacheState,
    StructuredReusePipeline,
)
from driftguard.training.vspw_segformer import (  # noqa: E402
    NUM_CLASSES,
    StreamingSegmentationMetrics,
    VSPWFrameDataset,
    load_vspw_class_names,
)


POLICIES = {"Full": None, "FI-2": 2, "FI-3": 3, "FI-4": 4, "FI-6": 6, "FI-8": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy", choices=tuple(POLICIES), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-frames", type=int)
    return parser.parse_args()


def make_loader(dataset: VSPWFrameDataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def select_dataset(root: Path, limit_videos: int | None, limit_frames: int | None) -> VSPWFrameDataset:
    dataset = VSPWFrameDataset(root, "val", training=False)
    if limit_videos is None and limit_frames is None:
        return dataset
    selected = []
    seen_videos: list[str] = []
    per_video = defaultdict(int)
    for sample in dataset.samples:
        if sample.video_id not in seen_videos:
            seen_videos.append(sample.video_id)
        if limit_videos is not None and seen_videos.index(sample.video_id) >= limit_videos:
            continue
        if limit_frames is not None and per_video[sample.video_id] >= limit_frames:
            continue
        selected.append(sample)
        per_video[sample.video_id] += 1
    dataset.samples = selected
    return dataset


def precision_context(precision: str):
    if precision == "fp32":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def update_metric(
    metric: StreamingSegmentationMetrics,
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Accumulate a GPU batch by copying only the 124x124 confusion matrix."""
    prediction = prediction.to(torch.int64)
    labels = labels.to(torch.int64)
    invalid = (labels != 255) & ((labels < 0) | (labels >= NUM_CLASSES))
    if bool(invalid.any().item()):
        raise ValueError("canonical VSPW labels contain an out-of-range ID")
    valid = labels != 255
    encoded = labels[valid] * NUM_CLASSES + prediction[valid]
    if encoded.numel():
        counts = torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
        metric.confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def metric_payload(metric: StreamingSegmentationMetrics) -> dict:
    result = metric.compute()
    confusion = result.pop("confusion")
    result["support_pixels"] = confusion.sum(dim=1).tolist()
    result["predicted_pixels"] = confusion.sum(dim=0).tolist()
    return result


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_limits(dataset: VSPWFrameDataset, limit_videos: int | None, limit_frames: int | None) -> VSPWFrameDataset:
    if limit_videos is None and limit_frames is None:
        return dataset
    selected = []
    video_order = []
    per_video = defaultdict(int)
    for sample in dataset.samples:
        if sample.video_id not in video_order:
            video_order.append(sample.video_id)
        video_index = video_order.index(sample.video_id)
        if limit_videos is not None and video_index >= limit_videos:
            continue
        if limit_frames is not None and per_video[sample.video_id] >= limit_frames:
            continue
        selected.append(sample)
        per_video[sample.video_id] += 1
    dataset.samples = selected
    return dataset


def run_full(model, dataset, device, precision, batch_size, workers, class_names, output_root):
    loader = make_loader(dataset, batch_size, workers)
    overall = StreamingSegmentationMetrics()
    per_video = defaultdict(StreamingSegmentationMetrics)
    prediction_histogram = torch.zeros(NUM_CLASSES, dtype=torch.int64)
    frame_rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with precision_context(precision):
                logits = model(pixel_values=pixels).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)
            update_metric(overall, prediction, labels)
            prediction_histogram += torch.bincount(
                prediction.reshape(-1), minlength=NUM_CLASSES
            ).cpu()
            for video_id in dict.fromkeys(batch["video_id"]):
                indices = [index for index, value in enumerate(batch["video_id"]) if value == video_id]
                selected = torch.tensor(indices, device=device)
                update_metric(per_video[video_id], prediction.index_select(0, selected), labels.index_select(0, selected))
            for index, video_id in enumerate(batch["video_id"]):
                frame_rows.append({"video_id": video_id, "frame_id": batch["frame_id"][index], "refresh": 1, "cache_age": 0})
    metrics = metric_payload(overall)
    metrics.update(
        {
            "policy": "Full",
            "refresh_interval": 1,
            "refresh_frames": len(dataset),
            "reuse_frames": 0,
            "refresh_ratio": 1.0,
            "stage4_calls": len(dataset),
            "decoder_frames": len(dataset),
            "total_frames": len(dataset),
            "total_videos": len(per_video),
            "elapsed_seconds": time.perf_counter() - started,
            "prediction_histogram": prediction_histogram.tolist(),
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "persistent_cache_bytes": 0,
        }
    )
    write_outputs(output_root, metrics, per_video, frame_rows, class_names)
    return metrics


def segment_indices(dataset: VSPWFrameDataset, interval: int) -> list[list[int]]:
    by_video = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_video[sample.video_id].append(index)
    segments = []
    for indices in by_video.values():
        for start in range(0, len(indices), interval):
            segments.append(indices[start : start + interval])
    return segments


class SegmentBatchSampler:
    def __init__(self, segments: list[list[int]]):
        self.segments = segments

    def __iter__(self):
        return iter(self.segments)

    def __len__(self):
        return len(self.segments)


def make_segment_loader(dataset, interval, workers):
    return DataLoader(
        dataset,
        batch_sampler=SegmentBatchSampler(segment_indices(dataset, interval)),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def run_fixed(model, dataset, device, precision, interval, workers, class_names, output_root):
    loader = make_segment_loader(dataset, interval, workers)
    pipeline = StructuredReusePipeline(model)
    overall = StreamingSegmentationMetrics()
    per_video = defaultdict(StreamingSegmentationMetrics)
    frame_positions = {}
    per_video_position = defaultdict(int)
    for sample in dataset.samples:
        frame_positions[sample.image_path.as_posix()] = per_video_position[sample.video_id]
        per_video_position[sample.video_id] += 1
    current_video = None
    state: Stage4CacheState | None = None
    refresh_frames = 0
    frame_rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch in loader:
            video_ids = list(batch["video_id"])
            video_id = video_ids[0]
            if any(value != video_id for value in video_ids):
                raise RuntimeError("segment batch crosses video boundary")
            if current_video != video_id:
                current_video = video_id
                state = None
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            positions = [frame_positions[Path(path).as_posix()] for path in batch["image_path"]]
            if positions[0] % interval != 0:
                raise RuntimeError("segment does not start at a fixed refresh boundary")
            with precision_context(precision):
                f1, f2, f3 = pipeline.fresh_stage_1_to_3(pixels)
                if state is None:
                    state = Stage4CacheState()
                f4 = pipeline.stage4_cache.run_dense(f3[:1])
                state.cached_f4 = f4.detach()
                state.initialized = True
                state.age = 0
                refresh_frames += 1
                f4_batch = f4.expand(pixels.shape[0], -1, -1, -1)
                logits = model.decode_head([f1, f2, f3, f4_batch])
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            prediction = logits.argmax(dim=1)
            update_metric(overall, prediction, labels)
            update_metric(per_video[video_id], prediction, labels)
            for index, (frame_id, position) in enumerate(zip(batch["frame_id"], positions)):
                if index > 0:
                    state.age += 1
                frame_rows.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "refresh": int(index == 0),
                        "cache_age": state.age,
                        "stage4_computed": int(index == 0),
                        "position": position,
                    }
                )
    total_frames = len(dataset)
    metrics = metric_payload(overall)
    metrics.update(
        {
            "policy": f"FI-{interval}",
            "refresh_interval": interval,
            "refresh_frames": refresh_frames,
            "reuse_frames": total_frames - refresh_frames,
            "refresh_ratio": refresh_frames / total_frames,
            "stage4_calls": refresh_frames,
            "decoder_frames": total_frames,
            "total_frames": total_frames,
            "total_videos": len(per_video),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "persistent_cache_bytes": 256 * 16 * 16 * 2,
        }
    )
    write_outputs(output_root, metrics, per_video, frame_rows, class_names)
    return metrics


def write_outputs(output_root, metrics, per_video, frame_rows, class_names):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    per_video_rows = []
    for video_id, metric in sorted(per_video.items()):
        payload = metric_payload(metric)
        per_video_rows.append(
            {
                "policy": metrics["policy"],
                "video_id": video_id,
                "frames": sum(1 for row in frame_rows if row["video_id"] == video_id),
                "miou": payload["miou"],
                "pixel_accuracy": payload["pixel_accuracy"],
                "mean_class_accuracy": payload["mean_class_accuracy"],
            }
        )
    write_csv(output_root / "per_video_metrics.csv", per_video_rows)
    write_csv(output_root / "frame_records.csv", frame_rows)
    per_class_rows = []
    for index, iou in enumerate(metrics["per_class_iou"]):
        per_class_rows.append(
            {
                "class_id": index,
                "class_name": class_names[index],
                "IoU": iou,
                "support_pixels": metrics["support_pixels"][index],
                "predicted_pixels": metrics["predicted_pixels"][index],
            }
        )
    write_csv(output_root / "per_class_iou.csv", per_class_rows)


def main():
    args = parse_args()
    if args.policy not in POLICIES:
        raise ValueError(args.policy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round 7 formal benchmark requires CUDA")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device)
    model.eval()
    if int(model.config.num_labels) != NUM_CLASSES:
        raise RuntimeError("frozen model is not VSPW-124")
    class_names = load_vspw_class_names(args.dataset_root)
    dataset = select_dataset(args.dataset_root, args.limit_videos, args.limit_frames)
    if args.policy == "Full":
        metrics = run_full(model, dataset, device, args.precision, args.batch_size, args.workers, class_names, args.output_root)
    else:
        metrics = run_fixed(model, dataset, device, args.precision, POLICIES[args.policy], args.workers, class_names, args.output_root)
    print(json.dumps({key: metrics[key] for key in ("policy", "miou", "pixel_accuracy", "refresh_ratio", "stage4_calls", "total_frames", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
