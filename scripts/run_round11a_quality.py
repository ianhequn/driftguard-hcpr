"""Evaluate a frozen HCPR lambda grid on chronological VSPW validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import NUM_CLASSES, StreamingSegmentationMetrics, VSPWFrameDataset  # noqa: E402


def load_grid(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("values", payload)


def limit_dataset(dataset, limit_videos: int | None, limit_frames: int | None):
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


def update_metric(metrics, logits, labels):
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.argmax(dim=1).to(torch.int64)
    metrics.update(prediction, labels)


def evaluate_lambda(model, dataset, lambda_value, hardware, scales, max_stale_run, workers, full_miou):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    pipeline = HardwareCalibratedProgressiveReusePipeline(
        model, stage_costs=hardware, drift_scales=scales,
        lambda_value=lambda_value, max_stale_run=max_stale_run,
    )
    metrics = StreamingSegmentationMetrics()
    routes = Counter()
    calls = Counter()
    age_sums = Counter()
    age_max = Counter()
    stale_run = Counter()
    stale_max = Counter()
    previous_video = None
    state = None
    videos = set()
    with torch.inference_mode():
        for batch in loader:
            video_id = batch["video_id"][0]
            if video_id != previous_video:
                state = None
                previous_video = video_id
            videos.add(video_id)
            pixels = batch["pixel_values"].to("cuda", non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, state, stats = pipeline.forward(pixels, state=state)
            update_metric(metrics, logits, labels)
            routes[stats["route"]] += 1
            for key, value in stats["stage_calls"].items():
                calls[key] += int(value)
            for stage, age in (("age2", state.age2), ("age3", state.age3), ("age4", state.age4)):
                age_sums[stage] += age
                age_max[stage] = max(age_max[stage], age)
            for stage, key in ((2, "stage2"), (3, "stage3"), (4, "stage4")):
                if stats["stage_calls"][key]:
                    stale_run[stage] = 0
                else:
                    stale_run[stage] += 1
                    stale_max[stage] = max(stale_max[stage], stale_run[stage])
    result = metrics.compute()
    result.pop("confusion")
    frames = len(dataset)
    payload = {
        "model": model.name_or_path if hasattr(model, "name_or_path") else "unknown",
        "lambda": float(lambda_value),
        "max_stale_run": max_stale_run,
        "videos": len(videos), "frames": frames,
        "mIoU": result["miou"], "mIoU_drop_pp": (full_miou - result["miou"]) * 100.0,
        "pixel_accuracy": result["pixel_accuracy"],
        "mean_class_accuracy": result["mean_class_accuracy"],
        "route_counts": dict(routes),
        "route_percentages": {key: value / frames for key, value in routes.items()},
        "stage_calls": {f"stage{i}": int(calls[f"stage{i}"]) for i in range(1, 5)} | {"decoder": int(calls["decoder"])},
        "actual_stage_refresh_ratios": {f"stage{i}": calls[f"stage{i}"] / frames for i in (2, 3, 4)},
        "average_age": {key: age_sums[key] / frames for key in ("age2", "age3", "age4")},
        "maximum_age": dict(age_max),
        "maximum_stale_runs": {f"stage{stage}": int(stale_max[stage]) for stage in (2, 3, 4)},
        "persistent_cache_bytes": state.persistent_bytes if state is not None else 0,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=Path, required=True)
    parser.add_argument("--hardware-costs", type=Path, required=True)
    parser.add_argument("--drift-statistics", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-stale-run", type=int, default=7)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError("Round 11A quality requires RTX 4090 D")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = limit_dataset(VSPWFrameDataset(args.dataset_root, "val", training=False), args.limit_videos, args.limit_frames)
    grid = load_grid(args.lambda_grid)
    hardware = json.loads(args.hardware_costs.read_text(encoding="utf-8"))
    scales = {int(key[-1]) + 1: value for key, value in json.loads(args.drift_statistics.read_text(encoding="utf-8"))["scales"].items()}
    full_payload = json.loads(args.full_results.read_text(encoding="utf-8"))
    if isinstance(full_payload, list):
        model_key = "B1" if "b1" in args.model_dir.name.lower() else "B0"
        full_payload = next(row for row in full_payload if row.get("Model", row.get("model")) == model_key and str(row.get("Boundary", row.get("boundary", ""))).upper() == "FULL")
    full_miou = float(full_payload.get("mIoU", full_payload.get("miou")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in grid:
        value = float(entry["lambda"] if isinstance(entry, dict) else entry)
        result = evaluate_lambda(model, dataset, value, hardware, scales, args.max_stale_run, args.workers, full_miou)
        result["quantile"] = entry.get("quantile") if isinstance(entry, dict) else None
        rows.append(result)
        (args.output_dir / f"lambda_{value:.8g}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("lambda", "mIoU", "mIoU_drop_pp", "route_counts", "stage_calls")}, indent=2), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"model": args.model_dir.name, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
