"""Full VSPW quality evaluation for one frozen Round 11B controller variant."""

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
from driftguard.structured_reuse.r11b import make_hcpr_variant  # noqa: E402
from driftguard.training.vspw_segformer import StreamingSegmentationMetrics, VSPWFrameDataset  # noqa: E402


def limit_dataset(dataset, limit_videos: int | None, limit_frames: int | None):
    if limit_videos is None and limit_frames is None:
        return dataset
    selected, order, counts = [], [], defaultdict(int)
    for sample in dataset.samples:
        if sample.video_id not in order:
            order.append(sample.video_id)
        if limit_videos is not None and order.index(sample.video_id) >= limit_videos:
            continue
        if limit_frames is not None and counts[sample.video_id] >= limit_frames:
            continue
        selected.append(sample); counts[sample.video_id] += 1
    dataset.samples = selected
    return dataset


def update_metric(metrics, logits, labels):
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    metrics.update(logits.argmax(dim=1).to(torch.int64), labels)


def evaluate(model, dataset, variant, lambda_value, old_costs, new_costs, scales, max_age, workers, full_miou):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    pipeline = make_hcpr_variant(model, variant, lambda_value, scales, old_costs, new_costs, max_age)
    metrics, routes, calls = StreamingSegmentationMetrics(), Counter(), Counter()
    age_sums, age_max, stale_run, stale_max, videos = Counter(), Counter(), Counter(), Counter(), set()
    state = None; previous_video = None
    with torch.inference_mode():
        for batch in loader:
            video_id = batch["video_id"][0]
            if video_id != previous_video:
                state = None; previous_video = video_id
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
                age_sums[stage] += age; age_max[stage] = max(age_max[stage], age)
            for stage, key in ((2, "stage2"), (3, "stage3"), (4, "stage4")):
                if stats["stage_calls"][key]:
                    stale_run[stage] = 0
                else:
                    stale_run[stage] += 1; stale_max[stage] = max(stale_max[stage], stale_run[stage])
    result = metrics.compute(); result.pop("confusion")
    frames = len(dataset)
    return {
        "variant": variant, "lambda": float(lambda_value), "max_stale_run": max_age,
        "videos": len(videos), "frames": frames, "mIoU": result["miou"], "mIoU_drop_pp": (full_miou - result["miou"]) * 100.0,
        "pixel_accuracy": result["pixel_accuracy"], "mean_class_accuracy": result["mean_class_accuracy"],
        "route_counts": dict(routes), "route_percentages": {k: v / frames for k, v in routes.items()},
        "stage_calls": {f"stage{i}": int(calls[f"stage{i}"]) for i in range(1, 5)} | {"decoder": int(calls["decoder"])},
        "actual_stage_refresh_ratios": {f"stage{i}": calls[f"stage{i}"] / frames for i in (2, 3, 4)},
        "average_age": {key: age_sums[key] / frames for key in ("age2", "age3", "age4")},
        "maximum_age": dict(age_max), "maximum_stale_runs": {f"stage{i}": int(stale_max[i]) for i in (2, 3, 4)},
        "persistent_cache_bytes": state.persistent_bytes if state is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True); parser.add_argument("--model-key", choices=("B0", "B1"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--round11a-state", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True); parser.add_argument("--new-cost", type=Path, required=True); parser.add_argument("--variant", choices=("HCPR-NewCost", "HCPR-OldCost", "ProgressiveDriftOnly"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--workers", type=int, default=8); parser.add_argument("--limit-videos", type=int); parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "3080 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Round 11B quality requires the RTX 3080 Ti")
    state = json.loads(args.round11a_state.read_text(encoding="utf-8"))
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = limit_dataset(VSPWFrameDataset(args.dataset_root, "val", training=False), args.limit_videos, args.limit_frames)
    old_costs = state["stage_costs"][args.model_key]
    new_costs = json.loads(args.new_cost.read_text(encoding="utf-8"))
    scales = {2: state["drift_scales"][args.model_key]["d1"], 3: state["drift_scales"][args.model_key]["d2"], 4: state["drift_scales"][args.model_key]["d3"]}
    full_payload = json.loads(args.full_results.read_text(encoding="utf-8"))
    if isinstance(full_payload, list):
        full_payload = next(row for row in full_payload if row.get("Model", row.get("model")) == args.model_key and str(row.get("Boundary", row.get("boundary", ""))).upper() == "FULL")
    full_miou = float(full_payload.get("mIoU", full_payload.get("miou")))
    grid = state["lambda_grid"][args.model_key]["values"]
    args.output_dir.mkdir(parents=True, exist_ok=True); rows = []
    for entry in grid:
        value = float(entry["lambda"])
        result = evaluate(model, dataset, args.variant, value, old_costs, new_costs, scales, 7, args.workers, full_miou)
        result["quantile"] = entry["quantile"]; result["model"] = args.model_key; rows.append(result)
        (args.output_dir / f"lambda_{value:.8g}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("variant", "lambda", "mIoU", "mIoU_drop_pp", "route_counts", "stage_calls")}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"model": args.model_key, "variant": args.variant, "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
