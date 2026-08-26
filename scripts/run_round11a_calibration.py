"""Unlabeled VSPW-train calibration for HCPR drift and hardware costs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from driftguard.models.segformer import get_stage_features  # noqa: E402
from driftguard.training.vspw_segformer import IMAGE_MEAN, IMAGE_STD, VSPWFrameDataset  # noqa: E402
from driftguard.structured_reuse.hcpr import HardwareCalibratedProgressiveReusePipeline  # noqa: E402


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    values = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    return ((values - IMAGE_MEAN) / IMAGE_STD).unsqueeze(0)


def cosine_drift(current: torch.Tensor, previous: torch.Tensor) -> float:
    current = F.normalize(F.adaptive_avg_pool2d(current.float(), 1).flatten(1), dim=1)
    previous = F.normalize(F.adaptive_avg_pool2d(previous.float(), 1).flatten(1), dim=1)
    return float((1.0 - (current * previous).sum(dim=1)).mean().item())


def timed(fn, warmup: int, measurements: int) -> list[float]:
    values = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(measurements):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); fn(); end.record(); end.synchronize()
            values.append(float(start.elapsed_time(end)))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(array.mean()), "median_ms": float(np.median(array)),
            "p95_ms": float(np.quantile(array, .95)), "p99_ms": float(np.quantile(array, .99)),
            "std_ms": float(array.std()), "measurements": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=32)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        raise RuntimeError("Round 11A calibration requires RTX 4090 D")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "train", training=False)
    by_video: dict[str, list] = {}
    for sample in dataset.samples:
        by_video.setdefault(sample.video_id, []).append(sample)
    selected_ids = sorted(by_video)[:args.videos]
    if len(selected_ids) != args.videos:
        raise RuntimeError(f"expected {args.videos} training videos, found {len(selected_ids)}")
    drifts = {"d1": [], "d2": [], "d3": []}
    with torch.inference_mode():
        for video_id in selected_ids:
            previous = None
            for sample in by_video[video_id][:args.frames_per_video]:
                pixels = image_tensor(sample.image_path).to("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
                features = get_stage_features(outputs, model=model.segformer)
                if previous is not None:
                    for index, key in enumerate(("d1", "d2", "d3")):
                        drifts[key].append(cosine_drift(features[index], previous[index]))
                previous = [feature.detach() for feature in features]
    scales = {key: float(np.percentile(values, 90)) for key, values in drifts.items()}
    pipeline = HardwareCalibratedProgressiveReusePipeline(model)
    pixels = image_tensor(dataset.samples[0].image_path).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        features = pipeline._full_frame(pixels, pipeline.new_state())
    inputs = {2: features[0], 3: features[1], 4: features[2]}
    timings: dict[str, list[float]] = {"Full": [] , "Stage2": [], "Stage3": [], "Stage4": []}
    for _ in range(args.repetitions):
        for key, fn in {
            "Full": lambda: model(pixel_values=pixels).logits,
            "Stage2": lambda: pipeline._run_stage(1, inputs[2]),
            "Stage3": lambda: pipeline._run_stage(2, inputs[3]),
            "Stage4": lambda: pipeline._run_stage(3, inputs[4]),
        }.items():
            timings[key].extend(timed(fn, args.warmup, args.measurements))
    costs = {key: summarize(value) for key, value in timings.items()}
    full_ms = costs["Full"]["mean_ms"]
    hardware = {"c2_ms": costs["Stage2"]["mean_ms"], "c3_ms": costs["Stage3"]["mean_ms"], "c4_ms": costs["Stage4"]["mean_ms"],
                "full_ms": full_ms, "k2": costs["Stage2"]["mean_ms"] / full_ms, "k3": costs["Stage3"]["mean_ms"] / full_ms, "k4": costs["Stage4"]["mean_ms"] / full_ms,
                "timings": costs}
    scores = []
    for key, values in drifts.items():
        downstream_stage = int(key[-1]) + 1
        normalized = np.asarray(values) / (scales[key] + 1e-12)
        scores.extend((normalized / hardware[f"k{downstream_stage}"]).tolist())
    score_array = np.asarray(scores, dtype=np.float64)
    quantiles = [20, 35, 50, 65, 80, 90, 95]
    lambdas = [{"quantile": q, "lambda": float(np.quantile(score_array, q / 100.0))} for q in quantiles]
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    (output / "calibration_videos.txt").write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    (output / "drift_statistics.json").write_text(json.dumps({"seed": 42, "videos": selected_ids, "counts": {k: len(v) for k, v in drifts.items()}, "scales": scales, "values": drifts}, indent=2) + "\n", encoding="utf-8")
    (output / "hardware_costs.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    (output / "lambda_grid.json").write_text(json.dumps({"quantiles": quantiles, "values": lambdas, "score_count": len(scores)}, indent=2) + "\n", encoding="utf-8")
    with (output / "score_histograms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["score"]); writer.writerows([[value] for value in scores])
    print(json.dumps({"videos": selected_ids, "scales": scales, "hardware": hardware, "lambda_grid": lambdas}, indent=2))


if __name__ == "__main__":
    main()
