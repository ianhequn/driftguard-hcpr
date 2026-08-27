"""Paired CUDA latency for frozen Round 11B variants on the second GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from driftguard.structured_reuse.boundaries import StructuredBoundaryReusePipeline  # noqa: E402
from driftguard.structured_reuse.r11b import make_hcpr_variant  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.quantile(a, .95)), "p99_ms": float(np.quantile(a, .99)), "std_ms": float(a.std()), "fps": float(1000.0 / a.mean()), "measurements": len(a)}


def bootstrap(values, seed=42, samples=10000):
    a = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed)
    return [float(np.quantile(rng.choice(a, size=(samples, len(a)), replace=True).mean(axis=1), q)) for q in (.025, .975)]


def event_call(fn):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(); fn(); end.record(); end.synchronize()
    return float(start.elapsed_time(end))


def run_stream(model, kind, config, pixels, video_ids, state_data, new_costs, warmup, measurements):
    if kind == "fixed":
        pipeline = StructuredBoundaryReusePipeline(model, config["boundary"])
    else:
        scales = {2: state_data["drift_scales"][config["model_key"]]["d1"], 3: state_data["drift_scales"][config["model_key"]]["d2"], 4: state_data["drift_scales"][config["model_key"]]["d3"]}
        pipeline = make_hcpr_variant(model, config["variant"], config["lambda"], scales, state_data["stage_costs"][config["model_key"]], new_costs, 7)
    state = None; previous_video = None; positions = {}; values = []; calls = {f"stage{i}": 0 for i in range(1, 5)} | {"decoder": 0}; routes = {}

    def one(emitted, collect):
        nonlocal state, previous_video
        index = emitted % pixels.shape[0]; video_id = video_ids[index]
        if video_id != previous_video:
            state = None; positions[video_id] = 0; previous_video = video_id
        position = positions[video_id]; positions[video_id] += 1; pixel = pixels[index:index + 1]; result = {}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if kind == "fixed":
                refresh = position % config["interval"] == 0 or state is None
                elapsed = event_call(lambda: result.setdefault("value", pipeline.forward(pixel, refresh=refresh, state=state)))
                _, state, stats = result["value"]; route = config["name"]
            else:
                elapsed = event_call(lambda: result.setdefault("value", pipeline.forward(pixel, state=state)))
                _, state, stats = result["value"]; route = stats["route"]
            for key, value in stats["stage_calls"].items(): calls[key] += int(value)
        routes[route] = routes.get(route, 0) + 1
        if collect: values.append(elapsed)

    for emitted in range(warmup): one(emitted, False)
    if kind == "fixed": pipeline = StructuredBoundaryReusePipeline(model, config["boundary"])
    else:
        scales = {2: state_data["drift_scales"][config["model_key"]]["d1"], 3: state_data["drift_scales"][config["model_key"]]["d2"], 4: state_data["drift_scales"][config["model_key"]]["d3"]}
        pipeline = make_hcpr_variant(model, config["variant"], config["lambda"], scales, state_data["stage_costs"][config["model_key"]], new_costs, 7)
    state = None; previous_video = None; positions = {}; calls = {f"stage{i}": 0 for i in range(1, 5)} | {"decoder": 0}; routes = {}
    for emitted in range(measurements): one(emitted, True)
    return values, {"stage_calls": calls, "route_counts": routes}


def paired_full(model, pixels, warmup, measurements):
    values = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for emitted in range(warmup):
            pixel = pixels[emitted % pixels.shape[0]:emitted % pixels.shape[0] + 1]; event_call(lambda: model(pixel_values=pixel).logits)
        for emitted in range(measurements):
            pixel = pixels[emitted % pixels.shape[0]:emitted % pixels.shape[0] + 1]; values.append(event_call(lambda: model(pixel_values=pixel).logits))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True); parser.add_argument("--model-key", choices=("B0", "B1"), required=True); parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--round11a-state", type=Path, required=True); parser.add_argument("--new-cost", type=Path, required=True); parser.add_argument("--variant", choices=("HCPR-NewCost", "HCPR-OldCost", "ProgressiveDriftOnly"), default="HCPR-NewCost"); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--include-fixed", action="store_true"); parser.add_argument("--quantiles", default="20,35,50,65,80,90,95"); parser.add_argument("--warmup", type=int, default=200); parser.add_argument("--measurements", type=int, default=1000); parser.add_argument("--repetitions", type=int, default=5); parser.add_argument("--pool", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "3080 Ti" not in torch.cuda.get_device_name(0): raise RuntimeError("Round 11B latency requires the RTX 3080 Ti")
    state_data = json.loads(args.round11a_state.read_text(encoding="utf-8")); new_costs = json.loads(args.new_cost.read_text(encoding="utf-8")); model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False); samples = [dataset[i] for i in range(min(args.pool, len(dataset)))]; pixels = torch.stack([sample["pixel_values"] for sample in samples]).to("cuda"); video_ids = [sample["video_id"] for sample in samples]
    values = {int(q): next(float(x["lambda"]) for x in state_data["lambda_grid"][args.model_key]["values"] if int(x["quantile"]) == int(q)) for q in args.quantiles.split(",")}
    configs = []
    if args.include_fixed:
        configs = [("fixed", {"name": "S4/FI2", "boundary": "s4", "interval": 2}), ("fixed", {"name": "S3-S4/FI2", "boundary": "s3_s4", "interval": 2}), ("fixed", {"name": "S2-S4/FI2", "boundary": "s2_s4", "interval": 2}), ("fixed", {"name": "S2-S4/FI4", "boundary": "s2_s4", "interval": 4}), ("fixed", {"name": "S2-S4/FI8", "boundary": "s2_s4", "interval": 8})]
    for q, value in values.items(): configs.append(("hcpr", {"name": f"{args.variant}/q{q}", "variant": args.variant, "lambda": value, "quantile": q, "model_key": args.model_key}))
    rows = []
    for repetition in range(args.repetitions):
        for kind, config in configs:
            policy_values, diagnostics = run_stream(model, kind, config, pixels, video_ids, state_data, new_costs, args.warmup, args.measurements); full_values = paired_full(model, pixels, args.warmup, args.measurements); policy = summarize(policy_values); full = summarize(full_values)
            row = {"method": config["name"], "kind": kind, "repetition": repetition + 1, "lambda": config.get("lambda"), "quantile": config.get("quantile"), "variant": config.get("variant"), "paired_full": full, "policy": policy, "paired_speedup": full["mean_ms"] / policy["mean_ms"], **diagnostics}; rows.append(row); print(json.dumps({"method": row["method"], "repetition": row["repetition"], "policy_mean_ms": policy["mean_ms"], "full_mean_ms": full["mean_ms"], "paired_speedup": row["paired_speedup"]}), flush=True)
    summary = {}
    for _, config in configs:
        vals = [row["paired_speedup"] for row in rows if row["method"] == config["name"]]; summary[config["name"]] = {"repetitions": len(vals), "mean_speedup": float(np.mean(vals)), "std_speedup": float(np.std(vals)), "min_speedup": float(np.min(vals)), "max_speedup": float(np.max(vals)), "ci95": bootstrap(vals)}
    payload = {"model": args.model_key, "gpu": torch.cuda.get_device_name(0), "variant": args.variant, "warmup": args.warmup, "measurements": args.measurements, "repetitions": args.repetitions, "pool": args.pool, "precision": "bf16", "timing_boundary": "CUDA-resident input to logits, controller included, paired Full immediately after each policy", "rows": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
