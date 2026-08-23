"""Extract and summarize SegFormer MiT stage 1-4 features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import (  # noqa: E402
    DEFAULT_IMAGE_URL,
    get_feature_stats,
    get_stage_features,
    load_image,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed, start_run  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", default=None)
    source.add_argument("--video", default=None)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, logger = start_run("extract_features", vars(args))
    set_seed(args.seed)
    device = choose_device(args.device)
    processor, model = load_segformer(args.model, device=device)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "feature_stats.csv"
    fieldnames = [
        "sample_id",
        "stage",
        "shape",
        "dtype",
        "mean",
        "std",
        "min",
        "max",
        "has_nan",
        "has_inf",
        "all_finite",
    ]

    if args.video:
        frame_iterator = iter_video_frames(args.video, max_frames=args.max_frames)
        samples = ((f"frame_{frame_id:06d}", Image.fromarray(frame, "RGB")) for frame_id, frame in frame_iterator)
    else:
        image_source = args.image or DEFAULT_IMAGE_URL
        samples = (("image_000000", load_image(image_source, PROJECT_ROOT / "data" / "samples")),)

    rows = []
    total_samples = 0
    for sample_id, image in samples:
        inputs = prepare_inputs(processor, image, device)
        with torch.inference_mode():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
        features = get_stage_features(outputs, model=model)
        stats = get_feature_stats(features)
        arrays = {f"stage{index}": feature[0].detach().cpu().numpy() for index, feature in enumerate(features, 1)}
        np.savez_compressed(output_dir / f"{sample_id}.npz", **arrays)
        for item in stats:
            rows.append({"sample_id": sample_id, **item})
            print(
                f"{sample_id} stage={item['stage']} shape={item['shape']} "
                f"dtype={item['dtype']} mean={item['mean']:.6g} std={item['std']:.6g} "
                f"finite={item['all_finite']}"
            )
        total_samples += 1
        logger.info("Extracted %s", sample_id)

    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if total_samples == 0:
        raise RuntimeError("No samples were processed.")
    print(f"device={device}")
    print(f"samples={total_samples}")
    print(f"stats={stats_path.resolve()}")
    print(f"features_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
