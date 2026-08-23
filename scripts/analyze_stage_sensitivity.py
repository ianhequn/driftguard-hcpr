"""Measure decoder damage when one SegFormer stage is replaced by stale features."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.analysis.stage_sensitivity import stale_feature_metrics  # noqa: E402
from driftguard.models.segformer import (  # noqa: E402
    get_stage_features,
    load_segformer,
    prepare_inputs,
)
from driftguard.utils.common import choose_device, set_seed  # noqa: E402
from driftguard.utils.video import iter_video_frames  # noqa: E402


FIELDS = [
    "video_id",
    "current_frame",
    "reference_frame",
    "offset",
    "stale_stage",
    "kl_divergence",
    "pixel_disagreement_ratio",
    "logit_relative_l2",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="results/E02_stage_sensitivity/stage_sensitivity.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    offsets = sorted({int(offset) for offset in args.offsets})
    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("--offsets must contain positive integers.")

    processor, model = load_segformer(args.model, device=device)
    history: dict[int, list[torch.Tensor]] = {}
    max_offset = max(offsets)
    rows = []
    start_time = time.perf_counter()

    for frame_id, frame_rgb in iter_video_frames(args.video, max_frames=args.max_frames):
        image = Image.fromarray(frame_rgb, mode="RGB")
        inputs = prepare_inputs(processor, image, device)
        # Keep regular no-grad tensors because the decoder may save inputs
        # for backward internally even when gradients are disabled.
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
                return_dict=True,
            )
        current_features = [
            feature.detach()
            for feature in get_stage_features(outputs, model=model)
        ]
        full_logits = model.decode_head(current_features)
        for offset in offsets:
            reference_frame = frame_id - offset
            reference_features = history.get(reference_frame)
            if reference_features is None:
                continue
            for stale_stage in range(4):
                stale_features = list(current_features)
                stale_features[stale_stage] = reference_features[stale_stage]
                metrics = stale_feature_metrics(
                    model=model,
                    current_features=current_features,
                    stale_features=stale_features,
                    full_logits=full_logits,
                )
                rows.append(
                    {
                        "video_id": Path(args.video).stem,
                        "current_frame": frame_id,
                        "reference_frame": reference_frame,
                        "offset": offset,
                        "stale_stage": stale_stage + 1,
                        **metrics,
                    }
                )
        history[frame_id] = current_features
        for old_frame in list(history):
            if old_frame < frame_id - max_offset:
                del history[old_frame]

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"device={device}")
    print(f"frames={args.max_frames}")
    print(f"rows={len(rows)}")
    print(f"analysis_time_sec={time.perf_counter() - start_time:.3f}")
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
