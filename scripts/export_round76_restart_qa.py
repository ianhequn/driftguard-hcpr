"""Export a small RGB/GT/prediction QA set for the frozen restart model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import (  # noqa: E402
    IMAGE_MEAN,
    IMAGE_STD,
    VSPWFrameDataset,
)


def colorize(mask: np.ndarray) -> Image.Image:
    """Use a stable grayscale ID map; 255 remains visibly distinct as white."""
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if args.count < 20:
        raise ValueError("Round 7.6 Restart QA requires at least 20 frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device)
    model.eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.inference_mode():
        for index in range(min(args.count, len(dataset))):
            sample = dataset[index]
            pixels = sample["pixel_values"].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                logits = model(pixel_values=pixels).logits
            prediction = torch.nn.functional.interpolate(
                logits,
                size=sample["labels"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            rgb = sample["pixel_values"].cpu() * IMAGE_STD + IMAGE_MEAN
            rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            gt = sample["labels"].numpy().astype(np.uint8)
            stem = f"frame_{index:04d}"
            Image.fromarray(rgb, mode="RGB").save(args.output_dir / f"{stem}_rgb.png")
            colorize(gt).save(args.output_dir / f"{stem}_gt.png")
            colorize(prediction).save(args.output_dir / f"{stem}_prediction.png")
            manifest.append(
                {
                    "index": index,
                    "video_id": sample["video_id"],
                    "frame_id": sample["frame_id"],
                    "image_path": sample["image_path"],
                    "rgb": f"{stem}_rgb.png",
                    "gt": f"{stem}_gt.png",
                    "prediction": f"{stem}_prediction.png",
                }
            )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "count": len(manifest),
                "model_dir": str(args.model_dir),
                "contract": "canonical VSPW IDs 0..123, ignore 255",
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(manifest), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
