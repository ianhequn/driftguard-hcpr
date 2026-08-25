"""Evaluate a frozen VSPW-124 checkpoint in a fresh process."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import (  # noqa: E402
    NUM_CLASSES,
    VSPWFrameDataset,
    evaluate_model,
    load_vspw_class_names,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device)
    model.eval()
    class_names = load_vspw_class_names(args.dataset_root)
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    alignment = [id2label.get(index) == name for index, name in enumerate(class_names)]
    if int(model.config.num_labels) != NUM_CLASSES or not all(alignment):
        raise RuntimeError("frozen model class alignment failed")

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
    started = time.perf_counter()
    result = evaluate_model(model, loader, device, args.precision)
    elapsed = time.perf_counter() - started
    confusion = result.pop("confusion")
    support = confusion.sum(dim=1).tolist()
    predicted = confusion.sum(dim=0).tolist()
    result.update(
        {
            "model_dir": str(args.model_dir),
            "validation_samples": len(dataset),
            "elapsed_seconds": elapsed,
            "device": str(device),
            "num_labels": int(model.config.num_labels),
            "class_alignment": f"{sum(alignment)} / {len(alignment)}",
            "never_predicted_classes": [index for index, value in enumerate(predicted) if value == 0],
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "IoU", "support_pixels", "predicted_pixels"],
        )
        writer.writeheader()
        for index, (iou, gt_count, pred_count) in enumerate(
            zip(result["per_class_iou"], support, predicted)
        ):
            writer.writerow(
                {
                    "class_id": index,
                    "class_name": class_names[index],
                    "IoU": iou,
                    "support_pixels": gt_count,
                    "predicted_pixels": pred_count,
                }
            )
    (args.output_dir / "class_alignment.json").write_text(
        json.dumps(
            {
                "num_labels": int(model.config.num_labels),
                "verified": all(alignment),
                "verified_count": sum(alignment),
                "total": len(alignment),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("miou", "pixel_accuracy", "mean_class_accuracy", "elapsed_seconds", "class_alignment")}, indent=2))


if __name__ == "__main__":
    main()
