"""Check frozen B1 standard forward against whole-Stage4 full refresh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import Stage4CacheState, StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("reuse equivalence requires CUDA")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    pipeline = StructuredReusePipeline(model)
    rows = []
    state: Stage4CacheState | None = None
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= args.frames:
                break
            pixel_values = batch["pixel_values"].to("cuda", non_blocking=True)
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            enabled = args.precision != "fp32"
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                reference = model(pixel_values=pixel_values).logits
                candidate, state, stats = pipeline.forward(
                    pixel_values, refresh=True, state=state
                )
            difference = (candidate - reference).float()
            reference_float = reference.float()
            rows.append(
                {
                    "frame_index": index,
                    "max_abs_error": float(difference.abs().max().item()),
                    "mean_abs_error": float(difference.abs().mean().item()),
                    "relative_l2": float(
                        torch.linalg.vector_norm(difference)
                        / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
                    ),
                    "stage4_calls_delta": int(stats["stage4_calls_delta"]),
                    "actual_refresh": bool(stats["actual_refresh"]),
                }
            )
    if not rows:
        raise RuntimeError("no validation frames were loaded")
    summary = {
        "architecture": "SegFormer-B1",
        "frames_checked": len(rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "mean_abs_error": sum(row["mean_abs_error"] for row in rows) / len(rows),
        "max_relative_l2": max(row["relative_l2"] for row in rows),
        "all_refresh": all(row["actual_refresh"] for row in rows),
        "stage4_call_deltas": [row["stage4_calls_delta"] for row in rows],
        "allclose_atol_1e-5_rtol_1e-4": all(
            row["max_abs_error"] <= 1e-5 or row["relative_l2"] <= 1e-4 for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["allclose_atol_1e-5_rtol_1e-4"] or not summary["all_refresh"]:
        raise SystemExit("whole-stage full-refresh equivalence failed")


if __name__ == "__main__":
    main()
