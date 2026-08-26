"""Check 100% refresh equivalence for all three Round-10 boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftguard.structured_reuse.boundaries import (  # noqa: E402
    BOUNDARIES,
    StructuredBoundaryReusePipeline,
)
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Round 10 equivalence requires CUDA")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_dir, local_files_only=True).to("cuda").eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    rows = []
    with torch.inference_mode():
        for index in range(args.frames):
            pixels = dataset[index]["pixel_values"].unsqueeze(0).to("cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reference = model(pixel_values=pixels).logits
                for boundary in BOUNDARIES:
                    pipeline = StructuredBoundaryReusePipeline(model, boundary)
                    actual, _, _ = pipeline.forward(pixels, refresh=True, state=None)
                    delta = (actual - reference).float()
                    rows.append({
                        "frame": index, "boundary": boundary,
                        "max_abs": float(delta.abs().max().item()),
                        "mean_abs": float(delta.abs().mean().item()),
                        "relative_l2": float(delta.norm().item() / reference.float().norm().item()),
                        "allclose": bool(torch.allclose(actual, reference, rtol=1e-3, atol=1e-3)),
                    })
    payload = {"frames": args.frames, "rows": rows, "pass": all(row["allclose"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": args.frames, "pass": payload["pass"]}, indent=2))


if __name__ == "__main__":
    main()
