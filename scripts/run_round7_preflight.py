"""Run Round 7 frozen-artifact, architecture, and 100% refresh gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


EXPECTED_SHA256 = "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    model_path = args.model_dir / "model.safetensors"
    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"MODEL ARTIFACT MISMATCH: {actual_sha256}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Round 7 requires CUDA")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device)
    model.eval()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    pipeline = StructuredReusePipeline(model)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    max_errors = []
    mean_errors = []
    relative_l2 = []
    disagreements = []
    stage_shapes = None
    with torch.inference_mode():
        for index in range(min(args.frames, len(dataset))):
            pixels = dataset[index]["pixel_values"].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32"):
                full = model(pixel_values=pixels, return_dict=True).logits
                actual, _, stats = pipeline.forward(pixels, refresh=True, state=None)
            actual = F.interpolate(actual, size=full.shape[-2:], mode="bilinear", align_corners=False)
            difference = actual - full
            max_errors.append(float(difference.abs().max()))
            mean_errors.append(float(difference.abs().mean()))
            relative_l2.append(float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(full).clamp_min(1e-12)))
            disagreements.append(float((actual.argmax(dim=1) != full.argmax(dim=1)).float().mean()))
            if stats["stage4_calls_delta"] != 1:
                raise RuntimeError("100% refresh did not execute Stage 4")
            if stage_shapes is None:
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32"):
                    features = pipeline.fresh_stage_1_to_3(pixels)
                    stage4 = pipeline.stage4_cache.run_dense(features[-1])
                stage_shapes = [list(feature.shape) for feature in [*features, stage4]]
    result = {
        "model_sha256": actual_sha256,
        "expected_sha256": EXPECTED_SHA256,
        "sha256_pass": actual_sha256 == EXPECTED_SHA256,
        "num_labels": int(model.config.num_labels),
        "frames": min(args.frames, len(dataset)),
        "stage_shapes": stage_shapes,
        "max_abs_error": max(max_errors),
        "mean_abs_error": sum(mean_errors) / len(mean_errors),
        "relative_l2": sum(relative_l2) / len(relative_l2),
        "max_prediction_disagreement": max(disagreements),
        "allclose": max(max_errors) <= 1e-6,
        "stage4_calls": min(args.frames, len(dataset)),
        "decoder_frames": min(args.frames, len(dataset)),
        "temporal_reuse_executed": False,
    }
    if not result["allclose"] or result["max_prediction_disagreement"] != 0.0:
        raise RuntimeError(f"100% refresh equivalence failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
