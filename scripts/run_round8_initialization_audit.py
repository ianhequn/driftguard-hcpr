"""Construct VSPW-124 SegFormer-B1 and audit structured-reuse compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.structured_reuse.pipeline import StructuredReusePipeline  # noqa: E402
from driftguard.training.vspw_segformer import (  # noqa: E402
    IGNORE_INDEX,
    NUM_CLASSES,
    build_vspw_segformer,
    load_vspw_class_names,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("B1 initialization audit requires CUDA")
    class_names = load_vspw_class_names(args.dataset_root)
    model, initialization = build_vspw_segformer(args.source_dir, class_names)
    model = model.to("cuda").eval()
    pipeline = StructuredReusePipeline(model)
    pixel = torch.zeros(1, 3, 512, 512, device="cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        f1, f2, f3 = pipeline.fresh_stage_1_to_3(pixel)
        f4 = pipeline.stage4_cache.run_dense(f3)
        logits = model.decode_head([f1, f2, f3, f4])
    checks = {
        "num_labels_124": int(model.config.num_labels) == NUM_CLASSES,
        "ignore_index_255": int(model.config.semantic_loss_ignore_index) == IGNORE_INDEX,
        "four_features": len((f1, f2, f3, f4)) == 4,
        "stage4_callable_from_f3": pipeline.stage4_cache.stage4_forward_calls == 1,
        "decoder_consumes_f1_f4": list(logits.shape) == [1, NUM_CLASSES, 128, 128],
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
    }
    payload = {
        "architecture": "SegFormer-B1",
        "source_dir": str(args.source_dir),
        "num_labels": int(model.config.num_labels),
        "class_names": len(class_names),
        "ignore_index": int(model.config.semantic_loss_ignore_index),
        "initialization": initialization.__dict__,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "feature_shapes": [list(feature.shape) for feature in (f1, f2, f3, f4)],
        "logits_shape": list(logits.shape),
        "encoder_depths": list(model.config.depths),
        "hidden_sizes": list(model.config.hidden_sizes),
        "attention_heads": list(model.config.num_attention_heads),
        "sr_ratios": list(model.config.sr_ratios),
        "decoder_hidden_size": int(model.config.decoder_hidden_size),
        "stage4_cache_f4_shape": list(f4.shape),
        "structured_reuse_compatibility": "DIRECT" if all(checks.values()) else "MAJOR ADAPTATION",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not all(checks.values()):
        raise SystemExit("B1 initialization or structured reuse audit failed")


if __name__ == "__main__":
    main()
