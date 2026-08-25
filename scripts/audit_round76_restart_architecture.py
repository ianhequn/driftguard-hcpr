"""Audit frozen SegFormer-B0 stage shapes and decoder inputs without temporal reuse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import VSPWFrameDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_dir, local_files_only=True
    ).to(device)
    model.eval()
    sample = VSPWFrameDataset(args.dataset_root, "val", training=False)[0]
    inputs = sample["pixel_values"].unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(pixel_values=inputs, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    stages = []
    for index, hidden in enumerate(hidden_states, start=1):
        if hidden.ndim == 4:
            shape = list(hidden.shape)
            channels, height, width = shape[1], shape[2], shape[3]
        else:
            batch, tokens, channels = hidden.shape
            height = width = int(tokens**0.5)
            if height * width != tokens:
                raise RuntimeError(f"stage {index} token grid is not square: {tokens}")
            shape = [batch, channels, height, width]
        stages.append(
            {
                "stage": index,
                "hidden_shape": list(hidden.shape),
                "feature_shape_bchw": shape,
            }
        )
    decoder = model.decode_head
    audit = {
        "model_dir": str(args.model_dir),
        "device": str(device),
        "input_shape": list(inputs.shape),
        "logits_shape": list(outputs.logits.shape),
        "stages": stages,
        "stage4": stages[-1],
        "decoder_linear_c_count": len(decoder.linear_c),
        "decoder_consumes_stage_count": len(decoder.linear_c),
        "decoder_classifier_out_channels": int(decoder.classifier.out_channels),
        "num_labels": int(model.config.num_labels),
        "structured_reuse_compatibility": (
            "DIRECT" if len(decoder.linear_c) == 4 and len(stages) == 4 else "MAJOR ADAPTATION"
        ),
        "temporal_reuse_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
