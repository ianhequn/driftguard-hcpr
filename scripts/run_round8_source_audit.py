"""Download and audit the verified SegFormer-B1 initialization source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoImageProcessor, SegformerConfig, SegformerForSemanticSegmentation


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", default="nvidia/segformer-b1-finetuned-ade-512-512")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    args.source_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_path = snapshot_download(repo_id=args.source_id, local_dir=args.source_dir, revision="main")
        info = HfApi().model_info(args.source_id, revision="main")
        config = SegformerConfig.from_pretrained(args.source_dir, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(args.source_dir, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(args.source_dir, local_files_only=True)
        model.eval()
        with torch.inference_mode():
            output = model(pixel_values=torch.zeros(1, 3, 512, 512)).logits
    except Exception as exc:
        payload = {
            "source_identifier": args.source_id,
            "source_gate": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "reason": "verified public checkpoint could not be obtained from Hugging Face and no local snapshot was available",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        raise SystemExit("ROUND 8 SOURCE CHECKPOINT BLOCKED")

    core_files = {}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"):
        path = args.source_dir / name
        if path.exists():
            core_files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

    payload = {
        "source_identifier": args.source_id,
        "resolved_commit": getattr(info, "sha", None),
        "snapshot_path": str(snapshot_path),
        "architecture": model.__class__.__name__,
        "model_type": config.model_type,
        "source_task": "ADE20K semantic segmentation",
        "num_labels": int(config.num_labels),
        "id2label_count": len(config.id2label),
        "encoder_depths": list(config.depths),
        "hidden_sizes": list(config.hidden_sizes),
        "sr_ratios": list(config.sr_ratios),
        "decoder_hidden_size": int(config.decoder_hidden_size),
        "processor_class": processor.__class__.__name__,
        "processor_size": processor.size,
        "logits_shape_512": list(output.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "core_files": core_files,
        "checks": {
            "architecture_is_segformer_b1": config.model_type == "segformer" and list(config.depths) == [2, 2, 2, 2] and list(config.hidden_sizes) == [64, 128, 320, 512],
            "semantic_segmentation_class": isinstance(model, SegformerForSemanticSegmentation),
            "source_has_150_classes": int(config.num_labels) == 150,
            "checkpoint_loadable": True,
            "logits_shape_valid": list(output.shape) == [1, 150, 128, 128],
        },
    }
    payload["source_gate"] = "PASS" if all(payload["checks"].values()) else "BLOCKED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["source_gate"] != "PASS":
        raise SystemExit("ROUND 8 SOURCE CHECKPOINT BLOCKED")


if __name__ == "__main__":
    main()
