"""Recover and verify the predeclared SegFormer-B1 source through a mirror."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoImageProcessor, SegformerConfig, SegformerForSemanticSegmentation


SOURCE_ID = "nvidia/segformer-b1-finetuned-ade-512-512"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", default=SOURCE_ID)
    parser.add_argument("--download-endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    try:
        os.environ["HF_ENDPOINT"] = args.download_endpoint
        from huggingface_hub import HfApi, snapshot_download

        args.source_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(repo_id=args.source_id, local_dir=args.source_dir, revision="main")
        try:
            info = HfApi(endpoint=args.download_endpoint).model_info(args.source_id, revision="main")
            resolved_revision = getattr(info, "sha", None) or "unavailable"
        except Exception:
            resolved_revision = "unavailable"

        conversion = None
        source_weight = args.source_dir / "model.safetensors"
        original_weight = "model.safetensors"
        if not source_weight.is_file():
            pytorch_weight = args.source_dir / "pytorch_model.bin"
            if not pytorch_weight.is_file():
                raise RuntimeError("official snapshot contains neither model.safetensors nor pytorch_model.bin")
            state = torch.load(pytorch_weight, map_location="cpu", weights_only=True)
            save_file(state, source_weight, metadata={"format": "pt", "source": "official pytorch_model.bin"})
            original_weight = "pytorch_model.bin"
            conversion = {
                "from": "pytorch_model.bin",
                "to": "model.safetensors",
                "method": "safetensors.torch.save_file",
                "source_sha256": sha256_file(pytorch_weight),
                "converted_sha256": sha256_file(source_weight),
            }

        required = ("config.json", "model.safetensors", "preprocessor_config.json")
        missing = [name for name in required if not (args.source_dir / name).is_file()]
        zero_byte = [name for name in required if (args.source_dir / name).is_file() and (args.source_dir / name).stat().st_size == 0]
        suspicious = [str(path.relative_to(args.source_dir)) for path in args.source_dir.rglob("*") if path.is_file() and (path.name.endswith(".partial") or path.name.endswith(".incomplete"))]
        if missing or zero_byte or suspicious:
            raise RuntimeError(f"incomplete source snapshot: missing={missing}, zero_byte={zero_byte}, suspicious={suspicious}")

        config = SegformerConfig.from_pretrained(args.source_dir, local_files_only=True)
        processor = AutoImageProcessor.from_pretrained(args.source_dir, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(args.source_dir, local_files_only=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.inference_mode():
            logits = model(pixel_values=torch.zeros(1, 3, 512, 512, device=device)).logits
        finite = bool(torch.isfinite(logits).all().item())

        hashes = {}
        files_to_hash = list(required)
        if (args.source_dir / "pytorch_model.bin").is_file():
            files_to_hash.append("pytorch_model.bin")
        for name in files_to_hash:
            path = args.source_dir / name
            hashes[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        write_json(args.output_root / "source_hashes.json", hashes)

        checks = {
            "identifier_verified": args.source_id == SOURCE_ID,
            "architecture_verified": config.model_type == "segformer" and list(config.depths) == [2, 2, 2, 2] and list(config.hidden_sizes) == [64, 128, 320, 512],
            "source_num_labels_verified": int(config.num_labels) == 150,
            "weights_complete": not missing and not zero_byte and not suspicious,
            "fresh_load_pass": True,
            "forward_smoke_pass": finite and list(logits.shape) == [1, 150, 128, 128],
        }
        provenance = {
            "source_identifier": args.source_id,
            "download_endpoint": args.download_endpoint,
            "resolved_revision": resolved_revision,
            "snapshot_path": str(snapshot_path),
            "architecture": "SegFormer-B1",
            "model_type": config.model_type,
            "source_task": "ADE20K semantic segmentation",
            "source_num_labels": int(config.num_labels),
            "encoder_depths": list(config.depths),
            "hidden_sizes": list(config.hidden_sizes),
            "attention_heads": list(config.num_attention_heads),
            "sr_ratios": list(config.sr_ratios),
            "decoder_hidden_size": int(config.decoder_hidden_size),
            "processor_class": processor.__class__.__name__,
            "processor_size": processor.size,
            "logits_shape": list(logits.shape),
            "logits_finite": finite,
            "device": str(device),
            "original_weight_file": original_weight,
            "weight_conversion": conversion,
            "download_time": started,
            "file_hashes": hashes,
            "checks": checks,
            "source_gate": "PASS" if all(checks.values()) else "BLOCKED",
        }
        write_json(args.output_root / "source_provenance.json", provenance)
        print(json.dumps(provenance, indent=2))
        if provenance["source_gate"] != "PASS":
            raise SystemExit("ROUND 8 SOURCE RECOVERY BLOCKED")
    except SystemExit:
        raise
    except Exception as exc:
        payload = {
            "source_identifier": args.source_id,
            "download_endpoint": args.download_endpoint,
            "download_time": started,
            "source_gate": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(args.output_root / "source_provenance.json", payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit("ROUND 8 SOURCE RECOVERY BLOCKED")


if __name__ == "__main__":
    main()
