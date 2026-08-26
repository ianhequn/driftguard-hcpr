"""Freeze the selected Round 7.6 Restart checkpoint as a VSPW-124 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.training.vspw_segformer import (
    IGNORE_INDEX,
    NUM_CLASSES,
    build_vspw_segformer,
    load_vspw_class_names,
)


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--architecture", default="SegFormer-B0")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty frozen directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_vspw_class_names(args.dataset_root)
    model, initialization = build_vspw_segformer(args.source, class_names)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint reload mismatch: {incompatible}")
    model.eval()
    model.save_pretrained(args.output_dir, safe_serialization=True)

    preprocessor = {
        "do_resize": True,
        "size": {"height": 512, "width": 512},
        "resample": 2,
        "do_rescale": True,
        "rescale_factor": 0.00392156862745098,
        "do_normalize": True,
        "image_mean": MEAN,
        "image_std": STD,
        "do_reduce_labels": False,
        "label_ignore_index": IGNORE_INDEX,
    }
    write_json(args.output_dir / "preprocessor_config.json", preprocessor)
    write_json(
        args.output_dir / "class_names.json",
        {"num_classes": NUM_CLASSES, "classes": class_names},
    )
    write_json(
        args.output_dir / "label_contract.json",
        {
            "num_classes": NUM_CLASSES,
            "valid_ids": [0, NUM_CLASSES - 1],
            "ignore_index": IGNORE_INDEX,
            "normalization": "official normalization then out-of-range to 255",
            "raw253": "253 -> 252 -> 255 -> excluded",
        },
    )
    write_json(
        args.output_dir / "source_provenance.json",
        {
            "initialization_source": str(args.source),
            "initialization_model_sha256": sha256_file(args.source / "model.safetensors"),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "loaded_parameter_tensors": initialization.loaded_parameter_tensors,
            "missing_keys": initialization.missing_keys,
            "unexpected_keys": initialization.unexpected_keys,
            "reinitialized_parameters": initialization.reinitialized_parameters,
        },
    )
    if args.training_config:
        shutil.copy2(args.training_config, args.output_dir / "training_config.yaml")
    else:
        (args.output_dir / "training_config.yaml").write_text(
            f"architecture: {args.architecture}\nnum_classes: 124\nimage_size: 512\n"
            "precision: bf16\nbatch_size: 8\nseed: 42\n",
            encoding="utf-8",
        )
    selection = {}
    if args.selection and args.selection.exists():
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
    write_json(
        args.output_dir / "freeze_summary.json",
        {
            "architecture": args.architecture,
            "num_labels": int(model.config.num_labels),
            "selected_step": selection.get("step", state.get("step")),
            "selected_validation_miou": selection.get("miou"),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "model_sha256": sha256_file(args.output_dir / "model.safetensors"),
            "class_alignment_count": len(class_names),
            "encoder_frozen": any(not parameter.requires_grad for parameter in model.parameters()),
        },
    )
    (args.output_dir / "sha256.txt").write_text(
        f"{sha256_file(args.output_dir / 'model.safetensors')}  model.safetensors\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads((args.output_dir / "freeze_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
