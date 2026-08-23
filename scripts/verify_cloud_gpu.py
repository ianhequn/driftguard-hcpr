"""Verify a CUDA environment for later cloud GPU experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.segformer import load_segformer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Hugging Face model id or a local model directory.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print(f"torch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. Check the NVIDIA driver, CUDA wheel, and runtime.")
        return 1

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU name: {properties.name}")
    print(f"VRAM: {properties.total_memory / (1024**3):.2f} GB")
    print(f"device capability: {properties.major}.{properties.minor}")

    try:
        processor, model = load_segformer(
            args.model,
            device=device,
            local_files_only=args.local_files_only,
        )
        del processor
        pixel_values = torch.randn(1, 3, 128, 128, device=device)
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values, return_dict=True)
        torch.cuda.synchronize(device)
        print(f"smoke logits shape: {tuple(outputs.logits.shape)}")
        print("CLOUD GPU READY")
        return 0
    except Exception as error:
        print(f"ERROR: SegFormer CUDA smoke test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
