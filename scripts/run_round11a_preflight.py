"""Verify the Round 11A hardware, software, checkpoint, and data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from driftguard.training.vspw_segformer import VSPWFrameDataset  # noqa: E402


EXPECTED = {
    "B0": "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b",
    "B1": "892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    videos = len({sample.video_id for sample in dataset.samples})
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        driver = "unavailable"
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "dataset": {"split": "val", "videos": videos, "frames": len(dataset), "classes": 124, "ignore": 255},
        "precision": "bf16",
        "resolution": [512, 512],
        "batch_size": 1,
        "checkpoints": {"B0": {"path": str(args.b0_dir), "sha256": sha256_file(args.b0_dir / "model.safetensors")},
                         "B1": {"path": str(args.b1_dir), "sha256": sha256_file(args.b1_dir / "model.safetensors")}},
    }
    payload["contract_pass"] = (
        payload["gpu"] == "NVIDIA GeForce RTX 4090 D"
        and payload["checkpoints"]["B0"]["sha256"] == EXPECTED["B0"]
        and payload["checkpoints"]["B1"]["sha256"] == EXPECTED["B1"]
        and videos == 343 and len(dataset) == 24392
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["contract_pass"]:
        raise SystemExit("Round 11A preflight contract failed")


if __name__ == "__main__":
    main()
