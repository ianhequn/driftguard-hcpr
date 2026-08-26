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


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(file_path.read_bytes())
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
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "dataset": {"split": "val", "videos": videos, "frames": len(dataset), "classes": 124, "ignore": 255},
        "precision": "bf16",
        "resolution": [512, 512],
        "batch_size": 1,
        "checkpoints": {"B0": {"path": str(args.b0_dir), "sha256": sha256_directory(args.b0_dir)},
                         "B1": {"path": str(args.b1_dir), "sha256": sha256_directory(args.b1_dir)}},
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
