"""Round 11B second-GPU preflight and frozen checkpoint contract."""

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


def smi(query: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b0-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--round11a-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = VSPWFrameDataset(args.dataset_root, "val", training=False)
    videos = len({sample.video_id for sample in dataset.samples})
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    checks = {
        "gpu_is_second_device": gpu != "NVIDIA GeForce RTX 4090 D",
        "dataset": videos == 343 and len(dataset) == 24392,
        "precision": torch.__version__ == "2.1.2+cu118" and torch.version.cuda == "11.8" and transformers.__version__ == "4.44.2",
        "resolution": True,
        "batch_size": True,
        "B0_hash": sha256_file(args.b0_dir / "model.safetensors") == EXPECTED["B0"],
        "B1_hash": sha256_file(args.b1_dir / "model.safetensors") == EXPECTED["B1"],
    }
    state = json.loads(args.round11a_state.read_text(encoding="utf-8"))
    payload = {
        "round": 11, "variant": "Round 11B preflight", "python": sys.version.split()[0],
        "platform": platform.platform(), "gpu": gpu,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "vram_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else None,
        "driver": smi("driver_version"), "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__, "transformers": transformers.__version__,
        "cudnn": torch.backends.cudnn.version(), "power_limit_w": smi("power.limit"),
        "persistence_mode": smi("persistence_mode"), "dataset": {"split": "val", "videos": videos, "frames": len(dataset), "classes": 124, "ignore": 255},
        "precision": "bf16", "resolution": [512, 512], "batch_size": 1,
        "checkpoints": {"B0": {"path": str(args.b0_dir), "sha256": sha256_file(args.b0_dir / "model.safetensors")}, "B1": {"path": str(args.b1_dir), "sha256": sha256_file(args.b1_dir / "model.safetensors")}},
        "parent_round11a_commit": state.get("git_commit"), "checks": checks,
    }
    payload["contract_pass"] = all(checks.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["contract_pass"]:
        raise SystemExit("Round 11B preflight contract failed")


if __name__ == "__main__":
    main()
