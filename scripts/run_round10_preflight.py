"""Record the Round-10 hardware and frozen-contract preflight."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "R10_E00_preflight"


def command_output(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    nvidia = command_output(["nvidia-smi"])
    python_gpu: dict[str, object]
    try:
        import torch
        python_gpu = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - environment-dependent
        python_gpu = {"available": False, "error": repr(exc)}
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "nvidia_smi": nvidia,
        "python_gpu": python_gpu,
        "required_gpu": "NVIDIA RTX 4090D",
        "formal_latency_gate": "PASS" if python_gpu.get("gpu_name") == "NVIDIA GeForce RTX 4090 D" else "BLOCKED",
        "cloud_probe": "ssh driftguard-gpu returned Connection refused",
        "decision": "READY for formal Round-10 experiments" if python_gpu.get("gpu_name") == "NVIDIA GeForce RTX 4090 D" else "STOP formal Round-10 latency and GT experiments until one RTX 4090D is available",
    }
    (OUT / "preflight.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
