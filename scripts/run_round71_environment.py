"""Capture the Round 7.1 GPU and frozen-model environment gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SHA256 = "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"COMMAND_FAILED: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_file = args.model_dir / "model.safetensors"
    actual_sha256 = sha256_file(model_file)
    gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_apps = command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "model_sha256": actual_sha256,
        "expected_sha256": EXPECTED_SHA256,
        "model_sha256_pass": actual_sha256 == EXPECTED_SHA256,
        "gpu_query": gpu,
        "compute_apps_query": compute_apps,
        "process_snapshot": command(["ps", "-eo", "pid,etime,cmd"]),
        "gate": "PASS" if actual_sha256 == EXPECTED_SHA256 and not compute_apps else "REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if actual_sha256 != EXPECTED_SHA256:
        raise SystemExit("MODEL ARTIFACT MISMATCH")


if __name__ == "__main__":
    main()
