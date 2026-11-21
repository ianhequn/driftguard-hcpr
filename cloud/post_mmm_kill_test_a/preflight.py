from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

from common import sha256_file


EXPECTED = {
    "B0": "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b",
    "B1": "892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=("B0", "B1"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    weight = args.model_dir / "model.safetensors"
    payload = {
        "protocol_id": "POSTMMM-R1",
        "model_key": args.model_key,
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "repo_root": str(args.repo_root),
        "dry_run": args.dry_run,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": None,
        "checkpoint_sha256": None,
        "cuda": None,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not weight.is_file() or not args.dataset_root.is_dir() or not args.repo_root.is_dir():
        raise SystemExit("missing required path")
    digest = sha256_file(weight)
    if digest != EXPECTED[args.model_key]:
        raise SystemExit(f"checkpoint hash mismatch for {args.model_key}: {digest}")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    payload["checkpoint_sha256"] = digest
    payload["cuda"] = torch.version.cuda
    payload["git_commit"] = subprocess.check_output(["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
