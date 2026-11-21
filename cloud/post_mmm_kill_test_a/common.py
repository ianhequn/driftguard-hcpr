"""Shared fail-closed checks for the POSTMMM-R1 cloud bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-key", choices=("B0", "B1"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def validate_common(args: argparse.Namespace) -> dict:
    if args.dry_run == args.execute:
        raise SystemExit("choose exactly one of --dry-run or --execute")
    manifest = load_manifest()
    if manifest["protocol_id"] != "POSTMMM-R1":
        raise SystemExit("protocol manifest mismatch")
    if not args.execute:
        return manifest
    if not args.model_dir.is_dir() or not args.dataset_root.is_dir() or not args.repo_root.is_dir():
        raise SystemExit("missing model, dataset, or repository root")
    weight = args.model_dir / "model.safetensors"
    if not weight.is_file():
        raise SystemExit("model.safetensors is required; do not substitute ADE/base weights")
    digest = sha256_file(weight)
    if len(digest) != 64:
        raise SystemExit("invalid checkpoint SHA256")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for --execute") from exc
    if not torch.cuda.is_available():
        raise SystemExit("POSTMMM-R1 requires CUDA; refusing CPU experiment")
    return manifest


def print_plan(kind: str, args: argparse.Namespace, manifest: dict) -> None:
    plan = {
        "kind": kind,
        "protocol_id": manifest["protocol_id"],
        "model_key": args.model_key,
        "model_dir": str(args.model_dir),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "fixed_grid": manifest["fixed_grid"],
        "hcpr_quantiles": manifest["hcpr_quantiles"],
        "timing": manifest["timing"],
        "execution": "not_started" if args.dry_run else "guarded",
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
