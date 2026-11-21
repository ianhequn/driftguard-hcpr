"""Read-only inventory and SHA256 audit for historical model checkpoints.

The scanner intentionally does not open image files, modify source trees, extract
archives, or follow directory names that are known to contain VSPW frames/masks.
It records archive members without extraction so the inventory itself is safe to
run on a workstation containing large datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable


EXPECTED_HASHES = {
    "EXACT_B0": "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b",
    "EXACT_B1": "892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb",
}
KNOWN_BASE_HASHES = {
    "6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9",
    "4dbca19129b1373ca6bce75d68834a246d6b1e5e5202886183cd284d2f41abef",
    "ac49bf62c8c0578dcf17ff048930b4f886ab0bc080d23f5591669afad5ddf7da",
}

MODEL_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
    ".7z",
}
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z")
SUSPICIOUS_TOKENS = re.compile(
    r"(?:segformer|vspw|driftguard|checkpoint|training_state|freeze|"
    r"model\.safetensors|pytorch_model|r(?:7|8|10|11)|b[01])",
    re.IGNORECASE,
)
ARCHIVE_MEMBER_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "data",
    "datasets",
    "VSPW",
    "VSPW_480p",
    "mask",
    "masks",
    "frames",
    "images",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_hash(
    digest: str | None,
    path_text: str,
    *,
    is_archive: bool = False,
) -> str:
    """Classify a hashed candidate without inferring a model from size alone."""

    normalized = (digest or "").lower()
    for label, expected in EXPECTED_HASHES.items():
        if normalized == expected:
            return label
    if normalized in KNOWN_BASE_HASHES:
        return "BASE_MODEL_ONLY"
    lower_path = path_text.lower()
    if "finetuned-ade" in lower_path or "ade512" in lower_path:
        return "BASE_MODEL_ONLY"
    if is_archive:
        return "UNKNOWN_CANDIDATE"
    if digest is not None:
        return "HASH_MISMATCH"
    return "UNKNOWN_CANDIDATE"


def is_excluded(path: Path, excluded_names: set[str]) -> bool:
    return any(part in excluded_names for part in path.parts)


def is_candidate(path: Path) -> bool:
    name = path.name.lower()
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() not in MODEL_EXTENSIONS and suffixes not in ARCHIVE_SUFFIXES:
        return False
    return bool(SUSPICIOUS_TOKENS.search(str(path)) or name in {"model.safetensors", "pytorch_model.bin"})


def suspicious_archive_member(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(ARCHIVE_MEMBER_SUFFIXES) and bool(SUSPICIOUS_TOKENS.search(lower))


def list_archive_members(path: Path) -> tuple[list[str], str | None]:
    """Return only checkpoint-like members; never extract an archive."""

    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                members = sorted(
                    name for name in archive.namelist() if suspicious_archive_member(name)
                )
            return members, None
        if path.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(path, "r:*") as archive:
                members = sorted(
                    item.name for item in archive.getmembers() if suspicious_archive_member(item.name)
                )
            return members, None
        return [], "unsupported_archive_format"
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def iter_candidates(roots: Iterable[Path], excluded_names: set[str]) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.resolve()
        if root.is_file():
            paths = [root]
        elif root.exists():
            def walk() -> Iterable[Path]:
                for current, directory_names, file_names in os.walk(root, followlinks=False):
                    directory_names[:] = [name for name in directory_names if name not in excluded_names]
                    current_path = Path(current)
                    for name in file_names:
                        yield current_path / name

            paths = walk()
        else:
            continue
        for path in paths:
            if is_excluded(path, excluded_names) or not is_candidate(path):
                continue
            key = os.path.normcase(str(path.resolve()))
            if key not in seen:
                seen.add(key)
                yield path.resolve()


def inspect_candidate(path: Path, max_hash_bytes: int) -> dict:
    stat = path.stat()
    is_archive = path.name.lower().endswith(ARCHIVE_SUFFIXES)
    digest = None
    hash_status = None
    if stat.st_size <= max_hash_bytes:
        try:
            digest = sha256_file(path)
        except OSError as exc:
            hash_status = f"{type(exc).__name__}: {exc}"
    else:
        hash_status = "skipped_over_max_hash_bytes"
    members, archive_error = list_archive_members(path) if is_archive else ([], None)
    classification = classify_hash(digest, str(path), is_archive=is_archive)
    if classification == "UNKNOWN_CANDIDATE" and any("finetuned-ade" in m.lower() for m in members):
        classification = "BASE_MODEL_ONLY"
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": digest,
        "hash_status": hash_status,
        "classification": classification,
        "archive_members": members,
        "archive_error": archive_error,
    }


def build_inventory(roots: list[Path], min_size: int, max_size: int, max_hash_bytes: int, excluded_names: set[str]) -> dict:
    entries = []
    for path in iter_candidates(roots, excluded_names):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if min_size <= size <= max_size:
            entries.append(inspect_candidate(path, max_hash_bytes))
    entries.sort(key=lambda item: item["path"].lower())
    counts: dict[str, int] = {}
    for item in entries:
        label = item["classification"]
        counts[label] = counts.get(label, 0) + 1
    return {
        "schema_version": 1,
        "purpose": "read-only historical SegFormer B0/B1 checkpoint recovery audit",
        "roots": [str(root.resolve()) for root in roots],
        "expected_hashes": EXPECTED_HASHES,
        "known_base_hashes": sorted(KNOWN_BASE_HASHES),
        "excluded_directory_names": sorted(excluded_names),
        "filters": {
            "min_size_bytes": min_size,
            "max_size_bytes": max_size,
            "max_hash_bytes": max_hash_bytes,
            "extensions": sorted(MODEL_EXTENSIONS),
        },
        "counts": counts,
        "entries": entries,
    }


def render_markdown(inventory: dict) -> str:
    entries = inventory["entries"]
    exact = [item for item in entries if item["classification"].startswith("EXACT_")]
    lines = [
        "# Checkpoint inventory",
        "",
        "只读扫描；未修改源文件、未提取压缩包、未读取 VSPW 图像/掩码。",
        "",
        f"- roots: {', '.join(inventory['roots'])}",
        f"- candidates: {len(entries)}",
        f"- exact expected checkpoints: {len(exact)}",
        "- expected B0: `8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b`",
        "- expected B1: `892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb`",
        "",
        "| classification | path | bytes | SHA256 | archive members |",
        "|---|---|---:|---|---|",
    ]
    for item in entries:
        members = "<br>".join(item["archive_members"]) or "-"
        lines.append(
            f"| {item['classification']} | `{item['path']}` | {item['size_bytes']} | "
            f"`{item['sha256'] or item['hash_status'] or '-'}` | {members} |"
        )
    if not entries:
        lines.append("| - | no candidates | - | - | - |")
    lines += ["", "## Interpretation", "", "EXACT_B0/EXACT_B1 require a full-file SHA256 match; filename or size is never sufficient.", ""]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--min-size", type=int, default=1024)
    parser.add_argument("--max-size", type=int, default=5 * 1024**3)
    parser.add_argument("--max-hash-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--exclude-dir", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="scan and print counts without writing output files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded = DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)
    inventory = build_inventory(args.roots, args.min_size, args.max_size, args.max_hash_bytes, excluded)
    payload = json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json and not args.dry_run:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    if args.output_md and not args.dry_run:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(inventory), encoding="utf-8")
    print(json.dumps({"dry_run": args.dry_run, "counts": inventory["counts"], "total": len(inventory["entries"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
