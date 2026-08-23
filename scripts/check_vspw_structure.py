"""Inspect a local VSPW directory without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.datasets.vspw import inspect_vspw_structure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/VSPW")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = inspect_vspw_structure(PROJECT_ROOT / args.root, split=args.split)
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        output = PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
