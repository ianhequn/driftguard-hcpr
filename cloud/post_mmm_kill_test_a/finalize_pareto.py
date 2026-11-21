"""Finalize only artifact rows with one exact model/device/protocol contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def nondominated(rows: list[dict]) -> list[dict]:
    result = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["miou_damage_pp"] <= candidate["miou_damage_pp"]
            and other["speedup"] >= candidate["speedup"]
            and (
                other["miou_damage_pp"] < candidate["miou_damage_pp"]
                or other["speedup"] > candidate["speedup"]
            )
            for other in rows
        )
        if not dominated:
            result.append(candidate)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    if not isinstance(rows, list) or not rows:
        raise SystemExit("input must contain a non-empty rows list")
    keys = {tuple(row.get(key) for key in ("model", "device", "protocol_id", "checkpoint_sha256", "dataset_manifest_sha256")) for row in rows}
    if len(keys) != 1 or any(None in key for key in next(iter(keys))):
        raise SystemExit("mixed or incomplete model/device/protocol/checkpoint/dataset contract")
    output = {
        "protocol_id": "POSTMMM-R1",
        "contract": list(next(iter(keys))),
        "rows": rows,
        "pareto_rows": nondominated(rows),
        "status": "valid_same_contract",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
