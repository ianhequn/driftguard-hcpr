"""Select a cache cut from measured redundancy, latency, and stale damage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-summary", default="results/E02_real_video_redundancy/summary.csv")
    parser.add_argument("--sequential-summary", default="results/E03_sequential_stale/summary.csv")
    parser.add_argument("--latency", default="results/E03_sequential_stale/cache_cut_latency.csv")
    parser.add_argument("--state", default="reports/research_state.json")
    parser.add_argument("--output", default="results/E03_sequential_stale/cache_cut_selection.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real_rows = _read(PROJECT_ROOT / args.real_summary)
    sequential_rows = _read(PROJECT_ROOT / args.sequential_summary)
    latency_rows = _read(PROJECT_ROOT / args.latency)
    real = {
        (int(row["stage"]), int(row["offset"])): row
        for row in real_rows
        if int(row["block_grid"]) == 4
    }
    sequential = {
        (int(row["cut"]), int(row["offset"])): row
        for row in sequential_rows
    }
    latency = {
        row["cut"]: float(row["mean_latency_ms"])
        for row in latency_rows
    }
    full_latency = latency["full"]
    rows = []
    for cut in range(1, 5):
        real_k1 = real[(cut, 1)]
        real_k2 = real[(cut, 2)]
        seq_k1 = sequential[(cut, 1)]
        seq_k2 = sequential[(cut, 2)]
        damage_k1 = float(seq_k1["logit_relative_l2_mean"])
        damage_k2 = float(seq_k2["logit_relative_l2_mean"])
        saved = full_latency - latency[f"cut_{cut}"]
        utility = saved / (damage_k1 + damage_k2 + 1e-6)
        rows.append(
            {
                "cut": cut,
                "saved_latency_ms": saved,
                "speedup": full_latency / latency[f"cut_{cut}"],
                "cosine_k1": float(real_k1["cosine_mean"]),
                "cosine_k2": float(real_k2["cosine_mean"]),
                "damage_k1": damage_k1,
                "damage_k2": damage_k2,
                "damage_k4": float(
                    sequential[(cut, 4)]["logit_relative_l2_mean"]
                ),
                "utility": utility,
                "priority": 0,
            }
        )
    ranked = sorted(rows, key=lambda row: row["utility"], reverse=True)
    for priority, row in enumerate(ranked, start=1):
        row["priority"] = priority
    selected = ranked[0]["cut"]

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["cut"]))

    state_path = PROJECT_ROOT / args.state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["selected_cache_cut"] = selected
    state["current_phase"] = "3"
    state["last_update"] = "2026-08-23T00:00:00+08:00"
    state["notes"] = state.get("notes", []) + [
        f"Phase 2.8 selected cache cut {selected} from measured utility."
    ]
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"selected_cache_cut={selected}")
    print(f"output={output_path.resolve()}")


if __name__ == "__main__":
    main()
