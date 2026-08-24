"""Build Round 5 Amdahl and break-even tables from measured latency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SELECTOR_MS = {
    0.25: 0.12554203858599067,
    0.375: 1.2321898695081472,
    0.5: 0.11443394608795643,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="results/R5_E00_compute_economics/summary.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/R5_E00_compute_economics",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    full_ms = float(profile["direct_full_mean_ms"])
    stage4_ms = float(profile["stage4_mean_ms"])
    other_ms = full_ms - stage4_ms
    rows = []
    for budget, selector_ms in SELECTOR_MS.items():
        ideal_partial_ms = budget * stage4_ms
        maximum_partial_ms = stage4_ms - selector_ms
        ideal_total_ms = other_ms + selector_ms + ideal_partial_ms
        rows.append(
            {
                "budget": budget,
                "full_stage4_ms": stage4_ms,
                "selector_ms": selector_ms,
                "ideal_partial_ms": ideal_partial_ms,
                "maximum_partial_ms_for_any_speedup": maximum_partial_ms,
                "ideal_total_ms": ideal_total_ms,
                "ideal_system_speedup": full_ms / ideal_total_ms,
            }
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "break_even_table.csv", rows)
    summary = {
        "formula_note": (
            "Physical latency uses T_ideal = T_other + r*T_stage4 + "
            "T_selector. The minus signs in the task brief are treated as a "
            "typographical error because component latencies add."
        ),
        "full_ms": full_ms,
        "stage4_ms": stage4_ms,
        "other_ms": other_ms,
        "stage4_fraction": stage4_ms / full_ms,
        "budgets": rows,
        "early_stop_a": {
            "threshold_speedup": 1.03,
            "core_budgets_below_threshold": sum(
                row["ideal_system_speedup"] < 1.03 for row in rows
            ),
            "triggered": sum(
                row["ideal_system_speedup"] < 1.03 for row in rows
            )
            >= 2,
        },
    }
    (output_root / "economics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
