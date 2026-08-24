"""Finalize Round 5 artifacts after an authorized early stop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CORE = (0.25, 0.375, 0.5)
SELECTOR_MS = {
    0.25: 0.12554203858599067,
    0.375: 1.2321898695081472,
    0.5: 0.11443394608795643,
}
OFFLINE_AFVT = {
    0.25: (0.017958917, 0.004282, 0.011275),
    0.375: (0.012978835, 0.002393, 0.008814),
    0.5: (0.010579057, 0.001491, 0.005988),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--economics",
        default="results/R5_E00_compute_economics/economics_summary.json",
    )
    parser.add_argument(
        "--partial",
        default="results/R5_E03_approx_partial/summary.json",
    )
    parser.add_argument(
        "--output-root",
        default="results/R5_E05_accuracy_latency",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def placeholder_plot(path: Path, title: str, reason: str) -> None:
    plt.figure(figsize=(7.5, 5.0))
    plt.text(
        0.5,
        0.5,
        f"{title}\n\n{reason}",
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    economics = json.loads(Path(args.economics).read_text(encoding="utf-8"))
    partial = json.loads(Path(args.partial).read_text(encoding="utf-8"))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    full_ms = float(economics["full_ms"])
    stage4_economics_ms = float(economics["stage4_ms"])
    other_ms = float(economics["other_ms"])
    full_stage4_isolated_ms = float(partial["full_stage4"]["mean_ms"])
    partial_by_budget = {
        float(row["budget"]): row for row in partial["partial"]
    }

    economics_rows = []
    compute_rows = []
    for budget in CORE:
        row = partial_by_budget[budget]
        selector_ms = SELECTOR_MS[budget]
        ideal_partial = budget * stage4_economics_ms
        ideal_total = other_ms + selector_ms + ideal_partial
        actual_total = other_ms + selector_ms + float(row["mean_ms"])
        ideal_speedup = full_ms / ideal_total
        actual_speedup = full_ms / actual_total
        sparse_efficiency = (
            (full_ms - actual_total) / (full_ms - ideal_total)
            if full_ms != ideal_total
            else float("nan")
        )
        economics_rows.append(
            {
                "budget": budget,
                "ideal_stage4_latency": ideal_partial,
                "actual_stage4_latency": float(row["mean_ms"]),
                "ideal_total_latency": ideal_total,
                "actual_total_latency": actual_total,
                "ideal_speedup": ideal_speedup,
                "actual_speedup": actual_speedup,
                "sparse_efficiency": sparse_efficiency,
                "latency_status": "component_composed_projection",
            }
        )
        compute_rows.append(
            {
                "video": "R5_stage4_microbenchmark",
                "frame": 0,
                "budget": budget,
                "logical_refresh_blocks": row["logical_refresh_blocks"],
                "physical_tokens": row["physical_tokens"],
                "full_tokens": row["full_tokens"],
                "physical_compute_ratio": row["physical_compute_ratio"],
                "halo_tokens": row["halo_tokens"],
                "cache_reads": 2 * (
                    2 * row["full_tokens"] + 9 * row["physical_tokens"]
                ),
                "cache_writes": row["physical_tokens"] * 12,
            }
        )

    write_csv(output_root / "ideal_vs_actual.csv", economics_rows)
    write_csv(
        Path("results/R5_E03_approx_partial") / "partial_compute_stats.csv",
        compute_rows,
    )

    main_rows = [
        {
            "Method": "Full",
            "Budget": 1.0,
            "Logical Refresh": 1.0,
            "Physical Compute": 1.0,
            "Latency": full_ms,
            "P95": 25.455249,
            "FPS": 1000.0 / full_ms,
            "Logit L2": 0.0,
            "KL": 0.0,
            "Pixel Disagreement": 0.0,
            "Status": "measured CUDA-resident full baseline",
        }
    ]
    for method in ("Age Real Partial", "AFVT Real Partial"):
        for budget in CORE:
            main_rows.append(
                {
                    "Method": method,
                    "Budget": budget,
                    "Logical Refresh": budget,
                    "Physical Compute": "",
                    "Latency": "",
                    "P95": "",
                    "FPS": "",
                    "Logit L2": "",
                    "KL": "",
                    "Pixel Disagreement": "",
                    "Status": "NOT RUN - Early Stop B before model integration",
                }
            )
    for budget in CORE:
        l2, kl, pixel = OFFLINE_AFVT[budget]
        main_rows.append(
            {
                "Method": "Offline AFVT Reference",
                "Budget": budget,
                "Logical Refresh": budget,
                "Physical Compute": "",
                "Latency": "",
                "P95": "",
                "FPS": "",
                "Logit L2": l2,
                "KL": kl,
                "Pixel Disagreement": pixel,
                "Status": "Round 3 offline reference; not deployable latency",
            }
        )
    write_csv(output_root / "real_partial_main_table.csv", main_rows)

    ratios = np.asarray([row["budget"] for row in economics_rows]) * 100
    ideal_speedups = np.asarray(
        [row["ideal_speedup"] for row in economics_rows]
    )
    actual_speedups = np.asarray(
        [row["actual_speedup"] for row in economics_rows]
    )
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(ratios, ideal_speedups, marker="o", label="Ideal linear system")
    plt.plot(
        ratios,
        actual_speedups,
        marker="o",
        label="Measured Stage-4 component projection",
    )
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xlabel("Refresh budget (%)")
    plt.ylabel("Speedup vs full system")
    plt.title("Ideal vs measured partial speedup")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "ideal_vs_measured_speedup.png", dpi=180)
    plt.close()

    partial_latency = np.asarray(
        [partial_by_budget[budget]["mean_ms"] for budget in CORE]
    )
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(ratios, partial_latency, marker="o", label="Cached-context partial")
    plt.axhline(
        full_stage4_isolated_ms,
        color="black",
        linestyle="--",
        label="Full Stage 4",
    )
    plt.xlabel("Logical refresh budget (%)")
    plt.ylabel("Measured Stage-4 latency (ms)")
    plt.title("Refresh ratio vs real Stage-4 latency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "refresh_ratio_vs_real_latency.png", dpi=180)
    plt.close()

    reason = (
        "NOT RUN: Early Stop B. Every core Route-A Stage-4 measurement was "
        "slower than full Stage 4, so AFVT/ Age model integration was stopped."
    )
    placeholder_plot(
        output_root / "accuracy_latency_pareto.png",
        "Accuracy / latency Pareto",
        reason,
    )
    placeholder_plot(
        output_root / "refresh_ratio_vs_task_damage.png",
        "Refresh ratio vs task damage",
        reason,
    )

    (output_root / "NOT_RUN.txt").write_text(reason + "\n", encoding="utf-8")
    (Path("results/R5_E04_real_integration") / "README.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (Path("results/R5_E04_real_integration") / "README.md").write_text(
        "NOT RUN - Early Stop B: Route A Stage-4 latency exceeded full Stage 4.\n",
        encoding="utf-8",
    )
    (Path("results/R5_E06_profiling") / "README.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (Path("results/R5_E06_profiling") / "README.md").write_text(
        "Stage-4 profiling artifacts are stored in R5_E00 and R5_E03.\n"
        "End-to-end sequential profiling was not run after Early Stop B.\n",
        encoding="utf-8",
    )

    state = {
        "round": 5,
        "status": "completed",
        "scheduler": "AFVT",
        "cache_cut": 4,
        "compute_economics": {
            "full_ms": full_ms,
            "stage4_ms": stage4_economics_ms,
            "stage4_fraction": stage4_economics_ms / full_ms,
            "early_stop_a": False,
        },
        "exact_partial_feasible": "computationally_pointless",
        "approx_partial_feasible": "feasible",
        "selected_partial_route": "Cached-Context Approximate Partial",
        "real_speedup": {
            "measured_end_to_end": False,
            "early_stop_b": True,
            "stage4_full_isolated_ms": full_stage4_isolated_ms,
            "core_partial_stage4_ms": {
                str(budget): float(partial_by_budget[budget]["mean_ms"])
                for budget in CORE
            },
        },
        "quality_result": "not_run_early_stop_b",
        "final_decision": "ROUND 5 = REAL PARTIAL NO-GO",
    }
    Path("reports/round5_state.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
