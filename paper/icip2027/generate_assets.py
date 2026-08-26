"""Generate the final Round 10 manuscript assets from frozen artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
R10 = ROOT / "results" / "R10_E06_analysis"
REPORTS = ROOT / "reports"


def copy_round10_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for source, target in {
        "R10_F01_boundary_pareto.pdf": "round10_boundary_pareto.pdf",
        "R10_F03_runtime_opportunity_vs_speed.pdf": "round10_runtime_opportunity.pdf",
    }.items():
        shutil.copyfile(R10 / source, FIGURES / target)


def method_overview() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 2.55), sharex=True)
    stages = ["F1", "F2", "F3", "F4"]
    boundaries = [("S4", 3), ("S3-S4", 2), ("S2-S4", 1)]
    for ax, (name, fresh_count) in zip(axes, boundaries):
        ax.set_xlim(-0.5, 5.4); ax.set_ylim(-0.34, 0.45); ax.axis("off")
        ax.text(-0.45, 0.05, name, ha="left", va="center", fontsize=9, fontweight="bold")
        for index, stage in enumerate(stages):
            fresh = index < fresh_count
            ax.text(index + 0.35, 0.05, stage, ha="center", va="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.28", "fc": "0.90" if fresh else "0.76", "ec": "0.15", "lw": 0.8})
            if index < len(stages) - 1:
                ax.annotate("", xy=(index + 0.95, 0.05), xytext=(index + 0.68, 0.05),
                            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"})
        ax.annotate("", xy=(4.85, 0.05), xytext=(4.25, 0.05), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.text(5.05, 0.05, "Decoder", ha="center", va="center", fontsize=8,
                bbox={"boxstyle": "round,pad=0.28", "fc": "0.90", "ec": "0.15", "lw": 0.8})
        ax.text(2.2, -0.25, "fresh prefix" if fresh_count < 4 else "all stages fresh",
                ha="center", va="center", fontsize=7, color="0.25")
        ax.text(3.35, -0.25, "cached suffix on reuse frames" if fresh_count < 4 else "",
                ha="center", va="center", fontsize=7, color="0.25")
    fig.tight_layout(pad=0.25)
    fig.savefig(FIGURES / "method_overview.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    source = REPORTS / "round10_boundary_quality_latency_table.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["Boundary"] != "FULL"]
    fields = ["Model", "Boundary", "Interval", "Actual Refresh", "mIoU drop (pp)", "Mean Speedup", "CI Low", "CI High"]
    with (TABLES / "round10_main_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in selected:
            writer.writerow({"Model": row["Model"], "Boundary": row["Boundary"], "Interval": row["Interval"],
                             "Actual Refresh": row["Actual Refresh"], "mIoU drop (pp)": row["mIoU_drop_pp"],
                             "Mean Speedup": row["Mean Speedup"], "CI Low": row["CI Low"], "CI High": row["CI High"]})
    profile = []
    with (REPORTS / "round10_profile_table.csv").open(encoding="utf-8", newline="") as handle:
        profile = list(csv.DictReader(handle))
    with (TABLES / "round10_profile_cache.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile[0])); writer.writeheader(); writer.writerows(profile)
    (TABLES / "round10_sources.json").write_text(json.dumps({
        "main": "reports/round10_boundary_quality_latency_table.csv",
        "profile": "reports/round10_profile_table.csv",
        "pareto": "reports/round10_pareto_frontier.csv",
        "cache": "reports/round10_cache_memory_table.csv",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    copy_round10_figures()
    method_overview()
    write_tables()
