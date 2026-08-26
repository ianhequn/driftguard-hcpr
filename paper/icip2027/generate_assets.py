"""Generate paper figures and source sidecars from frozen DriftGuard artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
FIGURES = PAPER / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES = PAPER / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def read_cross_scale() -> list[dict[str, str]]:
    path = ROOT / "results" / "R8R_E10_cross_scale" / "cross_scale_table.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sidecar(name: str, sources: list[str], claims: list[str]) -> None:
    (FIGURES / name).write_text(
        json.dumps({"sources": sources, "claims": claims}, indent=2) + "\n",
        encoding="utf-8",
    )


def write_tables(rows: list[dict[str, str]]) -> None:
    main_path = TABLES / "main_results.csv"
    with main_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Model", "Policy", "Full_mIoU", "Policy_mIoU", "Drop_pp",
            "Repeated_speedup", "Stage4_fraction", "F4_cache_MB",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (TABLES / "table_1_sources.json").write_text(json.dumps({
        "source": "results/R8R_E10_cross_scale/cross_scale_table.csv",
        "description": "Cross-scale frozen VSPW quality and repeated speed table",
    }, indent=2) + "\n", encoding="utf-8")

    b0 = json.loads((ROOT / "results" / "R71_E02_profile" / "profile_repetitions.json").read_text(encoding="utf-8"))
    b1 = json.loads((ROOT / "results" / "R8R_E10_cross_scale" / "stage_profile.json").read_text(encoding="utf-8"))
    profile_rows = [
        {"Model": "SegFormer-B0", "Stage4_fraction": f"{b0['stage4_fraction_of_full']:.12f}",
         "F4_dtype": b0["precision"], "F4_cache_MB": "0.125"},
        {"Model": "SegFormer-B1", "Stage4_fraction": f"{b1['mean_stage4_fraction']:.12f}",
         "F4_dtype": b1["f4_dtype"], "F4_cache_MB": f"{b1['persistent_cache_mb']:.3f}"},
    ]
    profile_path = TABLES / "stage4_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    (TABLES / "table_2_sources.json").write_text(json.dumps({
        "sources": [
            "results/R71_E02_profile/profile_repetitions.json",
            "results/R8R_E10_cross_scale/stage_profile.json",
        ],
        "description": "Measured Stage4 share and final F4 cache footprint",
    }, indent=2) + "\n", encoding="utf-8")


def save_figure(name: str) -> None:
    plt.savefig(FIGURES / name, bbox_inches="tight", pad_inches=0.03)
    plt.close()


def method_overview() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axis("off")
    boxes = [
        (0.02, "Frame $t$\nCUDA tensor"),
        (0.22, "Stages 1--3\nfresh"),
        (0.44, "Fixed interval\nrefresh decision"),
        (0.68, "Stage 4\ndense refresh / F4 reuse"),
        (0.90, "Decoder\nper frame"),
    ]
    for x, label in boxes:
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.45", "fc": "0.94", "ec": "0.18", "lw": 1.0})
    for (x0, _), (x1, _) in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(x1 - 0.065, 0.5), xytext=(x0 + 0.065, 0.5),
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "0.2"})
    ax.text(0.56, 0.12, "reuse returns cached final F4; no Stage 4 forward is executed",
            ha="center", va="center", fontsize=8, color="0.25")
    save_figure("method_overview.pdf")
    write_sidecar("figure_1_sources.json", [
        "reports/round6_final_report.md",
        "reports/round7_restart_equivalence.md",
        "driftguard/structured_reuse/pipeline.py",
    ], ["Stage 1-3 fresh", "whole Stage 4 dense refresh or cached final F4", "decoder per frame"])


def pareto(rows: list[dict[str, str]]) -> None:
    wanted = [("B0", "FI-2", "0.15"), ("B1", "FI-2", "0.35"), ("B1", "FI-8", "0.55")]
    selected = []
    for model, policy, shade in wanted:
        row = next(item for item in rows if item["Model"] == model and item["Policy"] == policy)
        speed = float(row["Repeated_speedup"].split()[0])
        drop = float(row["Drop_pp"])
        selected.append((f"{model} {policy}", speed, drop, shade))
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for label, speed, drop, shade in selected:
        ax.scatter(speed, drop, s=34, color=shade, edgecolor="black", linewidth=0.5)
        ax.annotate(label, (speed, drop), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axvline(1.0, color="0.65", lw=0.7, ls="--")
    ax.set_xlabel("Paired end-to-end speedup (x)")
    ax.set_ylabel("mIoU drop (percentage points)")
    ax.grid(True, color="0.88", lw=0.5)
    fig.tight_layout(pad=0.5)
    save_figure("accuracy_latency_pareto.pdf")
    write_sidecar("figure_2_sources.json", [
        "results/R8R_E10_cross_scale/cross_scale_table.csv",
        "reports/round71_latency_robustness.md",
        "reports/round8_resume_cross_scale.md",
    ], ["B0 FI-2", "B1 FI-2", "B1 FI-8"])


def stage_share() -> None:
    labels = ["SegFormer-B0", "SegFormer-B1"]
    values = [0.23975421800348004 * 100, 0.18990660459834086 * 100]
    fig, ax = plt.subplots(figsize=(3.15, 2.35))
    bars = ax.bar(labels, values, color=["0.25", "0.55"], width=0.55)
    ax.set_ylabel("Stage 4 share of full latency (%)")
    ax.set_ylim(0, 30)
    ax.grid(axis="y", color="0.88", lw=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout(pad=0.5)
    save_figure("stage4_runtime_share.pdf")
    write_sidecar("figure_3_sources.json", [
        "results/R71_E02_profile/profile_repetitions.json",
        "results/R8R_E10_cross_scale/stage_profile.json",
    ], ["B0 Stage 4 share", "B1 Stage 4 share"])


if __name__ == "__main__":
    rows = read_cross_scale()
    if len(rows) < 3:
        raise RuntimeError("cross-scale source table is incomplete")
    write_tables(rows)
    method_overview()
    pareto(rows)
    stage_share()
