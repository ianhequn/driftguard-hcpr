"""Create the frozen-result figures used by the ICASSP HCPR manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
FIGURES = PAPER / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def method_figure() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 1.75))
    ax.set_xlim(-0.25, 7.2)
    ax.set_ylim(-0.45, 2.95)
    ax.axis("off")
    stages = ["F1", "F2", "F3", "F4", "Decoder"]
    colors = {"fresh": "#dbe9f6", "cache": "#f3dfb3", "controller": "#d9ead3"}
    routes = [
        ("S2-S4", ["fresh", "cache", "cache", "cache", "fresh"]),
        ("S3-S4", ["fresh", "fresh", "cache", "cache", "fresh"]),
        ("S4", ["fresh", "fresh", "fresh", "cache", "fresh"]),
    ]
    x0, gap, box_w, box_h = 0.72, 1.08, 0.74, 0.48
    for row, (label, states) in enumerate(routes):
        y = 2.25 - row * 1.02
        ax.text(0.02, y, label, ha="left", va="center", fontsize=9, fontweight="bold")
        for i, (stage, state) in enumerate(zip(stages, states)):
            x = x0 + i * gap
            ax.add_patch(plt.Rectangle((x, y - box_h / 2), box_w, box_h,
                                       facecolor=colors[state], edgecolor="#263238",
                                       linewidth=0.75))
            ax.text(x + box_w / 2, y, stage, ha="center", va="center", fontsize=8)
            if i < len(stages) - 1:
                ax.annotate("", xy=(x + gap - 0.1, y), xytext=(x + box_w + 0.1, y),
                            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#455a64"})
        ax.text(x0 + gap * 1.5, y - 0.38, "fresh prefix", ha="center", va="center",
                fontsize=6.8, color="#455a64")
        if "cache" in states:
            first_cache = states.index("cache")
            ax.text(x0 + gap * (first_cache + 1.5), y - 0.38, "cached suffix",
                    ha="center", va="center", fontsize=6.8, color="#7a5b00")
    ax.add_patch(plt.Rectangle((5.94, 0.05), 0.28, 0.22, facecolor=colors["controller"],
                               edgecolor="#263238", linewidth=0.6))
    ax.text(6.32, 0.16, "HCPR controller: drift + target-device cost", va="center", fontsize=7.2)
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "hcpr_method.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def pareto_figure() -> None:
    r11a = read_rows(ROOT / "reports" / "round11a_hcpr_quality_latency_table.csv")
    r11b = read_rows(ROOT / "reports" / "round11b_cross_hardware_table.csv")
    # Round 11A has one row per model/quantile; use the exact frozen values.
    a_rows = {(row["Model"], row["Quantile"]): row for row in r11a
              if row["Quantile"] in {"20", "65", "95"}}
    b_rows = {(row["model"], row["quantile"]): row for row in r11b
              if row["variant"] == "HCPR-NewCost"}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), sharey=True)
    colors = {"20": "#1b6ca8", "65": "#e07a2d", "95": "#2b8a3e"}
    for ax, model in zip(axes, ["B0", "B1"]):
        for quantile in ["20", "65", "95"]:
            old = a_rows[(model, quantile)]
            new = b_rows[(model, quantile)]
            ax.scatter(float(old["mIoU_drop_pp"]), float(old["paired_speedup"]),
                       marker="o", s=30, color=colors[quantile], label=f"q{quantile} 4090D")
            ax.scatter(float(new["damage_pp"]), float(new["mean_speedup"]),
                       marker="s", s=30, facecolors="none", edgecolors=colors[quantile],
                       linewidths=1.2, label=f"q{quantile} 3080Ti")
            ax.annotate(f"q{quantile}", (float(new["damage_pp"]), float(new["mean_speedup"])),
                        xytext=(3, 3), textcoords="offset points", fontsize=6.5)
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xlabel("mIoU damage (percentage points)", fontsize=7.5)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Paired speedup (x)", fontsize=7.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.12, 1, 1), pad=0.5, w_pad=1.0)
    fig.savefig(FIGURES / "hcpr_cross_hardware_pareto.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    method_figure()
    pareto_figure()
