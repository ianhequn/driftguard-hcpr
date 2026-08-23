"""Plot Phase 2 temporal redundancy CSV outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stage"])].append(row)
    colors = ["#0f766e", "#2563eb", "#ca8a04", "#be123c"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for stage in sorted(grouped):
        values = sorted(grouped[stage], key=lambda item: int(item["offset"]))
        ax.plot(
            [int(item["offset"]) for item in values],
            [float(item["cosine_mean"]) for item in values],
            marker="o",
            label=f"Stage {stage}",
            color=colors[(stage - 1) % len(colors)],
        )
    ax.set_xlabel("Temporal offset k")
    ax.set_ylabel("Global cosine similarity")
    ax.set_title("SegFormer temporal feature similarity")
    ax.set_xticks(sorted({int(row["offset"]) for row in rows}))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "cosine_similarity_vs_offset.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for stage in sorted(grouped):
        values = sorted(grouped[stage], key=lambda item: int(item["offset"]))
        ax.plot(
            [int(item["offset"]) for item in values],
            [float(item["l2_mean"]) for item in values],
            marker="o",
            label=f"Stage {stage}",
            color=colors[(stage - 1) % len(colors)],
        )
    ax.set_xlabel("Temporal offset k")
    ax.set_ylabel("Relative L2 distance")
    ax.set_title("SegFormer temporal feature displacement")
    ax.set_xticks(sorted({int(row["offset"]) for row in rows}))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "relative_l2_vs_offset.png", dpi=180)
    plt.close(fig)


def _plot_heatmap(rows: list[dict[str, str]], output_dir: Path, stage: int) -> None:
    candidates = [
        row for row in rows
        if int(row["stage"]) == stage and int(row["offset"]) == 1
    ]
    if not candidates:
        raise ValueError("No offset=1 block metrics available for heatmap.")
    first_pair = (candidates[0]["current_frame"], candidates[0]["reference_frame"])
    selected = [
        row for row in candidates
        if (row["current_frame"], row["reference_frame"]) == first_pair
    ]
    grid_size = max(
        max(int(row["block_row"]) for row in selected),
        max(int(row["block_col"]) for row in selected),
    ) + 1
    heatmap = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for row in selected:
        heatmap[int(row["block_row"]), int(row["block_col"])] = float(row["cosine"])

    finite_values = heatmap[np.isfinite(heatmap)]
    value_min = float(finite_values.min())
    value_max = float(finite_values.max())
    spread = max(value_max - value_min, 1e-4)
    color_min = max(-1.0, value_min - 0.08 * spread)
    color_max = min(1.0, value_max + 0.08 * spread)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(
        heatmap,
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
    )
    ax.set_title(
        f"Block cosine similarity, Stage {stage}\n"
        f"frame {first_pair[1]} -> {first_pair[0]}"
    )
    ax.set_xlabel("Block column")
    ax.set_ylabel("Block row")
    for row in range(grid_size):
        for col in range(grid_size):
            if np.isfinite(heatmap[row, col]):
                ax.text(
                    col,
                    row,
                    f"{heatmap[row, col]:.5f}",
                    ha="center",
                    va="center",
                    color="white" if heatmap[row, col] < (color_min + color_max) / 2 else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.savefig(output_dir / "block_similarity_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/E01_temporal_redundancy",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/E01_temporal_redundancy",
    )
    parser.add_argument("--heatmap-stage", type=int, default=1)
    args = parser.parse_args()
    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _read_csv(input_dir / "temporal_redundancy_summary.csv")
    block_rows = _read_csv(input_dir / "block_metrics.csv")
    if not summary_rows:
        raise ValueError("Summary CSV is empty.")
    _plot_summary(summary_rows, output_dir)
    _plot_heatmap(block_rows, output_dir, stage=args.heatmap_stage)
    print(f"summary_rows={len(summary_rows)}")
    print(f"block_rows={len(block_rows)}")
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
