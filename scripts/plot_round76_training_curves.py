"""Plot the retained Round 7.6 pilot history, including an aborted validation panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "step": float(row["step"]),
                    "loss": float(row["train_loss"]),
                    "learning_rate": float(row["learning_rate"]),
                }
            )
    return rows


def save_line(x, y, ylabel: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(x, y, color="#176B87", linewidth=1.5)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_history(args.history)
    steps = [row["step"] for row in rows]
    save_line(steps, [row["loss"] for row in rows], "Training cross-entropy", args.output_dir / "training_loss_curve.png")
    save_line(steps, [row["learning_rate"] for row in rows], "Learning rate", args.output_dir / "learning_rate_curve.png")

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Validation mIoU")
    axis.set_xlim(0, max(5000, max(steps)))
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.text(
        0.5,
        0.52,
        "NO VALID POINT\nValidation aborted by raw label ID 253",
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="#A23B3B",
        fontsize=13,
    )
    figure.savefig(args.output_dir / "validation_miou_curve.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
