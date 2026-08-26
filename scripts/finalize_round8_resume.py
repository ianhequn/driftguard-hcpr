"""Create Round 8 B0/B1 tables and figures from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


B0 = {
    "model": "B0",
    "policy": "FI-2",
    "full_miou": 0.30129196806407454,
    "policy_miou": 0.3003732475378488,
    "drop_pp": 0.091872052622574,
    "speedup_mean": 1.0923767346812105,
    "speedup_std": 0.07572837516181272,
    "stage4_fraction": 0.23975,
    "cache_mb": 0.125,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/R8R_E10_cross_scale"))
    args = parser.parse_args()
    gt = {
        policy: load_json(args.root / "R8R_E08_gt_curve" / policy / "metrics.json")
        for policy in ("Full", "FI-2", "FI-4", "FI-8")
    }
    latency = load_json(args.root / "R8R_E09_latency" / "repeated_latency.json")["rows"]
    profile = load_json(args.root / "R8R_E10_cross_scale" / "stage_profile.json")
    grouped = {policy: [row for row in latency if row["policy"] == policy] for policy in ("FI-2", "FI-4", "FI-8")}
    rows = []
    for policy in ("FI-2", "FI-4", "FI-8"):
        values = np.asarray([row["paired_speedup"] for row in grouped[policy]], dtype=float)
        latency_values = np.asarray([row["policy_latency"]["mean_ms"] for row in grouped[policy]], dtype=float)
        full_values = np.asarray([row["paired_full_latency"]["mean_ms"] for row in grouped[policy]], dtype=float)
        rows.append(
            {
                "Model": "B1",
                "Policy": policy,
                "Full_mIoU": gt["Full"]["miou"],
                "Policy_mIoU": gt[policy]["miou"],
                "Drop_pp": 100.0 * (gt["Full"]["miou"] - gt[policy]["miou"]),
                "Refresh_ratio": gt[policy]["refresh_ratio"],
                "Stage4_calls": gt[policy]["stage4_calls"],
                "Repeated_speedup_mean": float(values.mean()),
                "Repeated_speedup_std": float(values.std(ddof=1)),
                "Repeated_speedup_ci95_low": float(values.mean() - 4.303 * values.std(ddof=1) / np.sqrt(len(values))),
                "Repeated_speedup_ci95_high": float(values.mean() + 4.303 * values.std(ddof=1) / np.sqrt(len(values))),
                "Repeated_speedup_min": float(values.min()),
                "Repeated_speedup_max": float(values.max()),
                "Latency_mean_ms": float(latency_values.mean()),
                "Latency_p95_ms_mean": float(np.mean([row["policy_latency"]["p95_ms"] for row in grouped[policy]])),
                "Latency_p99_ms_mean": float(np.mean([row["policy_latency"]["p99_ms"] for row in grouped[policy]])),
                "FPS_from_mean_latency": float(1000.0 / latency_values.mean()),
                "Paired_full_mean_ms": float(full_values.mean()),
                "Stage4_fraction": profile["mean_stage4_fraction"],
                "F4_cache_MB": profile["persistent_cache_mb"],
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "b1_latency_quality_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    cross_rows = [
        {
            "Model": B0["model"],
            "Policy": B0["policy"],
            "Full_mIoU": B0["full_miou"],
            "Policy_mIoU": B0["policy_miou"],
            "Drop_pp": B0["drop_pp"],
            "Repeated_speedup": f'{B0["speedup_mean"]:.6f} +/- {B0["speedup_std"]:.6f}',
            "Stage4_fraction": B0["stage4_fraction"],
            "F4_cache_MB": B0["cache_mb"],
        }
    ]
    for row in rows:
        cross_rows.append(
            {
                "Model": row["Model"],
                "Policy": row["Policy"],
                "Full_mIoU": row["Full_mIoU"],
                "Policy_mIoU": row["Policy_mIoU"],
                "Drop_pp": row["Drop_pp"],
                "Repeated_speedup": f'{row["Repeated_speedup_mean"]:.6f} +/- {row["Repeated_speedup_std"]:.6f}',
                "Stage4_fraction": row["Stage4_fraction"],
                "F4_cache_MB": row["F4_cache_MB"],
            }
        )
    with (args.output / "cross_scale_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cross_rows[0]))
        writer.writeheader()
        writer.writerows(cross_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    ax.scatter([B0["speedup_mean"]], [B0["policy_miou"]], s=90, label="B0 FI-2")
    for row in rows:
        ax.scatter(row["Repeated_speedup_mean"], row["Policy_mIoU"], s=90, label=f'B1 {row["Policy"]}')
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Paired CUDA speedup")
    ax.set_ylabel("VSPW validation mIoU")
    ax.set_title("DriftGuard Round 8: B0/B1 accuracy-latency comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "b0_b1_accuracy_latency.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5.5), dpi=160)
    refresh = [100.0 * row["Refresh_ratio"] for row in rows]
    damage = [row["Drop_pp"] for row in rows]
    speed = [row["Repeated_speedup_mean"] for row in rows]
    labels = [row["Policy"] for row in rows]
    ax1.plot(refresh, damage, marker="o", label="mIoU drop (pp)")
    ax1.set_xlabel("Logical refresh ratio (%)")
    ax1.set_ylabel("mIoU drop (percentage points)")
    ax2 = ax1.twinx()
    ax2.plot(refresh, speed, marker="s", color="tab:orange", label="paired speedup")
    ax2.set_ylabel("Paired CUDA speedup")
    ax1.set_xticks(refresh, labels)
    ax1.set_title("B1 whole-Stage4 reuse trade-off")
    fig.tight_layout()
    fig.savefig(args.output / "b1_refresh_quality_latency.png")
    plt.close(fig)
    print(json.dumps({"cross_scale_rows": cross_rows, "b1_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
