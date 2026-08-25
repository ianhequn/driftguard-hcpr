"""Aggregate Round 7 quality/latency results and apply frozen FI-2 gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_MIOU = 0.30129196806407454
POLICY_ORDER = ["Full", "FI-2", "FI-3", "FI-4", "FI-6", "FI-8"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policy_paths(root: Path) -> dict[str, Path]:
    return {
        "Full": root / "R7R_E02_full_gt",
        "FI-2": root / "R7R_E03_fi2_gate",
        "FI-3": root / "R7R_E04_full_curve" / "FI-3",
        "FI-4": root / "R7R_E04_full_curve" / "FI-4",
        "FI-6": root / "R7R_E04_full_curve" / "FI-6",
        "FI-8": root / "R7R_E04_full_curve" / "FI-8",
    }


def bootstrap_deltas(paths: dict[str, Path], output: Path):
    full_rows = list(csv.DictReader((paths["Full"] / "per_video_metrics.csv").open(encoding="utf-8")))
    full = {row["video_id"]: float(row["miou"]) for row in full_rows}
    result = []
    rng = np.random.default_rng(42)
    for policy in POLICY_ORDER[1:]:
        path = paths[policy] / "per_video_metrics.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        deltas = np.asarray([float(row["miou"]) - full[row["video_id"]] for row in rows], dtype=np.float64)
        samples = rng.integers(0, len(deltas), size=(10000, len(deltas)))
        boot = deltas[samples].mean(axis=1)
        result.append(
            {
                "policy": policy,
                "videos": len(deltas),
                "mean_delta_miou": float(deltas.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "seed": 42,
                "samples": 10000,
            }
        )
    write_csv(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    paths = policy_paths(args.results_root)
    latency_payload = read_json(args.latency)
    latency = {row["policy"]: row for row in latency_payload["rows"]}
    metrics = {policy: read_json(path / "metrics.json") for policy, path in paths.items() if (path / "metrics.json").exists()}
    if "Full" not in metrics or "FI-2" not in metrics:
        raise RuntimeError("Full and FI-2 quality results are required")
    full_miou = float(metrics["Full"]["miou"])
    baseline_delta = full_miou - REFERENCE_MIOU
    if abs(baseline_delta) > 0.001:
        raise RuntimeError(f"Full mIoU baseline mismatch: {full_miou} vs {REFERENCE_MIOU}")

    rows = []
    for policy in POLICY_ORDER:
        if policy not in metrics or policy not in latency:
            continue
        quality = metrics[policy]
        timing = latency[policy]
        policy_latency = timing["latency"]
        paired = timing["paired_full_latency"]
        rows.append(
            {
                "policy": policy,
                "refresh_interval": quality["refresh_interval"],
                "refresh_ratio": quality["refresh_ratio"],
                "miou": quality["miou"],
                "delta_miou": quality["miou"] - full_miou,
                "drop_miou_points": 100.0 * (full_miou - quality["miou"]),
                "pixel_accuracy": quality["pixel_accuracy"],
                "mean_class_accuracy": quality["mean_class_accuracy"],
                "mean_latency_ms": policy_latency["mean_ms"],
                "median_latency_ms": policy_latency["median_ms"],
                "p95_latency_ms": policy_latency["p95_ms"],
                "p99_latency_ms": policy_latency["p99_ms"],
                "fps": policy_latency["fps"],
                "paired_full_latency_ms": paired["mean_ms"],
                "speedup": timing["speedup"],
                "stage4_calls": quality["stage4_calls"],
                "total_frames": quality["total_frames"],
            }
        )
    write_csv(args.output_root / "main_results.csv", rows)
    bootstrap = bootstrap_deltas(paths, args.output_root / "video_bootstrap.csv")

    full_classes = read_json(paths["Full"] / "metrics.json")
    fi2_classes = read_json(paths["FI-2"] / "metrics.json")
    class_rows = []
    class_names = []
    with (paths["Full"] / "per_class_iou.csv").open(encoding="utf-8") as handle:
        class_names = [row["class_name"] for row in csv.DictReader(handle)]
    for index, name in enumerate(class_names):
        class_rows.append(
            {
                "class_id": index,
                "class_name": name,
                "support_pixels": full_classes["support_pixels"][index],
                "IoU_Full": full_classes["per_class_iou"][index],
                "IoU_FI2": fi2_classes["per_class_iou"][index],
                "delta_IoU": fi2_classes["per_class_iou"][index] - full_classes["per_class_iou"][index],
            }
        )
    write_csv(args.output_root / "full_vs_fi2_per_class.csv", class_rows)

    speedup_fi2 = next(row["speedup"] for row in rows if row["policy"] == "FI-2")
    drop_fi2 = next(row["drop_miou_points"] for row in rows if row["policy"] == "FI-2")
    if drop_fi2 > 5.0:
        decision = "QUALITY HARD STOP"
    elif speedup_fi2 >= 1.10 and drop_fi2 <= 0.5:
        decision = "ROUND 7 RESTART = STRONG GO"
    elif speedup_fi2 >= 1.10 and drop_fi2 <= 1.0:
        decision = "ROUND 7 RESTART = GO"
    elif speedup_fi2 > 1.0 and drop_fi2 > 1.0:
        decision = "ROUND 7 RESTART = CONDITIONAL"
    else:
        decision = "ROUND 7 RESTART = NO-GO"
    decision_payload = {
        "full_reference_miou": REFERENCE_MIOU,
        "full_benchmark_miou": full_miou,
        "baseline_delta": baseline_delta,
        "primary_policy": "FI-2",
        "fi2_speedup": speedup_fi2,
        "fi2_drop_miou_points": drop_fi2,
        "decision": decision,
        "bootstrap_fi2": next(item for item in bootstrap if item["policy"] == "FI-2"),
    }
    (args.output_root / "decision.json").write_text(json.dumps(decision_payload, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(7.5, 5.5))
    for row in rows:
        plt.scatter(row["mean_latency_ms"], row["miou"], s=70)
        plt.annotate(row["policy"], (row["mean_latency_ms"], row["miou"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Measured CUDA-resident latency (ms)")
    plt.ylabel("Ground-truth mIoU")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_root / "accuracy_latency_pareto.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.5))
    for row in rows:
        plt.scatter(row["speedup"], row["drop_miou_points"], s=70)
        plt.annotate(row["policy"], (row["speedup"], row["drop_miou_points"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Measured speedup (x)")
    plt.ylabel("mIoU drop (percentage points)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_root / "miou_drop_vs_speedup.png", dpi=180)
    plt.close()

    print(json.dumps(decision_payload, indent=2))


if __name__ == "__main__":
    main()
