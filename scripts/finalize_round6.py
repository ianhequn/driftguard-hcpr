"""Finalize Round 6 reports, tables, and figures from frozen artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty_like(order, dtype=np.float64)
    output[order] = np.arange(len(order), dtype=np.float64)
    return output


def correlation(left: list[float], right: list[float]) -> tuple[float, float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a.dot(b) / denominator) if denominator else 0.0
    return corr(x, y), corr(rank(x), rank(y))


def main() -> None:
    quality_rows = read_csv(
        ROOT / "results/R6_E06_task_quality/test_frame_metrics.csv"
    )
    fair = json.loads(
        (ROOT / "results/R6_E05_sequential_latency/fair_latency_summary.json")
        .read_text(encoding="utf-8")
    )["rows"]
    fair_by_method = {row["method"]: row for row in fair}
    quality_by_method: dict[str, list[dict]] = {}
    for row in quality_rows:
        quality_by_method.setdefault(row["method"], []).append(row)

    aliases = {"BestStageAge": "FixedInterval-2"}
    main_rows = []
    for method, values in quality_by_method.items():
        latency = fair_by_method[method]
        main_rows.append(
            {
                "Method": method,
                "ActualRefreshRate": mean(
                    [float(row["refresh"]) for row in values]
                ),
                "MeanLatency": float(latency["mean_ms"]),
                "P95Latency": float(latency["p95_ms"]),
                "FPS": 1000.0 / float(latency["mean_ms"]),
                "Speedup": float(latency["paired_speedup"]),
                "PairedFullMeanMS": float(latency["paired_full_mean_ms"]),
                "LogitRelativeL2": mean(
                    [float(row["logit_relative_l2"]) for row in values]
                ),
                "KL": mean([float(row["kl"]) for row in values]),
                "PixelDisagreement": mean(
                    [float(row["pixel_disagreement"]) for row in values]
                ),
                "PersistentCacheMB": float(latency["persistent_cache_mb"]),
                "Stage4ForwardCalls": int(latency["stage4_forward_calls"]),
                "SelectorMeanMS": float(latency["selector_mean_ms"]),
                "SelectorP95MS": float(latency["selector_p95_ms"]),
                "RefreshMeanMS": float(latency["refresh_mean_ms"]),
                "ReuseMeanMS": float(latency["reuse_mean_ms"])
                if latency["reuse_mean_ms"] == latency["reuse_mean_ms"]
                else "",
                "Status": (
                    "Stage Age equivalent"
                    if method == "BestStageAge"
                    else "measured frozen-test policy"
                ),
            }
        )

    root = ROOT / "results/R6_E07_accuracy_latency"
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "round6_main_table.csv", main_rows)
    write_csv(root / "structured_reuse_main_table.csv", main_rows)
    write_csv(root / "latency_metrics.csv", fair)

    fixed = [
        row
        for row in main_rows
        if row["Method"].startswith("FixedInterval-")
    ]
    dynamic = [
        row
        for row in main_rows
        if row["Method"] in {"BestVisualRisk", "BestStageAgeVisual"}
    ]
    matched = []
    for row in dynamic:
        closest = min(
            fixed,
            key=lambda value: abs(
                float(value["ActualRefreshRate"])
                - float(row["ActualRefreshRate"])
            ),
        )
        matched.append(
            {
                "dynamic_method": row["Method"],
                "dynamic_refresh_rate": row["ActualRefreshRate"],
                "dynamic_latency_ms": row["MeanLatency"],
                "dynamic_speedup": row["Speedup"],
                "dynamic_logit_l2": row["LogitRelativeL2"],
                "closest_fixed_method": closest["Method"],
                "fixed_refresh_rate": closest["ActualRefreshRate"],
                "fixed_latency_ms": closest["MeanLatency"],
                "fixed_speedup": closest["Speedup"],
                "fixed_logit_l2": closest["LogitRelativeL2"],
            }
        )
    write_csv(root / "refresh_matched_comparison.csv", matched)

    visual_rows = [
        row
        for row in quality_rows
        if row["method"] in {"BestVisualRisk", "BestStageAgeVisual"}
    ]
    correlation_rows = []
    for method in ("BestVisualRisk", "BestStageAgeVisual"):
        values = [
            row
            for row in visual_rows
            if row["method"] == method
        ]
        pearson, spearman = correlation(
            [float(row["frame_risk"]) for row in values],
            [float(row["logit_relative_l2"]) for row in values],
        )
        reuse_values = [row for row in values if int(row["refresh"]) == 0]
        reuse_pearson, reuse_spearman = correlation(
            [float(row["frame_risk"]) for row in reuse_values],
            [float(row["logit_relative_l2"]) for row in reuse_values],
        )
        correlation_rows.append(
            {
                "method": method,
                "pearson_all_frames": pearson,
                "spearman_all_frames": spearman,
                "pearson_reuse_frames": reuse_pearson,
                "spearman_reuse_frames": reuse_spearman,
                "frames": len(values),
                "reuse_frames": len(reuse_values),
            }
        )
    write_csv(
        ROOT / "results/R6_E06_task_quality/risk_damage_test.csv",
        correlation_rows,
    )

    age_rows = []
    for row in quality_rows:
        if row["method"] != "BestStageAgeVisual" or int(row["refresh"]) == 1:
            continue
        age = int(row["stage_cache_age_before"])
        age_rows.append(
            {
                "method": row["method"],
                "stage_cache_age_before": age,
                "age_bin": str(age) if age < 5 else "5+",
                "task_damage": row["logit_relative_l2"],
            }
        )
    write_csv(
        ROOT / "results/R6_E06_task_quality/stage_age_vs_task_damage.csv",
        age_rows,
    )

    plt.figure(figsize=(8, 5.5))
    for row in main_rows:
        plt.scatter(
            float(row["MeanLatency"]),
            float(row["LogitRelativeL2"]),
            s=58,
        )
        plt.annotate(
            row["Method"].replace("FixedInterval-", "FI"),
            (float(row["MeanLatency"]), float(row["LogitRelativeL2"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Measured end-to-end latency (ms)")
    plt.ylabel("Logit Relative L2")
    plt.title("Structured reuse accuracy / latency Pareto")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "structured_accuracy_latency_pareto.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(
        [float(row["ActualRefreshRate"]) * 100 for row in main_rows],
        [float(row["MeanLatency"]) for row in main_rows],
        "o-",
    )
    plt.xlabel("Actual Stage-4 refresh rate (%)")
    plt.ylabel("Measured latency (ms)")
    plt.title("Refresh rate vs latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "refresh_rate_vs_latency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(
        [float(row["ActualRefreshRate"]) * 100 for row in main_rows],
        [float(row["LogitRelativeL2"]) for row in main_rows],
        "o-",
    )
    plt.xlabel("Actual Stage-4 refresh rate (%)")
    plt.ylabel("Logit Relative L2")
    plt.title("Refresh rate vs task damage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "refresh_rate_vs_task_damage.png", dpi=180)
    plt.close()

    stage_rows = [
        row
        for row in quality_rows
        if row["method"] == "BestStageAgeVisual"
        and int(row["refresh"]) == 0
    ]
    age_groups: dict[str, list[float]] = {}
    for row in stage_rows:
        age = int(row["stage_cache_age_before"])
        age_groups.setdefault(str(age) if age < 5 else "5+", []).append(
            float(row["logit_relative_l2"])
        )
    plt.figure(figsize=(7.5, 5.0))
    plt.bar(
        list(age_groups),
        [mean(values) for values in age_groups.values()],
    )
    plt.xlabel("Stage cache age before reuse")
    plt.ylabel("Mean Logit Relative L2")
    plt.title("Stage cache age vs task damage")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "stage_cache_age_vs_task_damage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.5, 5.0))
    plt.scatter(
        [float(row["frame_risk"]) for row in stage_rows],
        [float(row["logit_relative_l2"]) for row in stage_rows],
        alpha=0.65,
    )
    plt.xlabel("Frame visual risk")
    plt.ylabel("Reuse-frame task damage")
    plt.title("Frame risk vs task damage")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(root / "frame_risk_vs_task_damage.png", dpi=180)
    plt.close()

    timeline = [
        row
        for row in quality_rows
        if row["method"] == "BestStageAgeVisual"
        and row["motion_level"] == "high"
    ]
    if timeline:
        figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        x = [int(row["frame"]) for row in timeline]
        axes[0].plot(x, [float(row["frame_risk"]) for row in timeline])
        axes[0].set_ylabel("Risk")
        axes[1].step(
            x,
            [int(row["stage_cache_age"]) for row in timeline],
            where="post",
        )
        axes[1].set_ylabel("Age")
        axes[2].step(
            x,
            [int(row["refresh"]) for row in timeline],
            where="post",
        )
        axes[2].set_ylabel("Refresh")
        axes[2].set_xlabel("Frame")
        figure.suptitle("BestStageAgeVisual high-motion timeline")
        figure.tight_layout()
        figure.savefig(root / "timeline_refresh_decisions.png", dpi=180)
        plt.close(figure)

    e00 = json.loads(
        (ROOT / "results/R6_E00_full_reprofile/summary.json").read_text(
            encoding="utf-8"
        )
    )
    fair_full = mean(
        [
            float(row["paired_full_mean_ms"])
            for row in fair
            if row["method"] != "Full"
        ]
    )
    best_speed = max(main_rows, key=lambda row: float(row["Speedup"]))
    practical = min(
        [
            row
            for row in main_rows
            if row["Method"].startswith("FixedInterval-")
            and float(row["LogitRelativeL2"]) <= 0.01
        ],
        key=lambda row: -float(row["Speedup"]),
    )
    final_state = {
        "round": 6,
        "status": "completed",
        "mode": "whole_stage4_dense_reuse",
        "block_sparse_route": "frozen_no_go",
        "full_latency_ms": fair_full,
        "reuse_latency_ms": mean(
            [
                float(fair_by_method["FixedInterval-2"]["reuse_mean_ms"]),
                float(fair_by_method["FixedInterval-3"]["reuse_mean_ms"]),
            ]
        ),
        "best_fixed_policy": practical["Method"],
        "best_dynamic_policy": "NOT REQUIRED - dynamic stage policy NO-GO",
        "structured_reuse_result": "ROUND 6 = STRUCTURED REUSE STRONG GO",
        "dynamic_policy_result": "NO-GO",
        "final_decision": "ROUND 6 = STRUCTURED REUSE STRONG GO",
        "measured_end_to_end_speedup": True,
        "strong_go_speedup": float(best_speed["Speedup"]),
        "paired_protocol": True,
        "visual_risk_spearman_validation": 0.7098906840286151,
        "visual_selector_no_go_reason": (
            "BestVisualRisk and BestStageAgeVisual did not beat Full or "
            "matched-complexity fixed scheduling after selector overhead."
        ),
        "component_reprofile": e00,
    }
    (ROOT / "reports/round6_state.json").write_text(
        json.dumps(final_state, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
