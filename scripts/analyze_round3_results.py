"""Analyze Round 3 schedulers and produce the frozen scientific decision."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_RATIOS = (0.25, 0.375, 0.5)
METHODS = (
    "random",
    "residual_mean",
    "age",
    "cnn_patch_age",
    "afvt",
    "magvr",
    "oracle",
)
NEW_METHODS = ("afvt", "magvr")
DISPLAY = {
    "random": "Random",
    "residual_mean": "Residual",
    "age": "Age",
    "cnn_patch_age": "Pure CNN",
    "afvt": "AFVT",
    "magvr": "MAG-VR",
    "oracle": "Oracle",
}
PROFILE_DISPLAY = {
    "age": "Age",
    "cnn_patch_age": "Pure CNN",
    "afvt": "AFVT",
    "magvr": "MAG-VR",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict], fields: tuple[str, ...]):
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    return sorted(groups.items())


def _float_list(text: str) -> np.ndarray:
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def _per_video(args: argparse.Namespace) -> list[dict]:
    metrics = _read_csv(PROJECT_ROOT / args.metrics)
    rows = []
    for (video_id, motion, method, budget), values in _group_rows(
        metrics, ("video_id", "motion_group", "method", "budget")
    ):
        rows.append(
            {
                "video_id": video_id,
                "motion_group": motion,
                "method": method,
                "method_display": DISPLAY[method],
                "budget": float(budget),
                "logit_l2": float(
                    np.mean(
                        [float(row["logit_relative_l2"]) for row in values]
                    )
                ),
                "kl": float(
                    np.mean([float(row["global_kl"]) for row in values])
                ),
                "pixel_disagreement": float(
                    np.mean(
                        [float(row["pixel_disagreement"]) for row in values]
                    )
                ),
            }
        )

    random_rows = [
        row
        for row in _read_csv(PROJECT_ROOT / args.round25_per_video)
        if row["policy"] == "random"
    ]
    for row in random_rows:
        rows.append(
            {
                "video_id": row["video_id"],
                "motion_group": row["motion_level"],
                "method": "random",
                "method_display": "Random",
                "budget": float(row["refresh_ratio"]),
                "logit_l2": float(row["logit_relative_l2_mean"]),
                "kl": float(row["global_kl_mean"]),
                "pixel_disagreement": float(
                    row["pixel_disagreement_mean"]
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["method"],
            row["budget"],
            row["video_id"],
        ),
    )


def _summary(per_video: list[dict]) -> list[dict]:
    rows = []
    for (method, budget), values in _group_rows(
        per_video, ("method", "budget")
    ):
        rows.append(
            {
                "method": method,
                "method_display": DISPLAY[method],
                "budget": float(budget),
                "video_count": len(values),
                "logit_relative_l2": float(
                    np.mean([float(row["logit_l2"]) for row in values])
                ),
                "global_kl": float(
                    np.mean([float(row["kl"]) for row in values])
                ),
                "pixel_disagreement": float(
                    np.mean(
                        [float(row["pixel_disagreement"]) for row in values]
                    )
                ),
            }
        )
    return rows


def _motion_conditioned(per_video: list[dict]) -> list[dict]:
    rows = []
    for (motion, method, budget), values in _group_rows(
        per_video, ("motion_group", "method", "budget")
    ):
        rows.append(
            {
                "motion_group": motion,
                "method": method,
                "method_display": DISPLAY[method],
                "budget": float(budget),
                "video_count": len(values),
                "logit_relative_l2": float(
                    np.mean([float(row["logit_l2"]) for row in values])
                ),
                "global_kl": float(
                    np.mean([float(row["kl"]) for row in values])
                ),
                "pixel_disagreement": float(
                    np.mean(
                        [float(row["pixel_disagreement"]) for row in values]
                    )
                ),
            }
        )
    return rows


def _fairness(args: argparse.Namespace) -> list[dict]:
    frame_rows = _read_csv(PROJECT_ROOT / args.frame_diagnostics)
    per_video = _read_csv(PROJECT_ROOT / args.fairness_per_video)
    rows = []
    for (method, budget), frames in _group_rows(
        frame_rows, ("method", "budget")
    ):
        ages = np.concatenate(
            [_float_list(row["ages_after"]) for row in frames]
        )
        fairness = [
            row
            for row in per_video
            if row["method"] == method
            and float(row["budget"]) == float(budget)
        ]
        stale = np.concatenate(
            [_float_list(row["max_stale_runs"]) for row in fairness]
        )
        rows.append(
            {
                "Method": DISPLAY[method],
                "MethodKey": method,
                "Budget": float(budget),
                "MeanAge": float(ages.mean()),
                "MedianAge": float(np.median(ages)),
                "P90Age": float(np.quantile(ages, 0.90)),
                "MaxAge": float(ages.max()),
                "AgeStd": float(ages.std()),
                "MeanMaxStaleRun": float(stale.mean()),
                "P95MaxStaleRun": float(np.quantile(stale, 0.95)),
                "MaxStaleRun": int(stale.max()),
                "RefreshCountCV": float(
                    np.mean(
                        [float(row["refresh_count_cv"]) for row in fairness]
                    )
                ),
                "JainFairness": float(
                    np.mean(
                        [float(row["jain_fairness"]) for row in fairness]
                    )
                ),
            }
        )
    return rows


def _visual_usage(args: argparse.Namespace) -> list[dict]:
    frame_rows = _read_csv(PROJECT_ROOT / args.frame_diagnostics)
    rows = []
    for (method, budget), values in _group_rows(
        frame_rows, ("method", "budget")
    ):
        cnn = np.asarray(
            [float(row["cnn_evaluated_blocks"]) for row in values]
        )
        rows.append(
            {
                "Method": DISPLAY[method],
                "MethodKey": method,
                "Budget": float(budget),
                "MeanCNNBlocksPerFrame": float(cnn.mean()),
                "P95CNNBlocksPerFrame": float(np.quantile(cnn, 0.95)),
                "FramesUsingCNNRatio": float((cnn > 0).mean()),
                "MeanTieGroupSize": float(
                    np.mean(
                        [float(row["tie_group_size"]) for row in values]
                    )
                ),
                "MeanForcedBlocks": float(
                    np.mean([float(row["forced_blocks"]) for row in values])
                ),
                "MeanVisualSelectedBlocks": float(
                    np.mean(
                        [
                            float(row["visual_selected_blocks"])
                            for row in values
                        ]
                    )
                ),
                "ForcedOverflowRate": float(
                    np.mean(
                        [float(row["forced_overflow"]) for row in values]
                    )
                ),
            }
        )
    return rows


def _oracle_overlap(args: argparse.Namespace) -> list[dict]:
    rows = _read_csv(PROJECT_ROOT / args.oracle_overlap)
    output = []
    for (method, budget), values in _group_rows(
        rows, ("method", "budget")
    ):
        count = int(round(16 * float(budget)))
        output.append(
            {
                "Method": DISPLAY[method],
                "MethodKey": method,
                "Budget": float(budget),
                "TopK": count,
                "MeanOracleOverlap": float(
                    np.mean([float(row["oracle_overlap"]) for row in values])
                ),
            }
        )
    return output


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(1000, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_rows(per_video: list[dict]) -> list[dict]:
    comparisons = (
        ("afvt", "age"),
        ("magvr", "age"),
        ("cnn_patch_age", "age"),
        ("afvt", "cnn_patch_age"),
        ("magvr", "cnn_patch_age"),
    )
    rows = []
    for method, baseline in comparisons:
        for budget in CORE_RATIOS:
            candidate = {
                row["video_id"]: float(row["logit_l2"])
                for row in per_video
                if row["method"] == method
                and float(row["budget"]) == budget
            }
            reference = {
                row["video_id"]: float(row["logit_l2"])
                for row in per_video
                if row["method"] == baseline
                and float(row["budget"]) == budget
            }
            delta = np.asarray(
                [
                    candidate[key] - reference[key]
                    for key in sorted(candidate)
                ],
                dtype=np.float64,
            )
            low, high = _bootstrap(
                delta,
                seed=42 + len(rows) + int(budget * 1000),
            )
            rows.append(
                {
                    "Method": DISPLAY[method],
                    "MethodKey": method,
                    "Baseline": DISPLAY[baseline],
                    "BaselineKey": baseline,
                    "Budget": budget,
                    "MeanDeltaVsBaseline": float(delta.mean()),
                    "CI95Low": low,
                    "CI95High": high,
                    "BetterVideos": int((delta < 0).sum()),
                    "VideoCount": len(delta),
                }
            )
    return rows


def _lookup(summary: list[dict], method: str, budget: float) -> dict:
    return next(
        row
        for row in summary
        if row["method"] == method and float(row["budget"]) == budget
    )


def _profile_lookup(
    profiles: list[dict], method: str, budget: float
) -> dict | None:
    name = PROFILE_DISPLAY.get(method)
    if name is None:
        return None
    return next(
        row
        for row in profiles
        if row["method"] == name and float(row["budget"]) == budget
    )


def _main_table(
    summary: list[dict],
    profiles: list[dict],
    round25_overhead: list[dict],
) -> list[dict]:
    residual = next(
        row for row in round25_overhead if row["method"] == "Residual Mean"
    )
    rows = []
    for method in METHODS:
        profile = _profile_lookup(profiles, method, 0.375)
        selector = ""
        p95 = ""
        if profile:
            selector = profile["total_selector_ms"]
            p95 = profile["p95_total_ms"]
        elif method == "residual_mean":
            selector = residual["total_selector_ms"]
            p95 = residual["p95_total_ms"]
        values = [_lookup(summary, method, ratio) for ratio in CORE_RATIOS]
        rows.append(
            {
                "Method": DISPLAY[method],
                "MethodKey": method,
                "25%": values[0]["logit_relative_l2"],
                "37.5%": values[1]["logit_relative_l2"],
                "50%": values[2]["logit_relative_l2"],
                "SelectorMs": selector,
                "P95Ms": p95,
            }
        )
    return rows


def _decision(
    summary: list[dict],
    per_video: list[dict],
    bootstrap: list[dict],
    fairness: list[dict],
    profiles: list[dict],
) -> dict:
    age_damage = np.asarray(
        [
            float(_lookup(summary, "age", ratio)["logit_relative_l2"])
            for ratio in CORE_RATIOS
        ]
    )
    age_pixel = np.asarray(
        [
            float(_lookup(summary, "age", ratio)["pixel_disagreement"])
            for ratio in CORE_RATIOS
        ]
    )
    evidence = {}
    for method in NEW_METHODS:
        damage = np.asarray(
            [
                float(_lookup(summary, method, ratio)["logit_relative_l2"])
                for ratio in CORE_RATIOS
            ]
        )
        pixel = np.asarray(
            [
                float(_lookup(summary, method, ratio)["pixel_disagreement"])
                for ratio in CORE_RATIOS
            ]
        )
        boot = [
            row
            for row in bootstrap
            if row["MethodKey"] == method
            and row["BaselineKey"] == "age"
        ]
        stable_budgets = sum(
            int(row["BetterVideos"]) == 3
            and float(row["MeanDeltaVsBaseline"]) < 0
            for row in boot
        )
        strict_ci = sum(float(row["CI95High"]) < 0 for row in boot)
        secondary_warning = (pixel > 1.05 * age_pixel)
        method_fairness = [
            row
            for row in fairness
            if row["MethodKey"] == method
            and float(row["Budget"]) in CORE_RATIOS
        ]
        age_fairness = {
            float(row["Budget"]): row
            for row in fairness
            if row["MethodKey"] == "age"
            and float(row["Budget"]) in CORE_RATIOS
        }
        starvation_ratios = [
            float(row["MaxStaleRun"])
            / max(1.0, float(age_fairness[float(row["Budget"])]["MaxStaleRun"]))
            for row in method_fairness
        ]
        severe_starvation = any(
            float(row["MaxStaleRun"])
            > max(
                2.0
                * float(
                    age_fairness[float(row["Budget"])]["MaxStaleRun"]
                ),
                float(
                    age_fairness[float(row["Budget"])]["MaxStaleRun"]
                )
                + 2.0,
            )
            for row in method_fairness
        )
        evidence[method] = {
            "damage_by_budget": damage.tolist(),
            "delta_vs_age": (damage - age_damage).tolist(),
            "noninferior_core_budgets": int(
                (damage <= age_damage + 1e-8).sum()
            ),
            "strict_negative_ci_count": int(strict_ci),
            "stable_3_of_3_video_budget_count": int(stable_budgets),
            "secondary_warning_budgets": [
                CORE_RATIOS[index]
                for index, warning in enumerate(secondary_warning)
                if warning
            ],
            "max_stale_run_ratio_vs_age": starvation_ratios,
            "severe_starvation": severe_starvation,
        }

    best = min(
        NEW_METHODS,
        key=lambda method: np.mean(evidence[method]["damage_by_budget"]),
    )
    best_evidence = evidence[best]
    strong = (
        best_evidence["noninferior_core_budgets"] == 3
        and best_evidence["strict_negative_ci_count"] >= 2
        and best_evidence["stable_3_of_3_video_budget_count"] >= 2
        and len(best_evidence["secondary_warning_budgets"]) < 2
        and not best_evidence["severe_starvation"]
    )
    go = (
        best_evidence["noninferior_core_budgets"] == 3
        and best_evidence["strict_negative_ci_count"] >= 1
        and best_evidence["stable_3_of_3_video_budget_count"] >= 1
        and not best_evidence["severe_starvation"]
    )
    age_dominant = all(
        value["noninferior_core_budgets"] <= 1
        and value["strict_negative_ci_count"] == 0
        for value in evidence.values()
    )

    if strong:
        scientific_result = "STRONG GO"
        winner = DISPLAY[best]
    elif go:
        scientific_result = "GO"
        winner = DISPLAY[best]
    elif age_dominant:
        scientific_result = "AGE DOMINANT"
        winner = "Age"
    else:
        scientific_result = "MARGINAL"
        winner = "Marginal / No Clear Winner"

    selector_ms = None
    if winner in {"AFVT", "MAG-VR"}:
        profile = next(
            row
            for row in profiles
            if row["method"] == winner
            and float(row["budget"]) == 0.375
        )
        selector_ms = float(profile["total_selector_ms"])
    if scientific_result in {"STRONG GO", "GO"}:
        deployment = (
            "LOCK AND GO"
            if selector_ms is not None and selector_ms <= 2.0
            else "ALGORITHM LOCKED - PIPELINE OPTIMIZATION REQUIRED"
        )
    elif scientific_result == "AGE DOMINANT":
        deployment = "AGE-ONLY PIVOT"
    else:
        deployment = "MORE SCIENTIFIC WORK REQUIRED"

    return {
        "scientific_result": scientific_result,
        "algorithm_winner": winner,
        "deployment_readiness": deployment,
        "best_new_candidate": DISPLAY[best],
        "selector_ms_at_37_5": selector_ms,
        "evidence": evidence,
        "decision_rules": {
            "primary_noninferior_all_core_budgets": True,
            "strong_go_strict_bootstrap_budgets": 2,
            "secondary_warning_threshold": "pixel disagreement > 1.05x Age",
            "selector_readiness_threshold_ms": 2.0,
        },
    }


def _plot_curves(summary, metric, ylabel, path):
    plt.figure(figsize=(9, 5.5))
    for method in METHODS:
        values = sorted(
            [row for row in summary if row["method"] == method],
            key=lambda row: float(row["budget"]),
        )
        plt.plot(
            [float(row["budget"]) * 100 for row in values],
            [float(row[metric]) for row in values],
            marker="o",
            label=DISPLAY[method],
        )
    plt.xlabel("Refresh budget (%)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_age_distribution(args, path):
    frames = _read_csv(PROJECT_ROOT / args.frame_diagnostics)
    methods = ("age", "cnn_patch_age", "afvt", "magvr")
    buckets = np.arange(0, 7)
    width = 0.18
    plt.figure(figsize=(9, 5.5))
    for offset, method in enumerate(methods):
        values = [
            row
            for row in frames
            if row["method"] == method
            and float(row["budget"]) == 0.375
        ]
        ages = np.concatenate(
            [_float_list(row["ages_after"]) for row in values]
        )
        counts = [
            float((ages == age).mean()) for age in range(6)
        ] + [float((ages >= 6).mean())]
        plt.bar(
            buckets + (offset - 1.5) * width,
            counts,
            width,
            label=DISPLAY[method],
        )
    plt.xticks(buckets, ["0", "1", "2", "3", "4", "5", "6+"])
    plt.xlabel("Cache age after refresh at 37.5%")
    plt.ylabel("Block fraction")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_max_stale(fairness, path):
    methods = ("age", "cnn_patch_age", "afvt", "magvr")
    x = np.arange(len(methods))
    width = 0.24
    plt.figure(figsize=(8.5, 5.3))
    for offset, budget in enumerate(CORE_RATIOS):
        values = [
            next(
                float(row["MaxStaleRun"])
                for row in fairness
                if row["MethodKey"] == method
                and float(row["Budget"]) == budget
            )
            for method in methods
        ]
        plt.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=f"{budget * 100:g}%",
        )
    plt.xticks(x, [DISPLAY[method] for method in methods])
    plt.ylabel("Global maximum stale run")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_latency(summary, profiles, path):
    plt.figure(figsize=(7.5, 5.2))
    for method in ("age", "cnn_patch_age", "afvt", "magvr"):
        profile = _profile_lookup(profiles, method, 0.375)
        damage = float(_lookup(summary, method, 0.375)["logit_relative_l2"])
        latency = float(profile["total_selector_ms"])
        plt.scatter(latency, damage, s=60)
        plt.annotate(
            DISPLAY[method],
            (latency, damage),
            xytext=(5, 4),
            textcoords="offset points",
        )
    plt.xlabel("Total selector latency (ms/frame)")
    plt.ylabel("LogitRelativeL2 at 37.5%")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_afvt(visual, path):
    rows = [
        row
        for row in visual
        if row["MethodKey"] == "afvt"
        and float(row["Budget"]) in CORE_RATIOS
    ]
    x = np.arange(len(rows))
    width = 0.35
    plt.figure(figsize=(7.5, 5.2))
    plt.bar(
        x - width / 2,
        [float(row["MeanTieGroupSize"]) for row in rows],
        width,
        label="Tie group",
    )
    plt.bar(
        x + width / 2,
        [float(row["MeanCNNBlocksPerFrame"]) for row in rows],
        width,
        label="CNN evaluated",
    )
    plt.xticks(x, [f"{float(row['Budget']) * 100:g}%" for row in rows])
    plt.ylabel("Mean blocks/frame")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_magvr(visual, path):
    rows = [
        row
        for row in visual
        if row["MethodKey"] == "magvr"
        and float(row["Budget"]) in CORE_RATIOS
    ]
    x = np.arange(len(rows))
    forced = np.asarray([float(row["MeanForcedBlocks"]) for row in rows])
    visual_selected = np.asarray(
        [float(row["MeanVisualSelectedBlocks"]) for row in rows]
    )
    plt.figure(figsize=(7.5, 5.2))
    plt.bar(x, forced, label="Forced set size")
    plt.bar(x, visual_selected, bottom=forced, label="Visual allocation")
    plt.xticks(x, [f"{float(row['Budget']) * 100:g}%" for row in rows])
    plt.ylabel("Mean blocks/frame")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="results/R3_E03_scheduler_comparison/metrics.csv",
    )
    parser.add_argument(
        "--frame-diagnostics",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "scheduler_frame_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--fairness-per-video",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "cache_fairness_per_video.csv"
        ),
    )
    parser.add_argument(
        "--oracle-overlap",
        default=(
            "results/R3_E04_scheduler_diagnostics/"
            "oracle_overlap_per_frame.csv"
        ),
    )
    parser.add_argument(
        "--profiles",
        default="results/R3_E05_scheduler_profiling/selector_overhead.csv",
    )
    parser.add_argument(
        "--round25-per-video",
        default="results/R25_E01_selector_closure/per_video_results.csv",
    )
    parser.add_argument(
        "--round25-overhead",
        default="results/R25_E03_selector_profiling/selector_overhead.csv",
    )
    parser.add_argument(
        "--comparison-root",
        default="results/R3_E03_scheduler_comparison",
    )
    parser.add_argument(
        "--diagnostics-root",
        default="results/R3_E04_scheduler_diagnostics",
    )
    parser.add_argument("--afvt-root", default="results/R3_E01_afvt")
    parser.add_argument("--magvr-root", default="results/R3_E02_magvr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    comparison_root = PROJECT_ROOT / args.comparison_root
    diagnostics_root = PROJECT_ROOT / args.diagnostics_root
    afvt_root = PROJECT_ROOT / args.afvt_root
    magvr_root = PROJECT_ROOT / args.magvr_root
    for root in (comparison_root, diagnostics_root, afvt_root, magvr_root):
        root.mkdir(parents=True, exist_ok=True)

    per_video = _per_video(args)
    summary = _summary(per_video)
    motion = _motion_conditioned(per_video)
    fairness = _fairness(args)
    visual = _visual_usage(args)
    overlap = _oracle_overlap(args)
    bootstrap = _bootstrap_rows(per_video)
    profiles = _read_csv(PROJECT_ROOT / args.profiles)
    round25_overhead = _read_csv(PROJECT_ROOT / args.round25_overhead)
    main_table = _main_table(summary, profiles, round25_overhead)
    decision = _decision(
        summary,
        per_video,
        bootstrap,
        fairness,
        profiles,
    )

    _write_csv(comparison_root / "per_video_results.csv", per_video)
    _write_csv(comparison_root / "summary.csv", summary)
    _write_csv(comparison_root / "motion_conditioned.csv", motion)
    _write_csv(comparison_root / "main_scheduler_table.csv", main_table)
    _write_csv(comparison_root / "secondary_metrics.csv", summary)
    _write_csv(
        comparison_root / "bootstrap_vs_age.csv",
        [row for row in bootstrap if row["BaselineKey"] == "age"],
    )
    _write_csv(comparison_root / "bootstrap_comparisons.csv", bootstrap)
    _write_csv(diagnostics_root / "cache_fairness.csv", fairness)
    _write_csv(diagnostics_root / "visual_compute_usage.csv", visual)
    _write_csv(diagnostics_root / "oracle_overlap.csv", overlap)
    (diagnostics_root / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )

    for method, root in (("afvt", afvt_root), ("magvr", magvr_root)):
        method_per_video = [
            row for row in per_video if row["method"] == method
        ]
        method_summary = [
            row for row in summary if row["method"] == method
        ]
        _write_csv(root / "metrics.csv", method_per_video)
        _write_csv(root / "summary.csv", method_summary)
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "method": DISPLAY[method],
                    "frozen_predictor": "TinyRiskCNN Patch+Age",
                    "tunable_hyperparameters": 0,
                    "summary": method_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "config.yaml").write_text(
            "predictor: TinyRiskCNN Patch+Age (frozen)\n"
            "cache_cut: Stage 4\nblock_grid: 4\n"
            + (
                "scheduler: age_first_visual_tie_break\n"
                if method == "afvt"
                else "scheduler: budget_derived_max_age_guard\n"
                "max_age: ceil(16 / K) - 1\n"
            ),
            encoding="utf-8",
        )
        (root / "command.txt").write_text(
            "derived from frozen Round 3 scheduler test run\n",
            encoding="utf-8",
        )
        (root / "runtime.txt").write_text(
            f"{time.perf_counter() - start:.6f}\n",
            encoding="utf-8",
        )

    _plot_curves(
        summary,
        "logit_relative_l2",
        "Mean test LogitRelativeL2",
        comparison_root / "scheduler_budget_vs_task_damage.png",
    )
    _plot_curves(
        summary,
        "pixel_disagreement",
        "Mean test pixel disagreement",
        comparison_root / "scheduler_budget_vs_pixel_disagreement.png",
    )
    _plot_age_distribution(
        args,
        diagnostics_root / "cache_age_distribution_by_policy.png",
    )
    _plot_max_stale(
        fairness,
        diagnostics_root / "max_stale_run_by_policy.png",
    )
    _plot_latency(
        summary,
        profiles,
        diagnostics_root / "selector_latency_vs_task_damage.png",
    )
    _plot_afvt(
        visual,
        diagnostics_root / "afvt_tie_group_distribution.png",
    )
    _plot_magvr(
        visual,
        diagnostics_root / "magvr_budget_allocation.png",
    )
    config = (
        "bootstrap_resamples: 1000\nbootstrap_unit: video\n"
        "primary_metric: logit_relative_l2\n"
        "secondary_pixel_warning_ratio: 1.05\n"
        "selector_readiness_ms: 2.0\n"
    )
    for root in (comparison_root, diagnostics_root):
        (root / "analysis_config.yaml").write_text(config, encoding="utf-8")
        (root / "analysis_command.txt").write_text(
            " ".join(sys.argv) + "\n", encoding="utf-8"
        )
        (root / "analysis_runtime.txt").write_text(
            f"{time.perf_counter() - start:.6f}\n",
            encoding="utf-8",
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
