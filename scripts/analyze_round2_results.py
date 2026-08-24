"""Summarize Round 2 test-video stability and bootstrap comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_RATIOS = (0.25, 0.375, 0.5)
BASELINES = ("residual_mean", "residual_p90", "random")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _video_means(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float, int], list[float]] = {}
    for row in rows:
        key = (
            row["video_id"],
            row["policy"],
            float(row["refresh_ratio"]),
            int(row["seed"]),
        )
        groups.setdefault(key, []).append(float(row["logit_relative_l2"]))
    output = []
    for (video_id, policy, ratio, seed), values in sorted(groups.items()):
        output.append(
            {
                "video_id": video_id,
                "policy": policy,
                "refresh_ratio": ratio,
                "seed": seed,
                "logit_relative_l2_mean": float(np.mean(values)),
                "global_kl_mean": float(
                    np.mean(
                        [
                            float(row["global_kl"])
                            for row in rows
                            if row["video_id"] == video_id
                            and row["policy"] == policy
                            and float(row["refresh_ratio"]) == ratio
                            and int(row["seed"]) == seed
                        ]
                    )
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [
                            float(row["global_pixel_disagreement"])
                            for row in rows
                            if row["video_id"] == video_id
                            and row["policy"] == policy
                            and float(row["refresh_ratio"]) == ratio
                            and int(row["seed"]) == seed
                        ]
                    )
                ),
            }
        )
    return output


def _bootstrap_ci(values: np.ndarray, seed: int, samples: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    resampled = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/R2_E04_refresh_simulation/policy_metrics.csv")
    parser.add_argument("--output-root", default="results/R2_E04_refresh_simulation")
    parser.add_argument("--latency", default="results/R2_E02_predictor/latency.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    rows = _read_csv(PROJECT_ROOT / args.input)
    means = _video_means(rows)
    _write_csv(output_root / "per_video_summary.csv", means)

    ci_rows = []
    stability = {}
    for ratio in TARGET_RATIOS:
        tiny = {
            row["video_id"]: row["logit_relative_l2_mean"]
            for row in means
            if row["policy"] == "tiny_risk_cnn" and row["refresh_ratio"] == ratio
        }
        ratio_stability = {}
        for baseline in BASELINES:
            if baseline == "random":
                baseline_values = {}
                for video_id in tiny:
                    values = [
                        row["logit_relative_l2_mean"]
                        for row in means
                        if row["video_id"] == video_id
                        and row["policy"] == "random"
                        and row["refresh_ratio"] == ratio
                    ]
                    baseline_values[video_id] = float(np.mean(values))
            else:
                baseline_values = {
                    row["video_id"]: row["logit_relative_l2_mean"]
                    for row in means
                    if row["policy"] == baseline and row["refresh_ratio"] == ratio
                }
            differences = np.asarray(
                [tiny[video_id] - baseline_values[video_id] for video_id in tiny],
                dtype=np.float64,
            )
            ci_low, ci_high = _bootstrap_ci(differences, seed=int(ratio * 1000))
            better_count = int(np.sum(differences < 0))
            ratio_stability[baseline] = {
                "better_videos": better_count,
                "video_count": len(differences),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
            ci_rows.append(
                {
                    "refresh_ratio": ratio,
                    "comparison": f"tiny_risk_cnn_minus_{baseline}",
                    "video_count": len(differences),
                    "mean_difference": float(differences.mean()),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "better_videos": better_count,
                }
            )
        stability[str(ratio)] = ratio_stability

    latency_rows = _read_csv(PROJECT_ROOT / args.latency)
    all_blocks = next(row for row in latency_rows if row["mode"] == "all_16_blocks")
    latency_ms = float(all_blocks["mean_ms"])
    strong_go = (
        latency_ms < 1.0
        and all(
            stability[str(ratio)][baseline]["better_videos"] == stability[str(ratio)][baseline]["video_count"]
            for ratio in TARGET_RATIOS
            for baseline in BASELINES
        )
    )
    decision = "STRONG GO" if strong_go else "GO"
    _write_csv(output_root / "bootstrap_ci.csv", ci_rows)
    (output_root / "go_no_go.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "latency_all_16_blocks_mean_ms": latency_ms,
                "target_ratios": list(TARGET_RATIOS),
                "stability": stability,
                "note": "Project-internal decision rule; not a universal statistical standard.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "latency_ms": latency_ms, "stability": stability}))


if __name__ == "__main__":
    main()
