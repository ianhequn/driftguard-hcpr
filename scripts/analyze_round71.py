"""Analyze the Round 7.1 repeated latency audit and emit lightweight artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = "8b673cb6bc173f3e986da8446c2c3debacc77779109b4916b230af36f197f17b"
POLICIES = ("Full", "FI-2", "FI-3", "FI-4", "FI-6", "FI-8")


def classify_fi4(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    if np.all(array >= 1.0) and float(array.mean()) > 1.0:
        return "TRANSIENT OUTLIER"
    if float(array.mean()) < 1.0 and int(np.sum(array < 1.0)) >= 3:
        return "REPRODUCIBLE PATHOLOGY"
    return "UNRESOLVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    rows = payload["rows"]
    args.output_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "repetition", "policy", "order_position", "mean_ms", "median_ms", "p95_ms", "p99_ms", "fps",
        "paired_full_mean_ms", "paired_full_median_ms", "paired_full_p95_ms", "paired_full_p99_ms", "speedup",
        "gpu_temperature", "gpu_memory_used", "gpu_clock_sm", "gpu_utilization", "refresh_frames", "refresh_ratio",
    ]
    with (args.output_root / "repeated_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    validity_checks = {
        "model_sha256": payload.get("model_sha256") == EXPECTED_SHA256,
        "environment_sha256": environment.get("model_sha256") == EXPECTED_SHA256,
        "environment_gate": environment.get("gate") == "PASS",
        "five_repetitions": payload.get("repetitions") == 5,
        "warmup_200": payload.get("warmup") == 200,
        "measurements_1000": payload.get("measurements") == 1000,
        "bf16": payload.get("precision") == "bf16",
        "all_policies_present": all(sum(row["policy"] == policy for row in rows) == 5 for policy in POLICIES),
        "paired_full_fields_present": all("paired_full_mean_ms" in row for row in rows),
        "profile_three_repetitions": profile.get("repetitions") == 3,
    }
    implementation_valid = all(validity_checks.values())

    summary = payload["summary_by_policy"]
    fi2 = summary["FI-2"]
    fi4_values = [float(row["speedup"]) for row in rows if row["policy"] == "FI-4"]
    fi4_status = classify_fi4(fi4_values)
    if not implementation_valid:
        round_status = "INVALID"
        round8_gate = "BLOCKED"
    elif fi2["mean_speedup"] >= 1.10 and fi2["ci95_low"] > 1.0:
        round_status = "ROBUST STRONG GO"
        round8_gate = "AUTHORIZED"
    elif fi2["mean_speedup"] > 1.0 and fi2["ci95_low"] > 1.0:
        round_status = "ROBUST GO"
        round8_gate = "AUTHORIZED"
    elif fi2["mean_speedup"] > 1.0:
        round_status = "UNSTABLE"
        round8_gate = "AUTHORIZED"
    else:
        round_status = "UNSTABLE"
        round8_gate = "BLOCKED"

    decision = {
        "round": "7.1",
        "round71": round_status,
        "implementation": "VALID" if implementation_valid else "INVALID",
        "round8_long_training": round8_gate,
        "fi2": summary["FI-2"],
        "fi4": {
            "status": fi4_status,
            "speedups": fi4_values,
            "historical_round7_speedup": 0.8684246131951041,
            "historical_round7_p95_ms": 43.733145523071265,
        },
        "validity_checks": validity_checks,
        "profile": profile["summary_by_component"],
        "profile_stage4_fraction": profile["stage4_fraction_of_full"],
        "profile_boundary": profile["profile_boundary"],
        "environment": {
            "gpu_query": environment.get("gpu_query"),
            "compute_apps_query": environment.get("compute_apps_query"),
        },
        "interpretation": {
            "fi2_primary": "mean speedup is above 1.0 and its five-repetition bootstrap CI is above 1.0",
            "fi4": "Round 7's low FI-4 row was not reproduced across the five fixed-order repetitions",
            "boundary_warning": "profile-derived Amdahl and paired end-to-end speedup use different decompositions",
        },
    }
    (args.output_root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for policy in ("FI-2", "FI-3", "FI-4", "FI-6", "FI-8"):
            values = [float(row["speedup"]) for row in rows if row["policy"] == policy]
            ax.plot(range(1, 6), values, marker="o", label=policy)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.axhline(1.10, color="gray", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Independent repetition", ylabel="Paired speedup", title="Round 7.1 repeated paired CUDA speedup")
        ax.legend(ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(args.output_root / "repeated_speedup.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
