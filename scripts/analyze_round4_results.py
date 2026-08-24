"""Package Round 4 stage artifacts and apply the frozen engineering rules."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LATENCY_MS = 38.40393970999867
CORE_RATIOS = (0.25, 0.375, 0.5)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_stage(
    root: Path,
    name: str,
    rows: list[dict],
    description: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "summary.csv", rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "stage": name,
                "description": description,
                "results": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "algorithm: AFVT\npredictor: TinyRiskCNN Patch+Age\n"
        "precision: fp32\npolicy_changed: false\n",
        encoding="utf-8",
    )
    (root / "command.txt").write_text(
        "derived from frozen Round 4 formal profiling\n",
        encoding="utf-8",
    )


def main() -> None:
    profiling = PROJECT_ROOT / "results/R4_E07_profiling"
    equivalence = PROJECT_ROOT / "results/R4_E06_equivalence"
    waterfall = _read_csv(profiling / "optimization_waterfall.csv")
    latency = _read_csv(profiling / "selector_latency.csv")
    eq_summary = json.loads(
        (equivalence / "summary.json").read_text(encoding="utf-8")
    )
    measured_o0 = float(waterfall[0]["Mean"])
    for row in waterfall:
        mean = float(row["Mean"])
        row["SpeedupVsO0"] = measured_o0 / mean
        row["SpeedupVsFrozenRound3"] = ORIGINAL_LATENCY_MS / mean
    _write_csv(profiling / "optimization_waterfall.csv", waterfall)
    by_stage = {row["OptimizationStage"]: row for row in waterfall}

    audit = PROJECT_ROOT / "results/R4_E00_pipeline_audit"
    audit.mkdir(parents=True, exist_ok=True)
    source = {
        "round3_reference": "immediately previous decoded RGB frame",
        "residual": "abs(current.float()/255 - previous.float()/255)",
        "grid": "4x4",
        "resize": "torch bilinear 16x16 align_corners=False",
        "metadata": "cache age standardized by frozen checkpoint mean/std",
        "source_semantics_ambiguity": (
            "Terminology ambiguity resolved from source and stored outputs; "
            "no executable ambiguity remained."
        ),
        "final_extraction": "vectorize all 16 logical blocks, then index candidates",
    }
    (audit / "source_semantics.json").write_text(
        json.dumps(source, indent=2) + "\n",
        encoding="utf-8",
    )
    (audit / "command.txt").write_text(
        "manual source audit plus frozen artifact verification\n",
        encoding="utf-8",
    )

    stage_map = (
        (
            "R4_E01_gpu_residual",
            "GPU residual",
            ["O1 GPU Residual"],
            "Move normalized RGB residual construction to CUDA.",
        ),
        (
            "R4_E02_gpu_patch",
            "GPU patch and resize",
            ["O2 GPU Patch", "O3 GPU Resize"],
            "Vectorize the 4x4 partition and batch bilinear resize.",
        ),
        (
            "R4_E03_candidate_batch",
            "Candidate batch",
            ["O4 Candidate Batch"],
            "Index AFVT candidates before batched resize and CNN inference.",
        ),
        (
            "R4_E04_fast_path",
            "Fast path",
            ["O5 Fast Path"],
            "Skip residual, patch, and CNN work when no visual rank is needed.",
        ),
        (
            "R4_E05_final_selector",
            "Final selector",
            ["O6 Direct Gather", "O7 Final"],
            "Compare direct gather and freeze the faster all-block strategy.",
        ),
    )
    for directory, name, stages, description in stage_map:
        _write_stage(
            PROJECT_ROOT / "results" / directory,
            name,
            [by_stage[stage] for stage in stages],
            description,
        )

    latency_by_budget = {
        float(row["Budget"]): row for row in latency
    }
    exact = float(eq_summary["exact_match_rate"])
    means_pass = all(
        float(latency_by_budget[ratio]["Mean"]) < 2.0
        for ratio in CORE_RATIOS
    )
    p95_pass = all(
        float(latency_by_budget[ratio]["P95"]) < 3.0
        for ratio in CORE_RATIOS
    )
    memory_pass = all(
        float(latency_by_budget[ratio]["PeakMemoryMB"]) < 100.0
        for ratio in CORE_RATIOS
    )
    if exact == 1.0 and means_pass and p95_pass and memory_pass:
        result = "STRONG ENGINEERING GO"
        readiness = "SELECTOR READY FOR PARTIAL-COMPUTE INTEGRATION"
    elif exact == 1.0 and means_pass and memory_pass:
        result = "ENGINEERING GO"
        readiness = "SELECTOR READY FOR PARTIAL-COMPUTE INTEGRATION"
    elif (
        exact == 1.0
        and 2.0
        < float(latency_by_budget[0.375]["Mean"])
        <= 4.0
        and ORIGINAL_LATENCY_MS
        / float(latency_by_budget[0.375]["Mean"])
        > 10.0
    ):
        result = "CONDITIONAL GO"
        readiness = "MORE SELECTOR PIPELINE OPTIMIZATION REQUIRED"
    else:
        result = "ENGINEERING NO-GO"
        readiness = "SELECTOR NOT READY"

    formal_37 = float(latency_by_budget[0.375]["Mean"])
    decision = {
        "engineering_result": result,
        "deployment_readiness": readiness,
        "algorithm": "AFVT",
        "predictor": "TinyRiskCNN Patch+Age",
        "precision": "FP32",
        "final_extraction": "all_blocks_vectorized_then_candidate_index",
        "policy_exact_match_rate": exact,
        "max_risk_diff": eq_summary["max_risk_diff"],
        "mean_risk_diff": eq_summary["mean_risk_diff"],
        "formal_latency_ms": {
            str(ratio): {
                key: float(latency_by_budget[ratio][key])
                for key in (
                    "Mean",
                    "Median",
                    "P95",
                    "P99",
                    "IncludingH2DMean",
                    "PeakMemoryMB",
                )
            }
            for ratio in CORE_RATIOS
        },
        "formal_speedup_vs_round3_at_37_5": (
            ORIGINAL_LATENCY_MS / formal_37
        ),
        "round3_scientific_result_reopened": False,
        "partial_computation_executed": False,
    }
    (profiling / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
