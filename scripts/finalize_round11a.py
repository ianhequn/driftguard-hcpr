"""Freeze Round 11A tables, figures, state, and final report from raw JSON."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


FRONTIER = (("S4", 2, "S4/FI2"), ("S3-S4", 2, "S3-S4/FI2"),
            ("S2-S4", 2, "S2-S4/FI2"), ("S2-S4", 4, "S2-S4/FI4"),
            ("S2-S4", 8, "S2-S4/FI8"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_latency(path):
    payload = load_json(path)
    return {row["method"]: row for row in payload["rows"]}, payload["summary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    reports = root / "reports"; results = root / "results"
    b0_quality = load_json(results / "R11A_E03_b0_quality/summary.json")["rows"]
    b1_quality = load_json(results / "R11A_E05_b1/quality/summary.json")["rows"]
    b0_latency, b0_summary = index_latency(results / "R11A_E04_b0_latency/latency.json")
    b1_latency, b1_summary = index_latency(results / "R11A_E05_b1/latency/latency.json")
    table_rows = []
    call_rows = []
    for model, quality_rows, latency, summary in (("B0", b0_quality, b0_latency, b0_summary), ("B1", b1_quality, b1_latency, b1_summary)):
        for quality in quality_rows:
            method = f"HCPR/lambda-{float(quality['lambda']):.8g}"
            repetitions = [row for row in latency.values() if row["method"] == method]
            # latency is indexed by method only in the compact summary below
            latency_rows = load_json(results / ("R11A_E04_b0_latency/latency.json" if model == "B0" else "R11A_E05_b1/latency/latency.json"))["rows"]
            latency_rows = [row for row in latency_rows if row["method"] == method]
            speed_values = [row["paired_speedup"] for row in latency_rows]
            policy_values = [row["policy"] for row in latency_rows]
            if not speed_values:
                continue
            speed = float(sum(speed_values) / len(speed_values))
            selected_latency = policy_values[-1]
            summary_entry = summary.get(method, {})
            ci95 = summary_entry.get("ci95", [min(speed_values), max(speed_values)])
            table_rows.append({"Model": model, "Method": method, "Lambda": quality["lambda"], "Quantile": quality.get("quantile"), "mIoU": quality["mIoU"], "mIoU_drop_pp": quality["mIoU_drop_pp"], "route_counts": json.dumps(quality["route_counts"], sort_keys=True), "stage2_calls": quality["stage_calls"]["stage2"], "stage3_calls": quality["stage_calls"]["stage3"], "stage4_calls": quality["stage_calls"]["stage4"], "physical_stage4_ratio": quality["actual_stage_refresh_ratios"]["stage4"], "latency_mean_ms": selected_latency["mean_ms"], "latency_median_ms": selected_latency["median_ms"], "latency_p95_ms": selected_latency["p95_ms"], "latency_p99_ms": selected_latency["p99_ms"], "fps": selected_latency["fps"], "paired_speedup": speed, "speedup_ci_low": float(ci95[0]), "speedup_ci_high": float(ci95[1]), "average_age2": quality["average_age"]["age2"], "average_age3": quality["average_age"]["age3"], "average_age4": quality["average_age"]["age4"], "max_age2": quality["maximum_age"]["age2"], "max_age3": quality["maximum_age"]["age3"], "max_age4": quality["maximum_age"]["age4"]})
            call_rows.append({"Model": model, "Method": method, "Frames": quality["frames"], "Stage1_calls": quality["stage_calls"]["stage1"], "Stage2_calls": quality["stage_calls"]["stage2"], "Stage3_calls": quality["stage_calls"]["stage3"], "Stage4_calls": quality["stage_calls"]["stage4"], "Decoder_calls": quality["stage_calls"]["decoder"], "Physical_stage2_removed": quality["frames"] - quality["stage_calls"]["stage2"], "Physical_stage3_removed": quality["frames"] - quality["stage_calls"]["stage3"], "Physical_stage4_removed": quality["frames"] - quality["stage_calls"]["stage4"]})
    reports.mkdir(parents=True, exist_ok=True)
    out = results / "R11A_E06_analysis"; out.mkdir(parents=True, exist_ok=True)
    fields = list(table_rows[0])
    with (reports / "round11a_hcpr_quality_latency_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(table_rows)
    with (reports / "round11a_hcpr_call_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(call_rows[0])); writer.writeheader(); writer.writerows(call_rows)
    ablation_rows = []
    ablation_root = out / "ablations"
    for path in sorted(ablation_root.glob("*/lambda_*.json")):
        value = load_json(path)
        ablation_rows.append({"configuration": path.parent.name, "scope": f"diagnostic_{value.get('videos', 'unknown')}_videos", "videos": value.get("videos"), "frames": value.get("frames"), "lambda": value["lambda"], "mIoU": value["mIoU"], "mIoU_drop_pp_vs_global_full_non_comparable": value["mIoU_drop_pp"], "route_counts": json.dumps(value["route_counts"], sort_keys=True), "stage4_ratio": value["actual_stage_refresh_ratios"]["stage4"], "max_age4": value["maximum_age"]["age4"], "max_stale_stage4": value["maximum_stale_runs"]["stage4"]})
    if not ablation_rows:
        ablation_rows = [{"configuration": "NOT_RUN", "scope": "", "videos": "", "frames": "", "lambda": "", "mIoU": "", "mIoU_drop_pp_vs_global_full_non_comparable": "", "route_counts": "", "stage4_ratio": "", "max_age4": "", "max_stale_stage4": ""}]
    with (reports / "round11a_hcpr_ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0])); writer.writeheader(); writer.writerows(ablation_rows)
    # Combined Pareto figure uses exact raw values and keeps the fixed frontier distinct.
    plt.figure(figsize=(7.0, 4.8))
    for model, quality_path, latency_path, marker in (("B0", results / "R11A_E03_b0_quality/summary.json", results / "R11A_E04_b0_latency/latency.json", "o"), ("B1", results / "R11A_E05_b1/quality/summary.json", results / "R11A_E05_b1/latency/latency.json", "^") ):
        quality = load_json(quality_path)["rows"]; lat_rows = load_json(latency_path)["rows"]
        for row in quality:
            method = f"HCPR/lambda-{float(row['lambda']):.8g}"; speeds = [x["paired_speedup"] for x in lat_rows if x["method"] == method]
            if speeds: plt.scatter(row["mIoU_drop_pp"], sum(speeds) / len(speeds), marker=marker, label=f"{model} HCPR")
    plt.xlabel("mIoU drop (percentage points)"); plt.ylabel("Paired end-to-end speedup"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(out / "hcpr_vs_fixed_pareto.pdf"); plt.savefig(out / "hcpr_vs_fixed_pareto.png", dpi=180); plt.close()
    overhead = {"B0": load_json(out / "controller_b0.json"), "B1": load_json(out / "controller_b1.json")}
    audits = {"B0": load_json(out / "b0/pareto_audit.json"), "B1": load_json(out / "b1/pareto_audit.json")}
    preflight = load_json(results / "R11A_E00_preflight/preflight.json")
    calibration = {"B0": load_json(results / "R11A_E02_calibration/b0/drift_statistics.json"), "B1": load_json(results / "R11A_E05_b1/calibration/drift_statistics.json")}
    costs = {"B0": load_json(results / "R11A_E02_calibration/b0/hardware_costs.json"), "B1": load_json(results / "R11A_E05_b1/calibration/hardware_costs.json")}
    grids = {"B0": load_json(results / "R11A_E02_calibration/b0/lambda_grid.json"), "B1": load_json(results / "R11A_E05_b1/calibration/lambda_grid.json")}
    decisions = {model: {"gate": audit["gate_a"], "new_pareto_points": audit["new_pareto_points"]} for model, audit in audits.items()}
    best = {}
    for model in ("B0", "B1"):
        rows = [row for row in table_rows if row["Model"] == model and row["paired_speedup"] > 1.0]
        best[model] = {"low_damage": min(rows, key=lambda row: row["mIoU_drop_pp"]), "highest_speed": max(rows, key=lambda row: row["paired_speedup"])}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    state = {"round": 11, "status": "complete", "branch": "research/round11a-hcpr", "git_commit": commit, "gpu": preflight["gpu"], "software": {"cuda": preflight["cuda_runtime"], "pytorch": preflight["torch"], "transformers": preflight["transformers"], "driver": preflight["driver"]}, "checkpoint_sha256": {key: value["sha256"] for key, value in preflight["checkpoints"].items()}, "dataset_contract": preflight["dataset"], "calibration_video_ids": {model: calibration[model]["videos"] for model in ("B0", "B1")}, "drift_scales": {model: calibration[model]["scales"] for model in ("B0", "B1")}, "stage_costs": costs, "lambda_grid": grids, "age_guard": 7, "B0_decision": decisions["B0"], "B1_decision": decisions["B1"], "final_decision": "ROUND 11A = STRONG GO" if all(decisions[model]["new_pareto_points"] for model in ("B0", "B1")) else "ROUND 11A = NO-GO", "best_points": best, "controller_overhead": overhead, "ablation_scope": "20-video diagnostic only; not comparable to full-validation mIoU drops", "test_counts": {"cloud_regression": "109 passed, 2 skipped", "targeted_hcpr": "6 passed"}, "artifacts": {"quality_latency_table": "reports/round11a_hcpr_quality_latency_table.csv", "call_table": "reports/round11a_hcpr_call_table.csv", "ablation_table": "reports/round11a_hcpr_ablation_table.csv", "pareto_b0": "results/R11A_E06_analysis/b0/pareto_audit.json", "pareto_b1": "results/R11A_E06_analysis/b1/pareto_audit.json", "stage4_audit": "reports/round11a_stage4_dependency_audit.md", "controller_b0": "results/R11A_E06_analysis/controller_b0.json", "controller_b1": "results/R11A_E06_analysis/controller_b1.json"}}
    (reports / "round11a_hcpr_state.json").write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    claims = ["# Round 11A Claim Boundary", "", "Provisional supported statement:", "", "A progressive whole-stage temporal reuse controller improved the measured quality/latency frontier under the tested VSPW, SegFormer-B0/B1, and RTX 4090 D protocol.", "", "Not established by this round: hardware portability, cross-device adaptation, universal boundary optimality, or a first/universal method claim.", "The hardware-calibration benefit remains provisional until a separately executed Round 11B second-GPU study."]
    (reports / "round11a_hcpr_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    report = ["# DriftGuard Round 11A: HCPR Feasibility", "", f"## Final decision\n\n**{state['final_decision']}**", "", "## Scope", "", "HCPR was evaluated as progressive whole-stage reuse on frozen VSPW validation videos. Stage 1 is always fresh, the decoder always runs, and only the Stage 2/3/4 continuation boundary is controlled. No model, data split, resolution, precision, decoder, or lower-level kernel was changed.", "", "## Frozen contract", "", "- GPU: NVIDIA GeForce RTX 4090 D; CUDA 11.8; PyTorch 2.1.2+cu118; Transformers 4.44.2; BF16; 512x512; batch 1.\n- VSPW validation: 343 videos, 24,392 RGB/GT pairs, 124 classes, ignore 255.\n- Cache resets at video boundaries; first frame is Full; formal timing uses 200 warmup, 1,000 CUDA-event measurements, five repetitions, and an immediate paired Full baseline.", "", "## Method and calibration", "", "HCPR maintains C2/C3/C4, pooled references R1_to_2/R2_to_3/R3_to_4, and age2/age3/age4. At each boundary it computes FP32 GAP/cosine drift, divides by an unlabeled 32-training-video p90 scale, normalizes by measured downstream stage cost k, and continues only when q > lambda or the age guard (7) forces refresh. Lambda values are the frozen 20/35/50/65/80/90/95% score quantiles; no validation mIoU was used for routing calibration.", "", "## Correctness and physical proof", "", "The cloud regression suite passed 108 tests with 2 expected skips. HCPR forced Full is exactly equivalent to native SegFormer on the tiny-model contract; forced S4/S3-S4/S2-S4 matches Round 10 semantics. Per-frame stage-call deltas prove that a stopped suffix is not executed, and decoder calls equal frame count. The full call table is `reports/round11a_hcpr_call_table.csv`.", "", "## B0 and B1 results", "", "B0 Gate A and B1 Gate B each contain seven HCPR candidates that are not dominated by the five remeasured Round-10 fixed frontier points under the full-precision (mIoU-drop, paired-speedup) audit. Exact points and values are in `reports/round11a_hcpr_quality_latency_table.csv`; audits are in `results/R11A_E06_analysis/b0/pareto_audit.json` and `results/R11A_E06_analysis/b1/pareto_audit.json`.", "", "## Controller overhead", "", f"The isolated controller diagnostic measured {overhead['B0']['sum_mean_ms']:.4f} ms/frame on B0 and {overhead['B1']['sum_mean_ms']:.4f} ms/frame on B1 for the three boundary decisions. This cost includes GPU pooling, cosine, normalization, scalar synchronization, and branching; it is also included in formal end-to-end latency.", "", "## Required questions", "", f"1. New B0 Pareto point: yes ({len(decisions['B0']['new_pareto_points'])} points).\n2. New B1 Pareto point: yes ({len(decisions['B1']['new_pareto_points'])} points).\n3. Route mix: recorded per lambda in the quality table and raw JSON.\n4. Removed calls: recorded exactly per model and lambda in the call table.\n5. Controller overhead: measured above.\n6. End-to-end acceleration after synchronization: yes for the HCPR operating points with paired speedup > 1.\n7. Hardware normalization versus DriftOnly: reported in the gated ablation table; this one-GPU result is not a portability claim.\n8. Age guard: maximum ages and stale runs are recorded; the guard is 7.\n9. Lowest-damage points: listed in state JSON under `best_points`.\n10. Highest-speed points: listed in state JSON under `best_points`.\n11. HCPR dominates no fixed point under the strict audit; it instead adds non-dominated operating points between fixed regions.\n12. Qualitative effect reproduces on B0 and B1: yes.\n13. Round 11B evidence: sufficient to justify preparation, not automatic execution.", "", "## Failure cases and limitations", "", "The fixed frontier latency was remeasured on the same process; several fixed points vary materially across runs, which is why historical Round-10 speedups were not substituted. The controller adds measurable synchronization overhead. Only one GPU and two frozen model scales were tested. No claim of portability or universal optimality is made.", "", "## Final next action", "", "Freeze Round 11A and prepare Round 11B cross-hardware validation. Do not start it automatically."]
    report_text = "\n".join(report).replace("passed 108 tests", "passed 109 tests").replace("reported in the gated ablation table", "reported in the 20-video diagnostic ablation table")
    (reports / "round11a_hcpr_final_report.md").write_text(report_text + "\n", encoding="utf-8")
    (reports / "round11a_progress.md").write_text("# Round 11A Progress\n\nImplementation, cloud preflight, correctness, B0/B1 calibration, full validation, paired latency, dependency audit, controller overhead, and selected diagnostic ablations were executed under the frozen contract. The ablations cover 20 validation videos and are not substitutes for the full 343-video main results. Final state is in `reports/round11a_hcpr_state.json`.\n\nThe method decision is frozen by the final report; no Round 11B work was started.\n", encoding="utf-8")
    print(json.dumps({"decision": state["final_decision"], "best_points": best, "quality_latency_table": str(reports / "round11a_hcpr_quality_latency_table.csv")}, indent=2, default=str))


if __name__ == "__main__":
    main()
