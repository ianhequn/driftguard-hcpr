"""Build Round 11A B0 quality/latency tables and Gate-A Pareto audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


FRONTIER = (("S4", 2, "S4/FI2"), ("S3-S4", 2, "S3-S4/FI2"),
            ("S2-S4", 2, "S2-S4/FI2"), ("S2-S4", 4, "S2-S4/FI4"),
            ("S2-S4", 8, "S2-S4/FI8"))


def fixed_quality(path: Path, model_name: str):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    result = {}
    for boundary, interval, name in FRONTIER:
        row = next(row for row in rows if row["Model"] == model_name and row["Boundary"] == boundary and int(row["Interval"]) == interval)
        result[name] = {"method": name, "kind": "fixed", "budget": f"{boundary}/FI{interval}",
                        "mIoU": float(row["mIoU"]), "mIoU_drop_pp": float(row["mIoU_drop_pp"]),
                        "pixel_accuracy": float(row["pixel_accuracy"]), "mean_class_accuracy": float(row["mean_class_accuracy"]),
                        "physical_compute_ratio": float(row["S4_calls"]) / float(row["Frames"])}
    return result


def bootstrap_speed(rows):
    values = [float(row["paired_speedup"]) for row in rows]
    return {"mean_speedup": sum(values) / len(values), "min_speedup": min(values), "max_speedup": max(values), "repetitions": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--round10-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="B0")
    args = parser.parse_args()
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    fixed = fixed_quality(args.round10_table, args.model)
    quality = json.loads(args.quality.read_text(encoding="utf-8"))["rows"]
    latency = json.loads(args.latency.read_text(encoding="utf-8"))
    latency_by_method = {}
    for row in latency["rows"]:
        latency_by_method.setdefault(row["method"], []).append(row)
    candidates = []
    for row in quality:
        method = f"HCPR/lambda-{float(row['lambda']):.8g}"
        speed_rows = latency_by_method.get(method, [])
        if not speed_rows:
            continue
        speed = bootstrap_speed(speed_rows)
        candidates.append({"method": method, "kind": "hcpr", "lambda": float(row["lambda"]), "quantile": row.get("quantile"),
                           "mIoU": row["mIoU"], "mIoU_drop_pp": row["mIoU_drop_pp"], "pixel_accuracy": row["pixel_accuracy"],
                           "mean_class_accuracy": row["mean_class_accuracy"], "physical_compute_ratio": row["actual_stage_refresh_ratios"]["stage4"],
                           "stage_calls": row["stage_calls"], "route_counts": row["route_counts"], **speed})
    for method, row in fixed.items():
        speed_rows = latency_by_method.get(method, [])
        if not speed_rows:
            continue
        fixed[method].update(bootstrap_speed(speed_rows))
    all_rows = list(fixed.values()) + candidates
    with (output / "all_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "kind", "lambda", "quantile", "mIoU", "mIoU_drop_pp", "physical_compute_ratio", "mean_speedup", "min_speedup", "max_speedup", "repetitions", "stage_calls", "route_counts"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    audits = []
    for candidate in candidates:
        dominators = [row["method"] for row in fixed.values() if row["mIoU_drop_pp"] <= candidate["mIoU_drop_pp"] and row["mean_speedup"] >= candidate["mean_speedup"] and (row["mIoU_drop_pp"] < candidate["mIoU_drop_pp"] or row["mean_speedup"] > candidate["mean_speedup"])]
        audits.append({"method": candidate["method"], "lambda": candidate["lambda"], "dominated_by_fixed": dominators, "new_pareto_point": not dominators})
    audit = {"axis": {"x": "mIoU_drop_pp_lower_better", "y": "paired_speedup_higher_better"}, "fixed_frontier": list(fixed.values()), "hcpr": audits, "new_pareto_points": [row["method"] for row in audits if row["new_pareto_point"]], "gate_a": "GO" if any(row["new_pareto_point"] for row in audits) else "NO-GO"}
    (output / "pareto_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "controller_overhead.json").write_text(json.dumps({"status": "not separately isolated in E04; included in paired end-to-end latency", "measurement_boundary": "HCPR forward including GAP, cosine, normalization, scalar synchronization, branch, cache, encoder, decoder"}, indent=2) + "\n", encoding="utf-8")
    (output / "ablation_table.csv").write_text("status,reason\nNOT_RUN,ablations are gated on B0 and B1 frontier GO\n", encoding="utf-8")
    figure = output / "hcpr_vs_fixed_pareto.pdf"
    plt.figure(figsize=(6.5, 4.5))
    for row in fixed.values(): plt.scatter(row["mIoU_drop_pp"], row["mean_speedup"], marker="s", label=row["method"])
    for row in candidates: plt.scatter(row["mIoU_drop_pp"], row["mean_speedup"], marker="o", label=row["method"])
    plt.xlabel("mIoU drop (percentage points)"); plt.ylabel("Paired speedup"); plt.grid(alpha=.25); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figure); plt.close()
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
