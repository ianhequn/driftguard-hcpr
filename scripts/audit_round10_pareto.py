"""Audit mean speedup/mIoU-drop Pareto frontiers from the final Round 10 CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "round10_boundary_quality_latency_table.csv"
REPORTS = ROOT / "reports"


def dominates(a: dict, b: dict) -> bool:
    return (a["Mean Speedup"] >= b["Mean Speedup"] and
            a["mIoU_drop_pp"] <= b["mIoU_drop_pp"] and
            (a["Mean Speedup"] > b["Mean Speedup"] or a["mIoU_drop_pp"] < b["mIoU_drop_pp"]))


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frontier_rows = []
    report = [
        "# Round 10 Pareto Audit",
        "",
        "## Definition",
        "",
        "Point A dominates point B when mean paired speedup is no smaller, mIoU",
        "drop is no larger, and at least one inequality is strict. Full is retained",
        "as the reference row but the reuse frontier is computed over the nine",
        "non-Full boundary/interval settings for each model.",
        "",
    ]
    for model in ("B0", "B1"):
        candidates = [row for row in rows if row["Model"] == model and row["Boundary"] != "FULL"]
        for row in candidates:
            row["Mean Speedup"] = float(row["Mean Speedup"])
            row["mIoU_drop_pp"] = float(row["mIoU_drop_pp"])
        frontier = [row for row in candidates if not any(dominates(other, row) for other in candidates)]
        frontier.sort(key=lambda row: (float(row["mIoU_drop_pp"]), -float(row["Mean Speedup"])))
        report.append(f"## {model}")
        report.append("")
        report.append("| Boundary | Interval | mIoU drop (pp) | Mean speedup |")
        report.append("|---|---:|---:|---:|")
        for row in frontier:
            report.append(f"| {row['Boundary']} | FI-{row['Interval']} | {float(row['mIoU_drop_pp']):.6f} | {float(row['Mean Speedup']):.6f} |")
            frontier_rows.append({"Model": model, "Boundary": row["Boundary"], "Interval": row["Interval"], "mIoU_drop_pp": row["mIoU_drop_pp"], "Mean Speedup": row["Mean Speedup"], "CI Low": row["CI Low"], "CI High": row["CI High"]})
        dominated = [row for row in candidates if row not in frontier]
        report.append("")
        report.append("Dominated settings: " + (", ".join(f"{r['Boundary']}/FI-{r['Interval']}" for r in sorted(dominated, key=lambda r: (r['Boundary'], int(r['Interval']))) ) or "none") + ".")
        report.append("")
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "round10_pareto_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier_rows[0]))
        writer.writeheader(); writer.writerows(frontier_rows)
    (REPORTS / "round10_pareto_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"PARETO_AUDIT_PASS frontier_rows={len(frontier_rows)}")


if __name__ == "__main__":
    main()
