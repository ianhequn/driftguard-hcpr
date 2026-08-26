"""Assemble Round-10 tables and figures from new GPU artifacts only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = ROOT / "results" / "R10_E03_quality"
LATENCY_ROOT = ROOT / "results" / "R10_E04_latency"
PROFILE_ROOT = ROOT / "results" / "R10_E05_profile"
OUT = ROOT / "results" / "R10_E06_analysis"
REPORTS = ROOT / "reports"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_quality() -> list[dict]:
    rows = []
    for model_dir in sorted(QUALITY_ROOT.glob("b*")):
        payloads = sorted(model_dir.glob("*.json"))
        full = json.loads((model_dir / "full.json").read_text(encoding="utf-8"))
        for path in payloads:
            data = json.loads(path.read_text(encoding="utf-8"))
            boundary = data["boundary"]
            interval = int(data["interval"])
            rows.append({
                "Model": "B0" if data["model"].startswith("segformer-b0") else "B1",
                "Boundary": boundary.upper().replace("_", "-"),
                "Interval": interval,
                "Nominal Refresh": data["nominal_refresh"],
                "Actual Refresh": data["actual_refresh"],
                "Videos": data["videos"], "Frames": data["frames"],
                "S1_calls": data["S1_calls"], "S2_calls": data["S2_calls"],
                "S3_calls": data["S3_calls"], "S4_calls": data["S4_calls"],
                "Decoder_calls": data["decoder_calls"], "mIoU": data["mIoU"],
                "mIoU_drop_pp": 100.0 * (full["mIoU"] - data["mIoU"]),
                "pixel_accuracy": data["pixel_accuracy"],
                "mean_class_accuracy": data["mean_class_accuracy"],
                "cache_bytes": data["cache_bytes"], "cache_mib": data["cache_mib"],
            })
    return rows


def load_latency() -> list[dict]:
    rows = []
    for path in sorted(LATENCY_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = "B0" if payload["model"].startswith("segformer-b0") else "B1"
        for key, summary in payload["summary"].items():
            boundary, interval = key.split("/FI-")
            raw = [r["policy"] for r in payload["rows"] if r["boundary"] == boundary and r["interval"] == int(interval)]
            mean_ms = sum(r["mean_ms"] for r in raw) / len(raw)
            rows.append({"Model": model, "Boundary": boundary.upper().replace("_", "-"),
                         "Interval": int(interval), "Mean Latency ms": mean_ms,
                         "P95 ms": sum(r["p95_ms"] for r in raw) / len(raw),
                         "P99 ms": sum(r["p99_ms"] for r in raw) / len(raw),
                         "FPS": 1000.0 / mean_ms,
                         "Mean Speedup": summary["mean_speedup"], "Speedup Std": summary["std_speedup"],
                         "Speedup Min": summary["min_speedup"], "Speedup Max": summary["max_speedup"],
                         "CI Low": summary["ci95"][0], "CI High": summary["ci95"][1],
                         "Latency Repetitions": summary["repetitions"]})
    return rows


def load_full_latency() -> dict[str, dict]:
    result = {}
    for path in sorted(LATENCY_ROOT.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = "B0" if payload["model"].startswith("segformer-b0") else "B1"
        raw = [r["paired_full"] for r in payload["rows"]]
        mean_ms = sum(r["mean_ms"] for r in raw) / len(raw)
        result[model] = {"Mean Latency ms": mean_ms,
                         "P95 ms": sum(r["p95_ms"] for r in raw) / len(raw),
                         "P99 ms": sum(r["p99_ms"] for r in raw) / len(raw),
                         "FPS": 1000.0 / mean_ms}
    return result


def make_figures(rows: list[dict], profile: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    markers = {"S4": "o", "S3-S4": "s", "S2-S4": "^"}
    colors = {"B0": "0.20", "B1": "0.60"}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, model in zip(axes, ("B0", "B1")):
        for row in rows:
            if row["Model"] != model or row.get("Mean Speedup") is None or row["Boundary"] == "FULL":
                continue
            ax.errorbar(row["Mean Speedup"], row["mIoU_drop_pp"],
                        xerr=[[row["Mean Speedup"] - row["CI Low"]], [row["CI High"] - row["Mean Speedup"]]],
                        fmt=markers[row["Boundary"]], color=colors[model], ms=5, capsize=2, label=row["Boundary"])
            ax.annotate(f"FI-{row['Interval']}", (row["Mean Speedup"], row["mIoU_drop_pp"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.axvline(1.0, color="0.65", ls="--", lw=.7); ax.set_title(model); ax.grid(True, color="0.88", lw=.5); ax.set_xlabel("Mean paired speedup (x)")
    axes[0].set_ylabel("mIoU drop (percentage points)"); fig.tight_layout(pad=.5); fig.savefig(OUT / "R10_F01_boundary_pareto.pdf", bbox_inches="tight"); fig.savefig(OUT / "R10_F01_boundary_pareto.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    for ax, model in zip(axes, ("B0", "B1")):
        for boundary in markers:
            points = sorted((r for r in rows if r["Model"] == model and r["Boundary"] == boundary), key=lambda r: r["Actual Refresh"])
            ax.plot([r["Actual Refresh"] * 100 for r in points], [r["mIoU_drop_pp"] for r in points], marker=markers[boundary], color=colors[model], label=boundary)
        ax.set_title(model); ax.set_xlabel("Actual refresh (%)"); ax.grid(True, color="0.88", lw=.5)
    axes[0].set_ylabel("mIoU drop (percentage points)"); fig.tight_layout(pad=.5); fig.savefig(OUT / "R10_F02_quality_refresh.pdf", bbox_inches="tight"); fig.savefig(OUT / "R10_F02_quality_refresh.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for row in rows:
        if row.get("Mean Speedup") is None or row["Boundary"] == "FULL":
            continue
        opportunity = profile[row["Model"]][row["Boundary"]] * (1.0 - row["Actual Refresh"])
        ax.scatter(opportunity * 100, row["Mean Speedup"], marker=markers[row["Boundary"]], color=colors[row["Model"]], s=28)
        ax.annotate(f"{row['Model']} {row['Boundary']} FI-{row['Interval']}", (opportunity * 100, row["Mean Speedup"]), xytext=(3, 3), textcoords="offset points", fontsize=5.7)
    ax.axhline(1.0, color="0.65", ls="--", lw=.7); ax.set_xlabel("Profiled skipped runtime opportunity (%)"); ax.set_ylabel("Mean paired speedup (x)"); ax.grid(True, color="0.88", lw=.5); fig.tight_layout(pad=.5); fig.savefig(OUT / "R10_F03_runtime_opportunity_vs_speed.pdf", bbox_inches="tight"); fig.savefig(OUT / "R10_F03_runtime_opportunity_vs_speed.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    quality = load_quality()
    latency = load_latency()
    full_latency = load_full_latency()
    if len(quality) != 20 or len(latency) != 18:
        raise RuntimeError(f"incomplete Round-10 matrix: quality={len(quality)}, latency={len(latency)}")
    quality_export = []
    for row in quality:
        quality_export.append({
            "model": row["Model"], "boundary": row["Boundary"], "interval": row["Interval"],
            "nominal_refresh": row["Nominal Refresh"], "actual_refresh": row["Actual Refresh"],
            "videos": row["Videos"], "frames": row["Frames"],
            "S1_calls": row["S1_calls"], "S2_calls": row["S2_calls"],
            "S3_calls": row["S3_calls"], "S4_calls": row["S4_calls"],
            "decoder_calls": row["Decoder_calls"], "mIoU": row["mIoU"],
            "mIoU_drop_pp": row["mIoU_drop_pp"], "pixel_accuracy": row["pixel_accuracy"],
            "mean_class_accuracy": row["mean_class_accuracy"], "cache_bytes": row["cache_bytes"],
            "cache_mib": row["cache_mib"],
        })
    write_csv(ROOT / "results" / "R10_quality" / "full_results.csv", quality_export)
    (ROOT / "results" / "R10_quality" / "full_results.json").write_text(json.dumps(quality_export, indent=2) + "\n", encoding="utf-8")
    latency_map = {(r["Model"], r["Boundary"], r["Interval"]): r for r in latency}
    rows = []
    for quality_row in quality:
        merged = dict(quality_row)
        if quality_row["Boundary"] != "FULL":
            merged.update(latency_map[(quality_row["Model"], quality_row["Boundary"], quality_row["Interval"])])
        else:
            merged.update({**full_latency[quality_row["Model"]], "Mean Speedup": 1.0,
                           "Speedup Std": 0.0, "Speedup Min": 1.0, "Speedup Max": 1.0,
                           "CI Low": 1.0, "CI High": 1.0, "Latency Repetitions": 45})
        rows.append(merged)
    write_csv(REPORTS / "round10_boundary_quality_latency_table.csv", rows)
    write_csv(REPORTS / "round10_call_table.csv", [{key: row[key] for key in ("Model", "Boundary", "Interval", "Frames", "S1_calls", "S2_calls", "S3_calls", "S4_calls", "Decoder_calls")} for row in rows])
    profile_payload = {"B0": {}, "B1": {}}
    profile_rows = []
    for path in sorted(PROFILE_ROOT.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8")); model = "B0" if data["model"].startswith("segformer-b0") else "B1"
        profile_payload[model] = {"S4": data["shares"]["S4"], "S3-S4": data["shares"]["S3-S4"], "S2-S4": data["shares"]["S2-S4"]}
        profile_rows.append({"Model": model, **{key: data["summary"][key]["mean_ms"] for key in ("Stage1", "Stage2", "Stage3", "Stage4", "Decoder", "Full")}, "S4 Share": data["shares"]["S4"], "S3-S4 Share": data["shares"]["S3-S4"], "S2-S4 Share": data["shares"]["S2-S4"]})
    write_csv(REPORTS / "round10_profile_table.csv", profile_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "round10_rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    make_figures(rows, profile_payload)
    print(json.dumps({"quality_rows": len(quality), "latency_rows": len(latency), "figures": 3}, indent=2))


if __name__ == "__main__":
    main()
