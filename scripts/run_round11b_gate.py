"""Gate A: test whether transferred HCPR points are non-dominated on GPU 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_quality(path: Path) -> dict[int, dict]:
    rows = {}
    for item in json.loads(path.read_text(encoding="utf-8"))["rows"]:
        rows[int(item["quantile"])] = item
    return rows


def load_latency(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: value for name, value in payload["summary"].items()}


def fixed_frontier(path: Path, model: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if row["Model"] != model or row["Boundary"] == "FULL":
            continue
        result.append({
            "method": f"{row['Boundary']}/FI{row['Interval']}",
            "damage_pp": float(row["mIoU_drop_pp"]),
            "speedup": None,
            "source": "Round10 quality",
        })
    return result


def nondominated(points: list[dict]) -> list[dict]:
    result = []
    for point in points:
        dominated = any(
            other is not point
            and other["damage_pp"] <= point["damage_pp"]
            and other["speedup"] >= point["speedup"]
            and (other["damage_pp"] < point["damage_pp"] or other["speedup"] > point["speedup"])
            for other in points
            if other["speedup"] is not None
        )
        if not dominated:
            result.append(point)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("B0", "B1"), required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--round10-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    quality = load_quality(args.quality)
    latency = load_latency(args.latency)
    points = fixed_frontier(args.round10_table, args.model)
    for name, summary in latency.items():
        if not name.startswith("S"):
            continue
        matching = next((p for p in points if p["method"] == name), None)
        if matching is not None:
            matching["speedup"] = float(summary["mean_speedup"])
    points = [p for p in points if p["speedup"] is not None]
    hcpr = []
    for quantile, row in sorted(quality.items()):
        name = f"HCPR-NewCost/q{quantile}"
        hcpr.append({
            "method": name,
            "quantile": quantile,
            "damage_pp": float(row["mIoU_drop_pp"]),
            "speedup": float(latency[name]["mean_speedup"]),
            "ci95": latency[name]["ci95"],
            "source": "Round11B E03/E04",
        })
    all_points = points + hcpr
    nd = nondominated(all_points)
    transfer_nd = [p for p in nd if p["source"] == "Round11B E03/E04"]
    payload = {
        "model": args.model,
        "gate": "A",
        "definition": "A transferred HCPR-NewCost point is accepted when it is non-dominated against the second-GPU fixed frontier using mIoU damage and paired mean speedup.",
        "fixed_points": points,
        "hcpr_points": hcpr,
        "non_dominated": nd,
        "transfer_non_dominated": transfer_nd,
        "pass": bool(transfer_nd),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"pareto_audit_{args.model.lower()}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter([p["damage_pp"] for p in points], [p["speedup"] for p in points], label="Round10 fixed frontier", marker="s")
    ax.scatter([p["damage_pp"] for p in hcpr], [p["speedup"] for p in hcpr], label="HCPR-NewCost", marker="o")
    for p in transfer_nd:
        ax.annotate(p["method"].split("/")[-1], (p["damage_pp"], p["speedup"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mIoU damage (percentage points)")
    ax.set_ylabel("Paired mean speedup")
    ax.set_title(f"Round 11B Gate A: {args.model}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"cross_hardware_pareto_{args.model.lower()}.png")
    fig.savefig(args.output_dir / f"cross_hardware_pareto_{args.model.lower()}.pdf")
    plt.close(fig)
    print(json.dumps({"model": args.model, "pass": bool(transfer_nd), "transfer_non_dominated": transfer_nd}, indent=2))


if __name__ == "__main__":
    main()
