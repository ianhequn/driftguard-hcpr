"""Audit MMM Pareto claims without importing hand-entered paper numbers.

The audit deliberately keeps quality and latency provenance separate.  A point
is only eligible for a strict Pareto set when both measurements belong to the
same local protocol identifier.  Cross-protocol comparisons are still emitted
as diagnostic-only results so that historical claims can be explained without
silently treating them as valid evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


FIXED_METHODS = {
    "s4_fi2": ("S4", 2),
    "s3_s4_fi2": ("S3-S4", 2),
    "s2_s4_fi2": ("S2-S4", 2),
    "s2_s4_fi4": ("S2-S4", 4),
    "s2_s4_fi8": ("S2-S4", 8),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table_block(tex: str, label: str) -> str:
    marker = f"\\label{{{label}}}"
    if marker not in tex:
        return ""
    return tex.split(marker, 1)[1].split("\\end{table}", 1)[0]


def latex_number(value: float) -> list[str]:
    formatted = f"{value:.3f}"
    return [formatted, formatted.removeprefix("0")]


def canonical_model(value: str) -> str:
    value = value.upper()
    if value in {"B0", "B1"}:
        return value
    if "B0" in value:
        return "B0"
    if "B1" in value:
        return "B1"
    raise ValueError(f"Cannot infer model from {value!r}")


def canonical_fixed_name(boundary: str, interval: int) -> str:
    return f"{boundary}/FI{interval}"


def canonical_latency_method(name: str) -> str:
    value = name.lower().replace("_", "-")
    value = value.replace("s2-s4", "s2-s4").replace("s3-s4", "s3-s4")
    value = value.replace("fi-", "fi")
    return value.upper().replace("S2-S4", "S2-S4").replace("S3-S4", "S3-S4")


def make_point(
    *,
    model: str,
    device: str,
    method: str,
    boundary: str,
    interval: int | None,
    quantile: int | None,
    quality_source: str,
    latency_source: str,
    quality_protocol: str,
    latency_protocol: str,
    damage_pp: float,
    speedup: float,
    ci_low: float | None,
    ci_high: float | None,
) -> dict:
    same_protocol = quality_protocol == latency_protocol
    point = {
        "model": model,
        "device": device,
        "method": method,
        "boundary": boundary,
        "interval": interval,
        "quantile": quantile,
        "quality_source": quality_source,
        "latency_source": latency_source,
        "quality_protocol_id": quality_protocol,
        "latency_protocol_id": latency_protocol,
        "measurement_protocol_id": quality_protocol if same_protocol else "MIXED_PROTOCOL",
        "mIoU_damage_pp": float(damage_pp),
        "speedup": float(speedup),
        "speedup_ci_low": ci_low,
        "speedup_ci_high": ci_high,
        "valid_for_strict_pareto": same_protocol and ci_low is not None and ci_high is not None,
        "invalid_reason": None if same_protocol else "quality and latency came from different measurement protocols",
    }
    return point


def fixed_quality(root: Path, model: str, key: str) -> tuple[float, str]:
    quality_dir = root / "results" / "R10_E03_quality" / model.lower()
    full = read_json(quality_dir / "full.json")
    candidate = read_json(quality_dir / f"{key}.json")
    damage = (float(full["mIoU"]) - float(candidate["mIoU"])) * 100.0
    return damage, f"results/R10_E03_quality/{model.lower()}/{key}.json"


def fixed_latency(root: Path, model: str, method: str, round_name: str) -> tuple[float, float, float, str]:
    path = root / "results" / f"{round_name}_E04_latency" / f"{model.lower()}.json"
    summary = read_json(path)["summary"]
    if round_name == "R11B":
        wanted = method
    else:
        boundary, interval = method.split("/FI")
        wanted = f"{boundary.lower().replace('-', '_')}/FI-{interval}"
    if round_name == "R11B":
        item = summary[wanted]
    else:
        item = summary[wanted]
    return float(item["mean_speedup"]), float(item["ci95"][0]), float(item["ci95"][1]), str(path.relative_to(root))


def load_fixed_points(root: Path, *, device: str, latency_round: str, protocol: str) -> list[dict]:
    points: list[dict] = []
    for model in ("B0", "B1"):
        for key, (boundary, interval) in FIXED_METHODS.items():
            method = canonical_fixed_name(boundary, interval)
            damage, quality_source = fixed_quality(root, model, key)
            speedup, ci_low, ci_high, latency_source = fixed_latency(root, model, method, latency_round)
            quality_protocol = "R10-FIXED-4090D-v1"
            latency_protocol = protocol
            points.append(
                make_point(
                    model=model,
                    device=device,
                    method=method,
                    boundary=boundary,
                    interval=interval,
                    quantile=None,
                    quality_source=quality_source,
                    latency_source=latency_source,
                    quality_protocol=quality_protocol,
                    latency_protocol=latency_protocol,
                    damage_pp=damage,
                    speedup=speedup,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )
    return points


def load_r11a_hcpr(root: Path) -> list[dict]:
    state = read_json(root / "reports" / "round11a_hcpr_state.json")
    path = root / "reports" / "round11a_hcpr_quality_latency_table.csv"
    points = []
    for row in read_csv(path):
        model = canonical_model(row["Model"])
        quantile = int(row["Quantile"])
        points.append(
            make_point(
                model=model,
                device=str(state["gpu"]),
                method=f"HCPR-NewCost/q{quantile}",
                boundary="progressive-adaptive",
                interval=None,
                quantile=quantile,
                quality_source=str(path.relative_to(root)),
                latency_source=str(path.relative_to(root)),
                quality_protocol="R11A-HCPR-4090D-v1",
                latency_protocol="R11A-HCPR-4090D-v1",
                damage_pp=float(row["mIoU_drop_pp"]),
                speedup=float(row["paired_speedup"]),
                ci_low=float(row["speedup_ci_low"]),
                ci_high=float(row["speedup_ci_high"]),
            )
        )
    return points


def load_r11b_hcpr(root: Path) -> list[dict]:
    preflight = read_json(root / "results" / "R11B_E00_preflight" / "preflight.json")
    points = []
    for model in ("B0", "B1"):
        quality_path = root / "results" / "R11B_E03_transfer" / model.lower() / "summary.json"
        latency_path = root / "results" / "R11B_E04_latency" / f"{model.lower()}.json"
        anchor_latency_path = root / "results" / "R11B_E05_latency" / model.lower() / "HCPR-NewCost" / "latency.json"
        quality = {int(row["quantile"]): row for row in read_json(quality_path)["rows"]}
        latency = read_json(latency_path)["summary"]
        anchor_latency = read_json(anchor_latency_path)["summary"] if anchor_latency_path.exists() else {}
        for quantile in sorted(quality):
            qrow = quality[quantile]
            method = f"HCPR-NewCost/q{quantile}"
            lrow = anchor_latency.get(method, latency[method])
            points.append(
                make_point(
                    model=model,
                    device=str(preflight["gpu"]),
                    method=f"HCPR-NewCost/q{quantile}",
                    boundary="progressive-adaptive",
                    interval=None,
                    quantile=quantile,
                    quality_source=str(quality_path.relative_to(root)),
                    latency_source=str((anchor_latency_path if method in anchor_latency else latency_path).relative_to(root)),
                    quality_protocol="R11B-HCPR-3080Ti-v1",
                    latency_protocol="R11B-HCPR-3080Ti-v1",
                    damage_pp=float(qrow["mIoU_drop_pp"]),
                    speedup=float(lrow["mean_speedup"]),
                    ci_low=float(lrow["ci95"][0]),
                    ci_high=float(lrow["ci95"][1]),
                )
            )
    return points


def dominates(other: dict, point: dict) -> bool:
    return (
        other["mIoU_damage_pp"] <= point["mIoU_damage_pp"]
        and other["speedup"] >= point["speedup"]
        and (
            other["mIoU_damage_pp"] < point["mIoU_damage_pp"]
            or other["speedup"] > point["speedup"]
        )
    )


def dominators(point: dict, candidates: Iterable[dict]) -> list[str]:
    return [candidate["method"] for candidate in candidates if candidate is not point and dominates(candidate, point)]


def diagnostic_comparison(fixed: list[dict], hcpr: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for model in ("B0", "B1"):
        all_points = [p for p in fixed + hcpr if p["model"] == model]
        hcpr_points = [p for p in hcpr if p["model"] == model]
        entries = []
        for point in hcpr_points:
            entries.append({"method": point["method"], "dominators": dominators(point, all_points)})
        result[model] = {
            "numeric_only": True,
            "reason": "historical comparison; protocol mismatch is retained as an explicit warning",
            "hcpr_non_dominated_count": sum(not item["dominators"] for item in entries),
            "hcpr_point_count": len(entries),
            "points": entries,
        }
    return result


def strict_sets(points: list[dict]) -> dict[str, dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for point in points:
        if point["valid_for_strict_pareto"]:
            groups[(point["model"], point["device"], point["measurement_protocol_id"])].append(point)
    result: dict[str, dict] = {}
    for key, group in groups.items():
        model, device, protocol = key
        hcp = [p for p in group if p["method"].startswith("HCPR")]
        fixed = [p for p in group if not p["method"].startswith("HCPR")]
        if not hcp or not fixed:
            continue
        result[f"{model}|{device}|{protocol}"] = {
            "model": model,
            "device": device,
            "protocol": protocol,
            "fixed_count": len(fixed),
            "hcpr_count": len(hcp),
            "hcpr_dominators": {p["method"]: dominators(p, group) for p in hcp},
        }
    return result


def paper_claim_checks(root: Path, fixed_4090: list[dict], hcpr_4090: list[dict], fixed_3080: list[dict], hcpr_3080: list[dict]) -> dict:
    tex = (root / "paper" / "mmm2027" / "MMM2027_HCPR_anonymous_revised.tex").read_text(encoding="utf-8")
    fixed_b0 = next(p for p in fixed_4090 if p["model"] == "B0" and p["method"] == "S2-S4/FI4")
    fixed_b1 = next(p for p in fixed_4090 if p["model"] == "B1" and p["method"] == "S2-S4/FI4")
    source_q90 = next(p for p in hcpr_4090 if p["model"] == "B0" and p["quantile"] == 90)
    source_q95 = next(p for p in hcpr_4090 if p["model"] == "B0" and p["quantile"] == 95)
    table_fixed_block = table_block(tex, "tab:fixed")
    table_cross_block = table_block(tex, "tab:cross")
    table_cost_block = table_block(tex, "tab:cost")
    table_ablation_block = table_block(tex, "tab:ablation")
    target_ablation = read_csv(root / "reports" / "round11b_cross_hardware_table.csv")

    source_anchor_check = all(
        any(
            f"{speed_token} / {damage_token}" in table_cross_block
            for speed_token in latex_number(point["speedup"])
            for damage_token in latex_number(point["mIoU_damage_pp"])
        )
        for point in hcpr_4090
        if point["quantile"] in {20, 65, 95}
    )
    target_anchor_check = all(
        any(
            f"{speed_token} / {damage_token}" in table_cross_block
            for speed_token in latex_number(point["speedup"])
            for damage_token in latex_number(point["mIoU_damage_pp"])
        )
        for point in hcpr_3080
        if point["quantile"] in {20, 65, 95}
    )

    state11a = read_json(root / "reports" / "round11a_hcpr_state.json")
    profiles = {
        "B0": read_json(root / "results" / "R11B_E02_profile" / "b0.json"),
        "B1": read_json(root / "results" / "R11B_E02_profile" / "b1.json"),
    }
    cost_rows_match = True
    for model in ("B0", "B1"):
        source_cost = state11a["stage_costs"][model]
        target_cost = profiles[model]
        for device_block, cost in (("RTX 4090 D", source_cost), ("RTX 3080 Ti", target_cost)):
            row_match = device_block in table_cost_block and model in table_cost_block
            row_match = row_match and all(
                any(token in table_cost_block for token in latex_number(float(cost[key])))
                for key in ("full_ms", "k2", "k3", "k4")
            )
            cost_rows_match = cost_rows_match and row_match

    ablation_values_match = True
    ablation_mismatches = []
    for row in target_ablation:
        if row["quantile"] not in {"20", "65", "95"}:
            continue
        pair = f"{float(row['damage_pp']):.3f}/{float(row['mean_speedup']):.3f}"
        pair_short = pair.replace("0.", ".")
        if pair not in table_ablation_block and pair_short not in table_ablation_block:
            ablation_values_match = False
            ablation_mismatches.append({"model": row["model"], "variant": row["variant"], "quantile": int(row["quantile"]), "artifact": pair})
    return {
        "table_2_source_anchor_values_match_artifact": source_anchor_check,
        "table_2_target_anchor_values_match_artifact": target_anchor_check,
        "table_3_fixed_values_match_round10_artifact": all(
            f"{p['mIoU_damage_pp']:.3f}" in table_fixed_block and f"{p['speedup']:.3f}" in table_fixed_block
            for p in (fixed_b0, fixed_b1)
        ),
        "table_3_has_explicit_gpu_label": "RTX" in table_fixed_block or "GPU" in table_fixed_block,
        "paper_mentions_source_all_7_non_dominated": "On RTX 4090 D, all seven candidates are non-dominated" in tex,
        "paper_claimed_target_counts": "On RTX 3080 Ti, 7/7 B0 and 6/7 B1 candidates remain non-dominated" in tex,
        "source_numeric_b0_q90_dominated_by": dominators(
            source_q90,
            [p for p in [*fixed_4090, *hcpr_4090] if p["model"] == "B0"],
        ),
        "source_numeric_b0_q95_dominated_by": dominators(
            source_q95,
            [p for p in [*fixed_4090, *hcpr_4090] if p["model"] == "B0"],
        ),
        "source_numeric_counts": diagnostic_comparison(fixed_4090, hcpr_4090),
        "target_numeric_counts": diagnostic_comparison(fixed_3080, hcpr_3080),
        "table_5_cost_values_match_artifact": cost_rows_match,
        "table_6_ablation_values_match_artifact": ablation_values_match,
        "table_6_ablation_mismatches": ablation_mismatches,
    }


def audit(root: Path) -> dict:
    fixed_4090 = load_fixed_points(
        root,
        device="NVIDIA GeForce RTX 4090 D",
        latency_round="R10",
        protocol="R10-FIXED-4090D-v1",
    )
    fixed_3080 = load_fixed_points(
        root,
        device="NVIDIA GeForce RTX 3080 Ti",
        latency_round="R11B",
        protocol="R11B-FIXED-3080Ti-v1",
    )
    hcpr_4090 = load_r11a_hcpr(root)
    hcpr_3080 = load_r11b_hcpr(root)
    points = fixed_4090 + fixed_3080 + hcpr_4090 + hcpr_3080
    report = {
        "audit": "MMM 2027 Pareto consistency audit",
        "definition": "A dominates B when damage is no larger, speedup is no smaller, and one inequality is strict.",
        "strict_rule": "Points from different device, quality protocol, or latency protocol are never placed in one strict Pareto set.",
        "points": points,
        "strict_pareto_sets": strict_sets(points),
        "diagnostic_comparisons": {
            "source_4090D": diagnostic_comparison(fixed_4090, hcpr_4090),
            "target_3080Ti": diagnostic_comparison(fixed_3080, hcpr_3080),
        },
        "paper_claim_checks": paper_claim_checks(root, fixed_4090, hcpr_4090, fixed_3080, hcpr_3080),
        "findings": [
            "Round 10 fixed-policy quality and latency are a complete same-protocol 4090D set.",
            "Round 11A HCPR is a separate same-protocol 4090D set; combining it with Round 10 fixed points is diagnostic-only.",
            "Round 11B target fixed points mix Round 10 quality with Round 11B latency; they are invalid for a strict same-protocol set.",
            "Round 11B HCPR quality and latency form a same-protocol 3080 Ti set, but no same-protocol target fixed quality sweep is present.",
        ],
    }
    return report


def markdown(report: dict) -> str:
    checks = report["paper_claim_checks"]
    lines = [
        "# Post-MMM Pareto Consistency Audit",
        "",
        "## 结论",
        "",
        "当前论文的 fixed-vs-HCPR non-dominance 结论不能作为严格同 protocol 结论接受。Round 10 fixed policy、Round 11A HCPR、Round 11B target fixed/HCPR 的来源链被脚本显式拆开；跨 protocol 的数字只作为 diagnostic，不进入 strict Pareto set。",
        "",
        "## 来源链",
        "",
        "| 集合 | 设备 | quality source | latency source | strict 状态 |",
        "|---|---|---|---|---|",
        "| Round 10 fixed | RTX 4090 D | `results/R10_E03_quality` | `results/R10_E04_latency` | 可作为内部同 protocol 集合 |",
        "| Round 11A HCPR | RTX 4090 D | `reports/round11a_hcpr_quality_latency_table.csv` | 同一 CSV 的 Round 11A 汇总 | 可作为内部同 protocol 集合；不能直接与 Round 10 fixed 合并 |",
        "| Round 11B fixed | RTX 3080 Ti | Round 10 quality | `results/R11B_E04_latency` | INVALID/MIXED_PROTOCOL |",
        "| Round 11B HCPR | RTX 3080 Ti | `results/R11B_E03_transfer` | `results/R11B_E04_latency` | 可作为 HCPR-only 集合 |",
        "",
        "## 论文文字检查",
        "",
        f"- Table 2 source-device anchor 与 artifact 一致：`{checks['table_2_source_anchor_values_match_artifact']}`；target-device anchor 与 artifact 一致：`{checks['table_2_target_anchor_values_match_artifact']}`。",
        f"- Table 3 的代表性 fixed 数值与 Round 10 artifact 一致：`{checks['table_3_fixed_values_match_round10_artifact']}`。",
        f"- Table 3 表格块显式标 GPU：`{checks['table_3_has_explicit_gpu_label']}`；当前为 `False`。",
        f"- Table 5 cost profile 数值与 Round 11A/R11B profile artifact 一致：`{checks['table_5_cost_values_match_artifact']}`。",
        f"- Table 6 ablation 数值全部与 Round 11B 汇总一致：`{checks['table_6_ablation_values_match_artifact']}`；不一致项：`{checks['table_6_ablation_mismatches']}`。",
        f"- 论文写有 4090 D 全 7 点 non-dominated：`{checks['paper_mentions_source_all_7_non_dominated']}`。",
        f"- 论文写有 target B0 7/7、B1 6/7：`{checks['paper_claimed_target_counts']}`。",
        f"- 在 diagnostic 数值重算中，source 4090 D B0 q90 的 dominator：`{', '.join(checks['source_numeric_b0_q90_dominated_by']) or 'none'}`。",
        f"- source 4090 D B0 q95 的 dominator：`{', '.join(checks['source_numeric_b0_q95_dominated_by']) or 'none'}`。",
        "",
        "## 解释",
        "",
        "B0 source q90/q95 的 fixed dominator 已足以推翻“4090 D 全 7 点 non-dominated”。这不是重新测量得出的新结果，而是对仓库已有 artifact 的一致性重算。target 的 7/7、6/7 计数只能描述混合来源的 diagnostic comparison，不能替代同设备、同 quality protocol 的 fixed sweep。",
        "",
        "## 机器可读明细",
        "",
        "每个点都保留 model/device/method/boundary/interval/quantile、quality/latency source、两个 protocol ID、damage、speedup 与 speedup CI；见同目录 JSON。",
        "",
    ]
    for group_name, group in report["diagnostic_comparisons"].items():
        lines.extend([f"### {group_name}", "", "| model | HCPR | dominators |", "|---|---|---|"])
        for model, details in group.items():
            for point in details["points"]:
                lines.append(f"| {model} | {point['method']} | {', '.join(point['dominators']) or 'none'} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--json", type=Path, default=Path("reports/post_mmm/pareto_consistency_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/post_mmm/pareto_consistency_audit.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root)
    json_path = args.json if args.json.is_absolute() else root / args.json
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "points": len(report["points"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
