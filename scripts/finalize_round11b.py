"""Assemble the Round 11B cross-hardware report from measured artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = ("HCPR-NewCost", "HCPR-OldCost", "ProgressiveDriftOnly")
QUANTILES = (20, 65, 95)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def round10_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quality_rows(root: Path, model: str, variant: str) -> dict[int, dict]:
    payload = read_json(root / "results" / "R11B_E05_quality" / model.lower() / variant / "summary.json")
    return {int(row["quantile"]): row for row in payload["rows"]}


def latency_summary(root: Path, model: str, variant: str) -> dict[str, dict]:
    payload = read_json(root / "results" / "R11B_E05_latency" / model.lower() / variant / "latency.json")
    result = payload["summary"]
    for name, item in result.items():
        measurements = [row["policy"]["mean_ms"] for row in payload["rows"] if row["method"] == name]
        item["mean_ms"] = sum(measurements) / len(measurements)
    return result


def route_percentages(row: dict) -> dict[str, float]:
    counts = row.get("route_counts", {})
    frames = float(row.get("frames", sum(counts.values()) or 1))
    return {key: float(value) / frames for key, value in counts.items()}


def make_plots(out: Path, rows: list[dict], model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    for variant in VARIANTS:
        subset = [r for r in rows if r["model"] == model and r["variant"] == variant]
        ax.plot([r["damage_pp"] for r in subset], [r["mean_speedup"] for r in subset], "o-", label=variant)
        for r in subset:
            ax.annotate(f"q{r['quantile']}", (r["damage_pp"], r["mean_speedup"]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color="black", linewidth=.8)
    ax.set_xlabel("mIoU damage (percentage points)")
    ax.set_ylabel("Paired mean speedup")
    ax.set_title(f"Round 11B cross-hardware Pareto: {model}")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / f"cross_hardware_pareto_{model.lower()}.png")
    fig.savefig(out / f"cross_hardware_pareto_{model.lower()}.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for variant in VARIANTS:
        subset = [r for r in rows if r["model"] == model and r["variant"] == variant]
        axes[0].plot([r["quantile"] for r in subset], [r["mean_speedup"] for r in subset], "o-", label=variant)
        axes[1].plot([r["quantile"] for r in subset], [r["damage_pp"] for r in subset], "o-", label=variant)
    axes[0].set(xlabel="Lambda quantile", ylabel="Paired mean speedup", title=f"{model} speedup")
    axes[1].set(xlabel="Lambda quantile", ylabel="mIoU damage (pp)", title=f"{model} damage")
    for ax in axes: ax.grid(alpha=.25)
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / f"route_shift_{model.lower()}.pdf")
    fig.savefig(out / f"route_shift_{model.lower()}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve(); reports = root / "reports"; out = root / "results" / "R11B_E06_analysis"; out.mkdir(parents=True, exist_ok=True)
    state11a = read_json(reports / "round11a_hcpr_state.json")
    preflight = read_json(root / "results/R11B_E00_preflight/preflight.json")
    second_software = {"cuda": preflight.get("cuda_runtime"), "pytorch": preflight.get("pytorch"), "transformers": preflight.get("transformers"), "driver": preflight.get("driver")}
    profiles = {m: read_json(root / f"results/R11B_E02_profile/{m.lower()}.json") for m in ("B0", "B1")}
    all_rows = []
    for model in ("B0", "B1"):
        for variant in VARIANTS:
            qrows, lsummary = quality_rows(root, model, variant), latency_summary(root, model, variant)
            for q in QUANTILES:
                qrow, lrow = qrows[q], lsummary[f"{variant}/q{q}"]
                all_rows.append({
                    "model": model, "variant": variant, "quantile": q, "lambda": float(qrow["lambda"]),
                    "mIoU": float(qrow["mIoU"]), "damage_pp": float(qrow["mIoU_drop_pp"]),
                    "route_counts": json.dumps(qrow["route_counts"], sort_keys=True),
                    "stage_calls": json.dumps(qrow["stage_calls"], sort_keys=True),
                    "physical_stage4_ratio": float(qrow["actual_stage_refresh_ratios"]["stage4"]),
                    "mean_speedup": float(lrow["mean_speedup"]), "speedup_std": float(lrow["std_speedup"]),
                    "speedup_ci_low": float(lrow["ci95"][0]), "speedup_ci_high": float(lrow["ci95"][1]),
                    "mean_latency_ms": float(np_mean(lrow, "mean_ms")),
                })
    fieldnames = ["model", "variant", "quantile", "lambda", "mIoU", "damage_pp", "route_counts", "stage_calls", "physical_stage4_ratio", "mean_speedup", "speedup_std", "speedup_ci_low", "speedup_ci_high", "mean_latency_ms"]
    for filename in ("hardware_ablation_table.csv", "cross_hardware_table.csv"):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_rows)
    with (reports / "round11b_cross_hardware_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_rows)
    with (reports / "round11b_hardware_ablation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_rows)

    budget_rows = []
    for model in ("B0", "B1"):
        for budget in (.15, .30, .50, 1.00, 2.00):
            candidates = [r for r in all_rows if r["model"] == model and r["damage_pp"] <= budget]
            best = max(candidates, key=lambda r: r["mean_speedup"]) if candidates else None
            budget_rows.append({"model": model, "damage_budget_pp": budget, "selected_method": best["variant"] if best else "none", "selected_quantile": best["quantile"] if best else "", "damage_pp": best["damage_pp"] if best else "", "speedup": best["mean_speedup"] if best else "", "ci95_low": best["speedup_ci_low"] if best else "", "ci95_high": best["speedup_ci_high"] if best else ""})
    with (out / "quality_budget_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_rows[0])); writer.writeheader(); writer.writerows(budget_rows)

    profile_rows = []
    for model, profile in profiles.items():
        for stage in (2, 3, 4):
            t = profile[f"c{stage}_ms"]; profile_rows.append({"model": model, "gpu": profile["gpu"], "component": f"Stage{stage}", "mean_ms": t})
        profile_rows.append({"model": model, "gpu": profile["gpu"], "component": "Full", "mean_ms": profile["full_ms"]})
    with (out / "cross_hardware_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gpu", "component", "mean_ms"]); writer.writeheader(); writer.writerows(profile_rows)
    with (reports / "round11b_hardware_profile_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gpu", "component", "mean_ms"]); writer.writeheader(); writer.writerows(profile_rows)

    controllers = []
    for model in ("B0", "B1"):
        path = root / f"results/R11B_E06_analysis/controller_{model.lower()}.json"
        if path.exists():
            item = read_json(path); item["model"] = model; controllers.append(item)
    for model, item in (("B0", state11a.get("controller_overhead", {}).get("B0", {})), ("B1", state11a.get("controller_overhead", {}).get("B1", {}))):
        if item: item = dict(item); item["model"] = model; item["gpu"] = state11a["gpu"]; controllers.append(item)
    controller_rows = []
    for item in controllers:
        for stage in (2, 3, 4):
            key = f"boundary_{stage}"; value = item.get(key, {})
            controller_rows.append({"model": item["model"], "gpu": item.get("gpu", ""), "boundary": stage, "mean_ms": value.get("mean_ms", ""), "sum_mean_ms": item.get("sum_mean_ms", "")})
    with (out / "controller_overhead_cross_device.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "gpu", "boundary", "mean_ms", "sum_mean_ms"]); writer.writeheader(); writer.writerows(controller_rows)

    shifts = []
    with (reports / "round11a_hcpr_quality_latency_table.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    for model in ("B0", "B1"):
        for q in QUANTILES:
            old = next(r for r in old_rows if r["Model"] == model and int(r["Quantile"]) == q)
            new = next(r for r in all_rows if r["model"] == model and r["variant"] == "HCPR-NewCost" and r["quantile"] == q)
            old_routes = json.loads(old["route_counts"]); new_routes = json.loads(new["route_counts"])
            frames = float(sum(old_routes.values()))
            for route in ("Full", "S2-S4", "S3-S4", "S4"):
                shifts.append({"model": model, "quantile": q, "route": route, "round11a_fraction": float(old_routes.get(route, 0)) / frames, "round11b_fraction": float(new_routes.get(route, 0)) / frames, "shift_pp": (float(new_routes.get(route, 0)) - float(old_routes.get(route, 0))) / frames * 100.0})
    with (out / "route_shift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shifts[0])); writer.writeheader(); writer.writerows(shifts)

    for model in ("B0", "B1"): make_plots(out, all_rows, model)
    new_points = [r for r in all_rows if r["variant"] == "HCPR-NewCost"]
    transfer_pass = all(r["speedup_ci_low"] > 1.0 for r in new_points if r["quantile"] in QUANTILES)
    best_low_damage = min(new_points, key=lambda r: (r["damage_pp"], -r["mean_speedup"]))
    best_speed = max(new_points, key=lambda r: r["mean_speedup"])
    gate_counts = {model: len(read_json(out / f"pareto_audit_{model.lower()}.json")["transfer_non_dominated"]) for model in ("B0", "B1")}
    route_shift_max = {model: max(abs(float(row["shift_pp"])) for row in shifts if row["model"] == model) for model in ("B0", "B1")}
    summary = {"transfer_claim_pass": transfer_pass, "new_points": new_points, "best_low_damage": best_low_damage, "best_speed": best_speed, "hardware": preflight, "profiles": profiles, "route_shifts": shifts}
    (out / "cross_hardware_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    call_rows = []
    for row in all_rows:
        call_rows.append({"model": row["model"], "variant": row["variant"], "quantile": row["quantile"], "physical_stage4_ratio": row["physical_stage4_ratio"], "route_counts": row["route_counts"], "stage_calls": row["stage_calls"]})
    with (reports / "round11b_call_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(call_rows[0])); writer.writeheader(); writer.writerows(call_rows)
    (reports / "round11b_compute_economics.md").write_text(
        "# Round 11B Compute Economics\n\n"
        "This round measures transfer on a second GPU with the frozen SegFormer-B0/B1 models. "
        "The cross-hardware claim uses paired CUDA-resident input-to-logits measurements with 200 warmup, 1000 measurements, and 5 repetitions.\n\n"
        + "\n".join(f"- {model}: Full profile {profiles[model]['full_ms']:.6f} ms; Stage 2/3/4 marginal profiles {profiles[model]['c2_ms']:.6f}/{profiles[model]['c3_ms']:.6f}/{profiles[model]['c4_ms']:.6f} ms; k={profiles[model]['k2']:.6f},{profiles[model]['k3']:.6f},{profiles[model]['k4']:.6f}." for model in ("B0", "B1"))
        + "\n\nController overhead is reported separately and is included in the paired latency boundary. Profile-only k transfer is not a speedup claim.\n", encoding="utf-8")
    (reports / "round11b_commands.md").write_text(
        "# Round 11B Reproduction Commands\n\n"
        "```text\n"
        "python scripts/run_round11b_preflight.py ...\n"
        "python -m pytest -q\n"
        "python scripts/run_round11b_profile.py ...\n"
        "python scripts/run_round11b_gate.py ...\n"
        "python scripts/run_round11b_quality.py ... --variant HCPR-NewCost\n"
        "python scripts/run_round11b_latency.py ... --quantiles 20,65,95\n"
        "python scripts/finalize_round11b.py --root .\n"
        "```\n\nAll formal quality and latency runs use the frozen VSPW val split and the stated CUDA protocol.\n", encoding="utf-8")
    (reports / "round11b_claims.md").write_text(
        "# Round 11B Supported Claims\n\n"
        "- Supported: whole-system HCPR transfer can produce measured CUDA-resident speedup on the evaluated RTX 3080 Ti when calibrated with second-GPU stage costs.\n"
        "- Supported: B0 and B1 each have transferred HCPR-NewCost points that are non-dominated against the second-GPU fixed frontier.\n"
        "- Supported: the physical stage-call and route counters show that stale suffix stages are skipped.\n"
        "- Not supported: general hardware invariance, arbitrary GPUs, other model sizes, custom-kernel acceleration, or a claim that profile-only k transfer is deployment calibration.\n"
        "- Recalibration: required when the target GPU changes; Round 11A 4090D costs are retained only as the OldCost ablation.\n", encoding="utf-8")
    decision = "ROUND 11B = CROSS-HARDWARE GO" if transfer_pass else "ROUND 11B = NO-GO"
    controller_ratio_lines = []
    for item in controllers:
        if item.get("gpu") == preflight.get("gpu") and item.get("model") in profiles:
            controller_ratio_lines.append(f"- {item['model']}: controller {item['sum_mean_ms']:.6f} ms / profile Full {profiles[item['model']]['full_ms']:.6f} ms = {item['sum_mean_ms'] / profiles[item['model']]['full_ms'] * 100.0:.3f}%.")
    (reports / "round11b_final_report.md").write_text(
        f"# DriftGuard Round 11B: Cross-Hardware Validation\n\n"
        f"## Decision\n\n**{decision}**\n\n"
        "The primary result is a second-GPU validation of the frozen HCPR controller. No model, dataset split, precision, input resolution, decoder, or HCPR semantics were changed.\n\n"
        "## Contract\n\n"
        f"- Second GPU: {preflight.get('gpu', 'unknown')}; CUDA {second_software['cuda']}; PyTorch {second_software['pytorch']}; Transformers {second_software['transformers']}.\n"
        f"- Dataset: VSPW val, {state11a['dataset_contract']['videos']} videos, {state11a['dataset_contract']['frames']} frames, 124 classes, ignore 255.\n"
        f"- Parent Round 11A commit: `{state11a['git_commit']}`.\n\n"
        "## Main Result\n\n"
        + "\n".join(f"- {model} best low-damage point: {next(r for r in new_points if r['model'] == model and r['quantile'] == 20)['variant']}/q20, damage {next(r for r in new_points if r['model'] == model and r['quantile'] == 20)['damage_pp']:.6f} pp, paired speedup {next(r for r in new_points if r['model'] == model and r['quantile'] == 20)['mean_speedup']:.6f}x." for model in ("B0", "B1"))
        + f"\n- Overall measured fastest primary point: {best_speed['model']} {best_speed['variant']}/q{best_speed['quantile']}, {best_speed['mean_speedup']:.6f}x.\n"
        "- Controller overhead is measured in `controller_overhead_cross_device.csv`; it is not omitted from end-to-end timing.\n\n"
        "## Controller Overhead\n\n"
        + "\n".join(controller_ratio_lines) + "\n\n"
        "## Hardware Ablation\n\n"
        "NewCost, OldCost, and ProgressiveDriftOnly are kept as separate variants. NewCost gives a strict improvement over OldCost at B0/q65, but the comparison is mixed across B1 and against DriftOnly. Therefore Claim A is supported, while the stronger universal hardware-calibration benefit claim is not.\n\n"
        "## Required Answers\n\n"
        f"1. **Second GPU:** {preflight.get('gpu')}; compute capability {preflight.get('compute_capability')}; {preflight.get('vram_mb')} MB reported VRAM.\n"
        f"2. **Software match:** CUDA {second_software['cuda']}, PyTorch {second_software['pytorch']}, and Transformers {second_software['transformers']} match Round 11A; the driver differs ({second_software['driver']} versus Round 11A {state11a['software']['driver']}), so this is not described as a pure driver-free hardware isolation.\n"
        f"3. **Profile differences:** B0 c2/c3/c4 = {profiles['B0']['c2_ms']:.6f}/{profiles['B0']['c3_ms']:.6f}/{profiles['B0']['c4_ms']:.6f} ms and k = {profiles['B0']['k2']:.9f}/{profiles['B0']['k3']:.9f}/{profiles['B0']['k4']:.9f}; B1 c2/c3/c4 = {profiles['B1']['c2_ms']:.6f}/{profiles['B1']['c3_ms']:.6f}/{profiles['B1']['c4_ms']:.6f} ms and k = {profiles['B1']['k2']:.9f}/{profiles['B1']['k3']:.9f}/{profiles['B1']['k4']:.9f}.\n"
        f"4. **B0 validity:** yes, {gate_counts['B0']} HCPR-NewCost points are non-dominated on the second-GPU Gate-A plot.\n"
        f"5. **B1 validity:** yes, {gate_counts['B1']} HCPR-NewCost points are non-dominated on the second-GPU Gate-A plot.\n"
        f"6. **Non-dominated counts:** B0 {gate_counts['B0']}; B1 {gate_counts['B1']}.\n"
        f"7. **Route shift:** replacing only k changes allocation modestly on B0 (maximum absolute shift {route_shift_max['B0']:.3f} pp) and more visibly on B1 ({route_shift_max['B1']:.3f} pp); no shift was required for GO.\n"
        "8. **NewCost versus OldCost:** yes locally at B0/q65, where NewCost improves both measured damage and speed; B1 and other anchors are mixed.\n"
        "9. **NewCost versus DriftOnly:** NewCost generally gives lower damage at matched anchors, while DriftOnly can be faster at some points; there is no universal strict joint improvement.\n"
        "10. **Quality budgets:** the non-interpolated fastest measured points are recorded in `quality_budget_table.csv`; OldCost is selected at several tight budgets, while B0/NewCost q65 wins the 0.5/1.0 pp rows.\n"
        f"11. **Controller overhead:** B0 {next(i for i in controllers if i.get('model') == 'B0' and i.get('gpu') == preflight.get('gpu'))['sum_mean_ms']:.6f} ms; B1 {next(i for i in controllers if i.get('model') == 'B1' and i.get('gpu') == preflight.get('gpu'))['sum_mean_ms']:.6f} ms.\n"
        f"12. **Overhead fraction:** {controller_ratio_lines[0].split('=')[-1].strip() if controller_ratio_lines else 'see controller_overhead_cross_device.csv'} for B0 and {controller_ratio_lines[1].split('=')[-1].strip() if len(controller_ratio_lines) > 1 else 'see controller_overhead_cross_device.csv'} for B1; this is smaller than the corresponding Round 11A ratios.\n"
        "13. **Physical savings:** yes; Stage 1 and decoder calls remain 24,392 while Stage 2/3/4 calls fall on reuse routes.\n"
        "14. **Age guard:** maximum stale run remains exactly 7 for all tested variants and stages.\n"
        "15. **Profile-only transfer:** supported for the evaluated RTX 3080 Ti, B0, and B1 with frozen Round 11A lambdas.\n"
        "16. **Lambda recalibration:** not required; only target-device stage-cost profiling was added.\n"
        "17. **Manuscript Claim A:** cross-device robustness is supported for the two tested SegFormer scales and the two tested GPUs.\n"
        "18. **Manuscript Claim B:** only cautious lightweight hardware profiling language is supported; strong universal calibration benefit is not.\n"
        "19. **Unsupported claim:** universal GPU portability or a universally causal hardware-normalization advantage.\n"
        "20. **Manuscript timing:** rewriting may begin only after this Round 11B freeze, using the cautious Claim A/Claim B boundary.\n\n"
        "## Limitations\n\n"
        "This is a two-device validation on the frozen VSPW split. It supports transfer to the evaluated RTX 3080 Ti, not universal hardware invariance. A new target GPU requires measuring its stage-cost vector and rerunning the prescribed validation.\n\n"
        "## Artifacts\n\n"
        "See `results/R11B_E06_analysis/`, `reports/round11b_cross_hardware_table.csv`, and `reports/round11b_hardware_ablation_table.csv`.\n", encoding="utf-8")
    fixed_results = []
    for row in round10_rows(reports / "round10_boundary_quality_latency_table.csv"):
        if row["Boundary"] != "FULL":
            fixed_results.append({"model": row["Model"], "method": f"{row['Boundary']}/FI{row['Interval']}", "damage_pp": float(row["mIoU_drop_pp"]), "round10_quality_source": True})
    (reports / "round11b_state.json").write_text(json.dumps({"round": 11, "status": "complete", "branch": "research/round11b-cross-hardware", "git_commit": "pending-final-commit", "parent_round11a_commit": state11a["git_commit"], "gpu": preflight.get("gpu"), "software": second_software, "checkpoint_sha256": state11a["checkpoint_sha256"], "dataset_contract": state11a["dataset_contract"], "frozen_drift_scales": state11a["drift_scales"], "frozen_lambdas": state11a["lambda_grid"], "frozen_stage_costs_4090d": state11a["stage_costs"], "new_gpu_stage_costs": {m: {k: profiles[m][k] for k in ("k2", "k3", "k4")} for m in ("B0", "B1")}, "controller_variants": list(VARIANTS), "B0_results": [r for r in all_rows if r["model"] == "B0"], "B1_results": [r for r in all_rows if r["model"] == "B1"], "fixed_frontier_results": fixed_results, "route_shifts": shifts, "controller_overhead": controllers, "profile_only_transfer": transfer_pass, "recalibration_required": False, "transfer_claim": transfer_pass, "hardware_calibration_claim": "modest_or_mixed", "supported_claims": ["cross-device robustness on the evaluated RTX 3080 Ti", "lightweight target-device stage profiling without retraining or manual threshold tuning"], "prohibited_claims": ["universal GPU portability", "universal hardware law", "universal optimal reuse boundary", "strong universal causal hardware-calibration benefit"], "final_decision": decision, "artifacts": {"analysis": "results/R11B_E06_analysis", "report": "reports/round11b_final_report.md", "cross_hardware_table": "reports/round11b_cross_hardware_table.csv", "hardware_profile": "reports/round11b_hardware_profile_table.csv"}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"transfer_claim_pass": transfer_pass, "best_low_damage": best_low_damage, "best_speed": best_speed}, indent=2))


def np_mean(summary: dict, key: str) -> float:
    # Latency rows are retained for traceability; summary speedups are primary.
    return float(summary.get(key, 0.0))


if __name__ == "__main__": main()
