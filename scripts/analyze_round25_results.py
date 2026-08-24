"""Analyze Round 2.5 selector closure outputs and lock the final outcome."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CORE_RATIOS = (0.25, 0.375, 0.5)
RATIOS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0)
METHODS = (
    "random",
    "residual_mean",
    "residual_p90",
    "age",
    "old_tiny_mlp",
    "cnn_all",
    "cnn_patch_age",
    "oracle",
)
LEARNED = ("old_tiny_mlp", "cnn_all", "cnn_patch_age")
DISPLAY = {
    "random": "Random",
    "residual_mean": "Residual Mean",
    "residual_p90": "Residual P90",
    "age": "Age",
    "old_tiny_mlp": "Old Tiny MLP",
    "cnn_all": "CNN All",
    "cnn_patch_age": "CNN Patch+Age",
    "oracle": "Oracle",
}
PARAMS = {
    "random": 0,
    "residual_mean": 0,
    "residual_p90": 0,
    "age": 0,
    "old_tiny_mlp": 737,
    "cnn_all": 5865,
    "cnn_patch_age": 5737,
    "oracle": "",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(rows: list[dict], fields: tuple[str, ...]):
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        groups.setdefault(key, []).append(row)
    return sorted(groups.items())


def _policy_metrics(args: argparse.Namespace) -> list[dict]:
    original = _read_csv(PROJECT_ROOT / args.closure_root / "policy_metrics.csv")
    corrected_path = (
        PROJECT_ROOT / args.diagnostics_root / "corrected_policy_metrics.csv"
    )
    if not corrected_path.exists():
        return original
    corrected = _read_csv(corrected_path)
    return [row for row in original if row["policy"] == "random"] + corrected


def _per_video_seed(rows: list[dict]) -> list[dict]:
    output = []
    for (video_id, policy, ratio, seed), values in _group_rows(
        rows, ("video_id", "policy", "refresh_ratio", "seed")
    ):
        output.append(
            {
                "video_id": video_id,
                "motion_level": values[0]["motion_level"],
                "policy": policy,
                "refresh_ratio": float(ratio),
                "seed": int(seed),
                "frame_count": len(values),
                "logit_relative_l2_mean": float(
                    np.mean([float(row["logit_relative_l2"]) for row in values])
                ),
                "global_kl_mean": float(
                    np.mean([float(row["global_kl"]) for row in values])
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [
                            float(row["global_pixel_disagreement"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def _per_video(seed_rows: list[dict]) -> list[dict]:
    output = []
    for (video_id, policy, ratio), values in _group_rows(
        seed_rows, ("video_id", "policy", "refresh_ratio")
    ):
        damage = np.asarray(
            [float(row["logit_relative_l2_mean"]) for row in values]
        )
        output.append(
            {
                "video_id": video_id,
                "motion_level": values[0]["motion_level"],
                "policy": policy,
                "method": DISPLAY[policy],
                "refresh_ratio": float(ratio),
                "seed_count": len(values),
                "logit_relative_l2_mean": float(damage.mean()),
                "logit_relative_l2_seed_std": float(damage.std()),
                "global_kl_mean": float(
                    np.mean([float(row["global_kl_mean"]) for row in values])
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [float(row["pixel_disagreement_mean"]) for row in values]
                    )
                ),
            }
        )
    return output


def _summary(per_video: list[dict]) -> list[dict]:
    output = []
    for (policy, ratio), values in _group_rows(
        per_video, ("policy", "refresh_ratio")
    ):
        output.append(
            {
                "policy": policy,
                "method": DISPLAY[policy],
                "refresh_ratio": float(ratio),
                "video_count": len(values),
                "logit_relative_l2_mean": float(
                    np.mean(
                        [float(row["logit_relative_l2_mean"]) for row in values]
                    )
                ),
                "global_kl_mean": float(
                    np.mean([float(row["global_kl_mean"]) for row in values])
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [float(row["pixel_disagreement_mean"]) for row in values]
                    )
                ),
            }
        )
    return output


def _motion_conditioned(per_video: list[dict]) -> list[dict]:
    output = []
    for (motion, policy, ratio), values in _group_rows(
        per_video, ("motion_level", "policy", "refresh_ratio")
    ):
        output.append(
            {
                "motion_level": motion,
                "policy": policy,
                "method": DISPLAY[policy],
                "refresh_ratio": float(ratio),
                "video_count": len(values),
                "logit_relative_l2_mean": float(
                    np.mean(
                        [float(row["logit_relative_l2_mean"]) for row in values]
                    )
                ),
                "global_kl_mean": float(
                    np.mean([float(row["global_kl_mean"]) for row in values])
                ),
                "pixel_disagreement_mean": float(
                    np.mean(
                        [float(row["pixel_disagreement_mean"]) for row in values]
                    )
                ),
            }
        )
    return output


def _load_static_scores(args: argparse.Namespace):
    import torch

    from driftguard.models.risk_predictor import TinyRiskCNN
    from train_round25_old_mlp import HistoricalTinyMLP

    inventory = {
        row["video_id"]: row
        for row in _read_csv(PROJECT_ROOT / args.inventory)
        if row["split"] == "test"
    }
    shards = []
    for video_id, record in inventory.items():
        shard = np.load(
            PROJECT_ROOT
            / args.dataset_root
            / "shards"
            / f"{video_id}.npz",
            allow_pickle=False,
        )
        row = {key: shard[key] for key in shard.files}
        count = len(row["task_damage"])
        row["video_id"] = np.asarray([video_id] * count)
        row["motion_level"] = np.asarray([record["motion_level"]] * count)
        shards.append(row)
    keys = (
        "patch",
        "metadata",
        "task_damage",
        "frame_id",
        "cache_age",
        "block_index",
        "video_id",
        "motion_level",
    )
    data = {key: np.concatenate([row[key] for row in shards]) for key in keys}
    threshold = float(
        json.loads(
            (
                PROJECT_ROOT
                / args.dataset_root
                / "thresholds_train_only.json"
            ).read_text(encoding="utf-8")
        )["p80"]
    )
    y = (data["task_damage"] > threshold).astype(np.int32)

    old_ckpt = torch.load(PROJECT_ROOT / args.old_mlp, map_location="cpu")
    old = HistoricalTinyMLP(len(old_ckpt["feature_columns"]))
    old.load_state_dict(old_ckpt["state_dict"])
    old.eval()
    old_values = data["metadata"][:, [3, 0, 1, 2, 4]]
    old_mean = np.asarray(old_ckpt["mean"], dtype=np.float32)
    old_std = np.asarray(old_ckpt["std"], dtype=np.float32)
    old_std[old_std < 1e-8] = 1.0
    with torch.inference_mode():
        old_score = old(
            torch.from_numpy((old_values - old_mean) / old_std)
        ).numpy()

    def load_cnn(path: str, metadata_dim: int):
        checkpoint = torch.load(PROJECT_ROOT / path, map_location="cpu")
        model = TinyRiskCNN(metadata_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean = np.asarray(checkpoint["metadata_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["metadata_std"], dtype=np.float32)
        std[std < 1e-8] = 1.0
        indices = list(checkpoint.get("metadata_indices", range(metadata_dim)))
        with torch.inference_mode():
            return model(
                torch.from_numpy(data["patch"]),
                torch.from_numpy(
                    (data["metadata"][:, indices] - mean) / std
                ),
            ).numpy()

    scores = {
        "random": np.random.default_rng(42).random(len(y)).astype(np.float32),
        "residual_mean": data["metadata"][:, 3],
        "residual_p90": np.quantile(
            data["patch"].reshape(len(data["patch"]), -1), 0.90, axis=1
        ),
        "age": data["cache_age"].astype(np.float32),
        "old_tiny_mlp": old_score,
        "cnn_all": load_cnn(args.cnn_all, 5),
        "cnn_patch_age": load_cnn(args.cnn_patch_age, 1),
        "oracle": data["task_damage"],
    }
    return data, y, scores


def _top_indices(score: np.ndarray, count: int) -> np.ndarray:
    return np.lexsort((np.arange(len(score)), -score))[:count]


def _static_quality(args: argparse.Namespace):
    data, y, scores = _load_static_scores(args)
    groups = {}
    for index in range(len(y)):
        key = (
            str(data["video_id"][index]),
            int(data["frame_id"][index]),
            int(data["cache_age"][index]),
        )
        groups.setdefault(key, []).append(index)
    output = {}
    for method, score in scores.items():
        selected = np.zeros(len(y), dtype=bool)
        static_overlaps = {4: [], 6: [], 8: []}
        for indices in groups.values():
            indices = np.asarray(indices, dtype=np.int64)
            local_score = score[indices]
            local_damage = data["task_damage"][indices]
            selected[indices[_top_indices(local_score, 4)]] = True
            for count in static_overlaps:
                predicted = set(
                    indices[_top_indices(local_score, count)].tolist()
                )
                oracle = set(
                    indices[_top_indices(local_damage, count)].tolist()
                )
                static_overlaps[count].append(len(predicted & oracle) / count)
        true_positive = int(y[selected].sum())
        output[method] = {
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
            "spearman_task_damage": float(
                spearmanr(score, data["task_damage"]).statistic
            ),
            "precision_at_25": float(true_positive / max(1, selected.sum())),
            "recall_at_25": float(true_positive / max(1, y.sum())),
            "static_top4": float(np.mean(static_overlaps[4])),
            "static_top6": float(np.mean(static_overlaps[6])),
            "static_top8": float(np.mean(static_overlaps[8])),
        }
    return data, output


def _sequential_overlap(rows: list[dict]) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, list[float]]] = {}
    for (_, _, policy, _), group in _group_rows(
        rows, ("video_id", "frame_id", "policy", "refresh_ratio")
    ):
        score = np.asarray([float(row["predictor_score"]) for row in group])
        damage = np.asarray([float(row["oracle_task_damage"]) for row in group])
        values.setdefault(policy, {4: [], 6: [], 8: []})
        for count in (4, 6, 8):
            predicted = set(_top_indices(score, count).tolist())
            oracle = set(_top_indices(damage, count).tolist())
            values[policy][count].append(len(predicted & oracle) / count)
    return {
        policy: {
            count: float(np.mean(overlaps))
            for count, overlaps in counts.items()
        }
        for policy, counts in values.items()
    }


def _selector_quality(
    args: argparse.Namespace, diagnostic_rows: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    data, static = _static_quality(args)
    sequential = _sequential_overlap(diagnostic_rows)
    rows = []
    for method in METHODS:
        overlap = sequential.get(method)
        rows.append(
            {
                "method": DISPLAY[method],
                "method_key": method,
                "auroc": static[method]["auroc"],
                "auprc": static[method]["auprc"],
                "spearman_task_damage": static[method][
                    "spearman_task_damage"
                ],
                "precision_at_25": static[method]["precision_at_25"],
                "recall_at_25": static[method]["recall_at_25"],
                "oracle_top4_overlap": (
                    1.0
                    if method == "oracle"
                    else overlap[4]
                    if overlap
                    else static[method]["static_top4"]
                ),
                "oracle_top6_overlap": (
                    1.0
                    if method == "oracle"
                    else overlap[6]
                    if overlap
                    else static[method]["static_top6"]
                ),
                "oracle_top8_overlap": (
                    1.0
                    if method == "oracle"
                    else overlap[8]
                    if overlap
                    else static[method]["static_top8"]
                ),
                "overlap_source": (
                    "perfect_reference"
                    if method == "oracle"
                    else "sequential_same_cache"
                    if overlap
                    else "static_task_dataset"
                ),
            }
        )
    return data, rows


def _age_diagnostic(rows: list[dict]) -> list[dict]:
    age_rows = [row for row in rows if row["policy"] == "age"]
    output = []
    ages = sorted({int(float(row["age_before"])) for row in age_rows})
    for age in ages:
        values = [
            float(row["oracle_task_damage"])
            for row in age_rows
            if int(float(row["age_before"])) == age
        ]
        output.append(
            {
                "kind": "damage_by_age",
                "motion_level": "all",
                "refresh_ratio": "all_core",
                "age": age,
                "count": len(values),
                "fraction": "",
                "mean_task_damage": float(np.mean(values)),
                "spearman_age_damage": "",
            }
        )
    for motion in sorted({row["motion_level"] for row in age_rows}):
        values = [row for row in age_rows if row["motion_level"] == motion]
        output.append(
            {
                "kind": "motion_correlation",
                "motion_level": motion,
                "refresh_ratio": "all_core",
                "age": -1,
                "count": len(values),
                "fraction": "",
                "mean_task_damage": float(
                    np.mean(
                        [float(row["oracle_task_damage"]) for row in values]
                    )
                ),
                "spearman_age_damage": float(
                    spearmanr(
                        [float(row["age_before"]) for row in values],
                        [float(row["oracle_task_damage"]) for row in values],
                    ).statistic
                ),
            }
        )

    selected_ages = []
    representative = [
        row for row in age_rows if float(row["refresh_ratio"]) == 0.375
    ]
    for _, group in _group_rows(representative, ("video_id", "frame_id")):
        damage = np.asarray(
            [float(row["oracle_task_damage"]) for row in group]
        )
        for index in _top_indices(damage, 6):
            selected_ages.append(int(float(group[index]["age_before"])))
    for age in sorted(set(selected_ages)):
        count = selected_ages.count(age)
        output.append(
            {
                "kind": "oracle_top6_age_distribution",
                "motion_level": "all",
                "refresh_ratio": 0.375,
                "age": age,
                "count": count,
                "fraction": float(count / len(selected_ages)),
                "mean_task_damage": "",
                "spearman_age_damage": "",
            }
        )
    return output


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(1000, len(values)), replace=True)
    means = sampled.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_rows(per_video: list[dict]) -> list[dict]:
    comparisons = (
        ("old_tiny_mlp", "age"),
        ("cnn_patch_age", "age"),
        ("cnn_all", "age"),
        ("old_tiny_mlp", "residual_mean"),
        ("cnn_patch_age", "residual_mean"),
    )
    output = []
    for candidate, baseline in comparisons:
        for ratio in CORE_RATIOS:
            candidate_values = {
                row["video_id"]: float(row["logit_relative_l2_mean"])
                for row in per_video
                if row["policy"] == candidate
                and float(row["refresh_ratio"]) == ratio
            }
            baseline_values = {
                row["video_id"]: float(row["logit_relative_l2_mean"])
                for row in per_video
                if row["policy"] == baseline
                and float(row["refresh_ratio"]) == ratio
            }
            differences = np.asarray(
                [
                    candidate_values[key] - baseline_values[key]
                    for key in sorted(candidate_values)
                ],
                dtype=np.float64,
            )
            low, high = _bootstrap(
                differences,
                seed=42 + int(ratio * 1000) + len(output),
            )
            output.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "comparison": f"{candidate}_minus_{baseline}",
                    "refresh_ratio": ratio,
                    "video_count": len(differences),
                    "mean_delta": float(differences.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "better_videos": int((differences < 0).sum()),
                }
            )
    return output


def _lookup_summary(summary: list[dict], policy: str, ratio: float) -> dict:
    return next(
        row
        for row in summary
        if row["policy"] == policy
        and float(row["refresh_ratio"]) == ratio
    )


def _main_table(
    summary: list[dict], overhead: dict[str, dict]
) -> list[dict]:
    output = []
    for method in METHODS:
        values = [
            _lookup_summary(summary, method, ratio)
            for ratio in CORE_RATIOS
        ]
        output.append(
            {
                "method": DISPLAY[method],
                "method_key": method,
                "params": PARAMS[method],
                "selector_latency_ms": overhead.get(
                    DISPLAY[method], {}
                ).get("total_selector_ms", ""),
                "refresh_25": values[0]["logit_relative_l2_mean"],
                "refresh_37_5": values[1]["logit_relative_l2_mean"],
                "refresh_50": values[2]["logit_relative_l2_mean"],
                "kl_37_5": values[1]["global_kl_mean"],
                "pixel_disagreement_37_5": values[1][
                    "pixel_disagreement_mean"
                ],
            }
        )
    return output


def _decision(
    main_rows: list[dict],
    per_video: list[dict],
    overhead: dict[str, dict],
    bootstrap_rows: list[dict],
) -> dict:
    damage = {
        row["method_key"]: np.asarray(
            [
                float(row["refresh_25"]),
                float(row["refresh_37_5"]),
                float(row["refresh_50"]),
            ]
        )
        for row in main_rows
    }
    age = damage["age"]
    learned_evidence = {}
    for method in LEARNED:
        relative = (age - damage[method]) / age
        video_pairs = []
        for ratio in CORE_RATIOS:
            candidates = {
                row["video_id"]: float(row["logit_relative_l2_mean"])
                for row in per_video
                if row["policy"] == method
                and float(row["refresh_ratio"]) == ratio
            }
            ages = {
                row["video_id"]: float(row["logit_relative_l2_mean"])
                for row in per_video
                if row["policy"] == "age"
                and float(row["refresh_ratio"]) == ratio
            }
            video_pairs.extend(
                candidates[key] < ages[key] for key in sorted(candidates)
            )
        boot = [
            row
            for row in bootstrap_rows
            if row["candidate"] == method and row["baseline"] == "age"
        ]
        learned_evidence[method] = {
            "core_ratio_wins": int((damage[method] < age).sum()),
            "relative_reduction_by_ratio": relative.tolist(),
            "mean_relative_reduction": float(relative.mean()),
            "video_ratio_wins": int(sum(video_pairs)),
            "video_ratio_comparisons": len(video_pairs),
            "negative_bootstrap_mean_count": int(
                sum(float(row["mean_delta"]) < 0 for row in boot)
            ),
            "strict_negative_ci_count": int(
                sum(float(row["bootstrap_ci_high"]) < 0 for row in boot)
            ),
            "selector_latency_ms": float(
                overhead[DISPLAY[method]]["total_selector_ms"]
            ),
        }

    best_learned = min(LEARNED, key=lambda method: damage[method].mean())
    evidence = learned_evidence[best_learned]
    age_latency = float(overhead["Age"]["total_selector_ms"])
    latency_ratio = evidence["selector_latency_ms"] / max(age_latency, 1e-9)
    evidence["latency_vs_age_ratio"] = latency_ratio

    clear_learned_win = (
        evidence["core_ratio_wins"] == 3
        and min(evidence["relative_reduction_by_ratio"]) >= 0.02
        and evidence["video_ratio_wins"] >= 6
        and evidence["negative_bootstrap_mean_count"] == 3
    )
    age_dominant = all(
        learned_evidence[method]["core_ratio_wins"] <= 1
        and learned_evidence[method]["mean_relative_reduction"] <= 0.0
        for method in LEARNED
    )
    if clear_learned_win:
        final_selector = {
            "old_tiny_mlp": "TaskRiskMLP",
            "cnn_all": "TinyRiskCNN All",
            "cnn_patch_age": "TinyRiskCNN Patch+Age",
        }[best_learned]
        conclusion = "LOCK AND GO"
        outcome = "learned_selector_clear_win"
    elif age_dominant:
        final_selector = "Age"
        conclusion = "AGE-DOMINANT PIVOT"
        outcome = "age_dominant"
    else:
        final_selector = "Marginal / No Clear Winner"
        conclusion = "MARGINAL / MORE SCIENTIFIC WORK REQUIRED"
        outcome = "mixed_task_gain_with_cost_or_stability_tradeoff"

    return {
        "final_selector": final_selector,
        "scientific_conclusion": conclusion,
        "outcome": outcome,
        "best_learned_candidate": DISPLAY[best_learned],
        "best_learned_evidence": evidence,
        "all_learned_evidence": learned_evidence,
        "decision_rule": (
            "held-out task damage, cross-video stability, selector overhead, "
            "then parameter complexity; <2% or inconsistent gains are marginal"
        ),
    }


def _plot_refresh(summary: list[dict], path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for policy in METHODS:
        values = sorted(
            [row for row in summary if row["policy"] == policy],
            key=lambda row: float(row["refresh_ratio"]),
        )
        plt.plot(
            [float(row["refresh_ratio"]) * 100 for row in values],
            [float(row["logit_relative_l2_mean"]) for row in values],
            marker="o",
            linewidth=1.8,
            label=DISPLAY[policy],
        )
    plt.xlabel("Refresh budget (%)")
    plt.ylabel("Mean test LogitRelativeL2")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_overhead(
    main_rows: list[dict], overhead: dict[str, dict], path: Path
) -> None:
    plt.figure(figsize=(7.5, 5.2))
    for row in main_rows:
        method = row["method"]
        if method not in overhead:
            continue
        x = float(overhead[method]["total_selector_ms"])
        y = float(row["refresh_37_5"])
        plt.scatter(x, y, s=55)
        plt.annotate(method, (x, y), xytext=(5, 4), textcoords="offset points")
    plt.xlabel("Total selector latency (ms/frame)")
    plt.ylabel("LogitRelativeL2 at 37.5% refresh")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_age(rows: list[dict], path: Path) -> None:
    age_rows = [row for row in rows if row["policy"] == "age"]
    x = np.asarray([float(row["age_before"]) for row in age_rows])
    y = np.asarray([float(row["oracle_task_damage"]) for row in age_rows])
    plt.figure(figsize=(7.5, 5.2))
    rng = np.random.default_rng(42)
    if len(x) > 5000:
        indices = rng.choice(len(x), 5000, replace=False)
    else:
        indices = np.arange(len(x))
    plt.scatter(x[indices], y[indices], s=7, alpha=0.12, label="blocks")
    ages = sorted(set(x.tolist()))
    means = [float(y[x == age].mean()) for age in ages]
    plt.plot(ages, means, color="black", marker="o", label="mean damage")
    plt.xlabel("Cache age before refresh")
    plt.ylabel("Single-block task damage")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_overlap(quality: list[dict], path: Path) -> None:
    methods = [
        row
        for row in quality
        if row["method_key"] not in {"random", "oracle"}
    ]
    x = np.arange(len(methods))
    width = 0.25
    plt.figure(figsize=(10, 5.4))
    for offset, count in enumerate((4, 6, 8)):
        plt.bar(
            x + (offset - 1) * width,
            [float(row[f"oracle_top{count}_overlap"]) for row in methods],
            width,
            label=f"Top-{count}",
        )
    plt.xticks(x, [row["method"] for row in methods], rotation=20, ha="right")
    plt.ylabel("Overlap with same-cache Oracle")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closure-root", default="results/R25_E01_selector_closure"
    )
    parser.add_argument(
        "--dataset-root", default="results/R2_E01_task_damage_dataset"
    )
    parser.add_argument(
        "--inventory",
        default="results/R2_E00_video_dataset/video_inventory.csv",
    )
    parser.add_argument(
        "--old-mlp",
        default="results/R25_E01_selector_closure/old_tiny_mlp_taskrisk.pt",
    )
    parser.add_argument("--cnn-all", default="results/R2_E02_predictor/best_model.pt")
    parser.add_argument(
        "--cnn-patch-age",
        default="results/R2_E03_ablation/patch_age.pt",
    )
    parser.add_argument(
        "--profiling-root", default="results/R25_E03_selector_profiling"
    )
    parser.add_argument(
        "--diagnostics-root", default="results/R25_E02_selector_diagnostics"
    )
    args = parser.parse_args()

    closure_root = PROJECT_ROOT / args.closure_root
    diagnostics_root = PROJECT_ROOT / args.diagnostics_root
    profiling_root = PROJECT_ROOT / args.profiling_root
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    policy_rows = _policy_metrics(args)
    seed_rows = _per_video_seed(policy_rows)
    per_video = _per_video(seed_rows)
    summary = _summary(per_video)
    motion_rows = _motion_conditioned(per_video)
    _write_csv(closure_root / "per_video_seed_results.csv", seed_rows)
    _write_csv(closure_root / "per_video_results.csv", per_video)
    _write_csv(closure_root / "motion_conditioned.csv", motion_rows)
    _write_csv(closure_root / "corrected_summary.csv", summary)

    overhead_rows = _read_csv(profiling_root / "selector_overhead.csv")
    overhead = {row["method"]: row for row in overhead_rows}
    main_rows = _main_table(summary, overhead)
    _write_csv(closure_root / "main_refresh_table.csv", main_rows)

    diagnostic_rows = _read_csv(
        diagnostics_root / "sequential_block_diagnostics.csv"
    )
    _, quality = _selector_quality(args, diagnostic_rows)
    _write_csv(diagnostics_root / "selector_quality.csv", quality)
    age_rows = _age_diagnostic(diagnostic_rows)
    _write_csv(diagnostics_root / "age_diagnostic.csv", age_rows)
    bootstrap_rows = _bootstrap_rows(per_video)
    _write_csv(diagnostics_root / "bootstrap_comparisons.csv", bootstrap_rows)
    _write_csv(
        diagnostics_root / "bootstrap_vs_age.csv",
        [row for row in bootstrap_rows if row["baseline"] == "age"],
    )

    complexity = []
    required_inputs = {
        "Residual Mean": "current RGB, previous RGB",
        "Residual P90": "16 residual patches",
        "Age": "16 block ages",
        "Old Tiny MLP": "residual stats, age, entropy",
        "CNN All": "16 residual patches, age, entropy, residual stats",
        "CNN Patch+Age": "16 residual patches, 16 block ages",
    }
    for row in overhead_rows:
        complexity.append(
            {
                "method": row["method"],
                "learned": row["method"]
                in {"Old Tiny MLP", "CNN All", "CNN Patch+Age"},
                "params": row["params"],
                "required_inputs": required_inputs[row["method"]],
                "model_forward_ms": row["model_forward_ms"],
                "total_selector_ms": row["total_selector_ms"],
            }
        )
    _write_csv(profiling_root / "selector_complexity.csv", complexity)

    decision = _decision(main_rows, per_video, overhead, bootstrap_rows)
    (diagnostics_root / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    _plot_refresh(
        summary, closure_root / "refresh_budget_vs_task_damage.png"
    )
    _plot_overhead(
        main_rows,
        overhead,
        profiling_root / "selector_overhead_vs_damage.png",
    )
    _plot_age(
        diagnostic_rows, diagnostics_root / "age_vs_task_damage.png"
    )
    _plot_overlap(
        quality, diagnostics_root / "oracle_overlap_by_policy.png"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
