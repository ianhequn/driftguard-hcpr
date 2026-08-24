"""Minimal TinyRiskCNN metadata ablation after predictor-level success."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_round2_predictor import (  # noqa: E402
    _best_f1_threshold,
    _concat,
    _fit_standardizer,
    _metrics,
    _predict,
    _read_shards,
    _train_cnn,
)
from driftguard.models.risk_predictor import parameter_count  # noqa: E402
from driftguard.utils.common import choose_device, set_seed  # noqa: E402


VARIANTS = {
    "patch_only": [],
    "patch_age": [0],
    "patch_age_entropy": [0, 1],
    "patch_all_metadata": [0, 1, 2, 3, 4],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="results/R2_E01_task_damage_dataset")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--output-root", default="results/R2_E03_ablation")
    parser.add_argument("--threshold", choices=["p70", "p80", "p90"], default="p80")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _slice(values: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    output = dict(values)
    output["metadata"] = values["metadata"][:, indices]
    return output


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    root = PROJECT_ROOT / args.dataset_root
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shards = _read_shards(root, PROJECT_ROOT / args.inventory)
    train, val, test = (_concat(shards, split) for split in ("train", "val", "test"))
    thresholds = json.loads((root / "thresholds_train_only.json").read_text(encoding="utf-8"))
    task_threshold = float(thresholds[args.threshold])
    y_train = (train["task_damage"] > task_threshold).astype(np.int32)
    y_val = (val["task_damage"] > task_threshold).astype(np.int32)
    y_test = (test["task_damage"] > task_threshold).astype(np.int32)

    rows = []
    for name, indices in VARIANTS.items():
        variant_train = _slice(train, indices)
        variant_val = _slice(val, indices)
        variant_test = _slice(test, indices)
        mean, std = _fit_standardizer(variant_train["metadata"])
        model, log = _train_cnn(
            variant_train,
            variant_val,
            y_train,
            y_val,
            mean,
            std,
            device,
            args.seed,
        )
        val_score = _predict(model, variant_val, mean, std, device)
        test_score = _predict(model, variant_test, mean, std, device)
        decision_threshold = _best_f1_threshold(y_val, val_score)
        rows.append(
            {
                "variant": name,
                "metadata_dim": len(indices),
                "parameters": parameter_count(model),
                **_metrics(y_test, test_score, test["task_damage"], decision_threshold),
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata_indices": indices,
                "metadata_mean": mean,
                "metadata_std": std,
                "threshold": task_threshold,
                "decision_threshold": decision_threshold,
            },
            output_root / f"{name}.pt",
        )

    with (output_root / "ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "config.yaml").write_text(
        f"dataset_root: {args.dataset_root}\nthreshold: {args.threshold}\ndevice: {device}\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
