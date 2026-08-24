"""Restore the frozen Old Tiny MLP as a Round 2 task-risk classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_round2_predictor import _concat, _read_shards  # noqa: E402
from driftguard.utils.common import choose_device, set_seed  # noqa: E402


OLD_COLUMNS = [
    "local_residual",
    "cache_age",
    "previous_entropy",
    "global_residual",
    "local_residual_variance",
]
ROUND2_TO_OLD = [3, 0, 1, 2, 4]


class HistoricalTinyMLP(torch.nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="results/R2_E01_task_damage_dataset")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--output", default="results/R25_E01_selector_closure/old_tiny_mlp_taskrisk.pt")
    parser.add_argument("--threshold", choices=["p70", "p80", "p90"], default="p80")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    root = PROJECT_ROOT / args.dataset_root
    shards = _read_shards(root, PROJECT_ROOT / args.inventory)
    train, val, _ = (_concat(shards, split) for split in ("train", "val", "test"))
    thresholds = json.loads((root / "thresholds_train_only.json").read_text(encoding="utf-8"))
    task_threshold = float(thresholds[args.threshold])

    train_values = train["metadata"][:, ROUND2_TO_OLD].astype(np.float32)
    val_values = val["metadata"][:, ROUND2_TO_OLD].astype(np.float32)
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    y_train = (train["task_damage"] > task_threshold).astype(np.float32)
    y_val = (val["task_damage"] > task_threshold).astype(np.float32)

    model = HistoricalTinyMLP().to(device)
    positive_weight = max(
        1.0,
        float((len(y_train) - y_train.sum()) / max(1.0, y_train.sum())),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(_standardize(train_values, mean, std)),
            torch.from_numpy(y_train),
        ),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(_standardize(val_values, mean, std)),
            torch.from_numpy(y_val),
        ),
        batch_size=512,
        shuffle=False,
    )

    best_score = -np.inf
    best_state = None
    stale_epochs = 0
    log = []
    for epoch in range(1, 101):
        model.train()
        losses = []
        for values, target in train_loader:
            values = values.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(values).unsqueeze(1), target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        scores, targets = [], []
        with torch.inference_mode():
            for values, target in val_loader:
                scores.append(torch.sigmoid(model(values.to(device))).cpu().numpy())
                targets.append(target.numpy())
        val_score = average_precision_score(
            np.concatenate(targets),
            np.concatenate(scores),
        )
        log.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_auprc": float(val_score),
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break

    if best_state is None:
        raise RuntimeError("No Old Tiny MLP checkpoint was produced")
    model.load_state_dict(best_state)
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": OLD_COLUMNS,
            "metadata_order": "local,age,entropy,global,local_variance",
            "mean": mean,
            "std": std,
            "task_threshold": task_threshold,
            "threshold_name": args.threshold,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "best_val_auprc": float(best_score),
        },
        output,
    )
    with output.with_name("old_tiny_mlp_training_log.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    output.with_name("old_tiny_mlp_config.yaml").write_text(
        f"dataset_root: {args.dataset_root}\nthreshold: {args.threshold}\n"
        f"device: {device}\nseed: {args.seed}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best_val_auprc": float(best_score),
            }
        )
    )


if __name__ == "__main__":
    main()
