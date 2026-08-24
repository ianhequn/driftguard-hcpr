"""Train and evaluate Round 2 scalar baselines and TinyRiskCNN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from driftguard.models.risk_predictor import TinyRiskCNN, parameter_count  # noqa: E402
from driftguard.utils.common import choose_device, set_seed  # noqa: E402


METADATA_COLUMNS = ["cache_age", "previous_entropy", "global_residual", "local_residual_mean", "local_residual_variance"]


class HistoricalTinyMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
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


def _read_shards(root: Path, inventory: Path) -> dict[str, dict[str, np.ndarray]]:
    with inventory.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    output = {}
    for record in records:
        path = root / "shards" / f"{record['video_id']}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        values = dict(np.load(path, allow_pickle=False))
        values["split"] = np.asarray([record["split"]] * len(values["task_damage"]))
        output[record["video_id"]] = values
    return output


def _concat(shards: dict[str, dict[str, np.ndarray]], split: str) -> dict[str, np.ndarray]:
    values = [shard for shard in shards.values() if str(shard["split"][0]) == split]
    if not values:
        raise RuntimeError(f"No shards for split={split}")
    keys = ["patch", "metadata", "task_damage", "video_id", "frame_id", "block_index"]
    return {key: np.concatenate([item[key] for item in values], axis=0) for key in keys}


def _safe_auc(fn, y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(fn(y_true, score))
    except ValueError:
        return float("nan")


def _precision_at(score: np.ndarray, y_true: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(score) * fraction)))
    indices = np.argsort(score)[-count:]
    return float(np.mean(y_true[indices]))


def _metrics(y: np.ndarray, score: np.ndarray, damage: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = score >= 0.5
    return {
        "auroc": _safe_auc(roc_auc_score, y, score),
        "auprc": _safe_auc(average_precision_score, y, score),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "spearman_task_damage": float(spearmanr(score, damage).statistic),
        "precision_at_10": _precision_at(score, y, 0.10),
        "precision_at_25": _precision_at(score, y, 0.25),
        "recall_at_25": float(y[np.argsort(score)[-max(1, int(np.ceil(len(score) * 0.25))):]].sum() / max(1, y.sum())),
        "threshold": threshold,
        "positive_ratio": float(y.mean()),
    }


def _per_video(y, score, damage, video_ids, threshold) -> list[dict]:
    rows = []
    for video_id in sorted(set(video_ids.tolist())):
        mask = video_ids == video_id
        if mask.sum() < 2:
            continue
        row = {"video_id": str(video_id), "samples": int(mask.sum())}
        row.update(_metrics(y[mask], score[mask], damage[mask], threshold))
        rows.append(row)
    return rows


def _fit_standardizer(metadata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = metadata.mean(axis=0).astype(np.float32)
    std = metadata.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _train_cnn(
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[TinyRiskCNN, list[dict]]:
    set_seed(seed)
    model = TinyRiskCNN(metadata_dim=train["metadata"].shape[1]).to(device)
    if parameter_count(model) >= 50000:
        raise RuntimeError(f"TinyRiskCNN exceeds hard parameter limit: {parameter_count(model)}")
    positive = max(1.0, float((len(y_train) - y_train.sum()) / max(1.0, y_train.sum())))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_ds = TensorDataset(
        torch.from_numpy(train["patch"]),
        torch.from_numpy((train["metadata"] - mean) / std),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val["patch"]),
        torch.from_numpy((val["metadata"] - mean) / std),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    best_score = -np.inf
    best_state = None
    stale_epochs = 0
    log = []
    for epoch in range(1, 101):
        model.train()
        train_losses = []
        for patch, metadata, target in train_loader:
            patch, metadata, target = patch.to(device), metadata.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(patch, metadata).unsqueeze(1), target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_scores, val_targets = [], []
        with torch.inference_mode():
            for patch, metadata, target in val_loader:
                val_scores.append(torch.sigmoid(model(patch.to(device), metadata.to(device))).cpu().numpy())
                val_targets.append(target.numpy())
        scores = np.concatenate(val_scores)
        targets = np.concatenate(val_targets).astype(np.int32)
        score = _safe_auc(average_precision_score, targets, scores)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_auprc": score}
        log.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, log


def _predict(model, values, mean, std, device, batch_size=512) -> np.ndarray:
    ds = TensorDataset(
        torch.from_numpy(values["patch"]),
        torch.from_numpy((values["metadata"] - mean) / std),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    output = []
    model.eval()
    with torch.inference_mode():
        for patch, metadata in loader:
            output.append(torch.sigmoid(model(patch.to(device), metadata.to(device))).cpu().numpy())
    return np.concatenate(output)


def _benchmark(model, patch, metadata, mean, std, device) -> list[dict]:
    model.eval()
    rows = []
    for batch_size, count in ((1, 1), (16, 16), (16, 16)):
        batch_patch = torch.from_numpy(np.tile(patch[:1], (count, 1, 1, 1))).to(device)
        batch_metadata = torch.from_numpy(np.tile(((metadata[:1] - mean) / std), (count, 1))).to(device)
        for _ in range(10):
            with torch.inference_mode():
                model(batch_patch, batch_metadata)
        timings = []
        for _ in range(100):
            if device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.inference_mode():
                    model(batch_patch, batch_metadata)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))
            else:
                start = time.perf_counter()
                with torch.inference_mode():
                    model(batch_patch, batch_metadata)
                timings.append((time.perf_counter() - start) * 1000.0)
        rows.append(
            {
                "batch_size": batch_size if count == 1 else 16,
                "count": count,
                "mean_ms": float(np.mean(timings)),
                "median_ms": float(np.median(timings)),
                "p95_ms": float(np.quantile(timings, 0.95)),
                "p99_ms": float(np.quantile(timings, 0.99)),
                "risk_per_frame_mean_ms": float(np.mean(timings) if count == 16 else np.mean(timings) * 16),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="results/R2_E01_task_damage_dataset")
    parser.add_argument("--inventory", default="results/R2_E00_video_dataset/video_inventory.csv")
    parser.add_argument("--output-root", default="results/R2_E02_predictor")
    parser.add_argument("--threshold", choices=["p70", "p80", "p90"], default="p80")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    dataset_root = PROJECT_ROOT / args.dataset_root
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    shards = _read_shards(dataset_root, PROJECT_ROOT / args.inventory)
    train, val, test = (_concat(shards, split) for split in ("train", "val", "test"))
    thresholds = json.loads((dataset_root / "thresholds_train_only.json").read_text(encoding="utf-8"))
    threshold = float(thresholds[args.threshold])
    y_train = (train["task_damage"] > threshold).astype(np.int32)
    y_val = (val["task_damage"] > threshold).astype(np.int32)
    y_test = (test["task_damage"] > threshold).astype(np.int32)
    mean, std = _fit_standardizer(train["metadata"])

    rows = []
    models = {}
    baseline_scores = {
        "Residual Mean": test["metadata"][:, 3],
        "Residual P90": np.quantile(test["patch"].reshape(len(test["patch"]), -1), 0.90, axis=1),
        "Residual Max": test["patch"].reshape(len(test["patch"]), -1).max(axis=1),
    }
    train_scores = {
        "Residual Mean": train["metadata"][:, 3],
        "Residual P90": np.quantile(train["patch"].reshape(len(train["patch"]), -1), 0.90, axis=1),
        "Residual Max": train["patch"].reshape(len(train["patch"]), -1).max(axis=1),
    }
    for name, score in baseline_scores.items():
        normalized = (score - score.min()) / max(1e-8, score.max() - score.min())
        metric = _metrics(y_test, normalized, test["task_damage"], threshold)
        rows.append({"model": name, "parameters": 0, **metric})

    for name, estimator in (
        ("Logistic Regression", LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=args.seed, n_jobs=-1)),
    ):
        estimator.fit((train["metadata"] - mean) / std, y_train)
        score = estimator.predict_proba((test["metadata"] - mean) / std)[:, 1]
        rows.append({"model": name, "parameters": 0, **_metrics(y_test, score, test["task_damage"], threshold)})

    model, training_log = _train_cnn(train, val, y_train, y_val, mean, std, device, args.seed)
    score = _predict(model, test, mean, std, device)
    cnn_metrics = _metrics(y_test, score, test["task_damage"], threshold)
    rows.append({"model": "TinyRiskCNN", "parameters": parameter_count(model), **cnn_metrics})
    models["TinyRiskCNN"] = score

    old_model_path = PROJECT_ROOT / "results/E06_estimator/best_model.pt"
    if old_model_path.exists():
        checkpoint = torch.load(old_model_path, map_location="cpu")
        old = HistoricalTinyMLP(len(checkpoint["feature_columns"]))
        old.load_state_dict(checkpoint["state_dict"])
        old.eval()
        old_mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        old_std = np.asarray(checkpoint["std"], dtype=np.float32)
        old_std[old_std < 1e-8] = 1.0
        old_features = (test["metadata"] - old_mean) / old_std
        with torch.inference_mode():
            old_score = old(torch.from_numpy(old_features)).numpy()
        old_score = (old_score - old_score.min()) / max(1e-8, old_score.max() - old_score.min())
        rows.append({"model": "Old Tiny MLP (historical)", "parameters": sum(p.numel() for p in old.parameters()), **_metrics(y_test, old_score, test["task_damage"], threshold)})

    with (output_root / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_video = _per_video(y_test, score, test["task_damage"], test["video_id"], threshold)
    with (output_root / "tiny_risk_per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_video[0]))
        writer.writeheader()
        writer.writerows(per_video)
    with (output_root / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_log[0]))
        writer.writeheader()
        writer.writerows(training_log)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata_mean": mean,
            "metadata_std": std,
            "metadata_columns": METADATA_COLUMNS,
            "threshold": threshold,
            "threshold_name": args.threshold,
            "parameter_count": parameter_count(model),
        },
        output_root / "best_model.pt",
    )
    latency = _benchmark(model, test["patch"], test["metadata"], mean, std, device)
    with (output_root / "latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency[0]))
        writer.writeheader()
        writer.writerows(latency)
    (output_root / "config.yaml").write_text(
        f"dataset_root: {args.dataset_root}\nthreshold: {args.threshold}\ndevice: {device}\nseed: {args.seed}\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps({"threshold": args.threshold, "device": str(device), "tiny_risk_cnn": cnn_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"tiny_risk_cnn": cnn_metrics, "parameters": parameter_count(model)}))


if __name__ == "__main__":
    main()
