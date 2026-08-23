"""Train and evaluate classical baselines plus the tiny DriftGuard MLP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_FEATURE_COLUMNS = [
    "local_residual",
    "cache_age",
    "previous_entropy",
    "global_residual",
    "local_variance",
]


def _augment_features(values: np.ndarray, revision: str) -> tuple[np.ndarray, list[str]]:
    if revision == "baseline":
        return values, list(BASE_FEATURE_COLUMNS)
    if revision == "interactions":
        local = values[:, 0]
        age = values[:, 1]
        global_residual = values[:, 3]
        augmented = np.column_stack(
            [
                values,
                local * age,
                global_residual * age,
                np.log1p(age),
            ]
        ).astype(np.float32)
        return augmented, list(BASE_FEATURE_COLUMNS) + [
            "local_residual_x_age",
            "global_residual_x_age",
            "log1p_age",
        ]
    raise ValueError(f"Unknown feature revision: {revision}")


class TinyDriftEstimator(torch.nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="results/E05_drift_dataset")
    parser.add_argument("--output-root", default="results/E06_estimator")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--target-transform",
        choices=["identity", "log1p"],
        default="identity",
    )
    parser.add_argument("--target-clip-percentile", type=float, default=100.0)
    parser.add_argument(
        "--feature-revision",
        choices=["baseline", "interactions"],
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_split(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction, squared=False)),
        "pearson": float(pearsonr(y_true, prediction).statistic)
        if np.std(prediction) > 1e-12
        else 0.0,
        "spearman": float(spearmanr(y_true, prediction).statistic)
        if np.std(prediction) > 1e-12
        else 0.0,
    }


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> tuple[TinyDriftEstimator, list[dict]]:
    torch.manual_seed(seed)
    model = TinyDriftEstimator(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.HuberLoss()
    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    best_state = None
    best_val = math.inf
    bad_epochs = 0
    log = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        train_losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[indices])
            loss = loss_fn(prediction, train_y[indices])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_mae = float(torch.abs(val_prediction - val_y).mean().item())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_mae": val_mae,
        }
        log.append(row)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


def _predict_mlp(
    model: TinyDriftEstimator,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
    return values.astype(np.float32)


def _unsafe_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    train_y: np.ndarray,
    model_name: str,
) -> list[dict]:
    rows = []
    for percentile in (70, 80, 90):
        threshold = float(np.percentile(train_y, percentile))
        true_unsafe = (y_true > threshold).astype(np.int32)
        score = prediction
        predicted_unsafe = (score > threshold).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_unsafe,
            predicted_unsafe,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "model": model_name,
                "threshold_percentile": percentile,
                "threshold": threshold,
                "auroc": float(roc_auc_score(true_unsafe, score))
                if len(np.unique(true_unsafe)) > 1
                else 0.0,
                "auprc": float(average_precision_score(true_unsafe, score)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _benchmark_overhead(
    model: TinyDriftEstimator,
    x: np.ndarray,
    device: torch.device,
) -> dict[str, float | str]:
    model = model.to(device).eval()
    batch = torch.from_numpy(x[: min(16, len(x))]).to(device)
    for _ in range(100):
        with torch.no_grad():
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(1000):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                model(batch)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            with torch.no_grad():
                model(batch)
            values.append((time.perf_counter() - start_time) * 1000.0)
    array = np.asarray(values)
    return {
        "device": str(device),
        "batch_size": len(batch),
        "mean_latency_ms": float(array.mean()),
        "p95_latency_ms": float(np.quantile(array, 0.95)),
        "p99_latency_ms": float(np.quantile(array, 0.99)),
        "per_sample_mean_ms": float(array.mean() / len(batch)),
        "parameters": _parameter_count(model),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    dataset_root = PROJECT_ROOT / args.dataset_root
    output_root = PROJECT_ROOT / args.output_root
    train = _load_split(dataset_root, "train")
    val = _load_split(dataset_root, "val")
    test = _load_split(dataset_root, "test")
    x_train_base = train["X"].astype(np.float32)
    x_val_base = val["X"].astype(np.float32)
    x_test_base = test["X"].astype(np.float32)
    x_train, feature_columns = _augment_features(x_train_base, args.feature_revision)
    x_val, _ = _augment_features(x_val_base, args.feature_revision)
    x_test, _ = _augment_features(x_test_base, args.feature_revision)
    y_train, y_val = train["y"].astype(np.float32), val["y"].astype(np.float32)
    y_test = test["y"].astype(np.float32)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std
    if args.target_transform == "log1p":
        y_train_model = np.log1p(np.maximum(y_train, 0.0))
        y_val_model = np.log1p(np.maximum(y_val, 0.0))
    else:
        y_train_model = y_train
        y_val_model = y_val
    if not 0.0 < args.target_clip_percentile <= 100.0:
        raise ValueError("--target-clip-percentile must be in (0, 100]")
    if args.target_clip_percentile < 100.0:
        clip_value = float(np.percentile(y_train_model, args.target_clip_percentile))
        y_train_model = np.minimum(y_train_model, clip_value)
        y_val_model = np.minimum(y_val_model, clip_value)
    else:
        clip_value = None

    predictions: dict[str, np.ndarray] = {}
    models: dict[str, object] = {}

    age_model = LinearRegression().fit(x_train[:, [1]], y_train)
    predictions["age_only"] = age_model.predict(x_test[:, [1]])
    models["age_only"] = age_model
    residual_model = LinearRegression().fit(x_train[:, [0]], y_train)
    predictions["residual_only"] = residual_model.predict(x_test[:, [0]])
    models["residual_only"] = residual_model
    linear_model = LinearRegression().fit(x_train_scaled, y_train)
    predictions["linear_regression"] = linear_model.predict(x_test_scaled)
    models["linear_regression"] = linear_model
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=args.seed,
        n_jobs=2,
    ).fit(x_train, y_train)
    predictions["random_forest"] = forest.predict(x_test)
    models["random_forest"] = forest
    mlp, training_log = _train_mlp(
        x_train_scaled,
        y_train_model,
        x_val_scaled,
        y_val_model,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    predictions["tiny_mlp"] = _predict_mlp(mlp, x_test_scaled, device)
    if args.target_transform == "log1p":
        predictions["tiny_mlp"] = np.expm1(predictions["tiny_mlp"])
    models["tiny_mlp"] = mlp

    metric_rows = []
    unsafe_rows = []
    for name, prediction in predictions.items():
        metric_rows.append(
            {
                "model": name,
                "parameters": _parameter_count(models[name])
                if isinstance(models[name], torch.nn.Module)
                else getattr(models[name], "n_features_in_", 0),
                **_metrics(y_test, prediction),
            }
        )
        unsafe_rows.extend(_unsafe_metrics(y_test, prediction, y_train, name))

    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = PROJECT_ROOT / "figures" / "E06_estimator"
    figure_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "metrics.csv", metric_rows)
    _write_csv(output_root / "unsafe_metrics.csv", unsafe_rows)
    _write_csv(output_root / "training_log.csv", training_log)
    prediction_rows = [
        {
            "true_drift": float(y_test[index]),
            **{name: float(values[index]) for name, values in predictions.items()},
        }
        for index in range(len(y_test))
    ]
    _write_csv(output_root / "test_predictions.csv", prediction_rows)

    checkpoint = {
        "state_dict": mlp.state_dict(),
        "feature_columns": feature_columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_transform": args.target_transform,
        "target_clip_percentile": args.target_clip_percentile,
        "target_clip_value": clip_value,
        "feature_revision": args.feature_revision,
    }
    torch.save(checkpoint, output_root / "best_model.pt")
    (output_root / "scaler.json").write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "target_transform": args.target_transform,
                "target_clip_percentile": args.target_clip_percentile,
                "target_clip_value": clip_value,
                "feature_revision": args.feature_revision,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    overhead = _benchmark_overhead(mlp, x_test_scaled, device)
    _write_csv(output_root / "estimator_overhead.csv", [overhead])

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions["tiny_mlp"], s=8, alpha=0.35)
    limits = [float(min(y_test.min(), predictions["tiny_mlp"].min())), float(max(y_test.max(), predictions["tiny_mlp"].max()))]
    plt.plot(limits, limits, "k--")
    plt.xlabel("True feature drift")
    plt.ylabel("Predicted feature drift")
    plt.tight_layout()
    plt.savefig(figure_root / "predicted_vs_true_drift.png", dpi=160)
    plt.close()

    residuals = predictions["tiny_mlp"] - y_test
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40)
    plt.xlabel("Prediction residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figure_root / "drift_residual_histogram.png", dpi=160)
    plt.close()

    threshold = float(np.percentile(y_train, 80))
    unsafe = (y_test > threshold).astype(np.int32)
    if len(np.unique(unsafe)) > 1:
        fpr, tpr, _ = roc_curve(unsafe, predictions["tiny_mlp"])
        precision, recall, _ = precision_recall_curve(unsafe, predictions["tiny_mlp"])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.tight_layout()
        plt.savefig(figure_root / "estimator_roc.png", dpi=160)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(figure_root / "estimator_pr.png", dpi=160)
        plt.close()

    (output_root / "config.yaml").write_text(
        "\n".join(
            [
                f"dataset_root: {args.dataset_root}",
                f"device: {device}",
                f"epochs: {args.epochs}",
                f"patience: {args.patience}",
                f"batch_size: {args.batch_size}",
                f"target_transform: {args.target_transform}",
                f"target_clip_percentile: {args.target_clip_percentile}",
                f"feature_revision: {args.feature_revision}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    best = next(row for row in metric_rows if row["model"] == "tiny_mlp")
    (output_root / "best_estimator.json").write_text(
        json.dumps(best, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"device={device}")
    print(f"best_model_by_test_mae={best['model']}")
    print(f"tiny_mlp_parameters={_parameter_count(mlp)}")
    print(f"output_root={output_root.resolve()}")


if __name__ == "__main__":
    main()
