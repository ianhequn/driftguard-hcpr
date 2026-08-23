"""Reproducibility, device selection, and experiment bookkeeping helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - useful before dependency installation
    torch = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto"):
    if torch is None:
        raise RuntimeError("PyTorch is required for device selection.")
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def create_run_dir(base_dir: str | Path = "logs", name: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / f"{timestamp}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return (result.stdout + result.stderr).strip()
    except OSError as error:
        return f"unavailable: {error}"


def write_environment_snapshot(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Python executable: {sys.executable}",
        f"Python version: {sys.version.splitlines()[0]}",
        f"OS: {platform.platform()}",
        f"Machine: {platform.machine()}",
        "",
        "pip freeze:",
        _command_output([sys.executable, "-m", "pip", "freeze"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_logging(log_dir: str | Path, name: str = "driftguard") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_dir / "stdout.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def start_run(name: str, arguments: dict[str, Any], base_dir: str | Path = "logs"):
    run_dir = create_run_dir(base_dir=base_dir, name=name)
    save_yaml(arguments, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    write_environment_snapshot(run_dir / "environment.txt")
    logger = configure_logging(run_dir, name=f"driftguard.{name}")
    logger.info("Run directory: %s", run_dir.resolve())
    return run_dir, logger


def json_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
