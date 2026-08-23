"""Inspect the local runtime and write reports/environment_report.md."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PACKAGE_NAMES = {
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "OpenCV": "opencv-python",
    "NumPy": "numpy",
    "SciPy": "scipy",
    "scikit-learn": "scikit-learn",
    "pandas": "pandas",
    "Pillow": "Pillow",
    "PyYAML": "PyYAML",
    "tqdm": "tqdm",
    "psutil": "psutil",
    "pytest": "pytest",
}


def command_output(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        return "unavailable"
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout + result.stderr).strip()
    return output or f"exit_code={result.returncode}"


def detect_gpu() -> tuple[str, str, str]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            return (
                torch.cuda.get_device_name(0),
                str(torch.version.cuda),
                "true",
            )
    except Exception:
        pass
    if platform.system() == "Windows":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell:
            output = command_output(
                [
                    powershell,
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -ExpandProperty Name",
                ]
            )
            return output.replace("\r", "").replace("\n", ", "), "not available", "false"
    return "not detected", "not available", "false"


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_report() -> str:
    gpu, cuda, cuda_available = detect_gpu()
    try:
        import torch

        torch_details = (
            f"{torch.__version__} (torch.version.cuda={torch.version.cuda}, "
            f"cuda_available={torch.cuda.is_available()})"
        )
    except Exception as error:
        torch_details = f"unavailable: {error}"

    package_rows = []
    for label, distribution in PACKAGE_NAMES.items():
        version = package_version(distribution)
        if label == "PyTorch" and version != "not installed":
            version = torch_details
        package_rows.append(f"| {label} | {version} |")

    nvidia = command_output(["nvidia-smi"])
    nvcc = command_output(["nvcc", "--version"])
    lines = [
        "# DriftGuard Environment Report",
        "",
        f"- Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        f"- OS: {platform.platform()}",
        f"- Python executable: `{sys.executable}`",
        f"- Python: {platform.python_version()}",
        f"- GPU detected: {gpu}",
        f"- NVIDIA Driver / nvidia-smi: {nvidia}",
        f"- CUDA Toolkit / nvcc: {nvcc}",
        f"- CUDA driver compatibility: "
        f"{'available through PyTorch' if cuda_available == 'true' else 'not applicable; no NVIDIA CUDA runtime detected'}",
        f"- CUDA available to PyTorch: `{cuda_available}`",
        f"- PyTorch CUDA runtime: `{cuda}`",
        "",
        "## Installed packages",
        "",
        "| Package | Version |",
        "|---|---|",
        *package_rows,
        "",
        "## Interpretation",
        "",
        "This report records the runtime that actually executed the scripts.",
        "On a CUDA-capable machine, rerun this script inside the target environment",
        "and require `cuda_available=true` before collecting paper-grade GPU latency.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    report_path = PROJECT_ROOT / "reports" / "environment_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
