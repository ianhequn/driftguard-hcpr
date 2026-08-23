# DriftGuard

DriftGuard is a reproducible research codebase for temporal feature reuse in
edge video semantic segmentation. Phase 0 and Phase 1 establish a reliable
SegFormer baseline before any cache reuse or drift estimator is introduced.

## Current scope

- Environment inspection and reproducibility metadata.
- SegFormer-B0 image inference using the Hugging Face checkpoint
  `nvidia/segformer-b0-finetuned-ade-512-512`.
- Frame-by-frame video inference with saved masks and overlays.
- Extraction of the four MiT encoder stage features.
- CUDA-aware baseline profiling with a CPU fallback for development machines.
- VSPW dataset loader skeleton and structure checker.
- Offline unit tests that do not require downloading a model.

Drift estimation, temporal caching, partial computation, and latency control are
intentionally out of scope for the current analysis phase.

## Environment status on the development machine

The current Windows machine has an AMD Radeon 880M iGPU and no NVIDIA CUDA
runtime. The code therefore runs on CPU here and keeps the CUDA path ready for
an RTX/WSL machine. A CUDA success report must only be produced on a machine
where `torch.cuda.is_available()` is true.

## Install

The repository uses Python 3.12. On Windows:

```powershell
cd D:\DriftGuard
& C:\Users\ianhe\Tools\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On an NVIDIA machine, select the official PyTorch wheel matching the installed
driver and CUDA support before installing the remaining requirements.

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pytest tests
```

The environment report is written to
`reports/environment_report.md`.

## Image inference

The first run downloads the model and the default documentation image:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_segformer.py
```

Use a local image when offline:

```powershell
.\.venv\Scripts\python.exe scripts\infer_image.py `
  --image "D:\canmv sd盘备份\bus.jpg" `
  --device cpu
```

Outputs are written under `outputs/segmentation/`.

## Video inference

```powershell
.\.venv\Scripts\python.exe scripts\infer_video.py `
  --video path\to\video.mp4 `
  --max-frames 100 `
  --device auto
```

Each video gets a directory containing original frames, color masks, overlays,
and `metrics.csv`.

## Feature extraction

Extract the first few frames only. The script never saves an entire video by
default:

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py `
  --image "D:\canmv sd盘备份\bus.jpg" `
  --device auto
```

The saved `.npz` files contain `stage1` through `stage4`. The CSV report
contains shape, dtype, summary statistics, and NaN/Inf checks.

## Baseline profiling

The defaults follow the research protocol: 100 warmup iterations and 500
measured iterations. On the current CPU-only machine use a short run for
development:

```powershell
.\.venv\Scripts\python.exe scripts\profile_baseline.py `
  --device cpu --warmup 2 --iterations 5 --input-size 128
```

On CUDA, the profiler uses CUDA Events and synchronizes before reading timing
values. Results are written to `results/baseline_profile.csv`.

## VSPW layout

Do not download the complete VSPW dataset automatically. The loader accepts the
following practical layout:

```text
data/raw/VSPW/
├── train/
│   └── video_0001/
│       ├── origin/
│       │   ├── 000000.jpg
│       │   └── ...
│       └── mask/
│           ├── 000000.png
│           └── ...
└── val/
    └── ...
```

## Phase 2: Temporal Redundancy Analysis

Phase 2 measures the four SegFormer MiT stage features on a continuous video
stream. It computes global cosine similarity, relative L2 distance, and
corresponding spatial-block metrics for offsets such as `1, 2, 4, 8, 16`.
The analysis uses a bounded rolling feature buffer and does not save complete
feature tensors for the video.

CPU smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_temporal_redundancy.py `
  --video data\samples\synthetic_video.mp4 `
  --model nvidia/segformer-b0-finetuned-ade-512-512 `
  --max-frames 20 `
  --offsets 1 2 4 8 16 `
  --block-grid 4 `
  --device cpu

.\.venv\Scripts\python.exe scripts\plot_temporal_redundancy.py
```

The CSV outputs are written to `results/E01_temporal_redundancy/` and the
figures to `figures/E01_temporal_redundancy/`.

## Cloud GPU Workflow

The local AMD machine is for development and CPU validation. On a CUDA-capable
cloud machine, use a clean Python 3.12 environment and install the PyTorch
wheel matching the NVIDIA driver. The repository does not bind to a specific
cloud provider.

```bash
git clone <your-repository-url>
cd DriftGuard
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/verify_cloud_gpu.py
python scripts/smoke_test_segformer.py --device cuda
python scripts/analyze_temporal_redundancy.py \
  --video data/samples/test.mp4 \
  --max-frames 32 \
  --offsets 1 2 4 8 16 \
  --block-grid 4 \
  --device cuda
python scripts/plot_temporal_redundancy.py
```

The CUDA wheel index is an example only. Select the official PyTorch build
compatible with the actual driver on the target machine. A successful GPU
environment check prints `CLOUD GPU READY`.

The loader also checks `data/<video>/origin` and common aliases such as
`images`, `img`, `labels`, and `masks`. Use:

```powershell
.\.venv\Scripts\python.exe scripts\check_vspw_structure.py `
  --root data\raw\VSPW --split train
```

## Reproducibility

Each experiment creates a timestamped directory under `logs/` containing the
parsed configuration, command, environment snapshot, and application log.
Default seed is 42. Final paper experiments should explicitly run seeds 0, 1,
and 2.

Suggested commit after this phase:

```text
feat: establish SegFormer baseline inference pipeline
```

## Next phase

Phase 2 is Temporal Redundancy Analysis: compare stage features from adjacent
frames at offsets `k = 1, 2, 4, 8, 16` using cosine similarity and relative L2
distance before implementing any cache policy.
