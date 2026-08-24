# DriftGuard Round 4 Pipeline Audit

## Frozen Source Semantics

The Round 3 source of truth is `scripts/run_round3_scheduler_study.py`.

The visual predictor input is:

```text
residual = abs(current_rgb_float / 255 - previous_rgb_float / 255)
logical grid = 4 x 4
candidate crop = exact integer grid boundary crop
resize = PyTorch bilinear, 16 x 16, align_corners=False
metadata = block age standardized by checkpoint mean/std
predictor = frozen TinyRiskCNN Patch+Age, FP32, eval/inference mode
```

`previous_rgb` is the immediately preceding decoded video frame. It is shared
by all 16 blocks and is not selected from the feature-cache age. The Stage-4
feature cache remains per policy and per budget, but the RGB risk reference is
global previous-frame RGB. This terminology differs from the Round 4 brief's
"cached reference frame" wording; source code, tests, and frozen outputs make
the executable semantics unambiguous.

Actual test video frames are RGB `uint8`, `480 x 640 x 3`. The optimized path
must support any height and width divisible by the 4x4 grid.

## Current Hot Path

| Operation | Device | Input -> output | Allocation / transfer | Main issue |
|---|---|---|---|---|
| AFVT age partition | CPU/NumPy | 16 ages -> older/tie/q | small arrays | acceptable |
| RGB conversion | CPU/NumPy | two HWC uint8 frames -> float32 | two full frames | repeated |
| Residual | CPU/NumPy | two float frames -> HWC residual | full-frame buffer | CPU bandwidth |
| Candidate crop | CPU/Python | residual -> list of CHW crops | Python loop/list | per-block overhead |
| Patch stack | CPU/PyTorch | list -> Nc x 3 x h x w | copies all candidates | allocation |
| Bilinear resize | CPU/PyTorch | crops -> Nc x 3 x 16 x 16 | CPU tensor | dominant preparation |
| Tensor transfer | CPU -> GPU | patches and metadata | two transfers | synchronization boundary |
| TinyRiskCNN | GPU | patches/metadata -> Nc scores | FP32 | about 0.45 ms at 37.5% |
| Score transfer | GPU -> CPU | Nc scores -> NumPy | small synchronized copy | required by old rank path |
| Deterministic rank | CPU/NumPy | scores -> selected tie blocks | lexsort | about 0.085 ms |

## Device-Boundary Findings

The Round 3 selector hot path contains:

```text
numpy.astype / numpy.abs
torch.from_numpy
tensor.to(device)
tensor.detach().cpu().numpy()
```

It does not use OpenCV resize or `cv2.absdiff`; the costly path is still
CPU-resident because residual construction, Python candidate crops, and
bilinear resize happen before the CUDA transfer.

The optimized selector will:

```text
keep raw RGB frames as BCHW CUDA tensors
determine the AFVT tie group before any image operation
skip image/CNN work when visual ranking is unnecessary
gather only candidate spatial regions
batch bilinear resize and TinyRiskCNN inference
retain deterministic lower-index tie breaking
```

The 16-element age state may remain on CPU. Large image tensors must not make a
GPU -> CPU -> GPU round trip.
