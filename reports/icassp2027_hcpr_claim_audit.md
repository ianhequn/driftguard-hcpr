# ICASSP 2027 HCPR Claim Audit

## Supported

- HCPR is a progressive whole-stage controller over the frozen SegFormer
  pipeline.
- Stage 1 and the decoder remain active on every frame; stopped suffix stages
  are absent from the forward path.
- The evaluated RTX 3080 Ti transfer uses target-device stage-cost profiling
  with frozen Round 11A temporal scales and lambda thresholds.
- Measured CUDA-resident end-to-end speedup is supported for the evaluated
  VSPW, SegFormer-B0/B1, RTX 4090 D, and RTX 3080 Ti protocols.
- At B0/q65 versus OldCost and B1/q95 versus ProgressiveDriftOnly, NewCost
  shows strict joint improvements in the tested anchor comparisons.

## Deliberately bounded

- "Cross-device" means the two evaluated GPUs under matched CUDA 11.8,
  PyTorch 2.1.2+cu118, and Transformers 4.44.2; driver versions differ.
- Target-device stage profiling is required for transfer. Lambda recalibration
  was not required for the evaluated RTX 3080 Ti transfer.
- The controller is not claimed to be theoretically optimal, universally
  portable, or universally hardware-causal.
- Results are on the VSPW validation split, not an independent public test
  set, and do not establish generalization to arbitrary GPUs, model scales,
  architectures, or software stacks.

## Excluded claims

- No claim of universal GPU portability or a universal hardware law.
- No claim that target-device normalization strictly improves every model,
  quantile, or device.
- No claim of custom-kernel, TensorRT, token-sparse, or multi-stage reuse.
- No new experiments, new datasets, or modified frozen results were used in
  the manuscript rewrite.

## Citation audit

All 11 bibliography entries are inherited from the verified ICIP source,
remain cited, and have no undefined citation keys in the final build.

