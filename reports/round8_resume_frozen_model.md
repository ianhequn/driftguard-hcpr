# DriftGuard Round 8 Resume: Frozen B1 Model

The selected checkpoint is step 40,000 and is frozen at:

```text
/root/autodl-tmp/models/segformer-b1-vspw124
```

The artifact contains the VSPW-124 config, class names, label contract, preprocessor config, model safetensors, source provenance, and checksum. Selected validation mIoU is `0.3715982824`; checkpoint SHA256 is recorded in `freeze_summary.json`; frozen model SHA256 is `892a6589874d762c6b9c911e559e062fccbb226b33309aa6c8317dbec091fcfb`.

Fresh-process reload passed with:

```text
mIoU = 0.3715979313
pixel accuracy = 0.7490037485
class alignment = 124 / 124
absolute mIoU difference from training validation = 3.51e-7
```

`encoder_frozen=false` in the legacy freeze JSON refers only to the PyTorch `requires_grad` flag after loading. The deployed frozen artifact contains weights without optimizer state and is loaded in evaluation mode; no training mutation is performed in subsequent benchmarks.
