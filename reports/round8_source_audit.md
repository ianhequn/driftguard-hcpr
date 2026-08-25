# Round 8 Source Checkpoint Audit

## Required source

```text
identifier = nvidia/segformer-b1-finetuned-ade-512-512
architecture = SegFormer-B1
source task = ADE20K semantic segmentation
source output classes = 150
```

This identifier was the only source candidate authorized by the serial experiment. It was not replaced with B2/B3 or an unverified local model.

## Acquisition result

Cloud acquisition using `huggingface_hub.snapshot_download` failed with a connection timeout to `huggingface.co:443`. A second local connectivity check using `curl` also timed out after 15 seconds. The cloud model cache did not contain a B1 snapshot, and no checkpoint files were written to the source directory.

Artifact: `results/R8_E00_source_audit/source_audit.json`

## Gate

```text
ROUND 8 = SOURCE CHECKPOINT BLOCKED
```

Because the source could not be obtained and verified, architecture, initialization, 124-class construction, smoke training, calibration and 40k training are invalid to start. This is an acquisition block, not a model-quality or temporal-reuse conclusion.
