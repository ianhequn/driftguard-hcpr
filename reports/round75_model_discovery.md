# DriftGuard Round 7.5 Model Discovery

## Discovery Decision

```text
valid VSPW-trained SegFormer checkpoint: NOT FOUND
verified public compatible checkpoint: NOT FOUND
model status: NOT FOUND
training status: required only as a future fallback
```

No checkpoint was downloaded, loaded, or used for inference in Round 7.5.

## Candidates

| Candidate | Source | Architecture | VSPW training | Checkpoint | Validity |
|---|---|---|---|---|---|
| Official VSPW baseline | `VSPW-dataset/VSPW_baseline` | PSPNet/OCRNet and CNN baselines | training code/reference | no VSPW checkpoint in repository; ImageNet weights referenced | invalid |
| Official VSPW code | `VSPW-dataset/VSPW_code` | evaluation/code repository | unspecified | README asks users to contact author for training/testing code | invalid |
| Public reimplementation | `sssdddwww2/CVPR2021_VSPW_Implement` | PSP/OCR/temporal CNN variants | VSPW code | no VSPW segmentation checkpoint in repository tree | invalid |
| MPVSS official author repo | `ziplab/MPVSS` | Mask2Former/MPVSS | ResNet/Swin variants | no VSPW checkpoint artifact in repository tree; VSPW config only | config-compatible, model unavailable |
| Hugging Face search | public model hub | candidate discovery only | unverified | network/API unavailable in current environment; no verified candidate | blocked |

The official baseline README explicitly points to ImageNet-pretrained models,
not a frozen VSPW-trained SegFormer checkpoint. The official code README does
not publish a usable checkpoint. Repository tree audits found source/config
files but no VSPW segmentation weight artifact.

The official-author `ziplab/MPVSS` repository is the strongest
configuration-level candidate found. Its VSPW configuration explicitly
specifies `NUM_CLASSES: 124`, `IGNORE_VALUE: 255`, and VSPW train/validation
datasets. The repository tree contains no VSPW-trained segmentation
checkpoint, and its Mask2Former/MPVSS graph is not a direct Round 6
SegFormer Stage-4 reuse target.

## SegFormer Search

```text
VSPW-trained SegFormer-B0: not found
VSPW-trained SegFormer-B1: not found
VSPW-trained SegFormer-B2: not found
public checkpoint with verified VSPW class order: not found
```

No candidate met all required gates:

```text
checkpoint exists
config readable
124 output classes
explicit VSPW training provenance
official class order verified
loadable architecture
forward smoke pass
```

## Candidate Selection

```text
selected candidate: NONE
selection reason: no candidate passed provenance, checkpoint, and
                  VSPW taxonomy gates
```

The current ADE20K SegFormer-B0 is retained as the historical Round 6 model
only. It is not a Round 7.5 candidate.

## Future Fallback

If the official VSPW package becomes available and no public compatible
checkpoint is found, the project will require a separately trained and frozen
VSPW-compatible segmentation model. Training is not performed in this round.
