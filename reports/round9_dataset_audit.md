# DriftGuard Round 9 Dataset Audit

## Gate Result

```text
ROUND 9 = DATASET BLOCKED
```

The required official Cityscapes package is not present in the local or cloud
environment. No training, label scan, sequence manifest, or benchmark was
started.

## Search Scope

Local roots searched:

```text
D:/datasets
D:/DriftGuard
C:/Users/ianhe/Downloads
C:/Users/ianhe/Desktop
```

Cloud roots searched:

```text
/root/autodl-tmp
/root/autodl-tmp/DriftGuard
```

The local dataset directories found were KITTI and VSPW-related directories.
The only local filename containing `cityscapes` was the installed
`torchvision` dataset loader, which is code rather than dataset content. No
Cityscapes-named files, directories, or archives were found on the cloud GPU.

## Missing Official Components

The following official data components are required before the experiment can
continue:

```text
leftImg8bit train/val
gtFine train/val
leftImg8bit_sequence train/val
```

The intended contract is the official `labelTrainIds` representation with
semantic IDs `0..18` and ignore value `255`. The required 2:1 image processing
contract is `512 x 1024`; it has not been run because the source data is absent.

The official Cityscapes site requires an account login before data download:

```text
https://www.cityscapes-dataset.com/login/
https://www.cityscapes-dataset.com/cameradownload/
```

## Stop Boundary

The following Round 9 stages remain intentionally not run:

```text
dataset integrity audit
label contract scan
sequence-target manifest
B0/B1 Cityscapes training
fresh reload validation
temporal GT benchmark
paired latency
cross-dataset analysis
```

No Kaggle repack, community mirror, CamVid, BDD100K, KITTI, or other
replacement dataset was substituted.
