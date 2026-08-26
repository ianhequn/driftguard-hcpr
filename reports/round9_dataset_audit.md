# DriftGuard Round 9 Dataset Audit

## Gate Result

```text
ROUND 9 = DATASET BLOCKED
BLOCK REASON = STORAGE CAPACITY
```

The official download page is now accessible in the user's logged-in browser,
but the required packages are not present in the local or cloud environment.
No training, label scan, sequence manifest, or benchmark was started.

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

The browser page showed `Edit account` and `Logout`, confirming that the
Cityscapes account session is active.

## Missing Official Components

The following official data components are required before the experiment can
continue:

```text
leftImg8bit train/val
gtFine train/val
leftImg8bit_sequence train/val
```

The logged-in official page exposes these relevant packages:

```text
gtFine_trainvaltest.zip               241MB  packageID=1
leftImg8bit_trainvaltest.zip           11GB  packageID=3
leftImg8bit_sequence_trainvaltest.zip 324GB packageID=14
```

The compressed total is already about `335GB`, before extraction. Current
capacity is approximately `245GB` free on local `D:` and `95GB` free on the
cloud `/root/autodl-tmp` volume. Neither location can safely hold the required
sequence package and extraction workspace.

The intended contract is the official `labelTrainIds` representation with
semantic IDs `0..18` and ignore value `255`. The required 2:1 image processing
contract is `512 x 1024`; it has not been run because the source data is absent.

The official Cityscapes site requires an account login before data download;
the login requirement is now satisfied in the browser:

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
