# DriftGuard Round 9 Dataset Recovery

## Current Gate

```text
STATIC CITYSCAPES: BLOCKED
TEMPORAL DATASET: BLOCKED
ROUND 9: STILL BLOCKED
```

The official Cityscapes download page is accessible in the logged-in browser;
the page exposed `Edit account` and `Logout`. However, the static packages
have not been downloaded because the browser's file-handling endpoint was
blocked by the client, and the command-line request without browser session
authentication redirected to the official login page.

The missing official components are:

```text
gtFine_trainvaltest.zip
leftImg8bit_trainvaltest.zip
leftImg8bit_sequence_trainvaltest.zip
```

The first two are sufficient for static preparation but are not currently
present. The sequence package is required for the frozen temporal protocol;
its listed size is 324 GB and cannot fit either available storage location.

No browser cookies, credentials, or authentication values were read, saved, or
written to the repository.

## Resume Condition

Resume only after a storage location with enough capacity for the official
archives plus extraction workspace is provided. Restart at the dataset audit
gate. Do not start B0/B1 training or temporal benchmark before the static and
temporal data contracts pass.
