# DriftGuard Round 7.65 Final Report

## Final decision

```text
ROUND 7.65 = LABEL CONTRACT RESOLVED
DATA STATUS = READY WITH CANONICAL EVALUATION CONTRACT
ROUND 7.6 RESTART = AUTHORIZED, NOT EXECUTED
```

This round resolved label semantics only. It did not train a model, resume Round 7.6, run temporal reuse, run latency measurements, or claim a formal mIoU result.

## Scope and source of truth

The audit used the frozen VSPW package and the current project code. The primary sources were:

- VSPW dataset README: `https://github.com/VSPW-dataset/VSPW-dataset-download`
- Official loader: `https://raw.githubusercontent.com/VSPW-dataset/VSPW_baseline/master/dataset.py`
- Official evaluator: `https://raw.githubusercontent.com/VSPW-dataset/VSPW_baseline/master/utils.py`

## Questions resolved

### Q1. What does the README claim?

The README describes raw values `0..124` and `255`, with `0` as other and `255` as void.

### Q2. What does the official loader actually do?

It applies, in order:

```python
label[label == 0] = 255
label = label - 1
label[label == 254] = 255
```

This is the official normalization order used for this project.

### Q3. What does the official evaluator count?

For `num_class=124`, the evaluator includes only normalized ground-truth IDs satisfying:

```python
gt >= 0 and gt < 124
```

All other values are excluded from the confusion matrix.

### Q4. What is raw label 253?

Raw `253` becomes normalized `252` under the official loader. It is outside the evaluator domain and is therefore void/excluded. It is classified as an official released annotation artifact, not assigned to a semantic class and not treated as local or cloud transfer corruption.

### Q5. Is the anomaly reproducible across machines?

Yes. All 17 affected validation masks have matching SHA256 values between local and cloud copies. The first anomaly SHA256 is:

```text
8d0d7eb3d334f08d6fce9676199bb1d41ecf1b0b7cfbe740a49d635c6c08acac
```

### Q6. How large is the anomaly?

The full validation scan found:

```text
raw253 masks: 17 / 24392
raw253 pixels: 68314
validation pixel fraction: 0.00068691543%
affected validation videos: 1 / 343
```

The training split contains no raw-253 pixels.

### Q7. Does the official evaluator treat normalized 252 as a class?

No. Normalized `252` fails the official `[0, 124)` validity check and contributes neither to the official confusion matrix nor to mIoU.

### Q8. What canonical project representation is selected?

The project preserves official normalization first, then canonicalizes every normalized ID outside `[0, 123]` to ignore index `255`:

```text
raw mask
  -> official normalization
  -> IDs 0..123 remain semantic
  -> every other ID becomes 255
```

This includes raw `253 -> 252 -> 255`. The rule is generalized and does not special-case raw 253.

### Q9. Is this compatible with training?

Yes. Canonical masks contain only semantic IDs `0..123` and ignore index `255`. Cross-entropy uses `ignore_index=255`, and metrics exclude canonical `255` pixels.

### Q10. Does canonical evaluation reproduce the official evaluator?

Yes. The equivalence script passed on both synthetic boundary values and all 17 real anomaly masks:

```text
valid masks equal: true
confusion matrices equal: true
equivalence_pass: true
```

### Q11. Does the full-domain canonical scan pass?

Yes. Across train and validation, remaining invalid labels after canonicalization are `0`.

### Q12. Were dataset files modified?

No. The PNG files and archive were not rewritten. Canonicalization is applied at the project data-loader/metric boundary, with warnings and diagnostics for out-of-range normalized IDs.

### Q13. Can Round 7.6 restart?

Yes, the restart is authorized because the label contract is now explicit, reproducible, and evaluator-equivalent. The restart was not executed in this round. Historical `Round 7.6 = MODEL TRAINING NO-GO` is preserved as the historical pre-contract decision.

## Contract record

```text
num_classes       = 124
valid IDs         = 0..123
ignore index      = 255
normalization     = official loader order
out-of-range IDs  = canonicalize to 255
raw 253           = void/excluded, never a semantic class
loss              = CrossEntropyLoss(ignore_index=255)
metrics           = exclude canonical 255
```

## Evidence artifacts

- `results/R765_E00_official_audit/source_provenance.json`
- `results/R765_E01_domain_scan/full_domain_scan.json`
- `results/R765_E01_domain_scan/raw253_analysis.csv`
- `results/R765_E02_evaluator_equivalence/evaluator_equivalence.json`
- `results/R765_E02_evaluator_equivalence/real_mask_equivalence.csv`
- `results/R765_E03_contract_tests/vspw_label_contract.json`
- `results/R765_E03_contract_tests/test_summary.json`

## Stop boundary

Round 7.65 stops here. The next authorized activity is a separately controlled Round 7.6 training restart using this contract. No temporal reuse, latency benchmark, dataset expansion, checkpoint migration, or paper-writing phase is entered automatically.
