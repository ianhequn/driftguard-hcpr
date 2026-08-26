# Simulated ICIP Reviewer #2

## Summary

This paper studies whole-stage temporal feature reuse in hierarchical SegFormer video semantic segmentation. It varies S4, vanilla S3--S4, and S2--S4 boundaries, FI-2/FI-4/FI-8 refresh intervals, and frozen SegFormer-B0/B1 scales. It combines validation mIoU, physical stage-call counters, cache accounting, profiled suffix runtime, and paired CUDA input-to-logits timing on one RTX 4090 D.

The work is not a new reuse architecture. Its potential contribution is a controlled empirical result: the same five-point speed/quality frontier appears for B0 and B1, and moving the boundary earlier can dominate simply extending the interval at a later boundary. The result is useful for runtime design, but novelty remains borderline because temporal clocking and high-level reuse are established directions, and the study uses one dataset, two model scales, and one GPU.

## Strengths

- Honest scope: no first-work, SOTA, portability, or universal-optimum claim.
- S3--S4 is explicitly a vanilla baseline and is distinguished from TFC alignment, fusion, and correction.
- The full 18-setting matrix is reported.
- Physical call counters establish that cached suffix stages are absent on reuse frames; the decoder remains included.
- Paired Full baselines and CUDA-event timing are stronger than FLOPs-only reporting.
- The cache audit separates logical cache from allocated persistent F2--F4 state.
- The same audited frontier is reproduced on B0 and B1.

## Weaknesses

- The contribution is characterization rather than a new algorithm.
- VSPW validation results use frozen checkpoints selected in the same dataset context; they are not held-out test-set generalization.
- Hardware awareness is demonstrated on one RTX 4090 D and one software stack.
- Five paired repetitions are informative but not a population-level statistical guarantee.
- Fixed intervals are controlled study points, not a learned or adaptive scheduler.

## Novelty Assessment

After accounting for Clockwork Convnets and TFC, the remaining contribution is **BORDERLINE**. It is distinct as a controlled boundary x interval x model-scale x measured-GPU study, not as a reuse mechanism. The strongest non-obvious result is the audited dominance pattern: for both models, S3--S4/FI-4 and S3--S4/FI-8 are dominated by S2--S4/FI-2, while S4/FI-4 and S4/FI-8 are dominated by earlier-boundary points. Thus, a more aggressive boundary with a shorter interval can simultaneously provide higher speed and lower mIoU damage than a later boundary with a longer interval.

## Closest-Work Assessment

Clockwork Convnets motivate component-specific temporal rates, Deep Feature Flow motivates keyframe feature transfer, and Accel motivates efficient segmentation with correction. TFC is the closest conceptual concern because it studies efficient video semantic segmentation with high-level feature reuse. The manuscript sufficiently states that S3--S4 is only a vanilla high-level reuse baseline and does not reproduce TFC alignment, fusion, or sparse multi-scale deformable attention. It does not falsely claim that TFC failed to measure latency.

## Technical Soundness

Score: **4/5**. The execution equations, cache reset rule, first-frame refresh, decoder semantics, stage-call table, 100% refresh equivalence gate, and logical versus allocated cache definitions are coherent. The malformed table cross-reference was fixed before the reviewed build. Bootstrap intervals over five paired repetitions are appropriately bounded as repeated-protocol intervals.

## Experimental Rigor

Score: **4/5**. The protocol states 343 videos, 24,392 pairs, 124 classes, ignore label 255, batch size one, 512x512 input, BF16, 200 warmup iterations, 1,000 CUDA-event measurements, five repetitions, and an immediately paired Full baseline. Latency includes fresh stages, cache control, and decoder from CUDA-resident input to ready logits; exclusions are stated. Limitations are the single GPU, validation split, and limited model scales.

## Significance

Score: **3/5**. The paper gives a practical decision surface for choosing where reuse enters a hierarchical segmentation model. Its significance is strongest for readers who otherwise equate avoided computation with proportional GPU speedup. It is weaker as a general video segmentation method because no new architecture or scheduler is proposed.

## Presentation

Score: **4/5**. The revised PDF has four technical pages and a reference-only fifth page. Figure 1 explains boundaries; Figure 2 is the central Pareto result; Figure 3 adds quality versus refresh; Figure 4 shows runtime opportunity versus measured speedup. Tables 1--3 are evidence-bearing. Visual review found no clipping, broken equations, or identity leakage. Official-style underfull box warnings remain non-fatal.

## Major Concerns and Attacks

| Attack | Severity | Manuscript answer | Best bounded revision | New experiment? |
| --- | --- | --- | --- | --- |
| A. Temporal reuse and stage clocking already exist; novelty is unclear. | HIGH | PARTIAL but sufficient after narrowing. | Keep characterization framing; foreground full matrix, physical calls, and paired latency. | NO |
| B. TFC already studies high-level reuse on VSPW. | HIGH | YES. | Retain explicit non-reproduction of TFC alignment, fusion, and correction. | NO |
| C. Only VSPW is tested. | HIGH | YES through claim narrowing. | Use validation-system characterization, not generalization. | NO |
| D. Only one RTX 4090 D is tested. | MEDIUM | YES through claim narrowing. | Qualify every hardware claim by the tested GPU/protocol. | NO |
| E. Only B0 and B1 are evaluated. | MEDIUM | YES as a bounded scale comparison. | Say ordering is reproduced under two tested scales, not universally. | NO |
| F. The result is merely cache more, gain speed, lose quality. | MEDIUM | YES. | Explicitly foreground cross-setting dominance relations. | NO |
| G. FI-2/FI-4/FI-8 are arbitrary. | MEDIUM | PARTIAL. | Describe them as controlled study points, not an optimal scheduler. | NO |
| H. Excluding decoding and transfer weakens latency relevance. | MEDIUM | YES. | Call it CUDA-resident input-to-logits latency and state exclusions. | NO |
| I. Pareto analysis only reorganizes measurements. | MEDIUM | YES. | Retain the full matrix and explain nontrivial dominance cases. | NO |
| J. Five repeats make the intervals weak. | MEDIUM | PARTIAL. | Say bootstrap interval over five repeated paired measurements, not population confidence. | NO |
| K. Validation-set checkpoint selection contaminates quality. | HIGH | YES through scope narrowing. | Retain validation-system wording and remove test/generalization language. | NO |
| L. S3--S4 does not reproduce prior high-level methods. | MEDIUM | YES. | Keep vanilla high-level feature-reuse baseline. | NO |
| M. A characterization paper without a new model may not fit ICIP. | HIGH | PARTIAL but defensible. | Present measured frontier and physical evidence as the contribution. | NO |

## Minor Concerns

- Table 2 uses separate CI low/high columns; a single interval column could improve scanning but is not required.
- Reference [6] is correctly marked as an arXiv preprint; no formal venue should be fabricated.
- Hardware-aware is defensible only because GPU execution is directly measured; it is not cross-hardware validation.
- The final 2027 kit checker remains pending.

## Scorecard

| Category | Score |
| --- | ---: |
| Novelty | 3/5 |
| Technical soundness | 4/5 |
| Experimental rigor | 4/5 |
| Significance | 3/5 |
| Clarity | 4/5 |
| Reproducibility | 4/5 |
| Related work | 4/5 |
| ICIP fit | 4/5 |

## Overall Recommendation

**5/7 (Weak Accept).** The paper is publishable if ICIP accepts bounded, measurement-driven characterization work and the authors preserve the current claim discipline. The most likely rejection reason is that a reviewer expects a new algorithm and views the boundary study as incremental engineering.

## Confidence

**4/5**. The score is stable under the declared evidence, but novelty norms for characterization papers are venue- and reviewer-dependent.

## Top Five Rejection Risks

1. **Novelty overlap with TFC and Clockwork** — HIGH. Retain the explicit non-reproduction statement and foreground the measured boundary-first dominance result. New experiment: NO.
2. **Characterization lacks method novelty** — HIGH. Make the empirical question, full matrix, physical calls, and paired GPU evidence the contribution. New experiment: NO.
3. **Validation-set and single-dataset scope** — HIGH. Retain validation-system wording and no cross-dataset claims. New experiment: NO.
4. **One-GPU latency and excluded ingestion costs** — MEDIUM. Keep the RTX 4090 D qualifier next to hardware claims. New experiment: NO.
5. **Five-repeat bootstrap intervals** — MEDIUM. Use bounded repeated-protocol language. New experiment: NO.

## Fatal-Flaw Gate

**IS THERE ANY CURRENT FATAL SCIENTIFIC FLAW? NO.** The central conclusion is supported by physical stage-call counters, cache semantics, a complete matrix, an independent Pareto audit, and paired CUDA timing. The limitations affect scope and novelty risk, not interpretability of the tested result.

## Required Revisions

The required revisions are complete: the malformed table cross-reference was corrected, the TFC distinction is explicit, validation and one-GPU scopes are stated, and the final PDFs were rebuilt. Remaining actions are human authorship review and the final ICIP 2027 template checker.
