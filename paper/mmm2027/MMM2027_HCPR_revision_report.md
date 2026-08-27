# MMM 2027 HCPR Revision Report

## Baseline

Baseline: `MMM2027_HCPR_anonymous_audited.pdf` supplied on 2026-08-27.
The repository did not contain its TeX source, so a maintainable anonymous
source was reconstructed in `MMM2027_HCPR_anonymous_revised.tex` using the
baseline text, tables, frozen Round 11A/11B artifacts, and the official TFC
paper for external quantitative positioning.

## Applied revisions

1. Added published TFC VSPW values for MiT-B0/B1: mIoU, mVC8, mVC16, and FPS.
2. Added the HCPR mVC definition and stated that a matched HCPR mVC comparison
   cannot be computed from the archived aggregate outputs without per-frame
   prediction masks.
3. Added absolute Full mIoU values: B0 30.129 and B1 37.160.
4. Added all five fixed-policy reference values for both model scales.
5. Rewrote the hardware ablation discussion to state that NewCost and OldCost
   are close across the two GPUs, that OldCost wins both axes at B1/Q20 and
   B1/Q65, and that the larger quality separation is against DriftOnly at the
   low and middle operating points.
6. Changed the route-allocation header to uppercase `Q` and explained the
   Q80-to-Q90 B0 speedup decrease as CI-overlapping measurement noise.
7. Standardized the route-allocation shift wording to 1.29 percentage points
   for Full and 1.34 percentage points for S2--S4 at B1/Q20.

## Verification

- Final PDF: 10 pages, US Letter, below a 12-page inclusive limit.
- No placeholder author, email, TODO, or undefined citation/reference.
- No `Overfull \\hbox`; only ordinary reference-list `Underfull \\hbox` notices.
- Rendered pages 1--10 were visually checked for clipping, overlap, and table
  legibility.

## Remaining evidence gap

The manuscript does not claim a matched HCPR-versus-TFC mVC result. The next
scientifically valid step would be to regenerate per-frame HCPR prediction masks
under the TFC mVC8/mVC16 protocol and rerun the comparison on the same split and
hardware contract. This was not fabricated in the manuscript-only revision.
