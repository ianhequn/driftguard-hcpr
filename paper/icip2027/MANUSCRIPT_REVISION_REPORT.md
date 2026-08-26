# Manuscript Revision Report

Date: 2026-08-26

## Source fixes

- Added `\\cite{miao2021vspw}` to the VSPW validation paragraph.
- Reconciled the citation set: 10 cited keys, all present in BibTeX, verified,
  and marked `used_in_paper=yes`; see `citation_consistency.md`.
- Added compact, directly relevant citations for BOFP and AR-Seg rather than
  leaving audited bibliography entries unused.
- Replaced `task-aware schedule` with `fixed temporal schedule` for the final
  FixedInterval method.
- Removed unnecessary cache-age semantics from the final execution contract.
- Replaced strong causal language (`dominant reason`, `explaining`) with
  profile-bounded wording (`consistent with`, `associated with`).
- Changed the main table from nominal interval ratios to actual measured
  Stage4 call ratios: 50.381% for FI-2 and 13.041% for B1 FI-8.
- Added the separately scoped Round 5 Stage4-only quantitative negative result:
  4.366 ms dense versus 6.662 ms at the 37.5% cached-context route.
- Made RTX 4090, BF16, batch size 1, 512x512 input, CUDA Events, 200 warmups,
  and 1,000 measurements explicit.
- Removed the forced `\\clearpage` before the bibliography to avoid wasting a
  page and allow the normal ICIP layout to place references naturally.

## Build and PDF audit

- Built blind and publish wrappers with MiKTeX 25.12 pdfTeX using
  `pdflatex -> bibtex -> pdflatex -> pdflatex`.
- Both PDFs are 3-page US Letter documents with embedded fonts.
- No page 6 is present; the references-only page rule is not used.
- Undefined citations: 0. Undefined references: 0. Overfull boxes: 0.
- Blind extracted-text identity scan: PASS. PDF `Author` metadata: absent.
- Publish identity is intentionally the placeholder from `AUTHORS_TODO.md`.

## Remaining blockers

- Replace publish placeholders with the author list and affiliation only after
  the human authors provide them.
- Recheck the final ICIP 2027 author kit and run its official PDF checker.
- Underfull box warnings remain non-fatal and are recorded in `PAPER_STATUS.md`.

## Scientific boundary

No new experiment, model, dataset, GPU benchmark, Cityscapes result, or method
change was performed. The paper remains limited to the frozen VSPW validation
results for SegFormer-B0/B1.
