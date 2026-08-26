# DriftGuard ICIP 2027 Draft

This directory contains the ICIP manuscript source and its evidence-controlled
assets.

- `main.tex`: manuscript source.
- `blind.tex`: double-blind build wrapper.
- `publish.tex`: publish build wrapper with author placeholders.
- `evidence_matrix.md`: quantitative claim gate.
- `reference_audit.csv`: bibliography verification record.
- `figures/`: generated vector figures and source sidecars.
- `tables/`: generated CSV tables and source sidecars.
- `round9_paper_report.md`: draft status and remaining submission gates.
- `BUILD.md`: compilation and verification commands.

The numerical results are frozen VSPW validation-split results for
SegFormer-B0/B1. The directory does not contain Cityscapes results or claims
about an independent public VSPW test set.
